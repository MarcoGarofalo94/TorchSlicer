import time

import grpc
import torch

from torchslicer.config import RunConfig
from torchslicer.discovery.static import StaticDiscovery
from torchslicer.executors.distributed import DistributedExecutor
from torchslicer.executors.startup import init_worker_with_retry
from torchslicer.executors.worker import WorkerServicer
from torchslicer.retry import RetryPolicy
from torchslicer import testing_faults as fault_injection
from torchslicer.transport.grpc.coordinator import coordinator_service_pb2


class _FakeCoordinatorStub:
    def __init__(self):
        self.calls = []

    def report_worker_error(self, request, timeout=None):
        self.calls.append((request, timeout))

    def batch_done(self, request, timeout=None):
        self.calls.append((request, timeout))


class _FakeServer:
    def __init__(self):
        self.stop_calls = []

    def stop(self, grace=0):
        self.stop_calls.append(grace)


class _FakePrevStub:
    def __init__(self):
        self.calls = []

    def backward(self, request, timeout=None):
        self.calls.append((request, timeout))


class _UnavailablePrevStub:
    def backward(self, request, timeout=None):
        raise _UnavailableRpcError()


def test_run_config_load_applies_retry_env_overrides(monkeypatch):
    monkeypatch.setenv("DISCOVERY_REGISTRATION_MAX_ATTEMPTS", "9")
    monkeypatch.setenv("DISCOVERY_REGISTRATION_DELAY", "1.5")
    monkeypatch.setenv("DISCOVERY_REGISTRATION_RPC_TIMEOUT", "7.0")
    monkeypatch.setenv("DISCOVERY_WATCHDOG_INTERVAL", "22")
    monkeypatch.setenv("WORKER_INIT_MAX_ATTEMPTS", "14")
    monkeypatch.setenv("WORKER_INIT_DELAY", "2.5")
    monkeypatch.setenv("WORKER_INIT_RPC_TIMEOUT", "8.0")

    cfg = RunConfig.from_env()

    assert cfg.discovery.registration_max_attempts == 9
    assert cfg.discovery.registration_delay_s == 1.5
    assert cfg.discovery.registration_rpc_timeout_s == 7.0
    assert cfg.discovery.watchdog_interval_s == 22.0
    assert cfg.startup.worker_init_max_attempts == 14
    assert cfg.startup.worker_init_delay_s == 2.5
    assert cfg.startup.worker_init_rpc_timeout_s == 8.0


def test_run_config_load_applies_log_level_env_override(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    cfg = RunConfig.from_env()

    assert cfg.logging.level == "DEBUG"


def test_worker_servicer_reports_fatal_error_once():
    servicer = WorkerServicer()
    servicer._coord_stub = _FakeCoordinatorStub()
    servicer._server = _FakeServer()
    servicer._run_id = "run-123"
    servicer._worker_index = 2

    try:
        raise ValueError("boom")
    except ValueError as exc:
        servicer._report_fatal_error("backward", 17, exc)

    try:
        raise RuntimeError("ignored")
    except RuntimeError as exc:
        servicer._report_fatal_error("forward", 18, exc)

    time.sleep(0.05)

    assert len(servicer._coord_stub.calls) == 1
    request, timeout = servicer._coord_stub.calls[0]
    assert isinstance(request, coordinator_service_pb2.WorkerError)
    assert request.worker_index == 2
    assert request.batch_id == 17
    assert request.run_id == "run-123"
    assert request.phase == "backward"
    assert request.message == "boom"
    assert request.fatal is True
    assert timeout == 5.0
    assert servicer._server.stop_calls == [0]


def test_distributed_executor_records_runtime_error_and_unblocks_wait():
    executor = DistributedExecutor(
        discovery=StaticDiscovery(peers=["worker0:50051"]),
        coordinator_addr="coordinator:50054",
        run_config=RunConfig(),
    )

    request = coordinator_service_pb2.WorkerError(
        worker_index=0,
        batch_id=11,
        run_id="run-123",
        worker="worker0",
        phase="forward",
        message="shape mismatch",
        traceback="traceback",
        fatal=True,
    )

    executor._on_worker_runtime_error(request)

    assert executor._failure_event.is_set()
    assert executor._batch_done.is_set()
    assert 0 in executor._failed_workers
    assert executor._failure_reason[0] == "forward: shape mismatch"
    assert executor._failure_info[0] == "worker0"


def test_worker_send_backward_notifies_coordinator_when_first_worker():
    servicer = WorkerServicer()
    servicer._coord_stub = _FakeCoordinatorStub()
    servicer._run_id = "run-123"

    servicer._send_backward(batch_id=9, grad=None, is_last_micro=True)

    request, timeout = servicer._coord_stub.calls[0]
    assert isinstance(request, coordinator_service_pb2.BatchDoneRequest)
    assert request.batch_id == 9
    assert request.run_id == "run-123"
    assert timeout is None


def test_worker_send_backward_rejects_stale_generation():
    servicer = WorkerServicer()
    servicer._prev_stub = _FakePrevStub()
    servicer._generation = 2

    try:
        servicer._send_backward(batch_id=9, grad=None, is_last_micro=True, generation=1)
        assert False, "expected stale work rejection"
    except RuntimeError as exc:
        assert "stale work" in str(exc)

    assert servicer._prev_stub.calls == []


def test_worker_send_backward_reports_peer_unavailable_as_nonfatal_runtime_error():
    servicer = WorkerServicer()
    servicer._prev_stub = _UnavailablePrevStub()
    servicer.prev_worker = "worker0:50051"

    try:
        servicer._send_backward(
            batch_id=9,
            grad=torch.ones(1),
            is_last_micro=True,
            generation=0,
        )
        assert False, "expected peer unavailable error"
    except RuntimeError as exc:
        assert "peer unavailable" in str(exc)


def test_worker_fault_config_matches_runtime_filters():
    config = fault_injection.WorkerFaultConfig(
        runtime_phase="backward",
        runtime_batch_id=17,
        runtime_worker_index=2,
    )

    assert config.matches_runtime(worker_index=2, phase="backward", batch_id=17) is True
    assert config.matches_runtime(worker_index=1, phase="backward", batch_id=17) is False
    assert config.matches_runtime(worker_index=2, phase="forward", batch_id=17) is False
    assert config.matches_runtime(worker_index=2, phase="backward", batch_id=18) is False


def test_worker_fault_controller_raises_only_once():
    controller = fault_injection.WorkerFaultController(
        fault_injection.WorkerFaultConfig(
            runtime_phase="forward",
            runtime_batch_id=9,
            runtime_worker_index=1,
        )
    )

    try:
        controller.maybe_raise_runtime(worker_index=1, phase="forward", batch_id=9)
        assert False, "expected simulated worker failure"
    except RuntimeError as exc:
        assert "simulated worker failure" in str(exc)

    controller.maybe_raise_runtime(worker_index=1, phase="forward", batch_id=9)


def test_worker_fault_controller_uses_shared_marker(monkeypatch, tmp_path):
    monkeypatch.setenv("FAULT_STATE_DIR", str(tmp_path))
    config = fault_injection.WorkerFaultConfig(
        runtime_phase="backward",
        runtime_batch_id=-1,
        runtime_worker_index=0,
    )
    first = fault_injection.WorkerFaultController(config)
    second = fault_injection.WorkerFaultController(config)

    try:
        first.maybe_raise_runtime(worker_index=0, phase="backward", batch_id=12, run_id="run-1")
        assert False, "expected first simulated worker failure"
    except RuntimeError:
        pass

    second.maybe_raise_runtime(worker_index=0, phase="backward", batch_id=99, run_id="run-1")


def test_coordinator_crash_callback_schedules_exit_once(monkeypatch):
    scheduled = []

    def _fake_schedule(delay_s, exit_code, reason):
        scheduled.append((delay_s, exit_code, reason))

    monkeypatch.setattr(fault_injection, "schedule_process_exit", _fake_schedule)
    callback = fault_injection.CoordinatorCrashCallback(
        fault_injection.CoordinatorFaultConfig(exit_epoch=2, exit_delay_s=1.25, exit_code=99)
    )

    metrics = {"loss": 1.0}
    assert callback.on_epoch_end(1, metrics) is metrics
    assert scheduled == []

    assert callback.on_epoch_end(2, metrics) is metrics
    assert scheduled == [(1.25, 99, "coordinator crash epoch=2 threshold=2")]

    assert callback.on_epoch_end(3, metrics) is metrics
    assert len(scheduled) == 1


class _UnavailableRpcError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE

    def __str__(self):
        return "UNAVAILABLE"


class _FakeInitResponse:
    def __init__(self, ok=True, message="ok", hostname="worker0"):
        self.ok = ok
        self.message = message
        self.hostname = hostname


class _FakeInitStub:
    def __init__(self):
        self.calls = 0

    def init(self, slice_cfg, timeout=None):
        self.calls += 1
        if self.calls == 1:
            raise _UnavailableRpcError()
        return _FakeInitResponse()


class _CollectingLogger:
    def __init__(self):
        self.messages = []

    def warning(self, message, *args):
        self.messages.append(message % args)


def test_init_worker_with_retry_retries_unavailable_once():
    stub = _FakeInitStub()
    logger = _CollectingLogger()

    response = init_worker_with_retry(
        stub,
        object(),
        address="worker0:50051",
        policy=RetryPolicy(max_attempts=2, delay_s=0.0, rpc_timeout_s=5.0),
        logger=logger,
        phase="init",
    )

    assert response.ok is True
    assert stub.calls == 2
    assert len(logger.messages) == 1
    assert "retry 1/2" in logger.messages[0]
