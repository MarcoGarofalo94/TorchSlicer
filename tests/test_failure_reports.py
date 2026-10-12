"""A runtime-error report names a worker index but is not authenticated; the
coordinator must confirm it with the reported worker before recovering."""

import grpc
import pytest

from torchslicer.config import RunConfig
from torchslicer.discovery.static import StaticDiscovery
from torchslicer.executors.distributed import DistributedExecutor
from torchslicer.executors.worker import WorkerServicer
from torchslicer.transport.grpc.coordinator import coordinator_service_pb2
from torchslicer.transport.grpc.worker import worker_service_pb2


class _Unavailable(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE


class _Stub:
    def __init__(self, fatal_error="", raises=False):
        self.fatal_error, self.raises, self.calls = fatal_error, raises, 0

    def get_stats(self, request, timeout=None):
        self.calls += 1
        if self.raises:
            raise _Unavailable()
        return worker_service_pb2.WorkerStatsResponse(fatal_error=self.fatal_error)


class _Proxy:
    def __init__(self, stub, address):
        self._s, self.address = stub, address

    def stub(self):
        return self._s


def _executor(stubs, verify=True):
    cfg = RunConfig()
    cfg.fault_tolerance.verify_reports = verify
    ex = DistributedExecutor(discovery=StaticDiscovery(peers=["w0:1"]), coordinator_addr="c:1", run_config=cfg)
    ex._proxies = [_Proxy(s, f"w{i}:1") for i, s in enumerate(stubs)]
    return ex


def _report(idx):
    return coordinator_service_pb2.WorkerError(worker_index=idx, batch_id=7, run_id="r", worker="peer",
                                               phase="forward", message="peer unreachable", fatal=True)


def test_report_about_healthy_worker_is_rejected():
    victim = _Stub(fatal_error="")
    ex = _executor([_Stub(), victim, _Stub()])
    ex._on_worker_runtime_error(_report(1))
    assert victim.calls == 1
    assert not ex._failure_event.is_set()
    assert 1 not in ex._failed_workers


@pytest.mark.parametrize("stub", [_Stub(fatal_error="CUDA OOM"), _Stub(raises=True)],
                         ids=["self_reported_fatal", "unreachable"])
def test_confirmed_report_marks_failure(stub):
    ex = _executor([_Stub(), stub])
    ex._on_worker_runtime_error(_report(1))
    assert ex._failure_event.is_set()
    assert 1 in ex._failed_workers


def test_out_of_range_index_is_rejected():
    ex = _executor([_Stub(), _Stub()])
    ex._on_worker_runtime_error(_report(5))
    assert not ex._failure_event.is_set()


def test_verification_can_be_disabled():
    ex = _executor([_Stub(), _Stub(fatal_error="")], verify=False)
    ex._on_worker_runtime_error(_report(1))
    assert 1 in ex._failed_workers


def test_worker_stats_expose_fatal_error():
    w = WorkerServicer()
    req = worker_service_pb2.GetStatsRequest(run_id="", epoch=-1, verbosity=0)
    assert w.get_stats(req, None).fatal_error == ""
    w._fatal_error = {"phase": "forward", "batch_id": 3, "message": "boom"}
    assert w.get_stats(req, None).fatal_error == "boom"
