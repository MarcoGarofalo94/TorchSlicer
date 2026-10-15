"""P2P in the library (``ts.P2PExecutor``): the driver is worker 0 + coordinator, the
followers are plain workers. Training must match unsplit PyTorch, and a dead follower
must fail (or, with fault tolerance, recover) the run instead of hanging it."""
import socket
import time

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torchslicer as ts
from torchslicer.config import RunConfig
from torchslicer.executors.worker import WorkerServicer

SGD = {"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}
CE = {"name": "CrossEntropyLoss", "params": {}}


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(),
                         nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))


def _loader():
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(64, 8, generator=g), torch.randint(0, 4, (64,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)


def _reference(epochs, n_micro=1):
    model = _model()
    opt = torch.optim.SGD(model.parameters(), **SGD["params"])
    out = []
    for _ in range(epochs):
        losses = []
        for x, y in _loader():
            opt.zero_grad()
            parts = [nn.functional.cross_entropy(model(xm), ym) for xm, ym
                     in zip(x.chunk(n_micro), y.chunk(n_micro))]
            sum(p / n_micro for p in parts).backward()
            opt.step()
            losses.append(sum(p.item() for p in parts) / n_micro)
        out.append(sum(losses) / len(losses))
    return out


def _free_pair():
    """A port p with p+1 also free (TCP tensor server = port + 1)."""
    for _ in range(50):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
        try:
            with socket.socket() as a, socket.socket() as b:
                a.bind(("127.0.0.1", p))
                b.bind(("127.0.0.1", p + 1))
            return p
        except OSError:
            continue
    raise RuntimeError("no free port pair")


def _cfg(transport="grpc", epochs=2, **ft):
    cfg = RunConfig()
    cfg.run_id = "p2p-test"
    cfg.training.epochs = epochs
    cfg.training.optimizer = SGD
    cfg.training.criterion = CE
    cfg.logging.enabled = False
    cfg.transport.tensor = transport
    cfg.fault_tolerance.heartbeat_interval_s = 0.2
    cfg.fault_tolerance.ping_timeout_s = 0.5
    for k, v in ft.items():
        setattr(cfg.fault_tolerance, k, v)
    return cfg


def _followers(cfg, k, servicer_class=None):
    return [ts.start_worker(_free_pair(), host="127.0.0.1", run_config=cfg,
                            servicer_class=servicer_class) for _ in range(k)]


def _driver(cfg, followers):
    addr = f"127.0.0.1:{_free_pair()}"
    return ts.P2PExecutor(peers=[addr, *[w.address for w in followers]], address=addr,
                          bind_addr=addr, run_config=cfg)


@pytest.mark.parametrize("transport", ["grpc", "tcp"])
def test_p2p_matches_unsplit(transport):
    cfg = _cfg(transport)
    followers = _followers(cfg, 2)
    try:
        sliced = ts.slice(_model(), n=3, executor=_driver(cfg, followers))
        history = sliced.train(_loader(), run_config=cfg)
    finally:
        for w in followers:
            w.stop()
    assert [h["loss"] for h in history] == pytest.approx(_reference(2), abs=1e-5)


def test_p2p_gpipe_matches_unsplit():
    cfg = _cfg()
    cfg.pipeline.use_gpipe, cfg.pipeline.n_micro = True, 2
    followers = _followers(cfg, 1)
    try:
        sliced = ts.slice(_model(), n=2, executor=_driver(cfg, followers))
        history = sliced.train(_loader(), run_config=cfg)
    finally:
        for w in followers:
            w.stop()
    assert [h["loss"] for h in history] == pytest.approx(_reference(2, n_micro=2), abs=1e-5)


def test_p2p_driver_keeps_raw_inputs():
    """Slice 0 runs in the driver's process; followers never receive a raw input."""
    seen = []

    class Spy(WorkerServicer):
        def forward(self, request, context):
            if request.input.data and self.layer is not None and self._worker_index == 1:
                seen.append(tuple(request.input.shape))
            return super().forward(request, context)

    cfg = _cfg(epochs=1)
    followers = _followers(cfg, 1, servicer_class=Spy)
    try:
        ex = _driver(cfg, followers)
        ts.slice(_model(), cuts=["2"], executor=ex).train(_loader(), run_config=cfg)
    finally:
        for w in followers:
            w.stop()
    assert seen and all(shape[1] == 16 for shape in seen)   # activations, not 8-dim inputs


class _StopAfter(ts.TrainingCallback):
    """Stop a follower's server at the end of an epoch (simulated crash)."""

    def __init__(self, worker, epoch):
        self.worker, self.epoch = worker, epoch

    def on_epoch_end(self, epoch, metrics):
        if epoch == self.epoch:
            self.worker.stop()
        return metrics


@pytest.mark.parametrize("dead", [0, 1])   # middle follower (hung before M2), last follower
def test_p2p_dead_follower_fails_the_run_instead_of_hanging(dead):
    cfg = _cfg(epochs=3)            # fault_tolerance.enabled = False: no recovery
    followers = _followers(cfg, 2)
    try:
        sliced = ts.slice(_model(), n=3, executor=_driver(cfg, followers))
        t0 = time.monotonic()
        with pytest.raises(ts.WorkerFailureError) as err:
            sliced.train(_loader(), run_config=cfg, callbacks=[_StopAfter(followers[dead], 1)])
        assert time.monotonic() - t0 < 30
        assert err.value.worker_idx == dead + 1
    finally:
        for w in followers:
            w.stop()


def test_p2p_recovers_from_a_dead_follower(tmp_path):
    cfg = _cfg(epochs=3, enabled=True, max_retries=1)
    cfg.checkpoint.dir = str(tmp_path)
    followers = _followers(cfg, 2)
    try:
        ex = _driver(cfg, followers)
        sliced = ts.slice(_model(), n=3, executor=ex)
        history = sliced.train(_loader(), run_config=cfg, callbacks=[_StopAfter(followers[1], 1)])
        assert len(history) == 3 and all(h["loss"] == h["loss"] for h in history)
        assert [nd.address for nd in ex._nodes] == [ex.address, followers[0].address]
    finally:
        for w in followers:
            w.stop()


def test_p2p_driver_slice_failure_stops_the_run(tmp_path):
    cfg = _cfg(epochs=2, enabled=True, max_retries=2)
    cfg.checkpoint.dir = str(tmp_path)
    followers = _followers(cfg, 1)
    try:
        ex = _driver(cfg, followers)

        calls = []

        def boom(phase, batch_id):
            calls.append(batch_id)
            if len(calls) == 2:
                raise RuntimeError("injected")
        ex.local._before_runtime_phase = boom
        with pytest.raises(RuntimeError, match="driver's own slice failed"):
            ts.slice(_model(), n=2, executor=ex).train(_loader(), run_config=cfg)
    finally:
        for w in followers:
            w.stop()


def test_liveness_probe_does_not_reset_profiler_stats():
    from torchslicer.transport.grpc.worker import worker_service_pb2 as pb
    from torchslicer.monitor import WorkerProfiler
    w = WorkerServicer()
    w._profiler = WorkerProfiler(verbosity=1)
    w._profiler.begin_batch(0)
    with w._profiler.phase("forward"):
        pass
    w._profiler.end_batch()
    w.get_stats(pb.GetStatsRequest(epoch=-1), None)            # heartbeat
    assert w.get_stats(pb.GetStatsRequest(epoch=0, verbosity=1), None).forward.count == 1


def test_centralized_dead_worker_fails_the_run_instead_of_hanging():
    """Same detection in the centralized topology (it hung the same way before M2)."""
    cfg = _cfg(epochs=3)
    workers = _followers(cfg, 3)
    coord = f"127.0.0.1:{_free_pair()}"
    try:
        ex = ts.DistributedExecutor(discovery=ts.StaticDiscovery([w.address for w in workers]),
                                    coordinator_addr=coord, coordinator_bind_addr=coord,
                                    run_config=cfg)
        with pytest.raises(ts.WorkerFailureError) as err:
            ts.slice(_model(), n=3, executor=ex).train(
                _loader(), run_config=cfg, callbacks=[_StopAfter(workers[1], 1)])
        assert err.value.worker_idx == 1
    finally:
        for w in workers:
            w.stop()
