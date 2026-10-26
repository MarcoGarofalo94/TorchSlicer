"""After train(), the data owner's model holds the trained weights in every topology.

The local executor trains the model's own modules; distributed executors train copies on
the workers, so at the end of the run the data owner reads them back
(``training.sync_back``: all by default, or hosted / none). The reference is always the
simulator: the same run on the local executor.
"""
import pytest
import torch
import torch.nn as nn

import torchslicer as ts
from torchslicer.config import RunConfig
from torchslicer.deploy.runner import free_port
from torchslicer.discovery import StaticDiscovery


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))


def _data():
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(48, 8, generator=g), torch.randint(0, 4, (48,), generator=g)
    return [(x[i:i + 12], y[i:i + 12]) for i in range(0, 48, 12)]


def _cfg(topology="local", setting="", sync_back=None, transport="grpc"):
    c = RunConfig()
    c.training.epochs = 2
    c.training.optimizer = {"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}
    c.discovery.n_workers = 3
    c.deploy.topology = topology
    c.network.device = "cpu"
    c.logging.enabled = False
    c.transport.tensor = transport
    c.sl.setting = setting
    if sync_back is not None:
        c.training.sync_back = sync_back
    return c


def _trained(cfg):
    model = _model()
    ts.run(lambda: model, _data, run_config=cfg, verbose=False)
    return model


def _same(a, b):
    for (name, x), y in zip(a.state_dict().items(), b.state_dict().values()):
        assert torch.allclose(x, y, atol=1e-6), name


@pytest.mark.parametrize("topology,setting,transport", [
    ("centralized", "pipeline", "grpc"),
    ("centralized", "vanilla", "tcp"),
    ("centralized", "u_shaped", "grpc"),
    ("p2p", "vanilla", "grpc"),
    ("p2p", "u_shaped", "tcp"),
])
def test_model_holds_the_trained_weights_like_the_simulator(topology, setting, transport):
    ref = _trained(_cfg())
    assert not torch.equal(ref[0].weight, _model()[0].weight)          # it did train
    _same(_trained(_cfg(topology, setting, transport=transport)), ref)


@pytest.mark.parametrize("setting,hosted", [("vanilla", [0]), ("u_shaped", [0, 2]), ("pipeline", [])])
def test_sync_back_hosted_reads_only_the_data_owners_slices(setting, hosted):
    ref, init = _trained(_cfg()), _model()
    got = _trained(_cfg("centralized", setting, sync_back="hosted"))
    layers = {0: 0, 1: 2, 2: 4}                                        # slice -> its Linear
    for k, i in layers.items():
        _same(got[i], ref[i] if k in hosted else init[i])


def test_sync_back_none_leaves_the_model_as_it_was():
    _same(_trained(_cfg("centralized", "vanilla", sync_back="none")), _model())


def test_sync_back_refuses_an_unknown_value():
    with pytest.raises(ValueError, match="sync_back"):
        _trained(_cfg("centralized", "vanilla", sync_back="some"))


# ── client nodes: the model ends like the in-process session's ──────────────

C, CUTS = 2, ["1", "3"]


def _clients_data():
    out = []
    for k in range(C):
        g = torch.Generator().manual_seed(100 + k)
        out.append([(torch.randn(8, 8, generator=g), torch.randint(0, 4, (8,), generator=g))
                    for _ in range(3 + k)])
    return out


def _client_cfg():
    c = _cfg(setting="vanilla")
    c.sl.clients = C
    c.discovery.timeout = 30
    return c


@pytest.mark.parametrize("mode,algorithm", [("sequential", None), ("parallel", "splitfed")])
def test_client_nodes_model_ends_like_the_in_process_session(mode, algorithm):
    algo = ts.SplitFedAlgorithm if algorithm else (lambda: None)
    ref = _model()
    ts.slice(ref, cuts=CUTS).train(ts.ClientSession(_clients_data(), mode=mode),
                                   run_config=_client_cfg(), algorithm=algo())
    cfg = _client_cfg()
    offsets = cfg.transport.listen_offsets()
    servers = [ts.start_worker(free_port(offsets), host="127.0.0.1", run_config=cfg) for _ in range(2)]
    clients = [ts.start_worker(free_port(offsets), host="127.0.0.1", run_config=cfg, data=d, client_index=k)
               for k, d in enumerate(_clients_data())]
    addr = f"127.0.0.1:{free_port(cfg.transport.listen_offsets(drives=True))}"
    ex = ts.ClientNodesExecutor(discovery=StaticDiscovery(peers=[w.address for w in clients + servers]),
                                coordinator_addr=addr, run_config=cfg)
    got = _model()
    try:
        ts.slice(got, cuts=CUTS, executor=ex).train(ts.RemoteSession(C, mode=mode),
                                                    run_config=cfg, algorithm=algo())
    finally:
        for w in servers + clients:
            w.stop()
    _same(got, ref)


def test_sync_back_reads_weights_only():
    """sync_back needs no optimizer state: with Adam it would triple what each slice sends."""
    import io
    from torchslicer.executors.worker import WorkerServicer

    seen = []

    class _Recording(WorkerServicer):
        def get_state(self, request, context):
            reply = super().get_state(request, context)
            seen.append((request.weights_only, torch.load(io.BytesIO(reply.data), weights_only=True)))
            return reply

    cfg = _cfg("centralized", "pipeline")
    cfg.training.optimizer = {"name": "Adam", "params": {"lr": 1e-3}}
    workers = [ts.start_worker(free_port(cfg.transport.listen_offsets()), host="127.0.0.1", run_config=cfg,
                               servicer_class=_Recording) for _ in range(3)]
    addr = f"127.0.0.1:{free_port(cfg.transport.listen_offsets(drives=True))}"
    ex = ts.DistributedExecutor(discovery=StaticDiscovery(peers=[w.address for w in workers]),
                                coordinator_addr=addr, run_config=cfg)
    model = _model()
    try:
        ts.slice(model, n=3, executor=ex).train(_data(), run_config=cfg)
    finally:
        for w in workers:
            w.stop()
    assert len(seen) == 3 and all(wo and st["optimizer"] is None for wo, st in seen)
    ref = _model()
    ts.slice(ref, n=3).train(_data(), run_config=_adam_local())
    _same(model, ref)


def _adam_local():
    cfg = _cfg()
    cfg.training.optimizer = {"name": "Adam", "params": {"lr": 1e-3}}
    return cfg
