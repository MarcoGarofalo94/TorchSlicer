"""Flower strategies as SplitFed aggregation (``torchslicer.interop.flower``): Flower's FedAvg
gives exactly ``ts.SplitFedAlgorithm`` (V2) / ``SplitFedV1Algorithm`` (V1), both weighting
the clients by their sample counts; other strategies plug in. Skipped without ``flwr``."""
import pytest
import torch
import torch.nn as nn

import torchslicer as ts

flwr = pytest.importorskip("flwr")
from flwr.server.strategy import FedAdam, FedAvg, FedMedian  # noqa: E402

from torchslicer.interop.flower import FlowerAggregation, state_to_parameters  # noqa: E402

B, D = 8, 6


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(D, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))


def _datas():
    g = torch.Generator().manual_seed(6)
    return [[(torch.randn(B, D, generator=g), torch.randint(0, 4, (B,), generator=g)) for _ in range(n)]
            for n in (1, 2, 4)]                              # unequal clients: the weights matter


def _train(algorithm, epochs=3):
    cfg = ts.RunConfig()
    cfg.training.optimizer = {"name": "SGD", "params": {"lr": 0.1}}
    cfg.training.criterion = {"name": "CrossEntropyLoss", "params": {}}
    cfg.training.epochs = epochs
    cfg.sl.setting = "vanilla"
    cfg.logging.enabled = False
    session = ts.ClientSession(_datas(), mode="parallel")
    return [e["loss"] for e in ts.slice(_model(), n=2, cuts=["1"]).train(session, run_config=cfg,
                                                                           algorithm=algorithm)]


def test_flower_fedavg_is_splitfed_v2():
    assert _train(FlowerAggregation(FedAvg())) == pytest.approx(_train(ts.SplitFedAlgorithm()), abs=1e-6)


def test_flower_fedavg_is_splitfed_v1():
    assert _train(FlowerAggregation(FedAvg(), server_strategy=FedAvg())) == pytest.approx(
        _train(ts.SplitFedV1Algorithm()), abs=1e-6)


def test_other_strategies_plug_in():
    median = _train(FlowerAggregation(FedMedian()))
    assert median[1:] != pytest.approx(_train(ts.SplitFedAlgorithm())[1:], abs=1e-6)
    client = nn.Sequential(*list(_model())[:2])
    adam = FedAdam(initial_parameters=state_to_parameters({0: client}), eta=0.01)
    losses = _train(FlowerAggregation(adam))
    assert all(torch.isfinite(torch.tensor(losses)))


def test_state_to_parameters_round_trip():
    from flwr.common import parameters_to_ndarrays
    m = _model()
    arrays = parameters_to_ndarrays(state_to_parameters({0: m}))
    assert [a.shape for a in arrays] == [tuple(v.shape) for v in m.state_dict().values()]
    assert all((torch.from_numpy(a) == v).all() for a, v in zip(arrays, m.state_dict().values()))


def test_split_client_app_in_flowers_simulation_engine_is_splitfed_v2():
    """Flower's own engine (``run_simulation``) runs TorchSlicer client slices against a
    ``ts.SplitServer``: with one node it equals ``ts.SplitFedAlgorithm`` on one client; with
    three nodes, FedAvg aggregates and the loss goes down."""
    pytest.importorskip("ray")
    import socket
    from flwr.app import ArrayRecord, ConfigRecord, Context
    from flwr.serverapp import Grid, ServerApp
    from flwr.serverapp.strategy import FedAvg as MsgFedAvg
    from flwr.simulation import run_simulation
    from torchslicer.interop.flower import initial_client_arrays, split_client_app

    opt, crit = {"name": "SGD", "params": {"lr": 0.1, "momentum": 0.9}}, {"name": "CrossEntropyLoss", "params": {}}
    rounds = 3

    def model_fn():
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(D, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))

    def data_fn(pid, n):
        g = torch.Generator().manual_seed(100 + pid)
        d = [(torch.randn(B, D, generator=g), torch.randint(0, 4, (B,), generator=g)) for _ in range(4)]
        return d, d

    def simulate(nodes):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server = ts.SplitServer(model_fn(), ["1"], opt, crit, port=port).start()
        out = {}
        sapp = ServerApp()

        @sapp.main()
        def main(grid: Grid, context: Context):
            strategy = MsgFedAvg(min_train_nodes=nodes, min_evaluate_nodes=nodes, min_available_nodes=nodes)
            out["result"] = strategy.start(grid=grid, initial_arrays=initial_client_arrays(model_fn, ["1"]),
                                           num_rounds=rounds, train_config=ConfigRecord({"local-epochs": 1}))

        capp = split_client_app(model_fn, data_fn, f"127.0.0.1:{port}", ["1"], opt)
        try:
            run_simulation(server_app=sapp, client_app=capp, num_supernodes=nodes,
                           backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}})
        finally:
            server.stop()
        return out["result"], server

    result, server = simulate(1)
    client = result.arrays.to_torch_state_dict()
    cfg = ts.RunConfig()
    cfg.training.optimizer, cfg.training.criterion, cfg.training.epochs = opt, crit, rounds
    cfg.sl.setting, cfg.logging.enabled = "vanilla", False
    ref = model_fn()
    ts.slice(ref, cuts=["1"]).train(ts.ClientSession([data_fn(0, 1)[0]], mode="sequential"),
                                    run_config=cfg, algorithm=ts.SplitFedAlgorithm())
    want = list(ref.parameters())
    got = list(client.values()) + [p for s in server.slices for p in s.parameters()]
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert torch.allclose(a, b, atol=1e-5)

    result, _ = simulate(3)
    losses = [m["train_loss"] for _, m in sorted(result.train_metrics_clientapp.items())]
    assert len(losses) == rounds and losses[-1] < losses[0]
