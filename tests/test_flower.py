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
