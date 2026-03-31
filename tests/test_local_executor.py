import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
import torchslicer as ts
from torchslicer.core.model_graph import ModelGraph
from torchslicer.strategies.uniform import UniformSplitter
from torchslicer.executors.local import LocalExecutor
from torchslicer.pipeline import BasePipelineSchedule, StandardSchedule, GPipeSchedule


def _net():
    return nn.Sequential(
        nn.Linear(8, 16), nn.ReLU(),
        nn.Linear(16, 8), nn.ReLU(),
        nn.Linear(8, 4),
    )


def _loader(n=16, in_dim=8, n_classes=4):
    X = torch.randn(n, in_dim)
    y = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=4)


def _setup(n_parts=2, n_micro=1):
    graph = ModelGraph.from_module(_net())
    splitter = UniformSplitter()
    parts = splitter.split(graph, n_parts)
    splitter.validate(graph, parts)
    ex = LocalExecutor()
    ex.setup(
        graph, parts,
        optimizer_cfg={"name": "SGD", "params": {"lr": 0.01}},
        criterion_cfg={"name": "CrossEntropyLoss", "params": {}},
        n_micro_batches=n_micro,
    )
    return ex


# ── setup ──────────────────────────────────────────────────────────────────────

def test_setup_creates_correct_number_of_split_layers():
    assert len(_setup(n_parts=2).split_layers) == 2


def test_setup_all_split_layers_have_optimizer():
    for sl in _setup(n_parts=2).split_layers:
        assert sl.optimizer is not None


def test_setup_last_split_layer_is_last():
    ex = _setup(n_parts=2)
    assert ex.split_layers[-1]._is_last is True


def test_setup_non_last_split_layer_not_is_last():
    ex = _setup(n_parts=2)
    assert ex.split_layers[0]._is_last is False


# ── train_epoch ────────────────────────────────────────────────────────────────

def test_train_epoch_returns_dict_with_loss():
    ex = _setup()
    result = ex.train_epoch(_loader(), epoch=1)
    assert isinstance(result, dict)
    assert "loss" in result


def test_train_epoch_loss_is_finite_float():
    ex = _setup()
    result = ex.train_epoch(_loader(), epoch=1)
    assert isinstance(result["loss"], float)
    assert result["loss"] == result["loss"]  # not NaN
    assert result["loss"] < float("inf")


def test_train_epoch_multiple_epochs_runs():
    ex = _setup()
    loader = _loader()
    losses = [ex.train_epoch(loader, epoch=i)["loss"] for i in range(1, 4)]
    assert len(losses) == 3
    assert all(l < float("inf") for l in losses)


def test_train_epoch_single_partition():
    ex = _setup(n_parts=1)
    result = ex.train_epoch(_loader(), epoch=1)
    assert result["loss"] < float("inf")


# ── GPipe ──────────────────────────────────────────────────────────────────────

def test_gpipe_mode_returns_finite_loss():
    ex = _setup(n_parts=2, n_micro=2)
    result = ex.train_epoch(_loader(), epoch=1)
    assert result["loss"] < float("inf")


def test_gpipe_mode_4_microbatches():
    ex = _setup(n_parts=2, n_micro=4)
    result = ex.train_epoch(_loader(n=32), epoch=1)
    assert result["loss"] < float("inf")


# ── teardown ───────────────────────────────────────────────────────────────────

def test_teardown_clears_split_layers():
    ex = _setup()
    ex.train_epoch(_loader(), epoch=1)
    ex.teardown()
    assert ex.split_layers == []


def test_teardown_clears_criterion():
    ex = _setup()
    ex.teardown()
    assert ex.criterion is None


# ── BasePipelineSchedule / schedule injection ──────────────────────────────────

def test_default_schedule_is_standard():
    ex = _setup(n_micro=1)
    assert isinstance(ex.schedule, StandardSchedule)


def test_default_schedule_is_gpipe_when_n_micro_gt_1():
    ex = _setup(n_micro=2)
    assert isinstance(ex.schedule, GPipeSchedule)
    assert ex.schedule.n_micro == 2


def test_custom_schedule_injection():
    """An injected schedule is used instead of the default."""
    called = []

    class _TrackingSchedule(BasePipelineSchedule):
        def step(self, split_layers, inputs, labels, aux, criterion, profilers, batch_id):
            called.append(batch_id)
            # Delegate to standard so gradients actually flow
            return StandardSchedule().step(
                split_layers, inputs, labels, aux, criterion, profilers, batch_id
            )

    graph    = ModelGraph.from_module(_net())
    splitter = UniformSplitter()
    parts    = splitter.split(graph, 2)
    ex       = LocalExecutor(schedule=_TrackingSchedule())
    ex.setup(graph, parts,
             {"name": "SGD", "params": {"lr": 0.01}},
             {"name": "CrossEntropyLoss", "params": {}})
    ex.train_epoch(_loader(), epoch=1)
    ex.teardown()

    assert len(called) > 0, "custom schedule was never called"


def test_schedule_exported_from_top_level():
    assert ts.BasePipelineSchedule is BasePipelineSchedule
    assert ts.StandardSchedule is StandardSchedule
    assert ts.GPipeSchedule is GPipeSchedule
