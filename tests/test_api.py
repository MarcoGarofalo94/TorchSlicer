import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
import torchslicer as ts
from torchslicer import SlicedModel
from torchslicer.executors.local import LocalExecutor
from torchslicer.strategies.base import Partition


def _net():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def _loader():
    X = torch.randn(16, 8)
    y = torch.randint(0, 4, (16,))
    return DataLoader(TensorDataset(X, y), batch_size=4)


def _opt():
    return {"name": "SGD", "params": {"lr": 0.01}}


def _crit():
    return {"name": "CrossEntropyLoss", "params": {}}


# ── ts.slice() ─────────────────────────────────────────────────────────────────

def test_slice_returns_sliced_model():
    assert isinstance(ts.slice(_net(), n=2), SlicedModel)


def test_slice_default_executor_is_local():
    assert isinstance(ts.slice(_net(), n=2).executor, LocalExecutor)


def test_slice_partition_count():
    sliced = ts.slice(_net(), n=2)
    assert len(sliced.partitions) == 2


def test_slice_n1():
    sliced = ts.slice(_net(), n=1)
    assert len(sliced.partitions) == 1


def test_slice_with_pack_function():
    net = _net()

    def pack(m):
        return [ts.BlockStage(c) for c in m.children()]

    sliced = ts.slice(net, n=2, pack=pack)
    assert isinstance(sliced, SlicedModel)
    assert sliced.graph.is_packed()
    assert len(sliced.partitions) == 2


def test_slice_pack_bypasses_fx_tracing():
    # Model that would fail fx tracing; pack= should handle it fine
    class _Dynamic(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(4, 4)
            self.l2 = nn.Linear(4, 4)

        def forward(self, x):
            if x.sum() > 0:
                return self.l(x)
            return self.l2(x)

    sliced = ts.slice(_Dynamic(), n=1, pack=lambda m: [ts.BlockStage(c) for c in m.children()])
    assert isinstance(sliced, SlicedModel)


# ── SlicedModel.train() ────────────────────────────────────────────────────────

def test_train_returns_history_list():
    sliced = ts.slice(_net(), n=2)
    history = sliced.train(_loader(), _opt(), _crit(), epochs=2)
    assert isinstance(history, list)
    assert len(history) == 2


def test_train_history_contains_loss():
    sliced = ts.slice(_net(), n=2)
    history = sliced.train(_loader(), _opt(), _crit(), epochs=1)
    assert "loss" in history[0]


def test_train_loss_is_finite():
    sliced = ts.slice(_net(), n=2)
    history = sliced.train(_loader(), _opt(), _crit(), epochs=1)
    assert history[0]["loss"] < float("inf")


def test_train_n1_partition():
    sliced = ts.slice(_net(), n=1)
    history = sliced.train(_loader(), _opt(), _crit(), epochs=1)
    assert history[0]["loss"] < float("inf")


def test_train_with_gpipe():
    sliced = ts.slice(_net(), n=2)
    history = sliced.train(_loader(), _opt(), _crit(), epochs=1, use_gpipe=True, n_micro_batches=2)
    assert history[0]["loss"] < float("inf")


# ── ts.peft_unwrap() ───────────────────────────────────────────────────────────

def test_peft_unwrap_passthrough_non_peft():
    net = _net()
    assert ts.peft_unwrap(net) is net


# ── strategy registry ──────────────────────────────────────────────────────────

def test_register_and_use_custom_strategy():
    class _FirstTwo(ts.BaseSplitter):
        def split(self, graph, n):
            layers = list(range(len(graph)))
            mid = max(1, len(layers) // 2)
            return [
                Partition(index=0, layer_indices=layers[:mid]),
                Partition(index=1, layer_indices=layers[mid:]),
            ]

    ts.register_strategy("first_two", _FirstTwo)
    sliced = ts.slice(_net(), strategy="first_two", n=2)
    assert isinstance(sliced, SlicedModel)
    assert len(sliced.partitions) == 2


# ── optimizer / criterion string shorthands ────────────────────────────────────

@pytest.mark.parametrize("opt", ["adam", "adamw", "sgd", "rmsprop"])
def test_train_optimizer_string(opt):
    sliced = ts.slice(_net(), n=2)
    history = sliced.train(_loader(), optimizer=opt, criterion="cross_entropy", epochs=1)
    assert history[0]["loss"] < float("inf")


@pytest.mark.parametrize("crit", ["cross_entropy", "crossentropy"])
def test_train_criterion_string(crit):
    sliced = ts.slice(_net(), n=2)
    history = sliced.train(_loader(), optimizer="adam", criterion=crit, epochs=1)
    assert history[0]["loss"] < float("inf")


def test_train_mse_criterion():
    net = nn.Sequential(nn.Linear(8, 4))
    X = torch.randn(16, 8)
    y = torch.randn(16, 4)
    loader = DataLoader(TensorDataset(X, y), batch_size=4)
    sliced = ts.slice(net, n=1)
    history = sliced.train(loader, optimizer="adam", criterion="mse", epochs=1)
    assert history[0]["loss"] < float("inf")


def test_train_unknown_optimizer_raises():
    sliced = ts.slice(_net(), n=2)
    with pytest.raises(ValueError, match="Unknown optimizer shorthand"):
        sliced.train(_loader(), optimizer="nonesuch", criterion="cross_entropy", epochs=1)


def test_train_unknown_criterion_raises():
    sliced = ts.slice(_net(), n=2)
    with pytest.raises(ValueError, match="Unknown criterion shorthand"):
        sliced.train(_loader(), optimizer="adam", criterion="nonesuch", epochs=1)


# ── ts.run() ───────────────────────────────────────────────────────────────────

def test_run_returns_history():
    history = ts.run(_net(), _loader(), n=2, epochs=2)
    assert isinstance(history, list)
    assert len(history) == 2
    assert "loss" in history[0]


def test_run_loss_is_finite():
    history = ts.run(_net(), _loader(), n=2, epochs=1)
    assert history[0]["loss"] < float("inf")


def test_run_string_shortcuts():
    history = ts.run(_net(), _loader(), n=2, epochs=1,
                     optimizer="adamw", criterion="cross_entropy")
    assert history[0]["loss"] < float("inf")


def test_run_with_pack():
    def pack(m):
        return [ts.BlockStage(c) for c in m.children()]

    history = ts.run(_net(), _loader(), n=2, epochs=1, pack=pack)
    assert history[0]["loss"] < float("inf")


# ── ts.hf_pack() ──────────────────────────────────────────────────────────────

def test_hf_pack_known_architectures():
    for arch in ("gpt2", "bert", "distilbert", "llama", "mistral", "qwen2", "mixtral"):
        fn = ts.hf_pack(arch)
        assert callable(fn)


def test_hf_pack_case_insensitive():
    assert ts.hf_pack("GPT2") is ts.hf_pack("gpt2")


def test_hf_pack_unknown_raises():
    with pytest.raises(KeyError, match="No built-in pack function"):
        ts.hf_pack("unknown_arch_xyz")


# ── ts.from_pretrained() with nn.Module ───────────────────────────────────────

def test_from_pretrained_with_module():
    # Pass an nn.Module directly (no HF download needed)
    net = _net()
    sliced = ts.from_pretrained(net, n=2)
    assert isinstance(sliced, SlicedModel)
    assert len(sliced.partitions) == 2


def test_from_pretrained_module_trains():
    net = _net()
    sliced = ts.from_pretrained(net, n=2)
    history = sliced.train(_loader(), optimizer="adam", criterion="cross_entropy", epochs=1)
    assert history[0]["loss"] < float("inf")


def test_from_pretrained_rejects_hf_kwargs_with_module():
    with pytest.raises(ValueError, match="hf_kwargs are only used"):
        ts.from_pretrained(_net(), n=2, num_labels=2)


def test_from_pretrained_rejects_bad_type():
    with pytest.raises(TypeError):
        ts.from_pretrained(42, n=2)


def test_from_pretrained_no_transformers_raises_on_string(monkeypatch):
    import sys
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="transformers is required"):
        ts.from_pretrained("bert-base-uncased", n=2)
