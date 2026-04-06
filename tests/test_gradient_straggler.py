"""
Tests for backward gradient hooks and straggler policies.
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytest

import torchslicer as ts
from torchslicer.hooks import GradientSparsifyHook
from torchslicer.straggler import FixedDelayPolicy, RandomDelayPolicy
from torchslicer.pipeline import StandardSchedule, GPipeSchedule
from torchslicer.executors.local import LocalExecutor

torch.manual_seed(0)
X = torch.randn(64, 16)
y = torch.randint(0, 4, (64,))
loader = DataLoader(TensorDataset(X, y), batch_size=32)


def make_model():
    return nn.Sequential(
        nn.Linear(16, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 4),
    )


# ── ActivationHook backward interface ────────────────────────────────────────

def test_default_on_backward_smash_is_passthrough():
    """on_backward_smash default returns grad unchanged."""
    class ForwardOnlyHook(ts.ActivationHook):
        def on_forward_smash(self, smashed, raw_input, partition_idx):
            return smashed

    hook = ForwardOnlyHook()
    g = torch.randn(4, 8)
    assert hook.on_backward_smash(g, 0) is g


def test_default_on_forward_smash_is_passthrough():
    """on_forward_smash default returns smashed unchanged."""
    class BackwardOnlyHook(ts.ActivationHook):
        def on_backward_smash(self, grad, partition_idx):
            return grad * 0.5

    hook = BackwardOnlyHook()
    s = torch.randn(4, 8)
    assert hook.on_forward_smash(s, s, 0) is s


def test_backward_hook_called_during_training():
    """on_backward_smash is called during the backward pass."""
    calls = []

    class RecordingHook(ts.ActivationHook):
        def on_backward_smash(self, grad, partition_idx):
            calls.append((partition_idx, grad.shape))
            return grad

    schedule = StandardSchedule(hooks=[RecordingHook()])
    executor = LocalExecutor(schedule=schedule)
    ts.run(make_model(), loader, n=3, epochs=1, optimizer="adam",
           criterion="cross_entropy", executor=executor)

    assert len(calls) > 0
    # cut points: 0 and 1 for 3 partitions
    seen_indices = {c[0] for c in calls}
    assert seen_indices <= {0, 1}


def test_backward_hook_can_zero_gradients():
    """A hook that zeros gradients should produce different (worse) training."""
    class ZeroGradHook(ts.ActivationHook):
        def on_backward_smash(self, grad, partition_idx):
            return torch.zeros_like(grad)

    normal = ts.run(make_model(), loader, n=2, epochs=3,
                    optimizer="adam", criterion="cross_entropy")

    schedule = StandardSchedule(hooks=[ZeroGradHook()])
    executor = LocalExecutor(schedule=schedule)
    zeroed = ts.run(make_model(), loader, n=2, epochs=3,
                    optimizer="adam", criterion="cross_entropy", executor=executor)

    # With zeroed cut-point gradients only the last partition learns
    assert zeroed[-1]["loss"] > normal[-1]["loss"]


# ── GradientSparsifyHook ─────────────────────────────────────────────────────

def test_gradient_sparsify_sparsifies():
    """GradientSparsifyHook zeros out (1 - keep_ratio) fraction of gradient."""
    hook = GradientSparsifyHook(keep_ratio=0.1)
    grad = torch.randn(100)
    out = hook.on_backward_smash(grad, 0)
    nonzero_ratio = (out != 0).float().mean().item()
    assert nonzero_ratio <= 0.15  # allow slight rounding


def test_gradient_sparsify_keepratio_1_is_identity():
    hook = GradientSparsifyHook(keep_ratio=1.0)
    grad = torch.randn(50)
    out = hook.on_backward_smash(grad, 0)
    assert torch.allclose(out.abs().sum(), grad.abs().sum())


def test_gradient_sparsify_partition_filter():
    """partition_indices restricts which cut points are sparsified."""
    hook = GradientSparsifyHook(keep_ratio=0.01, partition_indices=[1])
    grad = torch.randn(100)
    # partition 0: pass-through
    out0 = hook.on_backward_smash(grad, 0)
    assert torch.equal(out0, grad)
    # partition 1: sparsified
    out1 = hook.on_backward_smash(grad, 1)
    assert (out1 != 0).float().mean().item() < 0.15


def test_gradient_sparsify_invalid_ratio():
    with pytest.raises(ValueError):
        GradientSparsifyHook(keep_ratio=0.0)
    with pytest.raises(ValueError):
        GradientSparsifyHook(keep_ratio=1.5)


def test_gradient_sparsify_training_runs():
    schedule = StandardSchedule(hooks=[GradientSparsifyHook(keep_ratio=0.1)])
    executor = LocalExecutor(schedule=schedule)
    history = ts.run(make_model(), loader, n=3, epochs=2,
                     optimizer="adamw", criterion="cross_entropy", executor=executor)
    assert all(h["loss"] < float("inf") for h in history)


def test_gradient_sparsify_with_gpipe():
    schedule = GPipeSchedule(n_micro=2, hooks=[GradientSparsifyHook(keep_ratio=0.2)])
    executor = LocalExecutor(schedule=schedule)
    history = ts.run(make_model(), loader, n=3, epochs=2,
                     optimizer="adamw", criterion="cross_entropy", executor=executor)
    assert all(h["loss"] < float("inf") for h in history)


# ── FixedDelayPolicy ─────────────────────────────────────────────────────────

def test_fixed_delay_returns_correct_values():
    policy = FixedDelayPolicy(forward_delays={1: 0.05}, backward_delays={0: 0.02})
    assert policy.delay(1, "forward") == pytest.approx(0.05)
    assert policy.delay(0, "backward") == pytest.approx(0.02)
    assert policy.delay(0, "forward") == pytest.approx(0.0)
    assert policy.delay(2, "forward") == pytest.approx(0.0)


def test_fixed_delay_unified_delays():
    policy = FixedDelayPolicy(delays={0: 0.1, 2: 0.2})
    assert policy.delay(0, "forward") == pytest.approx(0.1)
    assert policy.delay(0, "backward") == pytest.approx(0.1)
    assert policy.delay(2, "forward") == pytest.approx(0.2)
    assert policy.delay(1, "forward") == pytest.approx(0.0)


def test_fixed_delay_slows_training():
    """Training with a straggler takes longer than without."""
    t0 = time.perf_counter()
    ts.run(make_model(), loader, n=2, epochs=1,
           optimizer="adam", criterion="cross_entropy")
    t_normal = time.perf_counter() - t0

    policy = FixedDelayPolicy(forward_delays={0: 0.05})
    schedule = StandardSchedule(straggler_policy=policy)
    executor = LocalExecutor(schedule=schedule)

    t0 = time.perf_counter()
    ts.run(make_model(), loader, n=2, epochs=1,
           optimizer="adam", criterion="cross_entropy", executor=executor)
    t_delayed = time.perf_counter() - t0

    # 2 batches × 0.05s = 0.1s minimum added delay
    assert t_delayed > t_normal + 0.05


# ── RandomDelayPolicy ─────────────────────────────────────────────────────────

def test_random_delay_zero_prob_never_delays():
    policy = RandomDelayPolicy(prob=0.0, max_delay_s=1.0, seed=42)
    for _ in range(50):
        assert policy.delay(0, "forward") == 0.0


def test_random_delay_prob_one_always_delays():
    policy = RandomDelayPolicy(prob=1.0, max_delay_s=0.5, seed=42)
    for _ in range(20):
        assert policy.delay(0, "forward") > 0.0


def test_random_delay_partition_filter():
    policy = RandomDelayPolicy(prob=1.0, max_delay_s=0.1,
                               partition_indices=[1], seed=0)
    assert policy.delay(0, "forward") == 0.0
    assert policy.delay(1, "forward") > 0.0


def test_random_delay_phase_filter():
    policy = RandomDelayPolicy(prob=1.0, max_delay_s=0.1,
                               phases=["backward"], seed=0)
    assert policy.delay(0, "forward") == 0.0
    assert policy.delay(0, "backward") > 0.0


def test_random_delay_reproducible_with_seed():
    p1 = RandomDelayPolicy(prob=0.5, max_delay_s=0.2, seed=99)
    p2 = RandomDelayPolicy(prob=0.5, max_delay_s=0.2, seed=99)
    results = [(p1.delay(0, "forward"), p2.delay(0, "forward")) for _ in range(20)]
    assert all(a == b for a, b in results)


def test_random_delay_invalid_prob():
    with pytest.raises(ValueError):
        RandomDelayPolicy(prob=1.5)


def test_straggler_with_gpipe():
    policy = FixedDelayPolicy(delays={0: 0.01})
    schedule = GPipeSchedule(n_micro=2, straggler_policy=policy)
    executor = LocalExecutor(schedule=schedule)
    history = ts.run(make_model(), loader, n=3, epochs=1,
                     optimizer="adam", criterion="cross_entropy", executor=executor)
    assert all(h["loss"] < float("inf") for h in history)


# ── Combined: hooks + straggler ───────────────────────────────────────────────

def test_gradient_hook_and_straggler_combined():
    schedule = StandardSchedule(
        hooks=[GradientSparsifyHook(keep_ratio=0.2), ts.NoPeekHook(lambda_=0.01)],
        straggler_policy=RandomDelayPolicy(prob=0.3, max_delay_s=0.01, seed=7),
    )
    executor = LocalExecutor(schedule=schedule)
    history = ts.run(make_model(), loader, n=3, epochs=2,
                     optimizer="adamw", criterion="cross_entropy", executor=executor)
    assert all(h["loss"] < float("inf") for h in history)
