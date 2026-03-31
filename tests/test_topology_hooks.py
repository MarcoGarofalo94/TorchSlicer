"""Tests for Phase 4: topology variants and activation hooks."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
import torchslicer as ts
from torchslicer.topology import BaseSplitTopology, PipelineTopology, UShapedTopology
from torchslicer.hooks import ActivationHook, DPNoiseHook, NoPeekHook, distance_correlation
from torchslicer.pipeline import StandardSchedule, GPipeSchedule
from torchslicer.executors.local import LocalExecutor


def _net():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def _loader(n=16):
    X = torch.randn(n, 8)
    y = torch.randint(0, 4, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=4)


# ── PipelineTopology ───────────────────────────────────────────────────────────

def test_pipeline_topology_assign():
    sliced = ts.slice(_net(), n=3)
    topo   = PipelineTopology()
    roles  = topo.assign(sliced.partitions)
    assert "stage" in roles
    assert len(roles["stage"]) == 3


def test_pipeline_topology_loss_owner():
    topo = PipelineTopology()
    assert topo.loss_owner([]) == "stage"


def test_pipeline_topology_describe():
    assert "stage" in PipelineTopology().describe()


def test_pipeline_topology_repr():
    assert "PipelineTopology" in repr(PipelineTopology())


# ── UShapedTopology ────────────────────────────────────────────────────────────

def test_u_shaped_assign_3_partitions():
    sliced = ts.slice(_net(), n=3)
    topo   = UShapedTopology()
    roles  = topo.assign(sliced.partitions)
    assert len(roles["client"]) == 2   # bottom + head
    assert len(roles["server"]) == 1   # middle


def test_u_shaped_assign_4_partitions():
    net    = nn.Sequential(*[nn.Linear(4, 4) for _ in range(8)])
    sliced = ts.slice(net, n=4)
    topo   = UShapedTopology(n_client_bottom=1, n_client_head=1)
    roles  = topo.assign(sliced.partitions)
    assert len(roles["client"]) == 2
    assert len(roles["server"]) == 2


def test_u_shaped_loss_owner():
    assert UShapedTopology().loss_owner([]) == "client"


def test_u_shaped_too_few_partitions_raises():
    sliced = ts.slice(_net(), n=2)
    topo   = UShapedTopology()   # needs n_client_bottom=1 + n_client_head=1 + ≥1 server
    with pytest.raises(ValueError, match="at least"):
        topo.assign(sliced.partitions)


def test_u_shaped_bad_n_client_raises():
    with pytest.raises(ValueError):
        UShapedTopology(n_client_bottom=0)


def test_u_shaped_describe_contains_client_server():
    desc = UShapedTopology().describe()
    assert "client" in desc and "server" in desc


def test_topology_attached_to_sliced_model():
    sliced = ts.slice(_net(), n=3)
    sliced.topology = ts.UShapedTopology()
    roles  = sliced.topology.assign(sliced.partitions)
    assert "client" in roles


# ── DPNoiseHook ────────────────────────────────────────────────────────────────

def test_dp_noise_adds_noise():
    hook   = DPNoiseHook(sigma=1.0)
    x      = torch.zeros(4, 8)
    raw    = torch.zeros(4, 8)
    result = hook.on_forward_smash(x, raw, partition_idx=0)
    assert not torch.allclose(result, x)


def test_dp_noise_zero_sigma_passthrough():
    hook   = DPNoiseHook(sigma=0.0)
    x      = torch.randn(4, 8)
    result = hook.on_forward_smash(x, x, partition_idx=0)
    assert torch.allclose(result, x)


def test_dp_noise_partition_filter():
    hook   = DPNoiseHook(sigma=1.0, partition_indices=[1])
    x      = torch.zeros(4, 8)
    # Partition 0 should be a passthrough
    assert torch.allclose(hook.on_forward_smash(x, x, 0), x)
    # Partition 1 should be noisy
    assert not torch.allclose(hook.on_forward_smash(x, x, 1), x)


def test_dp_noise_no_aux_loss():
    hook = DPNoiseHook(sigma=0.1)
    hook.on_forward_smash(torch.randn(4, 8), torch.randn(4, 8), 0)
    assert hook.pop_aux_loss() is None


def test_dp_noise_negative_sigma_raises():
    with pytest.raises(ValueError):
        DPNoiseHook(sigma=-1.0)


def test_dp_noise_training_runs():
    schedule = StandardSchedule(hooks=[DPNoiseHook(sigma=0.05)])
    history  = ts.run(_net(), _loader(), n=2, epochs=1,
                      executor=LocalExecutor(schedule=schedule))
    assert history[0]["loss"] < float("inf")


# ── distance_correlation ───────────────────────────────────────────────────────

def test_dcor_identical_tensors_is_one():
    x    = torch.randn(16, 4)
    dcor = distance_correlation(x, x)
    assert abs(dcor.item() - 1.0) < 1e-4


def test_dcor_constant_tensor_is_zero():
    a = torch.randn(16, 4)
    b = torch.ones(16, 4)
    dcor = distance_correlation(a, b)
    assert dcor.item() == 0.0


def test_dcor_in_unit_interval():
    a    = torch.randn(16, 4)
    b    = torch.randn(16, 4)
    dcor = distance_correlation(a, b)
    assert 0.0 <= dcor.item() <= 1.0 + 1e-6


# ── NoPeekHook ────────────────────────────────────────────────────────────────

def test_nopeak_does_not_modify_smashed():
    hook   = NoPeekHook(lambda_=0.1)
    x      = torch.randn(8, 4, requires_grad=True)
    raw    = torch.randn(8, 8)
    result = hook.on_forward_smash(x, raw, partition_idx=0)
    assert result is x   # tensor unchanged


def test_nopeak_produces_aux_loss():
    hook = NoPeekHook(lambda_=0.1)
    x    = torch.randn(8, 4, requires_grad=True)
    raw  = torch.randn(8, 8)
    hook.on_forward_smash(x, raw, 0)
    loss = hook.pop_aux_loss()
    assert loss is not None
    assert loss.item() >= 0.0


def test_nopeak_pop_resets_state():
    hook = NoPeekHook(lambda_=0.1)
    x    = torch.randn(8, 4, requires_grad=True)
    raw  = torch.randn(8, 8)
    hook.on_forward_smash(x, raw, 0)
    hook.pop_aux_loss()
    assert hook.pop_aux_loss() is None   # state reset


def test_nopeak_skips_non_default_partition():
    hook = NoPeekHook(lambda_=0.1, partition_indices=[0])
    x    = torch.randn(8, 4, requires_grad=True)
    raw  = torch.randn(8, 8)
    hook.on_forward_smash(x, raw, partition_idx=1)   # should be skipped
    assert hook.pop_aux_loss() is None


def test_nopeak_zero_lambda_no_aux_loss():
    hook = NoPeekHook(lambda_=0.0)
    hook.on_forward_smash(torch.randn(8, 4), torch.randn(8, 8), 0)
    assert hook.pop_aux_loss() is None


def test_nopeak_training_runs():
    schedule = StandardSchedule(hooks=[NoPeekHook(lambda_=0.05)])
    history  = ts.run(_net(), _loader(), n=2, epochs=1,
                      executor=LocalExecutor(schedule=schedule))
    assert history[0]["loss"] < float("inf")


def test_nopeak_combined_with_dp_noise():
    schedule = StandardSchedule(hooks=[
        DPNoiseHook(sigma=0.01),
        NoPeekHook(lambda_=0.05),
    ])
    history = ts.run(_net(), _loader(), n=2, epochs=1,
                     executor=LocalExecutor(schedule=schedule))
    assert history[0]["loss"] < float("inf")


# ── hooks with GPipeSchedule ──────────────────────────────────────────────────

def test_dp_noise_with_gpipe():
    schedule = GPipeSchedule(n_micro=2, hooks=[DPNoiseHook(sigma=0.05)])
    history  = ts.run(_net(), _loader(), n=2, epochs=1, use_gpipe=True,
                      executor=LocalExecutor(schedule=schedule))
    assert history[0]["loss"] < float("inf")
