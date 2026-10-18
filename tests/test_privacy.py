"""M12 privacy suite: leakage metrics, attacks, defences.

Requirements: observing a cut never changes training (bit-exact losses); each attack
reproduces its paper's qualitative result on a small setting and its defence lowers it;
NoPeek trains exactly on ``task loss + alpha * dcor``; unsupported executors refuse.
"""
import itertools

import pytest
import torch
import torch.nn as nn

import torchslicer as ts
from torchslicer.privacy import (DirectionLabelAttack, DistanceCorrelation, InversionAttack, NoPeek,
                                 NormLabelAttack, auc, distance_correlation, psnr, ssim)

SGD = {"name": "SGD", "params": {"lr": 0.1}}
CE = {"name": "CrossEntropyLoss", "params": {}}


def _binary_data(n=1024, batch=64, seed=0):
    """Imbalanced binary task (~10 % positives), as in Li et al.'s setting."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 20, generator=g)
    s = x @ torch.randn(20, generator=g)
    y = (s > s.quantile(0.9)).long()
    return [(x[i:i + batch], y[i:i + batch]) for i in range(0, n, batch)]


def _mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))


def _train(privacy=None, hooks=None, data=None, epochs=2, model=None, **kw):
    sliced = ts.slice(model or _mlp(), n=2, cut_hooks=hooks)
    return sliced.train(data or _binary_data(), SGD, CE, epochs=epochs, privacy=privacy, **kw)


def _images(n=256, batch=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    # smooth 1x8x8 images: a low-resolution pattern upsampled, so they can be reconstructed
    low = torch.randn(n, 1, 4, 4, generator=g)
    x = nn.functional.interpolate(low, size=(8, 8), mode="bilinear", align_corners=False)
    y = (low.mean((1, 2, 3)) > 0).long()
    return [(x[i:i + batch], y[i:i + batch]) for i in range(0, n, batch)]


def _cnn():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 64, 2))


# ── metrics ─────────────────────────────────────────────────────────────────────

def test_auc_matches_pairwise_definition():
    g = torch.Generator().manual_seed(0)
    s = torch.randint(0, 5, (40,), generator=g).float()     # many ties
    y = torch.rand(40, generator=g) > 0.6
    pairs = [(1.0 if a > b else 0.5 if a == b else 0.0)
             for a, b in itertools.product(s[y].tolist(), s[~y].tolist())]
    assert auc(s, y) == pytest.approx(sum(pairs) / len(pairs))
    assert auc(torch.arange(4.0), torch.tensor([0, 0, 1, 1]).bool()) == 1.0
    assert auc(torch.ones(3), torch.ones(3).bool()) != auc(torch.ones(3), torch.ones(3).bool())  # nan


def test_distance_correlation():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(200, 5, generator=g)
    assert distance_correlation(x, x).item() == pytest.approx(1.0, abs=1e-5)
    assert distance_correlation(x, 3 * x @ torch.linalg.qr(torch.randn(5, 5, generator=g))[0] + 1).item() \
        == pytest.approx(1.0, abs=1e-4)
    assert distance_correlation(x, torch.randn(200, 5, generator=g)).item() < 0.3
    z = x.clone().requires_grad_(True)
    distance_correlation(x, z).backward()
    assert torch.isfinite(z.grad).all()


def test_psnr_ssim():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(4, 3, 16, 16, generator=g)
    assert ssim(x, x) == pytest.approx(1.0, abs=1e-5)
    noisy = x + 0.1 * torch.randn(x.shape, generator=g)
    assert psnr(noisy, x, data_range=1.0) == pytest.approx(20.0, abs=0.5)
    # uniform images (var 1/12) + noise var s^2: SSIM ~ 2 var / (2 var + s^2)
    assert ssim(noisy, x) == pytest.approx(2 / 12 / (2 / 12 + 0.01), abs=0.02)
    assert ssim(x + 0.3 * torch.randn(x.shape, generator=g), x) == pytest.approx(2 / 12 / (2 / 12 + 0.09), abs=0.03)
    assert psnr(x + 0.01, x, data_range=1.0) > psnr(noisy, x, data_range=1.0)


# ── observers don't change training ─────────────────────────────────────────────

def _all_attacks():
    return [NormLabelAttack(), DirectionLabelAttack(), DistanceCorrelation()]


@pytest.mark.parametrize("gpipe", [False, True])
def test_attacks_do_not_change_losses(gpipe):
    kw = {"use_gpipe": True, "n_micro_batches": 2} if gpipe else {}
    base = _train(**kw)
    seen = _train(privacy=_all_attacks(), **kw)
    assert [e["loss"] for e in base] == [e["loss"] for e in seen]
    assert all(k.startswith("leak_") for k in set(seen[-1]) - set(base[-1]))


def test_inversion_does_not_change_losses_and_leaves_slices_intact():
    data = _images()
    base = ts.slice(_cnn(), cuts=["1"]).train(data, SGD, CE, epochs=2)
    sliced = ts.slice(_cnn(), cuts=["1"])
    att = InversionAttack(aux=torch.cat([x for x, _ in _images(seed=1)]), n_aux=128, batch_size=32)
    h = sliced.train(data, SGD, CE, epochs=2, privacy=[att])
    assert [e["loss"] for e in base] == [e["loss"] for e in h]
    assert "leak_inversion_psnr" in h[-1] and "leak_inversion_ssim" in h[-1]
    assert h[-1]["leak_inversion_psnr"] > h[0]["leak_inversion_psnr"]   # the decoder learns


def test_inversion_defaults_to_eval_data_and_needs_aux():
    data = _images()
    h = ts.slice(_cnn(), cuts=["1"]).train(data, SGD, CE, epochs=1, eval_data=_images(seed=2),
                                          privacy=[InversionAttack(n_aux=64)])
    assert "leak_inversion_psnr" in h[0]
    with pytest.raises(ValueError, match="auxiliary inputs"):
        ts.slice(_cnn(), cuts=["1"]).train(data, SGD, CE, epochs=1, privacy=[InversionAttack()])


# ── qualitative results ─────────────────────────────────────────────────────────

def test_label_leakage_and_gradient_noise():
    """Li et al. 2022: near-perfect leak AUC without protection; noisy gradients lower it."""
    h = _train(privacy=[NormLabelAttack(), DirectionLabelAttack()])[-1]
    assert h["leak_label_norm_auc"] > 0.95 and h["leak_label_direction_auc"] > 0.95
    noisy = _train(privacy=[NormLabelAttack(), DirectionLabelAttack()],
                   hooks={0: ts.GaussianNoise(std=0.05, grad=True)})[-1]
    assert noisy["leak_label_norm_auc"] < 0.7 and noisy["leak_label_direction_auc"] < 0.7


def test_label_attack_needs_class_labels():
    data = [(x, nn.functional.one_hot(y, 2).float()) for x, y in _binary_data()]
    with pytest.raises(ValueError, match="one class label per sample"):
        ts.slice(_mlp(), n=2).train(data, SGD, {"name": "MSELoss", "params": {}}, epochs=1,
                                    privacy=[NormLabelAttack()])


# ── NoPeek ──────────────────────────────────────────────────────────────────────

def test_nopeek_alpha_zero_is_bit_exact():
    assert [e["loss"] for e in _train()] == [e["loss"] for e in _train(privacy=[NoPeek(alpha=0.0)])]


def test_nopeek_matches_reference_loop():
    """The client slice trains on CE + alpha * dcor(x, z); the server on CE only."""
    alpha, data = 0.7, _binary_data(n=256)
    model = _mlp()
    ts.slice(model, cuts=["1"]).train(data, SGD, CE, epochs=1, privacy=[NoPeek(alpha=alpha)])

    ref = _mlp()
    client, server = ref[:2], ref[2:]
    opt_c = torch.optim.SGD(client.parameters(), lr=0.1)
    opt_s = torch.optim.SGD(server.parameters(), lr=0.1)
    for x, y in data:
        z = client(x)
        zs = z.detach().requires_grad_(True)
        nn.functional.cross_entropy(server(zs), y).backward()
        opt_s.step(); opt_s.zero_grad()
        (z * zs.grad).sum().backward(retain_graph=True)
        (alpha * distance_correlation(x, z)).backward()
        opt_c.step(); opt_c.zero_grad()
    assert not torch.allclose(model[0].weight, _mlp()[0].weight)
    for a, b in zip(model.parameters(), ref.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_nopeek_lowers_distance_correlation():
    att = lambda: [DistanceCorrelation()]
    plain = _train(privacy=att(), epochs=3)[-1]["leak_dcor_value"]
    defended = _train(privacy=att() + [NoPeek(alpha=1.0)], epochs=3)[-1]["leak_dcor_value"]
    assert defended < plain - 0.05


# ── config and refusals ─────────────────────────────────────────────────────────

def test_config_section():
    cfg = ts.RunConfig()
    cfg.logging.enabled = False
    cfg.training.optimizer, cfg.training.criterion, cfg.training.epochs = SGD, CE, 1
    cfg.privacy.attacks = [{"name": "label_norm", "cut": 0}, {"name": "label_norm", "cut": 1}]
    cfg.privacy.defences = [{"name": "nopeek", "params": {"alpha": 0.1}}]
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
    h = ts.slice(model, n=3).train(_binary_data(), run_config=cfg)
    assert {"leak_label_norm_cut0_auc", "leak_label_norm_cut1_auc"} <= set(h[0])


def test_resolution_errors():
    with pytest.raises(ValueError, match="unknown privacy attack"):
        ts.privacy.resolve_privacy(None, ts.PrivacyConfig(attacks=[{"name": "nope"}]), 2)
    with pytest.raises(ValueError, match="cuts 0..0"):
        ts.privacy.resolve_privacy([NormLabelAttack(cut=1)], None, 2)
    with pytest.raises(ValueError, match="same key"):
        ts.privacy.resolve_privacy([NormLabelAttack(key="a"), DirectionLabelAttack(key="a")], None, 2)
    with pytest.raises(TypeError):
        ts.privacy.resolve_privacy([ts.GaussianNoise(1.0)], None, 2)


def test_distributed_refuses_privacy():
    cfg = ts.RunConfig()
    cfg.privacy.attacks = [{"name": "label_norm"}]
    ex = ts.DistributedExecutor(None, "127.0.0.1:0", run_config=cfg)
    sliced = ts.slice(_mlp(), n=2, executor=ex)
    with pytest.raises(ValueError, match="local executor"):
        sliced.train(_binary_data(), SGD, CE, epochs=1)
    with pytest.raises(ValueError, match="local executor"):
        ts.slice(_mlp(), n=2, executor=ts.DistributedExecutor(None, "127.0.0.1:0")).train(
            _binary_data(), SGD, CE, epochs=1, privacy=[NormLabelAttack()])


def test_offload_refuses_query_attacks_and_runs_label_attacks():
    data = _images()
    with pytest.raises(ValueError, match="paged out"):
        ts.slice(_cnn(), cuts=["1"], executor=ts.OffloadExecutor(device="cpu")).train(
            data, SGD, CE, epochs=1, privacy=[InversionAttack(aux=data[0][0])])
    local = ts.slice(_cnn(), cuts=["1"]).train(data, SGD, CE, epochs=1, privacy=[NormLabelAttack()])
    off = ts.slice(_cnn(), cuts=["1"], executor=ts.OffloadExecutor(device="cpu")).train(
        data, SGD, CE, epochs=1, privacy=[NormLabelAttack()])
    assert off[0]["leak_label_norm_auc"] == pytest.approx(local[0]["leak_label_norm_auc"], abs=1e-6)
