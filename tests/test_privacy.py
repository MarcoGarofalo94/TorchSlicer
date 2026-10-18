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


# ── step 4: client copies, a malicious server, LLM inversion, DP accounting ─────

class _Pool(nn.Module):
    def forward(self, h):
        return h.mean(1)


def _tokens(n=128, vocab=50, length=8, batch=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(vocab, (n, length), generator=g)
    return [(ids[i:i + batch], (ids[i:i + batch, 0] % 2).long()) for i in range(0, n, batch)]


def _tiny_lm(vocab=50):
    torch.manual_seed(0)
    return nn.Sequential(nn.Embedding(vocab, 16), nn.Linear(16, 16), nn.Tanh(), _Pool(), nn.Linear(16, 2))


def test_client_copy_attacks_do_not_change_training():
    from torchslicer.privacy import EmbeddingRecovery, UnSplit
    data = _images()
    base = ts.slice(_cnn(), cuts=["1"]).train(data, SGD, CE, epochs=2)
    h = ts.slice(_cnn(), cuts=["1"]).train(data, SGD, CE, epochs=2,
                                           privacy=[UnSplit(every=4, rounds=3, input_steps=5, model_steps=5)])
    assert [e["loss"] for e in base] == [e["loss"] for e in h]
    assert {"leak_unsplit_mse", "leak_unsplit_psnr", "leak_unsplit_clone_mse"} <= set(h[-1])
    lm = _tokens()
    base = ts.slice(_tiny_lm(), cuts=["1"]).train(lm, SGD, CE, epochs=1)
    h = ts.slice(_tiny_lm(), cuts=["1"]).train(lm, SGD, CE, epochs=1,
                                               privacy=[EmbeddingRecovery(steps=20, every=2)])
    assert [e["loss"] for e in base] == [e["loss"] for e in h]


def test_client_copy_is_frozen_and_fresh_reinitializes():
    from torchslicer.privacy import Attack
    copies = {}

    class Grab(Attack):
        name, needs_client = "grab", True

        def observe(self, obs):
            if not copies:
                copies["live"], copies["fresh"] = self.client(), self.client(fresh=True)

    sliced = ts.slice(_cnn(), cuts=["1"])
    sliced.train(_images(), SGD, CE, epochs=1, privacy=[Grab()])
    live, fresh = copies["live"], copies["fresh"]
    assert not any(p.requires_grad for p in live.parameters()) and not live.training
    x = torch.randn(2, 1, 8, 8, requires_grad=True)
    live(x).sum().backward()                  # gradients reach the input through the copy
    assert x.grad is not None
    w_live = next(live.parameters())
    assert not torch.equal(w_live, next(fresh.parameters()))


def test_unsplit_reconstructs_better_than_its_start():
    """Erdogan et al. 2022: coordinate descent on (x~, clone) reconstructs smooth images
    from a shallow cut without data or queries (x~ starts at 0, so MSE < E[x^2])."""
    from torchslicer.privacy import UnSplit
    data = _images()
    h = ts.slice(_cnn(), cuts=["1"]).train(data, SGD, CE, epochs=1,
                                           privacy=[UnSplit(every=2, rounds=20, input_steps=10, model_steps=10,
                                                            lr=0.01, tv=0.01)])
    energy = torch.cat([x for x, _ in data]).pow(2).mean().item()
    assert h[-1]["leak_unsplit_mse"] < 0.7 * energy


def test_embedding_recovery_inverts_a_shallow_cut():
    """Dong et al. 2025 (ER): white-box optimization of dummy embeddings recovers the tokens."""
    from torchslicer.privacy import EmbeddingRecovery, token_f1
    h = ts.slice(_tiny_lm(), cuts=["2"]).train(_tokens(), SGD, CE, epochs=1,
                                               privacy=[EmbeddingRecovery(steps=300, lr=0.05, every=1)])
    assert h[-1]["leak_embedding_recovery_token_acc"] > 0.95
    assert token_f1(torch.tensor([[1, 2, 2]]), torch.tensor([[2, 1, 3]])) == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="token ids"):
        ts.slice(_cnn(), cuts=["1"]).train(_images(), SGD, CE, epochs=1, privacy=[EmbeddingRecovery()])


@pytest.mark.parametrize("gpipe", [False, True])
def test_forged_gradient_replaces_the_honest_one(gpipe):
    """A forging attack's gradient is what the client backpropagates: zeros freeze the
    client slice (plain SGD), the server still trains."""
    from torchslicer.privacy import Attack

    class Zero(Attack):
        name, forges = "zero", True

        def observe(self, obs):
            pass

        def forge(self, obs):
            return torch.zeros_like(obs.tensor)

    kw = {"use_gpipe": True, "n_micro_batches": 2} if gpipe else {}
    model = _mlp()
    before = [p.detach().clone() for p in model.parameters()]
    sliced = ts.slice(model, cuts=["1"])
    sliced.train(_binary_data(), SGD, CE, epochs=1, privacy=[Zero()], **kw)
    after = [p.detach() for p in model.parameters()]     # the slices share the model's parameters
    assert torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])     # client: Linear(20, 32)
    assert not torch.equal(before[-1], after[-1])                                     # server


def test_fsha_hijacks_and_its_control_does_not():
    from torchslicer.privacy import FSHA
    data, aux = _images(), torch.cat([x for x, _ in _images(seed=1)])
    base = ts.slice(_cnn(), cuts=["1"]).train(data, SGD, CE, epochs=1)
    honest = ts.slice(_cnn(), cuts=["1"]).train(data, SGD, CE, epochs=1, privacy=[FSHA(aux=aux, forge=False)])
    hijack = ts.slice(_cnn(), cuts=["1"]).train(data, SGD, CE, epochs=1, privacy=[FSHA(aux=aux)])
    assert [e["loss"] for e in base] == [e["loss"] for e in honest]
    assert [e["loss"] for e in base] != [e["loss"] for e in hijack]
    assert {"leak_fsha_mse", "leak_fsha_psnr", "leak_fsha_ssim"} <= set(hijack[-1])


def test_gdp_accountant_matches_numerical_integration():
    """delta(eps) of mu-GDP = the hockey-stick divergence between N(mu, 1) and N(0, 1)."""
    from torchslicer.privacy import gaussian_epsilon, gdp_delta, gdp_epsilon
    x = torch.linspace(-30, 30, 600_001, dtype=torch.float64)
    dx = (x[1] - x[0]).item()
    pdf = lambda m: torch.exp(-(x - m) ** 2 / 2) / (2 * torch.pi) ** 0.5
    for mu, eps in [(0.5, 0.1), (1.0, 1.0), (2.0, 3.0), (4.0, 8.0)]:
        direct = ((pdf(mu) - torch.exp(torch.tensor(eps, dtype=torch.float64)) * pdf(0.0)).clamp_min(0).sum() * dx).item()
        assert gdp_delta(mu, eps) == pytest.approx(direct, rel=1e-6, abs=1e-12)
    eps = gdp_epsilon(1.0, 1e-5)
    assert gdp_delta(1.0, eps) == pytest.approx(1e-5, rel=1e-6)
    assert eps < (2 * torch.log(torch.tensor(1.25 / 1e-5))).sqrt().item()   # tighter than the classical bound
    assert gaussian_epsilon(2.0, 4, 1e-5) == pytest.approx(gdp_epsilon(2.0, 1e-5))  # mu = 2/2 * sqrt(4)
    assert gaussian_epsilon(1.0, 2, 1e-5) > gaussian_epsilon(1.0, 1, 1e-5) > gaussian_epsilon(2.0, 1, 1e-5)
    assert gaussian_epsilon(0.0, 1, 1e-5) == float("inf")


def test_clip_noise_clips_each_row():
    from torchslicer.sl.hooks import CutContext
    hook = ts.ClipNoise(clip=1.0, noise_multiplier=0.0, grad=True)
    t = torch.stack([torch.full((2, 3), 0.1), torch.full((2, 3), 5.0)])
    out = hook.encode(t, CutContext(0, "forward", 0, 2))
    assert torch.equal(out[0], t[0])
    assert out[1].norm().item() == pytest.approx(1.0)
    noisy = ts.ClipNoise(clip=1.0, noise_multiplier=2.0, forward=False, grad=True)
    assert torch.equal(noisy.encode(t, CutContext(0, "forward", 0, 2)), t)
    g = noisy.encode_grad(t, CutContext(0, "backward", 0, 2))
    assert (g - out).std().item() == pytest.approx(2.0, rel=0.5)


def test_dp_accountant_in_training_and_its_checks():
    from torchslicer.privacy import DPAccountant, gaussian_epsilon
    hook = {0: ts.ClipNoise(clip=0.05, noise_multiplier=1.5, forward=False, grad=True)}
    h = _train(privacy=[DPAccountant(delta=1e-5), NormLabelAttack()], hooks=hook, epochs=3)
    assert [e["leak_dp_releases"] for e in h] == [1, 2, 3]
    assert h[-1]["leak_dp_epsilon"] == pytest.approx(gaussian_epsilon(1.5, 3, 1e-5), rel=1e-5)
    for hooks, match in [(None, "ClipNoise"), ({0: ts.GaussianNoise(std=1.0, grad=True)}, "ClipNoise"),
                         ({0: ts.ClipNoise(clip=1.0, noise_multiplier=1.0)}, "grad=True")]:
        with pytest.raises(ValueError, match=match):
            _train(privacy=[DPAccountant()], hooks=hooks, epochs=1)
    h = _train(privacy=[DPAccountant(direction="forward")], epochs=1,
               hooks={0: [ts.ClipNoise(clip=1.0, noise_multiplier=1.0), ts.Quantize(bits=8)]})
    assert h[0]["leak_dp_releases"] == 1


def test_step4_config_names_and_offload_refusal():
    from torchslicer.privacy import UnSplit
    cfg = ts.RunConfig()
    cfg.logging.enabled = False
    cfg.sl.cut_hooks = [{"cut": 0, "name": "clip_noise",
                         "params": {"clip": 0.05, "noise_multiplier": 1.0, "forward": False, "grad": True}}]
    cfg.privacy.attacks = [{"name": "dp", "cut": 0}, {"name": "fsha", "cut": 0, "params": {"forge": False}},
                           {"name": "unsplit", "cut": 0, "params": {"every": 100, "rounds": 1}}]
    data = _images()
    h = ts.slice(_cnn(), cuts=["1"]).train(data, SGD, CE, epochs=1, run_config=cfg, eval_data=_images(seed=3))
    assert {"leak_dp_epsilon", "leak_fsha_psnr", "leak_unsplit_mse"} <= set(h[0])
    with pytest.raises(ValueError, match="paged out"):
        ts.slice(_cnn(), cuts=["1"], executor=ts.OffloadExecutor(device="cpu")).train(
            data, SGD, CE, epochs=1, privacy=[UnSplit()])


def test_forged_gradient_scale_under_gpipe():
    """forge returns the gradient of a per-(micro-)batch mean loss; with the 1/M scale the
    client gets the same update with and without micro-batches."""
    from torchslicer.privacy import Attack

    class MeanSum(Attack):
        name, forges = "mean_sum", True

        def observe(self, obs):
            pass

        def forge(self, obs):                    # d/dz mean_i sum_j z_ij
            return torch.full_like(obs.tensor, 1.0 / obs.tensor.shape[0])

    weights = []
    for kw in ({}, {"use_gpipe": True, "n_micro_batches": 4}):
        model = _mlp()
        ts.slice(model, cuts=["1"]).train(_binary_data(), SGD, CE, epochs=1, privacy=[MeanSum()], **kw)
        weights.append(model[0].weight.detach().clone())
    assert torch.allclose(weights[0], weights[1], atol=1e-6)
