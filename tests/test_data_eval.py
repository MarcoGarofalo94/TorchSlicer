"""M11 datasets, the config ``data`` section and evaluation: test metrics equal an unsplit
model's, evaluating changes nothing in training, and sessions report per-client results."""
import copy
import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torchslicer as ts
from torchslicer.data import load_dataset

SGD = {"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}
CE = {"name": "CrossEntropyLoss", "params": {}}


def _model(bn=False):
    torch.manual_seed(0)
    layers = [nn.Linear(8, 16)] + ([nn.BatchNorm1d(16), nn.Dropout(0.3)] if bn else []) + \
             [nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)]
    return nn.Sequential(*layers)


def _loader(seed, n=64, b=16, shuffle=False):
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.randn(n, 8, generator=g), torch.randint(0, 4, (n,), generator=g))
    return DataLoader(ds, batch_size=b, shuffle=shuffle)


def _cfg(**sl):
    cfg = ts.RunConfig()
    cfg.training.optimizer, cfg.training.criterion, cfg.training.epochs = SGD, CE, 2
    cfg.logging.enabled = False
    cfg.network.device = "cpu"
    for k, v in sl.items():
        setattr(cfg.sl, k, v)
    return cfg


def _manual_eval(model, loader):
    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            total += nn.functional.cross_entropy(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total / n, correct / n


def test_evaluate_matches_the_unsplit_model():
    train, test = _loader(1), _loader(2, n=40, b=16)       # last test batch is smaller
    hist = ts.slice(_model(), n=3, cuts=["1", "3"]).train(train, run_config=_cfg(), eval_data=test)
    ref = _model()
    opt = torch.optim.SGD(ref.parameters(), **SGD["params"])
    for epoch in range(2):
        ref.train()
        for x, y in train:
            opt.zero_grad()
            nn.functional.cross_entropy(ref(x), y).backward()
            opt.step()
        loss, acc = _manual_eval(ref, test)
        assert hist[epoch]["test_loss"] == pytest.approx(loss, abs=1e-6)
        assert hist[epoch]["test_accuracy"] == pytest.approx(acc, abs=1e-9)
        assert hist[epoch]["test_samples"] == 40


def test_evaluating_does_not_change_training():
    # BatchNorm running stats and dropout RNG must be untouched by evaluation.
    train, test = _loader(1, shuffle=False), _loader(2)
    def run(eval_data):
        torch.manual_seed(123)
        return ts.slice(_model(bn=True), n=2, cuts=["3"]).train(
            train, run_config=_cfg(), eval_data=eval_data, epochs=3)
    plain, evaluated = run(None), run(test)
    assert [h["loss"] for h in plain] == [h["loss"] for h in evaluated]
    assert "test_loss" in evaluated[0] and "test_loss" not in plain[0]


def test_eval_every():
    hist = ts.slice(_model(), n=2, cuts=["1"]).train(_loader(1), run_config=_cfg(), epochs=4,
                                                    eval_data=_loader(2), eval_every=2)
    assert ["test_loss" in h for h in hist] == [False, True, False, True]


def test_session_eval_per_client_and_after_fedavg():
    test = _loader(9)
    clients = lambda: [list(_loader(s, n=32)) for s in (3, 4, 5)]
    keep = ts.ClientSession(clients(), mode="sequential", handoff="keep")
    hist = ts.slice(_model(), n=2, cuts=["1"]).train(keep, run_config=_cfg(setting="vanilla"),
                                                    eval_data=test)
    per = hist[-1]["test_accuracy_clients"]
    assert len(per) == 3 and hist[-1]["test_accuracy"] == pytest.approx(sum(per) / 3, abs=1e-6)
    fed = ts.ClientSession(clients(), mode="parallel")
    hist = ts.slice(_model(), n=2, cuts=["1"]).train(fed, run_config=_cfg(setting="vanilla"),
                                                    eval_data=test, algorithm=ts.SplitFedAlgorithm())
    assert "test_accuracy" in hist[-1] and "test_accuracy_clients" not in hist[-1]


def test_session_eval_leaves_training_unchanged():
    test = _loader(9)
    def run(eval_data):
        s = ts.ClientSession([list(_loader(k, n=32)) for k in (3, 4)], mode="parallel")
        return ts.slice(_model(), n=2, cuts=["1"]).train(
            s, run_config=_cfg(setting="vanilla"), eval_data=eval_data, epochs=3)
    assert [h["loss"] for h in run(None)] == [h["loss"] for h in run(test)]


def test_synthetic_and_npz_datasets(tmp_path):
    tr, te = load_dataset("synthetic", train=True), load_dataset("synthetic", train=False)
    assert (len(tr), len(te)) == (1024, 256) and tuple(tr[0][0].shape) == (3, 32, 32)
    assert torch.equal(load_dataset("synthetic").images, tr.images)          # seeded
    x = np.random.default_rng(0).integers(0, 256, (20, 8, 8, 3), dtype=np.uint8)
    y = np.arange(20) % 4
    np.savez(tmp_path / "d.npz", x=x, y=y)
    ds = load_dataset("npz", path=str(tmp_path / "d.npz"))
    img, label = ds[3]
    assert tuple(img.shape) == (3, 8, 8) and int(label) == 3
    assert torch.allclose(img, torch.from_numpy(x[3]).permute(2, 0, 1).float() / 255)
    with pytest.raises(KeyError, match="available"):
        load_dataset("imagenet")


def _data_cfg(tmp_path, **data):
    cfg = _cfg()
    cfg.data.dataset = "synthetic"
    cfg.data.params = {"shape": [8], "classes": 4, "n": 256}
    for k, v in data.items():
        setattr(cfg.data, k, v)
    cfg.logging.dir = str(tmp_path)
    return cfg


def test_run_builds_clients_and_evaluates_from_the_config(tmp_path):
    cfg = _data_cfg(tmp_path, partition="dirichlet", partition_params={"alpha": 0.5}, batch_size=16)
    cfg.sl.setting, cfg.sl.clients, cfg.sl.client_mode = "vanilla", 3, "parallel"
    cfg.logging.enabled = True
    hist = ts.run(lambda: _model(), run_config=cfg, verbose=False, cuts=["1"])
    assert len(hist) == 2 and 0.0 <= hist[-1]["test_accuracy"] <= 1.0
    split = ts.data.ClientSplit.load(tmp_path / cfg.run_id / "client_split.json")
    assert split.method == "dirichlet" and sum(split.sizes()) == 256
    again = ts.run(lambda: _model(), run_config=copy.deepcopy(cfg), verbose=False, cuts=["1"])
    assert [h["loss"] for h in again] == [h["loss"] for h in hist]           # seeded end to end


def test_config_data_errors(tmp_path):
    cfg = _data_cfg(tmp_path)
    with pytest.raises(ValueError, match="either data_fn or data.dataset"):
        ts.run(lambda: _model(), lambda: _loader(1), run_config=cfg, verbose=False)
    with pytest.raises(ValueError, match="needs a data_fn"):
        ts.run(lambda: _model(), run_config=_cfg(), verbose=False)


@pytest.mark.parametrize("topology,setting,clients", [
    ("centralized", "pipeline", 1), ("centralized", "vanilla", 1), ("p2p", "vanilla", 1),
    ("p2p", "u_shaped", 1), ("centralized", "vanilla", 3),
])
def test_distributed_runs_evaluate_like_the_local_one(tmp_path, topology, setting, clients):
    # the data owner fetches every slice's weights and evaluates them on its test set:
    # the same numbers as the in-process run, which trains the same way
    def run(topo):
        cfg = _data_cfg(tmp_path, batch_size=16)
        cfg.deploy.topology, cfg.sl.setting, cfg.discovery.n_workers = topo, setting, 3
        cfg.sl.clients, cfg.sl.client_mode = clients, "parallel"
        return ts.run(lambda: _model(), run_config=cfg, verbose=False, cuts=["1", "3"])
    ref, got = run("local"), run(topology)
    for r, g in zip(ref, got):
        for key in ("loss", "test_loss", "test_accuracy", "test_samples"):
            assert g[key] == pytest.approx(r[key], abs=1e-6), key
        assert g["bytes_eval"] > 0 and g["bytes_total"] == r["bytes_total"]   # eval isn't training traffic


def test_executors_without_evaluation_refuse_eval_data():
    cfg = _cfg()
    cfg.offload.mode = "cpu"
    with pytest.raises(ValueError, match="doesn't support evaluation"):
        ts.slice(_model(), n=2, cuts=["1"]).train(_loader(1), run_config=cfg, eval_data=_loader(2))


@pytest.mark.skipif(not (ts.data.datasets.Path("data/MNIST/raw").exists()), reason="MNIST not downloaded")
def test_mnist_loads_normalized():
    tr = load_dataset("mnist", root="data", download=False)
    assert len(tr) == 60000 and tuple(tr[0][0].shape) == (1, 28, 28)
    xs = torch.stack([tr[i][0] for i in range(0, len(tr), 50)])
    assert abs(xs.mean().item()) < 0.05 and abs(xs.std().item() - 1) < 0.05


# ── heterogeneous client profiles (simulated) ─────────────────────────────────

FAST = {"name": "fast", "compute_scale": 0.0, "bandwidth_mbps": 100.0, "latency_ms": 5.0}
SLOW = {"name": "slow", "compute_scale": 0.0, "bandwidth_mbps": 10.0, "latency_ms": 50.0}


def test_assign_profiles_is_seeded_and_follows_shares():
    from torchslicer.sl.metrics import assign_profiles
    a = assign_profiles([dict(FAST, share=3), dict(SLOW, share=1)], 4000, seed=1)
    assert a == assign_profiles([dict(FAST, share=3), dict(SLOW, share=1)], 4000, seed=1)
    assert abs(a.count(0) / 4000 - 0.75) < 0.03
    with pytest.raises(ValueError, match="unknown keys"):
        assign_profiles([{"bandwith_mbps": 1}], 2)


def _profiled_run(mode, profiles, epochs=1):
    cfg = _cfg(setting="vanilla", client_profiles=profiles, profile_seed=0)
    s = ts.ClientSession([list(_loader(k, n=32)) for k in (3, 4)], mode=mode)
    return ts.slice(_model(), n=2, cuts=["1"]).train(s, run_config=cfg, epochs=epochs)


def test_client_times_follow_each_clients_link():
    # compute_scale 0: a client's time is exactly its link time. Per batch of 16 the client
    # sends a 16x16 fp32 activation (1024 B) and the labels (128 B) and gets a 1024 B
    # gradient back: 2176 B and 3 messages; 2 batches per client.
    h = _profiled_run("parallel", [FAST, SLOW])[0]
    tiers, times = h["client_tiers"], h["sim_client_time_s"]
    for k in (0, 1):
        p = [FAST, SLOW][tiers[k]]
        expect = 2 * (2176 * 8 / (p["bandwidth_mbps"] * 1e6) + 3 * p["latency_ms"] / 1e3)
        assert times[k] == pytest.approx(expect, abs=1e-6)


@pytest.mark.parametrize("mode", ["parallel", "sequential"])
def test_round_time_is_slowest_or_sum_plus_server(mode):
    h = _profiled_run(mode, [dict(FAST, compute_scale=2.0), dict(SLOW, compute_scale=8.0)])[0]
    t = list(h["sim_client_time_s"].values())
    clients = max(t) if mode == "parallel" else sum(t)
    assert h["sim_round_s"] == pytest.approx(clients + h["server_compute_s"], abs=1e-5)


def test_profiles_do_not_change_training():
    plain = _profiled_run("parallel", [], epochs=2)
    prof = _profiled_run("parallel", [FAST, SLOW], epochs=2)
    assert [e["loss"] for e in plain] == [e["loss"] for e in prof]
    assert "sim_round_s" not in plain[0] and "client_tiers" not in prof[1]


class _ToClassesFirst(nn.Module):
    def forward(self, x):                      # (B, T, V) -> (B, V, T) for CrossEntropyLoss
        return x.transpose(1, 2)


def test_token_accuracy_for_language_model_outputs():
    torch.manual_seed(0)
    vocab, t = 11, 6
    model = nn.Sequential(nn.Embedding(vocab, 8), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, vocab),
                          _ToClassesFirst())
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, vocab, (24, t), generator=g)
    loader = DataLoader(TensorDataset(x, x.roll(-1, 1)), batch_size=8)
    hist = ts.slice(model, n=2, cuts=["1"]).train(loader, run_config=_cfg(), epochs=1, eval_data=loader)
    with torch.no_grad():
        out = torch.cat([model(b) for b, _ in loader])
    acc = (out.argmax(1) == x.roll(-1, 1)).float().mean().item()   # same weights: trained in place
    assert hist[0]["test_accuracy"] == pytest.approx(acc, abs=1e-6)


@pytest.mark.skipif(not ts.data.datasets.Path("data/shakespeare/input.txt").exists(),
                    reason="tiny-shakespeare not downloaded")
def test_shakespeare_windows_and_natural_split():
    tr = load_dataset("shakespeare", root="data", download=False)
    x, y = tr[5]
    assert x.shape == y.shape == (64,) and torch.equal(x[1:], y[:-1])
    split = ts.data.partition(tr, 20, "natural", seed=0)
    owner = {}
    for k, idx in enumerate(split.indices):        # every speaking role on one client only
        for role in np.unique(tr.groups[idx]):
            assert owner.setdefault(role, k) == k
    assert len(owner) == len(np.unique(tr.groups))
