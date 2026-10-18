"""OffloadExecutor (M5): slices paged between a slow tier and the device must train exactly
like the in-memory LocalExecutor."""
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn as nn

import torchslicer as ts
from torchslicer.offload.executor import choose_optimizer_placement

OPT = {"name": "AdamW", "params": {"lr": 1e-2, "weight_decay": 0.01}}
CRIT = {"name": "CrossEntropyLoss", "params": {}}
CUDA = torch.cuda.is_available()


def _mlp(bn=False, dropout=0.1):
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(16, 32), *( [nn.BatchNorm1d(32)] if bn else [] ), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(32, 32), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 4),
    )


def _data(n=4, bs=8):
    g = torch.Generator().manual_seed(1)
    return [(torch.randn(bs, 16, generator=g), torch.randint(0, 4, (bs,), generator=g)) for _ in range(n)]


def _train(model, executor, gpipe=False, epochs=2, run_config=None):
    s = ts.slice(model, n=3, executor=executor)
    torch.manual_seed(5)
    h = s.train(_data(), OPT, CRIT, epochs=epochs, use_gpipe=gpipe,
                n_micro_batches=2 if gpipe else None, run_config=run_config)
    return [e["loss"] for e in h], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, h


def _same(a, b):
    la, wa = a[:2]
    lb, wb = b[:2]
    assert la == lb
    for k in wa:
        assert torch.equal(wa[k], wb[k]), k


@pytest.mark.parametrize("mode", ["cpu", "disk"])
@pytest.mark.parametrize("gpipe", [False, True])
def test_matches_local_executor(tmp_path, mode, gpipe):
    """Same losses and bit-identical weights as LocalExecutor, dropout included (the
    recompute replays the RNG state). With micro-batches, offload runs slice by slice
    (layer-major) and GPipe micro-batch by micro-batch, so dropout in several slices
    draws its masks in a different order: GPipe is compared without dropout."""
    dropout = 0.0 if gpipe else 0.1
    ref = _train(_mlp(dropout=dropout), ts.LocalExecutor(), gpipe)
    got = _train(_mlp(dropout=dropout), ts.OffloadExecutor(mode=mode, device="cpu",
                                                          directory=str(tmp_path)), gpipe)
    _same(ref, got)


def test_batchnorm_stats_update_once():
    """The recompute must not update BatchNorm running stats a second time."""
    ref = _train(_mlp(bn=True), ts.LocalExecutor())
    got = _train(_mlp(bn=True), ts.OffloadExecutor(device="cpu"))
    _same(ref, got)
    assert got[1]["1.num_batches_tracked"].item() == 8


def test_params_live_in_home_after_training():
    model = _mlp()
    _train(model, ts.OffloadExecutor(device="cpu"))
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_metrics_fields():
    _, _, h = _train(_mlp(), ts.OffloadExecutor(device="cpu"))
    e = h[-1]
    assert e["offload_h2d_bytes"] > 0 and e["offload_d2h_bytes"] > 0
    assert e["offload_optimizer"] == "gpu"          # "the device", which is the CPU here
    assert len(e["compute_s"]) == 3 and all(t > 0 for t in e["compute_s"])
    assert "bytes_fwd" in e                         # SL metrics still there


class _Shared(nn.Module):
    """Uses a buffer object shared with other modules (like HF rotary embeddings)."""
    def __init__(self, buf):
        super().__init__()
        self.lin = nn.Linear(16, 16)
        self.register_buffer("scale", buf)

    def forward(self, x):
        return self.lin(x) * self.scale


def test_tensor_shared_by_slices_stays_resident():
    torch.manual_seed(0)
    buf = torch.ones(16)
    model = nn.Sequential(_Shared(buf), nn.ReLU(), _Shared(buf), nn.ReLU(), _Shared(buf), nn.Linear(16, 4))
    ex = ts.OffloadExecutor(device="cpu")
    s = ts.slice(model, n=3, executor=ex)
    s.executor.setup(s.graph, s.partitions, OPT, CRIT)
    assert any(t is buf for t in ex.pager.resident)
    assert all(e.get() is not buf for pg in ex.pager.pages for e in pg.pb.entries)
    ex.teardown()


def test_tied_parameters_refuse_cpu_optimizer():
    torch.manual_seed(0)
    a, b = nn.Linear(16, 16), nn.Linear(16, 16)
    b.weight = a.weight
    model = nn.Sequential(a, nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), b, nn.Linear(16, 4))
    with pytest.raises(ValueError, match="shared by several slices"):
        _train(model, ts.OffloadExecutor(device="cpu", optimizer="cpu"))


def test_tied_parameters_train_like_local():
    def tied():
        torch.manual_seed(0)
        a, b = nn.Linear(16, 16), nn.Linear(16, 16)
        b.weight = a.weight
        return nn.Sequential(a, nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), b, nn.Linear(16, 4))
    _same(_train(tied(), ts.LocalExecutor()), _train(tied(), ts.OffloadExecutor(device="cpu")))


def test_config_selects_offload_executor():
    cfg = ts.RunConfig()
    cfg.offload.mode = "cpu"
    cfg.offload.device = "cpu"
    cfg.logging.enabled = False
    model = _mlp()
    s = ts.slice(model, n=3)
    torch.manual_seed(5)
    h = s.train(_data(), OPT, CRIT, epochs=1, run_config=cfg)
    assert isinstance(s.executor, ts.OffloadExecutor)
    assert "offload_h2d_bytes" in h[-1]


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("OFFLOAD", "disk")
    monkeypatch.setenv("OFFLOAD_DIR", "/tmp/x")
    monkeypatch.setenv("OFFLOAD_OPTIMIZER", "gpu")
    cfg = ts.RunConfig.load(None)
    assert (cfg.offload.mode, cfg.offload.dir, cfg.offload.optimizer) == ("disk", "/tmp/x", "gpu")


def test_disk_refuses_cpu_optimizer(tmp_path):
    with pytest.raises(ValueError, match="disk"):
        _train(_mlp(), ts.OffloadExecutor(mode="disk", optimizer="cpu", device="cpu", directory=str(tmp_path)))


def test_optimizer_placement_measure():
    choice, info = choose_optimizer_placement(OPT, 10_000, 20e9, 20e9)
    assert choice in ("gpu", "cpu") and info["state_tensors"] == 2


# ── CUDA ───────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("kw", [{"optimizer": "gpu"}, {"optimizer": "cpu"}, {"mode": "disk"},
                                {"mode": "disk", "direct_io": True}])
def test_cuda_matches_local(tmp_path, kw):
    torch.use_deterministic_algorithms(True, warn_only=True)
    def model():
        m = _mlp(dropout=0.1)
        return m
    ref = _train(model().cuda(), ts.LocalExecutor())
    got = _train(model(), ts.OffloadExecutor(device="cuda", directory=str(tmp_path), **kw))
    assert max(abs(a - b) for a, b in zip(ref[0], got[0])) < 1e-5
    for k in ref[1]:
        assert torch.allclose(ref[1][k], got[1][k], atol=1e-5), k


_CAPPED = textwrap.dedent("""
    import json, sys, torch, torch.nn as nn, torchslicer as ts
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    def model():
        torch.manual_seed(0)
        layers = []
        for _ in range(8):
            layers += [nn.Linear(1024, 1024), nn.ReLU()]
        return nn.Sequential(*layers, nn.Linear(1024, 8))
    g = torch.Generator().manual_seed(1)
    data = [(torch.randn(32, 1024, generator=g), torch.randint(0, 8, (32,), generator=g)) for _ in range(2)]
    opt = {"name": "AdamW", "params": {"lr": 1e-3}}
    crit = {"name": "CrossEntropyLoss", "params": {}}
    mode = sys.argv[1]
    if mode == "reference":                       # no cap
        h = ts.slice(model().cuda(), n=9).train(data, opt, crit, epochs=1)
    else:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(100 * 2**20 / total)   # 100 MB
        if mode == "local":
            h = ts.slice(model().cuda(), n=9).train(data, opt, crit, epochs=1)
        else:
            h = ts.slice(model(), n=9, executor=ts.OffloadExecutor(device="cuda")).train(data, opt, crit, epochs=1)
    print(json.dumps({"loss": h[-1]["loss"], "peak": torch.cuda.max_memory_allocated()}))
""")


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_cuda_trains_a_model_that_does_not_fit():
    """~34 MB of weights + 34 MB of grads + 67 MB of AdamW state > the 100 MB cap: the
    in-memory executor runs out of memory, offload trains it to the reference loss."""
    def run(mode):
        p = subprocess.run([sys.executable, "-c", _CAPPED, mode], capture_output=True, text=True, timeout=300)
        return p
    ref, local, off = run("reference"), run("local"), run("offload")
    assert ref.returncode == 0, ref.stderr[-2000:]
    # on a GPU without NVML access, PyTorch's OOM path fails an NVML assert instead
    assert local.returncode != 0 and ("OutOfMemoryError" in local.stderr
                                      or "CUDACachingAllocator" in local.stderr)
    assert off.returncode == 0, off.stderr[-2000:]
    import json
    r, o = json.loads(ref.stdout.splitlines()[-1]), json.loads(off.stdout.splitlines()[-1])
    assert abs(r["loss"] - o["loss"]) < 1e-5
    assert o["peak"] < 100 * 2**20


def test_cpu_tier_checks_host_memory(monkeypatch):
    """A cpu tier that can't fit fails at setup with the numbers, instead of being
    OOM-killed once the optimizer state appears."""
    import torchslicer.offload.executor as ex_mod
    monkeypatch.setattr(ex_mod, "host_memory_available", lambda: 1000)
    with pytest.raises(MemoryError, match="use mode 'disk'"):
        _train(_mlp(), ts.OffloadExecutor(device="cpu"))


# ── planning: memory estimate with activations, "memory" strategy ─────────────

def test_activation_bytes_counts_saved_tensors():
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    g = ts.trace(model)
    x = torch.randn(8, 16)
    acts = ts.activation_bytes(g, x)
    assert acts[0] == 8 * 16 * 4              # Linear saves its input, not its weight
    assert acts[1] == 8 * 32 * 4              # ReLU saves its output
    assert acts[2] == 8 * 32 * 4


def test_activation_bytes_leaves_model_untouched():
    model = _mlp(bn=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.randn(8, 16)
    rng = torch.get_rng_state()
    ts.activation_bytes(ts.trace(model), x)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def _wide():
    torch.manual_seed(0)
    layers = []
    for _ in range(4):
        layers += [nn.Linear(256, 256), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(256, 4))


def test_memory_strategy_fewest_slices_that_fit():
    x = torch.randn(8, 256)
    sp = ts.MemorySplitter(slot_mb=1.2, optimizer=OPT, example_input=x)
    s = ts.slice(_wide(), strategy=sp, n=1)
    mem = sp.last_memory
    per = [sum(mem[i] for i in p.layer_indices) for p in s.partitions]
    assert all(m <= 1.2 * 2**20 for m in per)
    n = len(s.partitions)
    with pytest.raises(ValueError):          # one slice fewer can't fit
        ts.balanced_cuts(sp.node_costs(s.graph), s.graph.valid_cuts(), n - 1,
                         mem=mem, caps=[1.2 * 2**20] * (n - 1))
    assert n == 4                             # each Linear(256, 256): ~1 MB of weights+grads+AdamW


def test_memory_strategy_trains_like_local():
    x = torch.randn(8, 256)
    def run(ex):
        model = _wide()
        s = ts.slice(model, strategy=ts.MemorySplitter(slot_mb=1.2, optimizer=OPT, example_input=x),
                     n=1, executor=ex)
        g = torch.Generator().manual_seed(1)
        data = [(torch.randn(8, 256, generator=g), torch.randint(0, 4, (8,), generator=g)) for _ in range(3)]
        return [e["loss"] for e in s.train(data, OPT, CRIT, epochs=2)]
    assert run(ts.LocalExecutor()) == run(ts.OffloadExecutor(device="cpu"))


def test_memory_strategy_too_small_slot():
    with pytest.raises(ValueError, match="no split fits"):
        ts.slice(_wide(), strategy=ts.MemorySplitter(slot_mb=0.1), n=1)
