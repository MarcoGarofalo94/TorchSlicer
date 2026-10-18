"""Out-of-core training benchmark (M5): one model, one batch, one GPU memory cap, several modes.

usage: bench.py CONFIG [mode ...]        (default: every mode in the config)
       bench.py CONFIG --one MODE        (internal: run one mode in this process)

Each mode runs in its own process, with ``torch.cuda.set_per_process_memory_fraction`` set
to the config's cap, so an OOM in one mode does not affect the others. Modes:

- ``reference``         unsplit PyTorch, no cap (the speed of light, if it fits at all)
- ``reference-cap``     unsplit PyTorch under the cap (shows the model does not fit)
- ``offload-gpu``       TorchSlicer OffloadExecutor, cpu tier, optimizer step on the GPU
- ``offload-cpu``       same, optimizer step on the CPU
- ``offload-disk``      disk tier (memory-mapped files; the page cache holds them when RAM allows)
- ``fsdp2-offload``     FSDP2 fully_shard + CPUOffloadPolicy, world size 1 (baseline)
- ``fsdp2-offload-ckpt`` the same with activation checkpointing per block (as offload recomputes)

``bench_deepspeed.py`` adds DeepSpeed ZeRO-3 / ZeRO-2 CPU offload (own environment).

Prints one JSON line per mode and writes results/<config name>.json.
"""
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


class BVT(nn.Module):
    """[B, T, V] logits → [B, V, T] for ``CrossEntropyLoss`` over tokens."""
    def forward(self, x):
        return x.transpose(1, 2)


class Embed(nn.Module):
    def __init__(self, vocab, d, seq):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, seq, d))

    def forward(self, ids):
        return self.tok(ids) + self.pos[:, :ids.shape[1]]


class Ckpt(nn.Module):
    """Activation checkpointing of one block (the baselines' standard memory saver;
    offload recomputes each slice the same way)."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint
        return checkpoint(self.m, x, use_reentrant=False)


def checkpointed(model: nn.Sequential) -> nn.Sequential:
    return nn.Sequential(*[Ckpt(m) if isinstance(m, nn.TransformerEncoderLayer) else m for m in model])


def build_model(m: dict) -> nn.Sequential:
    """A GPT-shaped stack of standard modules (pre-norm encoder blocks, no causal mask:
    the cost is what matters here, not the language modelling)."""
    blocks = [nn.TransformerEncoderLayer(m["d_model"], m["n_heads"], m["d_ff"], dropout=m.get("dropout", 0.0),
                                         batch_first=True, norm_first=True)
              for _ in range(m["n_layers"])]
    return nn.Sequential(Embed(m["vocab"], m["d_model"], m["seq"]), *blocks,
                         nn.LayerNorm(m["d_model"]), nn.Linear(m["d_model"], m["vocab"]), BVT())


def batches(cfg: dict, device="cpu"):
    g = torch.Generator().manual_seed(cfg["seed"] + 1)
    m = cfg["model"]
    for _ in range(cfg["steps"]):
        x = torch.randint(0, m["vocab"], (cfg["batch"], m["seq"]), generator=g)
        yield x.to(device), torch.roll(x, -1, dims=1).to(device)


def _cap(cfg):
    if torch.cuda.is_available() and cfg.get("gpu_cap_gb"):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, cfg["gpu_cap_gb"] * 2**30 / total))


def _summary(times, cfg, extra):
    t = times[cfg["warmup"]:]
    step = statistics.median(t)
    return {"step_s": round(step, 3), "step_s_all": [round(x, 3) for x in times],
            "samples_per_s": round(cfg["batch"] / step, 2),
            "tokens_per_s": round(cfg["batch"] * cfg["model"]["seq"] / step, 1),
            "peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else 0,
            **extra}


def run_reference(cfg, cap):
    if cap:
        _cap(cfg)
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg["model"]).cuda()
    opt = getattr(torch.optim, cfg["optimizer"]["name"])(model.parameters(), **cfg["optimizer"]["params"])
    crit = nn.CrossEntropyLoss()
    times, losses = [], []
    for x, y in batches(cfg, "cuda"):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        times.append(time.perf_counter() - t0)
    return _summary(times, cfg, {"losses": losses})


def run_offload(cfg, kind):
    import torchslicer as ts
    _cap(cfg)
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg["model"])
    o = cfg["offload"]
    kw = {"mode": "disk" if kind == "disk" else "cpu", "optimizer": "gpu" if kind == "disk" else kind,
          "device": "cuda", "slots": o.get("slots", 2)}
    if kind == "disk":
        kw["directory"] = tempfile.mkdtemp(prefix="ts-offload-", dir=o.get("dir") or None)
    n = o["n_slices"]
    s = ts.slice(model, strategy="params", n=n, executor=ts.OffloadExecutor(**kw))
    times, losses, entries = [], [], []
    # one epoch per step, so every step gets its own history entry (bytes, bound, peak)
    data = list(batches(cfg))
    s.executor.setup(s.graph, s.partitions, cfg["optimizer"], {"name": "CrossEntropyLoss", "params": {}},
                     n_micro_batches=o.get("n_micro", 1), model_name="bench", strategy_name="params")
    for step, (x, y) in enumerate(data):
        t0 = time.perf_counter()
        h = s.executor.train_epoch([(x, y)], epoch=step)
        times.append(time.perf_counter() - t0)
        losses.append(h["loss"])
        entries.append(h)
    info = s.executor.placement_info
    s.executor.teardown()
    if kind == "disk":
        import shutil
        shutil.rmtree(kw["directory"], ignore_errors=True)
    timed = entries[cfg["warmup"]:]
    keys = ["offload_h2d_bytes", "offload_d2h_bytes", "offload_transfer_bound_s", "offload_bound_fraction",
            "offload_home_mb", "offload_slots_mb", "offload_optimizer", "offload_h2d_gbps", "offload_d2h_gbps"]
    extra = {k: timed[-1].get(k) for k in keys}
    extra["compute_s"] = round(statistics.median(sum(e.get("compute_s", [0])) for e in timed), 3)
    extra["bound_fraction_median"] = round(statistics.median(e["offload_bound_fraction"] for e in timed), 3)
    extra["placement_info"] = info
    extra["losses"] = losses
    extra["n_slices"] = n
    return _summary(times, cfg, extra)


def run_fsdp2(cfg, ckpt=False):
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    _cap(cfg)
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg["model"])
    if ckpt:
        model = checkpointed(model)
    policy = CPUOffloadPolicy(pin_memory=True)
    for m in model:
        if any(True for _ in m.parameters()):
            fully_shard(m, offload_policy=policy)
    fully_shard(model, offload_policy=policy)
    opt = getattr(torch.optim, cfg["optimizer"]["name"])(model.parameters(), **cfg["optimizer"]["params"])
    crit = nn.CrossEntropyLoss()
    times, losses = [], []
    for x, y in batches(cfg, "cuda"):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    dist.destroy_process_group()
    return _summary(times, cfg, {"losses": losses})


MODES = {
    "reference": lambda c: run_reference(c, cap=False),
    "reference-cap": lambda c: run_reference(c, cap=True),
    "offload-gpu": lambda c: run_offload(c, "gpu"),
    "offload-cpu": lambda c: run_offload(c, "cpu"),
    "offload-disk": lambda c: run_offload(c, "disk"),
    "fsdp2-offload": run_fsdp2,
    "fsdp2-offload-ckpt": lambda c: run_fsdp2(c, ckpt=True),
}


def one(cfg, mode):
    torch.backends.cuda.matmul.allow_tf32 = cfg.get("tf32", True)
    torch.backends.cudnn.allow_tf32 = cfg.get("tf32", True)
    try:
        out = {"mode": mode, "ok": True, **MODES[mode](cfg)}
    except (torch.OutOfMemoryError, RuntimeError) as e:
        # on a GPU without NVML access, PyTorch's OOM path fails an NVML assert instead
        if not isinstance(e, torch.OutOfMemoryError) and "CUDACachingAllocator" not in str(e):
            raise
        out = {"mode": mode, "ok": False, "error": "OOM", "detail": str(e).splitlines()[0][:200]}
    print(json.dumps(out), flush=True)


def main():
    path = sys.argv[1]
    cfg = yaml.safe_load(open(path))
    if len(sys.argv) > 3 and sys.argv[2] == "--one":
        return one(cfg, sys.argv[3])
    modes = sys.argv[2:] or cfg["modes"]
    n_params = sum(p.numel() for p in build_model({**cfg["model"]}).parameters()) if cfg.get("count_params", True) else None
    results = {"config": cfg, "n_params": n_params, "modes": {}}
    print(f"model: {n_params / 1e6:.0f}M params", flush=True)
    env = {**os.environ, "PYTHONPATH": f"{REPO}:{os.environ.get('PYTHONPATH', '')}"}
    for mode in modes:
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, __file__, path, "--one", mode], env=env,
                               capture_output=True, text=True, timeout=cfg.get("timeout_s", 1200))
            line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith("{")), None)
            res = (json.loads(line) if line else
                   {"mode": mode, "ok": False, "returncode": p.returncode, "error": p.stderr[-2000:]})
        except subprocess.TimeoutExpired:   # e.g. thrashing at the host memory limit
            res = {"mode": mode, "ok": False, "error": f"timeout after {cfg.get('timeout_s', 1200)} s"}
        res["wall_s"] = round(time.time() - t0, 1)
        results["modes"][mode] = res
        print(json.dumps(res), flush=True)
    out = HERE / "results" / (Path(path).stem + ".json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
