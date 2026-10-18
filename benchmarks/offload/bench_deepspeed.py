"""DeepSpeed baseline for bench.py: ZeRO-3 with parameters and optimizer state offloaded to
CPU (ZeRO-Infinity's CPU tier), same model, batch, steps and GPU cap as bench.py.

usage: <python with deepspeed> bench_deepspeed.py CONFIG [zero3-offload|zero2-offload][-ckpt]

Runs in its own environment (DeepSpeed is not a TorchSlicer dependency). Without a CUDA
toolkit, DeepSpeed's compiled CPU-Adam can't be built, so the optimizer is PyTorch's
(``zero_force_ds_cpu_optimizer: false``); the step still runs on the CPU. fp32 throughout,
as in bench.py. Prints one JSON line and merges it into results/<config name>.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402  (model, data, cap, summary)


def main():
    path, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "zero3-offload")
    cfg = yaml.safe_load(open(path))
    for k, v in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1",
                 "MASTER_PORT": str(29600 + os.getpid() % 1000)}.items():
        os.environ.setdefault(k, v)
    import deepspeed
    torch.backends.cuda.matmul.allow_tf32 = cfg.get("tf32", True)
    torch.backends.cudnn.allow_tf32 = cfg.get("tf32", True)
    torch.cuda.set_device(0)
    bench._cap(cfg)
    torch.manual_seed(cfg["seed"])
    model = bench.build_model(cfg["model"])
    if mode.endswith("-ckpt"):
        model = bench.checkpointed(model)
    o = cfg["optimizer"]
    opt = getattr(torch.optim, o["name"])(model.parameters(), **o["params"])
    zero = {"stage": 3, "offload_param": {"device": "cpu", "pin_memory": True},
            "offload_optimizer": {"device": "cpu", "pin_memory": True}}
    if mode.startswith("zero2-offload"):
        zero = {"stage": 2, "offload_optimizer": {"device": "cpu", "pin_memory": True}}
    ds_cfg = {"train_micro_batch_size_per_gpu": cfg["batch"], "gradient_accumulation_steps": 1,
              "zero_optimization": zero, "zero_force_ds_cpu_optimizer": False,
              "zero_allow_untested_optimizer": True, "steps_per_print": 10**9}
    out = {"mode": mode, "ok": True}
    try:
        engine, *_ = deepspeed.initialize(model=model, optimizer=opt, config=ds_cfg)
        crit = nn.CrossEntropyLoss()
        times, losses = [], []
        for x, y in bench.batches(cfg, "cuda"):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            loss = crit(engine(x), y)
            engine.backward(loss)
            engine.step()
            losses.append(loss.item())
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        out.update(bench._summary(times, cfg, {"losses": losses, "deepspeed": deepspeed.__version__}))
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if not isinstance(e, torch.OutOfMemoryError) and "CUDACachingAllocator" not in str(e):
            raise
        out = {"mode": mode, "ok": False, "error": "OOM", "detail": str(e).splitlines()[0][:200]}
    print(json.dumps(out), flush=True)
    res = bench.HERE / "results" / (Path(path).stem + ".json")
    if res.exists():
        data = json.loads(res.read_text())
        data["modes"][mode] = out
        res.write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
