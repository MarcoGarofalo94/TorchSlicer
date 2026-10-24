"""Capability-aware planning on real heterogeneous nodes: CIFAR ResNet-18 (3x3 stem, no
max-pool) on synthetic CIFAR-shaped data, split by PLAN_STRATEGY ("uniform" or "flops"),
planned after discovery from the nodes' measured speed (and links, with
PLANNING_MEASURE_LINKS). Prints one RESULT line per epoch and the plan in the log
("plan slice=... nodes=a..b").

    python -m torchslicer launch benchmarks/testbed/plan.py -c benchmarks/testbed/plan_p2p.yaml \\
        --target baremetal --hosts DRIVER,FOLLOWER1,FOLLOWER2 --workdir /path/to/TorchSlicer
"""
import json
import os

import torch
import torch.nn as nn
import torchvision

import torchslicer as ts

BATCH, N_BATCHES, SEED = 32, 20, 0


def build():
    torch.manual_seed(SEED)
    m = torchvision.models.resnet18(num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


def data():
    g = torch.Generator().manual_seed(SEED + 1)
    return [(torch.randn(BATCH, 3, 32, 32, generator=g), torch.randint(0, 10, (BATCH,), generator=g))
            for _ in range(N_BATCHES)]


class _Epochs(ts.TrainingCallback):
    def __init__(self):
        self.metrics = []

    def on_epoch_end(self, epoch, metrics):
        self.metrics.append(dict(metrics))
        return metrics


if __name__ == "__main__":
    strategy = os.environ.get("PLAN_STRATEGY", "flops")
    kwargs = {"strategy": strategy}
    if strategy in ("flops", "profiled"):
        kwargs["example_input"] = torch.randn(BATCH, 3, 32, 32)
    epochs = _Epochs()
    history = ts.run(build, data, callbacks=[epochs], **kwargs)
    if history:
        for m in epochs.metrics:
            print("RESULT " + json.dumps({"strategy": strategy, "epoch": m["epoch"], "loss": m["loss"],
                                          "ms_per_batch": m["duration_s"] * 1e3 / N_BATCHES,
                                          "compute_s": m["compute_s"], "bytes_fwd": m["bytes_fwd"]}),
                  flush=True)
