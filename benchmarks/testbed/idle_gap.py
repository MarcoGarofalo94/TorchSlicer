"""GPU step time vs idle time between steps: does a device slow down when work is bursty?

    python benchmarks/testbed/idle_gap.py [--slice server|client] [GAP_MS ...]   (default 0 10 30 60 100)

Runs one slice of examples/data's SmallCNN cut at pool1, batch 32: the server (conv2 .. fc2 on
the 32x16x16 activation, with the loss) or the client (conv1, relu, pool1 on a 3x32x32 input,
backward from a fixed gradient at the cut): forward, backward, SGD step, synchronized, with
GAP_MS of sleep between steps. On an
always-busy GPU the step time doesn't depend on the gap; a GPU that clocks down or
power-gates when idle pays a wake-up per step. Prints one JSON line per gap.
"""
import json
import statistics
import sys
import time

import torch
import torch.nn as nn

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = sys.argv[1:]
which = "server"
if "--slice" in args:
    i = args.index("--slice")
    which = args[i + 1]
    args = args[:i] + args[i + 2:]
torch.manual_seed(0)
if which == "server":
    part = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                         nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 10)).to(dev)
    x = torch.randn(32, 32, 16, 16, device=dev)
    y = torch.randint(0, 10, (32,), device=dev)
    loss = lambda: nn.functional.cross_entropy(part(x), y).backward()
else:
    part = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)).to(dev)
    x = torch.randn(32, 3, 32, 32, device=dev)
    g = torch.randn(32, 32, 16, 16, device=dev)
    loss = lambda: part(x).backward(g)
opt = torch.optim.SGD(part.parameters(), lr=0.05, momentum=0.9)


def step():
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    opt.zero_grad()
    loss()
    opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t


for _ in range(20):
    step()
for gap in [float(a) for a in args] or [0, 10, 30, 60, 100]:
    times = []
    for _ in range(60):
        time.sleep(gap / 1e3)
        times.append(step())
    print(json.dumps({"slice": which, "device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
                      "gap_ms": gap, "step_ms_median": round(statistics.median(times) * 1e3, 2),
                      "step_ms_p90": round(sorted(times)[int(0.9 * len(times))] * 1e3, 2)}), flush=True)
