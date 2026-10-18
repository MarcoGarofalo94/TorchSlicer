"""Many-client simulation: client-switch cost and N-client rounds, under a memory watchdog.

    .venv/bin/python benchmarks/clients/bench_clients.py switch          # switch cost vs slice size
    .venv/bin/python benchmarks/clients/bench_clients.py round N WIDTH STORE [N WIDTH STORE ...]

A switch = save the active client's state (get_state) + load the next one (set_state).
A round: N parallel clients, one batch of 16 each, SplitFed V2, client slice = 2 Linear
layers of WIDTH (2*WIDTH^2 params), STORE = cpu | gpu. Each round runs in its own process.

Watchdog: the container is shut down if it exceeds its memory limit, so every process
polls the cgroup's memory.current and kills itself above BENCH_MEM_LIMIT_GB (default 11).
"""
import os
import resource
import signal
import statistics as st
import subprocess
import sys
import threading
import time

LIMIT = float(os.environ.get("BENCH_MEM_LIMIT_GB", 11)) * 2**30


def _cgroup_bytes() -> int:
    try:
        return int(open("/sys/fs/cgroup/memory.current").read())
    except OSError:
        return 0


def start_watchdog(limit=LIMIT, period=0.1):
    def watch():
        while True:
            if _cgroup_bytes() > limit:
                print(f"WATCHDOG: container memory {_cgroup_bytes() / 2**30:.1f} GB > "
                      f"{limit / 2**30:.1f} GB, killing this run", flush=True)
                os.kill(os.getpid(), signal.SIGKILL)
            time.sleep(period)
    threading.Thread(target=watch, daemon=True).start()


def switch_table():
    import torch
    import torch.nn as nn
    from torchslicer.core.split_layer import SplitLayer
    from torchslicer.sl.clients import SplitLayerClientState
    torch.set_num_threads(4)
    stores = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print("slices on " + ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"{'params':>12} {'state MB':>9} " + " ".join(f"{'switch ms (' + s + ' store)':>24}" for s in stores))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for width in (32, 256, 1024, 2048, 4096):
        m = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)).to(dev)
        sl = SplitLayer(list(m))
        sl.set_optimizer(torch.optim.SGD(sl.parameters(), lr=0.1, momentum=0.9))
        x = torch.randn(8, width, device=dev)
        out = sl(x)
        sl.backward(prev_g=torch.ones_like(out), out=out)
        sl.optimize()
        row = []
        for store in stores:
            ad = SplitLayerClientState({0: sl}, store=store)
            state = ad.get_state()
            ts_ = []
            for _ in range(7):
                if dev == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                saved = ad.get_state()
                ad.set_state(state)
                if dev == "cuda":
                    torch.cuda.synchronize()
                ts_.append(time.perf_counter() - t)
                del saved
            row.append(st.median(ts_) * 1e3)
        n = sum(p.numel() for p in sl.parameters())
        mb = 2 * n * 4 / 1e6
        print(f"{n:>12,} {mb:>9.1f} " + " ".join(f"{v:>24.2f}" for v in row), flush=True)


def one_round(n: int, width: int, store: str):
    import torch
    import torch.nn as nn
    import torchslicer as ts
    torch.set_num_threads(4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(0)
    data = [[(torch.randn(16, width, generator=g), torch.randint(0, 10, (16,), generator=g))] for _ in range(n)]
    cfg = ts.RunConfig()
    cfg.logging.enabled = False
    cfg.network.device = dev
    cfg.sl.setting, cfg.sl.client_states = "vanilla", store
    cfg.training.optimizer = {"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}
    cfg.training.criterion = {"name": "CrossEntropyLoss", "params": {}}
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                          nn.Linear(width, 10)).to(dev)
    start = _cgroup_bytes()
    h = ts.slice(model, n=2, cuts=["2"]).train(ts.ClientSession(data, mode="parallel"), run_config=cfg,
                                               epochs=2, algorithm=ts.SplitFedAlgorithm())
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    gpu = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0
    params = 2 * (width * width + width)
    d = h[-1]["duration_s"]
    print(f"N={n:5d} client slice {params / 1e6:5.2f}M store={store:4s}  round {d:7.2f}s "
          f"({d / n * 1e3:6.2f} ms/client)  peak RSS {rss:5.2f} GB  peak GPU {gpu:5.2f} GB  "
          f"(container before {start / 2**30:.1f} GB)", flush=True)


if __name__ == "__main__":
    start_watchdog()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib", "torchslicer"))
    cmd = sys.argv[1] if len(sys.argv) > 1 else "switch"
    if cmd == "switch":
        switch_table()
    elif cmd == "_round":
        one_round(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif cmd == "round":
        args = sys.argv[2:]
        for i in range(0, len(args), 3):       # one process per round: no memory carried over
            r = subprocess.run([sys.executable, __file__, "_round", *args[i:i + 3]],
                               capture_output=True, text=True)
            lines = [l for l in r.stdout.splitlines() if l.startswith(("N=", "WATCHDOG"))]
            err = r.stderr.strip().splitlines()[-1:] if r.returncode else []
            print(*(lines or [f"N={args[i]} width={args[i + 1]} {args[i + 2]}: failed rc={r.returncode} {err}"]),
                  sep="\n", flush=True)
