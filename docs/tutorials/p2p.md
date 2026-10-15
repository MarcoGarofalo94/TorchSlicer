# P2P training, and how to change how it works

In the P2P topology there is no coordinator process. The **driver** (worker 0) holds the
data and slice 0, plans the split, sends every other slice to a **follower**, and drives
training. Activations and gradients go from peer to peer. Raw inputs never leave the
driver, and labels go only to the last worker. With `sl.setting: u_shaped` the driver hosts
the last slice too, so the labels never leave it either (`docs/tutorials/sl_settings.md`).

The driver is `ts.P2PExecutor`, a `DistributedExecutor` whose worker 0 runs in-process. The
followers are plain TorchSlicer workers. Everything else (worker code, gRPC/TCP transport,
tied weights, heartbeats, recovery, run logs) is the same as in the centralized topology.

## Run it

Every node runs the same script; `IS_DRIVER` picks the role:

```bash
# followers
IS_DRIVER=false PORT=50051 python train.py
# driver (peers in slice order, driver first)
IS_DRIVER=true PORT=50051 WORKER_ADDRESS=w0:50051 \
  WORKER_PEERS=w0:50051,w1:50051,w2:50051 EXPERIMENT_CONFIG=run.yaml python train.py
```

where `train.py` is the model, its data and one call:

```text
import torchslicer as ts
ts.run_p2p(build_model, get_loader, strategy="params")   # any ts.slice option
```

`examples/train/p2p/worker/main.py` (ResNet-18/CIFAR-10) and `examples/train/lm/` (TinyGPT,
TinyGPT + LoRA) are complete scripts, and `make run-p2p` runs one in Docker. In Python, the same
run with in-process followers looks like this (handy for tests and simulations):

```python
import socket

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torchslicer as ts


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
                         nn.Linear(32, 4))


x, y = torch.randn(64, 8), torch.randint(0, 4, (64,))
loader = DataLoader(TensorDataset(x, y), batch_size=16)

cfg = ts.RunConfig()
cfg.training.epochs = 2
cfg.training.optimizer = {"name": "SGD", "params": {"lr": 0.05}}
cfg.logging.enabled = False

followers = [ts.start_worker(host="127.0.0.1", run_config=cfg) for _ in range(2)]
driver = f"127.0.0.1:{free_port()}"
executor = ts.P2PExecutor(peers=[f.address for f in followers], address=driver,
                          bind_addr=driver, run_config=cfg)
history = ts.slice(model(), n=3, executor=executor).train(loader, run_config=cfg)
print([round(h["loss"], 4) for h in history])
```

## The four extension points

A P2P run is four decisions. Each has a default, and you can replace each one:

| Step | Default | Replace it with |
|---|---|---|
| **Discovery**: who the followers are | `StaticDiscovery(peers)` | any `ts.BaseDiscovery`, passed as `P2PExecutor(discovery=...)` |
| **Plan agreement**: who holds which slice | the driver plans and sends the plan with `init`; followers accept it | the strategy given to `ts.slice` (`strategy=`, `cuts=`), or subclass and override `_order_nodes` (slice order) |
| **Scheduling**: how a batch is fed | one batch at a time, or GPipe micro-batches (`pipeline.use_gpipe`) | override `_send_batch(batch_id, inputs, labels)` |
| **Failure handling**: what a dead follower does to the run | heartbeats → `WorkerFailureError`; with `fault_tolerance.enabled`, `recover()` re-plans over the survivors | override `recover(last_good_epoch)` |

The driver's own slice never moves to a follower, because that would send the raw data off
the driver. A failure on the driver therefore ends the run, even with fault tolerance on.

### Worked example: replace discovery

`StaticDiscovery` needs every address up front. Below, followers announce themselves by
writing a file in a shared directory (an NFS mount, a Kubernetes volume, ...), and the
driver waits until enough files appear. The planner only sees what `discover()` returns,
a list of `ts.NodeInfo` in slice order, so this is the whole change:

```python
import json
import pathlib
import tempfile
import time


class FileDiscovery(ts.BaseDiscovery):
    """Rendezvous through a shared directory: one JSON file per follower."""

    def __init__(self, directory):
        self.dir = pathlib.Path(directory)

    def announce(self, node_info, coordinator_addr=None):
        """Called by a follower once its server is up."""
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f".{node_info.node_id}.tmp"
        tmp.write_text(json.dumps({"address": node_info.address}))
        tmp.rename(self.dir / f"{node_info.node_id}.json")   # atomic: no half-written files
        return ts.AnnounceResult(run_id="file", worker_index=-1)

    def discover(self, expected, timeout=60.0):
        """Called by the driver: the first `expected` followers, in file-name order."""
        deadline = time.monotonic() + timeout
        while True:
            files = sorted(self.dir.glob("*.json"))
            if len(files) >= expected:
                return [ts.NodeInfo(node_id=f.stem,
                                    address=json.loads(f.read_text())["address"])
                        for f in files[:expected]]
            if time.monotonic() > deadline:
                raise TimeoutError(f"{len(files)}/{expected} followers in {self.dir}")
            time.sleep(0.1)


rendezvous = tempfile.mkdtemp()
followers = [ts.start_worker(host="127.0.0.1", run_config=cfg) for _ in range(2)]
for i, f in enumerate(followers):                  # each follower, on its own machine
    FileDiscovery(rendezvous).announce(ts.NodeInfo(node_id=f"w{i}", address=f.address))

driver = f"127.0.0.1:{free_port()}"
executor = ts.P2PExecutor(discovery=FileDiscovery(rendezvous), address=driver,
                          bind_addr=driver, run_config=cfg)
history_file = ts.slice(model(), n=3, executor=executor).train(loader, run_config=cfg)
assert [h["loss"] for h in history_file] == [h["loss"] for h in history]
```

The losses are identical to the static run: discovery decides *who* trains, not *what* is
computed. A centralized coordinator can use the same class:
`DistributedExecutor(discovery=FileDiscovery(...), ...)`.

## Planning for the workers you actually have

The split is planned after discovery, once the workers are known, and in the same way for
P2P and centralized runs:

1. Each worker reports its **capabilities**: device, free memory, and compute speed (a
   short matmul benchmark, in GFLOP/s), plus anything declared in its config:
   `network.worker_compute_gflops` (which overrides the benchmark, for example in a
   simulation), `network.worker_max_memory_mb` (a cap for its slice), and
   `network.worker_first_slice` (it must hold slice 0, for example because it owns the data).
2. The executor measures the RTT and bandwidth to each worker, and, with
   `planning.measure_links`, from each worker to the next one.
3. It asks the strategy for the partitions, passing these `nodes`. The cost strategies
   (`params`, `flops`, `profiled`) minimise the slowest device's time, `cost / speed`.
   They keep each slice within its worker's memory cap, and for `flops`/`profiled` with
   measured links they add each cut's activation and gradient transfer time. Speeds that
   differ by less than 25% count as identical, so benchmark noise on identical workers
   doesn't move the cuts.
4. `recover()` plans again for the survivors, with their real capabilities.

Here two simulated followers declare their speed, and the fast last worker gets most of the
parameters:

```python
def node_cfg(gflops):
    c = ts.RunConfig()
    c.training, c.logging = cfg.training, cfg.logging
    c.network.worker_compute_gflops = gflops
    return c


followers = [ts.start_worker(host="127.0.0.1", run_config=node_cfg(g)) for g in (10.0, 80.0)]
driver = f"127.0.0.1:{free_port()}"
executor = ts.P2PExecutor(peers=[f.address for f in followers], address=driver,
                          bind_addr=driver, run_config=node_cfg(10.0))
sliced = ts.slice(model(), n=3, strategy="params", executor=executor)
print("before discovery:", [p.layer_indices for p in sliced.partitions])
sliced.train(loader, run_config=cfg)
print("planned:         ", [p.layer_indices for p in sliced.partitions])
```

A strategy of your own takes part in the same way: give its `split` a `nodes=None` argument
(see `custom_strategy.md`). Strategies without it ignore the workers. Set
`planning.heterogeneous: false` to keep the partitions `ts.slice` computed.
