"""
Integration test: intra-partition DAG wiring over gRPC.

Runs 2 worker processes on localhost + coordinator in-process.
Uses a model with an explicit top-level skip connection so the
multi-predecessor code path in SplitLayer is exercised on workers.

Model forward graph (7 layers, 2 partitions):

  x --> fc1 --> relu1 --> fc2 --> add --> fc3 --> relu2 --> fc4 --> out
                   |               ^
                   +---------------+   (skip: relu1 output bypasses fc2)

Partition 0 (4 layers): fc1, relu1, fc2, add
  add's local predecessors = [2, 1] (fc2 and relu1) -- intra-partition DAG

Partition 1 (3 layers): fc3, relu2, fc4
  purely sequential
"""

import os
import signal
import subprocess
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torchslicer as ts
from torchslicer.core.model_graph import ModelGraph
from torchslicer.strategies.uniform import UniformSplitter
from torchslicer.executors.distributed import DistributedExecutor
from torchslicer.discovery import CoordinatorDiscovery


N_WORKERS    = 2
COORD_PORT   = 59054
WORKER_PORTS = [59051, 59052]
EPOCHS       = 2
N_SAMPLES    = 64
BATCH_SIZE   = 16
IN_FEATURES  = 16
N_CLASSES    = 4


class SkipModel(nn.Module):
    """Simple MLP with one residual skip connection at the top level."""
    def __init__(self):
        super().__init__()
        self.fc1   = nn.Linear(IN_FEATURES, IN_FEATURES)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Linear(IN_FEATURES, IN_FEATURES)
        # skip: add(fc2_output, relu1_output) — operator.add auto-wrapped as _AddWrapper
        self.fc3   = nn.Linear(IN_FEATURES, IN_FEATURES)
        self.relu2 = nn.ReLU()
        self.fc4   = nn.Linear(IN_FEATURES, N_CLASSES)

    def forward(self, x):
        out  = self.fc1(x)
        out  = self.relu1(out)
        skip = out                  # save for later
        out  = self.fc2(out)
        out  = out + skip           # skip connection — traced as _AddWrapper
        out  = self.fc3(out)
        out  = self.relu2(out)
        return self.fc4(out)


def verify_dag(model):
    """Print partition structure and confirm DAG is non-sequential."""
    graph = ModelGraph.from_module(model)
    print(f"[verify] total layers: {len(graph.nodes)}")
    print(f"[verify] is_dag: {graph.is_dag()}")
    assert graph.is_dag(), "Model graph should be a DAG (has skip connection)"

    splitter = UniformSplitter()
    partitions = splitter.split(graph, N_WORKERS)

    layers = graph.get_layers()
    for p in partitions:
        names = [type(layers[j]).__name__ for j in p.layer_indices]
        print(f"[verify] partition {p.index}: {names}")
        print(f"         predecessors: {p.predecessors}")
        for i, preds in enumerate(p.predecessors):
            if len(preds) > 1:
                print(f"         ** layer {i} ({names[i]}) has {len(preds)} predecessors "
                      f"(DAG path will be exercised on worker)")

    # Confirm no cross-partition multi-input nodes
    splitter.validate(graph, partitions)
    print("[verify] validation passed")
    return True


def start_workers():
    worker_script = os.path.join(
        os.path.dirname(__file__),
        "train/centralized/GRPC/worker/main.py"
    )
    procs = []
    for i, port in enumerate(WORKER_PORTS):
        env = os.environ.copy()
        env["COORDINATOR_ADDRESS"] = f"localhost:{COORD_PORT}"
        env["WORKER_ADDRESS"]      = f"localhost:{port}"
        p = subprocess.Popen(
            [sys.executable, worker_script, str(port)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append(p)
        print(f"[test] worker {i} started (pid={p.pid}, port={port})")
    return procs


def drain_workers(procs):
    for i, p in enumerate(procs):
        out, _ = p.communicate(timeout=10)
        print(f"[worker {i}] exit={p.returncode}")
        for line in out.decode(errors="replace").splitlines():
            print(f"  | {line}")


def main():
    print("=" * 60)
    print("TorchSlicer — intra-partition DAG distributed integration test")
    print("=" * 60)

    model = SkipModel()
    verify_dag(model)

    # Random dataset
    X = torch.randn(N_SAMPLES, IN_FEATURES)
    y = torch.randint(0, N_CLASSES, (N_SAMPLES,))
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)

    procs = start_workers()
    time.sleep(2)  # give workers time to start gRPC server

    try:
        discovery = CoordinatorDiscovery(run_id="dag_test")
        executor  = DistributedExecutor(
            discovery=discovery,
            coordinator_addr=f"localhost:{COORD_PORT}",
        )

        sliced = ts.slice(model, strategy="uniform", n=N_WORKERS, executor=executor)
        sliced.train(
            loader,
            {"name": "SGD", "params": {"lr": 0.01}},
            {"name": "CrossEntropyLoss", "params": {}},
            epochs=EPOCHS,
            verbose=True,
        )

        print("\n[test] PASS — intra-partition DAG executed correctly over gRPC")

    except Exception as e:
        print(f"\n[test] FAIL — {e}")
        import traceback; traceback.print_exc()
        for p in procs:
            p.terminate()
        sys.exit(1)

    finally:
        for p in procs:
            p.terminate()
        drain_workers(procs)


if __name__ == "__main__":
    main()
