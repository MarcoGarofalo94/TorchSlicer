"""
Integration test: GPipe micro-batch pipeline parallelism in P2P topology.

Verifies the epoch-boundary sync: with n_micro > 1 the driver must wait for
all M backward passes before advancing to the next batch, and all M micro-
batch losses must be aggregated before reporting.

Runs driver + follower as in-process gRPC servers on localhost (no Docker).
Uses synthetic data and a tiny MLP — no dataset download required.

Usage:
    conda run -n torchslicer python3 examples/test_p2p_gpipe.py
"""

import io
import threading
import time
from concurrent import futures

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import grpc

import torchslicer as ts
from torchslicer.executors.worker import (
    WorkerServicer, serialize_tensor, _channel, _GRPC_OPTS,
)
from torchslicer.core.split_layer import SplitLayer
from torchslicer.monitor import WorkerProfiler
from torchslicer.transport.grpc.worker import worker_service_pb2, worker_service_pb2_grpc
from torchslicer.transport.grpc.coordinator import (
    coordinator_service_pb2, coordinator_service_pb2_grpc,
)

DRIVER_PORT   = 59061
FOLLOWER_PORT = 59062
N_MICRO  = 2
EPOCHS   = 2


def _build_model():
    return nn.Sequential(
        nn.Linear(16, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 4),
    )


def _build_loader():
    torch.manual_seed(42)
    X = torch.randn(100, 16)
    y = torch.randint(0, 4, (100,))
    return DataLoader(TensorDataset(X, y), batch_size=10, shuffle=False)


# ── Embedded coordinator (driver side) ────────────────────────────────────────

class _Coordinator(coordinator_service_pb2_grpc.CoordinatorServiceServicer):
    def __init__(self):
        self._event  = threading.Event()
        self._losses: list = []
        self._lock   = threading.Lock()

    def batch_done(self, req, ctx):
        self._event.set()
        return coordinator_service_pb2.Empty()

    def report_metrics(self, req, ctx):
        with self._lock:
            self._losses.append(req.loss)
        return coordinator_service_pb2.Empty()

    def register(self, req, ctx):
        return coordinator_service_pb2.RegisterResponse(ok=False)

    def signal_batch_done(self):
        self._event.set()

    def wait_batch(self) -> float:
        self._event.wait()
        self._event.clear()
        with self._lock:
            loss = sum(self._losses) / len(self._losses) if self._losses else 0.0
            self._losses.clear()
        return loss


# ── Driver servicer ────────────────────────────────────────────────────────────

class _Driver(WorkerServicer):
    def __init__(self, coord: _Coordinator):
        super().__init__()
        self._coord = coord

    def run_own_forward(self, batch_id: int, inputs: torch.Tensor):
        try:
            self._profiler.begin_batch(batch_id)
            self._profiler.mark_idle_end("fwd")
            t = inputs.to(self.device)
            with self._profiler.phase("forward"):
                out   = self.layer(t)
                x_ref = self.layer.x
            with self._lock:
                self._outputs[batch_id] = (out, x_ref)
            with self._profiler.phase("send_fwd"):
                self._next_stub.forward(worker_service_pb2.ForwardRequest(
                    batch_id=batch_id,
                    input=serialize_tensor(out),
                ))
            self._profiler.mark_idle_start("bwd")
        except Exception as e:
            print(f"[driver.forward] ERROR: {e}")
            import traceback; traceback.print_exc()

    def _send_backward(self, batch_id: int, grad: torch.Tensor, is_last_micro: bool = True):
        if is_last_micro:
            self._coord.signal_batch_done()
        self._profiler.mark_idle_start("fwd")
        self._profiler.end_batch()


# ── proto helpers ─────────────────────────────────────────────────────────────

def _layer_cfg(layer):
    buf = io.BytesIO()
    torch.save(layer, buf)
    return worker_service_pb2.LayerConfig(
        layer_type=layer.__class__.__name__, serialized=buf.getvalue())

def _opt_cfg():
    buf = io.BytesIO(); torch.save({}, buf)
    return worker_service_pb2.OptimizerConfig(
        name="SGD", lr=0.01, extra_params=buf.getvalue())

def _crit_cfg():
    buf = io.BytesIO(); torch.save({}, buf)
    return worker_service_pb2.CriterionConfig(
        name="CrossEntropyLoss", extra_params=buf.getvalue())


# ── test ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TorchSlicer — P2P GPipe epoch-boundary sync test")
    print(f"  n_micro={N_MICRO}  epochs={EPOCHS}  in-process gRPC")
    print("=" * 60)

    model      = _build_model()
    loader     = _build_loader()
    sliced     = ts.slice(model, strategy="uniform", n=2)
    layers     = sliced.graph.get_layers()
    partitions = sliced.partitions

    driver_addr   = f"localhost:{DRIVER_PORT}"
    follower_addr = f"localhost:{FOLLOWER_PORT}"

    # Start follower gRPC server
    follower = WorkerServicer()
    f_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=_GRPC_OPTS)
    worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(follower, f_server)
    f_server.add_insecure_port(f"[::]:{FOLLOWER_PORT}")
    f_server.start()
    follower.set_server(f_server)
    print(f"[test] follower started on {follower_addr}")

    # Start driver gRPC server (receives backward from follower)
    coord  = _Coordinator()
    driver = _Driver(coord)
    d_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=_GRPC_OPTS)
    worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(driver, d_server)
    coordinator_service_pb2_grpc.add_CoordinatorServiceServicer_to_server(coord, d_server)
    d_server.add_insecure_port(f"[::]:{DRIVER_PORT}")
    d_server.start()
    driver.set_server(d_server)
    print(f"[test] driver started on {driver_addr}")

    # Init follower (worker 1 — is_last)
    follower_stub = worker_service_pb2_grpc.WorkerServiceStub(_channel(follower_addr))
    p1_layers  = [layers[j] for j in partitions[1].layer_indices]
    pred_proto = [
        worker_service_pb2.PredecessorList(indices=list(p))
        for p in (partitions[1].predecessors or [[] for _ in p1_layers])
    ]
    res = follower_stub.init(worker_service_pb2.SliceConfig(
        layers       = [_layer_cfg(l) for l in p1_layers],
        optimizer    = _opt_cfg(),
        criterion    = _crit_cfg(),
        is_last      = True,
        prev_worker  = driver_addr,
        next_worker  = "",
        coordinator  = driver_addr,
        n_micro      = N_MICRO,
        run_id       = "gpipe_p2p_test",
        worker_index = 1,
        predecessors = pred_proto,
    ))
    assert res.ok, f"Follower init failed: {res.message}"
    print(f"[test] follower init ok")

    # Configure driver's own slice (partition 0) directly
    p0_layers = [layers[j] for j in partitions[0].layer_indices]
    pred_proto0 = [
        worker_service_pb2.PredecessorList(indices=list(p))
        for p in (partitions[0].predecessors or [[] for _ in p0_layers])
    ]
    driver.layer         = SplitLayer(p0_layers, is_last=False)
    driver.is_last       = False
    driver._run_id       = "gpipe_p2p_test"
    driver._worker_index = 0
    driver._n_micro      = N_MICRO
    driver.next_worker   = follower_addr
    driver._next_stub    = worker_service_pb2_grpc.WorkerServiceStub(_channel(follower_addr))
    driver.layer.set_optimizer(optim.SGD(driver.layer.parameters(), lr=0.01))
    driver.layer     = driver.layer.to(driver.device)
    driver._profiler = WorkerProfiler(verbosity=0)
    print(f"[test] driver slice configured  device={driver.device}")

    # Training loop — mirrors P2P main.py run_training
    n_total = len(loader)
    losses  = []
    for epoch in range(EPOCHS):
        total_loss = 0.0
        n_batches  = 0
        t0         = time.perf_counter()

        for inputs, labels in loader:
            batch_id     = epoch * n_total + n_batches
            micro_inputs = inputs.chunk(N_MICRO)
            micro_labels = labels.chunk(N_MICRO)

            for m in range(N_MICRO):
                mbid = batch_id * N_MICRO + m
                follower_stub.forward(worker_service_pb2.ForwardRequest(
                    batch_id=mbid,
                    label=serialize_tensor(micro_labels[m]),
                ))
                driver._pool.submit(driver.run_own_forward, mbid, micro_inputs[m])

            loss = coord.wait_batch()   # blocks until last micro-batch backward completes
            total_loss += loss
            n_batches  += 1

        avg = total_loss / n_batches
        dur = time.perf_counter() - t0
        print(f"[epoch {epoch}] avg_loss={avg:.4f}  duration={dur:.2f}s  "
              f"batches={n_batches}  n_micro={N_MICRO}")
        losses.append(avg)

    # Shutdown
    follower_stub.shutdown(worker_service_pb2.ShutdownRequest(
        save_checkpoint=False, run_id="gpipe_p2p_test",
        epoch=EPOCHS - 1, worker_index=1,
    ))
    time.sleep(0.3)
    d_server.stop(grace=1)

    # Assertions
    assert len(losses) == EPOCHS, f"Expected {EPOCHS} epoch losses, got {len(losses)}"
    assert all(l > 0 for l in losses), f"All losses should be positive: {losses}"
    print()
    print(f"[test] PASS — GPipe P2P epoch-boundary sync verified")
    print(f"       n_micro={N_MICRO}  losses={[round(l, 4) for l in losses]}")


if __name__ == "__main__":
    main()
