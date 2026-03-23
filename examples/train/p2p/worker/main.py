"""
P2P worker entry point — all workers run this script.

Role is determined by the IS_DRIVER environment variable:

  IS_DRIVER=true   → worker 0 (driver): owns the DataLoader, builds the model,
                     slices and distributes partitions, drives the training loop.
                     Also embeds a lightweight coordinator service so followers
                     can report metrics and signal batch completion.

  IS_DRIVER=false  → follower (workers 1..N-1): start a gRPC server, wait for
                     init() from the driver, then serve forward/backward RPCs.
                     Identical to centralized workers, except no registration step.

Key P2P properties:
  - No coordinator process: driver (worker 0) plays both worker-0 and coordinator roles.
  - Labels travel directly from driver to last worker — intermediate workers never
    see labels, preserving the split-learning privacy guarantee.
  - Memory-efficient: each worker holds only its own partition slice.
  - GPipe micro-batching supported: driver fans out M micro-batches then waits.

Environment variables:
  IS_DRIVER          true/false (default false)
  WORKER_INDEX       0-based index (default 0 for driver, set per container)
  WORKER_PEERS       comma-separated "host:port" list in slice-assignment order
                     (overrides discovery.peers from YAML)
  WORKER_ADDRESS     address advertised to peers (default hostname:port)
  EXPERIMENT_CONFIG  path to YAML experiment config
"""

import io
import os
import sys
import socket
import threading
import time
from concurrent import futures

import grpc
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset

import torchslicer as ts
from torchslicer.executors.worker import (
    WorkerServicer,
    serialize_tensor,
    get_available_memory_mb,
    _channel,
    _GRPC_OPTS,
)
from torchslicer.transport.grpc.worker import worker_service_pb2, worker_service_pb2_grpc
from torchslicer.transport.grpc.coordinator import (
    coordinator_service_pb2,
    coordinator_service_pb2_grpc,
)
from torchslicer.core.split_layer import SplitLayer
from torchslicer.monitor import tracer, WorkerProfiler
from torchslicer.config import RunConfig


# ── dataset / model (customize per experiment) ─────────────────────────────────

def get_dataset(data_dir: str = '/workspace/data', batch_size: int = 64,
                n_train: int = 10000):
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)
    indices = torch.randperm(len(ds))[:n_train].tolist()
    return DataLoader(Subset(ds, indices), batch_size=batch_size,
                      shuffle=True, num_workers=0)


def build_model():
    model         = torchvision.models.resnet18()
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(512, 10)
    return model


# ── embedded coordinator for the driver ────────────────────────────────────────

class _P2PCoordinatorServicer(
    coordinator_service_pb2_grpc.CoordinatorServiceServicer
):
    """Minimal coordinator embedded in the driver.

    The last worker sends report_metrics here instead of a separate coordinator
    process. batch_done is called directly (not via RPC) by the driver's own
    _send_backward override to avoid a loopback gRPC call.
    """

    def __init__(self):
        self._batch_done   = threading.Event()
        self._batch_losses: list[float] = []
        self._lock = threading.Lock()

    def batch_done(self, request, context):
        self._batch_done.set()
        return coordinator_service_pb2.Empty()

    def report_metrics(self, request, context):
        with self._lock:
            self._batch_losses.append(request.loss)
        return coordinator_service_pb2.Empty()

    def register(self, request, context):
        return coordinator_service_pb2.RegisterResponse(
            ok=False, run_id="", worker_index=0,
            message="P2P topology does not use coordinator registration",
        )

    def signal_batch_done(self):
        """Called directly by the driver (no RPC overhead)."""
        self._batch_done.set()

    def wait_batch(self) -> float:
        """Block until the full forward→backward chain completes; return avg loss."""
        self._batch_done.wait()
        self._batch_done.clear()
        with self._lock:
            loss = (sum(self._batch_losses) / len(self._batch_losses)
                    if self._batch_losses else 0.0)
            self._batch_losses.clear()
        return loss


# ── driver servicer ────────────────────────────────────────────────────────────

class P2PDriverServicer(WorkerServicer):
    """WorkerServicer for the driver node (worker 0).

    Extends base WorkerServicer with:
      - _coordinator_svc: reference to the embedded coordinator (no loopback RPC)
      - _last_stub: gRPC stub to the last worker for sending labels directly
      - run_own_forward(): driver's own partition forward (called from training loop)
      - _send_backward() override: signals batch completion via coordinator directly
    """

    def __init__(self, coordinator_svc: _P2PCoordinatorServicer):
        super().__init__()
        self._coordinator_svc = coordinator_svc
        self._last_stub = None

    def set_last_stub(self, stub):
        self._last_stub = stub

    def run_own_forward(self, batch_id: int, inputs: torch.Tensor):
        """Run driver's own partition forward and send activation to next worker."""
        try:
            self._profiler.begin_batch(batch_id)
            self._profiler.mark_idle_end("fwd")

            tensor = inputs.to(self.device)
            with self._profiler.phase("forward"):
                out   = self.layer(tensor)
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
            print(f"[p2p-driver forward] ERROR batch_id={batch_id}: {e}")
            import traceback; traceback.print_exc()

    def _send_backward(self, batch_id: int, grad: torch.Tensor, is_last_micro: bool = True):
        """Override: driver has no prev_worker — signal batch done directly."""
        if self._prev_stub:
            # Should not happen for driver (worker 0 has no predecessor), but
            # kept for safety if topology changes.
            super()._send_backward(batch_id, grad, is_last_micro)
        else:
            # Driver is the first worker — no upstream gradient to send.
            if is_last_micro:
                self._coordinator_svc.signal_batch_done()
            self._profiler.mark_idle_start("fwd")
            self._profiler.end_batch()


# ── SliceConfig helpers ────────────────────────────────────────────────────────

def _build_layer_configs(layers: list) -> list:
    configs = []
    for layer in layers:
        buf = io.BytesIO()
        torch.save(layer, buf)
        configs.append(worker_service_pb2.LayerConfig(
            layer_type=layer.__class__.__name__,
            serialized=buf.getvalue(),
        ))
    return configs


def _build_optimizer_config(cfg: dict):
    extra = {k: v for k, v in cfg.get("params", {}).items() if k != "lr"}
    buf   = io.BytesIO()
    torch.save(extra, buf)
    return worker_service_pb2.OptimizerConfig(
        name        = cfg["name"],
        lr          = float(cfg["params"].get("lr", 0.001)),
        extra_params= buf.getvalue(),
    )


def _build_criterion_config(cfg: dict):
    buf = io.BytesIO()
    torch.save(cfg.get("params", {}), buf)
    return worker_service_pb2.CriterionConfig(
        name        = cfg["name"],
        extra_params= buf.getvalue(),
    )


# ── driver: configure own slice ────────────────────────────────────────────────

def _configure_driver_slice(
    driver: P2PDriverServicer,
    partition,
    all_layers: list,
    peers: list,
    opt_cfg: dict,
    run_id: str,
    cfg: RunConfig,
):
    """Configure driver's own partition directly (no init() RPC self-call)."""
    driver._reset_state()

    own_layers   = [all_layers[j] for j in partition.layer_indices]
    predecessors = (
        [list(p) for p in partition.predecessors]
        if partition.predecessors else None
    )

    driver.layer           = SplitLayer(own_layers, is_last=False, predecessors=predecessors)
    driver.is_last         = False
    driver._run_id         = run_id
    driver._worker_index   = 0
    driver.prev_worker     = None
    driver.next_worker     = peers[1] if len(peers) > 1 else None
    driver._n_micro        = cfg.pipeline.n_micro if cfg.pipeline.use_gpipe else 1

    if driver.next_worker:
        driver._next_stub = worker_service_pb2_grpc.WorkerServiceStub(
            _channel(driver.next_worker))

    # No coordinator stub needed: _send_backward is overridden to signal directly.

    extra = {k: v for k, v in opt_cfg.get("params", {}).items() if k != "lr"}
    opt   = getattr(optim, opt_cfg["name"])(
        driver.layer.parameters(),
        lr=float(opt_cfg.get("params", {}).get("lr", 0.001)),
        **extra,
    )
    driver.layer.set_optimizer(opt)
    driver.layer = driver.layer.to(driver.device)

    driver._profiler = WorkerProfiler(
        verbosity = cfg.profile.verbosity,
        memory    = cfg.profile.memory,
        device    = driver.device,
    )

    layer_names = [type(l).__name__ for l in driver.layer.layers]
    print(f"[p2p-driver] own slice configured: layers={layer_names}  "
          f"next={driver.next_worker}  device={driver.device}")


# ── training loop (driver only) ────────────────────────────────────────────────

def run_training(
    driver: P2PDriverServicer,
    coordinator_svc: _P2PCoordinatorServicer,
    last_stub,           # gRPC stub to last worker (for label delivery)
    data_loader,
    cfg: RunConfig,
    verbose: bool = True,
):
    n_micro  = cfg.pipeline.n_micro if cfg.pipeline.use_gpipe else 1
    n_total  = len(data_loader)

    for epoch in range(cfg.training.epochs):
        total_loss = 0.0
        n_batches  = 0
        epoch_t0   = time.perf_counter()

        for inputs, labels in data_loader:
            batch_id = epoch * n_total + n_batches

            if n_micro > 1:
                micro_inputs = inputs.chunk(n_micro)
                micro_labels = labels.chunk(n_micro)
                for m in range(n_micro):
                    mbid = batch_id * n_micro + m
                    # Send label directly to last worker (privacy: bypasses intermediates)
                    last_stub.forward(worker_service_pb2.ForwardRequest(
                        batch_id=mbid,
                        label=serialize_tensor(micro_labels[m]),
                    ))
                    # Run driver's own forward in serialised compute pool
                    driver._pool.submit(driver.run_own_forward, mbid, micro_inputs[m])
            else:
                # Send label directly to last worker
                last_stub.forward(worker_service_pb2.ForwardRequest(
                    batch_id=batch_id,
                    label=serialize_tensor(labels),
                ))
                driver._pool.submit(driver.run_own_forward, batch_id, inputs)

            loss = coordinator_svc.wait_batch()
            total_loss += loss
            n_batches  += 1

            if verbose:
                print(f"  [epoch {epoch} | batch {n_batches}/{n_total}] loss={loss:.4f}")

        avg      = total_loss / n_batches if n_batches else 0.0
        duration = time.perf_counter() - epoch_t0
        print(f"[epoch {epoch}] avg_loss={avg:.4f}  duration={duration:.1f}s")


# ── helpers ────────────────────────────────────────────────────────────────────

def _init_with_retry(stub, slice_cfg, peer_addr: str, timeout: float = 60.0):
    """Send init() to a follower, retrying until it accepts or timeout elapses."""
    deadline = time.monotonic() + timeout
    attempt  = 0
    while time.monotonic() < deadline:
        try:
            return stub.init(slice_cfg)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                if attempt == 0:
                    print(f"[p2p-driver] waiting for follower {peer_addr} ...")
                time.sleep(1.0)
                attempt += 1
            else:
                print(f"[p2p-driver] init → {peer_addr}: FAILED — {e}")
                return None
    print(f"[p2p-driver] init → {peer_addr}: timeout after {timeout:.0f}s")
    return None


# ── entry point ────────────────────────────────────────────────────────────────

def serve():
    tracer.auto_configure_if_env()

    port         = sys.argv[1] if len(sys.argv) > 1 else "50051"
    is_driver    = os.environ.get("IS_DRIVER", "false").lower() in ("true", "1")
    worker_index = int(os.environ.get("WORKER_INDEX", "0"))
    peers_env    = os.environ.get("WORKER_PEERS", "")
    hostname     = socket.gethostname()
    node_address = os.environ.get("WORKER_ADDRESS", f"{hostname}:{port}")

    cfg   = RunConfig.load(os.environ.get("EXPERIMENT_CONFIG"))
    peers = (
        [p.strip() for p in peers_env.split(",") if p.strip()]
        if peers_env else cfg.discovery.peers
    )

    # ── Follower (workers 1..N-1) ───────────────────────────────────────────────
    if not is_driver:
        servicer = WorkerServicer()
        server   = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                               options=_GRPC_OPTS)
        worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
        server.add_insecure_port(f"[::]:{port}")
        server.start()
        servicer.set_server(server)
        print(f"[p2p-follower] worker_{worker_index} started on {node_address}")
        server.wait_for_termination()
        print(f"[p2p-follower] {hostname} terminated cleanly")
        return

    # ── Driver (worker 0) ───────────────────────────────────────────────────────
    if not peers:
        print("[p2p-driver] ERROR: WORKER_PEERS env var or discovery.peers in config must be set")
        sys.exit(1)

    n = len(peers)

    if n < 2:
        print("[p2p-driver] ERROR: P2P topology requires at least 2 workers")
        sys.exit(1)

    coordinator_svc = _P2PCoordinatorServicer()
    driver_svc      = P2PDriverServicer(coordinator_svc)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=_GRPC_OPTS)
    worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(driver_svc, server)
    coordinator_service_pb2_grpc.add_CoordinatorServiceServicer_to_server(
        coordinator_svc, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    driver_svc.set_server(server)
    print(f"[p2p-driver] started on {node_address}  (n_workers={n})")

    # Build model graph + partitions
    model        = build_model()
    sliced       = ts.slice(model, strategy="uniform", n=n)
    all_layers   = sliced.graph.get_layers()
    partitions   = sliced.partitions
    opt_cfg      = cfg.training.optimizer
    crit_cfg     = cfg.training.criterion
    run_id       = cfg.run_id

    # Connect to follower peers
    follower_stubs: list[tuple[str, object]] = []
    for peer_addr in peers[1:]:
        stub = worker_service_pb2_grpc.WorkerServiceStub(_channel(peer_addr))
        follower_stubs.append((peer_addr, stub))

    # Send SliceConfig to each follower
    print(f"[p2p-driver] distributing slices to {len(follower_stubs)} follower(s) ...")
    for fi, (peer_addr, stub) in enumerate(follower_stubs):
        wi      = fi + 1
        is_last = (wi == n - 1)
        partition_layers = [all_layers[j] for j in partitions[wi].layer_indices]
        pred_proto = [
            worker_service_pb2.PredecessorList(indices=list(p))
            for p in (partitions[wi].predecessors or [[] for _ in partition_layers])
        ]
        slice_cfg = worker_service_pb2.SliceConfig(
            layers            = _build_layer_configs(partition_layers),
            optimizer         = _build_optimizer_config(opt_cfg),
            criterion         = _build_criterion_config(crit_cfg) if is_last else None,
            is_last           = is_last,
            prev_worker       = peers[wi - 1],
            next_worker       = peers[wi + 1] if wi < n - 1 else "",
            coordinator       = node_address,   # driver acts as coordinator
            n_micro           = cfg.pipeline.n_micro if cfg.pipeline.use_gpipe else 1,
            run_id            = run_id,
            checkpoint_path   = "",
            worker_index      = wi,
            profile_verbosity = cfg.profile.verbosity,
            profile_memory    = cfg.profile.memory,
            predecessors      = pred_proto,
        )
        res = _init_with_retry(stub, slice_cfg, peer_addr)
        if res is None:
            sys.exit(1)
        print(f"[p2p-driver] init → {peer_addr}  ok={res.ok}  {res.message}")

    # Configure driver's own slice (partition 0)
    _configure_driver_slice(driver_svc, partitions[0], all_layers,
                            peers, opt_cfg, run_id, cfg)

    # Stubs used during training
    last_stub = follower_stubs[-1][1]   # last worker receives labels
    driver_svc.set_last_stub(last_stub)

    # Run training
    data_loader = get_dataset()
    verbose     = True
    print(f"[p2p-driver] training start  run_id={run_id}  epochs={cfg.training.epochs}  "
          f"gpipe={cfg.pipeline.use_gpipe}  n_micro={cfg.pipeline.n_micro}")
    run_training(driver_svc, coordinator_svc, last_stub, data_loader, cfg, verbose)

    # Graceful shutdown of all followers
    for fi, (peer_addr, stub) in enumerate(follower_stubs):
        try:
            stub.shutdown(worker_service_pb2.ShutdownRequest(
                save_checkpoint = cfg.checkpoint.enabled,
                checkpoint_dir  = (
                    os.path.join(cfg.logging.dir, run_id)
                    if cfg.logging.enabled else ""
                ),
                run_id          = run_id,
                epoch           = cfg.training.epochs - 1,
                worker_index    = fi + 1,
            ))
            print(f"[p2p-driver] shutdown → {peer_addr}")
        except Exception as e:
            print(f"[p2p-driver] shutdown failed → {peer_addr}: {e}")

    server.stop(grace=2)
    print(f"[p2p-driver] done  run_id={run_id}")


if __name__ == '__main__':
    serve()
