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
    _TensorPeerClient,
    _tensor_addr,
)
from torchslicer.transport.grpc.worker import worker_service_pb2, worker_service_pb2_grpc
from torchslicer.transport.grpc.coordinator import (
    coordinator_service_pb2,
    coordinator_service_pb2_grpc,
)
from torchslicer.core.split_layer import SplitLayer
from torchslicer.monitor import tracer, WorkerProfiler
from torchslicer.monitor.process_logger import get_logger, configure as configure_process_logging
from torchslicer.monitor.run_logger import RunLogger
from torchslicer.monitor.callback import TrainingCallback
from torchslicer.discovery.base import NodeInfo
from torchslicer.config import RunConfig
from torchslicer.retry import RetryPolicy
from torchslicer.executors.startup import init_worker_with_retry

_LOG = get_logger("torchslicer.p2p")


# ── dataset / model (customize per experiment) ─────────────────────────────────

def get_dataset(data_dir: str = '/workspace/data', batch_size: int = None,
                n_train: int = None):
    batch_size = batch_size or int(os.environ.get("BATCH_SIZE", 64))
    n_train    = n_train    or int(os.environ.get("N_TRAIN",    10000))
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
            self._run_forward_stage(batch_id, tensor, self._generation, aux={})
        except Exception as e:
            _LOG.exception("driver forward failed batch_id=%s error=%s", batch_id, e)

    def _send_backward(self, batch_id: int, grad: torch.Tensor, is_last_micro: bool = True,
                       generation: int | None = None):
        """Override: driver has no prev_worker — signal batch done directly."""
        if self._prev_stub:
            # Should not happen for driver (worker 0 has no predecessor), but
            # kept for safety if topology changes.
            super()._send_backward(batch_id, grad, is_last_micro, generation)
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


# ── stats helpers ─────────────────────────────────────────────────────────────

def _follower_stats_to_dict(resp, epoch: int) -> dict:
    """Convert WorkerStatsResponse proto → flat dict for worker_epoch.jsonl."""
    def _ps(ps, name: str) -> dict:
        if ps.count == 0:
            return {}
        return {
            f"{name}_avg_ms":   round(ps.avg_ms,   3),
            f"{name}_min_ms":   round(ps.min_ms,   3),
            f"{name}_max_ms":   round(ps.max_ms,   3),
            f"{name}_p95_ms":   round(ps.p95_ms,   3),
            f"{name}_total_ms": round(ps.total_ms, 3),
        }
    entry = {
        "step": epoch, "epoch": epoch,
        "worker": resp.worker_index, "phase": "worker_epoch",
        "n_batches": resp.n_batches,
    }
    for name, proto_field in [
        ("forward",   resp.forward),
        ("backward",  resp.backward),
        ("optimizer", resp.optimizer),
        ("send_fwd",  resp.send_fwd),
        ("send_bwd",  resp.send_bwd),
        ("idle_fwd",  resp.idle_fwd),
        ("idle_bwd",  resp.idle_bwd),
    ]:
        entry.update(_ps(proto_field, name))
    if resp.peak_mem_mb > 0:
        entry["peak_mem_mb"] = round(resp.peak_mem_mb, 2)
    if resp.end_mem_mb > 0:
        entry["end_mem_mb"] = round(resp.end_mem_mb, 2)
    return entry


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

    driver._tensor_transport   = cfg.transport.tensor
    driver._tensor_port_offset = cfg.transport.tensor_port_offset
    if driver.next_worker:
        driver._next_stub = worker_service_pb2_grpc.WorkerServiceStub(
            _channel(driver.next_worker))
        if cfg.transport.tensor == "tcp":
            driver._next_tensor = _TensorPeerClient(
                _tensor_addr(driver.next_worker, cfg.transport.tensor_port_offset)
            )

    # No coordinator stub needed: _send_backward is overridden to signal directly.

    extra     = {k: v for k, v in opt_cfg.get("params", {}).items() if k != "lr"}
    trainable = [p for p in driver.layer.parameters() if p.requires_grad]
    params    = trainable if trainable else list(driver.layer.parameters())
    opt       = getattr(optim, opt_cfg["name"])(
        params,
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
    _LOG.info(
        "driver slice configured layers=%s next=%s device=%s",
        layer_names,
        driver.next_worker,
        driver.device,
    )


# ── training loop (driver only) ────────────────────────────────────────────────

def run_training(
    driver: P2PDriverServicer,
    coordinator_svc: _P2PCoordinatorServicer,
    last_stub,           # gRPC stub to last worker (for label delivery)
    data_loader,
    cfg: RunConfig,
    follower_stubs: list = None,   # [(peer_addr, stub), ...]
    run_logger: RunLogger = None,
    callbacks: list = None,
    verbose: bool = True,
):
    n_micro        = cfg.pipeline.n_micro if cfg.pipeline.use_gpipe else 1
    n_total        = len(data_loader)
    callbacks      = callbacks or []
    follower_stubs = follower_stubs or []

    for cb in callbacks:
        try:
            cb.on_train_begin(run_id=cfg.run_id, config={})
        except Exception as e:
            _LOG.warning("callback on_train_begin failed: %s", e)

    for epoch in range(cfg.training.epochs):
        total_loss         = 0.0
        n_batches          = 0
        data_load_total_ms = 0.0
        send_total_ms      = 0.0
        wait_total_ms      = 0.0
        epoch_t0           = time.perf_counter()

        for cb in callbacks:
            try:
                cb.on_epoch_begin(epoch)
            except Exception as e:
                _LOG.warning("callback on_epoch_begin failed: %s", e)

        iter_t0 = time.perf_counter()
        for inputs, labels in data_loader:
            data_load_total_ms += (time.perf_counter() - iter_t0) * 1000.0
            batch_id = epoch * n_total + n_batches

            send_t0 = time.perf_counter()
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
            send_total_ms += (time.perf_counter() - send_t0) * 1000.0

            wait_t0 = time.perf_counter()
            loss = coordinator_svc.wait_batch()
            wait_total_ms += (time.perf_counter() - wait_t0) * 1000.0

            total_loss += loss
            n_batches  += 1

            if run_logger:
                run_logger.log(
                    step=batch_id, epoch=epoch, batch=n_batches,
                    loss=round(loss, 6), phase="batch",
                )

            if verbose:
                _LOG.info("epoch progress epoch=%s batch=%s/%s loss=%.4f", epoch, n_batches, n_total, loss)

            iter_t0 = time.perf_counter()

        avg      = total_loss / n_batches if n_batches else 0.0
        duration = time.perf_counter() - epoch_t0
        _LOG.info("epoch complete epoch=%s avg_loss=%.4f duration_s=%.1f", epoch, avg, duration)

        epoch_metrics = {
            "step":       epoch,
            "epoch":      epoch,
            "loss":       round(avg, 6),
            "duration_s": round(duration, 3),
            "phase":      "epoch",
        }
        for cb in callbacks:
            try:
                result = cb.on_epoch_end(epoch, epoch_metrics)
                if isinstance(result, dict):
                    epoch_metrics = result
            except Exception as e:
                _LOG.warning("callback on_epoch_end failed: %s", e)

        if run_logger:
            run_logger.log(**epoch_metrics)

            if cfg.profile.verbosity >= 1:
                run_logger.log(
                    step=epoch, epoch=epoch,
                    data_load_total_ms=round(data_load_total_ms, 3),
                    send_total_ms=round(send_total_ms, 3),
                    wait_total_ms=round(wait_total_ms, 3),
                    n_batches=n_batches,
                    phase="coordinator_epoch",
                )

                # Driver's own stats (no RPC — direct profiler access)
                driver_summary = driver._profiler.epoch_summary(epoch)
                driver._profiler.reset_epoch()
                driver_entry = {
                    "step": epoch, "epoch": epoch,
                    "worker": 0, "phase": "worker_epoch",
                    "n_batches": driver_summary.get("n_batches", 0),
                }
                for phase_name in ("forward", "backward", "optimizer",
                                   "send_fwd", "send_bwd", "idle_fwd", "idle_bwd"):
                    for stat in ("avg_ms", "min_ms", "max_ms", "p95_ms", "total_ms"):
                        k = f"{phase_name}_{stat}"
                        if k in driver_summary:
                            driver_entry[k] = driver_summary[k]
                if driver_summary.get("forward_peak_mem_mb"):
                    driver_entry["peak_mem_mb"] = driver_summary["forward_peak_mem_mb"]
                run_logger.log(**driver_entry)

                # Followers' stats via get_stats RPC
                for fi, (peer_addr, stub) in enumerate(follower_stubs):
                    try:
                        resp = stub.get_stats(worker_service_pb2.GetStatsRequest(
                            run_id    = cfg.run_id,
                            epoch     = epoch,
                            verbosity = cfg.profile.verbosity,
                        ))
                        run_logger.log(**_follower_stats_to_dict(resp, epoch))
                        if cfg.profile.verbosity >= 3:
                            for b in resp.batches:
                                run_logger.log(
                                    step=epoch, epoch=epoch,
                                    worker=resp.worker_index, batch_id=b.batch_id,
                                    forward_ms=b.forward_ms, backward_ms=b.backward_ms,
                                    optimizer_ms=b.optimizer_ms, send_fwd_ms=b.send_fwd_ms,
                                    send_bwd_ms=b.send_bwd_ms, idle_fwd_ms=b.idle_fwd_ms,
                                    idle_bwd_ms=b.idle_bwd_ms, peak_mem_mb=b.peak_mem_mb,
                                    phase="worker_batch",
                                )
                    except Exception as e:
                        _LOG.warning("get_stats failed follower=%s address=%s error=%s", fi + 1, peer_addr, e)

    log_history = run_logger.log_history if run_logger else []
    for cb in callbacks:
        try:
            cb.on_train_end(log_history)
        except Exception as e:
            _LOG.warning("callback on_train_end failed: %s", e)


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
    configure_process_logging(cfg.logging.level)
    peers = (
        [p.strip() for p in peers_env.split(",") if p.strip()]
        if peers_env else cfg.discovery.peers
    )

    # ── Follower (workers 1..N-1) ───────────────────────────────────────────────
    if not is_driver:
        # No coordinator registration — driver sends init() directly.
        ts.run_worker(port=int(port), coordinator_addr=None, worker_address=node_address)
        return

    # ── Driver (worker 0) ───────────────────────────────────────────────────────
    if not peers:
        _LOG.error("WORKER_PEERS env var or discovery.peers in config must be set")
        sys.exit(1)

    n = len(peers)

    if n < 2:
        _LOG.error("P2P topology requires at least 2 workers")
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
    _LOG.info("p2p driver started address=%s n_workers=%s", node_address, n)

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
    _LOG.info("distributing slices follower_count=%s", len(follower_stubs))
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
        res = init_worker_with_retry(
            stub,
            slice_cfg,
            address=peer_addr,
            policy=RetryPolicy(
                max_attempts=cfg.startup.worker_init_max_attempts,
                delay_s=cfg.startup.worker_init_delay_s,
                rpc_timeout_s=cfg.startup.worker_init_rpc_timeout_s,
            ),
            logger=_LOG,
            phase="p2p-init",
        )
        _LOG.info("follower init complete address=%s message=%s", peer_addr, res.message)

    # Configure driver's own slice (partition 0)
    _configure_driver_slice(driver_svc, partitions[0], all_layers,
                            peers, opt_cfg, run_id, cfg)

    # Start TCP tensor server on driver if transport=tcp (followers connect back for backward)
    driver_svc.start_tensor_server(int(port) + cfg.transport.tensor_port_offset)

    # Stubs used during training
    last_stub = follower_stubs[-1][1]   # last worker receives labels
    driver_svc.set_last_stub(last_stub)

    # Initialise RunLogger
    run_logger = None
    if cfg.logging.enabled:
        run_dir    = os.path.join(cfg.logging.dir, run_id)
        run_logger = RunLogger(run_id=run_id, run_dir=run_dir)
        layer_names = [type(l).__name__ for l in all_layers]
        run_logger.record_executor("p2p")
        run_logger.record_config(cfg)
        run_logger.record_model(build_model().__class__.__name__, layer_names)
        run_logger.record_split(partitions, layer_names, "uniform")
        nodes = [
            NodeInfo(node_id=f"worker{i}", address=peers[i],
                     device="unknown", memory_mb=0)
            for i in range(n)
        ]
        run_logger.record_workers(nodes)

    # Run training
    data_loader = get_dataset()
    _LOG.info(
        "training start run_id=%s epochs=%s gpipe=%s n_micro=%s",
        run_id,
        cfg.training.epochs,
        cfg.pipeline.use_gpipe,
        cfg.pipeline.n_micro,
    )
    run_training(
        driver_svc, coordinator_svc, last_stub, data_loader, cfg,
        follower_stubs = follower_stubs,
        run_logger     = run_logger,
        callbacks      = [],
        verbose        = True,
    )

    # Graceful shutdown of all followers
    run_dir = os.path.join(cfg.logging.dir, run_id) if cfg.logging.enabled else ""
    for fi, (peer_addr, stub) in enumerate(follower_stubs):
        try:
            stub.shutdown(worker_service_pb2.ShutdownRequest(
                save_checkpoint = cfg.checkpoint.enabled,
                checkpoint_dir  = run_dir,
                run_id          = run_id,
                epoch           = cfg.training.epochs - 1,
                worker_index    = fi + 1,
            ))
            _LOG.info("follower shutdown sent address=%s", peer_addr)
        except Exception as e:
            _LOG.warning("follower shutdown failed address=%s error=%s", peer_addr, e)

    driver_svc.stop_tensor_server()
    server.stop(grace=2)

    if cfg.checkpoint.enabled and run_logger:
        for i in range(n):
            run_logger.record_artifact(
                "checkpoint", f"worker_{i}_epoch_{cfg.training.epochs - 1}.pt")

    if run_logger:
        run_logger.flush()

    _LOG.info("p2p driver complete run_id=%s", run_id)


if __name__ == '__main__':
    serve()
