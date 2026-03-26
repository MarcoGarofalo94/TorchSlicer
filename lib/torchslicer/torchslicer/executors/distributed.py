import io
import json
import os
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field

import torch
from torch import nn

from ..core.split_layer import SplitLayer

from .base import BaseExecutor
from .startup import init_worker_with_retry
from ..monitor import tracer
from ..monitor.process_logger import get_logger, configure as configure_process_logging
from ..monitor.run_logger import RunLogger
from ..monitor.callback import TrainingCallback
from ..discovery.base import BaseDiscovery, NodeInfo
from ..config import RunConfig, CheckpointConfig
from ..retry import RetryPolicy
from ..strategies.base import Partition
from ..strategies.uniform import UniformSplitter

try:
    import grpc
    from torchslicer.transport.grpc.worker import worker_service_pb2, worker_service_pb2_grpc
    from torchslicer.transport.grpc.coordinator import (
        coordinator_service_pb2,
        coordinator_service_pb2_grpc,
    )
    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False

_LOG = get_logger("torchslicer.coordinator")

class WorkerFailureError(RuntimeError):
    """Raised by train_epoch() when a worker's heartbeat times out."""
    def __init__(self, worker_idx: int, address: str, epoch: int, reason: str | None = None):
        self.worker_idx = worker_idx
        self.address    = address
        self.epoch      = epoch
        self.reason     = reason or ""
        msg = f"Worker {worker_idx} ({address}) failed during epoch {epoch}"
        if self.reason:
            msg = f"{msg}: {self.reason}"
        super().__init__(msg)


_MAX_MSG = 256 * 1024 * 1024
_GRPC_OPTS = [
    ('grpc.max_send_message_length',    _MAX_MSG),
    ('grpc.max_receive_message_length', _MAX_MSG),
]

def _channel(addr): return grpc.insecure_channel(addr, options=_GRPC_OPTS)


# ── tensor helpers ─────────────────────────────────────────────────────────────

_TORCH_TO_DTYPE = {}
_DTYPE_TO_TORCH = {}

def _init_dtype_maps():
    if not _GRPC_AVAILABLE:
        return
    mapping = [
        (torch.float32,  worker_service_pb2.FLOAT32),
        (torch.float16,  worker_service_pb2.FLOAT16),
        (torch.bfloat16, worker_service_pb2.BFLOAT16),
        (torch.float64,  worker_service_pb2.FLOAT64),
        (torch.int64,    worker_service_pb2.INT64),
        (torch.int32,    worker_service_pb2.INT32),
    ]
    for torch_t, proto_t in mapping:
        _TORCH_TO_DTYPE[torch_t] = proto_t
        _DTYPE_TO_TORCH[proto_t] = torch_t

_init_dtype_maps()


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    arr = t.detach().cpu().contiguous()
    if arr.dtype == torch.bfloat16:
        return arr.view(torch.uint8).numpy().tobytes()
    return arr.numpy().tobytes()


def _serialize_tensor(t: torch.Tensor):
    return worker_service_pb2.Tensor(
        data=_tensor_to_bytes(t),
        shape=list(t.shape),
        dtype=_TORCH_TO_DTYPE.get(t.dtype, worker_service_pb2.FLOAT32),
    )


# ── config builders ────────────────────────────────────────────────────────────

def _unpack_inputs(inputs):
    """Unpack batch inputs into ``(main_tensor, aux_dict)``.

    See ``local._unpack_inputs`` for full documentation.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs, {}
    if isinstance(inputs, dict):
        main_key = "input_ids" if "input_ids" in inputs else next(iter(inputs))
        return inputs[main_key], {k: v for k, v in inputs.items() if k != main_key}
    if (isinstance(inputs, (list, tuple))
            and len(inputs) == 2
            and isinstance(inputs[1], dict)):
        return inputs[0], inputs[1]
    if isinstance(inputs, (list, tuple)):
        return inputs[0], {}
    return inputs, {}


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
    buf = io.BytesIO()
    torch.save(extra, buf)
    return worker_service_pb2.OptimizerConfig(
        name=cfg["name"],
        lr=float(cfg["params"].get("lr", 0.001)),
        extra_params=buf.getvalue(),
    )


def _build_criterion_config(cfg: dict):
    buf = io.BytesIO()
    torch.save(cfg.get("params", {}), buf)
    return worker_service_pb2.CriterionConfig(
        name=cfg["name"],
        extra_params=buf.getvalue(),
    )


# ── coordinator servicer (embedded gRPC server inside the executor) ────────────

class _CoordinatorServicer(
    coordinator_service_pb2_grpc.CoordinatorServiceServicer
    if _GRPC_AVAILABLE else object
):
    def __init__(self, executor: "DistributedExecutor"):
        self._executor = executor

    def batch_done(self, request, context):
        self._executor._batch_done.set()
        return coordinator_service_pb2.Empty()

    def report_metrics(self, request, context):
        with self._executor._lock:
            self._executor._batch_losses.append(request.loss)
        return coordinator_service_pb2.Empty()

    def register(self, request, context):
        """Delegate to the discovery object's handle_register()."""
        discovery = self._executor._discovery
        if not hasattr(discovery, "handle_register"):
            return coordinator_service_pb2.RegisterResponse(
                ok=False,
                run_id="",
                worker_index=0,
                message="Discovery backend does not support registration",
            )
        ok, worker_index, msg = discovery.handle_register(
            node_id   = request.node.node_id,
            address   = request.node.address,
            device    = request.node.device,
            memory_mb = request.node.memory_mb,
            run_id    = request.run_id,
            tags      = list(request.node.tags),
        )
        return coordinator_service_pb2.RegisterResponse(
            ok=ok,
            run_id=discovery.run_id,
            worker_index=worker_index,
            message=msg,
        )

    def report_worker_error(self, request, context):
        self._executor._on_worker_runtime_error(request)
        return coordinator_service_pb2.Empty()


# ── worker proxy ───────────────────────────────────────────────────────────────

@dataclass
class _WorkerProxy:
    name:         str
    address:      str
    worker_index: int
    _stub: object = field(default=None, repr=False)

    def connect(self):
        self._stub = worker_service_pb2_grpc.WorkerServiceStub(_channel(self.address))

    def stub(self):
        return self._stub


# ── DistributedExecutor ────────────────────────────────────────────────────────

class DistributedExecutor(BaseExecutor):
    """
    Centralized topology executor. This process acts as the coordinator:
      - Starts an embedded gRPC server (including the Register endpoint).
      - Waits for workers to self-register via discovery.discover().
      - Distributes model slices to workers via SliceConfig at setup() time.
      - Drives the training loop batch-by-batch synchronously.
      - On teardown(), signals all workers to shut down cleanly (with optional checkpoint).

    Usage with CoordinatorDiscovery (workers self-register)::

        from torchslicer.discovery import CoordinatorDiscovery
        from torchslicer.config import RunConfig

        cfg = RunConfig.load("experiments/run.yaml")
        discovery = CoordinatorDiscovery(run_id=cfg.run_id)
        executor = DistributedExecutor(
            discovery=discovery,
            coordinator_addr="coordinator:50054",
            run_config=cfg,
        )

    Usage with StaticDiscovery (P2P / fixed addresses, no registration)::

        from torchslicer.discovery import StaticDiscovery

        discovery = StaticDiscovery(peers=["worker1:50051", "worker2:50051"])
        executor = DistributedExecutor(
            discovery=discovery,
            coordinator_addr="coordinator:50054",
        )
    """

    def __init__(
        self,
        discovery: BaseDiscovery,
        coordinator_addr: str,
        coordinator_bind_addr: str | None = None,
        run_config: RunConfig = None,
    ):
        if not _GRPC_AVAILABLE:
            raise ImportError(
                "grpcio is required for DistributedExecutor. "
                "Install it with: pip install grpcio"
            )
        self._discovery           = discovery
        self.coordinator_addr     = coordinator_addr
        self.coordinator_bind_addr = coordinator_bind_addr
        self._run_config          = run_config or RunConfig()

        self._proxies: list[_WorkerProxy] = []
        self._nodes:   list               = []
        self._grpc_server = None
        self._n_micro  = 1
        self._last_epoch = 0

        # Synchronisation between gRPC callbacks and train_epoch loop
        self._batch_done   = threading.Event()
        self._batch_losses: list[float] = []
        self._lock = threading.Lock()

        # Logging / callback state — populated in setup()
        self._run_logger:  RunLogger                 = None
        self._callbacks:   list[TrainingCallback]    = []
        self._model_name:  str                       = "unknown"
        self._strategy_name: str                     = "unknown"

        # Fault tolerance state
        self._failure_event    = threading.Event()
        self._failed_workers:  set[int] = set()
        self._failure_info:    dict[int, str] = {}   # idx -> address
        self._failure_reason:  dict[int, str] = {}   # idx -> runtime failure detail
        self._heartbeat_threads: list[threading.Thread] = []
        self._heartbeat_stop   = threading.Event()
        self._max_fault_retries: int = 0
        # Layer/partition snapshot for recovery re-slicing
        self._stored_model_graph = None
        self._stored_layers:           list = []
        self._stored_partition_indices: list[list[int]] = []
        self._stored_opt_cfg = None
        self._stored_crit_cfg = None
        self._ft_checkpoint_dir: str = ""

    # ── BaseExecutor interface ─────────────────────────────────────────────────

    def setup(
        self,
        model_graph,
        partitions,
        optimizer_cfg: dict,
        criterion_cfg: dict,
        mixed_precision: bool = False,
        n_micro_batches: int = 1,
        callbacks: list = None,
        model_name: str = "unknown",
        strategy_name: str = "unknown",
        run_config=None,  # ignored — DistributedExecutor uses constructor-injected RunConfig
    ) -> None:
        layers = model_graph.get_layers()
        n = len(partitions)
        self._n_micro = max(1, n_micro_batches)
        self._callbacks = list(callbacks or [])
        self._model_name = model_name
        self._strategy_name = strategy_name
        cfg = self._run_config
        configure_process_logging(cfg.logging.level)

        # Start embedded coordinator gRPC server FIRST so the Register endpoint
        # is available before discover() blocks waiting for registrations.
        self._grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=4), options=_GRPC_OPTS
        )
        coordinator_service_pb2_grpc.add_CoordinatorServiceServicer_to_server(
            _CoordinatorServicer(self), self._grpc_server
        )
        bind_addr = self.coordinator_bind_addr
        if not bind_addr:
            port = self.coordinator_addr.split(":")[-1]
            bind_addr = f"[::]:{port}"
        self._grpc_server.add_insecure_port(bind_addr)
        self._grpc_server.start()
        _LOG.info("gRPC server started bind_addr=%s run_id=%s", bind_addr, self._run_id)

        # Wait for workers to register (CoordinatorDiscovery) or use static list
        tag_filter = cfg.discovery.tag_filter or []
        filter_str = f" with tags {tag_filter}" if tag_filter else ""
        _LOG.info(
            "waiting for workers expected=%s discovery=%s timeout_s=%.1f%s",
            n,
            type(self._discovery).__name__,
            cfg.discovery.timeout,
            filter_str,
        )
        discover_kwargs = {"expected": n, "timeout": cfg.discovery.timeout}
        if tag_filter and hasattr(self._discovery, "discover"):
            discover_kwargs["tag_filter"] = tag_filter
        self._nodes = self._discovery.discover(**discover_kwargs)
        _LOG.info("workers ready count=%s addresses=%s", n, [nd.address for nd in self._nodes])

        # Build proxies from discovered node list (addresses come from workers themselves)
        self._proxies = [
            _WorkerProxy(name=nd.node_id, address=nd.address, worker_index=i)
            for i, nd in enumerate(self._nodes)
        ]
        for proxy in self._proxies:
            proxy.connect()

        # Initialise RunLogger now that we have run_id and worker info
        if cfg.logging.enabled:
            run_dir = os.path.join(cfg.logging.dir, self._run_id)
            self._run_logger = RunLogger(run_id=self._run_id, run_dir=run_dir)
            layer_names = [type(l).__name__ for l in layers]
            self._run_logger.record_executor("distributed")
            self._run_logger.record_config(cfg)
            self._run_logger.record_model(model_name, layer_names)
            self._run_logger.record_split(partitions, layer_names, strategy_name)
            self._run_logger.record_workers(self._nodes)

        # Resolve resume checkpoint paths (empty dict if not resuming)
        checkpoint_paths = self._resolve_checkpoint_paths()

        # Snapshot layers and partitions for potential fault-recovery re-slicing
        self._stored_model_graph       = model_graph
        self._stored_layers            = layers
        self._stored_partition_indices = [list(p.layer_indices) for p in partitions]

        # Send SliceConfig to each worker
        opt_cfg  = _build_optimizer_config(optimizer_cfg)
        crit_cfg = _build_criterion_config(criterion_cfg)
        self._stored_opt_cfg  = opt_cfg
        self._stored_crit_cfg = crit_cfg

        for i, proxy in enumerate(self._proxies):
            is_last = (i == n - 1)
            partition_layers = [layers[j] for j in partitions[i].layer_indices]
            pred_proto = [
                worker_service_pb2.PredecessorList(indices=list(p))
                for p in (partitions[i].predecessors or [[] for _ in partition_layers])
            ]
            slice_cfg = worker_service_pb2.SliceConfig(
                layers             = _build_layer_configs(partition_layers),
                optimizer          = opt_cfg,
                criterion          = crit_cfg if is_last else None,
                is_last            = is_last,
                prev_worker        = self._proxies[i - 1].address if i > 0 else "",
                next_worker        = self._proxies[i + 1].address if i < n - 1 else "",
                coordinator        = self.coordinator_addr,
                n_micro            = self._n_micro,
                run_id             = self._run_id,
                checkpoint_path    = checkpoint_paths.get(i, ""),
                worker_index       = i,
                profile_verbosity  = cfg.profile.verbosity,
                profile_memory     = cfg.profile.memory,
                predecessors       = pred_proto,
            )
            res = self._init_worker(proxy, slice_cfg, phase="init")
            _LOG.info(
                "worker init complete worker=%s address=%s hostname=%s message=%s",
                proxy.name,
                proxy.address,
                res.hostname,
                res.message,
            )

        # Start heartbeat threads if fault tolerance is enabled
        ft = cfg.fault_tolerance
        if ft.enabled:
            self._max_fault_retries = ft.max_retries
            self._start_heartbeats(ft.heartbeat_interval_s, ft.ping_timeout_s)
            _LOG.info(
                "fault tolerance enabled heartbeat_interval_s=%.1f ping_timeout_s=%.1f",
                ft.heartbeat_interval_s,
                ft.ping_timeout_s,
            )

        # Notify callbacks that training is about to begin
        for cb in self._callbacks:
            try:
                cb.on_train_begin(
                    run_id=self._run_id,
                    config=cfg.logging.__dict__ if hasattr(cfg, "logging") else {},
                )
            except Exception as e:
                _LOG.warning("callback on_train_begin failed: %s", e)

    def train_epoch(self, data_loader, epoch: int = 0, verbose: bool = False,
                    total_epochs: int = 0) -> dict:
        self._last_epoch = epoch
        total_loss       = 0.0
        n_batches        = 0
        n_total          = len(data_loader)

        # Coordinator-side timing accumulators
        data_load_total_ms = 0.0
        send_total_ms      = 0.0
        wait_total_ms      = 0.0

        for cb in self._callbacks:
            try:
                cb.on_epoch_begin(epoch)
            except Exception as e:
                _LOG.warning("callback on_epoch_begin failed: %s", e)

        epoch_t0 = time.perf_counter()

        with tracer.span("torchslicer.epoch", epoch=epoch, n_workers=len(self._proxies)):
            iter_t0 = time.perf_counter()
            for inputs, labels in data_loader:
                data_load_total_ms += (time.perf_counter() - iter_t0) * 1000.0

                batch_id = epoch * n_total + n_batches
                with self._lock:
                    self._batch_losses.clear()
                self._batch_done.clear()

                _shape = (str(tuple(inputs.shape)) if isinstance(inputs, torch.Tensor)
                          else "multimodal")
                with tracer.span(
                    "torchslicer.batch",
                    epoch=epoch,
                    batch_id=batch_id,
                    batch_index=n_batches,
                    input_shape=_shape,
                ) as batch_span:
                    send_t0 = time.perf_counter()
                    try:
                        self._send_batch(batch_id, inputs, labels)
                    except grpc.RpcError:
                        # A gRPC error here means a worker died mid-batch.
                        # Wait briefly for the heartbeat to mark it failed, then raise.
                        self._failure_event.wait(timeout=5.0)
                        if self._failure_event.is_set():
                            failed_idx = next(iter(self._failed_workers))
                            raise WorkerFailureError(
                                failed_idx,
                                self._failure_info.get(failed_idx, "unknown"),
                                epoch,
                                reason=self._failure_reason.get(failed_idx),
                            )
                        raise
                    send_total_ms += (time.perf_counter() - send_t0) * 1000.0

                    wait_t0 = time.perf_counter()
                    # Poll with 1s timeout so a worker failure interrupts the wait
                    while not self._batch_done.wait(timeout=1.0):
                        if self._failure_event.is_set():
                            break
                    if self._failure_event.is_set():
                        failed_idx = next(iter(self._failed_workers))
                        raise WorkerFailureError(
                            failed_idx,
                            self._failure_info.get(failed_idx, "unknown"),
                            epoch,
                            reason=self._failure_reason.get(failed_idx),
                        )
                    wait_total_ms += (time.perf_counter() - wait_t0) * 1000.0

                    with self._lock:
                        loss = (
                            sum(self._batch_losses) / len(self._batch_losses)
                            if self._batch_losses else 0.0
                        )
                    if batch_span:
                        batch_span.set_attribute("loss", loss)

                total_loss += loss
                n_batches  += 1

                if self._run_logger:
                    self._run_logger.log(
                        step=batch_id, epoch=epoch, batch=n_batches,
                        loss=round(loss, 6), phase="batch",
                    )

                if verbose:
                    ep_suffix = f"/{total_epochs}" if total_epochs else ""
                    _LOG.info(
                        "epoch progress epoch=%s%s batch=%s/%s loss=%.4f",
                        epoch,
                        ep_suffix,
                        n_batches,
                        n_total,
                        loss,
                    )

                iter_t0 = time.perf_counter()

        epoch_duration_s = time.perf_counter() - epoch_t0
        avg = total_loss / n_batches if n_batches > 0 else 0.0
        if verbose:
            suffix = f"/{total_epochs}" if total_epochs else ""
            _LOG.info(
                "epoch complete epoch=%s%s avg_loss=%.4f duration_s=%.1f",
                epoch,
                suffix,
                avg,
                epoch_duration_s,
            )

        # Build epoch metrics dict, allow callbacks to augment it
        epoch_metrics = {
            "step":       epoch,
            "epoch":      epoch,
            "loss":       round(avg, 6),
            "duration_s": round(epoch_duration_s, 3),
            "phase":      "epoch",
        }
        for cb in self._callbacks:
            try:
                result = cb.on_epoch_end(epoch, epoch_metrics)
                if isinstance(result, dict):
                    epoch_metrics = result
            except Exception as e:
                _LOG.warning("callback on_epoch_end failed: %s", e)

        # Fault-tolerance: save per-epoch checkpoint for potential recovery
        if self._run_config.fault_tolerance.enabled:
            self._save_ft_checkpoint(epoch)

        if self._run_logger:
            self._run_logger.log(**epoch_metrics)

            # Coordinator-side overhead entry
            if self._run_config.profile.verbosity >= 1:
                self._run_logger.log(
                    step=epoch,
                    epoch=epoch,
                    data_load_total_ms=round(data_load_total_ms, 3),
                    send_total_ms=round(send_total_ms, 3),
                    wait_total_ms=round(wait_total_ms, 3),
                    n_batches=n_batches,
                    phase="coordinator_epoch",
                )

            # Collect per-worker profiling stats
            if self._run_config.profile.verbosity >= 1:
                worker_stats = self._collect_worker_stats(epoch)
                for ws in worker_stats:
                    self._run_logger.log(**ws)

        return {"loss": avg}

    def reinit(
        self,
        model_graph,
        partitions,
        optimizer_cfg: dict,
        criterion_cfg: dict,
        n_micro_batches: int = 1,
        model_name: str = "unknown",
        strategy_name: str = "unknown",
    ) -> None:
        """Re-initialise existing workers with a new model — no server restart or re-discovery.

        Workers stay alive between runs.  Call this after a completed training loop
        (but before ``teardown()``) to push a different model to the same worker set.
        The gRPC server and proxies must already be established via ``setup()``.
        """
        layers = model_graph.get_layers()
        n = len(self._proxies)
        self._n_micro     = max(1, n_micro_batches)
        self._model_name  = model_name
        self._strategy_name = strategy_name
        cfg = self._run_config

        opt_cfg  = _build_optimizer_config(optimizer_cfg)
        crit_cfg = _build_criterion_config(criterion_cfg)

        for i, proxy in enumerate(self._proxies):
            is_last = (i == n - 1)
            partition_layers = [layers[j] for j in partitions[i].layer_indices]
            pred_proto = [
                worker_service_pb2.PredecessorList(indices=list(p))
                for p in (partitions[i].predecessors or [[] for _ in partition_layers])
            ]
            slice_cfg = worker_service_pb2.SliceConfig(
                layers             = _build_layer_configs(partition_layers),
                optimizer          = opt_cfg,
                criterion          = crit_cfg if is_last else None,
                is_last            = is_last,
                prev_worker        = self._proxies[i - 1].address if i > 0 else "",
                next_worker        = self._proxies[i + 1].address if i < n - 1 else "",
                coordinator        = self.coordinator_addr,
                n_micro            = self._n_micro,
                run_id             = self._run_id,
                checkpoint_path    = "",
                worker_index       = i,
                profile_verbosity  = cfg.profile.verbosity,
                profile_memory     = cfg.profile.memory,
                predecessors       = pred_proto,
            )
            res = self._init_worker(proxy, slice_cfg, phase="reinit")
            _LOG.info(
                "worker reinit complete worker=%s address=%s hostname=%s message=%s",
                proxy.name,
                proxy.address,
                res.hostname,
                res.message,
            )

        _LOG.info("reinit complete workers=%s model=%s", n, model_name)

    def teardown(self) -> None:
        self._stop_heartbeats()
        cfg  = self._run_config

        # Resolve the run directory: if logging is enabled, use logging.dir;
        # otherwise fall back to checkpoint.dir for backward compat.
        run_dir = (
            os.path.join(cfg.logging.dir, self._run_id)
            if cfg.logging.enabled
            else os.path.join(cfg.checkpoint.dir, self._run_id)
        )
        ckpt = cfg.checkpoint

        # Signal every worker to shut down, optionally saving a checkpoint first.
        for i, proxy in enumerate(self._proxies):
            try:
                proxy.stub().shutdown(worker_service_pb2.ShutdownRequest(
                    save_checkpoint = ckpt.enabled,
                    checkpoint_dir  = run_dir,   # unified dir: logs + checkpoints together
                    run_id          = self._run_id,
                    epoch           = self._last_epoch,
                    worker_index    = i,
                ))
                _LOG.info(
                    "shutdown sent worker=%s address=%s checkpoint=%s",
                    proxy.name,
                    proxy.address,
                    "yes" if ckpt.enabled else "no",
                )
            except Exception as e:
                _LOG.warning("shutdown failed worker=%s address=%s error=%s", proxy.name, proxy.address, e)

        # Gracefully shut down workers that registered but weren't selected
        self._shutdown_idle_workers()

        if self._grpc_server:
            self._grpc_server.stop(grace=2)
            self._grpc_server = None
        self._proxies.clear()

        if ckpt.enabled:
            self._save_run_state(run_dir)
            # Register checkpoint artifacts in the logger
            if self._run_logger:
                for i in range(len(self._nodes)):
                    name = f"worker_{i}_epoch_{self._last_epoch}.pt"
                    self._run_logger.record_artifact("checkpoint", name)
                self._run_logger.record_artifact("run_state", "run_state.json")

        # Notify callbacks training is done
        log_history = self._run_logger.log_history if self._run_logger else []
        for cb in self._callbacks:
            try:
                cb.on_train_end(log_history)
            except Exception as e:
                _LOG.warning("callback on_train_end failed: %s", e)

        if self._run_logger:
            self._run_logger.flush()

    # ── fault tolerance ────────────────────────────────────────────────────────

    def _start_heartbeats(self, interval: float, timeout: float) -> None:
        self._heartbeat_stop.clear()
        self._heartbeat_threads.clear()
        for i, proxy in enumerate(self._proxies):
            t = threading.Thread(
                target=self._heartbeat_loop,
                args=(i, proxy, interval, timeout),
                daemon=True,
                name=f"heartbeat-{i}",
            )
            self._heartbeat_threads.append(t)
            t.start()

    def _heartbeat_loop(self, idx: int, proxy: "_WorkerProxy",
                        interval: float, timeout: float) -> None:
        while not self._heartbeat_stop.wait(interval):
            try:
                proxy.stub().get_stats(
                    worker_service_pb2.GetStatsRequest(
                        run_id=self._run_id, epoch=-1, verbosity=0
                    ),
                    timeout=timeout,
                )
            except grpc.RpcError:
                if not self._heartbeat_stop.is_set():
                    self._on_worker_failure(idx, proxy)
                break

    def _on_worker_failure(self, idx: int, proxy: "_WorkerProxy") -> None:
        with self._lock:
            if idx in self._failed_workers:
                return
            self._failed_workers.add(idx)
            self._failure_info[idx] = proxy.address
        _LOG.error("worker unreachable worker=%s address=%s", idx, proxy.address)
        # Unblock train_epoch()'s _batch_done.wait() poll loop
        self._batch_done.set()
        self._failure_event.set()

    def _on_worker_runtime_error(self, request) -> None:
        idx = int(request.worker_index)
        address = self._failure_info.get(idx)
        if address is None and 0 <= idx < len(self._proxies):
            address = self._proxies[idx].address
        reason = f"{request.phase}: {request.message}"
        with self._lock:
            if idx in self._failed_workers:
                return
            self._failed_workers.add(idx)
            self._failure_info[idx] = address or request.worker or "unknown"
            self._failure_reason[idx] = reason
        _LOG.error(
            "worker runtime error worker=%s address=%s phase=%s batch_id=%s message=%s",
            idx,
            self._failure_info[idx],
            request.phase,
            request.batch_id,
            request.message,
        )
        self._batch_done.set()
        self._failure_event.set()

    def _init_worker(self, proxy: "_WorkerProxy", slice_cfg, phase: str):
        policy = RetryPolicy(
            max_attempts=self._run_config.startup.worker_init_max_attempts,
            delay_s=self._run_config.startup.worker_init_delay_s,
            rpc_timeout_s=self._run_config.startup.worker_init_rpc_timeout_s,
        )

        return init_worker_with_retry(
            proxy.stub(),
            slice_cfg,
            address=f"{proxy.name} ({proxy.address})",
            policy=policy,
            logger=_LOG,
            phase=phase,
        )

    def _stop_heartbeats(self) -> None:
        self._heartbeat_stop.set()
        for t in self._heartbeat_threads:
            t.join(timeout=2.0)
        self._heartbeat_threads.clear()

    def _save_ft_checkpoint(self, epoch: int) -> None:
        """Ask all surviving workers to save a checkpoint (without stopping)."""
        cfg = self._run_config
        run_dir = (
            os.path.join(cfg.logging.dir, self._run_id)
            if cfg.logging.enabled
            else os.path.join(cfg.checkpoint.dir, self._run_id)
        )
        os.makedirs(run_dir, exist_ok=True)
        self._ft_checkpoint_dir = run_dir
        for i, proxy in enumerate(self._proxies):
            try:
                proxy.stub().save_checkpoint(
                    worker_service_pb2.ShutdownRequest(
                        save_checkpoint = True,
                        checkpoint_dir  = run_dir,
                        run_id          = self._run_id,
                        epoch           = epoch,
                        worker_index    = i,
                    ),
                    timeout=10.0,
                )
            except Exception as e:
                _LOG.warning("ft checkpoint failed worker=%s address=%s error=%s", i, proxy.address, e)

    def _load_all_slices(self, epoch: int) -> "list | None":
        """Load all workers' checkpoint state_dicts and apply them to _stored_layers.

        Returns the flat layer list with trained weights, or None if any checkpoint
        is missing (recovery will fall back to initial weights).
        """
        if not self._ft_checkpoint_dir or not self._stored_layers:
            return None
        all_layers = list(self._stored_layers)
        for i, indices in enumerate(self._stored_partition_indices):
            path = os.path.join(
                self._ft_checkpoint_dir, f"worker_{i}_epoch_{epoch}.pt"
            )
            if not os.path.exists(path):
                _LOG.warning("recovery checkpoint missing path=%s", path)
                return None
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                sd   = ckpt.get("layer_state_dict", {})
                partition_mods = [all_layers[j] for j in indices]
                tmp_sl = SplitLayer(partition_mods)
                tmp_sl.load_state_dict(sd, strict=False)
                # Weights updated in-place on the shared module objects in all_layers
            except Exception as e:
                _LOG.warning("recovery checkpoint load failed path=%s error=%s", path, e)
                return None
        return all_layers

    @staticmethod
    def _uniform_split(layers: list, n: int) -> "list[list]":
        k, r   = divmod(len(layers), n)
        groups, start = [], 0
        for i in range(n):
            size = k + (1 if i < r else 0)
            groups.append(layers[start : start + size])
            start += size
        return groups

    def _build_recovery_partitions(self, n_alive: int) -> list[Partition]:
        """Rebuild partitions for recovery at the same abstraction level as setup().

        If setup used an explicit pack(model) stage list, those packed stages
        remain the atomic units here. Recovery only supports the same partition
        semantics as normal execution: multi-input nodes must remain within a
        single partition.
        """
        if self._stored_model_graph is None:
            raise RuntimeError("[recovery] missing stored model graph")

        layer_groups = self._uniform_split(list(range(len(self._stored_model_graph))), n_alive)
        partitions = []
        for i, layer_indices in enumerate(layer_groups):
            predecessors = UniformSplitter._intra_predecessors(
                self._stored_model_graph, layer_indices
            )
            partitions.append(Partition(
                index=i,
                layer_indices=layer_indices,
                predecessors=predecessors,
            ))

        UniformSplitter().validate(self._stored_model_graph, partitions)
        return partitions

    def recover(self, last_good_epoch: int) -> None:
        """Re-slice the model across surviving workers and reinit from last checkpoint.

        Called by SlicedModel.train() after catching WorkerFailureError.
        Survivors keep running; failed workers are removed from the pool.
        If fault-tolerance checkpoints exist they are loaded so trained weights are preserved;
        otherwise recovery falls back to the initial layer weights from setup().
        """
        self._stop_heartbeats()
        cfg = self._run_config

        alive = [(i, p) for i, p in enumerate(self._proxies)
                 if i not in self._failed_workers]
        n_alive = len(alive)
        if n_alive == 0:
            raise RuntimeError("[recovery] all workers failed — cannot continue")

        _LOG.warning(
            "recovery reslicing failed_workers=%s survivors=%s addresses=%s",
            sorted(self._failed_workers),
            n_alive,
            [p.address for _, p in alive],
        )

        # Load trained weights from last epoch's FT checkpoints
        all_layers = (
            self._load_all_slices(last_good_epoch)
            if last_good_epoch > 0
            else None
        )
        if all_layers is None:
            _LOG.warning("recovery using initial layer weights because no checkpoint was available")
            all_layers = list(self._stored_layers)

        # Rebuild partitions at the same abstraction level as setup().
        partitions = self._build_recovery_partitions(n_alive)

        # Rebuild proxy + node lists with new sequential indices
        self._nodes   = [self._nodes[old_i]   for old_i, _ in alive]
        self._proxies = [proxy                 for _,     proxy in alive]
        for new_i, proxy in enumerate(self._proxies):
            proxy.worker_index = new_i

        # Update stored partition state so subsequent _load_all_slices is consistent
        self._stored_layers = all_layers
        self._stored_partition_indices = [list(p.layer_indices) for p in partitions]

        # Send new SliceConfig to each survivor (reinit wires up new prev/next stubs)
        for new_i, (proxy, partition) in enumerate(zip(self._proxies, partitions)):
            is_last  = (new_i == n_alive - 1)
            partition_layers = [all_layers[j] for j in partition.layer_indices]
            pred_proto = [
                worker_service_pb2.PredecessorList(indices=list(p))
                for p in partition.predecessors
            ]
            slice_cfg = worker_service_pb2.SliceConfig(
                layers            = _build_layer_configs(partition_layers),
                optimizer         = self._stored_opt_cfg,
                criterion         = self._stored_crit_cfg if is_last else None,
                is_last           = is_last,
                prev_worker       = self._proxies[new_i - 1].address if new_i > 0 else "",
                next_worker       = self._proxies[new_i + 1].address if new_i < n_alive - 1 else "",
                coordinator       = self.coordinator_addr,
                n_micro           = self._n_micro,
                run_id            = self._run_id,
                checkpoint_path   = "",
                worker_index      = new_i,
                profile_verbosity = cfg.profile.verbosity,
                profile_memory    = cfg.profile.memory,
                predecessors      = pred_proto,
            )
            res = self._init_worker(proxy, slice_cfg, phase="recovery")
            _LOG.info("recovery reinit complete worker=%s address=%s message=%s", new_i, proxy.address, res.message)

        # Reset failure state
        self._failed_workers.clear()
        self._failure_info.clear()
        self._failure_event.clear()
        self._batch_done.clear()

        # Restart heartbeats
        ft = cfg.fault_tolerance
        if ft.enabled:
            self._start_heartbeats(ft.heartbeat_interval_s, ft.ping_timeout_s)

        _LOG.info("recovery complete active_workers=%s", n_alive)

    # ── internal ───────────────────────────────────────────────────────────────

    @property
    def _run_id(self) -> str:
        if hasattr(self._discovery, "run_id"):
            return self._discovery.run_id
        return self._run_config.run_id

    def _shutdown_idle_workers(self) -> None:
        """Send Shutdown RPC to workers that registered but weren't selected."""
        idle = self._discovery.idle_nodes()
        if not idle:
            return
        _LOG.info("shutting down idle workers count=%s addresses=%s", len(idle), [nd.address for nd in idle])
        for nd in idle:
            try:
                stub = worker_service_pb2_grpc.WorkerServiceStub(_channel(nd.address))
                stub.shutdown(worker_service_pb2.ShutdownRequest(
                    save_checkpoint=False,
                    checkpoint_dir="",
                    run_id=self._run_id,
                    epoch=0,
                    worker_index=0,
                ), timeout=5.0)
                _LOG.info("idle worker shutdown complete node_id=%s address=%s", nd.node_id, nd.address)
            except Exception as e:
                _LOG.warning("idle worker shutdown failed address=%s error=%s", nd.address, e)

    def _collect_worker_stats(self, epoch: int) -> list:
        """Pull profiling stats from all workers after an epoch."""
        results = []
        verbosity = self._run_config.profile.verbosity
        for proxy in self._proxies:
            try:
                resp = proxy.stub().get_stats(worker_service_pb2.GetStatsRequest(
                    run_id    = self._run_id,
                    epoch     = epoch,
                    verbosity = verbosity,
                ))
                entry = self._worker_stats_to_dict(resp, epoch)
                results.append(entry)
                # At verbosity=3, also emit per-batch rows
                if verbosity >= 3:
                    for b in resp.batches:
                        results.append({
                            "step":         epoch,
                            "epoch":        epoch,
                            "worker":       resp.worker_index,
                            "batch_id":     b.batch_id,
                            "forward_ms":   b.forward_ms,
                            "backward_ms":  b.backward_ms,
                            "optimizer_ms": b.optimizer_ms,
                            "send_fwd_ms":  b.send_fwd_ms,
                            "send_bwd_ms":  b.send_bwd_ms,
                            "idle_fwd_ms":  b.idle_fwd_ms,
                            "idle_bwd_ms":  b.idle_bwd_ms,
                            "peak_mem_mb":  b.peak_mem_mb,
                            "phase":        "worker_batch",
                        })
            except Exception as e:
                _LOG.warning("get_stats failed worker=%s address=%s error=%s", proxy.name, proxy.address, e)
        return results

    def _worker_stats_to_dict(self, resp, epoch: int) -> dict:
        """Convert WorkerStatsResponse to a flat dict for metrics.jsonl."""
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
            "step":   epoch,
            "epoch":  epoch,
            "worker": resp.worker_index,
            "phase":  "worker_epoch",
        }
        for phase_name, proto_field in [
            ("forward",   resp.forward),
            ("backward",  resp.backward),
            ("optimizer", resp.optimizer),
            ("send_fwd",  resp.send_fwd),
            ("send_bwd",  resp.send_bwd),
            ("idle_fwd",  resp.idle_fwd),
            ("idle_bwd",  resp.idle_bwd),
        ]:
            entry.update(_ps(proto_field, phase_name))
        if resp.peak_mem_mb > 0:
            entry["peak_mem_mb"] = round(resp.peak_mem_mb, 2)
        if resp.end_mem_mb > 0:
            entry["end_mem_mb"] = round(resp.end_mem_mb, 2)
        entry["n_batches"] = resp.n_batches
        return entry

    def _resolve_checkpoint_paths(self) -> dict:
        """Return {worker_index: path} for resume, or {} if not resuming."""
        resume = self._run_config.checkpoint.resume
        if not resume:
            return {}
        try:
            with open(resume) as f:
                state = json.load(f)
            ckpt_dir = state.get("checkpoint_dir", self._run_config.checkpoint.dir)
            epoch    = state.get("epoch", 0)
            run_id   = state.get("run_id", self._run_id)
            paths = {}
            for i in range(len(self._proxies)):
                p = os.path.join(ckpt_dir, f"worker_{i}_epoch_{epoch}.pt")
                if os.path.exists(p):
                    paths[i] = p
            return paths
        except Exception as e:
            _LOG.warning("resume run_state load failed path=%s error=%s", resume, e)
            return {}

    def _save_run_state(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        state = {
            "run_id":         self._run_id,
            "epoch":          self._last_epoch,
            "checkpoint_dir": run_dir,
            "n_workers":      len(self._nodes) if self._nodes else
                              self._run_config.discovery.n_workers,
        }
        path = os.path.join(run_dir, "run_state.json")
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        _LOG.info("run_state saved path=%s", path)

    def _send_batch(self, batch_id: int, inputs, labels: torch.Tensor):
        main, aux = _unpack_inputs(inputs)
        M = self._n_micro

        if M > 1:
            micro_mains  = main.chunk(M)
            micro_labels = labels.chunk(M)
            # Chunk aux tensors that share the batch dim; broadcast others
            micro_aux = [
                {k: v.chunk(M)[m] if v.dim() > 0 and v.shape[0] == main.shape[0] else v
                 for k, v in aux.items()}
                for m in range(M)
            ]
            for m in range(M):
                mbid = batch_id * M + m
                self._proxies[-1].stub().forward(worker_service_pb2.ForwardRequest(
                    batch_id=mbid,
                    label=_serialize_tensor(micro_labels[m]),
                ))
                self._proxies[0].stub().forward(worker_service_pb2.ForwardRequest(
                    batch_id=mbid,
                    input=_serialize_tensor(micro_mains[m]),
                    aux_inputs={k: _serialize_tensor(v) for k, v in micro_aux[m].items()},
                ))
        else:
            self._proxies[-1].stub().forward(worker_service_pb2.ForwardRequest(
                batch_id=batch_id,
                label=_serialize_tensor(labels),
            ))
            self._proxies[0].stub().forward(worker_service_pb2.ForwardRequest(
                batch_id=batch_id,
                input=_serialize_tensor(main),
                aux_inputs={k: _serialize_tensor(v) for k, v in aux.items()},
            ))
