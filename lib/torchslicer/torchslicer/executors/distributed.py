import io
import json
import os
import threading
import time
import uuid
from concurrent import futures
from dataclasses import dataclass, field

import torch
from torch import nn

from ..core.split_layer import SplitLayer
from ..core.tied import find_tied_parameters

from .base import BaseExecutor
from .startup import init_worker_with_retry
from .worker import (_TensorPeerClient, _tensor_addr, measure_link, WorkerServicer,
                     resolve_device, get_available_memory_mb)
from ..monitor import tracer
from ..monitor.process_logger import get_logger, configure as configure_process_logging
from ..monitor.run_logger import RunLogger
from ..monitor.callback import TrainingCallback
from ..discovery.base import BaseDiscovery, NodeInfo
from ..config import RunConfig, CheckpointConfig
from ..retry import RetryPolicy
from ..strategies.base import Partition
from ..strategies.uniform import UniformSplitter
from ..core.tied import find_tied_parameters as _find_ties
from ..sl.hooks import resolve_cut_hooks, tensor_nbytes
from ..sl.roles import Placement, resolve_setting
from ..sl.metrics import CommMeter
from ..sl.clients import ClientSession, SplitLayerClientState
from ..transport.tunnel import Router

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
        (torch.uint8,    worker_service_pb2.UINT8),
        (torch.int8,     worker_service_pb2.INT8),
        (torch.bool,     worker_service_pb2.BOOL),
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
    with tracer.span(
        "torchslicer.tensor.serialize",
        device=str(t.device),
        dtype=str(t.dtype),
        shape=str(tuple(t.shape)),
        role="coordinator",
    ):
        data = _tensor_to_bytes(t)
    return worker_service_pb2.Tensor(
        data=data,
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


def build_tied_configs(partitions_layers: list, addresses: list) -> list:
    """Per-partition ``TiedParam`` lists for ``SliceConfig.tied`` (see core/tied.py).

    ``addresses[i]`` is the worker hosting partition ``i``.
    """
    configs = [[] for _ in partitions_layers]
    for g in find_tied_parameters(partitions_layers):
        for p_idx, names in g.members.items():
            owner = p_idx == g.owner
            peers = [addresses[r] for r in g.replicas] if owner else [addresses[g.owner]]
            configs[p_idx].append(worker_service_pb2.TiedParam(
                group=g.group_id, names=names, owner=owner,
                peers=peers if g.cross_partition else []))
    return configs


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
    # lr is also kept in extra_params: the proto ``lr`` field is float32, and a
    # rounded learning rate silently changes training (measurable with Adam).
    extra = dict(cfg.get("params", {}))
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
            compute_gflops = request.node.compute_gflops,
            max_memory_mb  = request.node.max_memory_mb,
            first_slice    = request.node.first_slice,
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

class _LocalStub:
    """Calls a servicer in this process with a gRPC stub's signature (no serialization
    of the call, no network). Used for the P2P driver's own worker and coordinator."""

    def __init__(self, servicer):
        self._servicer = servicer

    def __getattr__(self, name):
        method = getattr(self._servicer, name)
        return lambda request, timeout=None, **_: method(request, None)


@dataclass
class _WorkerProxy:
    name:         str
    address:      str
    worker_index: int
    _stub:          object = field(default=None, repr=False)
    _tensor_client: object = field(default=None, repr=False)
    local:          object = field(default=None, repr=False)   # in-process WorkerServicer

    def connect(self, router: Router, tensor_transport: str = "grpc", tensor_port_offset: int = 1):
        if self.local is not None:
            self._stub = _LocalStub(self.local)
            return
        self._stub = worker_service_pb2_grpc.WorkerServiceStub(router.channel(self.address))
        if tensor_transport == "tcp":
            self._tensor_client = _TensorPeerClient(
                _tensor_addr(self.address, tensor_port_offset)
            )

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

    executor_name = "distributed"   # recorded in the run manifest
    topology      = "centralized"   # default SL setting: pipeline (see sl/roles.py)
    _local_tag    = "data_owner"    # tag of the slices this process hosts
    _owner_name   = "the data owner (coordinator)"

    def __init__(
        self,
        discovery: BaseDiscovery,
        coordinator_addr: str,
        coordinator_bind_addr: str | None = None,
        run_config: RunConfig = None,
        *,
        device: str | None = None,
    ):
        """``device``: device of the slices this process hosts as the data owner
        (``sl.setting`` vanilla / u_shaped); default ``network.device``."""
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
        self._planner  = None   # planner(nodes) -> partitions, from setup()
        self.partitions: list = []   # the partitions in use (after planning / recovery)
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

        # Where this process's calls go (direct, by tunnel, or through a relay), and
        # what it serves to calls arriving by tunnel (transport/tunnel.py). Behind NAT
        # (transport.nat) it listens on nothing: it opens a tunnel to every remote
        # worker, and the workers' calls to it come back over those.
        cfg = self._run_config
        self._nat = bool(cfg.transport.nat)
        self._router = Router(cfg.transport.relay or None,
                              keepalive_s=cfg.transport.tunnel_keepalive_s)

        # Split-learning setting: the slices the data owner (this process) hosts
        # in-process: slice 0 ("front", served on the coordinator's gRPC server) for
        # vanilla / u_shaped, plus the last slice ("tail", own server) for u_shaped.
        self._setting = resolve_setting(cfg.sl.setting, self.topology)
        self.local = (self._new_local_servicer(True, [self._local_tag], device)
                      if self._setting != "pipeline" else None)
        self._tail = (self._new_local_servicer(False, [self._local_tag, "tail"], device)
                      if self._setting == "u_shaped" else None)
        self._tail_server = None
        self._local: dict = {}        # slice index -> in-process WorkerServicer
        self._placement: Placement = None
        self._cut_hooks_spec = None
        self._hooks: dict = {}
        self._meter: CommMeter = None

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
        planner=None,
        cut_hooks=None,
        algorithm=None,
    ) -> None:
        """``planner(nodes) -> partitions`` (from ``SlicedModel``) re-plans the split for
        the discovered nodes, in slice order, once their capabilities are known; it is
        also used by ``recover()``. Without it, ``partitions`` are used as given.
        ``cut_hooks``: ``ts.slice(cut_hooks=)``, added to the config's ``sl.cut_hooks``."""
        layers = model_graph.get_layers()
        n = len(partitions)
        self._planner = planner
        self._placement = Placement(self._setting, n)
        self._cut_hooks_spec = cut_hooks
        self._hooks = resolve_cut_hooks(cut_hooks, n, self._run_config.sl.cut_hooks)
        self._meter = CommMeter(n)
        self._local = {i: srv for i, (srv, _) in self._hosted(n).items()}
        self._n_micro = max(1, n_micro_batches)
        self._callbacks = list(callbacks or [])
        self._model_name = model_name
        self._strategy_name = strategy_name
        cfg = self._run_config
        configure_process_logging(cfg.logging.level)

        if (self._nat and not self._router.relay and n > len(self._local)
                and hasattr(self._discovery, "handle_register")):
            raise ValueError(
                f"{self._owner_name} is behind NAT (transport.nat), so workers can't register "
                "with it: list them (discovery.backend: static, discovery.peers) or route "
                "everything through a relay (transport.relay)")

        # Start embedded coordinator gRPC server FIRST so the Register endpoint
        # is available before discover() blocks waiting for registrations.
        self._start_server(n)
        self._nodes = self._discover(n)
        _LOG.info("workers ready count=%s addresses=%s", n, [nd.address for nd in self._nodes])

        # Build proxies from discovered node list (addresses come from workers themselves)
        self._proxies = [self._make_proxy(i, nd) for i, nd in enumerate(self._nodes)]
        for proxy in self._proxies:
            proxy.connect(
                self._router,
                tensor_transport=cfg.transport.tensor,
                tensor_port_offset=cfg.transport.tensor_port_offset,
            )
        self._open_tunnels()

        # Capabilities → slice order → plan (deferred planning)
        self._probe_nodes()
        self._order_nodes()
        if cfg.planning.measure_links:
            self._measure_links()
        if planner is not None and cfg.planning.heterogeneous:
            partitions = planner(list(self._nodes))
            if len(partitions) != n:
                raise RuntimeError(f"planner returned {len(partitions)} partitions for {n} nodes")
        self.partitions = partitions
        self._log_plan(model_graph, partitions)

        # Initialise RunLogger now that we have run_id and worker info
        if cfg.logging.enabled:
            run_dir = os.path.join(cfg.logging.dir, self._run_id)
            self._run_logger = RunLogger(run_id=self._run_id, run_dir=run_dir)
            layer_names = [type(l).__name__ for l in layers]
            self._run_logger.record_executor(self.executor_name)
            self._run_logger.record_config(cfg)
            self._run_logger.record_model(model_name, layer_names,
                                          node_names=model_graph.names(), expand=model_graph.expand)
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

        self._init_slices(layers, partitions, phase="init", checkpoint_paths=checkpoint_paths)

        # Failure detection (heartbeats) runs unless disabled; recovery only with
        # fault_tolerance.enabled. Without recovery a dead worker fails the run
        # (WorkerFailureError) instead of hanging it.
        ft = cfg.fault_tolerance
        if ft.enabled:
            self._max_fault_retries = ft.max_retries
        if ft.enabled or ft.detect_failures:
            self._start_heartbeats(ft.heartbeat_interval_s, ft.ping_timeout_s)
            _LOG.info(
                "failure detection on heartbeat_interval_s=%.1f ping_timeout_s=%.1f recovery=%s",
                ft.heartbeat_interval_s,
                ft.ping_timeout_s,
                ft.enabled,
            )

        # Algorithm
        from ..algorithm.base import Algorithm
        from ..algorithm.builtin import DefaultAlgorithm
        if algorithm is None and getattr(self._run_config.sl, "algorithm", ""):
            from ..algorithm.registry import get as _get_algo
            algorithm = _get_algo(self._run_config.sl.algorithm)
        self._algorithm = algorithm if algorithm is not None else DefaultAlgorithm()
        self._algorithm.configure(len(partitions), None, self._placement)

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
        session          = data_loader if isinstance(data_loader, ClientSession) else None
        if session is not None:
            adapter = self._session_adapter()
            batches = session.batches(adapter, epoch)
        else:
            batches = ((None, b) for b in data_loader)
        self._meter.reset()

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

        with tracer.span("torchslicer.epoch", kind="CHAIN", epoch=epoch, n_workers=len(self._proxies)):
            iter_t0 = time.perf_counter()
            for client, (inputs, labels) in batches:
                data_load_total_ms += (time.perf_counter() - iter_t0) * 1000.0

                batch_id = epoch * n_total + n_batches
                with self._lock:
                    self._batch_losses.clear()
                self._batch_done.clear()

                _shape = (str(tuple(inputs.shape)) if isinstance(inputs, torch.Tensor)
                          else "multimodal")
                with tracer.span(
                    "torchslicer.batch",
                    kind="CHAIN",
                    new_trace=True,
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
                    batch_timeout = self._run_config.fault_tolerance.batch_timeout_s
                    # Poll with 1s timeout so a worker failure interrupts the wait
                    while not self._batch_done.wait(timeout=1.0):
                        if self._failure_event.is_set():
                            break
                        if batch_timeout and time.perf_counter() - wait_t0 > batch_timeout:
                            raise RuntimeError(
                                f"batch {batch_id} not completed within {batch_timeout}s "
                                "and no worker failure was detected (lost message?)")
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
                    extra = {"client": client} if client is not None else {}
                    self._run_logger.log(
                        step=batch_id, epoch=epoch, batch=n_batches,
                        loss=round(loss, 6), phase="batch", **extra,
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
        if session is not None:
            session.finish(adapter)
        # one get_stats per worker: communication + compute (always), profiling (verbosity)
        # Algorithm round-end (aggregation)
        if session is not None and hasattr(self, "_algorithm"):
            new_states = self._algorithm.on_round_end(list(session.states), {}, epoch)
            if new_states is not None:
                session.states = list(new_states)
            agg_client, _ = self._algorithm.aggregate(list(session.states), [], epoch)
            if agg_client is not None:
                session.states = [agg_client] * len(session.states)
        stats = self._epoch_stats(epoch)
        self._add_worker_comm(stats)
        sl_metrics = self._meter.summary(self._placement, self._run_config.sl.link_bandwidth_mbps,
                                         self._run_config.sl.link_latency_ms)
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
            **sl_metrics,
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

            # Per-worker profiling stats
            if self._run_config.profile.verbosity >= 1:
                for ws in self._worker_stats_rows(stats, epoch):
                    self._run_logger.log(**ws)

        return {"loss": avg, **sl_metrics}

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

        self._stored_opt_cfg  = _build_optimizer_config(optimizer_cfg)
        self._stored_crit_cfg = _build_criterion_config(criterion_cfg)
        self._stored_model_graph       = model_graph
        self._stored_layers            = layers
        self._stored_partition_indices = [list(p.layer_indices) for p in partitions]
        self._init_slices(layers, partitions, phase="reinit")

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
        if ckpt.enabled:
            self._save_checkpoint_layout(self._last_epoch, run_dir)

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
        if self._tail_server:
            self._tail_server.stop(grace=1)
            self._tail_server = None
        for srv in (self.local, self._tail):
            if srv is not None:
                srv.stop_tensor_server()
        self._proxies.clear()
        self._router.close()

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
            except grpc.RpcError as exc:
                # A crashed worker refuses the connection (UNAVAILABLE) at once; a
                # live but overloaded one times out.  Confirm timeouts with a longer
                # probe so a stall is not mistaken for a failure.
                if exc.code() == grpc.StatusCode.DEADLINE_EXCEEDED and not self._heartbeat_stop.is_set():
                    try:
                        proxy.stub().get_stats(
                            worker_service_pb2.GetStatsRequest(run_id=self._run_id, epoch=-1, verbosity=0),
                            timeout=3 * timeout,
                        )
                        _LOG.warning("heartbeat timeout recovered worker=%s address=%s", idx, proxy.address)
                        continue
                    except grpc.RpcError:
                        pass
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

    def _report_confirmed(self, idx: int) -> bool:
        """A runtime-error report names a worker index but is not authenticated, so any
        worker could report another one.  Ask the reported worker: it is failed if it
        is unreachable or itself holds a fatal error."""
        ft = self._run_config.fault_tolerance
        if not ft.verify_reports or not self._proxies:   # nothing to probe before setup
            return True
        if not 0 <= idx < len(self._proxies):
            return False
        try:
            res = self._proxies[idx].stub().get_stats(
                worker_service_pb2.GetStatsRequest(run_id=self._run_id, epoch=-1, verbosity=0),
                timeout=max(ft.ping_timeout_s, 1.0),
            )
        except grpc.RpcError:
            return True
        return bool(res.fatal_error)

    def _on_worker_runtime_error(self, request) -> None:
        idx = int(request.worker_index)
        address = self._failure_info.get(idx)
        if address is None and 0 <= idx < len(self._proxies):
            address = self._proxies[idx].address
        reason = f"{request.phase}: {request.message}"
        if not self._report_confirmed(idx):
            _LOG.warning("failure report rejected: worker=%s is reachable and reports no fatal error "
                         "(phase=%s batch_id=%s message=%s)", idx, request.phase, request.batch_id, request.message)
            return
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
        self._save_checkpoint_layout(epoch, run_dir)
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

    def _build_recovery_partitions(self, n_alive: int, nodes: list | None = None) -> list[Partition]:
        """Re-plan for the survivors with the run's strategy (``planner``), given their
        capabilities. Falls back to a uniform split when there is no planner or it can't
        produce ``n_alive`` partitions (e.g. named cuts fixed for more workers).

        The uniform fallback works at the same abstraction level as setup().

        If setup used an explicit pack(model) stage list, those packed stages
        remain the atomic units here. Recovery only supports the same partition
        semantics as normal execution: multi-input nodes must remain within a
        single partition.
        """
        if self._stored_model_graph is None:
            raise RuntimeError("[recovery] missing stored model graph")

        planner = getattr(self, "_planner", None)
        if planner is not None and nodes is not None:
            try:
                partitions = planner(list(nodes))
                if len(partitions) == n_alive:
                    return partitions
                _LOG.warning("recovery planner made %s partitions for %s survivors; using uniform",
                             len(partitions), n_alive)
            except ValueError as e:
                _LOG.warning("recovery planner failed (%s); using uniform", e)

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

        lost_local = sorted(i for i in self._failed_workers
                            if i < len(self._proxies) and self._proxies[i].local is not None)
        if lost_local:
            raise RuntimeError(
                f"{self._owner_name}'s own slice failed (slice {lost_local[0]}: "
                f"{self._failure_reason.get(lost_local[0], 'see the log')}); it holds the data "
                "(and with u_shaped the labels), so its slices can't move to another node. "
                "The run stops.")
        alive = [(i, p) for i, p in enumerate(self._proxies)
                 if i not in self._failed_workers]
        for i in self._failed_workers:
            if i < len(self._proxies):
                self._router.release(self._proxies[i].address)
        n_alive = len(alive)
        if n_alive == 0:
            raise RuntimeError("[recovery] all workers failed — cannot continue")
        self._placement = Placement(self._setting, n_alive)   # u_shaped needs >= 3

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

        # Rebuild proxy + node lists with new sequential indices
        self._nodes   = [self._nodes[old_i]   for old_i, _ in alive]
        self._proxies = [proxy                 for _,     proxy in alive]
        for new_i, proxy in enumerate(self._proxies):
            proxy.worker_index = new_i
        self._local = {i: p.local for i, p in enumerate(self._proxies) if p.local is not None}
        self._hooks = resolve_cut_hooks(self._cut_hooks_spec, n_alive, cfg.sl.cut_hooks,
                                        drop_missing=True)
        self._meter = CommMeter(n_alive)
        if cfg.planning.measure_links:
            self._measure_links()   # survivors have new neighbours

        # Re-plan for the survivors (their real capabilities), see _build_recovery_partitions
        partitions = self._build_recovery_partitions(
            n_alive, self._nodes if cfg.planning.heterogeneous else None)
        self.partitions = partitions
        self._log_plan(self._stored_model_graph, partitions)

        # Update stored partition state so subsequent _load_all_slices is consistent
        self._stored_layers = all_layers
        self._stored_partition_indices = [list(p.layer_indices) for p in partitions]

        # Send new SliceConfig to each survivor (reinit wires up new prev/next stubs)
        self._init_slices(all_layers, partitions, phase="recovery")

        # Reset failure state
        self._failed_workers.clear()
        self._failure_info.clear()
        self._failure_event.clear()
        self._batch_done.clear()

        # Restart heartbeats
        ft = cfg.fault_tolerance
        if ft.enabled or ft.detect_failures:
            self._start_heartbeats(ft.heartbeat_interval_s, ft.ping_timeout_s)

        _LOG.info("recovery complete active_workers=%s", n_alive)

    # ── topology steps (overridden by P2PExecutor) ─────────────────────────────

    def _start_server(self, n: int) -> None:
        """Start the gRPC server hosting the CoordinatorService (and the data owner's
        slices). The same services are served to calls arriving by tunnel; behind NAT
        those are the only calls, and no server is started."""
        coordinator = _CoordinatorServicer(self)
        self._router.serve(self.coordinator_addr, coordinator,
                           coordinator_service_pb2_grpc.add_CoordinatorServiceServicer_to_server)
        for srv, address in self._hosted(n).values():
            self._router.serve(address, srv, worker_service_pb2_grpc.add_WorkerServiceServicer_to_server)
        if self._nat:
            _LOG.info("%s is behind NAT: no listening port; serving %s by tunnel%s",
                      self._owner_name, self._router.addresses(),
                      f" through relay {self._router.relay}" if self._router.relay else "")
            return
        # one thread per worker call in flight, plus one per open tunnel (NAT'd workers)
        self._grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max(8, 2 * n + 4)), options=_GRPC_OPTS
        )
        coordinator_service_pb2_grpc.add_CoordinatorServiceServicer_to_server(
            coordinator, self._grpc_server
        )
        self._add_servicers(self._grpc_server)
        self._router.add_to_server(self._grpc_server)
        bind_addr = self.coordinator_bind_addr
        if not bind_addr:
            port = self.coordinator_addr.split(":")[-1]
            bind_addr = f"[::]:{port}"
        self._grpc_server.add_insecure_port(bind_addr)
        self._grpc_server.start()
        _LOG.info("gRPC server started bind_addr=%s run_id=%s", bind_addr, self._run_id)
        offset = int(self._run_config.transport.tensor_port_offset)
        if self.local is not None and self.local._tensor_server is None:
            self.local._tensor_stop.clear()
            self.local.start_tensor_server(self._bind_port() + offset)
        if self._tail is not None and self._tail_server is None:
            port = self._tail_port()
            self._tail_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                                            options=_GRPC_OPTS)
            worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(self._tail, self._tail_server)
            self._router.add_to_server(self._tail_server)
            self._tail_server.add_insecure_port(f"{self._bind_host()}:{port}")
            self._tail_server.start()
            self._tail._tensor_stop.clear()
            self._tail.start_tensor_server(port + offset)
            _LOG.info("u_shaped tail slice served address=%s", self._tail_address())

    def _add_servicers(self, server) -> None:
        """Hook: extra services on the coordinator's gRPC server. The data owner's slice 0
        is served here (other slices reach it: gradients come back to slice 0)."""
        if self.local is not None:
            worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(self.local, server)

    # ── slices hosted in this process (the data owner's) ───────────────────────

    def _new_local_servicer(self, first_slice: bool, tags: list, device=None):
        cfg = self._run_config
        srv = WorkerServicer()
        srv.device = torch.device(resolve_device(device or cfg.network.device))
        srv._tensor_transport   = cfg.transport.tensor.strip().lower()
        srv._tensor_port_offset = int(cfg.transport.tensor_port_offset)
        srv._persistent = True   # teardown must not stop the shared gRPC server
        srv.declared = {
            "compute_gflops": float(cfg.network.worker_compute_gflops),
            "max_memory_mb":  int(cfg.network.worker_max_memory_mb),
            "first_slice":    first_slice,
            "tags":           [*tags, *cfg.network.worker_tags],
        }
        srv.local_coordinator = _LocalStub(_CoordinatorServicer(self))
        srv.router = self._router
        return srv

    def _open_tunnels(self) -> None:
        """Behind NAT: a tunnel to every remote worker, before their slices are sent,
        so that each can call back (gradients, metrics, errors, U-shaped activations).
        Through a relay the relay's tunnel serves instead."""
        if not self._nat or self._router.relay:
            return
        timeout = float(self._run_config.discovery.timeout)
        deadline = time.monotonic() + timeout
        remote = [p.address for p in self._proxies if p.local is None]
        for address in remote:
            self._router.keep(address)
        for address in remote:
            try:
                self._router.keep(address, wait=max(0.0, deadline - time.monotonic()))
            except grpc.RpcError as e:
                raise RuntimeError(f"{self._owner_name} is behind NAT and could not open a "
                                   f"tunnel to worker {address}: {e.details()}") from None
        _LOG.info("tunnels open to %s", remote)

    def _bind_port(self) -> int:
        return int((self.coordinator_bind_addr or self.coordinator_addr).rsplit(":", 1)[-1])

    def _bind_host(self) -> str:
        if self.coordinator_bind_addr:
            return self.coordinator_bind_addr.rsplit(":", 1)[0]
        return "[::]"

    def _tail_port(self) -> int:
        return int(self._run_config.sl.tail_port) or self._bind_port() + 2

    def _tail_address(self) -> str:
        return f"{self.coordinator_addr.rsplit(':', 1)[0]}:{self._tail_port()}"

    @property
    def hosted_slices(self) -> int:
        """How many slices this process hosts (0, 1, or 2 for u_shaped): a run on
        ``n`` slices needs ``n - hosted_slices`` remote workers."""
        return (self.local is not None) + (self._tail is not None)

    def _hosted(self, n: int) -> dict:
        """``{slice index: (servicer, address)}`` for the slices hosted in this process."""
        out = {}
        if self.local is not None:
            out[0] = (self.local, self.coordinator_addr)
        if self._tail is not None:
            out[n - 1] = (self._tail, self._tail_address())
        return out

    def _session_adapter(self):
        """Client-state swapping for a multi-client session: over the client slices,
        which this process hosts (so swapping needs no network)."""
        if not self._placement.client_slices:
            raise ValueError("a multi-client session needs the data owner to host the client "
                             "slices: set sl.setting to vanilla or u_shaped")
        if self._run_config.fault_tolerance.enabled:
            raise ValueError("multi-client sessions don't support fault_tolerance.enabled "
                             "(recovery would not restore every client's state)")
        from .local import check_session_ties
        layers = self._stored_layers
        check_session_ties(_find_ties([[layers[j] for j in p.layer_indices] for p in self.partitions]),
                           self._placement)
        return SplitLayerClientState({i: self._local[i].layer for i in self._placement.client_slices})

    def _discover(self, n: int) -> list:
        """The ``n`` nodes to run on, in slice order (slice i goes to node i): the slices
        this process hosts, and ``n`` minus those discovered remote workers."""
        hosted = self._hosted(n)
        remote = self._discover_remote(n - len(hosted)) if n > len(hosted) else []
        it = iter(remote)
        nodes = []
        for i in range(n):
            if i not in hosted:
                nodes.append(next(it))
                continue
            srv, addr = hosted[i]
            dev = str(srv.device)
            nodes.append(NodeInfo(node_id=addr, address=addr, device=dev,
                                  memory_mb=get_available_memory_mb(dev),
                                  tags=list(srv.declared["tags"]), first_slice=(i == 0)))
        return nodes

    def _discover_remote(self, n: int) -> list:
        """``n`` remote workers from the discovery backend, in discovery order."""
        cfg = self._run_config
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
        if tag_filter:
            discover_kwargs["tag_filter"] = tag_filter
        return self._discovery.discover(**discover_kwargs)

    def _probe_nodes(self) -> None:
        """Fill each node's capabilities from the worker itself (``describe``), and measure
        the network from this process to it (``planning.measure_network``)."""
        plan = self._run_config.planning
        # A static peer list names addresses, not live workers: wait (up to
        # discovery.timeout) for each to accept connections, so the plan doesn't
        # depend on which nodes happened to be up first.
        timeout = float(self._run_config.discovery.timeout)
        deadline = time.monotonic() + timeout
        for proxy in self._proxies:
            if proxy.local is None:
                try:
                    self._router.wait_ready(proxy.address, max(0.0, deadline - time.monotonic()))
                except TimeoutError:
                    raise RuntimeError(f"worker {proxy.address} not reachable within "
                                       f"discovery.timeout={timeout:.0f}s") from None
        for node, proxy in zip(self._nodes, self._proxies):
            try:
                d = proxy.stub().describe(worker_service_pb2.DescribeRequest(), timeout=120.0)
                node.device         = d.device or node.device
                node.memory_mb      = d.memory_mb or node.memory_mb
                node.compute_gflops = d.compute_gflops
                node.max_memory_mb  = d.max_memory_mb
                node.first_slice    = d.first_slice
                node.tags           = list(node.tags) or list(d.tags)
            except grpc.RpcError as e:   # e.g. a worker older than M2 (UNIMPLEMENTED)
                _LOG.warning("describe failed address=%s error=%s; planning without its capabilities",
                             node.address, e.code() if hasattr(e, "code") else e)
            if plan.measure_network and proxy.local is None:
                try:
                    node.rtt_ms, node.bandwidth_mbps = measure_link(proxy.stub(), plan.probe_bytes)
                except grpc.RpcError as e:
                    _LOG.warning("network probe failed address=%s error=%s", node.address, e)

    def _order_nodes(self) -> None:
        """Slice order: a node declaring ``first_slice`` (e.g. the data owner) goes first;
        the others keep discovery order. When this process hosts slice 0 (it is the data
        owner), it keeps it."""
        if self.local is not None:
            for nd, proxy in zip(self._nodes[1:], self._proxies[1:]):
                if nd.first_slice and proxy.local is None:
                    _LOG.warning("worker %s declares first_slice, but this %s holds the data "
                                 "and slice 0", nd.address, self.executor_name)
            return
        firsts = [i for i, nd in enumerate(self._nodes) if nd.first_slice]
        if len(firsts) > 1:
            raise ValueError("more than one worker declares first_slice: "
                             f"{[self._nodes[i].address for i in firsts]}")
        if firsts and firsts[0] != 0:
            order = [firsts[0]] + [i for i in range(len(self._nodes)) if i != firsts[0]]
            self._nodes   = [self._nodes[i] for i in order]
            self._proxies = [self._proxies[i] for i in order]
            for i, proxy in enumerate(self._proxies):
                proxy.worker_index = i

    def _measure_links(self) -> None:
        """Worker i measures its link to worker i+1 (the activations' path)."""
        plan = self._run_config.planning
        for node, nxt, proxy in zip(self._nodes, self._nodes[1:], self._proxies):
            try:
                r = proxy.stub().probe_link(worker_service_pb2.LinkProbeRequest(
                    target=nxt.address, n_bytes=plan.probe_bytes, repeats=3), timeout=120.0)
            except grpc.RpcError as e:
                r = None
                _LOG.warning("link probe failed %s -> %s error=%s", node.address, nxt.address, e)
            if r is not None and not r.error:
                node.link_rtt_ms, node.link_bandwidth_mbps = r.rtt_ms, r.bandwidth_mbps
            elif r is not None:
                _LOG.warning("link probe failed %s -> %s error=%s", node.address, nxt.address, r.error)
        if self._nodes:
            self._nodes[-1].link_rtt_ms = self._nodes[-1].link_bandwidth_mbps = 0.0

    def _log_plan(self, model_graph, partitions) -> None:
        for i, (nd, p) in enumerate(zip(self._nodes, partitions)):
            params = sum(prm.numel() for j in p.layer_indices
                         for prm in model_graph.nodes[j].module.parameters()) if model_graph.nodes else 0
            _LOG.info(
                "plan slice=%s address=%s device=%s gflops=%.1f mem_cap_mb=%s nodes=%s params=%s",
                i, nd.address, nd.device, nd.compute_gflops, nd.memory_cap_mb(),
                f"{p.layer_indices[0]}..{p.layer_indices[-1]}", params,
            )

    def _make_proxy(self, i: int, node) -> "_WorkerProxy":
        return _WorkerProxy(name=node.node_id, address=node.address, worker_index=i,
                            local=self._local.get(i))

    def _init_slices(self, layers: list, partitions: list, phase: str,
                     checkpoint_paths: dict | None = None) -> None:
        """Send each proxy its ``SliceConfig`` (slice i → proxy i) and wait for every ack."""
        cfg = self._run_config
        n = len(self._proxies)
        checkpoint_paths = checkpoint_paths or {}
        tied = build_tied_configs(
            [[layers[j] for j in p.layer_indices] for p in partitions],
            [proxy.address for proxy in self._proxies])
        init_token = uuid.uuid4().hex
        for i, (proxy, partition) in enumerate(zip(self._proxies, partitions)):
            is_last = (i == n - 1)
            partition_layers = [layers[j] for j in partition.layer_indices]
            hooks = {c: self._hooks[c] for c in (i - 1, i) if c in self._hooks}
            hook_buf = io.BytesIO()
            if hooks:
                torch.save(hooks, hook_buf)
            pred_proto = [
                worker_service_pb2.PredecessorList(indices=list(p))
                for p in (partition.predecessors or [[] for _ in partition_layers])
            ]
            slice_cfg = worker_service_pb2.SliceConfig(
                layers             = _build_layer_configs(partition_layers),
                optimizer          = self._stored_opt_cfg,
                criterion          = self._stored_crit_cfg if is_last else None,
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
                tied               = tied[i],
                init_token         = init_token,
                cut_hooks          = hook_buf.getvalue(),
                n_slices           = n,
            )
            res = self._init_worker(proxy, slice_cfg, phase=phase)
            _LOG.info(
                "worker %s complete worker=%s address=%s hostname=%s message=%s",
                phase,
                i,
                proxy.address,
                res.hostname,
                res.message,
            )

    # ── internal ───────────────────────────────────────────────────────────────

    @property
    def _run_id(self) -> str:
        if hasattr(self._discovery, "run_id"):
            return self._discovery.run_id
        return self._run_config.run_id

    def _shutdown_idle_workers(self) -> None:
        """Send Shutdown RPC to workers that registered but weren't selected."""
        active = {p.address for p in self._proxies}
        idle = [nd for nd in self._discovery.idle_nodes() if nd.address not in active]
        if not idle:
            return
        _LOG.info("shutting down idle workers count=%s addresses=%s", len(idle), [nd.address for nd in idle])
        for nd in idle:
            try:
                stub = worker_service_pb2_grpc.WorkerServiceStub(self._router.channel(nd.address))
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

    def _epoch_stats(self, epoch: int) -> list:
        """One ``get_stats`` per worker after an epoch (``None`` where it failed). It
        returns, and resets, the worker's communication counters and profiling stats."""
        out = []
        for proxy in self._proxies:
            try:
                out.append(proxy.stub().get_stats(worker_service_pb2.GetStatsRequest(
                    run_id    = self._run_id,
                    epoch     = epoch,
                    verbosity = self._run_config.profile.verbosity,
                ), timeout=30.0))
            except Exception as e:
                out.append(None)
                _LOG.warning("get_stats failed worker=%s address=%s error=%s", proxy.name, proxy.address, e)
        return out

    def _add_worker_comm(self, stats: list) -> None:
        """Bytes each slice sent across its cuts, and its compute time, into the meter."""
        n = len(stats)
        for i, resp in enumerate(stats):
            if resp is None:
                continue
            if i < n - 1:
                self._meter.add_fwd(i, resp.bytes_fwd, resp.raw_bytes_fwd, resp.msgs_fwd)
            if i > 0:
                self._meter.add_bwd(i - 1, resp.bytes_bwd, resp.raw_bytes_bwd, resp.msgs_bwd)
            self._meter.add_compute(i, resp.compute_ms / 1e3)

    def _collect_worker_stats(self, epoch: int) -> list:
        """Pull profiling stats from all workers after an epoch (rows for metrics.jsonl)."""
        return self._worker_stats_rows(self._epoch_stats(epoch), epoch)

    def _worker_stats_rows(self, stats: list, epoch: int) -> list:
        results = []
        verbosity = self._run_config.profile.verbosity
        for proxy, resp in zip(self._proxies, stats):
            if resp is None:
                continue
            try:
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

    def _save_checkpoint_layout(self, epoch: int, run_dir: str) -> None:
        path = os.path.join(run_dir, "checkpoint_layouts.json")
        payload = {"epochs": {}}
        if os.path.exists(path):
            try:
                with open(path) as handle:
                    payload = json.load(handle)
            except Exception as e:
                _LOG.warning("checkpoint layout load failed path=%s error=%s", path, e)
                payload = {"epochs": {}}

        payload.setdefault("epochs", {})
        payload["epochs"][str(epoch)] = [
            {
                "worker_index": i,
                "layer_indices": list(indices),
            }
            for i, indices in enumerate(self._stored_partition_indices)
        ]

        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)
        _LOG.info("checkpoint layout saved path=%s epoch=%s", path, epoch)

    def _send_label(self, batch_id: int, label: torch.Tensor) -> None:
        """Labels go to the last slice: over the network (counted), or in-process when the
        data owner hosts it (u_shaped)."""
        last = self._proxies[-1]
        if last.local is not None:
            last.local.submit_label(batch_id, label)
            return
        self._meter.add_label(tensor_nbytes(label))
        with tracer.span("torchslicer.coordinator.label_send", batch_id=batch_id, transport="grpc"):
            last.stub().forward(worker_service_pb2.ForwardRequest(
                batch_id=batch_id,
                label=_serialize_tensor(label),
            ))

    def _send_input_to_first_worker(self, batch_id: int, tensor: torch.Tensor, aux: dict) -> None:
        """Send the forward input tensor to worker0, using TCP when available and no aux inputs."""
        if self._proxies[0].local is not None:
            self._proxies[0].local.submit_input(batch_id, tensor, aux)
            return
        self._meter.add_input(tensor_nbytes(tensor) + sum(tensor_nbytes(v) for v in aux.values()))
        tcp_client = self._proxies[0]._tensor_client
        if tcp_client is not None and not aux and self._router.is_direct(self._proxies[0].address):
            with tracer.span("torchslicer.coordinator.input_send", batch_id=batch_id, transport="tcp"):
                tcp_client.send_tensor(b"F", batch_id, tensor)
        else:
            with tracer.span("torchslicer.coordinator.input_send", batch_id=batch_id, transport="grpc"):
                self._proxies[0].stub().forward(worker_service_pb2.ForwardRequest(
                    batch_id=batch_id,
                    input=_serialize_tensor(tensor),
                    aux_inputs={k: _serialize_tensor(v) for k, v in aux.items()},
                ))

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
                self._send_label(mbid, micro_labels[m])
                self._send_input_to_first_worker(mbid, micro_mains[m], micro_aux[m])
        else:
            self._send_label(batch_id, labels)
            self._send_input_to_first_worker(batch_id, main, aux)
        self._meter.batches += 1
