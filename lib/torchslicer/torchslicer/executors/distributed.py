import io
import json
import os
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field

import torch
from torch import nn

from .base import BaseExecutor
from ..monitor import tracer
from ..monitor.run_logger import RunLogger
from ..monitor.callback import TrainingCallback
from ..discovery.base import BaseDiscovery, NodeInfo
from ..config import RunConfig, CheckpointConfig

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
        )
        return coordinator_service_pb2.RegisterResponse(
            ok=ok,
            run_id=discovery.run_id,
            worker_index=worker_index,
            message=msg,
        )


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
        run_config: RunConfig = None,
    ):
        if not _GRPC_AVAILABLE:
            raise ImportError(
                "grpcio is required for DistributedExecutor. "
                "Install it with: pip install grpcio"
            )
        self._discovery       = discovery
        self.coordinator_addr = coordinator_addr
        self._run_config      = run_config or RunConfig()

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

        # Start embedded coordinator gRPC server FIRST so the Register endpoint
        # is available before discover() blocks waiting for registrations.
        self._grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=4), options=_GRPC_OPTS
        )
        coordinator_service_pb2_grpc.add_CoordinatorServiceServicer_to_server(
            _CoordinatorServicer(self), self._grpc_server
        )
        port = self.coordinator_addr.split(":")[-1]
        self._grpc_server.add_insecure_port(f"[::]:{port}")
        self._grpc_server.start()
        print(f"[coordinator] gRPC server started on port {port} (run_id={self._run_id})")

        # Wait for workers to register (CoordinatorDiscovery) or use static list
        print(f"[coordinator] waiting for {n} workers "
              f"via {type(self._discovery).__name__} ...")
        self._nodes = self._discovery.discover(expected=n, timeout=cfg.discovery.timeout)
        print(f"[coordinator] all {n} workers ready: {[nd.address for nd in self._nodes]}")

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

        # Send SliceConfig to each worker
        opt_cfg  = _build_optimizer_config(optimizer_cfg)
        crit_cfg = _build_criterion_config(criterion_cfg)

        for i, proxy in enumerate(self._proxies):
            is_last = (i == n - 1)
            partition_layers = [layers[j] for j in partitions[i].layer_indices]
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
            )
            try:
                res = proxy.stub().init(slice_cfg)
                print(f"[init] {proxy.name}: ok={res.ok}  {res.message}  ({res.hostname})")
            except grpc.RpcError as e:
                print(f"[init] {proxy.name}: ERROR — {e}")

        # Notify callbacks that training is about to begin
        for cb in self._callbacks:
            try:
                cb.on_train_begin(
                    run_id=self._run_id,
                    config=cfg.logging.__dict__ if hasattr(cfg, "logging") else {},
                )
            except Exception as e:
                print(f"[callback] on_train_begin error: {e}")

    def train_epoch(self, data_loader, epoch: int = 0, verbose: bool = False) -> dict:
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
                print(f"[callback] on_epoch_begin error: {e}")

        epoch_t0 = time.perf_counter()

        with tracer.span("torchslicer.epoch", epoch=epoch, n_workers=len(self._proxies)):
            iter_t0 = time.perf_counter()
            for inputs, labels in data_loader:
                data_load_total_ms += (time.perf_counter() - iter_t0) * 1000.0

                batch_id = epoch * n_total + n_batches
                with self._lock:
                    self._batch_losses.clear()
                self._batch_done.clear()

                with tracer.span(
                    "torchslicer.batch",
                    epoch=epoch,
                    batch_id=batch_id,
                    batch_index=n_batches,
                    input_shape=str(tuple(inputs.shape)),
                ) as batch_span:
                    send_t0 = time.perf_counter()
                    self._send_batch(batch_id, inputs, labels)
                    send_total_ms += (time.perf_counter() - send_t0) * 1000.0

                    wait_t0 = time.perf_counter()
                    self._batch_done.wait()
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

                if verbose:
                    print(f"  [epoch {epoch} | batch {n_batches}/{n_total}] loss={loss:.4f}")

                iter_t0 = time.perf_counter()

        epoch_duration_s = time.perf_counter() - epoch_t0
        avg = total_loss / n_batches if n_batches > 0 else 0.0
        if verbose:
            print(f"[epoch {epoch}] avg_loss={avg:.4f}")

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
                print(f"[callback] on_epoch_end error: {e}")

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

    def teardown(self) -> None:
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
                print(f"[teardown] {proxy.name}: shutdown sent "
                      f"(checkpoint={'yes' if ckpt.enabled else 'no'})")
            except Exception as e:
                print(f"[teardown] {proxy.name}: shutdown failed — {e}")

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
                print(f"[callback] on_train_end error: {e}")

        if self._run_logger:
            self._run_logger.flush()

    # ── internal ───────────────────────────────────────────────────────────────

    @property
    def _run_id(self) -> str:
        if hasattr(self._discovery, "run_id"):
            return self._discovery.run_id
        return self._run_config.run_id

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
                print(f"[get_stats] {proxy.name}: {e}")
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
                p = os.path.join(ckpt_dir, run_id, f"worker_{i}_epoch_{epoch}.pt")
                if os.path.exists(p):
                    paths[i] = p
            return paths
        except Exception as e:
            print(f"[resume] could not load run_state from {resume}: {e}")
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
        print(f"[checkpoint] run_state saved → {path}")

    def _send_batch(self, batch_id: int, inputs: torch.Tensor, labels: torch.Tensor):
        M = self._n_micro
        if M > 1:
            micro_inputs = inputs.chunk(M)
            micro_labels = labels.chunk(M)
            for m in range(M):
                mbid = batch_id * M + m
                self._proxies[-1].stub().forward(worker_service_pb2.ForwardRequest(
                    batch_id=mbid,
                    label=_serialize_tensor(micro_labels[m]),
                ))
                self._proxies[0].stub().forward(worker_service_pb2.ForwardRequest(
                    batch_id=mbid,
                    input=_serialize_tensor(micro_inputs[m]),
                ))
        else:
            self._proxies[-1].stub().forward(worker_service_pb2.ForwardRequest(
                batch_id=batch_id,
                label=_serialize_tensor(labels),
            ))
            self._proxies[0].stub().forward(worker_service_pb2.ForwardRequest(
                batch_id=batch_id,
                input=_serialize_tensor(inputs),
            ))
