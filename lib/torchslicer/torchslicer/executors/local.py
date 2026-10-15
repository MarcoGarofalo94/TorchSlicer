import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseExecutor
from ..core.split_layer import SplitLayer
from ..core.tied import find_tied_parameters, non_owned_names, optimizer_parameters
from ..monitor import tracer
from ..monitor.profiler import WorkerProfiler
from ..monitor.process_logger import get_logger, configure as configure_process_logging
from ..monitor.run_logger import RunLogger
from ..monitor.callback import TrainingCallback
from ..sl.hooks import CutContext, resolve_cut_hooks, payload_nbytes, tensor_nbytes
from ..sl.roles import Placement, resolve_setting
from ..sl.metrics import CommMeter
from ..sl.clients import ClientSession, SplitLayerClientState

_LOG = get_logger("torchslicer.local")

def _detach_payload(payload):
    if isinstance(payload, torch.Tensor):
        return payload.detach()
    return [t.detach() for t in payload]


def check_session_ties(tied_groups, placement) -> None:
    """Client states are swapped per client; a parameter tied across a client and a
    server slice would then be shared by every client, so it is refused."""
    for g in tied_groups:
        sides = {placement.is_client(p) for p in g.members}
        if len(sides) > 1:
            raise ValueError("multi-client sessions don't support parameters tied across a "
                             "client slice and a server slice (e.g. tied embeddings with "
                             "u_shaped); untie them or change the cuts")


def _unpack_inputs(inputs):
    """Unpack batch inputs into ``(main_tensor, aux_dict)``.

    Supports three DataLoader conventions:

    * ``torch.Tensor``           → ``(inputs, {})``
    * ``dict``                   → ``(inputs["input_ids"], rest)`` — key
      ``"input_ids"`` is the main tensor; everything else is aux.  If
      ``"input_ids"`` is absent, the first key is treated as main.
    * ``(Tensor, dict)``         → ``(inputs[0], inputs[1])``
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


class LocalExecutor(BaseExecutor):
    def __init__(self):
        self.split_layers: list[SplitLayer] = []
        self.criterion = None
        self._n_micro  = 1

        self._run_logger: RunLogger              = None
        self._callbacks:  list[TrainingCallback] = []
        self._profilers:  list[WorkerProfiler]   = []  # one per partition
        self._hooks: dict = {}
        self._placement: Placement = None
        self._meter: CommMeter = None
        self._link = (0.0, 0.0)
        self._sync = None

    def setup(
        self,
        model_graph,
        partitions,
        optimizer_cfg: dict,
        criterion_cfg: dict,
        mixed_precision: bool = False,
        n_micro_batches: int = 1,
        run_config=None,
        callbacks: list = None,
        model_name: str = "unknown",
        strategy_name: str = "unknown",
        cut_hooks=None,
    ) -> None:
        """``cut_hooks``: ``ts.slice(cut_hooks=)`` (see ``torchslicer.sl.hooks``); the
        config's ``sl`` section adds hooks, the setting and the link model."""
        layers = model_graph.get_layers()
        n = len(partitions)
        self.split_layers = []
        self._callbacks   = list(callbacks or [])

        partitions_layers = [[layers[j] for j in p.layer_indices] for p in partitions]
        # A tied parameter is one shared tensor here, so its .grad sums every use;
        # only its owner (lowest partition, which steps last) may update it.
        self.tied_groups = find_tied_parameters(partitions_layers)
        for i, partition in enumerate(partitions):
            is_last = (i == n - 1)
            partition_layers = partitions_layers[i]
            preds = partition.predecessors if partition.predecessors else None
            sl = SplitLayer(partition_layers, is_last=is_last, predecessors=preds,
                            mixed_precision=mixed_precision)
            sl.train()
            params = optimizer_parameters(sl, exclude=non_owned_names(i, self.tied_groups))
            opt = getattr(optim, optimizer_cfg["name"])(params, **optimizer_cfg["params"]) if params else None
            sl.set_optimizer(opt)
            self.split_layers.append(sl)

        self.criterion = getattr(nn, criterion_cfg["name"])(**criterion_cfg["params"])
        self._n_micro  = max(1, n_micro_batches)

        # Split-learning setting: where each slice would run, hooks on the cuts, metrics
        cfg = run_config
        sl_cfg = cfg.sl if cfg else None
        self._placement = Placement(resolve_setting(sl_cfg.setting if sl_cfg else "", "local"), n)
        self._hooks = resolve_cut_hooks(cut_hooks, n, sl_cfg.cut_hooks if sl_cfg else None)
        self._meter = CommMeter(n)
        self._link = ((sl_cfg.link_bandwidth_mbps, sl_cfg.link_latency_ms) if sl_cfg else (0.0, 0.0))

        # Initialise profilers (one per partition) and RunLogger
        if cfg:
            configure_process_logging(cfg.logging.level)
        verbosity = cfg.profile.verbosity if cfg else 0
        memory    = cfg.profile.memory    if cfg else False

        # For local execution the "device" is whatever the split_layers are on
        device = next(iter(self.split_layers[0].parameters()), torch.tensor(0)).device \
            if self.split_layers else None

        self._profilers = [
            WorkerProfiler(verbosity=verbosity, memory=memory, device=device)
            for _ in range(n)
        ]
        self._sync = ((lambda: torch.cuda.synchronize(device))
                      if device is not None and device.type == "cuda" else None)

        if cfg and cfg.logging.enabled:
            run_dir = os.path.join(cfg.logging.dir, cfg.run_id)
            self._run_logger = RunLogger(run_id=cfg.run_id, run_dir=run_dir)
            layer_names = [type(l).__name__ for l in layers]
            self._run_logger.record_executor("local")
            self._run_logger.record_config(cfg)
            self._run_logger.record_model(model_name, layer_names,
                                          node_names=model_graph.names(), expand=model_graph.expand)
            self._run_logger.record_split(partitions, layer_names, strategy_name)

        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb.on_train_begin(
                    run_id=cfg.run_id if cfg else "local",
                    config={},
                )
            except Exception as e:
                _LOG.warning("callback on_train_begin failed: %s", e)

    def train_epoch(self, data_loader, epoch: int = 0, verbose: bool = False,
                    total_epochs: int = 0) -> dict:
        total_loss = 0.0
        n_batches  = 0
        n_total    = len(data_loader)
        session    = data_loader if isinstance(data_loader, ClientSession) else None
        if session is not None:
            clients = self._placement.client_slices
            if not clients:
                raise ValueError("a multi-client session needs client slices: set sl.setting "
                                 "to vanilla or u_shaped (pipeline hosts none on the data owner)")
            check_session_ties(self.tied_groups, self._placement)
            adapter = SplitLayerClientState({i: self.split_layers[i] for i in clients})
            batches = session.batches(adapter, epoch)
        else:
            batches = ((None, b) for b in data_loader)
        self._meter.reset()

        for cb in self._callbacks:
            try:
                cb.on_epoch_begin(epoch)
            except Exception as e:
                _LOG.warning("callback on_epoch_begin failed: %s", e)

        epoch_t0 = time.perf_counter()

        with tracer.span("torchslicer.epoch", epoch=epoch) as epoch_span:
            for client, (inputs, labels) in batches:
                batch_id = epoch * n_total + n_batches
                _shape = (str(tuple(inputs.shape)) if isinstance(inputs, torch.Tensor)
                          else "multimodal")

                with tracer.span(
                    "torchslicer.batch",
                    epoch=epoch,
                    batch_id=batch_id,
                    batch_index=n_batches,
                    input_shape=_shape,
                ) as batch_span:

                    for p in self._profilers:
                        p.begin_batch(batch_id)

                    if self._n_micro > 1:
                        loss_val = self._gpipe_batch(inputs, labels, batch_id)
                    else:
                        loss_val = self._standard_batch(inputs, labels, batch_id)

                    for p in self._profilers:
                        p.end_batch()

                    total_loss += loss_val
                    n_batches  += 1
                    if batch_span:
                        batch_span.set_attribute("loss", loss_val)

                if self._run_logger:
                    extra = {"client": client} if client is not None else {}
                    self._run_logger.log(
                        step=batch_id, epoch=epoch, batch=n_batches,
                        loss=round(loss_val, 6), phase="batch", **extra,
                    )

                if verbose:
                    ep_suffix = f"/{total_epochs}" if total_epochs else ""
                    _LOG.info(
                        "epoch progress epoch=%s%s batch=%s/%s loss=%.4f",
                        epoch,
                        ep_suffix,
                        n_batches,
                        n_total,
                        loss_val,
                    )

        epoch_duration_s = time.perf_counter() - epoch_t0
        avg = total_loss / n_batches if n_batches > 0 else 0.0
        if session is not None:
            session.finish(adapter)
        if verbose:
            suffix = f"/{total_epochs}" if total_epochs else ""
            _LOG.info("epoch complete epoch=%s%s avg_loss=%.4f", epoch, suffix, avg)

        # Build epoch metrics, let callbacks augment
        epoch_metrics = {
            "step":       epoch,
            "epoch":      epoch,
            "loss":       round(avg, 6),
            "duration_s": round(epoch_duration_s, 3),
            "phase":      "epoch",
            **self._meter.summary(self._placement, *self._link),
        }
        for cb in self._callbacks:
            try:
                result = cb.on_epoch_end(epoch, epoch_metrics)
                if isinstance(result, dict):
                    epoch_metrics = result
            except Exception as e:
                _LOG.warning("callback on_epoch_end failed: %s", e)

        if self._run_logger:
            self._run_logger.log(**epoch_metrics)

            # Per-partition profiling entries
            for i, profiler in enumerate(self._profilers):
                if not profiler.is_active():
                    break
                summary = profiler.epoch_summary(epoch)
                if summary.get("n_batches", 0) > 0:
                    self._run_logger.log(
                        step=epoch, epoch=epoch, partition=i,
                        phase="partition_epoch",
                        **{k: v for k, v in summary.items() if k != "epoch"},
                    )
                    # Verbosity=3: raw per-batch records
                    for rec in profiler.batch_records():
                        self._run_logger.log(
                            step=epoch, epoch=epoch, partition=i,
                            phase="partition_batch",
                            **rec,
                        )
                profiler.reset_epoch()

        return epoch_metrics

    # ── cuts: hooks and byte counting ──────────────────────────────────────────

    def _clock(self) -> float:
        if self._sync is not None:
            self._sync()
        return time.perf_counter()

    def _cut_forward(self, cut: int, x, batch_id: int):
        """What slice ``cut + 1`` receives from slice ``cut`` (hooks applied, bytes counted)."""
        raw = tensor_nbytes(x)
        hook = self._hooks.get(cut)
        if hook is None:
            self._meter.add_fwd(cut, raw, raw)
            return x
        ctx = CutContext(cut, "forward", batch_id, len(self.split_layers))
        payload = _detach_payload(hook.encode(x, ctx))
        self._meter.add_fwd(cut, payload_nbytes(payload), raw)
        return hook.decode(payload, ctx)

    def _cut_backward(self, cut: int, g, out, batch_id: int):
        """The gradient slice ``cut`` backpropagates through ``out``, from ``g`` sent by
        slice ``cut + 1``."""
        if g is None:
            return None
        raw = tensor_nbytes(g)
        hook = self._hooks.get(cut)
        if hook is None:
            self._meter.add_bwd(cut, raw, raw)
            return g
        ctx = CutContext(cut, "backward", batch_id, len(self.split_layers))
        payload = _detach_payload(hook.encode_grad(g, ctx))
        self._meter.add_bwd(cut, payload_nbytes(payload), raw)
        g = hook.decode_grad(payload, ctx)
        return g.to(out.dtype) if g.dtype != out.dtype else g

    def _count_io(self, main, aux: dict, labels, msgs: int = 1) -> None:
        """Raw inputs / labels that would leave the data owner (``sl.setting``)."""
        if self._placement.inputs_leave:
            self._meter.add_input(tensor_nbytes(main) + sum(tensor_nbytes(v) for v in aux.values()), msgs)
        if self._placement.labels_leave:
            self._meter.add_label(tensor_nbytes(labels), msgs)
        self._meter.batches += 1

    # ── standard (non-GPipe) batch ─────────────────────────────────────────────

    def _standard_batch(self, inputs, labels, batch_id: int) -> float:
        device = next(self.split_layers[0].parameters()).device
        main, aux = _unpack_inputs(inputs)
        main   = main.to(device)
        labels = labels.to(device)
        aux    = {k: v.to(device) for k, v in aux.items()}
        self._count_io(main, aux, labels)

        # forward
        outputs = []
        x = main
        for i, sl in enumerate(self.split_layers):
            if i > 0:
                x = self._cut_forward(i - 1, x, batch_id)
            with tracer.span(
                "torchslicer.partition.forward",
                partition=i,
                batch_id=batch_id,
                input_shape=str(tuple(x.shape)),
            ) as fwd_span:
                t0 = self._clock()
                with self._profilers[i].phase("forward"):
                    x = sl(x, **aux) if (i == 0 and aux) else sl(x)
                self._meter.add_compute(i, self._clock() - t0)
                if fwd_span:
                    fwd_span.set_attribute("output_shape", str(tuple(x.shape)))
            outputs.append(x)

        t0 = self._clock()
        loss = self.criterion(outputs[-1], labels)

        # backward — pop MoE aux loss BEFORE backward (same graph, would fail after)
        with tracer.span("torchslicer.partition.backward",
                         partition=len(self.split_layers) - 1, batch_id=batch_id):
            with self._profilers[-1].phase("backward"):
                moe = self.split_layers[-1].pop_moe_aux_loss()
                total_loss = loss + moe if moe is not None else loss
                grad = self.split_layers[-1].backward(loss=total_loss)
            with self._profilers[-1].phase("optimizer"):
                self.split_layers[-1].optimize()
        self._meter.add_compute(len(self.split_layers) - 1, self._clock() - t0)

        for i in range(len(self.split_layers) - 2, -1, -1):
            grad = self._cut_backward(i, grad, outputs[i], batch_id)
            t0 = self._clock()
            with tracer.span("torchslicer.partition.backward", partition=i, batch_id=batch_id):
                with self._profilers[i].phase("backward"):
                    moe = self.split_layers[i].pop_moe_aux_loss()
                    if moe is not None:
                        # retain graph so moe.backward() can reuse the same nodes
                        outputs[i].backward(gradient=grad, retain_graph=True)
                        moe.backward()
                        sl_x = self.split_layers[i].x
                        grad = sl_x.grad if sl_x is not None else None
                    else:
                        grad = self.split_layers[i].backward(prev_g=grad, out=outputs[i])
                with self._profilers[i].phase("optimizer"):
                    self.split_layers[i].optimize()
            self._meter.add_compute(i, self._clock() - t0)

        return loss.item()

    # ── GPipe micro-batch batch ────────────────────────────────────────────────

    def _gpipe_batch(self, inputs, labels, batch_id: int) -> float:
        """
        GPipe-style micro-batch pipelining:
          1. Forward all M micro-batches, saving cut-point tensors (x_refs).
          2. Backward all M micro-batches, accumulating gradients (no zero_grad between).
          3. Single optimizer step across all split layers.

        Each micro-batch loss is scaled by 1/M so accumulated gradients match
        the magnitude of a full-batch gradient.
        """
        device = next(self.split_layers[0].parameters()).device
        main, aux = _unpack_inputs(inputs)
        main   = main.to(device)
        labels = labels.to(device)
        aux    = {k: v.to(device) for k, v in aux.items()}

        M    = self._n_micro
        n_sl = len(self.split_layers)
        micro_mains  = main.chunk(M)
        micro_labels = labels.chunk(M)
        self._count_io(main, aux, labels, msgs=len(micro_mains))
        # Chunk aux tensors that share the batch dimension; broadcast scalars/constants
        micro_aux = [
            {k: v.chunk(M)[m] if v.dim() > 0 and v.shape[0] == main.shape[0] else v
             for k, v in aux.items()}
            for m in range(M)
        ]

        # Step 1: forward all micro-batches; collect per-micro MoE aux losses immediately
        # (must pop before backward — they share the same computation graph nodes)
        micro_outs: list[list[torch.Tensor]] = []
        x_refs:     list[list[torch.Tensor]] = []
        micro_moe:  list[list] = []  # micro_moe[m][i] = Optional[Tensor]

        for m in range(M):
            outs: list[torch.Tensor] = []
            refs: list[torch.Tensor] = []
            x = micro_mains[m]
            for i, sl in enumerate(self.split_layers):
                if i > 0:
                    x = self._cut_forward(i - 1, x, batch_id * M + m)
                with tracer.span(
                    "torchslicer.partition.forward",
                    partition=i,
                    batch_id=batch_id * M + m,
                    input_shape=str(tuple(x.shape)),
                ):
                    t0 = self._clock()
                    with self._profilers[i].phase("forward"):
                        x = sl(x, **micro_aux[m]) if (i == 0 and micro_aux[m]) else sl(x)
                    self._meter.add_compute(i, self._clock() - t0)
                refs.append(sl.x)
                outs.append(x)
            micro_outs.append(outs)
            x_refs.append(refs)
            # Pop MoE aux per micro-batch right after forward so graph is still alive
            micro_moe.append([sl.pop_moe_aux_loss() for sl in self.split_layers])

        # Step 2: backward all micro-batches, accumulate gradients
        total_loss = 0.0
        for m in range(M):
            t0 = self._clock()
            loss_m = self.criterion(micro_outs[m][-1], micro_labels[m])
            total_loss += loss_m.item()
            scaled = loss_m / M

            # Last partition: add MoE aux (scaled by 1/M) to activation loss
            moe_last = micro_moe[m][-1]
            if moe_last is not None:
                scaled = scaled + moe_last / M

            with tracer.span("torchslicer.partition.backward",
                             partition=n_sl - 1, batch_id=batch_id * M + m):
                with self._profilers[-1].phase("backward"):
                    scaled.backward()
            self._meter.add_compute(n_sl - 1, self._clock() - t0)

            for i in range(n_sl - 2, -1, -1):
                g = self._cut_backward(i, x_refs[m][i + 1].grad, micro_outs[m][i], batch_id * M + m)
                t0 = self._clock()
                with tracer.span("torchslicer.partition.backward",
                                 partition=i, batch_id=batch_id * M + m):
                    with self._profilers[i].phase("backward"):
                        moe_i = micro_moe[m][i]
                        if moe_i is not None:
                            micro_outs[m][i].backward(g, retain_graph=True)
                            (moe_i / M).backward()
                        else:
                            micro_outs[m][i].backward(g)
                self._meter.add_compute(i, self._clock() - t0)

        # Step 3: optimizer
        for i, sl in enumerate(self.split_layers):
            t0 = self._clock()
            with self._profilers[i].phase("optimizer"):
                sl.optimize()
            self._meter.add_compute(i, self._clock() - t0)

        return total_loss / M

    def teardown(self) -> None:
        log_history = self._run_logger.log_history if self._run_logger else []
        for cb in self._callbacks:
            try:
                cb.on_train_end(log_history)
            except Exception as e:
                _LOG.warning("callback on_train_end failed: %s", e)

        if self._run_logger:
            self._run_logger.flush()

        self.split_layers = []
        self.criterion    = None
        self._profilers   = []
