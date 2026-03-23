import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseExecutor
from ..core.split_layer import SplitLayer
from ..monitor import tracer
from ..monitor.profiler import WorkerProfiler
from ..monitor.run_logger import RunLogger
from ..monitor.callback import TrainingCallback


class LocalExecutor(BaseExecutor):
    def __init__(self):
        self.split_layers: list[SplitLayer] = []
        self.criterion = None
        self._n_micro  = 1

        self._run_logger: RunLogger              = None
        self._callbacks:  list[TrainingCallback] = []
        self._profilers:  list[WorkerProfiler]   = []  # one per partition

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
    ) -> None:
        layers = model_graph.get_layers()
        n = len(partitions)
        self.split_layers = []
        self._callbacks   = list(callbacks or [])

        for i, partition in enumerate(partitions):
            is_last = (i == n - 1)
            partition_layers = [layers[j] for j in partition.layer_indices]
            preds = partition.predecessors if partition.predecessors else None
            sl = SplitLayer(partition_layers, is_last=is_last, predecessors=preds,
                            mixed_precision=mixed_precision)
            trainable = [p for p in sl.parameters() if p.requires_grad]
            params = trainable if trainable else list(sl.parameters())
            opt = getattr(optim, optimizer_cfg["name"])(params, **optimizer_cfg["params"])
            sl.set_optimizer(opt)
            self.split_layers.append(sl)

        self.criterion = getattr(nn, criterion_cfg["name"])(**criterion_cfg["params"])
        self._n_micro  = max(1, n_micro_batches)

        # Initialise profilers (one per partition) and RunLogger
        cfg = run_config
        verbosity = cfg.profile.verbosity if cfg else 0
        memory    = cfg.profile.memory    if cfg else False

        # For local execution the "device" is whatever the split_layers are on
        device = next(iter(self.split_layers[0].parameters()), torch.tensor(0)).device \
            if self.split_layers else None

        self._profilers = [
            WorkerProfiler(verbosity=verbosity, memory=memory, device=device)
            for _ in range(n)
        ]

        if cfg and cfg.logging.enabled:
            run_dir = os.path.join(cfg.logging.dir, cfg.run_id)
            self._run_logger = RunLogger(run_id=cfg.run_id, run_dir=run_dir)
            layer_names = [type(l).__name__ for l in layers]
            self._run_logger.record_executor("local")
            self._run_logger.record_config(cfg)
            self._run_logger.record_model(model_name, layer_names)
            self._run_logger.record_split(partitions, layer_names, strategy_name)

        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb.on_train_begin(
                    run_id=cfg.run_id if cfg else "local",
                    config={},
                )
            except Exception as e:
                print(f"[callback] on_train_begin error: {e}")

    def train_epoch(self, data_loader, epoch: int = 0, verbose: bool = False) -> dict:
        total_loss = 0.0
        n_batches  = 0
        n_total    = len(data_loader)

        for cb in self._callbacks:
            try:
                cb.on_epoch_begin(epoch)
            except Exception as e:
                print(f"[callback] on_epoch_begin error: {e}")

        epoch_t0 = time.perf_counter()

        with tracer.span("torchslicer.epoch", epoch=epoch) as epoch_span:
            for inputs, labels in data_loader:
                batch_id = epoch * n_total + n_batches

                with tracer.span(
                    "torchslicer.batch",
                    epoch=epoch,
                    batch_id=batch_id,
                    batch_index=n_batches,
                    input_shape=str(tuple(inputs.shape)),
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
                    self._run_logger.log(
                        step=batch_id, epoch=epoch, batch=n_batches,
                        loss=round(loss_val, 6), phase="batch",
                    )

                if verbose:
                    print(f"  [epoch {epoch} | batch {n_batches}/{n_total}] loss={loss_val:.4f}")

        epoch_duration_s = time.perf_counter() - epoch_t0
        avg = total_loss / n_batches if n_batches > 0 else 0.0
        if verbose:
            print(f"[epoch {epoch}] avg_loss={avg:.4f}")

        # Build epoch metrics, let callbacks augment
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

        return {"loss": avg}

    # ── standard (non-GPipe) batch ─────────────────────────────────────────────

    def _standard_batch(self, inputs, labels, batch_id: int) -> float:
        # forward
        outputs = []
        x = inputs
        for i, sl in enumerate(self.split_layers):
            with tracer.span(
                "torchslicer.partition.forward",
                partition=i,
                batch_id=batch_id,
                input_shape=str(tuple(x.shape)),
            ) as fwd_span:
                with self._profilers[i].phase("forward"):
                    x = sl(x)
                if fwd_span:
                    fwd_span.set_attribute("output_shape", str(tuple(x.shape)))
            outputs.append(x)

        loss = self.criterion(outputs[-1], labels)

        # backward
        with tracer.span("torchslicer.partition.backward",
                         partition=len(self.split_layers) - 1, batch_id=batch_id):
            with self._profilers[-1].phase("backward"):
                grad = self.split_layers[-1].backward(loss=loss)
            with self._profilers[-1].phase("optimizer"):
                self.split_layers[-1].optimize()

        for i in range(len(self.split_layers) - 2, -1, -1):
            with tracer.span("torchslicer.partition.backward", partition=i, batch_id=batch_id):
                with self._profilers[i].phase("backward"):
                    grad = self.split_layers[i].backward(prev_g=grad, out=outputs[i])
                with self._profilers[i].phase("optimizer"):
                    self.split_layers[i].optimize()

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
        M = self._n_micro
        n_sl = len(self.split_layers)
        micro_inputs = inputs.chunk(M)
        micro_labels = labels.chunk(M)

        # Step 1: forward all micro-batches
        micro_outs: list[list[torch.Tensor]] = []
        x_refs:     list[list[torch.Tensor]] = []

        for m in range(M):
            outs: list[torch.Tensor] = []
            refs: list[torch.Tensor] = []
            x = micro_inputs[m]
            for i, sl in enumerate(self.split_layers):
                with tracer.span(
                    "torchslicer.partition.forward",
                    partition=i,
                    batch_id=batch_id * M + m,
                    input_shape=str(tuple(x.shape)),
                ):
                    with self._profilers[i].phase("forward"):
                        x = sl(x)
                refs.append(sl.x)
                outs.append(x)
            micro_outs.append(outs)
            x_refs.append(refs)

        # Step 2: backward all micro-batches, accumulate gradients
        total_loss = 0.0
        for m in range(M):
            loss_m = self.criterion(micro_outs[m][-1], micro_labels[m])
            total_loss += loss_m.item()
            scaled = loss_m / M

            with tracer.span("torchslicer.partition.backward",
                             partition=n_sl - 1, batch_id=batch_id * M + m):
                with self._profilers[-1].phase("backward"):
                    scaled.backward()

            for i in range(n_sl - 2, -1, -1):
                with tracer.span("torchslicer.partition.backward",
                                 partition=i, batch_id=batch_id * M + m):
                    with self._profilers[i].phase("backward"):
                        micro_outs[m][i].backward(x_refs[m][i + 1].grad)

        # Step 3: single optimizer step for all split layers
        for i, sl in enumerate(self.split_layers):
            with self._profilers[i].phase("optimizer"):
                sl.optimizer.step()
                sl.optimizer.zero_grad()
            sl.x = None

        return total_loss / M

    def teardown(self) -> None:
        log_history = self._run_logger.log_history if self._run_logger else []
        for cb in self._callbacks:
            try:
                cb.on_train_end(log_history)
            except Exception as e:
                print(f"[callback] on_train_end error: {e}")

        if self._run_logger:
            self._run_logger.flush()

        self.split_layers = []
        self.criterion    = None
        self._profilers   = []
