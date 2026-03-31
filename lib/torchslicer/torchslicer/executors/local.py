import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseExecutor
from ..core.split_layer import SplitLayer
from ..monitor import tracer
from ..monitor.profiler import WorkerProfiler
from ..monitor.process_logger import get_logger, configure as configure_process_logging
from ..monitor.run_logger import RunLogger
from ..monitor.callback import TrainingCallback
from ..pipeline import BasePipelineSchedule, StandardSchedule, GPipeSchedule

_LOG = get_logger("torchslicer.local")

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
    def __init__(self, schedule: BasePipelineSchedule = None):
        self.split_layers: list[SplitLayer] = []
        self.criterion = None
        self._n_micro  = 1
        self.schedule: BasePipelineSchedule = schedule  # None → resolved in setup()

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
            sl.train()
            trainable = [p for p in sl.parameters() if p.requires_grad]
            params = trainable if trainable else list(sl.parameters())
            opt = getattr(optim, optimizer_cfg["name"])(params, **optimizer_cfg["params"])
            sl.set_optimizer(opt)
            self.split_layers.append(sl)

        self.criterion = getattr(nn, criterion_cfg["name"])(**criterion_cfg["params"])
        self._n_micro  = max(1, n_micro_batches)

        # Resolve schedule: injected > GPipe (n_micro>1) > Standard
        if self.schedule is None:
            self.schedule = (GPipeSchedule(self._n_micro)
                             if self._n_micro > 1 else StandardSchedule())

        # Initialise profilers (one per partition) and RunLogger
        cfg = run_config
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
                _LOG.warning("callback on_train_begin failed: %s", e)

    def train_epoch(self, data_loader, epoch: int = 0, verbose: bool = False,
                    total_epochs: int = 0) -> dict:
        total_loss = 0.0
        n_batches  = 0
        n_total    = len(data_loader)

        for cb in self._callbacks:
            try:
                cb.on_epoch_begin(epoch)
            except Exception as e:
                _LOG.warning("callback on_epoch_begin failed: %s", e)

        epoch_t0 = time.perf_counter()

        with tracer.span("torchslicer.epoch", epoch=epoch) as epoch_span:
            for inputs, labels in data_loader:
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

                    device = next(self.split_layers[0].parameters()).device
                    main, aux = _unpack_inputs(inputs)
                    main   = main.to(device)
                    labels = labels.to(device)
                    aux    = {k: v.to(device) for k, v in aux.items()}

                    loss_val = self.schedule.step(
                        self.split_layers, main, labels, aux,
                        self.criterion, self._profilers, batch_id,
                    )

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

        return {"loss": avg}

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
