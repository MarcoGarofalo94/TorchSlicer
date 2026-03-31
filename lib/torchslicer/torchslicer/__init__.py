__version__ = "0.2.0"
__author__ = 'Marco Garofalo'

from .core.slicer import Slicer
from .core.split_layer import SplitLayer
from .core.model_graph import ModelGraph, LayerNode, _FlattenWrapper, _AddWrapper
from .strategies.base import BaseSplitter, Partition
from .strategies.uniform import UniformSplitter
from .strategies.param_balanced import ParameterBalancedSplitter
from .strategies.explicit import ExplicitSplitter
from .strategies.registry import register as register_strategy, get as _get_strategy
from .executors.base import BaseExecutor
from .executors.local import LocalExecutor
from .executors.distributed import DistributedExecutor, WorkerFailureError
from .executors.worker import resolve_device, run_worker, WorkerServicer
from .transport.base import BaseTransport
from .discovery import BaseDiscovery, NodeInfo, AnnounceResult
from .discovery import CoordinatorDiscovery, announce_to_coordinator, StaticDiscovery
from .config import (
    RunConfig, TrainingConfig, PipelineConfig, DiscoveryConfig,
    StartupConfig, TransportConfig,
    CheckpointConfig, LoggingConfig, ProfileConfig, FaultToleranceConfig,
    NetworkConfig,
)
from .monitor import TrainingCallback, RunLogger
from .core.stages import (
    AuxInputStage,
    BlockStage,
    CausalLMHeadStage,
    GPT2EmbedStage,
    MoEBlockStage,
    SimpleEmbedStage,
)
from .checkpoint_utils import restore_model_from_run, resolve_checkpoint_epoch
from ._resolver import resolve_optimizer, resolve_criterion
from .pipeline import BasePipelineSchedule, StandardSchedule, GPipeSchedule
from .adapters.hf_packs import (
    hf_pack,
    pack_gpt2,
    pack_bert_seq_cls,
    pack_bert_mlm,
    pack_distilbert_seq_cls,
    pack_llama,
    pack_moe,
)


def peft_unwrap(model) -> "nn.Module":
    """
    Unwrap a HuggingFace PEFT model so that ts.slice() can see its
    internal structure.

    After get_peft_model(), the top-level object is a PeftModel whose only
    direct child is 'base_model' — ts.slice() would treat the whole thing as
    one unsplittable leaf.  peft_unwrap() returns model.base_model.model:
    the original model with LoRA adapters injected in-place, so from_module()
    sees the familiar children (embedding, transformer blocks, head, etc.).

    Workers must have peft installed to deserialise LoRA-modified layers
    (included in the Docker images; install manually with: pip install peft).

    Idempotent on non-PEFT models — safe to call unconditionally.

    Example::

        from peft import LoraConfig, get_peft_model
        import torchslicer as ts

        model    = MyTransformer()
        lora_cfg = LoraConfig(r=8, target_modules=["qkv"])
        model    = get_peft_model(model, lora_cfg)

        sliced = ts.slice(ts.peft_unwrap(model), strategy="uniform", n=2)
        sliced.train(loader, optimizer_cfg, criterion_cfg, epochs=5)
    """
    try:
        from peft import PeftModel
    except ImportError:
        return model
    if isinstance(model, PeftModel):
        return model.base_model.model
    return model


class SlicedModel:
    def __init__(self, graph, partitions, executor,
                 model_name: str = "unknown", strategy_name: str = "unknown"):
        self.graph          = graph
        self.partitions     = partitions
        self.executor       = executor
        self._model_name    = model_name
        self._strategy_name = strategy_name

    def train(
        self,
        data_loader,
        optimizer=None,
        criterion=None,
        *,
        epochs: int = None,
        verbose: bool = False,
        mixed_precision: bool = None,
        use_gpipe: bool = None,
        n_micro_batches: int = None,
        callbacks: list = None,
        run_config=None,
    ) -> list:
        """Train the sliced model.

        When ``run_config`` is supplied all training parameters default to the
        values in the config; explicit kwargs override individual fields::

            # Everything from config
            sliced.train(loader, run_config=cfg)

            # Override just epochs
            sliced.train(loader, epochs=5, run_config=cfg)

            # Pure Python API (no config file)
            sliced.train(loader, optimizer_cfg, criterion_cfg, epochs=10)
        """
        cfg = run_config
        resolved_optimizer  = resolve_optimizer(optimizer  or (cfg.training.optimizer  if cfg else None))
        resolved_criterion  = resolve_criterion(criterion  or (cfg.training.criterion  if cfg else None))
        resolved_epochs          = epochs          if epochs          is not None else (cfg.training.epochs          if cfg else 1)
        resolved_mixed_precision = mixed_precision if mixed_precision is not None else (cfg.training.mixed_precision if cfg else False)
        resolved_use_gpipe       = use_gpipe       if use_gpipe       is not None else (cfg.pipeline.use_gpipe       if cfg else False)
        resolved_n_micro         = n_micro_batches if n_micro_batches is not None else (cfg.pipeline.n_micro         if cfg else 4)

        if resolved_optimizer is None:
            raise ValueError("optimizer must be provided directly or via run_config.training.optimizer")
        if resolved_criterion is None:
            raise ValueError("criterion must be provided directly or via run_config.training.criterion")

        n_micro = resolved_n_micro if resolved_use_gpipe else 1
        self.executor.setup(
            self.graph,
            self.partitions,
            resolved_optimizer,
            resolved_criterion,
            mixed_precision=resolved_mixed_precision,
            n_micro_batches=n_micro,
            callbacks=callbacks,
            model_name=self._model_name,
            strategy_name=self._strategy_name,
            run_config=run_config,
        )
        history = []
        max_retries = getattr(self.executor, '_max_fault_retries', 0)
        for epoch in range(1, resolved_epochs + 1):
            for attempt in range(max_retries + 1):
                try:
                    history.append(self.executor.train_epoch(
                        data_loader, epoch=epoch, verbose=verbose, total_epochs=epochs))
                    break
                except WorkerFailureError as e:
                    if attempt >= max_retries or not hasattr(self.executor, 'recover'):
                        raise
                    print(f"[fault] epoch {epoch} attempt {attempt + 1} failed: {e}")
                    self.executor.recover(epoch - 1)
        self.executor.teardown()
        return history


def slice(model, strategy="uniform", n=2, executor=None, pack=None) -> SlicedModel:
    """Slice a model into ``n`` partitions for distributed training.

    Args:
        model:    Any ``nn.Module``.
        strategy: Splitting strategy name, class, or instance.
        n:        Number of partitions.
        executor: Executor to use. Defaults to ``LocalExecutor``.
        pack:     Optional callable ``(model) -> list[nn.Module]``.
                  When provided, the function is called to produce a flat
                  list of stages; torch.fx tracing is skipped entirely.
                  Use this for models with cross-partition residuals, MoE
                  blocks, or any architecture the default tracer cannot
                  handle. Packed stages are treated as authoritative units for
                  splitting and fault-tolerance recovery. Example::

                      def pack_qwen(model):
                          return [
                              ts.BlockStage(model.model.embed_tokens),
                              *[ts.BlockStage(l) for l in model.model.layers],
                              ts.BlockStage(model.model.norm),
                              ts.BlockStage(model.lm_head),
                          ]

                      sliced = ts.slice(model, n=4, pack=pack_qwen)

    Serialization constraint
    ------------------------
    Slices are shipped to workers via ``torch.save`` / ``torch.load``.
    Every class instantiated inside ``pack`` must be importable on workers
    under the same module path.  Prefer torchslicer library classes
    (``BlockStage``, ``GPT2EmbedStage``, ``CausalLMHeadStage``, etc.) and
    standard PyTorch (``nn.Sequential``) which are always available.
    For custom stage classes, either install them as a package on all
    workers or expose them via ``PYTHONPATH``.
    """
    if pack is not None:
        stages = pack(model)
        graph = ModelGraph.from_stages(stages)
    else:
        graph = ModelGraph.from_module(model)
    splitter = _get_strategy(strategy)
    partitions = splitter.split(graph, n)
    splitter.validate(graph, partitions)
    strategy_name = strategy if isinstance(strategy, str) else type(strategy).__name__
    return SlicedModel(
        graph, partitions,
        executor or LocalExecutor(),
        model_name=type(model).__name__,
        strategy_name=strategy_name,
    )


def run(
    model,
    data_loader,
    n: int = 2,
    *,
    epochs: int = 10,
    optimizer="adam",
    criterion="cross_entropy",
    strategy: str = "uniform",
    pack=None,
    verbose: bool = False,
    mixed_precision: bool = False,
    use_gpipe: bool = False,
    n_micro_batches: int = 4,
    callbacks: list = None,
) -> list:
    """Train a model with split learning in a single call.

    The simplest possible entry point — no config files, no Docker, no
    executor boilerplate.  Everything runs in-process via ``LocalExecutor``.

    Args:
        model:           Any ``nn.Module``.
        data_loader:     PyTorch ``DataLoader`` yielding ``(inputs, labels)`` batches.
        n:               Number of partitions (default 2).
        epochs:          Training epochs (default 10).
        optimizer:       String shorthand (``"adam"``, ``"adamw"``, ``"sgd"``,
                         ``"rmsprop"``) or a full dict
                         ``{"name": "Adam", "params": {"lr": 3e-4}}``.
        criterion:       String shorthand (``"cross_entropy"``, ``"mse"``, ``"bce"``,
                         ``"bce_with_logits"``, ``"nll"``, ``"l1"``) or a full dict.
        strategy:        Partitioning strategy name (default ``"uniform"``).
        pack:            Optional ``(model) -> list[nn.Module]`` for models that
                         torch.fx cannot trace (e.g. HuggingFace LLMs).
        verbose:         Print per-batch and per-epoch loss.
        mixed_precision: Enable bfloat16 autocast.
        use_gpipe:       Enable GPipe micro-batch pipelining.
        n_micro_batches: Number of GPipe micro-batches (used when ``use_gpipe=True``).
        callbacks:       List of ``TrainingCallback`` instances.

    Returns:
        List of per-epoch metric dicts, e.g. ``[{"loss": 0.42}, ...]``.

    Example::

        import torchslicer as ts

        metrics = ts.run(model, train_loader, n=4, epochs=5, optimizer="adamw")
    """
    sliced = slice(model, strategy=strategy, n=n, pack=pack)
    return sliced.train(
        data_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        verbose=verbose,
        mixed_precision=mixed_precision,
        use_gpipe=use_gpipe,
        n_micro_batches=n_micro_batches,
        callbacks=callbacks,
    )


def from_pretrained(
    model_name_or_model,
    n: int = 2,
    *,
    task: str = None,
    pack=None,
    strategy: str = "uniform",
    executor=None,
    **hf_kwargs,
) -> "SlicedModel":
    """Load (or wrap) a HuggingFace model and slice it for split learning.

    Args:
        model_name_or_model:
            Either a HuggingFace model-hub name string (e.g.
            ``"bert-base-uncased"``) **or** an already-loaded ``nn.Module``.
            When a string is supplied the model is loaded via the
            ``transformers`` library (must be installed).
        n:        Number of partitions (default 2).
        task:     HuggingFace task hint used when loading by name.
                  Supported values: ``"classification"`` (→
                  ``AutoModelForSequenceClassification``), ``"generation"``
                  (→ ``AutoModelForCausalLM``), ``"masked_lm"`` / ``"mlm"``
                  (→ ``AutoModelForMaskedLM``).  Defaults to ``AutoModel``.
        pack:     Explicit pack function.  When ``None`` the function is
                  auto-detected from ``model.config.model_type``.  Pass this
                  to override or to handle architectures not in the registry.
        strategy: Partitioning strategy (default ``"uniform"``).
        executor: Executor to use (default ``LocalExecutor``).
        **hf_kwargs:
                  Forwarded to ``from_pretrained()`` when loading by name
                  (e.g. ``num_labels=2``, ``torch_dtype=torch.float16``).

    Returns:
        A ``SlicedModel`` ready for ``.train()``.

    Examples::

        # Load from hub and auto-slice
        sliced = ts.from_pretrained("distilbert-base-uncased", n=4,
                                    task="classification", num_labels=2)
        sliced.train(train_loader, optimizer="adamw", criterion="cross_entropy",
                     epochs=3)

        # Wrap an already-loaded model
        from transformers import GPT2LMHeadModel
        model  = GPT2LMHeadModel.from_pretrained("gpt2")
        sliced = ts.from_pretrained(model, n=4)
    """
    import torch.nn as nn

    # ── Load from hub if a string was given ───────────────────────────────────
    if isinstance(model_name_or_model, str):
        try:
            from transformers import (
                AutoModel,
                AutoModelForCausalLM,
                AutoModelForMaskedLM,
                AutoModelForSequenceClassification,
            )
        except ImportError:
            raise ImportError(
                "transformers is required for ts.from_pretrained(). "
                "Install it with: pip install transformers"
            )
        _task_map = {
            "classification": AutoModelForSequenceClassification,
            "generation":     AutoModelForCausalLM,
            "causal_lm":      AutoModelForCausalLM,
            "masked_lm":      AutoModelForMaskedLM,
            "mlm":            AutoModelForMaskedLM,
        }
        auto_cls = _task_map.get(task, AutoModel) if task else AutoModel
        model = auto_cls.from_pretrained(model_name_or_model, **hf_kwargs)
    elif isinstance(model_name_or_model, nn.Module):
        model = model_name_or_model
        if hf_kwargs:
            raise ValueError(
                "hf_kwargs are only used when loading by model name string. "
                "Pass an already-configured nn.Module without extra kwargs."
            )
    else:
        raise TypeError(
            f"model_name_or_model must be a str or nn.Module, "
            f"got {type(model_name_or_model).__name__}"
        )

    # ── Auto-unwrap PEFT models ───────────────────────────────────────────────
    model = peft_unwrap(model)

    # ── Resolve pack function ─────────────────────────────────────────────────
    if pack is None:
        model_type = getattr(getattr(model, 'config', None), 'model_type', None)
        if model_type is not None:
            try:
                pack = hf_pack(model_type)
            except KeyError:
                pass  # fall back to FX tracing

    return slice(model, strategy=strategy, n=n, executor=executor, pack=pack)


def simulate(
    model,
    data_loader,
    n: int = 2,
    *,
    epochs: int = 10,
    optimizer="adam",
    criterion="cross_entropy",
    devices: list = None,
    strategy: str = "uniform",
    pack=None,
    verbose: bool = False,
    mixed_precision: bool = False,
    use_gpipe: bool = False,
    n_micro_batches: int = 4,
    callbacks: list = None,
) -> list:
    """Train with split learning across N simulated workers on this machine.

    Spawns N worker subprocesses locally (no Docker, no cluster) using
    ``DistributedExecutor`` + ``CoordinatorDiscovery``.  Useful for testing
    distributed code and measuring pipeline overhead without any
    infrastructure setup.

    Behaves identically to real distributed training — workers register via
    gRPC, tensors are serialised across process boundaries, fault-tolerance
    logic is exercised.  The only difference is that all processes share the
    same physical host.

    Args:
        model:           Any ``nn.Module``.
        data_loader:     PyTorch ``DataLoader`` yielding ``(inputs, labels)``.
        n:               Number of simulated workers / partitions (default 2).
        epochs:          Training epochs (default 10).
        optimizer:       String shorthand or dict (same as ``ts.run()``).
        criterion:       String shorthand or dict (same as ``ts.run()``).
        devices:         List of device strings, one per worker (e.g.
                         ``["cuda:0", "cuda:1"]``).  Defaults to ``["auto"] * n``.
        strategy:        Partitioning strategy (default ``"uniform"``).
        pack:            Optional pack function for HuggingFace / custom models.
        verbose:         Print per-batch and per-epoch progress.
        mixed_precision: Enable bfloat16 autocast.
        use_gpipe:       Enable GPipe micro-batch pipelining.
        n_micro_batches: GPipe micro-batch count (used when ``use_gpipe=True``).
        callbacks:       List of ``TrainingCallback`` instances.

    Returns:
        List of per-epoch metric dicts, e.g. ``[{"loss": 0.42}, ...]``.

    Example::

        import torchslicer as ts

        # 4-way split, all on CPU — good for unit-testing distributed logic
        metrics = ts.simulate(model, train_loader, n=4, epochs=3)

        # 2-way split on two GPUs
        metrics = ts.simulate(model, train_loader, n=2,
                              devices=["cuda:0", "cuda:1"])
    """
    import multiprocessing as mp
    from ._simulate import _find_free_port, _run_worker_subprocess

    if devices is None:
        devices = ["auto"] * n

    # ── Allocate ports ────────────────────────────────────────────────────────
    coordinator_port   = _find_free_port()
    worker_ports       = [_find_free_port() for _ in range(n)]
    coordinator_addr   = f"127.0.0.1:{coordinator_port}"
    coordinator_bind   = f"0.0.0.0:{coordinator_port}"

    # ── Spawn worker subprocesses (they retry registration until coordinator
    #    is up — controlled by discovery.registration_max_attempts) ──────────
    ctx   = mp.get_context("spawn")
    procs = []
    for i in range(n):
        device = devices[i] if i < len(devices) else "auto"
        p = ctx.Process(
            target=_run_worker_subprocess,
            args=(worker_ports[i], coordinator_addr, device),
            daemon=True,
        )
        p.start()
        procs.append(p)

    try:
        # ── Build RunConfig for coordinator ───────────────────────────────────
        cfg = RunConfig.create(
            n_workers=n,
            coordinator_address=coordinator_addr,
        )
        cfg.logging.enabled = False   # suppress run-log files in simulation mode

        discovery = CoordinatorDiscovery(run_id=cfg.run_id)
        executor  = DistributedExecutor(
            discovery=discovery,
            coordinator_addr=coordinator_addr,
            coordinator_bind_addr=coordinator_bind,
            run_config=cfg,
        )

        sliced = slice(model, strategy=strategy, n=n, executor=executor, pack=pack)
        return sliced.train(
            data_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            verbose=verbose,
            mixed_precision=mixed_precision,
            use_gpipe=use_gpipe,
            n_micro_batches=n_micro_batches,
            callbacks=callbacks,
        )
    finally:
        for p in procs:
            p.terminate()
            p.join(timeout=5)
