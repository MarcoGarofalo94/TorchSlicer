__version__ = "0.4.0"
__author__ = 'Marco Garofalo'

from .core.slicer import Slicer
from .core.split_layer import SplitLayer
from .core.model_graph import ModelGraph, LayerNode, TraceError, _FlattenWrapper, _AddWrapper
from .strategies.base import BaseSplitter, Partition
from .strategies.uniform import UniformSplitter
from .strategies.named import NamedCutSplitter
from .strategies.cost import CostSplitter, ParamSplitter, FlopsSplitter, ProfiledSplitter, balanced_cuts
from .strategies.registry import (
    register as register_strategy, get as _get_strategy, available as available_strategies,
)
from .executors.base import BaseExecutor
from .executors.local import LocalExecutor
from .executors.distributed import DistributedExecutor, WorkerFailureError
from .executors.p2p import P2PExecutor, run_p2p
from .executors.worker import resolve_device, run_worker, start_worker, WorkerHandle, WorkerServicer
from .transport.base import BaseTransport
from .discovery import BaseDiscovery, NodeInfo, AnnounceResult
from .discovery import CoordinatorDiscovery, announce_to_coordinator, StaticDiscovery
from .config import (
    RunConfig, TrainingConfig, PipelineConfig, DiscoveryConfig,
    StartupConfig, TransportConfig,
    CheckpointConfig, LoggingConfig, ProfileConfig, FaultToleranceConfig,
    NetworkConfig, PlanningConfig,
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


def _accepts_nodes(splitter) -> bool:
    import inspect
    try:
        params = inspect.signature(splitter.split).parameters
    except (TypeError, ValueError):
        return False
    return "nodes" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


class SlicedModel:
    def __init__(self, graph, partitions, executor,
                 model_name: str = "unknown", strategy_name: str = "unknown",
                 splitter=None):
        self.graph          = graph
        self.partitions     = partitions
        self.executor       = executor
        self._model_name    = model_name
        self._strategy_name = strategy_name
        self._splitter      = splitter

    def plan(self, nodes) -> list:
        """Partitions for these workers (``NodeInfo`` list, in slice order).

        Distributed executors call this after discovery, when the workers' measured
        capabilities are known, and again in recovery for the survivors. Strategies
        whose ``split`` takes ``nodes=`` use them (speed, memory caps, links); the
        others split as usual (``ts.slice`` already did, for the same count).
        """
        if self._splitter is None:
            if len(nodes) != len(self.partitions):
                raise ValueError("no strategy to re-plan for a different worker count")
            return self.partitions
        n = len(nodes)
        if _accepts_nodes(self._splitter):
            partitions = self._splitter.split(self.graph, n, nodes=list(nodes))
        elif n == len(self.partitions):
            return self.partitions
        else:
            partitions = self._splitter.split(self.graph, n)
        self._splitter.validate(self.graph, partitions)
        return partitions

    def summary(self) -> str:
        """Which nodes each partition holds, with its parameter count."""
        lines = [f"SlicedModel({self._model_name}, strategy={self._strategy_name}, "
                 f"partitions={len(self.partitions)})"]
        seen = set()
        for p in self.partitions:
            params = 0
            for i in p.layer_indices:
                for prm in self.graph.nodes[i].module.parameters():
                    if id(prm) not in seen:
                        seen.add(id(prm))
                        params += prm.numel()
            names = [self.graph.nodes[i].name or str(i) for i in p.layer_indices]
            shown = names if len(names) <= 6 else names[:3] + ["..."] + names[-2:]
            lines.append(f"  {p.index}: {len(names)} nodes, {params:,} params  {shown}")
        return "\n".join(lines)

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
        resolved_optimizer  = optimizer  or (cfg.training.optimizer  if cfg else None)
        resolved_criterion  = criterion  or (cfg.training.criterion  if cfg else None)
        resolved_epochs          = epochs          if epochs          is not None else (cfg.training.epochs          if cfg else 1)
        resolved_mixed_precision = mixed_precision if mixed_precision is not None else (cfg.training.mixed_precision if cfg else False)
        resolved_use_gpipe       = use_gpipe       if use_gpipe       is not None else (cfg.pipeline.use_gpipe       if cfg else False)
        resolved_n_micro         = n_micro_batches if n_micro_batches is not None else (cfg.pipeline.n_micro         if cfg else 4)

        if resolved_optimizer is None:
            raise ValueError("optimizer must be provided directly or via run_config.training.optimizer")
        if resolved_criterion is None:
            raise ValueError("criterion must be provided directly or via run_config.training.criterion")

        n_micro = resolved_n_micro if resolved_use_gpipe else 1
        import inspect
        extra = ({"planner": self.plan}
                 if "planner" in inspect.signature(self.executor.setup).parameters else {})
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
            **extra,
        )
        self.partitions = getattr(self.executor, "partitions", None) or self.partitions
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
                    self.partitions = getattr(self.executor, "partitions", None) or self.partitions
        self.executor.teardown()
        return history


def _merge_expand(a, b):
    if a == "all" or b == "all":
        return "all"
    a = [a] if isinstance(a, str) else list(a or [])
    b = [b] if isinstance(b, str) else list(b or [])
    return a + [x for x in b if x not in a] or None


def trace(model, expand=None, pack=None) -> ModelGraph:
    """Build the layer graph ``ts.slice`` would split, without splitting it.

    ``print(ts.trace(model).summary())`` lists every node's name (use them in
    ``cuts=``) and marks where a cut is allowed. ``expand`` and ``pack`` mean the
    same as in :func:`slice`.
    """
    if pack is not None:
        if expand:
            raise ValueError("expand= has no effect with pack=: packed stages are not traced")
        return ModelGraph.from_stages(list(pack(model)))
    return ModelGraph.from_module(model, expand=expand)


def slice(model, strategy="uniform", n=None, executor=None, pack=None, *,
          cuts=None, expand=None, example_input=None) -> SlicedModel:
    """Slice a model into ``n`` partitions for distributed training.

    Args:
        model:    Any ``nn.Module``.
        strategy: Splitting strategy name, class, or instance. Built in:
                  ``"uniform"`` (equal layer count, the default), ``"params"``
                  (balance parameter count), ``"flops"`` (balance forward FLOPs,
                  needs ``example_input``), ``"profiled"`` (balance measured
                  forward+backward time, needs ``example_input``). Custom ones:
                  subclass ``ts.BaseSplitter`` and optionally
                  ``ts.register_strategy(name, cls)``.
        n:        Number of partitions (default 2, or ``len(cuts) + 1``).
        executor: Executor to use. Defaults to ``LocalExecutor``.
        pack:     Optional callable ``(model) -> list[nn.Module]``.
                  When provided, the function is called to produce a flat
                  list of stages; torch.fx tracing is skipped entirely.
                  Use this for models with cross-partition residuals, MoE
                  blocks, or any architecture the default tracer cannot
                  handle (``ts.TraceError`` says so). Packed stages are treated
                  as authoritative units for splitting and fault-tolerance
                  recovery. See docs/tutorials/pack.md. Example::

                      def pack_qwen(model):
                          return [
                              ts.BlockStage(model.model.embed_tokens),
                              *[ts.BlockStage(l) for l in model.model.layers],
                              ts.BlockStage(model.model.norm),
                              ts.BlockStage(model.lm_head),
                          ]

                      sliced = ts.slice(model, n=4, pack=pack_qwen)
        cuts:     Named cuts: cut right after each listed node, e.g.
                  ``cuts=["layer2", "layer3.1"]``. Overrides ``strategy``.
                  Node names come from ``ts.trace(model).summary()``; a dotted
                  module name traces into its parent so you can cut inside it.
                  With ``pack=``, use stage indices.
        expand:   Submodules to trace through so their children become nodes
                  (finer cut granularity), e.g. ``["layer1", "layer2"]``, or
                  ``"all"``. Not allowed with ``pack=``.
        example_input: An input batch, for strategies that measure the model
                  (``"flops"``, ``"profiled"``). They run on a copy, so the
                  model's weights, buffers and RNG state are not changed.

    Serialization constraint
    ------------------------
    Slices are shipped to workers via ``torch.save`` / ``torch.load``.
    Every class instantiated inside ``pack`` must be importable on workers
    under the same module path.  Prefer torchslicer library classes
    (``BlockStage``, ``GPT2EmbedStage``, ``CausalLMHeadStage``, etc.) and
    standard PyTorch (``nn.Sequential``) which are always available.
    For custom stage classes, either install them as a package on all
    workers or expose them via ``PYTHONPATH``. The same holds for functions
    the traced ``forward`` calls that ``torch.fx`` records by reference.
    """
    if cuts is not None:
        splitter = NamedCutSplitter(cuts)
        if n is not None and n != len(splitter.cuts) + 1:
            raise ValueError(f"{len(splitter.cuts)} cuts make {len(splitter.cuts) + 1} partitions, not n={n}")
        n = len(splitter.cuts) + 1
        strategy_name = "named"
    else:
        splitter = _get_strategy(strategy)
        strategy_name = strategy if isinstance(strategy, str) else type(splitter).__name__
    n = 2 if n is None else n
    if example_input is not None and hasattr(splitter, "example_input"):
        splitter.example_input = example_input

    if pack is not None:
        graph = trace(model, expand=expand, pack=pack)
    else:
        graph = trace(model, expand=_merge_expand(expand, splitter.expand_for(model)))
    partitions = splitter.split(graph, n)
    splitter.validate(graph, partitions)
    return SlicedModel(
        graph, partitions,
        executor or LocalExecutor(),
        model_name=type(model).__name__,
        strategy_name=strategy_name,
        splitter=splitter,
    )
