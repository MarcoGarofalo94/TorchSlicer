__version__ = "0.13.0"
__author__ = 'Marco Garofalo'

from .core.slicer import Slicer
from .core.split_layer import SplitLayer
import copy as _copy_module
from .core.determinism import set_deterministic
from .core.model_graph import ModelGraph, LayerNode, TraceError, _FlattenWrapper, _AddWrapper
from .core.multicut import _RecvCut, _SendCut
from .strategies.base import BaseSplitter, Partition
from .strategies.uniform import UniformSplitter
from .strategies.named import NamedCutSplitter
from .strategies.cost import (
    CostSplitter, ParamSplitter, FlopsSplitter, ProfiledSplitter, MemorySplitter, balanced_cuts,
    activation_bytes,
)
from .strategies.registry import (
    register as register_strategy, get as _get_strategy, available as available_strategies,
)
from .executors.base import BaseExecutor
from .executors.local import LocalExecutor
from .executors.distributed import DistributedExecutor, WorkerFailureError
from .executors.p2p import P2PExecutor, run_p2p
from .executors.client_nodes import ClientNodesExecutor
from .offload import OffloadExecutor
from .executors.worker import resolve_device, run_worker, start_worker, WorkerHandle, WorkerServicer
from .transport.base import BaseTransport
from .discovery import BaseDiscovery, NodeInfo, AnnounceResult
from .discovery import CoordinatorDiscovery, announce_to_coordinator, StaticDiscovery
from .config import (
    normalize_component,
    RunConfig, TrainingConfig, PipelineConfig, DiscoveryConfig,
    StartupConfig, TransportConfig,
    CheckpointConfig, LoggingConfig, ProfileConfig, FaultToleranceConfig,
    NetworkConfig, PlanningConfig, DeployConfig, SLConfig, OffloadConfig, DataConfig, PrivacyConfig,
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
from .deploy import run
from . import sl
from . import data
from . import privacy
from .sl import (
    CutHook, CutContext, Float16, BFloat16, Quantize, TopK, CircularConv, GaussianNoise, ClipNoise, Recorder, Chain,
    register_hook, ClientSession, RemoteSession, Placement, SplitServer, SplitClient,
)
from . import algorithm as _algorithm_module  # noqa: ensure builtin registrations run
from .algorithm import (
    Algorithm, BackwardPolicy, AlgorithmContext,
    register_algorithm, get_algorithm, available_algorithms,
    DefaultAlgorithm, SplitFedAlgorithm, SplitFedV1Algorithm, LocalLossAlgorithm,
    fedavg,
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
                 splitter=None, cut_hooks=None):
        self.graph          = graph
        self.partitions     = partitions
        self.executor       = executor
        self._model_name    = model_name
        self._strategy_name = strategy_name
        self._splitter      = splitter
        self.cut_hooks      = cut_hooks

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

    def summary(self, params: bool = False, example_input=None, aux: dict | None = None) -> str:
        """Each slice with its nodes and parameter count, and the cut after it.

        A module whose nodes all sit in one slice is shown once, as ``layer1.0 (7 nodes)``.
        ``params=True`` adds each slice's size (parameters + buffers; a tied weight is
        counted once). ``example_input`` adds the shape and size of what crosses each
        cut. Both measure like :meth:`ModelGraph.summary`, on a copy of the model.
        """
        from .core.model_graph import _human_bytes
        sizes = self.graph._param_sizes()
        outs = self.graph._output_sizes(example_input, aux) if example_input is not None else None
        total = sum(s[0] for s in sizes)
        head = (f"SlicedModel({self._model_name}, strategy={self._strategy_name}, "
                f"{len(self.partitions)} slices, {total:,} params")
        if params:
            head += f", {_human_bytes(sum(s[1] for s in sizes))}"
        lines = [head + ")"]
        for p in self.partitions:
            count = sum(sizes[i][0] for i in p.layer_indices)
            n_nodes = len(p.layer_indices)
            line = f"  slice {p.index}: {n_nodes} node{'s' if n_nodes > 1 else ''}, {count:,} params"
            if params:
                line += f", {_human_bytes(sum(sizes[i][1] for i in p.layer_indices))}"
            lines.append(line)
            shown = [i for i in p.layer_indices
                     if not isinstance(self.graph.nodes[i].module, (_SendCut, _RecvCut))]
            lines.extend("      " + row for row in _wrap(self._group_names(shown), 96))
            last = p.layer_indices[-1]
            sent = self.graph.nodes[last].predecessors if isinstance(self.graph.nodes[last].module, _SendCut) else None
            if p.index < len(self.partitions) - 1 and sent is not None:
                names = [self._sent_name(i) for i in sent]
                after = self.graph.nodes[shown[-1]].name if shown else names[0]
                cut = (f"  ── cut {p.index} after {after}: {len(names)} values packed "
                       f"({', '.join(names)})")
            elif p.index < len(self.partitions) - 1:
                cut = f"  ── cut {p.index} after {self.graph.nodes[last].name}"
            elif outs:
                cut = "  ── output"
            else:
                continue
            if outs:
                shape, nbytes = outs[last]
                cut += f": {shape}, {_human_bytes(nbytes)}"
            lines.append(cut + " ──")
        return "\n".join(lines)

    def view(self, example_input=None, aux: dict | None = None, title: str | None = None):
        """An interactive picture of the graph and its cuts (``torchslicer.viz.SliceView``).

        Jupyter shows it inline; ``.save("split.html")`` writes the page. Click a gap to move
        the cuts and read the matching ``ts.slice(...)`` call. ``example_input`` adds what
        each cut sends, measured on a copy like :meth:`summary`. Cut with
        ``multi_tensor=True``, it shows the model's own graph and every candidate gap is clickable.
        """
        from .viz import SliceView
        return SliceView(self, example_input, aux, title)

    def slices(self, copy: bool = False) -> list:
        """The slices as modules (``SplitLayer``), in order: print one, run it with
        ``.run(x)``, count its parameters.

        They hold the model's own modules, so after ``train()`` they are trained (on a
        cluster too, with ``training.sync_back: all``), and running them in train mode
        or with gradients changes the model (BatchNorm statistics, ``.grad``). Use
        ``eval()`` and ``torch.no_grad()`` to only look, or ``copy=True`` for independent
        copies (tied weights stay tied among them).
        """
        layers = self.graph.get_layers()
        if copy:
            layers = _copy_module.deepcopy(layers)           # one memo: sharing kept
        last = len(self.partitions) - 1
        return [SplitLayer([layers[j] for j in p.layer_indices], is_last=(p.index == last),
                           predecessors=p.predecessors)
                for p in self.partitions]

    def _sent_name(self, i: int) -> str:
        """A value sent at a packed cut, named after the node it comes from (through recv)."""
        node = self.graph.nodes[i]
        if isinstance(node.module, _RecvCut):
            send = self.graph.nodes[node.predecessors[0]]
            return self._sent_name(send.predecessors[node.module.index])
        return node.name

    def _group_names(self, indices: list) -> list:
        """Node names of one slice, a module whose nodes are all in it shown once."""
        inside = set(indices)
        members = {}
        for node in self.graph.nodes:
            for scope in node.scope:
                members.setdefault(scope, []).append(node.index)
        shown, done = [], set()
        for i in indices:
            node = self.graph.nodes[i]
            whole = next((s for s in node.scope
                          if len(members[s]) > 1 and set(members[s]) <= inside), None)
            if whole is None:
                shown.append(node.name or str(i))
            elif whole not in done:
                done.add(whole)
                shown.append(f"{whole} ({len(members[whole])} nodes)")
        return shown

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
        algorithm=None,
        eval_data=None,
        eval_every: int = None,
        privacy=None,
    ) -> list:
        """Train the sliced model.

        ``data_loader`` yields ``(inputs, labels)``; a ``ts.ClientSession`` trains
        several clients (data owners) against the shared server slices.
        ``eval_data``: a test loader, evaluated every ``eval_every`` epochs (default
        ``data.eval_every``, else 1); ``test_loss``/``test_accuracy`` go into the history.
        ``privacy``: ``ts.privacy`` attacks / defences (or a list), added to the config's
        ``privacy`` section; attack metrics go into the history as ``leak_*``.

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
        resolved_optimizer = normalize_component(resolved_optimizer, "optimizer")
        resolved_criterion = normalize_component(resolved_criterion, "criterion")
        if cfg is not None and cfg.training.sync_back not in ("all", "hosted", "none"):
            raise ValueError(f"training.sync_back must be all, hosted or none; got {cfg.training.sync_back!r}")

        n_micro = resolved_n_micro if resolved_use_gpipe else 1
        if cfg is not None and cfg.training.deterministic:
            set_deterministic()
        if cfg is not None and cfg.offload.mode not in ("", "none") and type(self.executor) is LocalExecutor:
            self.executor = OffloadExecutor()          # offload.* comes from the config
        import inspect
        accepted = inspect.signature(self.executor.setup).parameters
        extra = {"planner": self.plan} if "planner" in accepted else {}
        if "cut_hooks" in accepted:
            extra["cut_hooks"] = self.cut_hooks
        elif self.cut_hooks:
            raise ValueError(f"{type(self.executor).__name__} does not support cut_hooks")
        if "algorithm" in accepted:
            extra["algorithm"] = algorithm
        if eval_data is not None:
            if "eval_data" not in accepted:
                raise ValueError(f"{type(self.executor).__name__} doesn't support evaluation yet "
                                 "(eval_data / data.eval_every); it runs on the local executor")
            extra["eval_data"] = eval_data
            extra["eval_every"] = (eval_every if eval_every is not None
                                   else (cfg.data.eval_every if cfg else 1))
        if privacy is not None:
            if "privacy" not in accepted and not isinstance(self.executor, LocalExecutor):
                raise ValueError(f"{type(self.executor).__name__} doesn't support privacy attacks or "
                                 "defences yet; they run on the local executor")
            extra["privacy"] = privacy
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
        from .core.determinism import rng_restore, rng_snapshot
        for epoch in range(1, resolved_epochs + 1):
            rng = rng_snapshot(data_loader)      # a retried epoch replays the same batches
            for attempt in range(max_retries + 1):
                if attempt:
                    rng_restore(rng, data_loader)
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


def _wrap(items: list, width: int) -> list:
    """Join ``items`` with commas into lines of at most ``width`` characters."""
    rows, row = [], ""
    for item in items:
        if row and len(row) + 2 + len(item) > width:
            rows.append(row + ",")
            row = item
        else:
            row = f"{row}, {item}" if row else item
    return rows + [row] if row else rows


def _check_traced_graph(graph, splitter, expand, pack):
    """``ts.slice(graph)`` cuts the graph as traced: refuse anything that would need another trace."""
    if expand is not None or pack is not None:
        raise ValueError("ts.slice(graph, ...) cuts the graph as traced: pass expand=/pack= to "
                         "ts.trace(model, ...) instead, or call ts.slice(model, expand=..., pack=...)")
    if graph.expand == "all" or graph.is_packed():
        return
    missing = [m for m in splitter.expand_for(getattr(graph, "model", None)) if m not in graph.expand]
    if missing:
        raise ValueError(f"cuts inside {missing}, which this graph does not expand "
                         f"(expand={graph.expand}): trace it with "
                         f"ts.trace(model, expand={sorted(set(graph.expand) | set(missing))}), "
                         f"or call ts.slice(model, cuts=...), which expands them for you")


def _merge_expand(a, b):
    if a == "all" or b == "all":
        return "all"
    a = [a] if isinstance(a, str) else list(a or [])
    b = [b] if isinstance(b, str) else list(b or [])
    return a + [x for x in b if x not in a] or None


def _multi_tensor_partitions(graph, splitter, n):
    """The strategy cuts anywhere; then cuts that several tensors cross get send/recv nodes."""
    from .core.multicut import nearest_valid_cuts, pack_cuts
    anywhere = _copy_module.copy(graph)
    anywhere.multi_tensor = True
    partitions = splitter.split(anywhere, n)
    flat = [i for p in partitions for i in p.layer_indices]
    if flat != list(range(len(graph))) or any(not p.layer_indices for p in partitions):
        splitter.validate(graph, partitions)          # not contiguous: the usual error
    cuts = [p.layer_indices[-1] for p in partitions[:-1]]
    if not set(cuts) <= set(anywhere.valid_cuts()):   # a strategy that counts nodes (uniform)
        cuts = nearest_valid_cuts(anywhere, cuts)     # cut between a weight and its reader
    packed, cuts = pack_cuts(graph, cuts)
    return packed, BaseSplitter.partitions_from_cuts(packed, cuts)


def trace(model, expand=None, pack=None) -> ModelGraph:
    """Build the layer graph ``ts.slice`` would split, without splitting it.

    ``print(ts.trace(model).summary())`` lists every node's name (use them in
    ``cuts=``) and marks where a cut is allowed. ``expand`` and ``pack`` mean the
    same as in :func:`slice`. ``ts.slice(graph, ...)`` then cuts exactly this graph.
    """
    if pack is not None:
        if expand:
            raise ValueError("expand= has no effect with pack=: packed stages are not traced")
        graph = ModelGraph.from_stages(list(pack(model)))
    else:
        graph = ModelGraph.from_module(model, expand=expand)
    graph.model = model
    return graph


def slice(model, strategy="uniform", n=None, executor=None, pack=None, *,
          cuts=None, expand=None, example_input=None, cut_hooks=None,
          multi_tensor=False) -> SlicedModel:
    """Slice a model into ``n`` partitions for distributed training.

    Args:
        model:    Any ``nn.Module``, or a graph from ``ts.trace(model, expand=...)``:
                  then that graph is cut as it is (not traced again), so the cuts
                  are the ones its ``summary()`` showed. ``expand``/``pack`` are
                  given to ``ts.trace`` in that case, not here.
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
        cut_hooks: What crosses the cuts: a ``ts.CutHook`` (every cut), or
                  ``{cut: hook}`` with cut ``k`` = between slice ``k`` and ``k+1``,
                  e.g. ``{0: ts.Quantize(bits=8)}``. Built in: ``Float16``,
                  ``BFloat16``, ``Quantize``, ``TopK``, ``CircularConv``, ``GaussianNoise``,
                  ``ClipNoise``, ``Recorder``, ``Chain``. See docs/tutorials/sl_settings.md.
        multi_tensor: Allow cuts that several tensors cross (a skip connection, a U-Net
                  level, the model input read by a later slice). Every gap becomes a
                  valid cut, except between a weight and its reader
                  (``graph.multi_tensor_cuts()``); at such a cut the graph gets a ``cut<k>.send`` node that
                  packs the tensors into one, and ``cut<k>.recv<i>`` nodes that unpack
                  them, so executors and transports still move one tensor (see
                  ``torchslicer.core.multicut``). Cuts one tensor crosses are unchanged.

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

    if isinstance(model, ModelGraph):
        graph = model
        _check_traced_graph(graph, splitter, expand, pack)
        model = getattr(graph, "model", None)
    elif pack is not None:
        graph = trace(model, expand=expand, pack=pack)
    else:
        graph = trace(model, expand=_merge_expand(expand, splitter.expand_for(model)))
    if multi_tensor:
        graph, partitions = _multi_tensor_partitions(graph, splitter, n)
    else:
        partitions = splitter.split(graph, n)
    splitter.validate(graph, partitions)
    return SlicedModel(
        graph, partitions,
        executor or LocalExecutor(),
        model_name=type(model).__name__ if model is not None else "ModelGraph",
        strategy_name=strategy_name,
        splitter=splitter,
        cut_hooks=cut_hooks,
    )
