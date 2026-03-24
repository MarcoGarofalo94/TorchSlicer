__version__ = "0.2.0"
__author__ = 'Marco Garofalo'

from .core.slicer import Slicer
from .core.split_layer import SplitLayer
from .core.model_graph import ModelGraph, LayerNode, _FlattenWrapper, _AddWrapper
from .strategies.base import BaseSplitter, Partition
from .strategies.uniform import UniformSplitter
from .strategies.registry import register as register_strategy, get as _get_strategy
from .executors.base import BaseExecutor
from .executors.local import LocalExecutor
from .executors.distributed import DistributedExecutor
from .transport.base import BaseTransport
from .topology.base import BaseTopology
from .discovery import BaseDiscovery, NodeInfo, AnnounceResult
from .discovery import CoordinatorDiscovery, announce_to_coordinator, StaticDiscovery
from .config import (
    RunConfig, TrainingConfig, PipelineConfig, DiscoveryConfig,
    CheckpointConfig, LoggingConfig, ProfileConfig,
)
from .monitor import TrainingCallback, RunLogger
from .adapters.hf import (
    HFAdapter, wrap_hf,
    AuxInputStage, register_hf_architecture,
    BlockStage, CausalLMHeadStage, MoEBlockStage, SimpleEmbedStage,
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
        raise ImportError(
            "peft is required for peft_unwrap(). "
            "Install it with: pip install peft"
        )
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
        optimizer,
        criterion,
        epochs: int = 1,
        devices=None,
        verbose: bool = False,
        mixed_precision: bool = False,
        use_gpipe: bool = False,
        n_micro_batches: int = 4,
        callbacks: list = None,
        run_config=None,
    ) -> list:
        n_micro = n_micro_batches if use_gpipe else 1
        self.executor.setup(
            self.graph,
            self.partitions,
            optimizer,
            criterion,
            mixed_precision=mixed_precision,
            n_micro_batches=n_micro,
            callbacks=callbacks,
            model_name=self._model_name,
            strategy_name=self._strategy_name,
            run_config=run_config,
        )
        history = []
        for epoch in range(1, epochs + 1):
            history.append(self.executor.train_epoch(
                data_loader, epoch=epoch, verbose=verbose, total_epochs=epochs))
        self.executor.teardown()
        return history


def slice(model, strategy="uniform", n=2, executor=None) -> SlicedModel:
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
