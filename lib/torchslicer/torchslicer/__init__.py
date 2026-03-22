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


class SlicedModel:
    def __init__(self, graph, partitions, executor):
        self.graph = graph
        self.partitions = partitions
        self.executor = executor

    def train(self, data_loader, optimizer, criterion, epochs=1, devices=None, verbose=False, mixed_precision=False) -> list:
        self.executor.setup(self.graph, self.partitions, optimizer, criterion, mixed_precision=mixed_precision)
        history = []
        for epoch in range(1, epochs + 1):
            history.append(self.executor.train_epoch(data_loader, epoch=epoch, verbose=verbose))
        self.executor.teardown()
        return history


def slice(model, strategy="uniform", n=2, executor=None) -> SlicedModel:
    graph = ModelGraph.from_module(model)
    splitter = _get_strategy(strategy)
    partitions = splitter.split(graph, n)
    splitter.validate(graph, partitions)
    return SlicedModel(graph, partitions, executor or LocalExecutor())
