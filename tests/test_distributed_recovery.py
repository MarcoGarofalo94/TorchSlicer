import pytest
import torch.nn as nn

from torchslicer.core.model_graph import ModelGraph
from torchslicer.discovery.base import AnnounceResult, BaseDiscovery
from torchslicer.executors.distributed import DistributedExecutor


class _DummyDiscovery(BaseDiscovery):
    def announce(self, node_info, coordinator_addr=None):
        return AnnounceResult(run_id="test-run", worker_index=0)

    def register(self, node):
        return None

    def discover(self, expected: int, timeout: float = 60.0, **kwargs):
        return []


class _ResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = nn.Linear(4, 4)
        self.block = nn.Linear(4, 4)
        self.block_out = nn.ReLU()
        self.proj_out = nn.Linear(4, 2)

    def forward(self, x):
        x = self.proj_in(x)
        skip = x
        x = self.block(x)
        x = x + skip
        x = self.block_out(x)
        x = self.proj_out(x)
        return x


def _executor():
    return DistributedExecutor(
        discovery=_DummyDiscovery(),
        coordinator_addr="127.0.0.1:50051",
    )


def test_recovery_partitions_preserve_intra_partition_predecessors_for_dag():
    graph = ModelGraph.from_module(_ResidualModel())
    if not graph.is_dag():
        pytest.skip("fx tracing fell back to sequential — DAG not captured")

    executor = _executor()
    executor._stored_model_graph = graph

    partitions = executor._build_recovery_partitions(2)

    assert [p.layer_indices for p in partitions] == [[0, 1, 2], [3, 4]]
    assert partitions[0].predecessors == [[], [0], [1, 0]]
    assert partitions[1].predecessors == [[], [0]]


def test_recovery_partitions_reject_cross_partition_multi_input_nodes():
    graph = ModelGraph.from_module(_ResidualModel())
    if not graph.is_dag():
        pytest.skip("fx tracing fell back to sequential — DAG not captured")

    executor = _executor()
    executor._stored_model_graph = graph

    with pytest.raises(ValueError, match="multi-input predecessors"):
        executor._build_recovery_partitions(3)


def test_recovery_partitions_keep_packed_stages_atomic():
    class _ResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x):
            return self.proj(x) + x

    graph = ModelGraph.from_stages([
        _ResidualBlock(),
        nn.ReLU(),
        nn.Linear(4, 2),
    ])

    executor = _executor()
    executor._stored_model_graph = graph

    partitions = executor._build_recovery_partitions(2)

    assert graph.is_packed()
    assert [p.layer_indices for p in partitions] == [[0, 1], [2]]
    assert partitions[0].predecessors == [[], [0]]
    assert partitions[1].predecessors == [[]]
