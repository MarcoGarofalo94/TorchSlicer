import pytest
import torch.nn as nn
import torchslicer as ts
from torchslicer.core.model_graph import ModelGraph
from torchslicer.strategies.uniform import UniformSplitter
from torchslicer.strategies.param_balanced import ParameterBalancedSplitter
from torchslicer.strategies.explicit import ExplicitSplitter
from torchslicer.strategies.base import Partition


def _linear_graph(n):
    return ModelGraph.from_sequential(nn.Sequential(*[nn.Linear(4, 4) for _ in range(n)]))


def _split(n_layers, n_parts):
    graph = _linear_graph(n_layers)
    return graph, UniformSplitter().split(graph, n_parts)


# ── UniformSplitter ────────────────────────────────────────────────────────────

def test_split_returns_correct_partition_count():
    _, parts = _split(4, 2)
    assert len(parts) == 2


def test_split_covers_all_layers():
    for n_layers in [3, 4, 5, 7]:
        for n_parts in [2, 3]:
            _, parts = _split(n_layers, n_parts)
            covered = sorted(idx for p in parts for idx in p.layer_indices)
            assert covered == list(range(n_layers))


def test_split_no_duplicate_indices():
    _, parts = _split(6, 3)
    all_indices = [idx for p in parts for idx in p.layer_indices]
    assert len(all_indices) == len(set(all_indices))


def test_split_indices_are_ordered_within_partition():
    _, parts = _split(6, 2)
    for p in parts:
        assert p.layer_indices == sorted(p.layer_indices)


def test_split_partitions_are_contiguous():
    _, parts = _split(6, 3)
    prev_end = -1
    for p in sorted(parts, key=lambda x: x.layer_indices[0]):
        assert p.layer_indices[0] == prev_end + 1
        prev_end = p.layer_indices[-1]


def test_split_fewer_layers_than_partitions():
    # 2 layers, 4 requested partitions -> only 2 partitions created
    _, parts = _split(2, 4)
    assert len(parts) == 2


def test_split_n_equals_layers():
    _, parts = _split(4, 4)
    assert len(parts) == 4
    for p in parts:
        assert len(p.layer_indices) == 1


def test_split_partition_indices_start_at_zero():
    _, parts = _split(4, 2)
    assert parts[0].index == 0
    assert parts[1].index == 1


# ── intra_predecessors ─────────────────────────────────────────────────────────

def test_intra_predecessors_sequential_first_partition():
    graph = _linear_graph(4)
    parts = UniformSplitter().split(graph, 2)
    # First layer of partition 0 has no predecessor
    assert parts[0].predecessors[0] == []


def test_intra_predecessors_sequential_chain_within_partition():
    graph = _linear_graph(4)
    parts = UniformSplitter().split(graph, 2)
    # Partition 0 covers [0, 1]: layer 1 has local predecessor 0
    assert parts[0].predecessors[1] == [0]


def test_intra_predecessors_first_layer_of_second_partition():
    graph = _linear_graph(4)
    parts = UniformSplitter().split(graph, 2)
    # First layer of partition 1 gets external input (predecessor is in partition 0)
    assert parts[1].predecessors[0] == []


# ── BaseSplitter.validate ──────────────────────────────────────────────────────

def test_validate_passes_for_valid_split():
    graph = _linear_graph(4)
    splitter = UniformSplitter()
    splitter.validate(graph, splitter.split(graph, 2))  # no exception


def test_validate_raises_for_missing_layers():
    graph = _linear_graph(4)
    bad = [Partition(index=0, layer_indices=[0, 1])]  # missing 2, 3
    with pytest.raises(ValueError, match="missing"):
        UniformSplitter().validate(graph, bad)


# ── ParameterBalancedSplitter ──────────────────────────────────────────────────

def test_param_balanced_covers_all_layers():
    graph = _linear_graph(6)
    parts = ParameterBalancedSplitter().split(graph, 3)
    covered = sorted(idx for p in parts for idx in p.layer_indices)
    assert covered == list(range(6))


def test_param_balanced_partition_count():
    graph = _linear_graph(6)
    parts = ParameterBalancedSplitter().split(graph, 3)
    assert len(parts) == 3


def test_param_balanced_contiguous():
    graph = _linear_graph(6)
    parts = ParameterBalancedSplitter().split(graph, 3)
    prev_end = -1
    for p in sorted(parts, key=lambda x: x.layer_indices[0]):
        assert p.layer_indices[0] == prev_end + 1
        prev_end = p.layer_indices[-1]


def test_param_balanced_no_params_falls_back_to_uniform():
    # ReLU has no parameters — should fall back without error
    graph = ModelGraph.from_sequential(nn.Sequential(*[nn.ReLU() for _ in range(4)]))
    parts = ParameterBalancedSplitter().split(graph, 2)
    assert len(parts) == 2
    covered = sorted(idx for p in parts for idx in p.layer_indices)
    assert covered == list(range(4))


def test_param_balanced_via_string_strategy():
    from torch.utils.data import DataLoader, TensorDataset
    import torch
    net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    X = torch.randn(8, 8)
    y = torch.randint(0, 4, (8,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4)
    history = ts.run(net, loader, n=2, epochs=1, strategy="param_balanced")
    assert history[0]["loss"] < float("inf")


# ── ExplicitSplitter ───────────────────────────────────────────────────────────

def test_explicit_basic():
    graph = _linear_graph(6)
    parts = ExplicitSplitter(boundaries=[3]).split(graph, 2)
    assert len(parts) == 2
    assert parts[0].layer_indices == list(range(3))
    assert parts[1].layer_indices == list(range(3, 6))


def test_explicit_multiple_boundaries():
    graph = _linear_graph(9)
    parts = ExplicitSplitter(boundaries=[3, 6]).split(graph, 3)
    assert len(parts) == 3
    assert parts[0].layer_indices == [0, 1, 2]
    assert parts[1].layer_indices == [3, 4, 5]
    assert parts[2].layer_indices == [6, 7, 8]


def test_explicit_wrong_n_raises():
    graph = _linear_graph(6)
    with pytest.raises(ValueError, match="2 partitions"):
        ExplicitSplitter(boundaries=[3]).split(graph, 3)


def test_explicit_out_of_range_boundary_raises():
    graph = _linear_graph(6)
    with pytest.raises(ValueError, match="out of range"):
        ExplicitSplitter(boundaries=[10]).split(graph, 2)


def test_explicit_empty_boundaries_raises():
    with pytest.raises(ValueError, match="at least one"):
        ExplicitSplitter(boundaries=[])


def test_explicit_via_ts_slice():
    from torch.utils.data import DataLoader, TensorDataset
    import torch
    net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    X = torch.randn(8, 8)
    y = torch.randint(0, 4, (8,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4)
    splitter = ts.ExplicitSplitter(boundaries=[2])
    history = ts.run(net, loader, n=2, epochs=1, strategy=splitter)
    assert history[0]["loss"] < float("inf")


# ── register_strategy decorator ───────────────────────────────────────────────

def test_register_strategy_decorator():
    @ts.register_strategy("_test_decorator_strategy")
    class _MyStrat(ts.BaseSplitter):
        def split(self, graph, n):
            return UniformSplitter().split(graph, n)

    from torchslicer.strategies.registry import _REGISTRY
    assert "_test_decorator_strategy" in _REGISTRY
    assert _REGISTRY["_test_decorator_strategy"] is _MyStrat


def test_register_strategy_decorator_preserves_class():
    @ts.register_strategy("_test_class_preserved")
    class _MyStrat2(ts.BaseSplitter):
        def split(self, graph, n):
            return UniformSplitter().split(graph, n)

    # Decorator must return the class unchanged
    assert issubclass(_MyStrat2, ts.BaseSplitter)


def test_register_strategy_decorator_usable_by_name():
    @ts.register_strategy("_test_by_name")
    class _MyStrat3(ts.BaseSplitter):
        def split(self, graph, n):
            return UniformSplitter().split(graph, n)

    graph = _linear_graph(4)
    from torchslicer.strategies.registry import get
    splitter = get("_test_by_name")
    parts = splitter.split(graph, 2)
    assert len(parts) == 2


def test_validate_raises_for_cross_partition_skip():
    class _SkipModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(4, 4)
            self.l2 = nn.Linear(4, 4)

        def forward(self, x):
            return self.l1(x) + self.l2(x)

    graph = ModelGraph.from_module(_SkipModel())
    if not graph.is_dag():
        pytest.skip("fx tracing fell back to sequential — DAG not captured")

    # Manually construct partitions that split around the add node's inputs
    # Graph: [0: l1, 1: l2, 2: add]; add.predecessors = [0, 1]
    # Split [0] | [1, 2]: add (node 2) has predecessor 0 across boundary
    bad = [
        Partition(index=0, layer_indices=[0]),
        Partition(index=1, layer_indices=list(range(1, len(graph)))),
    ]
    with pytest.raises(ValueError):
        UniformSplitter().validate(graph, bad)
