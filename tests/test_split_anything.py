"""M1 "split anything": tracing (functional ops, fan-out of the input, parameters read
in forward, expand=), clear errors instead of silent fallbacks, stricter partition
validation, named cuts and the cost-balancing strategies.

The invariant everywhere: a sliced model computes, and trains, exactly like the unsplit one.
"""
import itertools
import json
import random
import socket
from concurrent import futures

import grpc
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import torchslicer as ts
from torchslicer import ModelGraph, Partition, TraceError
from torchslicer.core.split_layer import SplitLayer
from torchslicer.strategies.cost import balanced_cuts

SGD = {"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}
CE = {"name": "CrossEntropyLoss", "params": {}}


# ── models ────────────────────────────────────────────────────────────────────

class Block(nn.Module):
    """Residual block: the skip connection lives inside, so it's one node unless expanded."""
    def __init__(self, d):
        super().__init__()
        self.fc1, self.fc2, self.act = nn.Linear(d, d), nn.Linear(d, d), nn.ReLU()

    def forward(self, x):
        return self.act(x + self.fc2(self.act(self.fc1(x))))


class ResMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(8, 16)
        self.stage1 = nn.Sequential(Block(16), Block(16))
        self.stage2 = nn.Sequential(Block(16), Block(16))
        self.head = nn.Linear(16, 3)

    def forward(self, x):
        return self.head(self.stage2(self.stage1(self.stem(x))))


class Functional(nn.Module):
    """Functional ops M0 dropped silently (F.relu) or mangled (view with x.size(0))."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc1 = nn.Linear(4 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        h = F.relu(self.conv(x))
        h = h.view(h.size(0), -1)
        h = torch.tanh(self.fc1(h))
        return self.fc2(h * 2.0)


class FanOut(nn.Module):
    """The model input feeds two nodes, and a parameter is read directly in forward."""
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.bias = nn.Parameter(torch.zeros(8))
        self.b = nn.Linear(16, 8)
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        h = x + self.a(x) + self.bias
        h = self.b(torch.cat([h, torch.relu(h)], dim=1))
        return self.head(h)


class InplaceStart(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.relu, self.fc2 = nn.Linear(8, 8), nn.ReLU(inplace=True), nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.bn, self.drop, self.fc2 = nn.Linear(8, 8), nn.BatchNorm1d(8), nn.Dropout(0.5), nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.drop(self.bn(self.fc1(x))))


MODELS = {"resmlp": ResMLP, "functional": Functional, "fanout": FanOut, "inplace": InplaceStart}


def _build(name):
    torch.manual_seed(0)
    return MODELS[name]()


def _data(name):
    g = torch.Generator().manual_seed(1)
    shape = (32, 3, 8, 8) if name == "functional" else (32, 8)
    return torch.randn(*shape, generator=g), torch.randint(0, 3, (32,), generator=g)


def _loader(name):
    return DataLoader(TensorDataset(*_data(name)), batch_size=8, shuffle=False)


def _reference_losses(name, epochs=2):
    model = _build(name)
    opt = torch.optim.SGD(model.parameters(), **SGD["params"])
    crit = nn.CrossEntropyLoss()
    out = []
    for _ in range(epochs):
        losses = []
        for x, y in _loader(name):
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        out.append(sum(losses) / len(losses))
    return out


def _chain_forward(sliced, x):
    out = x
    for p in sliced.partitions:
        layer = SplitLayer([sliced.graph.nodes[i].module for i in p.layer_indices],
                           predecessors=p.predecessors)
        out = layer(out)
    return out


# ── tracing ───────────────────────────────────────────────────────────────────

def test_simple_models_keep_their_m0_graph():
    """Graphs M0 got right must not change: same nodes, same edges (configs, checkpoints)."""
    torch.manual_seed(0)
    g = ModelGraph.from_module(nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)))
    assert [type(n.module).__name__ for n in g.nodes] == ["Linear", "ReLU", "Linear"]
    assert [n.predecessors for n in g.nodes] == [[], [0], [1]]
    assert g.names() == ["0", "1", "2"]
    g = ModelGraph.from_module(ResMLP())
    assert g.names() == ["stem", "stage1", "stage2", "head"]
    assert g.expand == []


@pytest.mark.parametrize("name", list(MODELS))
def test_every_valid_cut_reproduces_the_unsplit_forward(name):
    model = _build(name).eval()
    x, _ = _data(name)
    with torch.no_grad():
        ref = model(x)
        graph = ts.trace(model)
        assert graph.valid_cuts(), graph.summary()
        for k in graph.valid_cuts():
            sliced = ts.slice(model, cuts=[k])
            assert torch.equal(_chain_forward(sliced, x), ref), f"cut after node {k}"


def test_every_valid_cut_of_a_fully_expanded_resnet_reproduces_the_forward():
    """Cuts right after a residual block: the next block's add reads the cut-point input (-1)."""
    model = _build("resmlp").eval()
    x, _ = _data("resmlp")
    graph = ts.trace(model, expand="all")
    assert len(graph.valid_cuts()) >= 8
    with torch.no_grad():
        ref = model(x)
        for k in graph.valid_cuts():
            sliced = ts.slice(model, cuts=[k], expand="all")
            assert torch.equal(_chain_forward(sliced, x), ref), f"cut after {graph.nodes[k].name}"


def test_residual_add_right_after_a_cut_gets_the_external_input():
    sliced = ts.slice(_build("resmlp"), cuts=["stage1.0.act:1"], expand="all")
    p1 = sliced.partitions[1]
    names = [sliced.graph.nodes[i].name for i in p1.layer_indices]
    add = names.index(next(n for n in names if n.startswith("add")))
    assert -1 in p1.predecessors[add]
    history = sliced.train(_loader("resmlp"), SGD, CE, epochs=2)
    assert [h["loss"] for h in history] == pytest.approx(_reference_losses("resmlp"), abs=1e-5)


@pytest.mark.parametrize("name", list(MODELS))
def test_split_training_matches_unsplit(name):
    history = ts.slice(_build(name), strategy="params", n=2).train(_loader(name), SGD, CE, epochs=2)
    assert [h["loss"] for h in history] == pytest.approx(_reference_losses(name), abs=1e-5)


def test_functional_ops_are_nodes_not_dropped():
    graph = ts.trace(Functional())
    types = [type(n.module).__name__ for n in graph.nodes]
    assert types.count("_FunctionWrapper") >= 4          # relu, size, view, tanh, mul
    # view(h.size(0), -1) keeps its dynamic batch dimension: two inputs, not a constant shape
    view = next(n for n in graph.nodes if n.name == "view")
    assert len(view.predecessors) == 2


def test_input_fan_out_becomes_an_input_node():
    graph = ts.trace(FanOut())
    assert graph.nodes[0].name == "input" and graph.nodes[0].takes_input
    assert not any(n.takes_input for n in graph.nodes[1:])
    assert "bias" in graph.names()


def test_untraceable_model_raises_trace_error_pointing_to_pack():
    class Untraceable(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 4), nn.Linear(4, 2)

        def forward(self, x):
            if x.sum() > 0:            # data-dependent control flow
                x = self.a(x)
            return self.b(x)

    with pytest.raises(TraceError, match="pack="):
        ts.slice(Untraceable(), n=2)


def test_untraceable_child_is_fine_because_children_are_leaves():
    class Gate(nn.Module):
        def forward(self, x):
            return x if x.sum() > 0 else -x

    model = nn.Sequential(nn.Linear(4, 4), Gate(), nn.Linear(4, 2))
    assert len(ts.trace(model)) == 3
    with pytest.raises(TraceError):
        ts.trace(model, expand="all")


def test_several_inputs_or_outputs_raise_trace_error():
    class TwoIn(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 2)

        def forward(self, x, mask):
            return self.a(x * mask)

    class TwoOut(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 2)

        def forward(self, x):
            return self.a(x), x

    with pytest.raises(TraceError, match="second input"):
        ts.trace(TwoIn())
    with pytest.raises(TraceError, match="not a single tensor"):
        ts.trace(TwoOut())


def test_dead_in_place_op_raises():
    class DeadInplace(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)

        def forward(self, x):
            h = self.a(x)
            h.mul_(2)                  # result unused: the dependency is invisible to a graph
            return h

    with pytest.raises(TraceError, match="in-place"):
        ts.trace(DeadInplace())


def test_expand_traces_into_submodules():
    g = ts.trace(ResMLP(), expand=["stage1"])
    assert g.names() == ["stem", "stage1.0", "stage1.1", "stage2", "head"]
    assert g.valid_cuts() == [0, 1, 2, 3]
    deep = ts.trace(ResMLP(), expand=["stage1.0"])        # ancestors are expanded too
    assert "stage1.0.fc1" in deep.names() and "stage1.1" in deep.names()
    assert deep.expand == ["stage1", "stage1.0"]
    with pytest.raises(ValueError, match="no submodule"):
        ts.trace(ResMLP(), expand=["stage9"])


def test_valid_cuts_exclude_skip_connections():
    g = ts.trace(ResMLP(), expand="all")
    names = g.names()
    cut_names = {names[k] for k in g.valid_cuts()}
    assert "stage1.0.act:1" in cut_names            # a block's output: one tensor crosses
    assert "stage1.0.fc1" not in cut_names          # inside the residual: two would cross


def test_module_called_twice_gets_numbered_names():
    names = ts.trace(ResMLP(), expand=["stage1.0"]).names()
    assert "stage1.0.act" in names and "stage1.0.act:1" in names


def test_pack_with_expand_is_rejected():
    with pytest.raises(ValueError, match="pack"):
        ts.slice(ResMLP(), n=2, pack=lambda m: list(m.children()), expand=["stage1"])


# ── validation ────────────────────────────────────────────────────────────────

def _parts(graph, groups):
    return [Partition(i, g, ts.BaseSplitter._intra_predecessors(graph, g)) for i, g in enumerate(groups)]


def test_validate_rejects_non_contiguous_partitions():
    graph = ts.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)))
    with pytest.raises(ValueError, match="contiguous"):
        ts.UniformSplitter().validate(graph, _parts(graph, [[0, 2], [1]]))


def test_validate_rejects_single_input_edge_from_a_non_last_node():
    """M0 accepted this: node 3 would silently get node 2's output instead of node 1's."""
    class Bypass(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b, self.c = nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)

        def forward(self, x):
            h = self.a(x)
            return self.c(h) * self.b(h)

    graph = ts.trace(Bypass())               # a, b, c, mul
    with pytest.raises(ValueError, match="crosses a boundary"):
        ts.UniformSplitter().validate(graph, _parts(graph, [[0, 1], [2, 3]]))


def test_validate_rejects_input_reader_outside_first_partition():
    graph = ts.trace(nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2)))
    graph.nodes[1].takes_input = True
    with pytest.raises(ValueError, match="model input"):
        ts.UniformSplitter().validate(graph, _parts(graph, [[0], [1]]))


# ── named cuts ────────────────────────────────────────────────────────────────

def test_named_cuts_partition_where_asked():
    sliced = ts.slice(ResMLP(), cuts=["stem", "stage1.0"])
    names = sliced.graph.names()
    assert [[names[i] for i in p.layer_indices] for p in sliced.partitions] == \
        [["stem"], ["stage1.0"], ["stage1.1", "stage2", "head"]]
    assert sliced.graph.expand == ["stage1"]


def test_named_cut_on_an_expanded_module_means_after_its_output():
    graph = ts.trace(ResMLP(), expand="all")
    assert graph.nodes[graph.index_of("stage1.0")].name == "stage1.0.act:1"
    assert graph.nodes[graph.index_of("stage1")].name == "stage1.1.act:1"


def test_named_cut_inside_a_skip_connection_explains_why():
    with pytest.raises(ValueError, match="Nearest valid cuts"):
        ts.slice(ResMLP(), cuts=["stage1.0.fc1"])


def test_named_cuts_errors():
    with pytest.raises(KeyError, match="no graph node"):
        ts.slice(ResMLP(), cuts=["nope"])
    with pytest.raises(ValueError, match="order"):
        ts.slice(ResMLP(), cuts=["stage2", "stem"])
    with pytest.raises(ValueError, match="partitions"):
        ts.slice(ResMLP(), cuts=["stem"], n=3)


def test_named_cuts_on_packed_stages_use_indices():
    sliced = ts.slice(ResMLP(), cuts=[1], pack=lambda m: list(m.children()))
    assert [p.layer_indices for p in sliced.partitions] == [[0, 1], [2, 3]]


def test_named_cuts_train_like_unsplit():
    history = ts.slice(_build("resmlp"), cuts=["stage1.1", "stage2.0"]).train(
        _loader("resmlp"), SGD, CE, epochs=2)
    assert [h["loss"] for h in history] == pytest.approx(_reference_losses("resmlp"), abs=1e-5)


# ── cost strategies ───────────────────────────────────────────────────────────

def _brute_force(costs, valid, n):
    best = None
    for cuts in itertools.combinations(valid, n - 1):
        bounds = [-1, *cuts, len(costs) - 1]
        worst = max(sum(costs[bounds[i] + 1:bounds[i + 1] + 1]) for i in range(n))
        best = worst if best is None else min(best, worst)
    return best


def test_balanced_cuts_is_optimal():
    rng = random.Random(0)
    for _ in range(200):
        size = rng.randint(2, 9)
        costs = [rng.randint(0, 20) for _ in range(size)]
        valid = sorted(rng.sample(range(size - 1), rng.randint(1, size - 1)))
        n = rng.randint(1, len(valid) + 1)
        cuts = balanced_cuts(costs, valid, n)
        assert len(cuts) == n - 1 and set(cuts) <= set(valid)
        bounds = [-1, *cuts, size - 1]
        worst = max(sum(costs[bounds[i] + 1:bounds[i + 1] + 1]) for i in range(n))
        assert worst == _brute_force(costs, valid, n)


def test_too_many_partitions_for_cost_strategy():
    with pytest.raises(ValueError, match="valid cut points"):
        ts.slice(ResMLP(), strategy="params", n=5)


def test_param_strategy_balances_parameters():
    model = nn.Sequential(nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 10), nn.ReLU(),
                          nn.Linear(10, 10), nn.Linear(10, 10))
    sliced = ts.slice(model, strategy="params", n=2)
    assert [p.layer_indices for p in sliced.partitions] == [[0], [1, 2, 3, 4, 5]]


def test_param_strategy_counts_tied_parameters_once():
    emb, head = nn.Embedding(50, 8), nn.Linear(8, 50, bias=False)
    head.weight = emb.weight
    costs = ts.ParamSplitter().node_costs(ts.trace(nn.Sequential(emb, nn.Linear(8, 8), head)))
    assert costs == [400, 72, 0]


def test_flops_strategy_needs_and_uses_example_input():
    with pytest.raises(ValueError, match="example_input"):
        ts.slice(ResMLP(), strategy="flops", n=2)
    splitter = ts.FlopsSplitter()
    sliced = ts.slice(ResMLP(), strategy=splitter, n=2, example_input=torch.randn(4, 8),
                      expand=["stage1", "stage2"])
    assert splitter.last_costs[0] == 2 * 4 * 8 * 16       # stem: 2·B·in·out
    assert len(sliced.partitions) == 2


@pytest.mark.parametrize("strategy", ["flops", "profiled"])
def test_measuring_strategies_leave_model_and_rng_untouched(strategy):
    torch.manual_seed(0)
    model = BNNet().train()
    x = torch.randn(16, 8)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    ts.slice(model, strategy=strategy, n=2, example_input=x)
    assert torch.equal(torch.get_rng_state(), rng)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("strategy", ["params", "flops", "profiled"])
def test_cost_strategies_train_like_unsplit(strategy):
    x, _ = _data("resmlp")
    history = ts.slice(_build("resmlp"), strategy=strategy, n=3, example_input=x[:8],
                       expand="all").train(_loader("resmlp"), SGD, CE, epochs=2)
    assert [h["loss"] for h in history] == pytest.approx(_reference_losses("resmlp"), abs=1e-5)


def test_custom_strategy_from_the_tutorial():
    class LastLayerAlone(ts.BaseSplitter):
        """Server keeps only the head: cut at the last valid point."""
        def split(self, model_graph, n_partitions):
            return self.partitions_from_cuts(model_graph, [model_graph.valid_cuts()[-1]])

    ts.register_strategy("last-layer-alone", LastLayerAlone)
    sliced = ts.slice(ResMLP(), strategy="last-layer-alone")
    assert [p.layer_indices for p in sliced.partitions] == [[0, 1, 2], [3]]
    assert "last-layer-alone" in ts.available_strategies()


def test_inplace_first_layer_of_a_slice_trains():
    """M0 bug: a slice starting with ReLU(inplace=True) failed with an autograd error."""
    history = ts.slice(_build("inplace"), cuts=["fc1"]).train(_loader("inplace"), SGD, CE, epochs=2)
    assert [h["loss"] for h in history] == pytest.approx(_reference_losses("inplace"), abs=1e-5)


def test_manifest_records_node_names_and_expand(tmp_path):
    cfg = ts.RunConfig()
    cfg.run_id = "m1-manifest"
    cfg.logging.enabled = True
    cfg.logging.dir = str(tmp_path)
    ts.slice(_build("resmlp"), cuts=["stage1.0"]).train(_loader("resmlp"), SGD, CE, epochs=1, run_config=cfg)
    manifest = json.loads((tmp_path / "m1-manifest" / "run_manifest.json").read_text())
    assert manifest["model"]["expand"] == ["stage1"]
    assert manifest["model"]["node_names"][:3] == ["stem", "stage1.0", "stage1.1"]
    assert manifest["split"]["strategy"] == "named"


# ── distributed: generic op wrappers survive torch.save → worker ─────────────

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


DIST_CASES = {
    "fanout": dict(strategy="params", n=3),
    "resmlp": dict(cuts=["stage1.0.act:1", "stage2.0"], expand="all"),   # add reads the cut input
}


@pytest.mark.parametrize("name", list(DIST_CASES))
def test_distributed_traced_graph_matches_unsplit(name):
    from torchslicer.config import RunConfig
    from torchslicer.discovery.static import StaticDiscovery
    from torchslicer.executors.distributed import DistributedExecutor
    from torchslicer.executors.worker import WorkerServicer, _GRPC_OPTS
    from torchslicer.transport.grpc.worker import worker_service_pb2_grpc

    servers, peers = [], []
    for _ in range(3):
        port = _free_port()
        servicer = WorkerServicer()
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=_GRPC_OPTS)
        worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
        server.add_insecure_port(f"127.0.0.1:{port}")
        server.start()
        servicer.set_server(server)
        servers.append(server)
        peers.append(f"127.0.0.1:{port}")
    try:
        cfg = RunConfig()
        cfg.run_id = f"m1-{name}"
        cfg.discovery.backend = "static"
        cfg.discovery.n_workers = 3
        cfg.training.epochs = 2
        cfg.training.optimizer = SGD
        cfg.training.criterion = CE
        cfg.logging.enabled = False
        coord = f"127.0.0.1:{_free_port()}"
        executor = DistributedExecutor(discovery=StaticDiscovery(peers=peers),
                                       coordinator_addr=coord, coordinator_bind_addr=coord,
                                       run_config=cfg)
        sliced = ts.slice(_build(name), executor=executor, **DIST_CASES[name])
        history = sliced.train(_loader(name), run_config=cfg)
    finally:
        for s in servers:
            s.stop(0)
    assert [h["loss"] for h in history] == pytest.approx(_reference_losses(name), abs=1e-5)


def test_uniform_gives_the_requested_number_of_partitions():
    """It returned 2 partitions for n=3 over 4 layers (ceil sizes ran out), silently
    leaving a worker unused; cases that already gave n partitions are unchanged."""
    m = nn.Sequential(*[nn.Linear(4, 4) for _ in range(4)])
    assert [p.layer_indices for p in ts.slice(m, n=3).partitions] == [[0, 1], [2], [3]]
    m7 = nn.Sequential(*[nn.Linear(4, 4) for _ in range(7)])
    assert [p.layer_indices for p in ts.slice(m7, n=3).partitions] == [[0, 1, 2], [3, 4, 5], [6]]
    with pytest.raises(ValueError, match="can't make 5 partitions"):
        ts.slice(m, n=5)
