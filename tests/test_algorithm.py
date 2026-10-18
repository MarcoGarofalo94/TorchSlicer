"""M8 algorithm API tests.

Correctness requirement: DefaultAlgorithm must reproduce every run bit-for-bit
against no-algorithm runs. LocalLossAlgorithm must train and converge (losses
decrease). SplitFed variants must aggregate client weights and produce training
losses.
"""
import copy

_UNSET = object()

import pytest
import torch
import torch.nn as nn

import torchslicer as ts
from torchslicer.algorithm import (
    Algorithm, BackwardPolicy, AlgorithmContext,
    DefaultAlgorithm, SplitFedAlgorithm, SplitFedV1Algorithm, LocalLossAlgorithm,
    register_algorithm, get_algorithm, available_algorithms, fedavg,
)
from torchslicer.algorithm.base import BackwardPolicy

# ── shared fixtures ────────────────────────────────────────────────────────────

SGD = {"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}
CE  = {"name": "CrossEntropyLoss", "params": {}}
B, D_IN, D_H = 8, 8, 16


def _model():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(D_IN, D_H), nn.ReLU(),
        nn.Linear(D_H, D_H), nn.ReLU(),
        nn.Linear(D_H, 4),
    )


def _data(n_batches=4):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(n_batches * B, D_IN, generator=g)
    y = torch.randint(0, 4, (n_batches * B,), generator=g)
    return [(x[i:i+B], y[i:i+B]) for i in range(0, n_batches*B, B)]


def _cfg(**kw):
    cfg = ts.RunConfig()
    cfg.training.optimizer = SGD
    cfg.training.criterion = CE
    cfg.logging.enabled = False
    cfg.network.device = "cpu"
    for k, v in kw.items():
        setattr(cfg.training, k, v)
    return cfg


def _run(n_slices=3, cuts=_UNSET, algorithm=None, epochs=2, data=None):
    cfg = _cfg(epochs=epochs)
    resolved_cuts = (["1", "3"] if cuts is _UNSET else cuts)
    sliced = ts.slice(_model(), n=n_slices, cuts=resolved_cuts)
    return sliced.train(_data() if data is None else data,
                        run_config=cfg, algorithm=algorithm)


def _reference_losses(epochs=2):
    """Unsplit PyTorch reference losses."""
    model = _model()
    opt   = torch.optim.SGD(model.parameters(), **SGD["params"])
    batches = _data()
    out = []
    for _ in range(epochs):
        total = 0.0
        for x, y in batches:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        out.append(total / len(batches))
    return out


# ── registry ──────────────────────────────────────────────────────────────────

def test_registry_builtin():
    assert "default" in available_algorithms()
    assert "splitfed" in available_algorithms()
    assert "splitfed_v1" in available_algorithms()
    assert "local_loss" not in available_algorithms()  # needs args, not auto-instantiable


def test_register_custom():
    class MyAlgo(Algorithm):
        pass

    register_algorithm("my_test_algo", MyAlgo)
    assert "my_test_algo" in available_algorithms()
    inst = get_algorithm("my_test_algo")
    assert isinstance(inst, MyAlgo)


def test_register_bad_type():
    with pytest.raises(TypeError):
        register_algorithm("bad", object)


def test_get_unknown():
    with pytest.raises(KeyError, match="unknown algorithm"):
        get_algorithm("nonexistent_xyz")


# ── DefaultAlgorithm bit-exact reproduction ───────────────────────────────────

def test_default_algo_matches_no_algo():
    """DefaultAlgorithm must reproduce the same losses as passing no algorithm."""
    h1 = _run(algorithm=None)
    h2 = _run(algorithm=DefaultAlgorithm())
    assert len(h1) == len(h2)
    for e1, e2 in zip(h1, h2):
        assert abs(e1["loss"] - e2["loss"]) < 1e-9, f"loss mismatch: {e1['loss']} vs {e2['loss']}"


def test_default_algo_matches_reference():
    """DefaultAlgorithm losses must match unsplit PyTorch within 1e-5."""
    ref  = _reference_losses()
    hist = _run(algorithm=DefaultAlgorithm(), n_slices=1, cuts=None)
    for r, h in zip(ref, hist):
        assert abs(r - h["loss"]) < 1e-5, f"vs reference: {r:.6f} vs {h['loss']:.6f}"


# ── BackwardPolicy and base hooks ─────────────────────────────────────────────

def test_default_backward_policy():
    algo = DefaultAlgorithm()
    assert algo.backward_policy(0) is BackwardPolicy.FULL
    assert algo.backward_policy(5) is BackwardPolicy.FULL


def test_default_loss_returns_none():
    algo = DefaultAlgorithm()
    algo.configure(2, nn.CrossEntropyLoss(), None)
    x = torch.randn(4, 4)
    y = torch.randint(0, 4, (4,))
    ctx = AlgorithmContext(round=0, batch_id=0, n_slices=2)
    assert algo.loss(x, y, ctx) is None


def test_client_loss_not_overridden_raises():
    algo = DefaultAlgorithm()
    algo.configure(2, nn.CrossEntropyLoss(), None)
    ctx = AlgorithmContext(round=0, batch_id=0, n_slices=2)
    with pytest.raises(NotImplementedError):
        algo.client_loss(torch.randn(4, D_H), torch.randint(0, 4, (4,)), ctx)


# ── LocalLossAlgorithm ────────────────────────────────────────────────────────

def _aux_head():
    torch.manual_seed(42)
    return nn.Linear(D_H, 4)   # maps cut activation (D_H) → 4 classes


def test_local_loss_backward_policy():
    algo = LocalLossAlgorithm(_aux_head())
    assert algo.backward_policy(0) is BackwardPolicy.LOCAL_LOSS


def test_local_loss_trains():
    """LOCAL_LOSS: losses must be finite and decrease over 4 epochs."""
    algo   = LocalLossAlgorithm(_aux_head(), cut_slice_idx=0)
    hist   = _run(n_slices=3, algorithm=algo, epochs=4)
    losses = [h["loss"] for h in hist]
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), f"non-finite: {losses}"
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_local_loss_client_cut():
    """client_cut() with explicit idx returns it; with None uses placement."""
    algo = LocalLossAlgorithm(_aux_head(), cut_slice_idx=1)
    assert algo.client_cut() == 1

    algo2 = LocalLossAlgorithm(_aux_head())   # no explicit cut
    # configure with a mock placement that has client_slices = [0]
    class _P:
        client_slices = [0]
    algo2.configure(3, nn.CrossEntropyLoss(), _P())
    assert algo2.client_cut() == 0


def test_local_loss_no_server_forward():
    """Slices after the cut must not run (only client slice weights should change)."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(D_IN, D_H), nn.ReLU(),   # slice 0
        nn.Linear(D_H, D_H), nn.ReLU(),    # slice 1
        nn.Linear(D_H, 4),                  # slice 2
    )
    children = list(model.children())
    w0_before = children[0].weight.detach().clone()
    w1_before = children[2].weight.detach().clone()

    sliced = ts.slice(model, n=3, cuts=["1", "3"])
    cfg = _cfg(epochs=1)
    algo = LocalLossAlgorithm(_aux_head(), cut_slice_idx=0)
    sliced.train(_data(), run_config=cfg, algorithm=algo)

    # Children are the same nn.Module objects; their weights update in-place.
    w0_after = children[0].weight.detach()
    w1_after = children[2].weight.detach()

    assert not torch.allclose(w0_before, w0_after), "slice 0 weights unchanged"
    assert torch.allclose(w1_before, w1_after), "slice 1 weights changed (should not)"


# ── fedavg helper ─────────────────────────────────────────────────────────────

def test_fedavg_single():
    state = {0: {"model": {"w": torch.tensor([1.0, 2.0])}, "optimizer": None}}
    avg   = fedavg([state])
    assert torch.allclose(avg[0]["model"]["w"], state[0]["model"]["w"])


def test_fedavg_equal_weights():
    s1 = {0: {"model": {"w": torch.tensor([0.0, 4.0])}, "optimizer": None}}
    s2 = {0: {"model": {"w": torch.tensor([2.0, 0.0])}, "optimizer": None}}
    avg = fedavg([s1, s2])
    assert torch.allclose(avg[0]["model"]["w"], torch.tensor([1.0, 2.0]))


def test_fedavg_preserves_dtype():
    s1 = {0: {"model": {"w": torch.tensor([1, 2], dtype=torch.int32)}, "optimizer": None}}
    s2 = {0: {"model": {"w": torch.tensor([3, 4], dtype=torch.int32)}, "optimizer": None}}
    avg = fedavg([s1, s2])
    assert avg[0]["model"]["w"].dtype == torch.int32


def test_fedavg_empty():
    assert fedavg([]) is None


# ── SplitFedAlgorithm (V2) ────────────────────────────────────────────────────

def _parallel_session(n_clients=2, n_batches=4):
    loaders = [_data(n_batches) for _ in range(n_clients)]
    return ts.ClientSession(loaders, mode="parallel")


def test_splitfed_v2_trains():
    """SplitFedAlgorithm runs without error and produces decreasing loss."""
    cfg = ts.RunConfig()
    cfg.training.optimizer = SGD
    cfg.training.criterion = CE
    cfg.training.epochs = 4
    cfg.logging.enabled = False
    cfg.network.device = "cpu"
    cfg.sl.setting = "vanilla"

    algo = SplitFedAlgorithm()
    session = _parallel_session(n_clients=2)
    sliced = ts.slice(_model(), n=2, cuts=["1"])
    hist = sliced.train(session, run_config=cfg, algorithm=algo)
    losses = [h["loss"] for h in hist]
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_splitfed_v2_on_round_end():
    """on_round_end must return a list of averaged states, all equal."""
    algo = SplitFedAlgorithm()
    s1 = {0: {"model": {"w": torch.tensor([0.0])}, "optimizer": None}}
    s2 = {0: {"model": {"w": torch.tensor([2.0])}, "optimizer": None}}
    out = algo.on_round_end([s1, s2], {}, 0)
    assert out is not None
    assert len(out) == 2
    assert torch.allclose(out[0][0]["model"]["w"], torch.tensor([1.0]))
    assert torch.allclose(out[1][0]["model"]["w"], torch.tensor([1.0]))


# ── SplitFedV1Algorithm ────────────────────────────────────────────────────────

def test_splitfed_v1_needs_server_state():
    assert SplitFedV1Algorithm().needs_per_client_server_state() is True


def test_splitfed_v1_aggregate():
    """aggregate() must average both client and server states."""
    algo = SplitFedV1Algorithm()
    cs1 = {0: {"model": {"w": torch.tensor([0.0])}, "optimizer": None}}
    cs2 = {0: {"model": {"w": torch.tensor([4.0])}, "optimizer": None}}
    ss1 = {1: {"model": {"w": torch.tensor([1.0])}, "optimizer": None}}
    ss2 = {1: {"model": {"w": torch.tensor([3.0])}, "optimizer": None}}
    agg_c, agg_s = algo.aggregate([cs1, cs2], [ss1, ss2], 0)
    assert torch.allclose(agg_c[0]["model"]["w"], torch.tensor([2.0]))
    assert torch.allclose(agg_s[1]["model"]["w"], torch.tensor([2.0]))


def test_splitfed_v1_trains():
    """SplitFedV1Algorithm (per-client server copies) trains and converges."""
    cfg = ts.RunConfig()
    cfg.training.optimizer = SGD
    cfg.training.criterion = CE
    cfg.training.epochs = 4
    cfg.logging.enabled = False
    cfg.network.device = "cpu"
    cfg.sl.setting = "vanilla"

    algo    = SplitFedV1Algorithm()
    session = _parallel_session(n_clients=2)
    sliced  = ts.slice(_model(), n=2, cuts=["1"])
    hist    = sliced.train(session, run_config=cfg, algorithm=algo)
    losses  = [h["loss"] for h in hist]
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


# ── config-driven algorithm ────────────────────────────────────────────────────

def test_config_algorithm_name():
    """sl.algorithm = 'default' in config produces the same result as no algorithm."""
    cfg1 = _cfg(epochs=2)
    cfg2 = _cfg(epochs=2)
    cfg2.sl.algorithm = "default"
    h1 = ts.slice(_model(), n=3, cuts=["1", "3"]).train(_data(), run_config=cfg1)
    h2 = ts.slice(_model(), n=3, cuts=["1", "3"]).train(_data(), run_config=cfg2)
    for e1, e2 in zip(h1, h2):
        assert abs(e1["loss"] - e2["loss"]) < 1e-9


def test_config_unknown_algorithm_raises():
    cfg = _cfg(epochs=1)
    cfg.sl.algorithm = "does_not_exist_xyz"
    with pytest.raises(KeyError, match="unknown algorithm"):
        ts.slice(_model(), n=2).train(_data(), run_config=cfg)


# ── SplitFed against a plain-PyTorch reference ────────────────────────────────

def _splitfed_reference(datas, v1: bool, rounds=2, select=None):
    """SplitFed in plain PyTorch, parallel clients: each client has its own client model
    and optimizer; the server is shared (V2) or one copy per client (V1). At round end
    the client models (and V1's server copies) are averaged and their optimizer states
    start fresh (``fedavg`` drops them)."""
    full = _model()
    client0, server0 = nn.Sequential(*list(full)[:2]), nn.Sequential(*list(full)[2:])
    k = len(datas)
    clients = [copy.deepcopy(client0) for _ in range(k)]
    servers = [copy.deepcopy(server0) for _ in range(k)] if v1 else [server0] * k
    sgd = lambda m: torch.optim.SGD(m.parameters(), **SGD["params"])
    c_opts = [sgd(c) for c in clients]
    s_opts = [sgd(s) for s in servers] if v1 else [sgd(server0)] * k
    out = []
    for r in range(1, rounds + 1):
        part = select(r) if select else list(range(k))
        losses = []
        for b in range(len(datas[0])):
            for i in part:
                x, y = datas[i][b]
                a = clients[i](x)
                a_in = a.detach().requires_grad_(True)
                loss = nn.functional.cross_entropy(servers[i](a_in), y)
                s_opts[i].zero_grad(); loss.backward(); s_opts[i].step()
                c_opts[i].zero_grad(); a.backward(a_in.grad); c_opts[i].step()
                losses.append(loss.item())
        out.append(sum(losses) / len(losses))
        groups = [(clients, c_opts)] + ([(servers, s_opts)] if v1 else [])
        for models, opts in groups:
            with torch.no_grad():       # average the participants, give it to everyone
                for ps in zip(*(m.parameters() for m in models)):
                    avg = torch.stack([ps[i] for i in part]).mean(0)
                    for p in ps:
                        p.copy_(avg)
            for o in opts:
                o.state.clear()
    return out


@pytest.mark.parametrize("algo_cls,v1", [(SplitFedAlgorithm, False), (SplitFedV1Algorithm, True)])
def test_splitfed_matches_reference(algo_cls, v1):
    # Regressions: after aggregation the last client of a round kept its old weights, and
    # every client kept the previous client's momentum.
    g = torch.Generator().manual_seed(5)
    datas = [[(torch.randn(B, D_IN, generator=g), torch.randint(0, 4, (B,), generator=g))
              for _ in range(3)] for _ in range(3)]
    cfg = _cfg(epochs=2)
    cfg.sl.setting = "vanilla"
    session = ts.ClientSession(datas, mode="parallel")
    hist = ts.slice(_model(), n=2, cuts=["1"]).train(session, run_config=cfg, algorithm=algo_cls())
    assert [h["loss"] for h in hist] == pytest.approx(_splitfed_reference(datas, v1), abs=1e-6)


class _Sample(ts.Algorithm):
    """Two of three clients per round, rotating."""
    def select_clients(self, available, round):
        return [available[round % 3], available[(round + 1) % 3]]


class _SampledSplitFed(_Sample, SplitFedAlgorithm):
    pass


def _three_clients():
    g = torch.Generator().manual_seed(5)
    return [[(torch.randn(B, D_IN, generator=g), torch.randint(0, 4, (B,), generator=g))
             for _ in range(3)] for _ in range(3)]


def test_client_sampling_runs_only_the_selected_clients():
    datas = _three_clients()
    cfg = _cfg(epochs=3)
    cfg.sl.setting = "vanilla"
    session = ts.ClientSession(datas, mode="parallel")
    seen = []
    orig = session.batches
    def spy(adapter, epoch, clients=None):
        for k, b in orig(adapter, epoch, clients):
            seen.append((epoch, k))
            yield k, b
    session.batches = spy
    hist = ts.slice(_model(), n=2, cuts=["1"]).train(session, run_config=cfg, algorithm=_Sample())
    for r in (1, 2, 3):
        assert sorted({k for e, k in seen if e == r}) == sorted([r % 3, (r + 1) % 3])
        assert hist[r - 1]["clients"] == [r % 3, (r + 1) % 3]


def test_splitfed_with_sampling_matches_reference():
    datas = _three_clients()
    cfg = _cfg(epochs=3)
    cfg.sl.setting = "vanilla"
    session = ts.ClientSession(datas, mode="parallel")
    hist = ts.slice(_model(), n=2, cuts=["1"]).train(session, run_config=cfg, algorithm=_SampledSplitFed())
    ref = _splitfed_reference(datas, v1=False, rounds=3, select=lambda r: [r % 3, (r + 1) % 3])
    assert [h["loss"] for h in hist] == pytest.approx(ref, abs=1e-6)


def test_bad_client_selection_is_refused():
    class Bad(ts.Algorithm):
        def select_clients(self, available, round):
            return [0, 0]
    cfg = _cfg(epochs=1)
    cfg.sl.setting = "vanilla"
    with pytest.raises(ValueError, match="distinct client indices"):
        ts.slice(_model(), n=2, cuts=["1"]).train(ts.ClientSession(_three_clients(), mode="parallel"),
                                                 run_config=cfg, algorithm=Bad())
