"""Hardware-independent costs in every in-process epoch: memory and FLOPs per slice."""
import pytest
import torch
import torch.nn as nn

import torchslicer as ts
from torchslicer.executors.local import LocalExecutor

B, D, H, K = 8, 16, 32, 4


def _cfg(opt="SGD", **params):
    c = ts.RunConfig()
    c.training.epochs = 2
    c.training.optimizer = {"name": opt, "params": {"lr": 0.05, **params}}
    c.training.criterion = {"name": "CrossEntropyLoss", "params": {}}
    c.logging.enabled = False
    c.network.device = "cpu"
    c.sl.setting = "vanilla"
    return c


def _data(n=5, b=B):
    g = torch.Generator().manual_seed(0)
    return [(torch.randn(b, D, generator=g), torch.randint(0, K, (b,), generator=g)) for _ in range(n)]


def _mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, K))


@pytest.mark.parametrize("opt,params,state", [("SGD", {}, 0), ("SGD", {"momentum": 0.9}, 1), ("AdamW", {}, 2)])
def test_exact_values_on_an_mlp(opt, params, state):
    data = _data(n=5, b=B)
    h = ts.slice(_mlp(), cuts=["1"]).train(data, run_config=_cfg(opt, **params), verbose=False)
    e = h[0]
    w0, w1 = (D * H + H) * 4, (H * K + K) * 4                       # fp32 weights + biases
    act0 = B * D * 4 + B * H * 4                                      # Linear's input, ReLU's output
    act1 = B * H * 4                                                  # Linear's input
    assert e["mem_bytes"] == [w0 * (2 + state) + act0, w1 * (2 + state) + act1]
    fwd = [2 * B * D * H, 2 * B * H * K]                              # 2 FLOPs per multiply-add
    assert e["flops"] == [3 * f * 5 for f in fwd]                     # 5 batches, training = 3 x forward
    assert e["client_mem_bytes"] == e["mem_bytes"][0] and e["client_flops"] == e["flops"][0]
    assert e["costs_batch_size"] == B


def test_flops_follow_the_samples_of_the_epoch():
    data = _data(n=4, b=B) + [(torch.randn(2, D), torch.randint(0, K, (2,)))]   # a short last batch
    e = ts.slice(_mlp(), cuts=["1"]).train(data, run_config=_cfg(), verbose=False)[0]
    assert e["flops"][0] == 3 * 2 * D * H * (4 * B + 2)


def _bn_dropout():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(D, H), nn.BatchNorm1d(H), nn.ReLU(), nn.Dropout(0.3), nn.Linear(H, K))


def test_measuring_them_leaves_training_unchanged(monkeypatch):
    # the measuring forward runs once, before the first step: RNG forked (dropout), buffers
    # restored (BatchNorm running stats), no backward
    def run():
        torch.manual_seed(1)
        model = _bn_dropout()
        losses = [e["loss"] for e in ts.slice(model, cuts=["2"]).train(
            _data(), run_config=_cfg(momentum=0.9), verbose=False)]
        return losses, model.state_dict()          # running stats too (train mode ignores them)
    losses, state = run()
    monkeypatch.setattr(LocalExecutor, "_measure_costs", lambda self, inputs: [])
    losses2, state2 = run()
    assert losses2 == losses
    assert state2.keys() == state.keys() and all(torch.equal(state[k], state2[k]) for k in state)


def test_a_client_session_reports_one_clients_memory():
    data = [_data(n=3), _data(n=2)]
    session = ts.ClientSession(data, mode="sequential")
    e = ts.slice(_mlp(), cuts=["1"]).train(session, run_config=_cfg(), verbose=False)[0]
    assert e["client_mem_bytes"] == e["mem_bytes"][0]                # each client holds slice 0
    assert e["client_flops"] == 3 * 2 * D * H * 5 * B                 # every client's batches


def test_an_in_place_first_layer_is_measured():
    # the measuring input is cloned as in SplitLayer.forward (a leaf can't be changed in place)
    torch.manual_seed(0)
    model = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(D, H), nn.ReLU(inplace=True), nn.Linear(H, K))
    e = ts.slice(model, cuts=["1"]).train(_data(), run_config=_cfg(), verbose=False)[0]
    assert e["flops"][1] > 0 and len(e["mem_bytes"]) == 2


def test_simulated_time_is_deprecated():
    cfg = _cfg()
    cfg.sl.link_bandwidth_mbps = 10.0
    with pytest.warns(DeprecationWarning, match="simulated time"):
        e = ts.slice(_mlp(), cuts=["1"]).train(_data(), run_config=cfg, verbose=False)[0]
    assert "sim_comm_s" in e                       # still there until it's removed
