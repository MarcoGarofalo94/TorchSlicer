"""``ts.SplitServer`` / ``ts.SplitClient``: the server slices over gRPC, the client slice and
the data elsewhere. One client trained through the server equals the local executor's vanilla
SL (same weights after training), with and without a cut hook; several clients share the one
server copy."""
import socket

import pytest
import torch
import torch.nn as nn

import torchslicer as ts

pytest.importorskip("grpc")

OPT = {"name": "Adam", "params": {"lr": 0.01}}
CRIT = {"name": "CrossEntropyLoss", "params": {}}


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))


def _data(n=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(8, 6, generator=g), torch.randint(0, 4, (8,), generator=g)) for _ in range(n)]


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _local(data, hook=None, epochs=2):
    m = _model()
    cfg = ts.RunConfig()
    cfg.training.optimizer, cfg.training.criterion = OPT, CRIT
    cfg.training.epochs, cfg.logging.enabled, cfg.sl.setting = epochs, False, "vanilla"
    ts.slice(m, cuts=["1", "3"], cut_hooks={0: hook} if hook else None).train(data, run_config=cfg)
    return m


@pytest.mark.parametrize("hook", [None, "circular"])
def test_one_remote_client_equals_local_vanilla_sl(hook):
    mk = (lambda: ts.CircularConv(ratio=4, seed=2)) if hook else (lambda: None)
    data = _data()
    server = ts.SplitServer(_model(), cuts=["1", "3"], optimizer=OPT, criterion=CRIT, port=_port(),
                            cut_hook=mk()).start()
    try:
        client = ts.SplitClient(_model(), cuts=["1", "3"], address=f"127.0.0.1:{server.port}",
                                optimizer=OPT, cut_hook=mk())
        out = client.fit(data, epochs=2)
        ev = client.evaluate(data)
    finally:
        server.stop()
    assert out["num_examples"] == 64 and server.steps == 8
    assert ev["num_examples"] == 32 and 0 <= ev["accuracy"] <= 1
    ref = list(_local(data, mk()).parameters())
    got = list(client.slice.parameters()) + [p for s in server.slices for p in s.parameters()]
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        assert torch.allclose(a, b, atol=1e-5)


def test_clients_share_the_server_copy():
    server = ts.SplitServer(_model(), cuts=["1"], optimizer=OPT, criterion=CRIT, port=_port()).start()
    try:
        before = [p.clone() for s in server.slices for p in s.parameters()]
        for k in range(3):
            c = ts.SplitClient(_model(), cuts=["1"], address=f"127.0.0.1:{server.port}", optimizer=OPT)
            c.fit(_data(2, seed=10 + k))
            assert c.bytes_sent > 0 and c.bytes_received > 0
    finally:
        server.stop()
    assert server.steps == 6
    assert any(not torch.equal(a, b) for a, b in zip(before, (p for s in server.slices for p in s.parameters())))


@pytest.mark.parametrize("hook", [None, "circular"])
def test_a_cut_several_tensors_cross_trains_like_local_with_multi_tensor(hook):
    from test_multi_tensor import UNet

    def unet():
        torch.manual_seed(0)
        return UNet()

    mk = (lambda: ts.CircularConv(ratio=2, seed=2)) if hook else (lambda: None)
    g = torch.Generator().manual_seed(1)
    data = [(torch.randn(4, 3, 16, 16, generator=g), torch.randint(0, 3, (4, 16, 16), generator=g))
            for _ in range(3)]
    with pytest.raises(ValueError, match="multi_tensor=True"):
        ts.SplitClient(unet(), cuts=["e2"], address="127.0.0.1:1", optimizer=OPT)
    server = ts.SplitServer(unet(), cuts=["e2"], optimizer=OPT, criterion=CRIT, port=_port(),
                            cut_hook=mk(), multi_tensor=True).start()
    try:
        client = ts.SplitClient(unet(), cuts=["e2"], address=f"127.0.0.1:{server.port}",
                                optimizer=OPT, cut_hook=mk(), multi_tensor=True)
        client.fit(data, epochs=2)
    finally:
        server.stop()
    ref = unet()
    cfg = ts.RunConfig()
    cfg.training.optimizer, cfg.training.criterion = OPT, CRIT
    cfg.training.epochs, cfg.logging.enabled, cfg.sl.setting = 2, False, "vanilla"
    ts.slice(ref, cuts=["e2"], multi_tensor=True, cut_hooks={0: mk()} if hook else None).train(
        data, run_config=cfg)
    got = list(client.slice.parameters()) + [p for s in server.slices for p in s.parameters()]
    want = list(ref.parameters())
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert torch.allclose(a, b, atol=1e-5)
