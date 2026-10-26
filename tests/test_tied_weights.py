"""Tied weights (one parameter used by several layers, e.g. token embedding ↔ LM head)
must train exactly like the unsplit model, locally and across workers."""
import socket
from concurrent import futures

import grpc
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torchslicer as ts
from torchslicer.config import RunConfig
from torchslicer.core.tied import find_tied_parameters
from torchslicer.discovery.static import StaticDiscovery
from torchslicer.executors.distributed import DistributedExecutor
from torchslicer.executors.worker import WorkerServicer, _GRPC_OPTS
from torchslicer.transport.grpc.worker import worker_service_pb2_grpc

VOCAB, DIM = 32, 16
ADAM = {"name": "Adam", "params": {"lr": 1e-2}}
CE = {"name": "CrossEntropyLoss", "params": {}}


def _tied_lm():
    torch.manual_seed(0)
    emb = nn.Embedding(VOCAB, DIM)
    head = nn.Linear(DIM, VOCAB, bias=False)
    head.weight = emb.weight
    return nn.Sequential(emb, nn.Linear(DIM, DIM), nn.Tanh(), nn.Linear(DIM, DIM), head)


def _loader():
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, VOCAB, (48,), generator=g)
    return DataLoader(TensorDataset(x, (x + 1) % VOCAB), batch_size=8)


def _reference_losses(epochs=2, model=None):
    model = model if model is not None else _tied_lm()
    opt = torch.optim.Adam(model.parameters(), **ADAM["params"])
    crit = nn.CrossEntropyLoss()
    out = []
    for _ in range(epochs):
        losses = []
        for x, y in _loader():
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        out.append(sum(losses) / len(losses))
    return out


def test_find_tied_parameters_reports_cross_partition_group():
    model = _tied_lm()
    layers = list(model)
    groups = find_tied_parameters([layers[:2], layers[2:4], layers[4:]])
    assert len(groups) == 1
    assert groups[0].members == {0: ["layers.0.weight"], 2: ["layers.0.weight"]}
    assert groups[0].owner == 0 and groups[0].replicas == [2]


def test_find_tied_parameters_ignores_untied_models():
    layers = [nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)]
    assert find_tied_parameters([layers[:1], layers[1:]]) == []


@pytest.mark.parametrize("n_micro", [1, 2])
def test_local_tied_weights_match_unsplit_with_adam(n_micro):
    sliced = ts.slice(_tied_lm(), n=3)
    history = sliced.train(_loader(), ADAM, CE, epochs=2,
                           use_gpipe=n_micro > 1, n_micro_batches=n_micro)
    assert [h["loss"] for h in history] == pytest.approx(_reference_losses(), abs=1e-5)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_worker():
    port = _free_port()
    servicer = WorkerServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=_GRPC_OPTS)
    worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    servicer.set_server(server)
    return server, f"127.0.0.1:{port}"


def test_distributed_tied_weights_match_unsplit_with_adam():
    servers, peers = zip(*[_start_worker() for _ in range(3)])
    try:
        cfg = RunConfig()
        cfg.run_id = "tied-test"
        cfg.discovery.backend = "static"
        cfg.discovery.n_workers = 3
        cfg.training.epochs = 2
        cfg.training.optimizer = ADAM
        cfg.training.criterion = CE
        cfg.logging.enabled = False
        coord = f"127.0.0.1:{_free_port()}"
        executor = DistributedExecutor(discovery=StaticDiscovery(peers=list(peers)),
                                       coordinator_addr=coord, coordinator_bind_addr=coord,
                                       run_config=cfg)
        model = _tied_lm()
        sliced = ts.slice(model, n=3, executor=executor)
        history = sliced.train(_loader(), run_config=cfg)
    finally:
        for s in servers:
            s.stop(0)
    unsplit = _tied_lm()
    assert [h["loss"] for h in history] == pytest.approx(_reference_losses(model=unsplit), abs=1e-5)
    # sync_back: the model got the trained weights back, the tied pair still one tensor
    assert model[0].weight is model[4].weight
    for (name, x), y in zip(model.named_parameters(), unsplit.parameters()):
        assert torch.allclose(x, y, atol=1e-5), name
