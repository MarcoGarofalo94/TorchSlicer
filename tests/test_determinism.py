"""training.deterministic: deterministic GPU kernels in every process of a run.

Without it, cuDNN's convolution backward (atomics) makes two GPU runs from one seed drift
apart; with it the losses repeat exactly. The flag is process-global torch state, so every
test here puts it back.
"""
import os

import pytest
import torch
import torch.nn as nn

import torchslicer as ts
from torchslicer.transport.grpc.worker import worker_service_pb2 as pb

CUDA = torch.cuda.is_available()


@pytest.fixture(autouse=True)
def _restore():
    # set_deterministic() also sets CUBLAS_WORKSPACE_CONFIG: put back, else later tests'
    # subprocesses inherit a bigger cuBLAS workspace (and offload's 100 MiB cap fails)
    env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    state = (torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    yield
    torch.use_deterministic_algorithms(state[0], warn_only=state[1])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = state[2], state[3]
    if env is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = env


def test_env_override(monkeypatch):
    monkeypatch.setenv("DETERMINISTIC", "1")
    assert ts.RunConfig.load().training.deterministic is True


def test_a_worker_turns_it_on_from_its_slice_config():
    # workers started without ts.run (ts.start_worker) learn it from the data owner
    torch.use_deterministic_algorithms(False)
    ws = ts.WorkerServicer()
    ws.init(pb.SliceConfig(deterministic=True, is_last=True), None)
    assert torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.deterministic


def _cnn():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, 10)).cuda()


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_two_gpu_runs_repeat_exactly():
    cfg = ts.RunConfig()
    cfg.training.optimizer = {"name": "SGD", "params": {"lr": 0.05, "momentum": 0.9}}
    cfg.training.criterion = {"name": "CrossEntropyLoss", "params": {}}
    cfg.training.epochs = 3
    cfg.training.deterministic = True
    cfg.logging.enabled = False
    g = torch.Generator().manual_seed(1)
    data = [(torch.randn(32, 3, 32, 32, generator=g).cuda(), torch.randint(0, 10, (32,), generator=g).cuda())
            for _ in range(50)]
    runs = [[h["loss"] for h in ts.slice(_cnn(), n=2).train(data, run_config=cfg, verbose=False)]
            for _ in range(2)]
    assert runs[0] == runs[1]
