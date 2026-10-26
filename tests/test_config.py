from torchslicer.config import RunConfig


def test_run_config_load_applies_worker_tag_filter_env_override(tmp_path, monkeypatch):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "discovery:\n"
        "  tag_filter:\n"
        "    - yaml-tag\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("WORKER_TAG_FILTER", "env-a, env-b")

    cfg = RunConfig.load(str(config_path))

    assert cfg.discovery.tag_filter == ["env-a", "env-b"]


# ── optimizer / criterion configs: one meaning in the simulator and on a cluster ──

import pytest
import torch
import torch.nn as nn

import torchslicer as ts
from torchslicer.config import normalize_component


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))


def _data():
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(48, 8, generator=g), torch.randint(0, 4, (48,), generator=g)
    return [(x[i:i + 12], y[i:i + 12]) for i in range(0, 48, 12)]


def _losses(topology, optimizer, criterion):
    cfg = RunConfig()
    cfg.training.epochs = 2
    cfg.training.optimizer, cfg.training.criterion = optimizer, criterion
    cfg.discovery.n_workers = 3
    cfg.deploy.topology = topology
    cfg.network.device = "cpu"
    cfg.logging.enabled = False
    return [h["loss"] for h in ts.run(_model, _data, run_config=cfg, verbose=False)]


@pytest.mark.parametrize("topology", ["centralized", "p2p"])
def test_optimizer_without_params_trains_the_same_locally_and_on_workers(topology):
    # Adagrad's own default lr is 0.01; workers used to get lr=0.001 when none was given.
    optimizer, criterion = {"name": "Adagrad"}, {"name": "CrossEntropyLoss"}
    local = _losses("local", optimizer, criterion)
    assert _losses(topology, optimizer, criterion) == pytest.approx(local, abs=1e-6)
    explicit = _losses("local", {"name": "Adagrad", "params": {"lr": 0.01}}, criterion)
    assert explicit == pytest.approx(local, abs=1e-7)


def test_normalize_component_fills_params_and_refuses_mistakes():
    assert normalize_component({"name": "CrossEntropyLoss"}, "criterion") == \
        {"name": "CrossEntropyLoss", "params": {}}
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['parms'\]"):
        normalize_component({"name": "SGD", "parms": {"lr": 0.1}}, "optimizer")
    with pytest.raises(ValueError, match="'Sgd' not found in torch.optim"):
        normalize_component({"name": "Sgd", "params": {"lr": 0.1}}, "optimizer")
    with pytest.raises(ValueError, match=r"Adam does not take \['momentum'\]"):
        normalize_component({"name": "Adam", "params": {"momentum": 0.9}}, "optimizer")


def test_create_with_another_optimizer_drops_sgd_defaults():
    cfg = RunConfig.create(optimizer="Adam", lr=1e-3)
    assert cfg.training.optimizer == {"name": "Adam", "params": {"lr": 1e-3}}
    assert RunConfig.create(optimizer="SGD").training.optimizer == RunConfig().training.optimizer
