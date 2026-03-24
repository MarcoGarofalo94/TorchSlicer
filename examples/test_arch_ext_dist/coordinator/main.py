"""
Architecture extension distributed smoke test — coordinator.

Runs two tests sequentially on the same 2 workers:
  1. Mistral-tiny (from config, no download) — LLaMA/RoPE branch, 2 epochs
  2. Synthetic DeepSeek-V2 MoE structure — MoEBlockStage + aux loss backward, 2 epochs

Workers stay alive between tests; coordinator uses reinit() to push a new model.
"""

import argparse
import os
import signal
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

import torchslicer as ts
from torchslicer.config import RunConfig
from torchslicer.discovery import CoordinatorDiscovery
from torchslicer.executors.distributed import DistributedExecutor

# Shared synthetic model classes — must be importable with the same module path
# on both coordinator and workers so torch.load() can deserialize slices.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import DeepSeekMoEModel  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

VOCAB, SEQ, BATCH, N = 512, 16, 4, 32
EPOCHS = 2
OPT  = {"name": "AdamW",         "params": {"lr": 5e-4}}
CRIT = {"name": "CrossEntropyLoss", "params": {}}


def make_loader():
    ids    = torch.randint(0, VOCAB, (N, SEQ))
    labels = torch.randint(0, VOCAB, (N, SEQ))
    return DataLoader(TensorDataset(ids, labels), batch_size=BATCH, drop_last=True)


def run_epochs(executor, loader, epochs, name):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    history = []
    for epoch in range(1, epochs + 1):
        r = executor.train_epoch(loader, epoch=epoch, verbose=True, total_epochs=epochs)
        history.append(r)
    losses = [round(h["loss"], 4) for h in history]
    print(f"  loss: {losses[0]} → {losses[-1]}")
    assert losses[-1] < losses[0] * 1.1, "loss did not decrease"
    print("  OK")
    return history


# ── Test 1: Mistral-tiny ──────────────────────────────────────────────────────

def build_mistral():
    from transformers import MistralConfig, MistralForCausalLM
    config = MistralConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, _attn_implementation="eager",
    )
    return ts.wrap_hf(MistralForCausalLM(config), task="causal_lm")


# ── Test 2: Synthetic DeepSeek MoE ───────────────────────────────────────────

def build_deepseek():
    return ts.wrap_hf(DeepSeekMoEModel(vocab=VOCAB, d=64, n_layers=2), task="causal_lm")


# ── entry point ───────────────────────────────────────────────────────────────

def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("port", nargs="?", default="50054")
    args = parser.parse_args()

    cfg = RunConfig.load(os.environ.get("EXPERIMENT_CONFIG"))

    n_workers = cfg.discovery.n_workers
    print(f"[coordinator] run_id={cfg.run_id}  n_workers={n_workers}")

    discovery = CoordinatorDiscovery(run_id=cfg.run_id)
    executor  = DistributedExecutor(
        discovery        = discovery,
        coordinator_addr = f"coordinator:{args.port}",
        run_config       = cfg,
    )

    loader = make_loader()

    # ── Test 1: Mistral ───────────────────────────────────────────────────────
    adapter1 = build_mistral()
    stages1  = [type(s).__name__ for s in adapter1]
    print(f"[coordinator] Mistral stages ({len(adapter1)}): {stages1}")
    sliced1  = ts.slice(adapter1, strategy="uniform", n=n_workers, executor=executor)

    executor.setup(
        sliced1.graph, sliced1.partitions, OPT, CRIT,
        model_name="Mistral-tiny", strategy_name="uniform",
    )
    run_epochs(executor, loader, EPOCHS, "Test 1 — Mistral-tiny (LLaMA/RoPE branch)")

    # ── Test 2: DeepSeek MoE ─────────────────────────────────────────────────
    adapter2   = build_deepseek()
    stages2    = [type(s).__name__ for s in adapter2]
    moe_count  = sum(1 for s in adapter2 if isinstance(s, ts.MoEBlockStage))
    print(f"[coordinator] DeepSeek stages ({len(adapter2)}): {stages2}  "
          f"MoEBlockStage={moe_count}")
    sliced2 = ts.slice(adapter2, strategy="uniform", n=n_workers, executor=executor)

    executor.reinit(
        sliced2.graph, sliced2.partitions, OPT, CRIT,
        model_name="DeepSeek-MoE-synthetic", strategy_name="uniform",
    )
    run_epochs(executor, loader, EPOCHS, "Test 2 — DeepSeek-MoE-synthetic (MoEBlockStage)")

    # ── done ──────────────────────────────────────────────────────────────────
    executor.teardown()
    print("\n[coordinator] all architecture extension tests passed")
    print("[coordinator] waiting for SIGTERM (docker compose down)")
    signal.pause()


if __name__ == "__main__":
    serve()
