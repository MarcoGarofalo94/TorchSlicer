"""
HuggingFace GPT-2 split learning — centralized topology, DistributedExecutor.

Supports three experiment modes via YAML config:
  baseline : standard full-parameter fine-tuning
  lora     : LoRA (r=8, target c_attn + c_proj)
  gpipe    : baseline + GPipe micro-batch pipelining

Model   : distilgpt2 (82M params, 6 transformer blocks)
Dataset : WikiText-2 raw (HuggingFace datasets)
Split   : uniform across N workers (embed+blocks / blocks / ... / blocks+head)

The ``model`` section of the YAML is read directly here (RunConfig ignores it):

    model:
      name: distilgpt2          # any AutoModelForCausalLM identifier
      task: causal_lm
      use_lora: false
      lora_r: 8
      lora_alpha: 16
      lora_target_modules: [c_attn, c_proj]
      block_size: 128           # token sequence length
      n_train: 4000             # sequences sampled from WikiText-2 train split
      batch_size: 16

Usage:
    make run-hf-dist-gpu CONFIG=experiments/hf_gpt2_4gpu_baseline.yaml
    make run-hf-dist-gpu CONFIG=experiments/hf_gpt2_4gpu_lora.yaml
    make run-hf-dist-gpu CONFIG=experiments/hf_gpt2_4gpu_gpipe.yaml
"""

import argparse
import os
import signal
import sys

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import torchslicer as ts
from torchslicer.config import RunConfig
from torchslicer.discovery import CoordinatorDiscovery
from torchslicer.executors.distributed import DistributedExecutor
from torchslicer.monitor import tracer


# ── dataset ───────────────────────────────────────────────────────────────────

def build_loader(model_name: str, block_size: int, n_train: int, batch_size: int):
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("[coordinator] loading WikiText-2 ...")
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                       trust_remote_code=True)
    text = "\n".join(line for line in raw["text"] if line.strip())
    ids  = tokenizer.encode(text)

    n_available = (len(ids) - 1) // block_size
    n           = min(n_available, n_train)
    torch.manual_seed(42)
    perm   = torch.randperm(n_available)[:n]
    inputs = torch.stack([
        torch.tensor(ids[i * block_size     : (i + 1) * block_size],     dtype=torch.long)
        for i in perm.tolist()
    ])
    labels = torch.stack([
        torch.tensor(ids[i * block_size + 1 : (i + 1) * block_size + 1], dtype=torch.long)
        for i in perm.tolist()
    ])

    loader = DataLoader(TensorDataset(inputs, labels),
                        batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"[coordinator] dataset: {len(ids):,} tokens  "
          f"sequences={n:,}  block_size={block_size}  "
          f"batch_size={batch_size}  batches/epoch={len(loader)}")
    return loader


# ── model ─────────────────────────────────────────────────────────────────────

def build_model(model_cfg: dict) -> ts.HFAdapter:
    model_name = model_cfg.get("name", "distilgpt2")
    use_lora   = model_cfg.get("use_lora", False)
    task       = model_cfg.get("task", "causal_lm")

    print(f"[coordinator] loading {model_name} ...")
    model    = AutoModelForCausalLM.from_pretrained(model_name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[coordinator]   params: {n_params:,}")

    if use_lora:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r               = model_cfg.get("lora_r", 8),
            lora_alpha      = model_cfg.get("lora_alpha", 16),
            target_modules  = model_cfg.get("lora_target_modules", ["c_attn", "c_proj"]),
            lora_dropout    = 0.05,
            bias            = "none",
            task_type       = "CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        # In split learning the lm_head lands on the last worker as a standalone
        # module (weight tying to wte is broken at the serialization boundary).
        # Unfreeze it so the last worker can train the output projection.
        for p in model.base_model.model.lm_head.parameters():
            p.requires_grad_(True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[coordinator]   LoRA trainable: {trainable:,} / {n_params:,} "
              f"({100 * trainable / n_params:.2f}%)")
        model = ts.peft_unwrap(model)

    adapter = ts.wrap_hf(model, task=task)
    print(f"[coordinator]   {adapter}")
    return adapter


# ── entry point ───────────────────────────────────────────────────────────────

def serve():
    tracer.auto_configure_if_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("port", nargs="?", default="50054")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = RunConfig.load(args.config)

    # Read model section — RunConfig ignores unknown top-level keys
    model_cfg   = {}
    config_path = args.config or os.environ.get("EXPERIMENT_CONFIG")
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            model_cfg = yaml.safe_load(f).get("model", {})

    n = cfg.discovery.n_workers
    print(f"[coordinator] run_id={cfg.run_id}  n_workers={n}  "
          f"epochs={cfg.training.epochs}  gpipe={cfg.pipeline.use_gpipe}  "
          f"n_micro={cfg.pipeline.n_micro}  lora={model_cfg.get('use_lora', False)}")

    discovery = CoordinatorDiscovery(run_id=cfg.run_id)
    executor  = DistributedExecutor(
        discovery        = discovery,
        coordinator_addr = f"coordinator:{args.port}",
        run_config       = cfg,
    )

    model  = build_model(model_cfg)
    sliced = ts.slice(model, strategy="uniform", n=n, executor=executor)

    print(f"[coordinator] partitions ({n}):")
    for p in sliced.partitions:
        print(f"  {p.index}: layers {p.layer_indices}")

    loader = build_loader(
        model_name = model_cfg.get("name", "distilgpt2"),
        block_size = model_cfg.get("block_size", 128),
        n_train    = model_cfg.get("n_train", 4000),
        batch_size = model_cfg.get("batch_size", 16),
    )

    sliced.train(
        loader,
        cfg.training.optimizer,
        cfg.training.criterion,
        epochs          = cfg.training.epochs,
        verbose         = True,
        use_gpipe       = cfg.pipeline.use_gpipe,
        n_micro_batches = cfg.pipeline.n_micro,
        run_config      = cfg,
    )

    run_dir = os.path.join(cfg.logging.dir, cfg.run_id) if cfg.logging.enabled else ""
    print("[coordinator] training complete — waiting for SIGTERM (docker compose down)")
    if run_dir:
        print(f"[coordinator] metrics → {run_dir}/metrics.jsonl")
        print(f"[coordinator] workers → {run_dir}/worker_epoch.jsonl")
    signal.pause()


if __name__ == "__main__":
    serve()
