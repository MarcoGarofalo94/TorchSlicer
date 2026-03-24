"""
DistilGPT-2 causal LM fine-tuning via split learning (LocalExecutor).

Demonstrates ts.wrap_hf() replacing the manual wrapping in bert_sst2.py.
The adapter decomposes distilgpt2 (6 blocks) into:
  stage 0  : GPT-2 token + position embedding
  stages 1-6: 6 x GPT2Block (opaque — residuals stay internal)
  stage 7  : ln_f + lm_head  (output: [B, vocab_size, T])

ts.slice() calls ModelGraph.from_sequential() on the HFAdapter, producing a
purely linear DAG → validate() always passes regardless of n_partitions.

The head stage returns logits transposed to [B, V, T] so that PyTorch's
built-in CrossEntropyLoss(logits [B,V,T], labels [B,T]) works directly
(matches the [N, C, d1] convention).

Dataset: tiny synthetic text repeated to give enough batches for a loss curve.

Run:
    conda run -n torchslicer python3 examples/finetune/hf_gpt2.py

Requires: pip install transformers
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import torchslicer as ts


# ── dataset ───────────────────────────────────────────────────────────────────

CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "Split learning enables training on constrained devices by partitioning "
    "the model across multiple workers. Each worker holds only its slice. "
    "Activations travel forward and gradients travel backward across the cut. "
    "No worker ever sees the full model or the full dataset at once. "
    "TorchSlicer implements this for arbitrary PyTorch modules. "
) * 80   # ~4 KB corpus


def build_loader(tokenizer, seq_len: int = 64, batch_size: int = 8):
    ids    = tokenizer.encode(CORPUS, return_tensors="pt")[0]   # [N]
    n      = (len(ids) - 1) // seq_len
    inputs = torch.stack([ids[i * seq_len:(i + 1) * seq_len] for i in range(n)])
    labels = torch.stack([ids[i * seq_len + 1:(i + 1) * seq_len + 1] for i in range(n)])
    return DataLoader(TensorDataset(inputs, labels),
                      batch_size=batch_size, shuffle=True, drop_last=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    model_name   = "distilgpt2"
    n_partitions = 2
    seq_len      = 64
    batch_size   = 8
    epochs       = 3

    print(f"Loading {model_name} ...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model     = GPT2LMHeadModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    print(f"  parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  blocks     : {model.config.n_layer}")
    print(f"  device     : {device}")

    adapter = ts.wrap_hf(model, task="causal_lm")
    print(f"\n{adapter}")

    sliced = ts.slice(adapter, strategy="uniform", n=n_partitions)
    print(f"\nPartitions ({n_partitions}):")
    for p in sliced.partitions:
        print(f"  partition {p.index}: layers {p.layer_indices}")

    loader = build_loader(tokenizer, seq_len=seq_len, batch_size=batch_size)
    print(f"\nDataset: {len(loader.dataset)} sequences, "
          f"seq_len={seq_len}, batches={len(loader)}")

    print(f"\nTraining for {epochs} epochs ...\n")
    history = sliced.train(
        loader,
        optimizer  = {"name": "AdamW", "params": {"lr": 5e-5}},
        criterion  = {"name": "CrossEntropyLoss", "params": {}},
        epochs     = epochs,
        verbose    = True,
    )

    print("\nEpoch summary:")
    for i, h in enumerate(history, 1):
        print(f"  epoch {i}/{epochs}  loss={h['loss']:.4f}")


if __name__ == "__main__":
    main()
