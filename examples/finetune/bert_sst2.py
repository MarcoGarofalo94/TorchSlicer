"""
BERT-tiny SST-2 fine-tuning via split learning (2 partitions, LocalExecutor).

Model:  prajjwal1/bert-tiny  (4.4 M params, 2 transformer layers, hidden=128)
Task:   binary sentiment classification (positive / negative)
Split:  partition 0 → [Embedding, TransformerLayer-0]
        partition 1 → [TransformerLayer-1, Pooler+Classifier]

The model is wrapped into nn.Sequential-compatible modules so that each
partition receives a single tensor and returns a single tensor. Attention
masks are omitted (full attention) — acceptable for short, fixed-length
sequences; for production pass the mask explicitly.

Run:
    conda run -n torchslicer python3 examples/finetune/bert_sst2.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer

import torchslicer as ts


# ── Sequential-compatible wrappers ───────────────────────────────────────────

class EmbedLayer(nn.Module):
    """Embedding + position encodings. Accepts token-id LongTensor."""
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, input_ids):
        return self.embeddings(input_ids)          # → [B, L, H]


class TransformerLayer(nn.Module):
    """Single BERT encoder layer, sequential-compatible (no attention mask)."""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden):
        out = self.layer(hidden, attention_mask=None)
        # transformers <5 returns a tuple; ≥5 returns a Tensor directly
        return out[0] if isinstance(out, (tuple, list)) else out  # → [B, L, H]


class ClassifierHead(nn.Module):
    """Extracts [CLS] token, runs pooler dense + dropout + classifier."""
    def __init__(self, pooler, dropout, classifier):
        super().__init__()
        self.dense = pooler.dense        # nn.Linear(H, H)
        self.activation = pooler.activation  # nn.Tanh()
        self.dropout = dropout
        self.classifier = classifier

    def forward(self, hidden):
        # hidden: [B, L, H]
        cls = hidden[:, 0]                           # [B, H]
        pooled = self.activation(self.dense(cls))    # [B, H]
        return self.classifier(self.dropout(pooled)) # [B, C]


# ── Build sequential model ────────────────────────────────────────────────────

def build_model(model_name: str = "prajjwal1/bert-tiny", num_labels: int = 2):
    bert = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels)
    layers = [
        EmbedLayer(bert.bert.embeddings),
        *[TransformerLayer(l) for l in bert.bert.encoder.layer],
        ClassifierHead(bert.bert.pooler, bert.dropout, bert.classifier),
    ]
    return nn.Sequential(*layers)


# ── Dataset ───────────────────────────────────────────────────────────────────

SENTENCES = [
    ("This movie is absolutely wonderful!", 1),
    ("A brilliant and touching story.", 1),
    ("Funny, heartwarming, and beautifully acted.", 1),
    ("A masterpiece of modern cinema.", 1),
    ("The performances were outstanding.", 1),
    ("I loved every minute of it.", 1),
    ("Engaging from start to finish.", 1),
    ("A truly uplifting experience.", 1),
    ("Terrible film, waste of time.", 0),
    ("I fell asleep halfway through.", 0),
    ("One of the worst films I've ever seen.", 0),
    ("Complete waste of time and money.", 0),
    ("Painfully dull and predictable.", 0),
    ("The acting was wooden and unconvincing.", 0),
    ("A boring mess with no redeeming qualities.", 0),
    ("Absolutely unwatchable from start to finish.", 0),
]


def build_loader(tokenizer, batch_size: int = 16, repeats: int = 8,
                 max_length: int = 32):
    texts, labels = zip(*(SENTENCES * repeats))
    enc = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    ds = TensorDataset(enc["input_ids"], torch.tensor(list(labels)))
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    model_name = "prajjwal1/bert-tiny"
    tokenizer   = BertTokenizer.from_pretrained(model_name)
    model       = build_model(model_name)

    n_transformer_layers = len(list(model.children())) - 2  # excl. embed + head
    print(f"Model: {model_name}")
    print(f"  transformer layers : {n_transformer_layers}")
    print(f"  sequential modules : {len(list(model.children()))}")
    print(f"  total parameters   : {sum(p.numel() for p in model.parameters()):,}")

    loader = build_loader(tokenizer)
    n_partitions = 2

    sliced = ts.slice(model, strategy="uniform", n=n_partitions)
    print(f"\nPartitions ({n_partitions}):")
    for p in sliced.partitions:
        print(f"  partition {p.index}: layers {p.layer_indices}")

    print(f"\nTraining on {len(loader.dataset)} examples, "
          f"batch={loader.batch_size}, epochs=5\n")

    history = sliced.train(
        loader,
        optimizer={"name": "AdamW", "params": {"lr": 2e-5}},
        criterion={"name": "CrossEntropyLoss", "params": {}},
        epochs=5,
    )

    for i, h in enumerate(history):
        print(f"epoch {i + 1}/5  loss={h['loss']:.4f}")


if __name__ == "__main__":
    main()
