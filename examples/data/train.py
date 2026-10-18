"""A whole SL experiment from one config: dataset, client split, clients, evaluation.

    python examples/data/train.py -c experiments/m11_cifar10_dirichlet.yaml

The config's ``data`` section picks the dataset and how its training set is split
across ``sl.clients`` clients; ``ts.run`` builds the loaders and reports
``test_loss``/``test_accuracy`` every ``data.eval_every`` epochs.
"""
import argparse
import json

import torch
import torch.nn as nn

import torchslicer as ts


class SmallCNN(nn.Module):
    def __init__(self, in_ch=3, n_classes=10, side=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64 * (side // 4) ** 2, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return self.fc2(self.relu3(self.fc1(self.flat(x))))


class ClassesFirst(nn.Module):
    def forward(self, x):                    # (B, T, V) -> (B, V, T), as CrossEntropyLoss wants
        return x.transpose(1, 2)


class CharLM(nn.Module):
    """A small next-character model: embedding, a causal 1-D conv stack, per-token logits."""

    def __init__(self, vocab=65, d=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.to_ch = ClassesFirst()                       # (B, d, T)
        self.conv1 = nn.Conv1d(d, d, 5, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(d, d, 5, padding=4)
        self.relu2 = nn.ReLU()
        self.head = nn.Conv1d(d, vocab, 1)               # (B, vocab, T)

    def forward(self, x):
        h = self.relu1(self.conv1(self.to_ch(self.embed(x)))[..., :-4])  # causal: drop the padded tail
        h = self.relu2(self.conv2(h)[..., :-4])
        return self.head(h)


SHAPES = {"mnist": (1, 10, 28), "fashion_mnist": (1, 10, 28), "cifar10": (3, 10, 32),
          "cifar100": (3, 100, 32), "synthetic": (3, 10, 32)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    cfg = ts.RunConfig.load(ap.parse_args().config)
    torch.manual_seed(cfg.data.seed)
    if cfg.data.dataset == "shakespeare":
        history = ts.run(lambda: CharLM(), run_config=cfg, cuts=["relu1"])
    else:
        in_ch, n_classes, side = SHAPES[cfg.data.dataset]
        history = ts.run(lambda: SmallCNN(in_ch, n_classes, side), run_config=cfg, cuts=["pool1"])
    for h in history:
        print(f"epoch {h['epoch']}: loss {h['loss']:.4f}  test_acc {h.get('test_accuracy', float('nan')):.4f}  "
              f"bytes_fwd {sum(h['bytes_fwd']) if isinstance(h['bytes_fwd'], list) else h['bytes_fwd']}")
    print("HISTORY " + json.dumps([h["loss"] for h in history]), flush=True)
