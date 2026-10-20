"""Models used by the baselines, by name (``baseline.model`` in a baseline's config).

Each builder returns a flat ``nn.Sequential`` so a baseline's cuts are plain layer indices.
"""
import torch.nn as nn


def lenet(in_ch: int = 1, n_classes: int = 10, side: int = 28) -> nn.Sequential:
    """LeNet with ReLU and max pooling, as in SLPerf's ``LeNetComplete``:
    conv(0) relu(1) pool(2) | conv(3) relu(4) pool(5) flatten(6) | fc(7) relu(8) fc(9)
    relu(10) fc(11). 28x28 inputs give 16x4x4 = 256 features."""
    flat = 16 * ((((side - 4) // 2) - 4) // 2) ** 2
    return nn.Sequential(
        nn.Conv2d(in_ch, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Flatten(),
        nn.Linear(flat, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, n_classes))


MODELS = {"lenet": lenet}


def cnn(in_ch: int = 3, n_classes: int = 10, side: int = 32) -> nn.Sequential:
    """conv16(0) relu(1) pool(2) | conv32(3) relu(4) pool(5) flatten(6) fc128(7) relu(8) fc(9)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
        nn.Linear(32 * (side // 4) ** 2, 128), nn.ReLU(), nn.Linear(128, n_classes))


MODELS["cnn"] = cnn
