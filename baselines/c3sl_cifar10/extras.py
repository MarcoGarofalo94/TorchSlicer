"""C3-SL baseline: CIFAR-10 as the authors load it (random horizontal flip on the training
set, their normalization) and an ImageNet-pretrained VGG-16 as a flat layer list. Imported by
baselines/run.py.
"""
import os

import torch
import torch.nn as nn

import torchslicer as ts
import models

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)   # the authors' train.py


class Flip(torch.utils.data.Dataset):
    def __init__(self, base, train):
        self.base, self.train, self.targets = base, train, base.targets

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        if self.train and torch.rand(()) < 0.5:
            x = x.flip(-1)
        return x, y


def _cifar10_c3sl(train, root, download, seed):
    """The local CIFAR-10 npz files (``root/cifar10_{train,test}.npz``, as the label-leakage
    baseline) when present, else torchvision's download."""
    npz = [os.path.join(root, f"cifar10_{s}.npz") for s in ("train", "test")]
    if all(map(os.path.exists, npz)):
        base = ts.data.load_dataset("npz", train=train, path=npz[0], test_path=npz[1])
    else:
        base = ts.data.load_dataset("cifar10", train=train, root=root, download=download, seed=seed)
    return Flip(ts.data.ImageTensorDataset(base.images, base.targets, MEAN, STD), train)


ts.data.register_dataset("cifar10_c3sl", _cifar10_c3sl)


def vgg16_split(n_classes=10, pretrained=True):
    """torchvision VGG-16 (ImageNet weights) with a new last layer, flattened:
    features 0-30 (the 4th max-pool is 23), avgpool 31, flatten 32, classifier 33-39."""
    import torchvision
    m = torchvision.models.vgg16(weights="IMAGENET1K_V1" if pretrained else None)
    m.classifier[6] = nn.Linear(4096, n_classes)
    return nn.Sequential(*m.features, m.avgpool, nn.Flatten(), *m.classifier)


models.MODELS["vgg16_split"] = vgg16_split
