"""Standard datasets for SL experiments, as in-memory tensor datasets.

``load_dataset(name, root, train)`` returns a ``TensorDataset``-like object with
``targets`` (for the partitioners), normalized with the usual per-channel statistics.
torchvision is only used to download and parse the files (optional dependency); the
images are kept as one uint8 tensor and normalized per item, so no PIL per sample.

Names (``available_datasets()``):

- ``mnist``, ``fashion_mnist``: 1x28x28, 10 classes.
- ``cifar10``, ``cifar100``: 3x32x32, 10 / 100 classes.
- ``synthetic``: random Gaussian inputs and labels, fixed by ``seed`` (no download; for
  smoke tests). Params: ``shape`` (default [3, 32, 32]), ``classes`` (10), ``n``
  (1024 train / 256 test).
- ``npz``: a local ``.npz`` file with ``x`` (N, H, W[, C] uint8 or float) and ``y`` (N,)
  arrays, and optionally ``groups`` (N,) for the ``natural`` partitioner (e.g. LEAF's
  FEMNIST writers). Params: ``path`` (train file), ``test_path``, optional ``mean``/``std``.
- ``shakespeare``: next-character prediction on tiny-shakespeare (1 MB), in windows of
  ``seq_len`` (64) characters, each tagged with its speaking role in ``groups``, so the
  ``natural`` partitioner gives one or more roles per client (in the spirit of LEAF's
  Shakespeare; not the same corpus or split). ``x``, ``y``: int64 ``(seq_len,)``, ``y`` =
  ``x`` shifted by one. ``targets`` (what the partitioners read) are the role ids;
  ``vocab`` is the character list. A seeded 10 % of the windows is the test set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_STATS = {   # per-channel mean, std of the training set
    "mnist":         ((0.1307,), (0.3081,)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
    "cifar10":       ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100":      ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


class ImageTensorDataset(Dataset):
    """Images as one NCHW tensor (uint8 scaled to [0, 1], or float), normalized per
    item with ``mean``/``std``; ``targets`` is an int64 tensor."""

    def __init__(self, images: torch.Tensor, targets: torch.Tensor, mean=None, std=None):
        if len(images) != len(targets):
            raise ValueError(f"{len(images)} images but {len(targets)} targets")
        self.images = images
        self.targets = targets.long()
        c = images.shape[1] if images.dim() == 4 else 1
        self._mean = torch.tensor(mean if mean is not None else [0.0] * c).view(-1, 1, 1)
        self._std = torch.tensor(std if std is not None else [1.0] * c).view(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i):
        x = self.images[i]
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        return (x - self._mean) / self._std, self.targets[i]


def _nchw(x: np.ndarray) -> torch.Tensor:
    """(N, H, W) or (N, H, W, C) -> (N, C, H, W); (N, C, H, W) unchanged."""
    t = torch.from_numpy(np.ascontiguousarray(x))
    if t.dim() == 3:
        return t.unsqueeze(1)
    if t.dim() == 4 and t.shape[-1] in (1, 3) and t.shape[1] not in (1, 3):
        return t.permute(0, 3, 1, 2).contiguous()
    return t


def _torchvision(name: str, root: str, train: bool, download: bool):
    try:
        import torchvision.datasets as tvd
    except ImportError as e:
        raise ImportError(f"dataset {name!r} needs torchvision (pip install torchvision)") from e
    cls = {"mnist": tvd.MNIST, "fashion_mnist": tvd.FashionMNIST,
           "cifar10": tvd.CIFAR10, "cifar100": tvd.CIFAR100}[name]
    ds = cls(root=root, train=train, download=download)
    x = ds.data.numpy() if hasattr(ds.data, "numpy") else np.asarray(ds.data)
    y = torch.as_tensor(np.asarray(ds.targets), dtype=torch.long)
    mean, std = _STATS[name]
    return ImageTensorDataset(_nchw(x), y, mean, std)


def _synthetic(train: bool, seed: int = 0, shape=(3, 32, 32), classes: int = 10, n: int | None = None):
    n = n if n is not None else (1024 if train else 256)
    g = torch.Generator().manual_seed(seed * 2 + (0 if train else 1))
    x = torch.randn(n, *shape, generator=g)
    y = torch.randint(0, classes, (n,), generator=g)
    return ImageTensorDataset(x, y) if len(shape) == 3 else _Flat(x, y)


class _Flat(Dataset):
    def __init__(self, x, y):
        self.x, self.targets = x, y

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


def _npz(train: bool, path: str, test_path: str | None = None, mean=None, std=None):
    file = path if train else test_path
    if not file:
        raise ValueError("dataset npz: params.test_path is needed for the test set")
    z = np.load(Path(file))
    x = _nchw(z["x"])
    ds = ImageTensorDataset(x, torch.as_tensor(z["y"], dtype=torch.long), mean, std)
    if "groups" in z.files:
        ds.groups = np.asarray(z["groups"], dtype=np.int64)
    return ds


class TextLMDataset(Dataset):
    """Next-token windows over one token sequence: ``x = ids[s:s+T]``, ``y = ids[s+1:s+T+1]``."""

    def __init__(self, ids: torch.Tensor, starts: np.ndarray, seq_len: int, groups: np.ndarray, vocab):
        self.ids, self.starts, self.seq_len = ids, starts, seq_len
        self.groups = groups
        self.targets = torch.as_tensor(groups, dtype=torch.long)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        return self.ids[s:s + self.seq_len], self.ids[s + 1:s + self.seq_len + 1]


_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _shakespeare(root: str, train: bool, download: bool, seed: int = 0, seq_len: int = 64,
                 test_fraction: float = 0.1):
    path = Path(root) / "shakespeare" / "input.txt"
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} is missing and download is off")
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_SHAKESPEARE_URL, path)
    text = path.read_text()
    vocab = sorted(set(text))
    index = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([index[c] for c in text], dtype=torch.long)
    roles, starts, groups, pos = {}, [], [], 0
    for block in text.split("\n\n"):                 # "NAME:\nspeech..." per speech
        head, _, speech = block.partition("\n")
        begin = pos + len(head) + 1                    # the speech, after the name line
        if head.endswith(":") and len(speech) > seq_len:
            role = roles.setdefault(head[:-1].strip(), len(roles))
            for s in range(begin, begin + len(speech) - seq_len, seq_len):
                starts.append(s)
                groups.append(role)
        pos += len(block) + 2
    starts, groups = np.array(starts), np.array(groups)
    test = np.random.default_rng(seed).random(len(starts)) < test_fraction
    keep = ~test if train else test
    return TextLMDataset(ids, starts[keep], seq_len, groups[keep], vocab)


_DATASETS = ("mnist", "fashion_mnist", "cifar10", "cifar100", "synthetic", "npz", "shakespeare")
_REGISTERED: dict = {}


def register_dataset(name: str, fn) -> None:
    """``fn(train, root, download, seed, **params) -> Dataset`` under ``data.dataset: name``.
    The partitioners read its ``targets`` (or ``groups``) when it is split across clients."""
    _REGISTERED[name.strip().lower()] = fn


def available_datasets() -> list[str]:
    return list(_DATASETS) + sorted(_REGISTERED)


def load_dataset(name: str, root: str = "data", train: bool = True, download: bool = True,
                 seed: int = 0, **params):
    """One split (train or test) of a named dataset; see the module docstring."""
    name = name.strip().lower()
    if name in _REGISTERED:
        return _REGISTERED[name](train=train, root=root, download=download, seed=seed, **params)
    if name in _STATS:
        if params:
            raise ValueError(f"dataset {name!r} takes no params, got {sorted(params)}")
        return _torchvision(name, root, train, download)
    if name == "synthetic":
        return _synthetic(train, seed=seed, **params)
    if name == "npz":
        return _npz(train, **params)
    if name == "shakespeare":
        return _shakespeare(root, train, download, seed=seed, **params)
    raise KeyError(f"unknown dataset {name!r}; available: {available_datasets()}")
