"""Federated partitioners: split one dataset's samples across clients.

Each partitioner takes the labels (one int per sample), a client count and a seed, and
returns a ``ClientSplit``: one sorted index array per client, plus what produced it.
Same inputs, same seed, same split on every machine (``numpy.random.Generator``)::

    split = ts.data.partition(labels, n_clients=10, method="dirichlet", alpha=0.5, seed=0)
    split.label_counts(labels)      # clients x classes
    loaders = split.loaders(dataset, batch_size=64)
    split.save("runs/x/split.json")

Methods (``ts.data.available_partitioners()``):

- ``iid``: a random permutation cut into near-equal parts.
- ``dirichlet`` (``alpha``): label skew. For every class, the share of its samples
  each client gets is drawn from Dirichlet(alpha); small alpha = few classes per
  client. Redrawn until every client has ``min_size`` samples.
- ``shards`` (``shards_per_client``): pathological label skew. Samples sorted by label,
  cut into ``n_clients * shards_per_client`` equal shards, each client gets that many
  at random (with 2 shards, at most ~2 classes per client).
- ``quantity`` (``beta``): quantity skew. Client sizes drawn from Dirichlet(beta),
  samples assigned at random (labels stay IID).
- ``natural`` (``groups``): the data's own partition, e.g. writers or speakers. Groups
  are never split; each client gets whole groups, the largest first to the client with
  the fewest samples (one group each when there are as many clients as groups).
  ``groups`` defaults to the dataset's ``groups`` attribute.

Feature shift (the same labels, different inputs per client) is a transform on top of
any split: ``ClientSplit.loaders(..., feature_shift=FeatureShift("noise", 0.5))``.

Add one with ``ts.data.register_partitioner(name, fn)``, where
``fn(labels, n_clients, rng, **params) -> list[np.ndarray]``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def labels_of(data) -> np.ndarray:
    """The labels as an int array: from an array/list/tensor, or a dataset's
    ``targets``/``labels`` attribute, or by reading ``dataset[i][1]``."""
    for attr in ("targets", "labels"):
        if hasattr(data, attr) and not isinstance(data, np.ndarray):
            data = getattr(data, attr)
            break
    else:
        if hasattr(data, "__getitem__") and hasattr(data, "__len__") and len(data) \
                and isinstance(data[0], (tuple, list)):
            data = [data[i][1] for i in range(len(data))]
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    out = np.asarray(data)
    if out.ndim != 1 or not np.issubdtype(out.dtype, np.integer):
        raise ValueError(f"labels must be a 1-D integer array, got shape {out.shape} dtype {out.dtype}")
    return out


def _cut(idx: np.ndarray, shares: np.ndarray) -> list[np.ndarray]:
    """Split ``idx`` into ``len(shares)`` parts proportional to ``shares``."""
    bounds = (np.cumsum(shares)[:-1] * len(idx)).astype(int)
    return np.split(idx, bounds)


def _iid(labels, n_clients, rng):
    return np.array_split(rng.permutation(len(labels)), n_clients)


def _dirichlet(labels, n_clients, rng, alpha: float = 0.5, min_size: int = 1, max_tries: int = 1000):
    if alpha <= 0:
        raise ValueError(f"dirichlet: alpha must be > 0, got {alpha}")
    classes = np.unique(labels)
    for _ in range(max_tries):
        parts = [[] for _ in range(n_clients)]
        for c in classes:
            idx = rng.permutation(np.flatnonzero(labels == c))
            for k, chunk in enumerate(_cut(idx, rng.dirichlet(np.full(n_clients, alpha)))):
                parts[k].append(chunk)
        parts = [np.concatenate(p) for p in parts]
        if min(len(p) for p in parts) >= min_size:
            return parts
    raise ValueError(f"dirichlet: no split with every client >= {min_size} samples in {max_tries} "
                     f"draws (alpha={alpha}, {n_clients} clients, {len(labels)} samples); "
                     f"raise alpha or lower min_size")


def _shards(labels, n_clients, rng, shards_per_client: int = 2):
    n_shards = n_clients * shards_per_client
    if n_shards > len(labels):
        raise ValueError(f"shards: {n_shards} shards for {len(labels)} samples")
    order = rng.permutation(len(labels))
    order = order[np.argsort(labels[order], kind="stable")]   # by label, random within a class
    shards = np.array_split(order, n_shards)
    pick = rng.permutation(n_shards).reshape(n_clients, shards_per_client)
    return [np.concatenate([shards[s] for s in row]) for row in pick]


def _quantity(labels, n_clients, rng, beta: float = 0.5, min_size: int = 1, max_tries: int = 1000):
    if beta <= 0:
        raise ValueError(f"quantity: beta must be > 0, got {beta}")
    idx = rng.permutation(len(labels))
    for _ in range(max_tries):
        parts = _cut(idx, rng.dirichlet(np.full(n_clients, beta)))
        if min(len(p) for p in parts) >= min_size:
            return parts
    raise ValueError(f"quantity: no split with every client >= {min_size} samples in {max_tries} "
                     f"draws (beta={beta}, {n_clients} clients); raise beta or lower min_size")


def _natural(labels, n_clients, rng, groups=None):
    if groups is None:
        raise ValueError("natural: needs groups (one id per sample), e.g. the dataset's groups")
    groups = np.asarray(groups)
    if len(groups) != len(labels):
        raise ValueError(f"natural: {len(groups)} group ids for {len(labels)} samples")
    ids = rng.permutation(np.unique(groups))
    if n_clients > len(ids):
        raise ValueError(f"natural: {n_clients} clients but only {len(ids)} groups")
    members = {g: np.flatnonzero(groups == g) for g in ids}
    order = sorted(ids, key=lambda g: -len(members[g]))          # stable: ties keep the shuffle
    parts, sizes = [[] for _ in range(n_clients)], np.zeros(n_clients, dtype=np.int64)
    for g in order:
        k = int(np.argmin(sizes))
        parts[k].append(members[g])
        sizes[k] += len(members[g])
    return [np.concatenate(p) for p in parts]


_PARTITIONERS = {"iid": _iid, "dirichlet": _dirichlet, "shards": _shards, "quantity": _quantity,
                 "natural": _natural}


def register_partitioner(name: str, fn) -> None:
    _PARTITIONERS[name] = fn


def available_partitioners() -> list[str]:
    return list(_PARTITIONERS)


@dataclass
class ClientSplit:
    """One sorted index array per client, and the recipe that produced it."""
    indices: list
    method: str
    seed: int
    params: dict = field(default_factory=dict)

    @property
    def n_clients(self) -> int:
        return len(self.indices)

    def sizes(self) -> list[int]:
        return [len(i) for i in self.indices]

    def label_counts(self, labels, n_classes: int | None = None) -> np.ndarray:
        """``counts[k, c]`` = samples of class ``c`` on client ``k``."""
        labels = labels_of(labels)
        n_classes = n_classes or int(labels.max()) + 1
        return np.stack([np.bincount(labels[i], minlength=n_classes) for i in self.indices])

    def subsets(self, dataset) -> list:
        from torch.utils.data import Subset
        return [Subset(dataset, i.tolist()) for i in self.indices]

    def loaders(self, dataset, batch_size: int, shuffle: bool = True, feature_shift=None, **kwargs) -> list:
        """One ``DataLoader`` per client. Client ``k`` shuffles with seed ``seed + k``;
        ``feature_shift`` (a ``FeatureShift``) transforms client ``k``'s inputs."""
        import torch
        from torch.utils.data import DataLoader
        subs = self.subsets(dataset)
        if feature_shift is not None:
            subs = [_Shifted(dataset, idx, feature_shift, k, self.n_clients)
                    for k, idx in enumerate(self.indices)]
        out = []
        for k, sub in enumerate(subs):
            g = torch.Generator().manual_seed(self.seed + k) if shuffle else None
            out.append(DataLoader(sub, batch_size=batch_size, shuffle=shuffle, generator=g, **kwargs))
        return out

    def to_dict(self) -> dict:
        return {"method": self.method, "seed": self.seed, "params": self.params,
                "indices": [i.tolist() for i in self.indices]}

    @classmethod
    def from_dict(cls, d: dict) -> "ClientSplit":
        return cls([np.asarray(i, dtype=np.int64) for i in d["indices"]],
                   d["method"], int(d["seed"]), dict(d.get("params") or {}))

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()))
        return path

    @classmethod
    def load(cls, path) -> "ClientSplit":
        return cls.from_dict(json.loads(Path(path).read_text()))


def partition(labels, n_clients: int, method: str = "iid", seed: int = 0, **params) -> ClientSplit:
    """Split samples across ``n_clients`` with ``method`` (see the module docstring).
    ``labels``: the labels, or a dataset to read them from (``labels_of``)."""
    if method not in _PARTITIONERS:
        raise KeyError(f"unknown partitioner {method!r}; available: {available_partitioners()}")
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}")
    if method == "natural" and params.get("groups") is None and hasattr(labels, "groups"):
        params = dict(params, groups=labels.groups)
    labels = labels_of(labels)
    if n_clients > len(labels):
        raise ValueError(f"{n_clients} clients for {len(labels)} samples")
    parts = _PARTITIONERS[method](labels, n_clients, np.random.default_rng(seed), **params)
    if len(parts) != n_clients:
        raise ValueError(f"partitioner {method!r} returned {len(parts)} parts for {n_clients} clients")
    recipe = {k: v for k, v in params.items() if k != "groups"}     # saved with the split
    return ClientSplit([np.sort(np.asarray(p, dtype=np.int64)) for p in parts], method, seed, recipe)


class FeatureShift:
    """Per-client input shift, growing with the client index ``k`` of ``n`` (client 0
    unshifted, client n-1 at ``level``), as in noise-based feature skew:

    - ``noise``: add Gaussian noise of std ``level * k / (n - 1)``, fixed per sample
      (seeded by ``seed``, the client and the sample), so every epoch sees the same input.
    - ``rotate``: rotate images by ``level * k / (n - 1)`` degrees.
    """

    KINDS = ("noise", "rotate")

    def __init__(self, kind: str, level: float, seed: int = 0):
        if kind not in self.KINDS:
            raise ValueError(f"feature shift kind must be one of {self.KINDS}, got {kind!r}")
        self.kind, self.level, self.seed = kind, float(level), int(seed)

    def __call__(self, x, k: int, n: int, sample: int):
        import torch
        amount = self.level * (k / (n - 1) if n > 1 else 0.0)
        if amount == 0.0 or not x.is_floating_point():
            return x
        if self.kind == "noise":
            g = torch.Generator().manual_seed((self.seed * 1_000_003 + k) * 1_000_003 + sample)
            return x + amount * torch.randn(x.shape, generator=g, dtype=x.dtype)
        import math
        import torch.nn.functional as F
        a = math.radians(amount)
        theta = torch.tensor([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0]],
                             dtype=x.dtype).unsqueeze(0)
        img = x.unsqueeze(0) if x.dim() == 3 else x.view(1, 1, *x.shape)
        grid = F.affine_grid(theta, list(img.shape), align_corners=False)
        return F.grid_sample(img, grid, align_corners=False).view_as(x)


class _Shifted:
    """Client ``k``'s samples of ``dataset`` with ``shift`` applied to the inputs."""

    def __init__(self, dataset, indices, shift: FeatureShift, k: int, n: int):
        self.dataset, self.indices, self.shift, self.k, self.n = dataset, indices, shift, k, n

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        x, y = self.dataset[j]
        return self.shift(x, self.k, self.n, j), y
