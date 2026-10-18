"""Build an experiment's loaders from the config's ``data`` section.

With ``data.dataset`` set, ``ts.run(model_fn)`` needs no ``data_fn``: the training set is
loaded, optionally subsampled, split across ``sl.clients`` clients with
``data.partition``, and the test set is evaluated every ``data.eval_every`` epochs::

    data:
      dataset: cifar10
      partition: dirichlet
      partition_params: {alpha: 0.5}
      batch_size: 64
    sl:
      clients: 10
      client_mode: parallel
"""
from __future__ import annotations

import os

import numpy as np

from .datasets import load_dataset
from .partition import ClientSplit, labels_of, partition


def _subset(dataset, n: int, seed: int):
    if not n or n >= len(dataset):
        return dataset
    from torch.utils.data import Subset
    idx = np.sort(np.random.default_rng(seed).permutation(len(dataset))[:n])
    sub = Subset(dataset, idx.tolist())
    sub.targets = labels_of(dataset)[idx]
    return sub


def _loader(dataset, batch_size: int, shuffle: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader
    g = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)


def build_data(cfg):
    """``(train, test_loader, split)`` for ``cfg``: ``train`` is one loader, or a list of
    one loader per client when ``sl.clients`` > 1 (``split`` = the ``ClientSplit``,
    else ``None``); ``test_loader`` is ``None`` when ``data.eval_every`` is 0."""
    d = cfg.data
    if not d.dataset:
        raise ValueError("data.dataset is not set")
    common = dict(root=d.root, download=d.download, seed=d.seed, **(d.params or {}))
    train = _subset(load_dataset(d.dataset, train=True, **common), d.train_subset, d.seed)
    clients = int(cfg.sl.clients or 1)
    split = None
    if clients > 1:
        split = partition(train, clients, d.partition, seed=d.seed, **(d.partition_params or {}))
        train_out = split.loaders(train, d.batch_size)
        if cfg.logging.enabled:
            split.save(os.path.join(cfg.logging.dir, cfg.run_id, "client_split.json"))
    else:
        train_out = _loader(train, d.batch_size, shuffle=True, seed=d.seed)
    test = None
    if d.eval_every:
        test_ds = _subset(load_dataset(d.dataset, train=False, **common), d.test_subset, d.seed)
        test = _loader(test_ds, d.test_batch_size, shuffle=False, seed=d.seed)
    return train_out, test, split


__all__ = ["build_data", "ClientSplit"]
