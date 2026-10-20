"""SLPerf's non-IID split, registered as partitioner ``slperf_hetero`` (imported by
baselines/run.py before this baseline runs).

Ported from SLPerf's ``cifar10Partition.hetero`` (SLFrame/core/dataset/partition/
cifar10Partition.py), which its MNIST runs use: per class, Dirichlet(alpha) proportions over
the clients, zeroed for clients that already hold >= N / n_clients samples; repeated while
the smallest client has < 10 samples, **without** resetting the clients' lists between
attempts (as in the original, where the reset is commented out); then every client is cut
to the smallest client's size. The last step discards much of the data at small alpha,
which TorchSlicer's own ``dirichlet`` does not do.
"""
import numpy as np

import torchslicer as ts


def slperf_hetero(labels, n_clients, rng, alpha: float = 0.1):
    labels = np.asarray(labels)
    K, N = len(np.unique(labels)), len(labels)
    batches = [[] for _ in range(n_clients)]
    min_size = 0
    while min_size < 10:
        for k in range(K):
            idx_k = np.flatnonzero(labels == k)
            rng.shuffle(idx_k)
            p = rng.dirichlet(np.repeat(alpha, n_clients))
            p = np.array([q * (len(b) < N / n_clients) for q, b in zip(p, batches)])
            p = p / p.sum()
            cuts = (np.cumsum(p) * len(idx_k)).astype(int)[:-1]
            batches = [b + part.tolist() for b, part in zip(batches, np.split(idx_k, cuts))]
            min_size = min(len(b) for b in batches)
    smallest = min(len(b) for b in batches)
    out = []
    for b in batches:
        b = np.array(b)
        rng.shuffle(b)
        out.append(b[:smallest])
    return out


ts.data.register_partitioner("slperf_hetero", slperf_hetero)
