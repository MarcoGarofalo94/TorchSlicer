"""M11 federated partitioners: every split is a seeded partition of the samples, and its
statistics match the definition of its method."""
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

import torchslicer as ts
from torchslicer.data import ClientSplit, labels_of, partition

METHODS = [("iid", {}), ("dirichlet", {"alpha": 0.3}), ("shards", {"shards_per_client": 2}),
           ("quantity", {"beta": 0.5})]


def _labels(n_classes=10, per_class=100, seed=0):
    return np.random.default_rng(seed).permutation(np.repeat(np.arange(n_classes), per_class))


@pytest.mark.parametrize("method,params", METHODS)
def test_split_is_a_seeded_partition(method, params):
    labels = _labels()
    a = partition(labels, 10, method, seed=3, **params)
    assert a.n_clients == 10 and all(len(i) > 0 for i in a.indices)
    flat = np.concatenate(a.indices)
    assert len(flat) == len(labels) and np.array_equal(np.sort(flat), np.arange(len(labels)))
    assert all(np.array_equal(i, np.sort(i)) for i in a.indices)
    b = partition(labels, 10, method, seed=3, **params)
    assert all(np.array_equal(x, y) for x, y in zip(a.indices, b.indices))
    c = partition(labels, 10, method, seed=4, **params)
    assert not all(np.array_equal(x, y) for x, y in zip(a.indices, c.indices))


def test_iid_sizes_and_label_mix():
    labels = _labels(per_class=1000)
    split = partition(labels, 10, "iid")
    assert max(split.sizes()) - min(split.sizes()) <= 1
    share = split.label_counts(labels) / np.array(split.sizes())[:, None]
    assert np.abs(share - 0.1).max() < 0.03


def test_dirichlet_share_variance_matches_the_definition():
    # A class's shares over K clients ~ Dirichlet(alpha 1_K): var = (1/K)(1 - 1/K) / (K alpha + 1).
    k, n_classes = 10, 200
    labels = _labels(n_classes=n_classes, per_class=2000)
    for alpha in (0.1, 1.0, 10.0):
        counts = partition(labels, k, "dirichlet", alpha=alpha, min_size=0).label_counts(labels)
        share = counts / counts.sum(axis=0, keepdims=True)          # per class, over clients
        expected = (1 / k) * (1 - 1 / k) / (k * alpha + 1)
        assert share.var() == pytest.approx(expected, rel=0.2), alpha


def test_dirichlet_small_alpha_concentrates_classes():
    labels = _labels(per_class=500)
    def classes_per_client(alpha):
        counts = partition(labels, 10, "dirichlet", alpha=alpha).label_counts(labels)
        return (counts / counts.sum(1, keepdims=True) > 0.05).sum(1).mean()
    assert classes_per_client(0.05) < 3 < 8 < classes_per_client(100.0)


def test_dirichlet_min_size():
    labels = _labels(per_class=50)
    split = partition(labels, 20, "dirichlet", alpha=0.1, min_size=5)
    assert min(split.sizes()) >= 5
    with pytest.raises(ValueError, match="raise alpha"):
        partition(labels, 20, "dirichlet", alpha=0.01, min_size=40, max_tries=20)


def test_shards_give_each_client_its_shards_classes():
    labels = _labels(per_class=100)           # 20 shards of 50: one class each
    split = partition(labels, 10, "shards", shards_per_client=2)
    assert set(split.sizes()) == {100}
    assert ((split.label_counts(labels) > 0).sum(1) <= 2).all()


def test_quantity_sizes_vary_labels_stay_mixed():
    # Client shares ~ Dirichlet(beta 1_K): var = (1/K)(1 - 1/K) / (K beta + 1), over many seeds.
    k, beta = 10, 0.5
    labels = _labels(per_class=2000)
    shares = np.concatenate([np.array(partition(labels, k, "quantity", seed=s, beta=beta).sizes())
                             / len(labels) for s in range(100)])
    assert shares.var() == pytest.approx((1 / k) * (1 - 1 / k) / (k * beta + 1), rel=0.2)
    split = partition(labels, k, "quantity", beta=beta)
    big = int(np.argmax(split.sizes()))
    assert np.abs(split.label_counts(labels)[big] / split.sizes()[big] - 0.1).max() < 0.03


def test_save_load_and_loaders(tmp_path):
    x = torch.randn(1000, 4)
    y = torch.as_tensor(_labels(per_class=100))
    ds = TensorDataset(x, y)
    ds.targets = y
    split = ts.data.partition(ds, 4, "dirichlet", alpha=1.0, seed=7)
    back = ClientSplit.load(split.save(tmp_path / "split.json"))
    assert back.method == "dirichlet" and back.params == {"alpha": 1.0} and back.seed == 7
    assert all(np.array_equal(a, b) for a, b in zip(split.indices, back.indices))
    loaders = split.loaders(ds, batch_size=32)
    assert [sum(len(b[1]) for b in l) for l in loaders] == split.sizes()
    first = lambda: [next(iter(l))[1].tolist() for l in split.loaders(ds, batch_size=32)]
    assert first() == first()                                   # seeded shuffling


def test_labels_of_inputs():
    y = np.array([0, 2, 1])
    assert np.array_equal(labels_of(torch.tensor(y)), y)
    assert np.array_equal(labels_of([(None, 0), (None, 2), (None, 1)]), y)
    with pytest.raises(ValueError):
        labels_of(np.array([0.5, 1.0]))


def test_register_partitioner_and_errors():
    ts.data.register_partitioner("halves", lambda labels, n, rng: np.array_split(np.arange(len(labels)), n))
    assert partition(_labels(), 2, "halves").sizes() == [500, 500]
    with pytest.raises(KeyError, match="available"):
        partition(_labels(), 2, "nope")
    with pytest.raises(ValueError):
        partition(_labels(), 0)


def test_natural_keeps_groups_whole_and_balances():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(12), rng.integers(5, 60, 12))
    labels = np.zeros(len(groups), dtype=np.int64)
    split = partition(labels, 4, "natural", groups=groups, seed=2)
    owner = {}
    for k, idx in enumerate(split.indices):
        for g in np.unique(groups[idx]):
            assert owner.setdefault(g, k) == k                 # a group never spans clients
    sizes = split.sizes()
    assert max(sizes) - min(sizes) <= max(np.bincount(groups))  # greedy: within one group
    one_each = partition(labels, 12, "natural", groups=groups)
    assert sorted(len(np.unique(groups[i])) for i in one_each.indices) == [1] * 12
    assert "groups" not in split.params                        # not saved with the split
    with pytest.raises(ValueError, match="only 12 groups"):
        partition(labels, 13, "natural", groups=groups)


def test_feature_shift_noise_grows_with_the_client_and_is_fixed():
    from torchslicer.data import FeatureShift
    x = torch.zeros(4000)
    shift = FeatureShift("noise", 0.5, seed=1)
    assert torch.equal(shift(x, 0, 5, 7), x)                    # client 0 unshifted
    for k in (1, 4):
        out = shift(x, k, 5, 7)
        assert out.std().item() == pytest.approx(0.5 * k / 4, rel=0.05)
        assert torch.equal(out, shift(x, k, 5, 7))             # same every epoch
    assert not torch.equal(shift(x, 4, 5, 7), shift(x, 4, 5, 8))


def test_feature_shift_rotate_and_loaders():
    from torchslicer.data import FeatureShift
    img = torch.arange(16.).view(1, 4, 4)
    rot = FeatureShift("rotate", 90.0)(img, 1, 2, 0)
    assert torch.allclose(rot, torch.rot90(img, 1, (1, 2)), atol=1e-4) or \
           torch.allclose(rot, torch.rot90(img, -1, (1, 2)), atol=1e-4)
    ds = TensorDataset(torch.zeros(40, 1, 4, 4), torch.as_tensor(_labels(4, 10)))
    split = partition(ds.tensors[1], 2, "iid")
    l0, l1 = split.loaders(ds, 8, feature_shift=FeatureShift("noise", 1.0))
    assert torch.equal(next(iter(l0))[0], torch.zeros(8, 1, 4, 4))
    assert next(iter(l1))[0].abs().sum() > 0
def test_register_dataset_is_used_by_build_data():
    import torch
    from torch.utils.data import TensorDataset
    import torchslicer as ts

    def toy(train, root, download, seed, n=12):
        g = torch.Generator().manual_seed(seed + int(train))
        ds = TensorDataset(torch.randn(n, 3, generator=g), torch.arange(n) % 2)
        ds.targets = torch.arange(n) % 2
        return ds
    ts.data.register_dataset("toy_reg", toy)
    assert "toy_reg" in ts.data.available_datasets()
    cfg = ts.RunConfig()
    cfg.data.dataset, cfg.data.params, cfg.data.batch_size = "toy_reg", {"n": 12}, 4
    cfg.sl.clients = 2
    train, test, split = ts.data.build_data(cfg)
    assert len(train) == 2 and sum(len(i) for i in split.indices) == 12
