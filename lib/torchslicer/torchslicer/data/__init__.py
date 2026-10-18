"""Data for SL experiments (M11): federated partitioners, standard datasets, and the
config-driven loaders behind ``ts.run(model_fn)`` with a ``data`` section."""
from .build import build_data
from .datasets import ImageTensorDataset, available_datasets, load_dataset
from .partition import (
    ClientSplit,
    available_partitioners,
    labels_of,
    partition,
    register_partitioner,
)

__all__ = ["ClientSplit", "ImageTensorDataset", "available_datasets", "available_partitioners",
           "build_data", "labels_of", "load_dataset", "partition", "register_partitioner"]
