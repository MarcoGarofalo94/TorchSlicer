from abc import ABC, abstractmethod


class BaseExecutor(ABC):
    @abstractmethod
    def setup(self, model_graph, partitions, optimizer_cfg: dict, criterion_cfg: dict, mixed_precision: bool = False, n_micro_batches: int = 1) -> None:
        ...

    @abstractmethod
    def train_epoch(self, data_loader) -> dict:
        ...

    @abstractmethod
    def teardown(self) -> None:
        ...
