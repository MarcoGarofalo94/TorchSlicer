from abc import ABC, abstractmethod


class BaseExecutor(ABC):
    @abstractmethod
    def setup(self, model_graph, partitions, optimizer_cfg: dict, criterion_cfg: dict) -> None:
        ...

    @abstractmethod
    def train_epoch(self, data_loader) -> dict:
        ...

    @abstractmethod
    def teardown(self) -> None:
        ...
