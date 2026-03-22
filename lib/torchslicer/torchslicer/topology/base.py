from abc import ABC, abstractmethod


class BaseTopology(ABC):
    @abstractmethod
    def setup(self, cluster: list) -> None:
        ...

    @abstractmethod
    def teardown(self) -> None:
        ...
