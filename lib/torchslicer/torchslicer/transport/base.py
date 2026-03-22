from abc import ABC, abstractmethod


class BaseTransport(ABC):
    @abstractmethod
    def send(self, destination: str, data: bytes) -> bytes:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
