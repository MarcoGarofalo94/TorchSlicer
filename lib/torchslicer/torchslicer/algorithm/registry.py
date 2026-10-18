"""Algorithm registry: name → Algorithm class."""

from .base import Algorithm

_REGISTRY: dict[str, type] = {}


def register(name: str, cls: type) -> None:
    if not (isinstance(cls, type) and issubclass(cls, Algorithm)):
        raise TypeError(f"algorithm class must subclass Algorithm, got {cls!r}")
    _REGISTRY[name] = cls


def get(name: str) -> Algorithm:
    """Return a new instance of the algorithm registered under *name*."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown algorithm {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available() -> list[str]:
    return sorted(_REGISTRY)
