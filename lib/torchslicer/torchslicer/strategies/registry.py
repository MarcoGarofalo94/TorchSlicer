from .base import BaseSplitter
from .uniform import UniformSplitter
from .cost import ParamSplitter, FlopsSplitter, ProfiledSplitter, MemorySplitter

_REGISTRY: dict = {
    "uniform": UniformSplitter,
    "params": ParamSplitter,
    "flops": FlopsSplitter,
    "profiled": ProfiledSplitter,
    "memory": MemorySplitter,
}


def register(name: str, cls: type) -> None:
    _REGISTRY[name] = cls


def available() -> list:
    return list(_REGISTRY)


def get(name_or_instance) -> BaseSplitter:
    if isinstance(name_or_instance, BaseSplitter):
        return name_or_instance
    if isinstance(name_or_instance, type) and issubclass(name_or_instance, BaseSplitter):
        return name_or_instance()
    if isinstance(name_or_instance, str):
        if name_or_instance not in _REGISTRY:
            raise KeyError(f"Unknown strategy '{name_or_instance}'. Available: {list(_REGISTRY)}")
        return _REGISTRY[name_or_instance]()
    raise TypeError(f"Expected str, BaseSplitter subclass, or instance; got {type(name_or_instance)}")
