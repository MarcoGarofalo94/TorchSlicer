from .base import BaseSplitter
from .uniform import UniformSplitter
from .param_balanced import ParameterBalancedSplitter
from .explicit import ExplicitSplitter

_REGISTRY: dict = {
    "uniform":         UniformSplitter,
    "param_balanced":  ParameterBalancedSplitter,
}


def register(name: str, cls: type = None):
    """Register a custom splitter strategy under *name*.

    Can be used as a plain function call or as a class decorator::

        # Function call
        ts.register_strategy("my_strategy", MyStrategy)

        # Decorator
        @ts.register_strategy("my_strategy")
        class MyStrategy(ts.BaseSplitter):
            ...
    """
    if cls is None:
        # Called as @register("name") — return a decorator
        def decorator(klass):
            _REGISTRY[name] = klass
            return klass
        return decorator
    # Called as register("name", cls)
    _REGISTRY[name] = cls


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
