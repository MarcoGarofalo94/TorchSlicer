"""
Resolve optimizer and criterion from string shorthands or canonical dicts.

This allows callers to write::

    sliced.train(loader, optimizer="adam", criterion="cross_entropy")

instead of the verbose dict form.
"""

_OPTIMIZER_SHORTHANDS: dict[str, dict] = {
    "adam":    {"name": "Adam",    "params": {"lr": 1e-3}},
    "adamw":   {"name": "AdamW",   "params": {"lr": 1e-4, "weight_decay": 0.01}},
    "sgd":     {"name": "SGD",     "params": {"lr": 0.01, "momentum": 0.9}},
    "rmsprop": {"name": "RMSprop", "params": {"lr": 1e-3}},
}

_CRITERION_SHORTHANDS: dict[str, dict] = {
    "cross_entropy":    {"name": "CrossEntropyLoss",   "params": {}},
    "crossentropy":     {"name": "CrossEntropyLoss",   "params": {}},
    "mse":              {"name": "MSELoss",             "params": {}},
    "bce":              {"name": "BCELoss",             "params": {}},
    "bce_with_logits":  {"name": "BCEWithLogitsLoss",  "params": {}},
    "nll":              {"name": "NLLLoss",             "params": {}},
    "l1":               {"name": "L1Loss",              "params": {}},
}


def resolve_optimizer(optimizer) -> dict | None:
    """Normalise *optimizer* to a canonical ``{"name": ..., "params": {...}}`` dict.

    Accepts:
    - ``None``  → returned as-is (caller decides whether to raise)
    - ``dict``  → returned as-is
    - ``str``   → resolved from the shorthand table
    """
    if optimizer is None or isinstance(optimizer, dict):
        return optimizer
    if isinstance(optimizer, str):
        key = optimizer.lower()
        if key in _OPTIMIZER_SHORTHANDS:
            return _OPTIMIZER_SHORTHANDS[key]
        raise ValueError(
            f"Unknown optimizer shorthand '{optimizer}'. "
            f"Available: {list(_OPTIMIZER_SHORTHANDS)}. "
            "For custom settings pass a dict: "
            "{'name': 'Adam', 'params': {'lr': 3e-4}}."
        )
    raise TypeError(f"optimizer must be a str or dict, got {type(optimizer).__name__}")


def resolve_criterion(criterion) -> dict | None:
    """Normalise *criterion* to a canonical ``{"name": ..., "params": {...}}`` dict.

    Accepts:
    - ``None``  → returned as-is
    - ``dict``  → returned as-is
    - ``str``   → resolved from the shorthand table
    """
    if criterion is None or isinstance(criterion, dict):
        return criterion
    if isinstance(criterion, str):
        key = criterion.lower()
        if key in _CRITERION_SHORTHANDS:
            return _CRITERION_SHORTHANDS[key]
        raise ValueError(
            f"Unknown criterion shorthand '{criterion}'. "
            f"Available: {list(_CRITERION_SHORTHANDS)}. "
            "For custom settings pass a dict: "
            "{'name': 'CrossEntropyLoss', 'params': {}}."
        )
    raise TypeError(f"criterion must be a str or dict, got {type(criterion).__name__}")
