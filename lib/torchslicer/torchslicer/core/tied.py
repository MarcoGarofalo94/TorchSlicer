"""Parameters shared between layers (tied weights), e.g. GPT-2 / Llama-3.2-1B / Qwen2.5-0.5B
token embedding ↔ LM head.

A tied group is one ``nn.Parameter`` reachable under several names, possibly in
different partitions.  Training it correctly needs exactly one optimizer update per
step using the *sum* of the gradients of every use:

- the **owner** is the lowest partition index that uses it; only the owner's
  optimizer holds it.  The owner steps last in a batch (backward runs from the last
  partition to the first), so by then every other use has produced its gradient;
- in one process (local executor) all uses share the tensor, so its ``.grad``
  already accumulates every contribution;
- across processes each worker holds a copy: replicas send their gradient to the
  owner before the owner steps, and the owner pushes the updated value back
  before the batch completes (see ``WorkerServicer``).

Names are relative to a ``SplitLayer`` (``layers.<i>.<path>``), which is how both
the local executor and the workers address a partition's parameters.
"""
from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class TiedGroup:
    """One shared parameter: ``members`` maps partition index → its local names."""
    group_id: int
    members: dict = field(default_factory=dict)   # {partition_index: [name, ...]}
    requires_grad: bool = True                    # False: frozen, it never changes

    @property
    def owner(self) -> int:
        return min(self.members)

    @property
    def replicas(self) -> list:
        return sorted(p for p in self.members if p != self.owner)

    @property
    def cross_partition(self) -> bool:
        return len(self.members) > 1


def find_tied_parameters(partitions_layers: list) -> list:
    """Return the ``TiedGroup``s among ``partitions_layers`` (one list of layers per partition).

    Only parameters reachable under more than one name are reported.
    """
    by_param: dict = {}
    grad: dict = {}
    for p_idx, layers in enumerate(partitions_layers):
        container = nn.ModuleDict({"layers": nn.ModuleList(layers)})
        for name, param in container.named_parameters(remove_duplicate=False):
            by_param.setdefault(id(param), []).append((p_idx, name))
            grad[id(param)] = param.requires_grad
    groups = []
    for key, uses in by_param.items():
        if len(uses) < 2:
            continue
        group = TiedGroup(group_id=len(groups), requires_grad=grad[key])
        for p_idx, name in uses:
            group.members.setdefault(p_idx, []).append(name)
        groups.append(group)
    return groups


def non_owned_names(partition_index: int, groups: list) -> set:
    """Local names of the tied parameters that ``partition_index`` uses but does not own."""
    return {
        name for g in groups
        if partition_index in g.members and partition_index != g.owner
        for name in g.members[partition_index]
    }


def optimizer_parameters(module: nn.Module, exclude: set = frozenset()) -> list:
    """Parameters ``module``'s optimizer should hold: trainable ones (all if none are),
    minus the names in ``exclude`` (tied parameters owned by another partition)."""
    params = [p for n, p in module.named_parameters(remove_duplicate=False) if n not in exclude]
    params = list({id(p): p for p in params}.values())
    trainable = [p for p in params if p.requires_grad]
    return trainable if trainable else params


def retie(module: nn.Module, names: list) -> nn.Parameter:
    """Make every name in ``names`` refer to the parameter at ``names[0]``.

    Needed after layers were serialized one by one (each copy got its own tensor).
    Returns the shared parameter.
    """
    shared = module.get_parameter(names[0])
    for name in names[1:]:
        parent_name, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attr, shared)
    return shared
