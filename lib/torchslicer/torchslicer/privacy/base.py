"""Attack and Defence interfaces, their registry and resolution from config.

An :class:`Attack` observes one cut from the side that *receives* it: the activation for a
``"forward"`` attack (the server sees what the client sends), the gradient for a
``"backward"`` attack (the client sees what the server returns). It sees what that side
really gets, after the cut hooks (so noise or quantization defend against it). The
executor also hands it the batch's true inputs and labels, **for scoring only**: an attack
that used them to attack would be cheating, and the built-in ones don't.

A :class:`Defence` runs on the client side of a cut during training and adds a term to the
gradient that side backpropagates (NoPeek's distance-correlation loss). Defences that
change what crosses the cut are cut hooks (``GaussianNoise``, ``Quantize``, ``TopK``).

Both are simulator features: they run on the local executor (and offload, except
attacks that query or copy the client slices); the distributed executors refuse them.
"""

from dataclasses import dataclass

import torch

from ..sl.hooks import CutContext


@dataclass
class Observation:
    tensor: torch.Tensor              # what the receiving side got (detached)
    ctx:    CutContext
    inputs: "torch.Tensor | None"     # the (micro-)batch's raw inputs: scoring only
    labels: "torch.Tensor | None"     # its labels: scoring only


class Attack:
    """Base class. Set ``name`` and ``direction``, override ``observe`` and ``metrics``.

    ``metrics()`` returns the epoch's numbers; the executor prefixes them with
    ``leak_<key>_`` (``key`` defaults to ``name``) and calls ``reset()`` at each epoch start.
    The executor grants more access on request:

    - ``needs_query = True``: ``bind(query)``; ``query(x)`` runs the client slices up to
      the cut on attacker-chosen inputs (eval mode, no gradients, the hooks applied) and
      returns what the receiver would get, i.e. black-box query access.
    - ``needs_client = True``: ``bind_client(client)``; ``client(fresh=False)`` returns a
      copy of the client slices up to the cut as one module (eval mode, parameters frozen,
      gradients flow to its input; cut hooks *not* applied). ``fresh=True`` re-initializes
      its parameters: the architecture only (UnSplit). The copy never touches training.
    - ``forges = True`` (a malicious server, FSHA): before the gradient at the cut is sent
      back, ``forge(obs)`` may return the gradient to send instead (``obs.tensor`` = the
      honest one; return the gradient of a per-(micro-)batch mean loss, the executor
      applies the micro-batch scale) or ``None`` to send the honest one. Unlike the
      others, this changes training, on purpose.
    """
    name = "attack"
    direction = "forward"
    needs_query = False
    needs_client = False
    forges = False

    def __init__(self, cut: int = 0, key: str | None = None):
        self.cut = int(cut)
        self.key = key

    def bind(self, query) -> None:
        self.query = query

    def bind_client(self, client) -> None:
        self.client = client

    def forge(self, obs: "Observation") -> "torch.Tensor | None":
        return None

    def reset(self) -> None:
        pass

    def observe(self, obs: Observation) -> None:
        raise NotImplementedError

    def metrics(self) -> dict:
        return {}


class Defence:
    """Base class. ``grad_term(z, obs)`` returns the extra gradient of the client's loss
    w.r.t. its cut activation ``z`` (``obs.tensor`` is the gradient received), or ``None``."""
    name = "defence"

    def __init__(self, cut: int = 0):
        self.cut = int(cut)

    def grad_term(self, z: torch.Tensor, obs: Observation) -> "torch.Tensor | None":
        return None


_ATTACKS: dict = {}
_DEFENCES: dict = {}


def register_attack(name: str, cls) -> None:
    """Make ``cls`` available to config files as ``privacy.attacks: [{name: ...}]``."""
    _ATTACKS[name] = cls


def register_defence(name: str, cls) -> None:
    """Make ``cls`` available to config files as ``privacy.defences: [{name: ...}]``."""
    _DEFENCES[name] = cls


def _from_config(entries, registry, what):
    out = []
    for e in entries or []:
        name = e.get("name")
        if name not in registry:
            raise ValueError(f"unknown privacy {what} {name!r}; known: {sorted(registry)}")
        out.append(registry[name](cut=e.get("cut", 0), **(e.get("params") or {})))
    return out


def resolve_privacy(spec, cfg_privacy, n_slices: int):
    """``(attacks, defences)`` from ``train(privacy=)`` (an Attack/Defence or a list of
    them) plus the config's ``privacy`` section. Checks the cuts and gives each attack a
    unique metric key."""
    items = [] if spec is None else (list(spec) if isinstance(spec, (list, tuple)) else [spec])
    attacks = _from_config(getattr(cfg_privacy, "attacks", None), _ATTACKS, "attack")
    defences = _from_config(getattr(cfg_privacy, "defences", None), _DEFENCES, "defence")
    for it in items:
        if isinstance(it, Attack):
            attacks.append(it)
        elif isinstance(it, Defence):
            defences.append(it)
        else:
            raise TypeError(f"privacy= takes ts.privacy Attack/Defence objects, got {type(it).__name__}")
    for obj in attacks + defences:
        if not 0 <= obj.cut < n_slices - 1:
            raise ValueError(f"{type(obj).__name__} on cut {obj.cut}, but {n_slices} slices "
                             f"have cuts 0..{n_slices - 2}")
    names = [a.name for a in attacks if a.key is None]
    for a in attacks:
        if a.key is None:
            a.key = a.name if names.count(a.name) == 1 else f"{a.name}_cut{a.cut}"
    keys = [a.key for a in attacks]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        raise ValueError(f"two attacks report under the same key {sorted(dup)}; set key=")
    return attacks, defences
