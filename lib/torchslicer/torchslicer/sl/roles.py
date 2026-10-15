"""Split-learning settings: which slices the data owner hosts, and what leaves it.

The *data owner* is the node that holds the data: the process that calls
``data_fn`` (``ts.run``) or iterates the loader. It is the P2P driver, the
centralized coordinator, or the one process of a local run. A setting says which
slices it hosts in-process (so what it hosts never crosses the network) and,
therefore, what leaves it:

============  =======================  ===========  ============  =====================
setting       data owner hosts         raw inputs   labels        typical name
============  =======================  ===========  ============  =====================
pipeline      nothing                  leave        leave         pipeline parallelism
vanilla       slice 0                  stay         leave         split learning (label sharing)
u_shaped      slice 0 and the last     stay         stay          U-shaped split learning
============  =======================  ===========  ============  =====================

The default is ``pipeline`` for the centralized topology (its coordinator holds no
layers, as before M4) and ``vanilla`` elsewhere (a P2P driver always holds slice 0).
"""

from dataclasses import dataclass

SETTINGS = ("pipeline", "vanilla", "u_shaped")


def resolve_setting(setting: str | None, topology: str = "local") -> str:
    s = (setting or "").strip().lower().replace("-", "_")
    if s == "ushaped":
        s = "u_shaped"
    if not s:
        return "pipeline" if topology == "centralized" else "vanilla"
    if s not in SETTINGS:
        raise ValueError(f"sl.setting must be one of {SETTINGS}, got {setting!r}")
    if s == "pipeline" and topology == "p2p":
        raise ValueError("sl.setting=pipeline is not possible in P2P: the driver holds the "
                         "data and always hosts slice 0 (use vanilla or u_shaped)")
    return s


@dataclass(frozen=True)
class Placement:
    """Where each slice runs, from the data owner's point of view."""
    setting:  str
    n_slices: int

    def __post_init__(self):
        if self.setting not in SETTINGS:
            raise ValueError(f"unknown setting {self.setting!r}")
        if self.setting == "u_shaped" and self.n_slices < 3:
            raise ValueError("u_shaped needs at least 3 slices (client front, server, client tail), "
                             f"got {self.n_slices}")

    @property
    def client_slices(self) -> tuple:
        """Slices hosted by the data owner."""
        if self.setting == "pipeline":
            return ()
        if self.setting == "vanilla":
            return (0,)
        return (0, self.n_slices - 1)

    @property
    def inputs_leave(self) -> bool:
        return self.setting == "pipeline"

    @property
    def labels_leave(self) -> bool:
        return self.setting != "u_shaped"

    def is_client(self, i: int) -> bool:
        return i in self.client_slices

    def cut_crosses_owner(self, cut: int) -> bool:
        """Cut ``cut`` joins a data-owner slice and a server slice."""
        return self.is_client(cut) != self.is_client(cut + 1)
