"""Multi-client sessions: several data owners (clients) share the server slices.

Each client has its own data and its own copy of the client-side slices (the slices
the data owner hosts: slice 0, plus the last one for U-shaped). The server slices are
shared. The executor holds one copy of the client slices and swaps the clients'
states in and out between batches, so a session runs on every executor that hosts the
client slices in-process (local; P2P and centralized with ``sl.setting`` vanilla or
u_shaped)::

    session = ts.ClientSession([loader_a, loader_b, loader_c], mode="sequential")
    sliced.train(session, run_config=cfg)
    session.states[1]     # client 1's client-side state after training

Modes:

- ``sequential``: client 0 trains on all its batches, then client 1, ... (one round
  = one epoch). Moving from client *a* to client *b* calls ``handoff``:
  ``"copy"`` (default) gives *b* the state *a* ended with (weights and optimizer
  state: the client model travels from client to client, as in multi-client
  SplitNN); ``"keep"`` gives *b* its own last state (or the initial one), so only
  the server is shared; a callable ``handoff(prev_state, next_state, info) -> state``
  does anything else (``next_state`` is ``None`` on a client's first turn).
- ``parallel``: the clients advance together, one batch each in turn (client 0's
  first batch, client 1's first batch, ...), each with its own client-side state; a
  client whose loader is exhausted drops out of the round. The server consumes the
  batches one at a time, in that order, and updates after each (no server-side
  averaging or batching across clients). At the end of every round
  ``on_round_end(states, info) -> states`` may combine the clients' states (e.g.
  FedAvg for SplitFed); TorchSlicer itself never averages.

A state is ``{slice_index: {"model": state_dict, "optimizer": state_dict | None}}``
(CPU copies).
"""

import copy


def _copy_state(state):
    return copy.deepcopy(state)


def _handoff_copy(prev_state, next_state, info):
    return prev_state


def _handoff_keep(prev_state, next_state, info):
    return next_state if next_state is not None else info["initial"]


HANDOFFS = {"copy": _handoff_copy, "keep": _handoff_keep}
MODES = ("sequential", "parallel")


class ClientSession:
    def __init__(self, loaders, mode: str = "sequential", handoff="copy", on_round_end=None):
        self.loaders = list(loaders)
        if not self.loaders:
            raise ValueError("ClientSession needs at least one client loader")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if isinstance(handoff, str):
            if handoff not in HANDOFFS:
                raise ValueError(f"handoff must be one of {sorted(HANDOFFS)} or a callable")
            handoff = HANDOFFS[handoff]
        if mode == "parallel" and handoff is not _handoff_copy:
            raise ValueError("handoff applies to sequential sessions; in parallel ones use on_round_end")
        self.mode = mode
        self.handoff = handoff
        self.on_round_end = on_round_end
        self.states: list = [None] * len(self.loaders)
        self.current: int | None = None
        self._initial = None

    @property
    def n_clients(self) -> int:
        return len(self.loaders)

    def __len__(self) -> int:
        return sum(len(l) for l in self.loaders)

    def _order(self):
        if self.mode == "sequential":
            for k, loader in enumerate(self.loaders):
                for batch in loader:
                    yield k, batch
            return
        its = [iter(l) for l in self.loaders]
        live = list(range(len(its)))
        while live:
            for k in list(live):
                try:
                    batch = next(its[k])
                except StopIteration:
                    live.remove(k)
                    continue
                yield k, batch

    def batches(self, adapter, epoch: int = 0):
        """Yield ``(client, (inputs, labels))`` for one round, swapping client states
        in ``adapter`` (``get_state()`` / ``set_state(state)``) between clients."""
        if self._initial is None:
            self._initial = _copy_state(adapter.get_state())
        for k, batch in self._order():
            if k != self.current:
                self._switch(adapter, k, epoch)
            yield k, batch
        if self.mode == "parallel" and self.current is not None:
            self.states[self.current] = _copy_state(adapter.get_state())
            if self.on_round_end is not None:
                states = [s if s is not None else _copy_state(self._initial) for s in self.states]
                out = self.on_round_end(states, {"epoch": epoch})
                self.states = list(out if out is not None else states)
            adapter.set_state(self.states[self.current])

    def replace_states(self, states, adapter) -> None:
        """Replace every client's state (e.g. after aggregation) and load the active
        client's new one into the live slices. Assigning ``states`` alone leaves the
        slices on the active client's old state, which the next switch saves over the
        new one."""
        self.states = list(states)
        if self.current is not None and self.states[self.current] is not None:
            adapter.set_state(_copy_state(self.states[self.current]))

    def finish(self, adapter) -> None:
        """Store the active client's final state in ``states``."""
        if self.current is not None:
            self.states[self.current] = _copy_state(adapter.get_state())

    def _switch(self, adapter, k: int, epoch: int) -> None:
        if self.current is None and self.states[k] is None:
            self.current = k           # the slices already hold the initial state
            return
        if self.current is not None:
            self.states[self.current] = _copy_state(adapter.get_state())
        if self.mode == "sequential" and self.current is not None:
            info = {"from": self.current, "to": k, "epoch": epoch, "initial": _copy_state(self._initial)}
            new = self.handoff(self.states[self.current], self.states[k], info)
        else:
            new = self.states[k] if self.states[k] is not None else self._initial
        adapter.set_state(_copy_state(new))
        self.current = k


def data_for_run(cfg, data_fn):
    """``data_fn()`` for ``ts.run`` / ``ts.run_p2p``: with ``sl.clients`` > 1 it returns
    one loader per client, wrapped in a ``ClientSession`` (``sl.client_mode``,
    ``sl.handoff``)."""
    data = data_fn()
    k = int(cfg.sl.clients or 1)
    if k <= 1 or isinstance(data, ClientSession):
        return data
    loaders = list(data)
    if len(loaders) != k:
        raise ValueError(f"sl.clients={k}: data_fn() must return {k} loaders, got {len(loaders)}")
    mode = cfg.sl.client_mode or "sequential"
    return ClientSession(loaders, mode=mode, handoff=cfg.sl.handoff if mode == "sequential" else "copy")


class SplitLayerClientState:
    """``get_state`` / ``set_state`` over in-process ``SplitLayer``s (by slice index)."""

    def __init__(self, layers: dict):
        self.layers = layers     # {slice_index: SplitLayer}

    def get_state(self) -> dict:
        out = {}
        for i, sl in self.layers.items():
            out[i] = {
                "model": {k: v.detach().cpu().clone() for k, v in sl.state_dict().items()},
                "optimizer": copy.deepcopy(_cpu(sl.optimizer.state_dict())) if sl.optimizer else None,
            }
        return out

    def set_state(self, state: dict) -> None:
        for i, sl in self.layers.items():
            s = state[i]
            sl.load_state_dict(s["model"])
            if sl.optimizer is None:
                continue
            if s.get("optimizer") is not None:
                sl.optimizer.load_state_dict(s["optimizer"])
            else:
                # no optimizer state (e.g. after fedavg): start fresh, rather than keep
                # the state of whichever client ran last
                sl.optimizer.state.clear()


def _cpu(obj):
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu(v) for v in obj]
    return obj
