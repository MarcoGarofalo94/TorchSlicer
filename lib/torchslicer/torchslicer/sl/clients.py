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

    def n_samples(self, k: int) -> int:
        """Client ``k``'s training samples (its loader's dataset size; else its batches)."""
        loader = self.loaders[k]
        try:
            return len(loader.dataset)
        except (AttributeError, TypeError):
            return len(loader)

    def _order(self, clients):
        if self.mode == "sequential":
            for k in clients:
                for batch in self.loaders[k]:
                    yield k, batch
            return
        its = {k: iter(self.loaders[k]) for k in clients}
        live = list(clients)
        while live:
            for k in list(live):
                try:
                    batch = next(its[k])
                except StopIteration:
                    live.remove(k)
                    continue
                yield k, batch

    def batches(self, adapter, epoch: int = 0, clients=None):
        """Yield ``(client, (inputs, labels))`` for one round, swapping client states
        in ``adapter`` (``get_state()`` / ``set_state(state)``) between clients.
        ``clients``: the participating clients this round (default all), in order."""
        if self._initial is None:
            self._initial = adapter.get_state()          # get_state returns a fresh copy
        clients = list(range(self.n_clients)) if clients is None else list(clients)
        for k, batch in self._order(clients):
            if k != self.current:
                self._switch(adapter, k, epoch)
            yield k, batch
        if self.mode == "parallel" and self.current is not None:
            self.states[self.current] = adapter.get_state()
            if self.on_round_end is not None:
                states = [s if s is not None else _copy_state(self._initial) for s in self.states]
                out = self.on_round_end(states, {"epoch": epoch, "clients": clients})
                self.states = list(out if out is not None else states)
            adapter.set_state(self.states[self.current])

    def replace_states(self, states, adapter) -> None:
        """Replace every client's state (e.g. after aggregation) and load the active
        client's new one into the live slices. Assigning ``states`` alone leaves the
        slices on the active client's old state, which the next switch saves over the
        new one."""
        self.states = list(states)
        if self.current is not None and self.states[self.current] is not None:
            adapter.set_state(self.states[self.current])

    def finish(self, adapter) -> None:
        """Store the active client's final state in ``states``."""
        if self.current is not None:
            self.states[self.current] = adapter.get_state()

    def _switch(self, adapter, k: int, epoch: int) -> None:
        if self.current is None and self.states[k] is None:
            self.current = k           # the slices already hold the initial state
            return
        # A switch costs one copy out (get_state) and one in (set_state); neither aliases
        # a stored state, so stored states are never copied again here.
        if self.current is not None:
            self.states[self.current] = adapter.get_state()
        if self.mode == "sequential" and self.current is not None:
            builtin = self.handoff in (_handoff_copy, _handoff_keep)   # these don't mutate it
            initial = self._initial if builtin else _copy_state(self._initial)
            info = {"from": self.current, "to": k, "epoch": epoch, "initial": initial}
            new = self.handoff(self.states[self.current], self.states[k], info)
        else:
            new = self.states[k] if self.states[k] is not None else self._initial
        adapter.set_state(new)
        self.current = k


class RemoteSession(ClientSession):
    """A session whose clients are nodes (``sl.client_nodes``): each client node holds its
    own loader and its own client slice, and this object, on the coordinator, only holds
    the schedule. Same modes, handoffs and round-end hooks as :class:`ClientSession`, and
    the same order of batches, so a run gives the same losses as the in-process session::

        sliced.train(ts.RemoteSession(3, mode="parallel"), run_config=cfg)

    Client states stay on their nodes; they cross the network only when the schedule needs
    them: a sequential handoff (``copy``: the previous client's state to the next client;
    ``keep``: nothing), and the round-end hooks (``on_round_end``, an ``Algorithm``'s
    ``on_round_end`` / ``aggregate``), which get and set the participants' states.
    ``states[k]`` holds the last state fetched from client ``k``; after training, every
    client's final state (read before the nodes shut down).
    """

    def __init__(self, n_clients: int, mode: str = "sequential", handoff="copy", on_round_end=None):
        super().__init__([None] * int(n_clients), mode=mode, handoff=handoff, on_round_end=on_round_end)
        self._nodes = None      # the executor's client-node adapter (get / set / info), from bind()
        self._info = None       # [(n_batches, n_samples)] per client
        self._trained: set = set()

    def bind(self, nodes) -> None:
        """Called by the executor before each round: ``nodes.info(k)`` gives client ``k``'s
        ``(n_batches, n_samples)``; ``nodes.get(k)`` / ``nodes.set(k, state)`` move states."""
        self._nodes = nodes
        self._info = [nodes.info(k) for k in range(self.n_clients)]

    def _bound(self):
        if self._info is None:
            raise RuntimeError("RemoteSession: no client nodes yet (it runs with sl.client_nodes, "
                               "on the coordinator)")
        return self._info

    def __len__(self) -> int:
        return sum(nb for nb, _ in self._bound())

    def n_samples(self, k: int) -> int:
        return self._bound()[k][1]

    def _order(self, clients):
        counts = {k: self._bound()[k][0] for k in clients}
        if self.mode == "sequential":
            for k in clients:
                for i in range(counts[k]):
                    yield k, i
            return
        live, done = list(clients), dict.fromkeys(clients, 0)
        while live:
            for k in list(live):
                if done[k] == counts[k]:
                    live.remove(k)
                    continue
                yield k, done[k]
                done[k] += 1

    def batches(self, adapter=None, epoch: int = 0, clients=None):
        """Yield ``(client, (ClientBatch, None))``: which client runs which of its batches.
        The executor turns each into a ``client_step`` on that node."""
        nodes = self._nodes
        clients = list(range(self.n_clients)) if clients is None else list(clients)
        if self._initial is None and clients:
            self._initial = nodes.get(clients[0])     # every client starts from the same weights
        for k, i in self._order(clients):
            if k != self.current:
                self._switch_remote(k, epoch)
            yield k, (ClientBatch(k, i), None)
        if self.mode == "parallel" and self.on_round_end is not None:
            states = [nodes.get(k) for k in range(self.n_clients)]
            out = self.on_round_end(states, {"epoch": epoch, "clients": clients})
            self.states = list(states)
            if out is not None:
                self.replace_states(out, adapter)

    def _switch_remote(self, k: int, epoch: int) -> None:
        nodes, prev = self._nodes, self.current
        self.current = k
        if self.mode != "sequential" or prev is None:
            self._trained.add(k)
            return
        self._trained.add(prev)
        if self.handoff is _handoff_copy:
            nodes.set(k, nodes.get(prev))
        elif self.handoff is not _handoff_keep:      # a callable: it sees both states
            info = {"from": prev, "to": k, "epoch": epoch, "initial": _copy_state(self._initial)}
            nodes.set(k, self.handoff(nodes.get(prev), nodes.get(k) if k in self._trained else None, info))
        self._trained.add(k)

    def fetch_states(self, clients=None) -> list:
        """Read the clients' states from their nodes into ``states`` (all by default)."""
        for k in (range(self.n_clients) if clients is None else clients):
            self.states[k] = self._nodes.get(k)
        return self.states

    def replace_states(self, states, adapter=None) -> None:
        """Send each client whose state changed (another object than ``states[k]``) its new one."""
        states = list(states)
        for k, st in enumerate(states):
            if st is not None and st is not self.states[k]:
                self._nodes.set(k, st)
        self.states = states

    def finish(self, adapter=None) -> None:
        """Nothing to store: the states stay on the client nodes."""


class ClientBatch:
    """Batch ``index`` of client ``client``'s epoch (a :class:`RemoteSession` item)."""

    def __init__(self, client: int, index: int):
        self.client, self.index = client, index

    @property
    def shape(self):          # what the executor logs as the batch's input shape
        return ("client", self.client, self.index)


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
    """``get_state`` / ``set_state`` over in-process ``SplitLayer``s (by slice index).

    ``get_state`` returns a fresh copy on ``store`` ("cpu", the default, or e.g. "cuda"
    to keep many clients' states in GPU memory); ``set_state`` copies the state in and
    never aliases it (``Optimizer.load_state_dict`` alone keeps tensors that are
    already on the parameter's device), so the caller may keep and reuse it."""

    def __init__(self, layers: dict, store="cpu"):
        self.layers = layers     # {slice_index: SplitLayer}
        self.store = store

    def get_state(self) -> dict:
        out = {}
        for i, sl in self.layers.items():
            out[i] = {
                "model": {k: v.detach().to(self.store, copy=True) for k, v in sl.state_dict().items()},
                "optimizer": _copy_optimizer_state(sl.optimizer.state_dict(), self.store)
                             if sl.optimizer else None,
            }
        return out

    def set_state(self, state: dict) -> None:
        for i, sl in self.layers.items():
            s = state[i]
            sl.load_state_dict(s["model"])
            if sl.optimizer is None:
                continue
            if s.get("optimizer") is not None:
                device = next(iter(sl.parameters())).device
                sl.optimizer.load_state_dict(_copy_optimizer_state(s["optimizer"], device))
            else:
                # no optimizer state (e.g. after fedavg): start fresh, rather than keep
                # the state of whichever client ran last
                sl.optimizer.state.clear()


def _copy_optimizer_state(osd: dict, device) -> dict:
    """A copy of an optimizer ``state_dict`` with its tensors copied to ``device``,
    except ``step`` counters, which stay where they are (Adam keeps them on the CPU)."""
    import torch
    def leaf(key, v):
        if isinstance(v, torch.Tensor):
            return v.detach().to(v.device if key == "step" else device, copy=True)
        return copy.deepcopy(v)
    return {"state": {pid: {k: leaf(k, v) for k, v in st.items()} for pid, st in osd["state"].items()},
            "param_groups": copy.deepcopy(osd["param_groups"])}


# ── where client states live, and whether they fit ────────────────────────────

_OPT_SLOTS = {"SGD": None, "Adam": 2, "AdamW": 2, "Adamax": 2, "NAdam": 2, "RAdam": 2,
              "Adagrad": 1, "Adadelta": 2, "RMSprop": None}
_BUDGET = 0.8      # of the free memory a store may fill


def state_nbytes(layers: dict) -> int:
    """Estimated bytes of one client state over ``layers`` ({slice: SplitLayer}):
    parameters and buffers, plus the optimizer's per-parameter tensors (by type,
    before its first step; 2 per parameter when unknown)."""
    total = 0
    for sl in layers.values():
        sd = sl.state_dict()
        total += sum(v.numel() * v.element_size() for v in sd.values())
        opt = sl.optimizer
        if opt is None:
            continue
        name = type(opt).__name__
        slots = _OPT_SLOTS.get(name, 2)
        if slots is None:              # SGD / RMSprop: one buffer per momentum term
            g = opt.param_groups[0]
            slots = int(bool(g.get("momentum", 0))) + (1 if name == "RMSprop" else 0) \
                + int(name == "RMSprop" and bool(g.get("centered", False)))
        total += slots * sum(p.numel() * p.element_size() for g in opt.param_groups for p in g["params"])
    return total


def available_bytes(device: str) -> int:
    """Free memory for ``device``: GPU free memory, or for the host the smaller of
    the cgroup's remaining limit and ``MemAvailable``."""
    import torch
    if str(device).startswith("cuda"):
        return torch.cuda.mem_get_info(torch.device(device))[0]
    free = []
    try:
        limit = open("/sys/fs/cgroup/memory.max").read().strip()
        if limit != "max":
            free.append(int(limit) - int(open("/sys/fs/cgroup/memory.current").read()))
    except (OSError, ValueError):
        pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                free.append(int(line.split()[1]) * 1024)
    except OSError:
        pass
    return min(free) if free else 1 << 62


def resolve_client_store(kind: str, layers: dict, n_states: int, what: str = "client") -> str:
    """The device for ``n_states`` states over ``layers``: ``kind`` is "cpu", "gpu" /
    "cuda[:i]" or "auto" (the slices' GPU if they fit there, else the host). Raises,
    before training starts, when they don't fit in ``_BUDGET`` of the free memory:
    running out of host memory mid-run can take the whole machine down."""
    import torch
    kind = (kind or "cpu").strip().lower()
    device = next((p.device for sl in layers.values() for p in sl.parameters()), torch.device("cpu"))
    gpu = str(device) if device.type == "cuda" else ("cuda" if torch.cuda.is_available() else None)
    need = state_nbytes(layers) * n_states
    if kind in ("gpu", "cuda") or kind.startswith("cuda:"):
        if gpu is None:
            raise ValueError(f"sl.client_states={kind!r} but CUDA isn't available")
        candidates = [gpu if kind == "gpu" else kind]
    elif kind == "cpu":
        candidates = ["cpu"]
    elif kind == "auto":
        candidates = ([gpu] if device.type == "cuda" else []) + ["cpu"]
    else:
        raise ValueError(f"sl.client_states must be cpu | gpu | cuda[:i] | auto, got {kind!r}")
    tried = []
    for c in candidates:
        free = available_bytes(c)
        if need <= _BUDGET * free:
            return c
        tried.append(f"{c}: {free / 2**30:.1f} GB free")
    per = need / max(n_states, 1)
    raise ValueError(f"{n_states} {what} states need ~{need / 2**30:.2f} GB ({per / 2**20:.1f} MB each), "
                     f"more than {_BUDGET:.0%} of the free memory ({'; '.join(tried)}). Use fewer "
                     f"clients, a smaller client slice, or another sl.client_states (cpu | gpu | auto)")


def select_clients(algorithm, n_clients: int, round: int) -> list:
    """The round's participants from ``algorithm.select_clients``, checked."""
    sel = [int(k) for k in algorithm.select_clients(list(range(n_clients)), round)]
    if not sel:
        raise ValueError(f"{type(algorithm).__name__}.select_clients chose no client in round {round}")
    if len(set(sel)) != len(sel) or not all(0 <= k < n_clients for k in sel):
        raise ValueError(f"{type(algorithm).__name__}.select_clients must return distinct client indices "
                         f"in [0, {n_clients}), got {sel}")
    return sel


def algorithm_round_end(session, algorithm, adapter, clients: list, metrics: dict, round: int,
                        server_states: dict | None = None):
    """``on_round_end`` and ``aggregate`` over the round's participants ``clients``.
    ``on_round_end``'s states go back to those clients; ``aggregate``'s client state is
    the new global model, given to every client (non-participants get it too, as in
    FedAvg). Returns ``aggregate``'s server state (``None``: unchanged)."""
    metrics = {**metrics, "client_samples": [session.n_samples(k) for k in clients]}
    new = algorithm.on_round_end([session.states[k] for k in clients], metrics, round)
    if new is not None:
        new = list(new)
        if len(new) != len(clients):
            raise ValueError(f"{type(algorithm).__name__}.on_round_end returned {len(new)} states "
                             f"for {len(clients)} participating clients")
        states = list(session.states)
        for k, st in zip(clients, new):
            states[k] = st
        session.replace_states(states, adapter)
    servers = [server_states[k] for k in clients if k in server_states] if server_states else []
    agg_client, agg_server = algorithm.aggregate([session.states[k] for k in clients], servers, round)
    if agg_client is not None:
        session.replace_states([agg_client] * session.n_clients, adapter)
    return agg_server

