"""``ts.run``: one training script for every environment.

The script defines the model and the data and makes one call::

    import torchslicer as ts
    ts.run(build_model, get_loader, strategy="params")   # any ts.slice option

What the call does depends on the config (``deploy.topology``) and on this node's role
(``deploy.role``, env ``ROLE``):

=============  ==============================================================
topology       roles
=============  ==============================================================
local          ``all``: ``LocalExecutor``, one process, no network
centralized    ``coordinator`` | ``worker`` | ``all`` (+ ``client`` with ``sl.client_nodes``)
p2p            ``driver`` | ``follower`` | ``all``
=============  ==============================================================

``all`` (the default) runs every node in this process: the workers/followers are
started with ``ts.start_worker`` on loopback and the run goes through the real gRPC
(or TCP) stack, so it is a faithful local simulation of the distributed run.
``python -m torchslicer launch`` starts or renders the other roles from the same
config (one OS process per node, compose, k8s, bare metal).
"""

import copy
import os
import socket

from ..config import RunConfig
from ..monitor.process_logger import get_logger

_LOG = get_logger("torchslicer.run")

TOPOLOGIES = ("local", "centralized", "p2p")
ROLES = {
    "local":       ("all",),
    "centralized": ("all", "coordinator", "worker", "client"),
    "p2p":         ("all", "driver", "follower"),
}


def resolve_role(cfg: RunConfig) -> tuple[str, str]:
    """``(topology, role)`` for this node, validated. An empty role means ``all``; in
    P2P, the legacy ``IS_DRIVER`` env var still picks driver/follower."""
    topology = (cfg.deploy.topology or "local").strip().lower()
    if topology not in TOPOLOGIES:
        raise ValueError(f"deploy.topology must be one of {TOPOLOGIES}, got {topology!r}")
    role = (cfg.deploy.role or "").strip().lower()
    if not role and topology == "p2p" and "IS_DRIVER" in os.environ:
        role = "driver" if os.environ["IS_DRIVER"].lower() in ("true", "1") else "follower"
    role = role or "all"
    if role == "client" and not cfg.sl.client_nodes:
        raise ValueError("ROLE=client needs sl.client_nodes (env SL_CLIENT_NODES=true)")
    if cfg.sl.client_nodes and topology != "centralized":
        raise ValueError(f"sl.client_nodes runs on the centralized topology, not {topology!r}")
    if role not in ROLES[topology]:
        raise ValueError(f"role {role!r} is not valid for topology {topology!r} "
                         f"(expected one of {ROLES[topology]})")
    return topology, role


def hosted_slices(cfg: RunConfig, topology: str, n: int) -> int:
    """Slices the driving node hosts in-process (``sl.setting``): 0 for pipeline, 1 for
    vanilla, 2 for u_shaped. The other ``n - hosted`` slices go to workers/followers."""
    from ..sl.roles import Placement, resolve_setting
    if topology == "local":
        return n
    if cfg.sl.client_nodes:
        return 1     # slice 0 is on the client nodes (client_nodes()), not on a worker
    return len(Placement(resolve_setting(cfg.sl.setting, topology), n).client_slices)


def client_nodes(cfg: RunConfig) -> int:
    """Client nodes of the run (``sl.client_nodes``: one per client), else 0."""
    return int(cfg.sl.clients or 1) if cfg.sl.client_nodes else 0


def _config_data(cfg, data_fn, topology: str, drives: bool, evaluates: bool = False):
    """``(data_fn, eval_data)``: with ``data.dataset`` set, TorchSlicer builds the loaders
    (``torchslicer.data.build_data``) and ``data_fn`` must be ``None``. ``drives``: this
    node trains on the data; ``evaluates``: it runs the evaluation (the data owner, or a
    client-node coordinator, which holds the test set but no training data)."""
    if not cfg.data.dataset:
        if data_fn is None and drives:
            raise ValueError("ts.run needs a data_fn, or data.dataset in the config")
        return data_fn, None
    if data_fn is not None:
        raise ValueError("set either data_fn or data.dataset, not both")
    wants_eval = evaluates and bool(cfg.data.eval_every)
    if not (drives or wants_eval):
        return None, None
    from ..data.build import build_data
    train, test, _ = build_data(cfg)
    return ((lambda: train) if drives else None), (test if wants_eval else None)


def run(model_fn, data_fn=None, *, run_config: RunConfig | None = None, role: str | None = None,
        verbose: bool = True, callbacks: list | None = None, **slice_kwargs) -> list:
    """Run this process as its node of the experiment; return the training history on
    the node that drives training (local, coordinator, driver, ``all``), else ``[]``.

    ``model_fn()`` and ``data_fn()`` are only called on the driving node: workers and
    followers receive their slice. ``slice_kwargs`` go to ``ts.slice`` (``strategy``,
    ``cuts``, ``expand``, ``pack``, ``example_input``). ``run_config`` defaults to
    ``RunConfig.load()`` (env ``EXPERIMENT_CONFIG`` + env overrides); ``role``
    overrides ``deploy.role``. The number of slices is ``discovery.n_workers``;
    the data owner (coordinator / driver) hosts ``sl.setting``'s share of them
    in-process (pipeline: none, vanilla: slice 0, u_shaped: slice 0 and the last) and
    workers / followers the rest. With ``sl.clients`` > 1, ``data_fn()`` returns one
    loader per client (a ``ts.ClientSession``). With ``data.dataset`` set there is no
    ``data_fn``: the loaders, the client split and the test set come from the config's
    ``data`` section (``torchslicer.data``).
    """
    from ..monitor import tracer
    from ..monitor.process_logger import configure as configure_process_logging

    tracer.auto_configure_if_env()
    cfg = run_config or RunConfig.load()
    if role is not None:
        cfg = copy.deepcopy(cfg)
        cfg.deploy.role = role
    configure_process_logging(cfg.logging.level)
    if cfg.training.deterministic:     # before model_fn or a loader touches the GPU
        from ..core.determinism import set_deterministic
        set_deterministic()
    topology, role = resolve_role(cfg)
    n = int(cfg.discovery.n_workers)
    if cfg.planning.multi_tensor:
        slice_kwargs.setdefault("multi_tensor", True)
    _LOG.info("run topology=%s role=%s n=%s run_id=%s", topology, role, n, cfg.run_id)
    # who reads the data: the driving node, or with client nodes, each client
    needs_data = role in ("client", "all") or (role not in ("worker", "follower") and not cfg.sl.client_nodes)
    evaluates = role not in ("worker", "follower", "client")
    data_fn, eval_data = _config_data(cfg, data_fn, topology, needs_data, evaluates)

    if topology == "local":
        import torch
        from .. import slice as ts_slice
        from ..executors.worker import resolve_device
        # network.device means the same thing as on a worker ("auto": cuda if present)
        from ..sl.clients import data_for_run
        model = model_fn().to(torch.device(resolve_device(cfg.network.device)))
        sliced = ts_slice(model, n=n, **slice_kwargs)
        return sliced.train(data_for_run(cfg, data_fn), run_config=cfg, verbose=verbose,
                            callbacks=callbacks, eval_data=eval_data)
    if role in ("worker", "follower"):
        return _serve(cfg, topology)
    if role == "client":
        return _client(cfg, data_fn)
    if role == "all":
        return _run_all(cfg, topology, n, model_fn, data_fn, verbose, callbacks, slice_kwargs, eval_data)
    if topology == "centralized":
        return _coordinate(cfg, n, model_fn, data_fn, verbose, callbacks, slice_kwargs, eval_data)
    return _drive(cfg, model_fn, data_fn, verbose, callbacks, slice_kwargs, eval_data)


# ── roles ──────────────────────────────────────────────────────────────────────

def _serve(cfg: RunConfig, topology: str) -> list:
    """A worker / follower: serve a slice until the coordinator or driver shuts it down.
    It registers with the coordinator only for ``discovery.backend: coordinator``."""
    from ..executors.worker import run_worker
    register = topology == "centralized" and cfg.discovery.backend == "coordinator"
    coordinator = cfg.network.coordinator_address if register else None
    if register and not coordinator:
        raise ValueError("a centralized worker with discovery.backend=coordinator needs "
                         "network.coordinator_address (env COORDINATOR_ADDRESS)")
    run_worker(port=cfg.network.worker_port, coordinator_addr=coordinator)
    return []


def _client_loaders(cfg: RunConfig, data_fn) -> list:
    """The clients' loaders: ``data_fn()`` gives one per client (or a ``ClientSession``)."""
    from ..sl.clients import ClientSession
    data = data_fn()
    loaders = list(data.loaders if isinstance(data, ClientSession) else data)
    c = int(cfg.sl.clients or 1)
    if len(loaders) != c:
        raise ValueError(f"sl.clients={c}: data_fn() must return {c} loaders, got {len(loaders)}")
    return loaders


def _client(cfg: RunConfig, data_fn) -> list:
    """A client node (``sl.client_nodes``): slice 0 and its own data, until shut down."""
    from ..executors.worker import run_worker
    k, c = int(cfg.sl.client_index), int(cfg.sl.clients or 1)
    if not 0 <= k < c:
        raise ValueError(f"a client node needs sl.client_index in 0..{c - 1} (env CLIENT_INDEX), got {k}")
    register = cfg.discovery.backend == "coordinator"
    run_worker(port=cfg.network.worker_port,
               coordinator_addr=cfg.network.coordinator_address if register else None,
               data=_client_loaders(cfg, data_fn)[k], client_index=k)
    return []


def _coordinate(cfg, n, model_fn, data_fn, verbose, callbacks, slice_kwargs, eval_data=None) -> list:
    """The centralized coordinator: discovery per ``discovery.backend``, then train."""
    from .. import slice as ts_slice
    from ..discovery import CoordinatorDiscovery, StaticDiscovery
    from ..executors.distributed import DistributedExecutor
    from ..sl.clients import data_for_run

    bind = cfg.network.coordinator_bind_address
    port = int(bind.rsplit(":", 1)[-1])
    remote = n - hosted_slices(cfg, "centralized", n) + client_nodes(cfg)
    if cfg.discovery.backend == "static":
        if len(cfg.discovery.peers) < remote:
            raise ValueError(f"discovery.backend=static needs {remote} discovery.peers "
                             f"(env WORKER_PEERS), got {cfg.discovery.peers}")
        discovery = StaticDiscovery(peers=list(cfg.discovery.peers))
    else:
        discovery = CoordinatorDiscovery(run_id=cfg.run_id)
    # the address workers report to (batch_done, metrics): advertised, else hostname
    address = cfg.network.coordinator_address or f"{socket.gethostname()}:{port}"
    if cfg.sl.client_nodes:
        from ..executors.client_nodes import ClientNodesExecutor
        from ..sl.clients import RemoteSession
        executor = ClientNodesExecutor(discovery=discovery, coordinator_addr=address,
                                       coordinator_bind_addr=bind, run_config=cfg)
        mode = cfg.sl.client_mode or "sequential"
        data = RemoteSession(int(cfg.sl.clients or 1), mode=mode,
                             handoff=cfg.sl.handoff if mode == "sequential" else "copy")
    else:
        executor = DistributedExecutor(discovery=discovery, coordinator_addr=address,
                                       coordinator_bind_addr=bind, run_config=cfg)
        data = data_for_run(cfg, data_fn)
    sliced = ts_slice(model_fn(), n=n, executor=executor, **slice_kwargs)
    _LOG.info("coordinator start run_id=%s\n%s", cfg.run_id, sliced.summary())
    return sliced.train(data, run_config=cfg, verbose=verbose, callbacks=callbacks, eval_data=eval_data)


def _drive(cfg, model_fn, data_fn, verbose, callbacks, slice_kwargs, eval_data=None) -> list:
    """The P2P driver (worker 0 + embedded coordinator)."""
    from ..executors.p2p import run_p2p
    return run_p2p(model_fn, data_fn, is_driver=True, run_config=cfg, verbose=verbose,
                   callbacks=callbacks, eval_data=eval_data, **slice_kwargs)


def _run_all(cfg, topology, n, model_fn, data_fn, verbose, callbacks, slice_kwargs, eval_data=None) -> list:
    """Every node in this process: workers on loopback, then the driving role."""
    from ..executors.worker import start_worker

    cfg = copy.deepcopy(cfg)
    serves = cfg.transport.listen_offsets()               # a worker (and the tail slice)
    drives = cfg.transport.listen_offsets(drives=True)    # the coordinator / driver
    n_workers = n - hosted_slices(cfg, topology, n)
    if (cfg.sl.setting or "").strip().lower().replace("-", "_") in ("u_shaped", "ushaped") \
            and not cfg.sl.tail_port:
        cfg.sl.tail_port = free_port(serves)   # the data owner's last slice
    workers, relay = [], None
    # transport.nat stands for the data owner (the client) here: the workers stay
    # reachable. Any transport.relay is replaced by one on loopback.
    worker_cfg = copy.deepcopy(cfg)
    worker_cfg.transport.nat = False
    try:
        if cfg.transport.relay:
            from ..transport.tunnel import start_relay
            relay = start_relay(host="127.0.0.1")
            cfg.transport.relay = worker_cfg.transport.relay = relay.address
        for _ in range(n_workers):
            port = free_port(serves)
            workers.append(start_worker(port, host="127.0.0.1", run_config=worker_cfg))
        loaders = _client_loaders(cfg, data_fn) if client_nodes(cfg) else []
        for k, loader in enumerate(loaders):
            workers.append(start_worker(free_port(serves), host="127.0.0.1", run_config=worker_cfg,
                                        data=loader, client_index=k))
        peers = [w.address for w in workers]
        cfg.discovery.backend = "static"
        if topology == "centralized":
            cfg.discovery.peers = peers
            cfg.network.coordinator_bind_address = f"127.0.0.1:{free_port(drives)}"
            cfg.network.coordinator_address = cfg.network.coordinator_bind_address
            return _coordinate(cfg, n, model_fn, data_fn, verbose, callbacks, slice_kwargs, eval_data)
        port = free_port(drives)
        cfg.network.worker_port = port
        cfg.network.worker_address = f"127.0.0.1:{port}"
        cfg.discovery.peers = [cfg.network.worker_address, *peers]
        return _drive(cfg, model_fn, data_fn, verbose, callbacks, slice_kwargs, eval_data)
    finally:
        for w in workers:
            w.stop()
        if relay is not None:
            relay.stop()


_PORT_RANGE = (20000, 32000)    # below Linux's ephemeral range (32768-60999)


def free_port(offsets=(0,), host: str = "127.0.0.1") -> int:
    """A port ``p`` with ``p + o`` free for every ``o`` in ``offsets`` (a node's gRPC port,
    its tensor listener, the NCCL store: ``TransportConfig.listen_offsets``; an int ``k``
    means ``(0, k)``). Drawn below the ephemeral range, so the outgoing connections the
    OS opens meanwhile can't take it between this check and the bind."""
    import random
    offsets = (0, offsets) if isinstance(offsets, int) else tuple(offsets)
    rng = random.SystemRandom()
    for _ in range(200):
        port = rng.randrange(_PORT_RANGE[0], _PORT_RANGE[1] - max(offsets))
        try:
            for o in offsets:
                with socket.socket() as s:
                    s.bind((host, port + o))
            return port
        except OSError:
            continue
    raise RuntimeError(f"no free port with offsets {offsets} in {_PORT_RANGE}")
