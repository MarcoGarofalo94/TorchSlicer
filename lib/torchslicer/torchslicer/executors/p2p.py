"""P2P topology: the driver is worker 0 *and* the coordinator, in one process.

There is no coordinator process. The driver holds the data and slice 0, trains its
own slice in-process, and drives the followers (plain workers started with
``ts.run_worker(coordinator_addr=None)`` or ``ts.start_worker()``) exactly like the
centralized coordinator does: it ships each follower its slice with ``init``, sends
labels to the last worker, and waits for ``batch_done``. Activations and gradients go
peer to peer. Raw inputs never leave the driver; labels go only to the last worker.

Everything else is shared with :class:`DistributedExecutor` (servicer, transports,
heartbeats, recovery, run logging). The steps you can replace are documented in
``docs/tutorials/p2p.md``:

- **discovery**: who the followers are (``BaseDiscovery``; default ``StaticDiscovery``
  over ``discovery.peers``);
- **plan agreement**: the driver plans and pushes the plan with ``init``; followers
  accept it (the strategy given to ``ts.slice``);
- **scheduling**: how a batch is fed (``_send_batch``: one batch, or GPipe micro-batches);
- **failure handling**: heartbeats raise ``WorkerFailureError``; ``recover()`` re-plans
  over the survivors. The driver's own slice can't be moved to a follower, since that
  would send raw data off the driver, so its failure ends the run.

Hosting the data owner's slices in-process is shared with the centralized
coordinator (``DistributedExecutor`` with ``sl.setting`` vanilla / u_shaped); in P2P
it is always on.
"""

from ..config import RunConfig
from ..discovery.static import StaticDiscovery
from ..monitor.process_logger import get_logger
from .distributed import DistributedExecutor

_LOG = get_logger("torchslicer.p2p")


class P2PExecutor(DistributedExecutor):
    """Driver of a P2P run (worker 0 + embedded coordinator).

    Args:
        peers:    Follower addresses in slice order. The driver's own ``address``
                  may appear in the list (as in a shared ``discovery.peers``
                  config); it is dropped, since the driver is always worker 0.
                  Defaults to ``run_config.discovery.peers``.
        address:  Address followers use to reach the driver. Defaults to
                  ``network.worker_address``, else the first peer.
        bind_addr: Local bind address (default ``[::]:<port of address>``).
        discovery: A ``BaseDiscovery`` returning the followers, replacing ``peers``.
        run_config: ``RunConfig`` (transport, fault tolerance, logging, ``sl``, ...).
        device:   Device for the driver's own slices (default ``network.device``).

    The driver is the data owner: it hosts slice 0 (``sl.setting`` vanilla, the
    default) or slice 0 and the last slice (``u_shaped``; the last one is served on
    ``sl.tail_port``, default the driver's port + 2), so followers host the others.

    Example::

        ex = ts.P2PExecutor(peers=["w1:50051", "w2:50051"], address="w0:50051",
                            run_config=cfg)
        ts.slice(model, n=3, executor=ex).train(loader, run_config=cfg)
    """

    executor_name = "p2p"
    topology      = "p2p"
    _local_tag    = "driver"
    _owner_name   = "P2P driver"

    def __init__(self, peers=None, *, address: str | None = None, bind_addr: str | None = None,
                 discovery=None, run_config: RunConfig | None = None, device: str | None = None):
        cfg = run_config or RunConfig()
        peers = list(cfg.discovery.peers if peers is None else peers)
        address = address or _advertised(cfg) or (peers[0] if peers else None)
        if not address:
            raise ValueError("P2PExecutor needs the driver's address (address=, "
                             "network.worker_address or the first entry of peers)")
        if discovery is None:
            discovery = StaticDiscovery([p for p in peers if p != address])
        super().__init__(discovery=discovery, coordinator_addr=address,
                         coordinator_bind_addr=bind_addr, run_config=cfg, device=device)
        self.address = address


def _advertised(cfg: RunConfig) -> str | None:
    raw = cfg.network.worker_address
    if not raw:
        return None
    return raw if ":" in raw else f"{raw}:{cfg.network.worker_port}"


def run_p2p(model_fn, data_fn, *, port: int | None = None, is_driver: bool | None = None,
            address: str | None = None, run_config: RunConfig | None = None,
            verbose: bool = True, callbacks: list | None = None, eval_data=None, **slice_kwargs) -> list:
    """Run this process as one node of a P2P run, and return the history on the driver.

    Every node runs the same script; the role comes from ``is_driver``, else
    ``deploy.role`` (env ``ROLE``: driver/follower), else env ``IS_DRIVER``. A
    follower serves its slice until the driver shuts it down and
    returns ``[]``. The driver calls ``model_fn()`` and ``data_fn()`` (followers never
    do: they get only their slice), slices the model over ``discovery.peers`` (env
    ``WORKER_PEERS``, driver first) and trains with ``P2PExecutor``. With
    ``sl.setting: u_shaped`` the driver hosts the last slice too, so the model is cut
    into ``len(followers) + 2`` slices.

    ``slice_kwargs`` go to ``ts.slice`` (``strategy``, ``cuts``, ``expand``, ``pack``,
    ``example_input``). ``run_config`` defaults to ``RunConfig.load()`` (env
    ``EXPERIMENT_CONFIG``)::

        import torchslicer as ts
        ts.run_p2p(build_model, get_loader, strategy="params")
    """
    import os
    import socket
    from .. import slice as ts_slice
    from ..monitor import tracer
    from ..monitor.process_logger import configure as configure_process_logging
    from .worker import run_worker

    tracer.auto_configure_if_env()
    cfg = run_config or RunConfig.load(os.environ.get("EXPERIMENT_CONFIG"))
    configure_process_logging(cfg.logging.level)
    port = int(port or cfg.network.worker_port)
    if is_driver is None:
        role = (cfg.deploy.role or "").strip().lower()   # env ROLE (ts.run, the launcher)
        is_driver = (role == "driver" if role in ("driver", "follower")
                     else os.environ.get("IS_DRIVER", "false").lower() in ("true", "1"))
    address = address or _advertised(cfg) or f"{socket.gethostname()}:{port}"

    if not is_driver:
        run_worker(port=port, coordinator_addr=None, worker_address=address)
        return []

    peers = list(cfg.discovery.peers)
    if not peers:
        raise ValueError("P2P driver: set discovery.peers (env WORKER_PEERS), driver first")
    followers = [p for p in peers if p != address]
    if len(followers) == len(peers):
        followers = peers[1:]   # the driver's entry is spelled differently (e.g. alias)
    executor = P2PExecutor(peers=followers, address=address, bind_addr=f"[::]:{port}",
                           run_config=cfg)
    if cfg.planning.multi_tensor:
        slice_kwargs.setdefault("multi_tensor", True)
    sliced = ts_slice(model_fn(), n=len(followers) + executor.hosted_slices, executor=executor,
                      **slice_kwargs)
    _LOG.info("p2p driver start run_id=%s\n%s", cfg.run_id, sliced.summary())
    from ..sl.clients import data_for_run
    history = sliced.train(data_for_run(cfg, data_fn), run_config=cfg, verbose=verbose,
                           callbacks=callbacks, eval_data=eval_data)
    _LOG.info("p2p driver complete run_id=%s", cfg.run_id)
    return history
