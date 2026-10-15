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
"""

import torch

from ..config import RunConfig
from ..discovery.base import NodeInfo
from ..discovery.static import StaticDiscovery
from ..monitor.process_logger import get_logger
from .distributed import DistributedExecutor, _LocalStub, _CoordinatorServicer
from .worker import WorkerServicer, resolve_device, get_available_memory_mb

try:
    from torchslicer.transport.grpc.worker import worker_service_pb2_grpc
except ImportError:  # grpc missing: DistributedExecutor.__init__ raises a clear error
    pass

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
        run_config: ``RunConfig`` (transport, fault tolerance, logging, ...).
        device:   Device for the driver's own slice (default ``network.device``).

    Example::

        ex = ts.P2PExecutor(peers=["w1:50051", "w2:50051"], address="w0:50051",
                            run_config=cfg)
        ts.slice(model, n=3, executor=ex).train(loader, run_config=cfg)
    """

    executor_name = "p2p"

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
                         coordinator_bind_addr=bind_addr, run_config=cfg)
        self.address = address

        # the driver's own worker: same servicer as every follower, called in-process
        self.local = WorkerServicer()
        self.local.device = torch.device(resolve_device(device or cfg.network.device))
        self.local._tensor_transport   = cfg.transport.tensor.strip().lower()
        self.local._tensor_port_offset = int(cfg.transport.tensor_port_offset)
        self.local._persistent = True   # teardown must not stop the shared gRPC server
        self.local.declared = {
            "compute_gflops": float(cfg.network.worker_compute_gflops),
            "max_memory_mb":  int(cfg.network.worker_max_memory_mb),
            "first_slice":    True,      # it holds the data
            "tags":           ["driver", *cfg.network.worker_tags],
        }
        self.local.local_coordinator = _LocalStub(_CoordinatorServicer(self))

    # ── topology steps ─────────────────────────────────────────────────────────

    def _add_servicers(self, server) -> None:
        # followers reach the driver's slice (gradients back to slice 0) on this server
        worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(self.local, server)

    def _start_server(self, n: int) -> None:
        super()._start_server(n)
        port = int(self.address.rsplit(":", 1)[-1])
        if self.local._tensor_server is None:
            self.local._tensor_stop.clear()
            self.local.start_tensor_server(port + self.local._tensor_port_offset)

    def _discover(self, n: int) -> list:
        followers = super()._discover(n - 1) if n > 1 else []
        dev = str(self.local.device)
        driver = NodeInfo(node_id=self.address, address=self.address, device=dev,
                          memory_mb=get_available_memory_mb(dev), tags=["driver"],
                          first_slice=True)
        return [driver, *followers]

    def _order_nodes(self) -> None:
        # the driver holds the data, so it always keeps slice 0
        for nd in self._nodes[1:]:
            if nd.first_slice:
                _LOG.warning("follower %s declares first_slice; in P2P the driver holds slice 0",
                             nd.address)

    def _make_proxy(self, i: int, node):
        proxy = super()._make_proxy(i, node)
        if i == 0:
            proxy.local = self.local
        return proxy

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def recover(self, last_good_epoch: int) -> None:
        if 0 in self._failed_workers:
            raise RuntimeError(
                "P2P driver's own slice failed "
                f"({self._failure_reason.get(0, 'see the log')}); it holds the data, so its "
                "slice can't move to a follower. The run stops.")
        super().recover(last_good_epoch)

    def teardown(self) -> None:
        super().teardown()
        self.local.stop_tensor_server()


def _advertised(cfg: RunConfig) -> str | None:
    raw = cfg.network.worker_address
    if not raw:
        return None
    return raw if ":" in raw else f"{raw}:{cfg.network.worker_port}"


def run_p2p(model_fn, data_fn, *, port: int | None = None, is_driver: bool | None = None,
            address: str | None = None, run_config: RunConfig | None = None,
            verbose: bool = True, **slice_kwargs) -> list:
    """Run this process as one node of a P2P run, and return the history on the driver.

    Every node runs the same script; the role comes from ``is_driver`` (env
    ``IS_DRIVER``). A follower serves its slice until the driver shuts it down and
    returns ``[]``. The driver calls ``model_fn()`` and ``data_fn()`` (followers never
    do: they get only their slice), slices the model over ``discovery.peers`` (env
    ``WORKER_PEERS``, driver first) and trains with ``P2PExecutor``.

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
        is_driver = os.environ.get("IS_DRIVER", "false").lower() in ("true", "1")
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
    sliced = ts_slice(model_fn(), n=len(followers) + 1, executor=executor, **slice_kwargs)
    _LOG.info("p2p driver start run_id=%s\n%s", cfg.run_id, sliced.summary())
    history = sliced.train(data_fn(), run_config=cfg, verbose=verbose)
    _LOG.info("p2p driver complete run_id=%s", cfg.run_id)
    return history
