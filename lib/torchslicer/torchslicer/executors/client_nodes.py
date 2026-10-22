"""Clients as nodes (``sl.client_nodes``, M10): the coordinator of a run whose clients are
separate processes or machines, each with its own data and its own copy of slice 0.

    coordinator  ── client_step(batch_id) ──▶  client k  (slice 0 + its loader)
                                                  │ labels ──────────────▶ last slice
                                                  │ activation ──▶ slice 1 ─▶ ... ─▶ last
    client k  ◀── gradient (batch_id → k) ───── slice 1   (shared server slices)

The server slices are today's worker chain. Slice 1 knows every client's address
(``SliceConfig.prev_workers``) and sends each gradient back to the client of its batch,
``(batch_id // n_micro) % n_clients``: this executor picks batch ids so, and every
transport (gRPC, TCP, IPC, NAT tunnels) carries them unchanged. The coordinator holds no
data and no slice; it runs the schedule (:class:`torchslicer.sl.clients.RemoteSession`),
the client selection and the round-end hooks, moving client states only when they need
to (sequential handoff, aggregation). With per-client server copies (SplitFed V1) every
server worker keeps one copy of its slice per client and swaps it in when that client's
batch arrives (``SliceConfig.per_client_state``). One batch is in flight at a time, so the
losses equal an in-process ``ClientSession`` on the same clients.

Client nodes register with the tags ``client`` and ``client=<k>`` (``ts.start_worker(data=,
client_index=)``, or ``ts.run`` with ``ROLE=client``); the other registered workers are
the server, in discovery order.
"""
from __future__ import annotations

import io
import os
import threading
import time

import grpc
import torch

from ..algorithm.base import overridden
from ..core.tied import find_tied_parameters
from ..sl.clients import ClientBatch, RemoteSession
from ..transport.grpc.worker import worker_service_pb2
from ..monitor.process_logger import get_logger
from ..core.split_layer import SplitLayer
from ..sl.hooks import resolve_cut_hooks
from ..sl.metrics import CommMeter
from ..sl.roles import Placement
from .distributed import DistributedExecutor, build_tied_configs, _build_layer_configs, _PHASES

_LOG = get_logger("torchslicer.client_nodes")
_MAX_BATCH_ID = 2 ** 32


class _ClientNodes:
    """Client states and data sizes over the network (``RemoteSession.bind``)."""

    def __init__(self, executor: "ClientNodesExecutor"):
        self._ex = executor

    def _stub(self, k: int):
        return self._ex._clients[k].stub()

    def lost(self, k: int) -> bool:
        return k in self._ex._lost

    def info(self, k: int) -> tuple:
        """``(n_batches, n_samples)``: asked once; a lost client keeps its last answer."""
        cache = self._ex._client_info
        if not self.lost(k):
            r = self._call(k, "client_info", worker_service_pb2.ClientInfoRequest())
            if r is not None:
                cache[k] = (int(r.n_batches), int(r.n_samples))
        return cache.get(k, (0, 0))

    def _call(self, k: int, method: str, request):
        """A call to client ``k``; ``None`` if its node turns out to be gone (it leaves)."""
        try:
            return getattr(self._stub(k), method)(request, timeout=300.0)
        except grpc.RpcError:
            if not (self._ex._drop and self._ex._gone(k)):
                raise
            self._ex._client_left(k)
            return None

    def get(self, k: int) -> "dict | None":
        """Client ``k``'s state, or ``None`` when its node is gone."""
        blob = self._call(k, "get_state", worker_service_pb2.StateRequest())
        if blob is None:
            return None
        self._ex._state_bytes += len(blob.data)
        return {0: torch.load(io.BytesIO(blob.data), weights_only=True)}

    def set(self, k: int, state: dict) -> None:
        buf = io.BytesIO()
        torch.save(state[0], buf)
        if self._call(k, "set_state", worker_service_pb2.StateBlob(data=buf.getvalue())) is not None:
            self._ex._state_bytes += buf.tell()


class ClientNodesExecutor(DistributedExecutor):
    """The coordinator when clients are nodes (``sl.client_nodes``). Train it with a
    :class:`~torchslicer.sl.clients.RemoteSession` (``ts.run`` builds one from
    ``sl.clients`` / ``sl.client_mode`` / ``sl.handoff``)."""

    executor_name = "client_nodes"
    _owner_name   = "the coordinator (client nodes hold the data)"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self._run_config
        if (cfg.sl.setting or "vanilla") != "vanilla":
            raise ValueError(f"sl.client_nodes runs sl.setting vanilla (client nodes hold slice 0), "
                             f"not {cfg.sl.setting!r}")
        if cfg.transport.tensor == "nccl":
            raise ValueError("sl.client_nodes doesn't support transport.tensor=nccl: clients join "
                             "and leave, and an NCCL process group is fixed at init")
        self._drop = bool(cfg.sl.drop_failed_clients)
        self._lost: set = set()       # client nodes that died and left the run
        self._client_info: dict = {}
        self._inflight = None         # the client of the batch in flight
        self._abandoned = False       # its node died: the batch won't complete
        self._seq = 0                 # batches sent: batch ids stay unique whatever is skipped
        self._rep = None              # the client proxy standing for slice 0 in the chain
        self._joined: list = []       # clients that joined at this round's start
        self._client_ckpt: dict = {}  # recovery: client -> its last epoch checkpoint (on its disk)
        self._session_snap = None     # recovery: the schedule's state at that checkpoint
        self._start_snap = None       # ... and at the run's start (no checkpoint)
        self._setting = "vanilla"    # client nodes host slice 0; this process hosts none
        self.n_clients = int(cfg.sl.clients or 1)
        self._clients: list = []     # client proxies, by client index (client 0 is also _proxies[0])
        self._state_bytes = 0        # client states moved this epoch (handoffs, aggregation)
        self._epoch = 0
        self._session = None         # the RemoteSession trained on: its states are read at teardown
        self._per_client = False     # SplitFed V1: the server workers keep one copy per client

    def setup(self, model_graph, partitions, optimizer_cfg: dict, criterion_cfg: dict,
              mixed_precision: bool = False, n_micro_batches: int = 1, callbacks: list = None,
              model_name: str = "unknown", strategy_name: str = "unknown", run_config=None,
              planner=None, cut_hooks=None, algorithm=None) -> None:
        """As ``DistributedExecutor.setup`` (same signature: ``SlicedModel.train`` passes
        what it names); an algorithm that needs per-client server state
        (``needs_per_client_server_state``, SplitFed V1) makes every server worker keep one
        copy of its slice per client."""
        if algorithm is None and self._run_config.sl.algorithm:
            from ..algorithm.registry import get as _get_algo
            algorithm = _get_algo(self._run_config.sl.algorithm)
        self._per_client = bool(algorithm is not None and algorithm.needs_per_client_server_state())
        if self._per_client and self._run_config.fault_tolerance.enabled:
            raise ValueError("fault_tolerance.enabled with per-client server copies (SplitFed V1) "
                             "isn't supported on client nodes yet: the copies aren't checkpointed")
        super().setup(model_graph, partitions, optimizer_cfg, criterion_cfg,
                      mixed_precision=mixed_precision, n_micro_batches=n_micro_batches,
                      callbacks=callbacks, model_name=model_name, strategy_name=strategy_name,
                      run_config=run_config, planner=planner, cut_hooks=cut_hooks, algorithm=algorithm)

    def _unsupported_hooks(self) -> dict:
        hooks = super()._unsupported_hooks()
        del hooks["needs_per_client_server_state"]      # server workers keep the copies
        return hooks

    def _new_local_servicer(self, first_slice, tags, device=None):
        return None                  # the coordinator hosts no slice

    def hosted_slices(self) -> int:
        return 0

    # ── nodes ────────────────────────────────────────────────────────────────────

    def _discover(self, n: int) -> list:
        """``n - 1`` server workers plus the client nodes up now (all of them, or at least
        ``sl.min_clients``; the others may join at a round's start). Returns the slice chain
        ``[a client, server 1, ..., server n-1]``: the first client present stands for slice
        0 in the plan; every client is in ``_clients`` (``None`` = not there yet)."""
        need = int(self._run_config.sl.min_clients or 0) or self.n_clients
        if not hasattr(self._discovery, "known_nodes"):     # a custom backend: all, as listed
            servers, clients = self._classify(self._discover_remote(n - 1 + self.n_clients),
                                              timeout=float(self._run_config.discovery.timeout))
        else:
            from ..discovery import StaticDiscovery
            fixed = isinstance(self._discovery, StaticDiscovery)     # no node will register later
            deadline = time.monotonic() + float(self._run_config.discovery.timeout)
            while True:
                known = self._discovery.known_nodes()
                servers, clients = self._classify(known, timeout=1.0)
                enough = len(servers) >= n - 1 and len(clients) >= need
                all_up = len(servers) + len(clients) == len(known)
                if enough or (fixed and all_up) or time.monotonic() > deadline:
                    break
                time.sleep(0.5)
        if len(clients) < need or len(servers) < n - 1:
            raise ValueError(f"sl.client_nodes with {n} slices and {self.n_clients} clients needs "
                             f"{need} client nodes and {n - 1} server workers; got "
                             f"{len(clients)} and {len(servers)}")
        servers = servers[:n - 1]
        cfg = self._run_config
        self._clients = [None] * self.n_clients
        for k, nd in clients.items():
            proxy = super()._make_proxy(0, nd)
            proxy.connect(self._router, tensor_transport=cfg.transport.tensor,
                          tensor_port_offset=cfg.transport.tensor_port_offset)
            self._clients[k] = proxy
        self._lost |= {k for k in range(self.n_clients) if self._clients[k] is None}
        if self._lost:
            _LOG.info("starting without clients %s: they may join at a round's start", sorted(self._lost))
        first = min(clients)
        self._rep = self._clients[first]
        return [clients[first], *servers]

    def _classify(self, nodes: list, timeout: float) -> tuple:
        """``(servers, {client index: node})`` among ``nodes``; a node whose tags can't be
        read within ``timeout`` (not up) is left out."""
        clients, servers = {}, []
        for nd in nodes:
            try:
                tags = list(nd.tags) or self._describe_tags(nd.address, timeout)
            except (TimeoutError, grpc.RpcError):
                continue
            if "client" not in tags:
                servers.append(nd)
                continue
            k = next((int(t.split("=", 1)[1]) for t in tags if t.startswith("client=")), None)
            if k is None or not 0 <= k < self.n_clients or (k in clients and clients[k].address != nd.address):
                raise ValueError(f"client node {nd.address}: client index {k} is missing, out of "
                                 f"range for sl.clients={self.n_clients}, or taken")
            clients[k] = nd
        return servers, clients

    def _describe_tags(self, address: str, timeout: float) -> list:
        """A node's declared tags from the node itself (static discovery has none)."""
        from ..transport.grpc.worker import worker_service_pb2_grpc
        self._router.wait_ready(address, timeout)
        stub = worker_service_pb2_grpc.WorkerServiceStub(self._router.channel(address))
        return list(stub.describe(worker_service_pb2.DescribeRequest(), timeout=max(timeout, 5.0)).tags)

    def _make_proxy(self, i: int, node):
        if i == 0 and self._clients:
            return self._rep         # the first client present is slice 0 of the chain
        return super()._make_proxy(i, node)

    def _other_clients(self) -> list:
        """The client proxies besides the one standing for slice 0 in the chain."""
        return [p for p in self._clients if p is not None and p is not self._rep]

    def _open_tunnels(self) -> None:
        extra = self._other_clients()
        self._proxies.extend(extra)  # the base opens tunnels to every proxy
        try:
            super()._open_tunnels()
        finally:
            del self._proxies[len(self._proxies) - len(extra):]

    def _probe_nodes(self) -> None:
        super()._probe_nodes()
        timeout = float(self._run_config.discovery.timeout)
        for proxy in self._other_clients():
            self._router.wait_ready(proxy.address, timeout)

    def _order_nodes(self) -> None:
        """Slice 0 is the clients' (one of them stands for them in the plan); the server
        keeps discovery order."""

    def _start_heartbeats(self, interval: float, timeout: float) -> None:
        super()._start_heartbeats(interval, timeout)
        self._hb = (interval, timeout)
        for proxy in self._other_clients():
            self._watch_client(proxy)

    def _watch_client(self, proxy) -> None:
        k = self._clients.index(proxy)
        interval, timeout = self._hb
        t = threading.Thread(target=self._heartbeat_loop, args=(len(self._proxies) + k, proxy, interval, timeout),
                             daemon=True, name=f"heartbeat-client-{k}")
        self._heartbeat_threads.append(t)
        t.start()

    def _admit(self) -> None:
        """At a round's start: a client node that wasn't there, or that left, and now
        answers, gets slice 0 and joins (from the last broadcast client state, e.g. an
        aggregate, else the initial weights); slice 1 learns its address."""
        missing = sorted(self._lost)
        if not missing or not hasattr(self._discovery, "known_nodes"):
            return
        live = {p.address for p in self._proxies} | {p.address for p in self._other_clients()
                                                     if self._clients.index(p) not in self._lost}
        candidates = [nd for nd in self._discovery.known_nodes() if nd.address not in live]
        for nd in candidates:
            self._router.reset(nd.address)     # dial afresh: it may be a new process there
        _, found = self._classify(candidates, timeout=1.0)
        joined = []
        for k in missing:
            nd = found.get(k)
            if nd is None:
                continue
            try:
                self._join(k, nd)
                joined.append(k)
            except grpc.RpcError as e:
                _LOG.warning("client %s at %s didn't join: %s", k, nd.address, e)
        if joined:
            self._proxies[1].stub().set_peers(worker_service_pb2.PeersRequest(
                prev_workers=[p.address if p is not None and k not in self._lost else ""
                              for k, p in enumerate(self._clients)]), timeout=60.0)
            _LOG.info("clients joined %s", joined)
        self._joined = joined

    def _join(self, k: int, node) -> None:
        cfg = self._run_config
        proxy = super()._make_proxy(0, node)
        proxy.connect(self._router, tensor_transport=cfg.transport.tensor,
                      tensor_port_offset=cfg.transport.tensor_port_offset)
        self._router.wait_ready(proxy.address, 10.0)
        self._init_worker(proxy, self._slice0(), phase="join")
        state = getattr(self._session, "broadcast", None)
        if state is not None:
            buf = io.BytesIO()
            torch.save(state[0], buf)
            proxy.stub().set_state(worker_service_pb2.StateBlob(data=buf.getvalue()), timeout=300.0)
            self._state_bytes += buf.tell()
        old = self._clients[k]
        self._clients[k] = proxy
        if old is self._rep:         # the chain's slice 0 stand-in left: this one takes its place
            self._rep = self._proxies[0] = proxy
        with self._lock:
            self._lost.discard(k)
        self._client_info.pop(k, None)
        if getattr(self, "_hb", None):
            self._watch_client(proxy)

    # ── slices ───────────────────────────────────────────────────────────────────

    def _init_slices(self, layers: list, partitions: list, phase: str,
                     checkpoint_paths: dict | None = None) -> None:
        """Slice 0 to every client node, slices 1.. to the server; slice 1 learns every
        client's address, each client the last slice's (for its labels)."""
        parts = [[layers[j] for j in p.layer_indices] for p in partitions]
        for g in find_tied_parameters(parts):
            if 0 in g.members and len(g.members) > 1 and g.requires_grad:
                raise ValueError("sl.client_nodes: a trainable parameter is tied across the client "
                                 "and the server slices; client nodes can't keep the copies equal")
            if self._per_client and len(g.members) > 1 and g.requires_grad:
                raise ValueError("per-client server copies (SplitFed V1) don't support a trainable "
                                 "parameter tied across server slices")
        n = len(self._proxies)
        tied = build_tied_configs(parts, [p.address for p in self._proxies])
        clients = [p.address if p is not None else "" for p in self._clients]
        last = self._proxies[-1].address
        import uuid
        token = uuid.uuid4().hex
        self._slice0_args = (n, parts[0], partitions[0], tied[0], token, last)
        for i, partition in enumerate(partitions):
            targets = ([(k, p) for k, p in enumerate(self._clients) if p is not None and k not in self._lost]
                       if i == 0 else [(None, self._proxies[i])])
            for k, proxy in targets:
                slice_cfg = self._slice_config(i, n, parts[i], partition, tied[i], token)
                if k is not None:
                    slice_cfg.checkpoint_path = self._client_ckpt.get(k, "")
                if i == 0:
                    slice_cfg.label_worker = last
                if i == 1:
                    slice_cfg.prev_worker = ""
                    slice_cfg.prev_workers.extend(clients)
                if i > 0:
                    slice_cfg.n_clients = self.n_clients
                    slice_cfg.per_client_state = self._per_client
                res = self._init_worker(proxy, slice_cfg, phase=phase)
                _LOG.info("worker %s complete slice=%s address=%s hostname=%s", phase, i,
                          proxy.address, res.hostname)

    def _slice0(self):
        """The ``SliceConfig`` every client node gets (also a client joining later)."""
        n, layers, partition, tied, token, last = self._slice0_args
        slice_cfg = self._slice_config(0, n, layers, partition, tied, token)
        slice_cfg.label_worker = last
        return slice_cfg

    def _slice_config(self, i: int, n: int, partition_layers: list, partition, tied, token: str):
        cfg = self._run_config
        hooks = {c: self._hooks[c] for c in (i - 1, i) if c in self._hooks}
        hook_buf = io.BytesIO()
        if hooks:
            torch.save(hooks, hook_buf)
        preds = [worker_service_pb2.PredecessorList(indices=list(p))
                 for p in (partition.predecessors or [[] for _ in partition_layers])]
        return worker_service_pb2.SliceConfig(
            layers=_build_layer_configs(partition_layers), optimizer=self._stored_opt_cfg,
            criterion=self._stored_crit_cfg if i == n - 1 else None, is_last=(i == n - 1),
            prev_worker=self._proxies[i - 1].address if i > 0 else "",
            next_worker=self._proxies[i + 1].address if i < n - 1 else "",
            coordinator=self.coordinator_addr, n_micro=self._n_micro, run_id=self._run_id,
            worker_index=i, profile_verbosity=cfg.profile.verbosity, profile_memory=cfg.profile.memory,
            predecessors=preds, tied=tied, init_token=token, cut_hooks=hook_buf.getvalue(), n_slices=n,
            deterministic=cfg.training.deterministic)

    def _log_plan(self, model_graph, partitions) -> None:
        super()._log_plan(model_graph, partitions)
        _LOG.info("client nodes count=%s addresses=%s", len(self._clients),
                  [p.address if p is not None else None for p in self._clients])

    # ── training ─────────────────────────────────────────────────────────────────

    def train_epoch(self, data_loader, epoch: int = 0, verbose: bool = False,
                    total_epochs: int = 0) -> dict:
        if not isinstance(data_loader, RemoteSession):
            raise TypeError("sl.client_nodes trains on a ts.RemoteSession (the data is on the "
                            f"client nodes), not {type(data_loader).__name__}")
        if data_loader.n_clients != self.n_clients:
            raise ValueError(f"RemoteSession has {data_loader.n_clients} clients, sl.clients="
                             f"{self.n_clients}")
        self._epoch, self._state_bytes = epoch, 0
        self._session = data_loader
        if self._start_snap is None:            # recovery without a checkpoint replays from here
            self._start_snap = data_loader.snapshot()
        self._joined = []
        self._admit()
        data_loader.round_deadline_s = float(self._run_config.sl.round_deadline_s or 0.0)
        data_loader.bind(_ClientNodes(self))
        if len(self._lost) == self.n_clients:
            raise RuntimeError("every client node has left the run")
        out = super().train_epoch(data_loader, epoch, verbose, total_epochs)
        out["bytes_state"] = self._state_bytes
        last = data_loader.last_round
        if self._joined:
            out["clients_joined"] = self._joined
        if last["lost"]:
            out["clients_lost"] = last["lost"]
        if last["skipped"]:
            out["batches_skipped"] = last["skipped"]
        return out

    def _session_adapter(self, n_clients: int):
        return None                  # states live on the nodes (RemoteSession moves them)

    def _batch_id(self, epoch: int, n_total: int, n_batches: int, client) -> int:
        """``batch_id % n_clients`` is the batch's client: slice 1 routes the gradient by it.
        Counted per batch sent, so an abandoned batch's id is never reused."""
        self._seq += 1
        bid = self._seq * self.n_clients + int(client)
        if bid * self._n_micro >= _MAX_BATCH_ID:
            raise OverflowError(f"batch id {bid} (x {self._n_micro} micro-batches) exceeds 32 bits: "
                                "fewer epochs, batches or clients per run")
        return bid

    def _send_batch(self, batch_id: int, inputs, labels) -> None:
        if not isinstance(inputs, ClientBatch):
            raise TypeError("ClientNodesExecutor sends ClientBatch items (a RemoteSession)")
        k = inputs.client
        with self._lock:
            self._inflight, self._abandoned = k, False
        try:
            self._clients[k].stub().client_step(worker_service_pb2.ClientStepRequest(
                batch_id=batch_id, epoch=self._epoch, index=inputs.index), timeout=120.0)
        except grpc.RpcError:
            if not (self._drop and self._gone(k)):
                raise                    # a live client refused the batch: a real error
            self._client_left(k)
            return
        self._meter.batches += 1

    def _batch_lost(self, batch_id: int) -> bool:
        with self._lock:
            lost, self._abandoned, self._inflight = self._abandoned, False, None
        return lost

    def _gone(self, k: int) -> bool:
        """Whether client ``k``'s node is down: it doesn't answer a liveness probe."""
        try:
            self._clients[k].stub().get_stats(worker_service_pb2.GetStatsRequest(
                run_id=self._run_id, epoch=-1), timeout=max(self._run_config.fault_tolerance.ping_timeout_s, 1.0))
            return False
        except grpc.RpcError:
            return True

    def _client_left(self, k: int) -> None:
        """Client ``k``'s node died: it leaves the run; its batch in flight is abandoned."""
        with self._lock:
            if k in self._lost:
                return
            self._lost.add(k)
            if self._inflight == k:
                self._abandoned = True
                self._batch_done.set()       # unblock the wait for its batch
        _LOG.warning("client node lost client=%s address=%s; the run goes on without it",
                     k, self._clients[k].address)

    def _report_confirmed(self, idx: int) -> bool:
        """Every client node reports as slice 0: a report naming slice 0 holds if a live
        client holds a fatal error (a client's bug stops the run; a dead client is dropped
        by the heartbeats instead)."""
        if idx != 0 or not self._run_config.fault_tolerance.verify_reports or not self._clients:
            return super()._report_confirmed(idx)
        timeout = max(self._run_config.fault_tolerance.ping_timeout_s, 1.0)
        for k, proxy in enumerate(self._clients):
            if k in self._lost:
                continue
            try:
                r = proxy.stub().get_stats(worker_service_pb2.GetStatsRequest(
                    run_id=self._run_id, epoch=-1, verbosity=0), timeout=timeout)
            except grpc.RpcError:
                continue
            if r.fatal_error:
                return True
        return False

    def _on_worker_failure(self, idx: int, proxy) -> None:
        """A dead client node leaves the run (``sl.drop_failed_clients``); a dead server
        worker fails it, as in every distributed run."""
        if self._drop and proxy in self._clients:
            self._client_left(self._clients.index(proxy))
            return
        super()._on_worker_failure(idx, proxy)

    def _round_end(self, session, adapter, selected: list, epoch: int) -> None:
        """Move the participants' states only when an algorithm hook uses them."""
        algo = self._algorithm
        if not (overridden(algo, "on_round_end") or overridden(algo, "aggregate")):
            return
        selected = [k for k in selected if k not in self._lost]      # who finished the round
        if not selected:
            return
        session.fetch_states(selected)
        if not self._per_client:
            super()._round_end(session, adapter, selected, epoch)
            return
        from ..sl.clients import algorithm_round_end
        servers = self._proxies[1:]
        server_states = {k: self._server_copy(servers, k) for k in selected}
        agg = algorithm_round_end(session, algo, adapter, selected, {}, epoch, server_states)
        for i, proxy in enumerate(servers, start=1):    # the round's result, then a new round
            data = b""
            if agg is not None:
                buf = io.BytesIO()
                torch.save(agg[i], buf)
                data = buf.getvalue()
                self._state_bytes += len(data)
            proxy.stub().set_state(worker_service_pb2.StateBlob(data=data, reset_clients=True),
                                   timeout=300.0)

    # ── recovery (fault_tolerance.enabled): the server side ───────────────────────

    def _save_ft_checkpoint(self, epoch: int) -> None:
        """The server workers' checkpoints (as in every distributed run), plus each live
        client's on its own node (``worker_<n + k>``) and the schedule's state."""
        super()._save_ft_checkpoint(epoch)
        n = len(self._proxies)
        for k, proxy in enumerate(self._clients):
            if proxy is None or k in self._lost:
                continue
            try:
                proxy.stub().save_checkpoint(worker_service_pb2.ShutdownRequest(
                    save_checkpoint=True, checkpoint_dir=self._ft_checkpoint_dir, run_id=self._run_id,
                    epoch=epoch, worker_index=n + k), timeout=60.0)
                self._client_ckpt[k] = os.path.join(self._ft_checkpoint_dir, f"worker_{n + k}_epoch_{epoch}.pt")
            except grpc.RpcError as e:
                _LOG.warning("client checkpoint failed client=%s error=%s", k, e)
        if self._session is not None:
            self._session_snap = self._session.snapshot()

    def recover(self, last_good_epoch: int) -> None:
        """A server worker died: re-split the server slices over the surviving server
        workers, from their last checkpoint; slice 0 stays on the clients, each restored
        from its own checkpoint, and the schedule replays the round from its start."""
        self._stop_heartbeats()
        failed = set(self._failed_workers)
        if 0 in failed:
            raise RuntimeError("a client node failed with an error (not a disconnect); a client "
                               f"holds its own data, so it can't move: {self._failure_reason.get(0, '')}")
        alive = [(i, p) for i, p in enumerate(self._proxies) if i > 0 and i not in failed]
        if not alive:
            raise RuntimeError("[recovery] every server worker failed: cannot continue")
        for i in failed:
            if i < len(self._proxies):
                self._router.release(self._proxies[i].address)
        cfg = self._run_config
        all_layers = self._load_server_slices(last_good_epoch) if last_good_epoch > 0 else None
        snap = self._session_snap
        if all_layers is None:       # everything back to the start, so server and clients agree
            _LOG.warning("recovery from the initial weights: no complete checkpoint")
            all_layers = list(self._stored_layers)
            self._client_ckpt.clear()
            snap = self._start_snap
        n = 1 + len(alive)
        partitions = self._server_split(len(alive))
        _LOG.warning("recovery: server slices %s -> %s workers %s", len(self._proxies) - 1, len(alive),
                     [p.address for _, p in alive])
        self._nodes = [self._nodes[0], *(self._nodes[i] for i, _ in alive)]
        self._proxies = [self._proxies[0], *(p for _, p in alive)]
        for i, proxy in enumerate(self._proxies):
            proxy.worker_index = i
        self._hooks = resolve_cut_hooks(self._cut_hooks_spec, n, cfg.sl.cut_hooks, drop_missing=True)
        self._meter = CommMeter(n)
        self._placement = Placement(self._setting, n)
        self.partitions = partitions
        self._log_plan(self._stored_model_graph, partitions)
        self._stored_layers = all_layers
        self._stored_partition_indices = [list(p.layer_indices) for p in partitions]
        if self._session is not None and snap is not None:
            self._session.restore(snap)
        self._init_slices(all_layers, partitions, phase="recovery")
        self._failed_workers.clear()
        self._failure_info.clear()
        self._failure_event.clear()
        self._batch_done.clear()
        ft = cfg.fault_tolerance
        if ft.enabled or ft.detect_failures:
            self._start_heartbeats(ft.heartbeat_interval_s, ft.ping_timeout_s)
        _LOG.info("recovery complete server_workers=%s", len(alive))

    def _load_server_slices(self, epoch: int) -> "list | None":
        """The stored layers with the server slices' weights from ``epoch``'s checkpoints
        (slice 0 is restored on each client from its own)."""
        import copy as _copy
        if not self._ft_checkpoint_dir:
            return None
        layers = list(self._stored_layers)
        for i, indices in enumerate(self._stored_partition_indices):
            if i == 0:
                continue
            path = os.path.join(self._ft_checkpoint_dir, f"worker_{i}_epoch_{epoch}.pt")
            if not os.path.exists(path):
                _LOG.warning("recovery checkpoint missing path=%s", path)
                return None
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            SplitLayer([layers[j] for j in indices]).load_state_dict(ckpt.get("layer_state_dict", {}), strict=False)
        return layers

    def _server_split(self, n_servers: int) -> list:
        """Partition 0 as it is (the clients' slice); the server layers split evenly over
        ``n_servers`` workers."""
        from ..strategies.base import Partition
        from ..strategies.uniform import UniformSplitter
        graph = self._stored_model_graph
        first = list(self._stored_partition_indices[0])
        rest = [j for j in range(len(graph)) if j > first[-1]]
        if len(rest) < n_servers:
            raise RuntimeError(f"[recovery] {len(rest)} server layers can't go on {n_servers} workers")
        groups = [first, *self._uniform_split(rest, n_servers)]
        parts = [Partition(index=i, layer_indices=g,
                           predecessors=UniformSplitter._intra_predecessors(graph, g))
                 for i, g in enumerate(groups)]
        UniformSplitter().validate(graph, parts)
        return parts

    def _server_copy(self, servers: list, k: int) -> dict:
        """Client ``k``'s copy of the server slices, ``{slice: state}``, from every worker."""
        out = {}
        for i, proxy in enumerate(servers, start=1):
            blob = proxy.stub().get_state(worker_service_pb2.StateRequest(per_client=True, client=k),
                                          timeout=300.0)
            self._state_bytes += len(blob.data)
            out[i] = torch.load(io.BytesIO(blob.data), weights_only=True)
        return out

    def _epoch_stats(self, epoch: int) -> list:
        """The chain's stats, with slice 0's summed over the live client nodes."""
        saved = self._proxies
        try:
            self._proxies = saved[1:]
            servers = super()._epoch_stats(epoch)
            self._proxies = [p for k, p in enumerate(self._clients) if k not in self._lost]
            clients = super()._epoch_stats(epoch)
        finally:
            self._proxies = saved
        merged = worker_service_pb2.WorkerStatsResponse()
        for r in clients:
            if r is None:
                continue
            for f in ("bytes_fwd", "bytes_bwd", "raw_bytes_fwd", "raw_bytes_bwd", "msgs_fwd",
                      "msgs_bwd", "bytes_label", "msgs_label"):
                setattr(merged, f, getattr(merged, f) + getattr(r, f))
            merged.compute_ms += r.compute_ms
            for p in _PHASES:                     # profiler phases (profile.verbosity >= 1)
                src, dst = getattr(r, p), getattr(merged, p)
                if src.count:
                    dst.min_ms = min(dst.min_ms, src.min_ms) if dst.count else src.min_ms
                    dst.max_ms = max(dst.max_ms, src.max_ms)
                    dst.p95_ms = max(dst.p95_ms, src.p95_ms)      # an upper bound
                    dst.total_ms += src.total_ms
                    dst.count += src.count
                    dst.avg_ms = dst.total_ms / dst.count
        return [merged, *servers]

    def _add_worker_comm(self, stats: list) -> None:
        super()._add_worker_comm(stats)
        if stats and stats[0] is not None:
            self._meter.add_label(stats[0].bytes_label, stats[0].msgs_label)

    def teardown(self) -> None:
        """Read every client's final state into ``session.states`` (as an in-process
        session ends with them), then shut the nodes down."""
        if self._session is not None and self._clients:
            try:
                self._session.fetch_states()
            except Exception as e:
                _LOG.warning("final client states not read: %s", e)
        for k, proxy in enumerate(self._clients):
            if k in self._lost or proxy is None or proxy is self._rep:   # the base shuts _rep down
                continue
            try:
                proxy.stub().shutdown(worker_service_pb2.ShutdownRequest(run_id=self._run_id))
            except Exception as e:
                _LOG.warning("shutdown failed client=%s error=%s", proxy.address, e)
        super().teardown()
