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
import threading

import grpc
import torch

from ..algorithm.base import overridden
from ..core.tied import find_tied_parameters
from ..sl.clients import ClientBatch, RemoteSession
from ..transport.grpc.worker import worker_service_pb2
from ..monitor.process_logger import get_logger
from .distributed import DistributedExecutor, build_tied_configs, _build_layer_configs

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
        if cfg.fault_tolerance.enabled:
            raise ValueError("sl.client_nodes doesn't support fault_tolerance.enabled yet")
        self._drop = bool(cfg.sl.drop_failed_clients)
        self._lost: set = set()       # client nodes that died and left the run
        self._client_info: dict = {}
        self._inflight = None         # the client of the batch in flight
        self._abandoned = False       # its node died: the batch won't complete
        self._seq = 0                 # batches sent: batch ids stay unique whatever is skipped
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
        """``n - 1`` server workers plus ``n_clients`` client nodes. Returns the slice
        chain ``[client 0, server 1, ..., server n-1]``; every client is in ``_clients``."""
        nodes = self._discover_remote(n - 1 + self.n_clients)
        clients, servers = {}, []
        for nd in nodes:
            tags = list(nd.tags) or self._describe_tags(nd.address)
            if "client" not in tags:
                servers.append(nd)
                continue
            k = next((int(t.split("=", 1)[1]) for t in tags if t.startswith("client=")), None)
            if k is None or not 0 <= k < self.n_clients or k in clients:
                raise ValueError(f"client node {nd.address}: client index {k} is missing, out of "
                                 f"range for sl.clients={self.n_clients}, or taken")
            clients[k] = nd
        if len(clients) != self.n_clients or len(servers) != n - 1:
            raise ValueError(f"sl.client_nodes with {n} slices and {self.n_clients} clients needs "
                             f"{self.n_clients} client nodes and {n - 1} server workers; got "
                             f"{len(clients)} and {len(servers)}")
        self._client_nodes = [clients[k] for k in range(self.n_clients)]
        self._clients = [self._make_proxy(0, nd) for nd in self._client_nodes]
        cfg = self._run_config
        for proxy in self._clients[1:]:
            proxy.connect(self._router, tensor_transport=cfg.transport.tensor,
                          tensor_port_offset=cfg.transport.tensor_port_offset)
        return [self._client_nodes[0], *servers]

    def _describe_tags(self, address: str) -> list:
        """A node's declared tags from the node itself (static discovery has none)."""
        from ..transport.grpc.worker import worker_service_pb2_grpc
        self._router.wait_ready(address, float(self._run_config.discovery.timeout))
        stub = worker_service_pb2_grpc.WorkerServiceStub(self._router.channel(address))
        return list(stub.describe(worker_service_pb2.DescribeRequest(), timeout=120.0).tags)

    def _make_proxy(self, i: int, node):
        if i == 0 and self._clients and node is self._client_nodes[0]:
            return self._clients[0]  # client 0 is slice 0 of the chain
        return super()._make_proxy(i, node)

    def _open_tunnels(self) -> None:
        extra = self._clients[1:]
        self._proxies.extend(extra)  # the base opens tunnels to every proxy
        try:
            super()._open_tunnels()
        finally:
            del self._proxies[len(self._proxies) - len(extra):]

    def _probe_nodes(self) -> None:
        super()._probe_nodes()
        timeout = float(self._run_config.discovery.timeout)
        for proxy in self._clients[1:]:
            self._router.wait_ready(proxy.address, timeout)

    def _order_nodes(self) -> None:
        """Slice 0 is the clients' (client 0 stands for them in the plan); the server
        keeps discovery order."""

    def _start_heartbeats(self, interval: float, timeout: float) -> None:
        super()._start_heartbeats(interval, timeout)
        n = len(self._proxies)
        for j, proxy in enumerate(self._clients[1:], start=1):
            t = threading.Thread(target=self._heartbeat_loop, args=(n + j - 1, proxy, interval, timeout),
                                 daemon=True, name=f"heartbeat-client-{j}")
            self._heartbeat_threads.append(t)
            t.start()

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
        clients = [p.address for p in self._clients]
        last = self._proxies[-1].address
        import uuid
        token = uuid.uuid4().hex
        for i, partition in enumerate(partitions):
            targets = self._clients if i == 0 else [self._proxies[i]]
            for proxy in targets:
                slice_cfg = self._slice_config(i, n, parts[i], partition, tied[i], token)
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
            predecessors=preds, tied=tied, init_token=token, cut_hooks=hook_buf.getvalue(), n_slices=n)

    def _log_plan(self, model_graph, partitions) -> None:
        super()._log_plan(model_graph, partitions)
        _LOG.info("client nodes count=%s addresses=%s", len(self._clients),
                  [p.address for p in self._clients])

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
        data_loader.round_deadline_s = float(self._run_config.sl.round_deadline_s or 0.0)
        data_loader.bind(_ClientNodes(self))
        if len(self._lost) == self.n_clients:
            raise RuntimeError("every client node has left the run")
        out = super().train_epoch(data_loader, epoch, verbose, total_epochs)
        out["bytes_state"] = self._state_bytes
        last = data_loader.last_round
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
        for k, proxy in enumerate(self._clients[1:], start=1):
            if k in self._lost:
                continue
            try:
                proxy.stub().shutdown(worker_service_pb2.ShutdownRequest(run_id=self._run_id))
            except Exception as e:
                _LOG.warning("shutdown failed client=%s error=%s", proxy.address, e)
        super().teardown()
