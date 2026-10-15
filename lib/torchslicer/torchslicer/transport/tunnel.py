"""NAT traversal: tunnels, the per-node router and the relay (M6).

A node behind NAT (``transport.nat``) accepts no connections. It dials each node that
has to call it and opens one ``TunnelService.open`` stream there. Calls *to* the NAT'd
node then travel back down that stream. The NAT'd node's own calls go out directly as
usual (outbound connections are allowed), so it needs only outbound connections:

- the gradient for slice 0 (and, U-shaped, the activation for the tail slice) comes back
  on the connection the data owner opened;
- so do the metrics, error reports, heartbeats and tied-weight messages.

Two NAT'd nodes can't reach each other this way. With a relay (``transport.relay``) every
node dials only the relay, which forwards each call by its target address. A relay is
a reachable process that holds no slice: ``python -m torchslicer relay --port 50060``.

Every node has one :class:`Router`. It maps an address to a route: a service hosted in
this process, an inbound tunnel the NAT'd node opened, the relay, or a plain gRPC
channel. :meth:`Router.channel` returns a drop-in replacement for
``grpc.insecure_channel``, so the generated stubs and every call site stay unchanged. A
tunnel carries the serialized unary calls of every TorchSlicer service, the same bytes
gRPC would send, so what crosses a cut does not change.
"""

import itertools
import queue
import threading
import time
from concurrent import futures

import grpc

from ..monitor.process_logger import get_logger
from .grpc.worker import worker_service_pb2 as _pb, worker_service_pb2_grpc as _pb_grpc

_LOG = get_logger("torchslicer.tunnel")

_MAX_MSG = 256 * 1024 * 1024
GRPC_OPTS = [
    ("grpc.max_send_message_length",    _MAX_MSG),
    ("grpc.max_receive_message_length", _MAX_MSG),
]

Frame = _pb.TunnelFrame
# Internal method: "can you route to `target`?", answered by the router itself (used
# to wait for a node that is reached through the relay).
_RESOLVE = "/torchslicer.tunnel/resolve"
_CODES = {c.value[0]: c for c in grpc.StatusCode}


class TunnelRpcError(grpc.RpcError):
    """A failed call over a tunnel. It has the same ``code()`` and ``details()`` as a
    failed gRPC call, so failure detection treats both alike."""

    def __init__(self, code: grpc.StatusCode, details: str = ""):
        super().__init__(details)
        self._code, self._details = code, details

    def code(self):
        return self._code

    def details(self):
        return self._details

    def __str__(self):
        return f"{self._code.name}: {self._details}"


def _error_of(exc: Exception) -> tuple:
    if isinstance(exc, grpc.RpcError) and callable(getattr(exc, "code", None)):
        code = exc.code()
        details = exc.details() if callable(getattr(exc, "details", None)) else str(exc)
        return code, details or ""
    return grpc.StatusCode.UNKNOWN, f"{type(exc).__name__}: {exc}"


class _Context:
    """The ``grpc.ServicerContext`` a servicer gets for a call that came over a tunnel."""

    def __init__(self, timeout_s: float):
        self._deadline = time.monotonic() + timeout_s if timeout_s else None
        self.code, self.details = None, ""

    def is_active(self):
        return True

    def time_remaining(self):
        return None if self._deadline is None else max(0.0, self._deadline - time.monotonic())

    def invocation_metadata(self):
        return ()

    def peer(self):
        return "tunnel"

    def add_callback(self, callback):
        return False

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details

    def abort(self, code, details):
        raise TunnelRpcError(code, details)


class _Capture:
    """Stands in for a ``grpc.Server`` in ``add_*Servicer_to_server``, to collect the
    generated method handlers (deserializer, servicer method, serializer)."""

    def __init__(self):
        self.handlers: dict = {}

    def add_generic_rpc_handlers(self, handlers):
        pass   # the same handlers also arrive through add_registered_method_handlers

    def add_registered_method_handlers(self, service: str, handlers: dict):
        for name, h in handlers.items():
            self.handlers[f"/{service}/{name}"] = h


# ── tunnel ─────────────────────────────────────────────────────────────────────

class Tunnel:
    """One end of an open ``TunnelService.open`` stream. Calls go both ways; each
    ``REQUEST`` is answered by a ``RESPONSE`` or ``ERROR`` with the same ``call_id``."""

    def __init__(self, router: "Router", name: str, dialed: bool):
        self.router = router
        self.name = name
        self.dialed = dialed               # True on the NAT'd (dialing) side
        self.addresses: list = []          # inbound side: what the far end serves
        self.acked = threading.Event()     # dialing side: the far end answered HELLO
        self.closed = threading.Event()
        self._out: queue.SimpleQueue = queue.SimpleQueue()
        self._pending: dict = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._call = None                  # dialing side: the grpc call, to cancel

    def outgoing(self):
        """The frames to send, for gRPC to consume (ends when the tunnel closes)."""
        while True:
            frame = self._out.get()
            if frame is None:
                return
            yield frame

    def send(self, frame) -> None:
        if not self.closed.is_set():
            self._out.put(frame)

    def read(self, frames) -> None:
        """Handle incoming frames until the stream ends, then close."""
        try:
            for frame in frames:
                self.router._on_frame(self, frame)
        except Exception as exc:
            if not self.closed.is_set():
                _LOG.debug("tunnel %s ended: %s", self.name, exc)
        finally:
            self.close()

    def call(self, target: str, method: str, payload: bytes, timeout: float | None) -> bytes:
        slot = [threading.Event(), None]
        with self._lock:
            if self.closed.is_set():
                raise TunnelRpcError(grpc.StatusCode.UNAVAILABLE, f"tunnel {self.name} is closed")
            call_id = next(self._ids)
            self._pending[call_id] = slot
        self._out.put(Frame(kind=Frame.REQUEST, call_id=call_id, target=target, method=method,
                            payload=payload, timeout_s=timeout or 0.0))
        if not slot[0].wait(timeout):
            with self._lock:
                self._pending.pop(call_id, None)
            raise TunnelRpcError(grpc.StatusCode.DEADLINE_EXCEEDED,
                                 f"{method} to {target} over tunnel {self.name}: no reply in {timeout}s")
        frame = slot[1]
        if frame is None:
            raise TunnelRpcError(grpc.StatusCode.UNAVAILABLE,
                                 f"tunnel {self.name} closed during {method} to {target}")
        if frame.kind == Frame.ERROR:
            raise TunnelRpcError(_CODES.get(frame.code, grpc.StatusCode.UNKNOWN), frame.details)
        return frame.payload

    def _reply(self, frame) -> None:
        with self._lock:
            slot = self._pending.pop(frame.call_id, None)
        if slot is not None:
            slot[1] = frame
            slot[0].set()

    def close(self) -> None:
        with self._lock:
            if self.closed.is_set():
                return
            self.closed.set()
            pending, self._pending = self._pending, {}
        self._out.put(None)
        for slot in pending.values():
            slot[0].set()                  # slot[1] stays None: UNAVAILABLE
        if self._call is not None:
            self._call.cancel()
        self.router._on_close(self)


class _TunnelServicer(_pb_grpc.TunnelServiceServicer):
    def __init__(self, router: "Router"):
        self._router = router

    def open(self, request_iterator, context):
        tunnel = Tunnel(self._router, f"from {context.peer()}", dialed=False)
        context.add_callback(tunnel.close)
        threading.Thread(target=tunnel.read, args=(request_iterator,), daemon=True,
                         name="tunnel-in").start()
        return tunnel.outgoing()


class _Keeper:
    """Keeps one tunnel open from this node to ``address``: dials, says HELLO, serves
    the stream, and dials again after it breaks, until released."""

    def __init__(self, router: "Router", address: str):
        self.router = router
        self.address = address
        self.tunnel: Tunnel | None = None
        self.error = ""
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name=f"tunnel-to-{address}").start()

    @property
    def ready(self) -> bool:
        t = self.tunnel
        return t is not None and t.acked.is_set() and not t.closed.is_set()

    def wait(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not self.ready:
            if self._stop.is_set() or time.monotonic() >= deadline:
                return False
            time.sleep(0.02)
        return True

    def stop(self) -> None:
        self._stop.set()
        if self.tunnel is not None:
            self.tunnel.close()

    def _run(self) -> None:
        delay = 0.1
        while not self._stop.is_set():
            channel = grpc.insecure_channel(self.address, options=GRPC_OPTS)
            try:
                grpc.channel_ready_future(channel).result(timeout=5.0)
                tunnel = Tunnel(self.router, f"to {self.address}", dialed=True)
                # set before reading addresses(): a concurrent serve() then either
                # lands in this HELLO or sees the tunnel and announces itself
                self.tunnel = tunnel
                tunnel.send(Frame(kind=Frame.HELLO, addresses=self.router.addresses()))
                tunnel._call = _pb_grpc.TunnelServiceStub(channel).open(tunnel.outgoing())
                if self._stop.is_set():
                    tunnel.close()
                delay = 0.1
                tunnel.read(tunnel._call)            # until the stream ends
                if not self._stop.is_set():
                    _LOG.warning("tunnel to %s closed; dialing again", self.address)
            except Exception as exc:
                self.error = str(exc) or type(exc).__name__
                _LOG.debug("tunnel to %s: %s", self.address, self.error)
            finally:
                channel.close()
            self._stop.wait(delay)
            delay = min(2 * delay, 2.0)


# ── router ─────────────────────────────────────────────────────────────────────

class Router:
    """Where this node's calls go, and what it serves to calls that arrive by tunnel.

    Args:
        relay:       address of a relay. Calls to every address that this node does not
                     host, and that no inbound tunnel announced, go through the relay, and
                     the node announces its own addresses there.
        forwarding:  this router is a relay: it forwards a call for an address it does
                     not host down the tunnel that announced it, else directly.
        keepalive_s: a dialed tunnel sends a PING after this long, so an idle NAT
                     mapping does not expire.
    """

    def __init__(self, relay: str | None = None, *, forwarding: bool = False,
                 keepalive_s: float = 20.0, dispatch_workers: int = 32):
        if relay and relay.strip().lower() == "auto":
            raise ValueError("transport.relay: auto means 'start one for me' (ROLE=all, or the "
                             "launcher's relay node); a node started by hand needs the relay's "
                             "address")
        self.relay = relay or None
        self.forwarding = forwarding
        self.keepalive_s = keepalive_s
        self._dispatch_workers = dispatch_workers
        self._services: dict = {}      # address -> {method: handler}
        self._routes: dict = {}        # address -> inbound Tunnel announcing it
        self._behind_tunnel: set = set()   # addresses ever reached by tunnel: no direct fallback
        self._keepers: dict = {}       # address -> _Keeper (tunnels this node dialed)
        self._channels: dict = {}
        self._calls: dict = {}
        self._lock = threading.Lock()
        self._pool = None
        self._closed = threading.Event()
        self._pinger = None
        self.stats = {"calls_out": 0, "bytes_out": 0, "calls_in": 0, "bytes_in": 0,
                      "forwarded": 0}
        if self.relay:
            self.keep(self.relay)

    # ── what this node serves ──────────────────────────────────────────────────

    def serve(self, address: str, servicer, add_to_server) -> None:
        """Serve ``servicer`` at ``address`` to tunneled calls. ``add_to_server`` is the
        generated ``add_<Service>Servicer_to_server`` function."""
        capture = _Capture()
        add_to_server(servicer, capture)
        with self._lock:
            self._services.setdefault(address, {}).update(capture.handlers)
        self._announce()

    def addresses(self) -> list:
        with self._lock:
            return list(self._services)

    def add_to_server(self, server) -> None:
        """Accept tunnels on ``server`` (every TorchSlicer gRPC server does)."""
        _pb_grpc.add_TunnelServiceServicer_to_server(_TunnelServicer(self), server)

    def _announce(self) -> None:
        hello = Frame(kind=Frame.HELLO, addresses=self.addresses())
        for keeper in list(self._keepers.values()):
            if keeper.tunnel is not None:
                keeper.tunnel.send(hello)

    # ── tunnels this node dials ────────────────────────────────────────────────

    def keep(self, address: str, wait: float | None = None) -> _Keeper:
        """Keep a tunnel open to ``address`` (it calls this node through it). With
        ``wait``, block until it is open, else raise UNAVAILABLE."""
        with self._lock:
            keeper = self._keepers.get(address)
            if keeper is None:
                keeper = self._keepers[address] = _Keeper(self, address)
            if self._pinger is None and self.keepalive_s > 0:
                self._pinger = threading.Thread(target=self._ping_loop, daemon=True,
                                                name="tunnel-keepalive")
                self._pinger.start()
        if wait is not None and not keeper.wait(wait):
            raise TunnelRpcError(grpc.StatusCode.UNAVAILABLE,
                                 f"could not open a tunnel to {address} within {wait:.0f}s"
                                 + (f" ({keeper.error})" if keeper.error else ""))
        return keeper

    def kept(self) -> list:
        return list(self._keepers)

    def release(self, address: str) -> None:
        with self._lock:
            keeper = self._keepers.pop(address, None)
        if keeper is not None:
            keeper.stop()

    def _ping_loop(self) -> None:
        closed = self._closed
        while not closed.wait(self.keepalive_s):
            for keeper in list(self._keepers.values()):
                if keeper.ready:
                    keeper.tunnel.send(Frame(kind=Frame.PING))

    # ── routing ────────────────────────────────────────────────────────────────

    def channel(self, address: str) -> "RoutedChannel":
        """Use in place of ``grpc.insecure_channel(address)``."""
        return RoutedChannel(self, address)

    def _route(self, address: str):
        """"local" | an inbound Tunnel | the relay's Tunnel | None (dial directly)."""
        if address in self._services:
            return "local"
        tunnel = self._routes.get(address)
        if tunnel is not None and not tunnel.closed.is_set():
            return tunnel
        if self.relay and address != self.relay:
            keeper = self._keepers.get(self.relay) or self.keep(self.relay)
            if not keeper.wait(2.0):
                raise TunnelRpcError(grpc.StatusCode.UNAVAILABLE, f"relay {self.relay} unreachable")
            return keeper.tunnel
        if address in self._behind_tunnel:
            raise TunnelRpcError(grpc.StatusCode.UNAVAILABLE,
                                 f"{address} is behind NAT and its tunnel is closed")
        return None

    def is_direct(self, address: str) -> bool:
        """Calls to ``address`` go over a plain connection (so raw TCP can be used)."""
        try:
            return self._route(address) is None
        except TunnelRpcError:
            return False

    def wait_ready(self, address: str, timeout: float) -> None:
        """Block until ``address`` can be called (directly, by tunnel or through the
        relay), else raise ``TimeoutError``."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                route = self._route(address)
            except TunnelRpcError:
                route = "down"
            if route == "local" or (isinstance(route, Tunnel) and not self._via_relay(route)):
                return
            if isinstance(route, Tunnel):
                try:
                    if route.call(address, _RESOLVE, b"", 5.0) == b"1":
                        return
                except TunnelRpcError:
                    pass
            elif route is None:
                try:
                    grpc.channel_ready_future(self._direct(address)).result(
                        timeout=max(0.0, min(0.5, deadline - time.monotonic())))
                    return
                except grpc.FutureTimeoutError:
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{address} not reachable within {timeout:.0f}s")
            time.sleep(0.05)

    def _via_relay(self, tunnel: Tunnel) -> bool:
        keeper = self._keepers.get(self.relay) if self.relay else None
        return keeper is not None and keeper.tunnel is tunnel

    def _can_reach(self, address: str) -> bool:
        try:
            route = self._route(address)
        except TunnelRpcError:
            return False
        if route is not None:
            return True
        if not self.forwarding:
            return False
        try:
            grpc.channel_ready_future(self._direct(address)).result(timeout=1.0)
            return True
        except grpc.FutureTimeoutError:
            return False

    def _direct(self, address: str):
        with self._lock:
            ch = self._channels.get(address)
            if ch is None:
                ch = self._channels[address] = grpc.insecure_channel(address, options=GRPC_OPTS)
            return ch

    def _direct_call(self, address: str, method: str, serializer, deserializer):
        key = (address, method, serializer, deserializer)
        call = self._calls.get(key)
        if call is None:
            call = self._calls[key] = self._direct(address).unary_unary(
                method, request_serializer=serializer, response_deserializer=deserializer)
        return call

    def call_bytes(self, address: str, method: str, payload: bytes, timeout: float | None) -> bytes:
        """One unary call with an already serialized request; returns the serialized
        response."""
        route = self._route(address)
        if route is None:
            return self._direct_call(address, method, None, None)(payload, timeout=timeout)
        if route == "local":
            return self._handle(address, method, payload, timeout)
        self.stats["calls_out"] += 1
        self.stats["bytes_out"] += len(payload)
        return route.call(address, method, payload, timeout)

    # ── incoming frames ────────────────────────────────────────────────────────

    def _on_frame(self, tunnel: Tunnel, frame) -> None:
        kind = frame.kind
        if kind == Frame.REQUEST:
            self.stats["calls_in"] += 1
            self.stats["bytes_in"] += len(frame.payload)
            self._dispatcher().submit(self._serve_frame, tunnel, frame)
        elif kind in (Frame.RESPONSE, Frame.ERROR):
            tunnel._reply(frame)
        elif kind == Frame.HELLO:
            if tunnel.dialed:
                tunnel.acked.set()             # the far end's answer to our HELLO
                return
            with self._lock:
                tunnel.addresses = list(dict.fromkeys([*tunnel.addresses, *frame.addresses]))
                for address in frame.addresses:
                    self._routes[address] = tunnel
                    self._behind_tunnel.add(address)
            _LOG.info("tunnel opened %s serving %s", tunnel.name, list(frame.addresses))
            tunnel.send(Frame(kind=Frame.HELLO, addresses=self.addresses()))

    def _dispatcher(self):
        if self._pool is None:
            with self._lock:
                if self._pool is None:
                    self._pool = futures.ThreadPoolExecutor(
                        max_workers=self._dispatch_workers, thread_name_prefix="tunnel-call")
        return self._pool

    def _serve_frame(self, tunnel: Tunnel, frame) -> None:
        try:
            payload = self._handle(frame.target, frame.method, frame.payload, frame.timeout_s)
            tunnel.send(Frame(kind=Frame.RESPONSE, call_id=frame.call_id, payload=payload))
        except Exception as exc:
            code, details = _error_of(exc)
            tunnel.send(Frame(kind=Frame.ERROR, call_id=frame.call_id, code=code.value[0],
                              details=details))

    def _handle(self, target: str, method: str, payload: bytes, timeout_s: float | None) -> bytes:
        if method == _RESOLVE:
            return b"1" if self._can_reach(target) else b""
        handlers = self._services.get(target)
        if handlers is not None:
            handler = handlers.get(method)
            if handler is None:
                raise TunnelRpcError(grpc.StatusCode.UNIMPLEMENTED, f"{method} not served at {target}")
            context = _Context(timeout_s or 0.0)
            response = handler.unary_unary(handler.request_deserializer(payload), context)
            if context.code not in (None, grpc.StatusCode.OK):
                raise TunnelRpcError(context.code, context.details)
            return handler.response_serializer(response)
        if self.forwarding:
            self.stats["forwarded"] += 1
            return self.call_bytes(target, method, payload, timeout_s or None)
        raise TunnelRpcError(grpc.StatusCode.UNAVAILABLE, f"{target} is not served here")

    def _on_close(self, tunnel: Tunnel) -> None:
        with self._lock:
            for address in tunnel.addresses:
                if self._routes.get(address) is tunnel:
                    del self._routes[address]
        if not tunnel.dialed and tunnel.addresses:
            _LOG.info("tunnel closed %s serving %s", tunnel.name, tunnel.addresses)

    def close(self) -> None:
        """Close every tunnel and channel. The router stays usable: a later call dials
        again (an executor can train more than once)."""
        with self._lock:
            self._closed.set()
            self._closed, self._pinger = threading.Event(), None
            keepers, self._keepers = list(self._keepers.values()), {}
            routes = list(set(self._routes.values()))
            channels, self._channels = list(self._channels.values()), {}
            self._calls = {}
            pool, self._pool = self._pool, None
        for keeper in keepers:
            keeper.stop()
        for tunnel in routes:
            tunnel.close()
        for ch in channels:
            ch.close()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


class RoutedChannel:
    """A ``grpc.Channel`` look-alike for generated stubs: each call is routed when it
    is made (see :meth:`Router._route`), so a tunnel opened later is used."""

    def __init__(self, router: Router, address: str):
        self._router, self.address = router, address

    def unary_unary(self, method, request_serializer=None, response_deserializer=None, **_):
        return _RoutedCall(self._router, self.address, method, request_serializer,
                           response_deserializer)

    def close(self):
        pass


class _RoutedCall:
    def __init__(self, router, address, method, serializer, deserializer):
        self._router, self._address, self._method = router, address, method
        self._ser, self._deser = serializer, deserializer

    def __call__(self, request, timeout=None, metadata=None, credentials=None,
                 wait_for_ready=None, compression=None):
        router = self._router
        route = router._route(self._address)
        if route is None:
            return router._direct_call(self._address, self._method, self._ser, self._deser)(
                request, timeout=timeout, metadata=metadata, wait_for_ready=wait_for_ready)
        payload = self._ser(request) if self._ser else request
        if route == "local":
            out = router._handle(self._address, self._method, payload, timeout)
        else:
            router.stats["calls_out"] += 1
            router.stats["bytes_out"] += len(payload)
            out = route.call(self._address, self._method, payload, timeout)
        return self._deser(out) if self._deser else out


# ── relay ──────────────────────────────────────────────────────────────────────

class RelayHandle:
    """A relay started in this process by :func:`start_relay`."""

    def __init__(self, router: Router, server, address: str):
        self.router, self.server, self.address = router, server, address

    def stop(self, grace: float = 0) -> None:
        self.router.close()
        self.server.stop(grace)


def start_relay(port: int = 0, *, host: str = "[::]", address: str | None = None,
                max_tunnels: int = 64) -> RelayHandle:
    """Start a relay without blocking. Every node configured with ``transport.relay``
    dials it and announces its addresses; the relay forwards each call to the node that
    announced the target address, or directly when no tunnel announced it. Each open
    tunnel holds one server thread, hence ``max_tunnels``."""
    import socket
    router = Router(forwarding=True)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_tunnels + 8), options=GRPC_OPTS)
    router.add_to_server(server)
    port = server.add_insecure_port(f"{host}:{port}")
    server.start()
    if address is None:
        address = (f"127.0.0.1:{port}" if host in ("127.0.0.1", "localhost")
                   else f"{socket.gethostname()}:{port}")
    _LOG.info("relay started address=%s", address)
    return RelayHandle(router, server, address)


def run_relay(port: int = 50060, *, host: str = "[::]", max_tunnels: int = 64) -> None:
    """Run a relay until the process is stopped (``python -m torchslicer relay``)."""
    handle = start_relay(port, host=host, max_tunnels=max_tunnels)
    try:
        handle.server.wait_for_termination()
    finally:
        handle.router.close()
