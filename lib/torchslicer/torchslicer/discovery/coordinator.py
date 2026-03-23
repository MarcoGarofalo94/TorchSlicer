"""
CoordinatorDiscovery — registration-based discovery for the centralized topology.

Server side (runs inside DistributedExecutor / _CoordinatorServicer):
    discovery = CoordinatorDiscovery(run_id="my_run")
    nodes = discovery.discover(expected=4, timeout=60)  # blocks until 4 workers register

Client side (runs in worker/main.py at startup):
    result = announce_to_coordinator("coordinator:50054", node_info)
    # result.run_id, result.worker_index
"""

import threading
import datetime
from typing import Callable, List, Optional

from .base import BaseDiscovery, NodeInfo, AnnounceResult


def _generate_run_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class CoordinatorDiscovery(BaseDiscovery):
    """
    Server-side discovery used by DistributedExecutor.

    Workers call the coordinator's Register RPC at startup.
    _CoordinatorServicer delegates to handle_register().
    Once enough workers have registered, discover() unblocks.
    """

    def __init__(self, run_id: str = None):
        self._run_id = run_id or _generate_run_id()
        self._nodes: list[NodeInfo] = []
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._expected = 0
        self._on_join:  Optional[Callable[[NodeInfo], None]] = None
        self._on_leave: Optional[Callable[[NodeInfo], None]] = None

    @property
    def run_id(self) -> str:
        return self._run_id

    def handle_register(self, node_id: str, address: str, device: str, memory_mb: int, run_id: str) -> tuple[bool, int, str]:
        """
        Called by _CoordinatorServicer.register().
        Returns (ok, worker_index, message).
        """
        with self._lock:
            worker_index = len(self._nodes)
            node = NodeInfo(
                node_id=node_id,
                address=address,
                device=device,
                memory_mb=memory_mb,
            )
            self._nodes.append(node)
            print(f"[discovery] registered: {node_id} @ {address} "
                  f"(index={worker_index}, device={device}, mem={memory_mb}MB)")
            if self._on_join:
                self._on_join(node)
            if self._expected > 0 and len(self._nodes) >= self._expected:
                self._ready.set()
        return True, worker_index, "ok"

    def announce(self, node_info: NodeInfo, coordinator_addr: str = None) -> AnnounceResult:
        raise NotImplementedError(
            "CoordinatorDiscovery is server-side only. "
            "Use announce_to_coordinator() in worker processes."
        )

    def discover(self, expected: int, timeout: float = 60.0) -> List[NodeInfo]:
        """Block until `expected` workers have registered via Register RPC."""
        self._expected = expected
        with self._lock:
            if len(self._nodes) >= expected:
                self._ready.set()
        if not self._ready.wait(timeout=timeout):
            with self._lock:
                n = len(self._nodes)
            raise TimeoutError(
                f"Discovery timeout after {timeout}s: "
                f"expected {expected} workers, got {n}"
            )
        with self._lock:
            return list(self._nodes[:expected])

    def watch(
        self,
        on_join:  Callable[[NodeInfo], None],
        on_leave: Callable[[NodeInfo], None],
    ) -> None:
        self._on_join  = on_join
        self._on_leave = on_leave


# ── client-side helper (used by workers) ──────────────────────────────────────

def announce_to_coordinator(
    coordinator_addr: str,
    node_info: NodeInfo,
    run_id: str = "",
    retries: int = 30,
    retry_delay: float = 3.0,
) -> AnnounceResult:
    """
    Client-side: call Register RPC on coordinator. Used by workers at startup.
    Retries until the coordinator is reachable (it may not be up yet).
    """
    import time
    try:
        import grpc
        from torchslicer.transport.grpc.coordinator import (
            coordinator_service_pb2,
            coordinator_service_pb2_grpc,
        )
    except ImportError as e:
        raise ImportError(f"grpcio required for CoordinatorDiscovery: {e}")

    _MAX_MSG = 256 * 1024 * 1024
    _OPTS = [
        ("grpc.max_send_message_length",    _MAX_MSG),
        ("grpc.max_receive_message_length", _MAX_MSG),
    ]

    node_proto = coordinator_service_pb2.NodeInfo(
        node_id=node_info.node_id,
        address=node_info.address,
        device=node_info.device,
        memory_mb=node_info.memory_mb,
    )
    req = coordinator_service_pb2.RegisterRequest(node=node_proto, run_id=run_id)

    last_exc = None
    for attempt in range(retries):
        try:
            channel = grpc.insecure_channel(coordinator_addr, options=_OPTS)
            stub = coordinator_service_pb2_grpc.CoordinatorServiceStub(channel)
            resp = stub.register(req, timeout=5.0)
            channel.close()
            if resp.ok:
                return AnnounceResult(run_id=resp.run_id, worker_index=resp.worker_index)
            raise RuntimeError(f"Coordinator rejected registration: {resp.message}")
        except Exception as exc:
            last_exc = exc
            print(f"[discovery] register attempt {attempt + 1}/{retries} failed: {exc}")
            time.sleep(retry_delay)

    raise RuntimeError(
        f"Could not register with coordinator at {coordinator_addr} "
        f"after {retries} attempts: {last_exc}"
    )
