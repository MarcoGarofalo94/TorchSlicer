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
import time
from typing import Callable, List, Optional

from .base import BaseDiscovery, NodeInfo, AnnounceResult
from ..monitor.process_logger import get_logger
from ..retry import RetryPolicy, call_with_retry

_LOG = get_logger("torchslicer.discovery")


def _generate_run_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class CoordinatorDiscovery(BaseDiscovery):
    """
    Server-side discovery used by DistributedExecutor.

    Workers call the coordinator's Register RPC at startup (and periodically to
    stay visible). handle_register() is idempotent: re-registering by the same
    node_id updates the node's address/device/memory and returns the same index.

    discover() accepts an optional tag_filter: only workers whose tags are a
    superset of tag_filter count toward the expected total.
    """

    def __init__(self, run_id: str = None):
        self._run_id = run_id or _generate_run_id()
        self._nodes: list[NodeInfo] = []
        self._cond = threading.Condition(threading.Lock())
        self._on_join:  Optional[Callable[[NodeInfo], None]] = None
        self._on_leave: Optional[Callable[[NodeInfo], None]] = None

    @property
    def run_id(self) -> str:
        return self._run_id

    def handle_register(
        self,
        node_id:   str,
        address:   str,
        device:    str,
        memory_mb: int,
        run_id:    str = "",
        tags:      list = None,
    ) -> tuple[bool, int, str]:
        """
        Called by _CoordinatorServicer.register().
        Idempotent: if node_id already registered, update its info and return
        the existing index. Returns (ok, worker_index, message).
        """
        tags = tags or []
        with self._cond:
            for i, existing in enumerate(self._nodes):
                if existing.node_id == node_id:
                    existing.address   = address
                    existing.device    = device
                    existing.memory_mb = memory_mb
                    existing.tags      = tags
                    tag_str = f"  tags=[{', '.join(tags)}]" if tags else ""
                    _LOG.info(
                        "worker re-registered node_id=%s address=%s index=%s device=%s mem_mb=%s%s",
                        node_id,
                        address,
                        i,
                        device,
                        memory_mb,
                        tag_str,
                    )
                    self._cond.notify_all()
                    return True, i, "ok"

            # New registration
            worker_index = len(self._nodes)
            node = NodeInfo(
                node_id=node_id,
                address=address,
                device=device,
                memory_mb=memory_mb,
                tags=tags,
            )
            self._nodes.append(node)
            tag_str = f"  tags=[{', '.join(tags)}]" if tags else ""
            _LOG.info(
                "worker registered node_id=%s address=%s index=%s device=%s mem_mb=%s%s",
                node_id,
                address,
                worker_index,
                device,
                memory_mb,
                tag_str,
            )
            if self._on_join:
                self._on_join(node)
            self._cond.notify_all()
        return True, worker_index, "ok"

    def announce(self, node_info: NodeInfo, coordinator_addr: str = None) -> AnnounceResult:
        raise NotImplementedError(
            "CoordinatorDiscovery is server-side only. "
            "Use announce_to_coordinator() in worker processes."
        )

    def discover(
        self,
        expected:    int,
        timeout:     float = 60.0,
        tag_filter:  list  = None,
    ) -> List[NodeInfo]:
        """
        Block until `expected` workers matching all tags in tag_filter have
        registered via Register RPC. Returns NodeInfos in registration order.
        """
        tag_filter = tag_filter or []
        deadline = time.monotonic() + timeout

        self._last_expected   = expected
        self._last_tag_filter = tag_filter

        with self._cond:
            while True:
                matching = self._matching(tag_filter)
                if len(matching) >= expected:
                    return matching[:expected]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    n = len(matching)
                    filter_str = f" with tags {tag_filter}" if tag_filter else ""
                    raise TimeoutError(
                        f"Discovery timeout after {timeout}s: "
                        f"expected {expected} workers{filter_str}, got {n}"
                    )
                self._cond.wait(timeout=min(1.0, remaining))

    def _matching(self, tag_filter: list) -> list:
        """Return nodes whose tag set is a superset of tag_filter."""
        if not tag_filter:
            return list(self._nodes)
        return [n for n in self._nodes if all(t in n.tags for t in tag_filter)]

    def idle_nodes(self) -> list:
        """Return nodes that registered but were not selected (index >= expected)."""
        with self._cond:
            selected = self._matching(getattr(self, "_last_tag_filter", []))
            selected_ids = {n.node_id for n in selected[:getattr(self, "_last_expected", 0)]}
            return [n for n in self._nodes if n.node_id not in selected_ids]

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
    rpc_timeout: float = 5.0,
    retry_policy: RetryPolicy | None = None,
) -> AnnounceResult:
    """
    Client-side: call Register RPC on coordinator. Used by workers at startup
    and by the coordinator watchdog for periodic re-registration.
    Retries until the coordinator is reachable.
    """
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
        tags=node_info.tags,
    )
    req = coordinator_service_pb2.RegisterRequest(node=node_proto, run_id=run_id)

    policy = retry_policy or RetryPolicy(
        max_attempts=retries,
        delay_s=retry_delay,
        rpc_timeout_s=rpc_timeout,
    )

    def _register():
        channel = grpc.insecure_channel(coordinator_addr, options=_OPTS)
        try:
            channel = grpc.insecure_channel(coordinator_addr, options=_OPTS)
            stub = coordinator_service_pb2_grpc.CoordinatorServiceStub(channel)
            resp = stub.register(req, timeout=policy.rpc_timeout_s)
            if resp.ok:
                return AnnounceResult(run_id=resp.run_id, worker_index=resp.worker_index)
            raise RuntimeError(f"Coordinator rejected registration: {resp.message}")
        finally:
            channel.close()

    def _on_retry(attempt: int, max_attempts: int, exc: Exception, delay_s: float):
        _LOG.warning(
            "registration retry %s/%s for %s failed: %s; next_delay=%.1fs",
            attempt,
            max_attempts,
            coordinator_addr,
            exc,
            delay_s,
        )

    try:
        return call_with_retry(_register, policy=policy, on_retry=_on_retry)
    except Exception as exc:
        raise RuntimeError(
            f"Could not register with coordinator at {coordinator_addr} "
            f"after {policy.max_attempts} attempts: {exc}"
        ) from exc
