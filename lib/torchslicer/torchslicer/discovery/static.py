"""
StaticDiscovery — config-driven peer list, no registration required.

Suitable for P2P topologies and fixed-address clusters where all node
addresses are known upfront (e.g. Docker Compose with fixed service names).

Usage::

    discovery = StaticDiscovery(peers=["worker1:50051", "worker2:50051"])
    nodes = discovery.discover(expected=2)   # returns immediately
"""

from typing import Callable, List

from .base import BaseDiscovery, NodeInfo, AnnounceResult


class StaticDiscovery(BaseDiscovery):
    """
    Peer list from config. No communication, no central point.
    discover() returns immediately with the pre-configured list.
    announce() is a no-op (index determined by position in the peers list).
    """

    def __init__(self, peers: List[str]):
        """
        peers: list of "host:port" strings in assignment order.
               Index 0 gets slice 0, index 1 gets slice 1, etc.
        """
        self._nodes = [
            NodeInfo(node_id=addr, address=addr)
            for addr in peers
        ]

    def announce(self, node_info: NodeInfo, coordinator_addr: str = None) -> AnnounceResult:
        """No-op for static discovery. Index is determined by position in peers list."""
        try:
            idx = next(i for i, n in enumerate(self._nodes) if n.address == node_info.address)
        except StopIteration:
            idx = 0
        return AnnounceResult(run_id="static", worker_index=idx)

    def discover(self, expected: int, timeout: float = 60.0) -> List[NodeInfo]:
        """Returns the pre-configured peer list immediately."""
        if len(self._nodes) < expected:
            raise ValueError(
                f"StaticDiscovery: configured {len(self._nodes)} peers "
                f"but {expected} expected"
            )
        return self._nodes[:expected]

    def watch(
        self,
        on_join:  Callable[[NodeInfo], None],
        on_leave: Callable[[NodeInfo], None],
    ) -> None:
        pass  # No dynamic membership in static mode
