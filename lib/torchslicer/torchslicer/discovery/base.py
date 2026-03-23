from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class NodeInfo:
    node_id:   str        # unique ID (hostname or UUID)
    address:   str        # host:port of the node's gRPC server
    device:    str = "cpu"
    memory_mb: int = 0


@dataclass
class AnnounceResult:
    run_id:       str
    worker_index: int


class BaseDiscovery(ABC):
    """
    Abstraction over cluster membership for both centralized and P2P topologies.

    Centralized topology:
        Coordinator uses CoordinatorDiscovery (server side) — waits for workers
        to push Register RPCs. Workers call announce_to_coordinator() at startup.

    P2P topology:
        All nodes use StaticDiscovery — peer list from config, no registration needed.
        Future: MDNSDiscovery for zero-config local network auto-discovery.

    The watch() method is a forward-compatibility hook for fault-tolerance:
    a heartbeat mechanism will fire on_leave(node) when a peer drops, letting
    the executor decide how to handle it (abort, checkpoint, reroute).
    """

    @abstractmethod
    def announce(self, node_info: NodeInfo, coordinator_addr: str = None) -> AnnounceResult:
        """Register this node with the cluster. Returns assigned run_id and worker_index."""
        ...

    @abstractmethod
    def discover(self, expected: int, timeout: float = 60.0) -> List[NodeInfo]:
        """Block until `expected` peers are known. Returns NodeInfos in assignment order."""
        ...

    @abstractmethod
    def watch(
        self,
        on_join:  Callable[[NodeInfo], None],
        on_leave: Callable[[NodeInfo], None],
    ) -> None:
        """Register callbacks for membership changes (fault-tolerance hook, stub for now)."""
        ...

    def idle_nodes(self) -> List[NodeInfo]:
        """
        Return nodes that registered but were not selected by discover().
        Default is empty — only CoordinatorDiscovery can have idle nodes.
        """
        return []
