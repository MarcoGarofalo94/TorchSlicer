from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class NodeInfo:
    """A worker as the planner sees it. Fields that are 0 are unknown.

    Declared at startup (``network.worker_*`` config / env) or measured: the worker
    reports ``device``, ``memory_mb`` and ``compute_gflops`` (a short matmul
    benchmark; a declared value wins), and the executor measures ``rtt_ms`` and
    ``bandwidth_mbps`` from itself, and ``link_*`` from this node to the next one in
    slice order (``planning.measure_links``).
    """
    node_id:   str        # unique ID (hostname or UUID)
    address:   str        # host:port of the node's gRPC server
    device:    str = "cpu"
    memory_mb: int = 0
    tags:      list = field(default_factory=list)  # user-defined labels, e.g. ["gpu", "high-memory"]
    compute_gflops: float = 0.0   # matmul GFLOP/s
    max_memory_mb:  int   = 0     # declared cap for this node's slice (0 = none)
    first_slice:    bool  = False # must hold slice 0 (e.g. the data owner)
    rtt_ms:         float = 0.0   # executor → node
    bandwidth_mbps: float = 0.0   # executor → node, megabits/s
    link_rtt_ms:         float = 0.0  # this node → next node in slice order
    link_bandwidth_mbps: float = 0.0

    def memory_cap_mb(self) -> int:
        """The memory a slice on this node may use: the declared cap, else what is free."""
        return self.max_memory_mb or self.memory_mb


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

    def watch(
        self,
        on_join:  Callable[[NodeInfo], None] = None,
        on_leave: Callable[[NodeInfo], None] = None,
    ) -> None:
        """Register callbacks for membership changes (no-op default; heartbeat wired in DistributedExecutor)."""
        pass

    def idle_nodes(self) -> List[NodeInfo]:
        """
        Return nodes that registered but were not selected by discover().
        Default is empty — only CoordinatorDiscovery can have idle nodes.
        """
        return []
