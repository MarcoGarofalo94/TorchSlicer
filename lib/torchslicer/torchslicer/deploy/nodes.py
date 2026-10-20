"""The nodes of a run, derived from its config: one list for every target.

``plan_nodes`` turns a ``RunConfig`` into ``NodeSpec``s (name, role, address, env).
Each launch target only decides the addresses and renders the same list: OS
processes on this host, compose services, k8s Jobs, or per-host bare-metal scripts.
Every node runs the same script with ``ts.run``; its env says which node it is.
"""

from dataclasses import dataclass, field

from ..config import RunConfig
from .runner import client_nodes, hosted_slices, resolve_role

RELAY_PORT = 50060   # the relay's port where the target picks fixed ports (compose, k8s, baremetal)


@dataclass
class NodeSpec:
    name:    str            # "coordinator", "worker-0", ..., "client-0", ... / "driver", "follower-0", ... / "relay"
    role:    str            # coordinator | worker | client | driver | follower | relay
    host:    str            # how the other nodes reach it
    port:    int            # its gRPC port (the coordinator's, for the coordinator)
    env:     dict = field(default_factory=dict)

    def argv(self, script: str) -> list:
        """What the node runs after the interpreter: the training script, or for the
        relay ``-m torchslicer relay`` (it needs no user code)."""
        if self.role == "relay":
            return ["-m", "torchslicer", "relay", "--port", str(self.port)]
        return [script]

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def drives(self) -> bool:
        """The node that runs training (and whose exit ends the run)."""
        return self.role in ("coordinator", "driver")


def node_names(cfg: RunConfig) -> list[tuple[str, str]]:
    """``(name, role)`` of every node, the driving one first. ``discovery.n_workers`` is
    the number of slices; the driving node hosts ``sl.setting``'s share of them
    (``hosted_slices``) and there is one worker/follower per remaining slice. With
    ``sl.client_nodes``, one client node per client follows the workers. With
    ``transport.relay: auto`` a relay node comes last."""
    topology, _ = resolve_role(_without_role(cfg))
    n = int(cfg.discovery.n_workers)
    if topology == "local":
        raise ValueError("deploy.topology is local: there is nothing to deploy "
                         "(set centralized or p2p)")
    if topology == "p2p" and n < 1:
        raise ValueError("p2p needs discovery.n_workers >= 1 (the driver holds one slice)")
    k = n - hosted_slices(cfg, topology, n)
    relay = [("relay", "relay")] if relay_node(cfg) else []
    if topology == "centralized":
        clients = [(f"client-{i}", "client") for i in range(client_nodes(cfg))]
        return [("coordinator", "coordinator")] + [(f"worker-{i}", "worker") for i in range(k)] + clients + relay
    return [("driver", "driver")] + [(f"follower-{i}", "follower") for i in range(k)] + relay


def relay_node(cfg: RunConfig) -> bool:
    """``transport.relay: auto``: the launcher adds a relay node. Any other non-empty
    value is the address of a relay that runs elsewhere."""
    return (cfg.transport.relay or "").strip().lower() == "auto"


def tail_slice(cfg: RunConfig) -> bool:
    """The driving node also hosts the last slice (``sl.setting: u_shaped``), served on
    its own port (``sl.tail_port``, default the driving node's port + 2)."""
    from ..sl.roles import resolve_setting
    topology, _ = resolve_role(_without_role(cfg))
    return topology != "local" and resolve_setting(cfg.sl.setting, topology) == "u_shaped"


def coordinator_port(cfg: RunConfig) -> int:
    return int(cfg.network.coordinator_bind_address.rsplit(":", 1)[-1])


def plan_nodes(cfg: RunConfig, hosts: list[str], ports: list[int], *,
               config_path: str | None = None, discovery: str | None = None,
               tail_port: int | None = None) -> list[NodeSpec]:
    """The nodes of this run, at ``hosts[i]:ports[i]`` (in ``node_names`` order).

    ``discovery`` overrides ``discovery.backend`` for centralized runs ("static":
    the coordinator gets the worker list; "coordinator": workers register).
    ``tail_port``: u_shaped, the driving node's port for the last slice (env
    ``SL_TAIL_PORT``); default ``sl.tail_port`` or its port + 2.

    ``transport.nat`` in the shared config means the driving node (the data owner) is
    behind NAT: it gets ``TRANSPORT_NAT=true`` and the others ``false`` (set
    ``TRANSPORT_NAT=true`` on a worker's own env to put it behind NAT). Every worker
    and follower gets ``COORDINATOR_ADDRESS``, the node a NAT'd one tunnels to.
    """
    topology, _ = resolve_role(_without_role(cfg))
    names = node_names(cfg)
    if len(hosts) != len(names) or len(ports) != len(names):
        raise ValueError(f"{len(names)} nodes need {len(names)} hosts and ports, "
                         f"got {len(hosts)} and {len(ports)}")
    nodes = [NodeSpec(name, role, h, p) for (name, role), h, p in zip(names, hosts, ports)]
    relay = nodes.pop() if relay_node(cfg) else None
    lead, others = nodes[0], nodes[1:]
    backend = discovery or cfg.discovery.backend
    nat = bool(cfg.transport.nat)
    if nat and not cfg.transport.relay and topology == "centralized" and backend == "coordinator" \
            and others:
        raise ValueError("the coordinator is behind NAT (transport.nat), so workers can't "
                         "register with it: use discovery.backend static, or a relay "
                         "(transport.relay)")
    common = {
        "PYTHONUNBUFFERED": "1",
        "TOPOLOGY": topology,
        "RUN_ID": cfg.run_id,
        "N_WORKERS": str(cfg.discovery.n_workers),
    }
    if config_path:
        common["EXPERIMENT_CONFIG"] = config_path
    for nd in nodes:
        nd.env = {**common, "ROLE": nd.role}
        if nd.role == "coordinator":
            nd.env["COORDINATOR_BIND_ADDRESS"] = f"0.0.0.0:{nd.port}"
            nd.env["COORDINATOR_ADDRESS"] = nd.address
            nd.env["DISCOVERY_BACKEND"] = backend
            if backend == "static":
                nd.env["WORKER_PEERS"] = ",".join(w.address for w in others)
        else:
            nd.env["PORT"] = str(nd.port)
            nd.env["WORKER_ADDRESS"] = nd.address
        if nd.role in ("worker", "client"):
            nd.env["DISCOVERY_BACKEND"] = backend
        if nd.role == "client":
            nd.env["CLIENT_INDEX"] = nd.name.split("-", 1)[1]
        if nd.role in ("worker", "follower", "client"):
            nd.env["COORDINATOR_ADDRESS"] = lead.address
        if nd.role == "driver":
            nd.env["WORKER_PEERS"] = ",".join(x.address for x in nodes)
        if nat:
            nd.env["TRANSPORT_NAT"] = "true" if nd is lead else "false"
        if relay is not None:
            nd.env["TRANSPORT_RELAY"] = relay.address
    if tail_slice(cfg):
        lead.env["SL_TAIL_PORT"] = str(tail_port or cfg.sl.tail_port or lead.port + 2)
    if relay is not None:
        relay.env = {"PYTHONUNBUFFERED": "1", "ROLE": "relay"}
        nodes.append(relay)
    return nodes


def _without_role(cfg: RunConfig) -> RunConfig:
    """The launcher plans every role, whatever ``deploy.role`` this config sets."""
    import copy
    cfg = copy.deepcopy(cfg)
    cfg.deploy.role = ""
    return cfg
