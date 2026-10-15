"""The nodes of a run, derived from its config: one list for every target.

``plan_nodes`` turns a ``RunConfig`` into ``NodeSpec``s (name, role, address, env).
Each launch target only decides the addresses and renders the same list: OS
processes on this host, compose services, k8s Jobs, or per-host bare-metal scripts.
Every node runs the same script with ``ts.run``; its env says which node it is.
"""

from dataclasses import dataclass, field

from ..config import RunConfig
from .runner import hosted_slices, resolve_role


@dataclass
class NodeSpec:
    name:    str            # "coordinator", "worker-0", ... / "driver", "follower-0", ...
    role:    str            # coordinator | worker | driver | follower
    host:    str            # how the other nodes reach it
    port:    int            # its gRPC port (the coordinator's, for the coordinator)
    env:     dict = field(default_factory=dict)

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
    (``hosted_slices``) and there is one worker/follower per remaining slice."""
    topology, _ = resolve_role(_without_role(cfg))
    n = int(cfg.discovery.n_workers)
    if topology == "local":
        raise ValueError("deploy.topology is local: there is nothing to deploy "
                         "(set centralized or p2p)")
    if topology == "p2p" and n < 1:
        raise ValueError("p2p needs discovery.n_workers >= 1 (the driver holds one slice)")
    k = n - hosted_slices(cfg, topology, n)
    if topology == "centralized":
        return [("coordinator", "coordinator")] + [(f"worker-{i}", "worker") for i in range(k)]
    return [("driver", "driver")] + [(f"follower-{i}", "follower") for i in range(k)]


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
    """
    topology, _ = resolve_role(_without_role(cfg))
    names = node_names(cfg)
    if len(hosts) != len(names) or len(ports) != len(names):
        raise ValueError(f"{len(names)} nodes need {len(names)} hosts and ports, "
                         f"got {len(hosts)} and {len(ports)}")
    nodes = [NodeSpec(name, role, h, p) for (name, role), h, p in zip(names, hosts, ports)]
    lead, others = nodes[0], nodes[1:]
    backend = discovery or cfg.discovery.backend
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
        if nd.role == "worker":
            nd.env["DISCOVERY_BACKEND"] = backend
            if backend == "coordinator":
                nd.env["COORDINATOR_ADDRESS"] = lead.address
        if nd.role == "driver":
            nd.env["WORKER_PEERS"] = ",".join(x.address for x in nodes)
    if tail_slice(cfg):
        lead.env["SL_TAIL_PORT"] = str(tail_port or cfg.sl.tail_port or lead.port + 2)
    return nodes


def _without_role(cfg: RunConfig) -> RunConfig:
    """The launcher plans every role, whatever ``deploy.role`` this config sets."""
    import copy
    cfg = copy.deepcopy(cfg)
    cfg.deploy.role = ""
    return cfg
