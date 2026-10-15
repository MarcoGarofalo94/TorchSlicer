"""``python -m torchslicer launch``: start or render one run for a target.

    python -m torchslicer launch SCRIPT -c CONFIG --target TARGET [options]

Targets (every one runs ``SCRIPT`` with ``ts.run`` on each node, from the same config):

- ``local``      runs ``SCRIPT`` here with ``ROLE=all``: the whole run in one process
                 (``LocalExecutor`` for ``topology: local``; loopback workers otherwise).
- ``processes``  starts one OS process per node on this host and waits for the run.
- ``compose``    writes ``docker-compose.yml`` (one service per node).
- ``k8s``        writes ``k8s.yaml``: a ConfigMap with the config, an Indexed Job of
                 workers/followers behind a headless Service (stable DNS names), and a
                 Job + Service for the coordinator/driver.
- ``baremetal``  writes one shell script per node for ``--hosts`` (driving node first).

The node list (names, roles, addresses, env) comes from :func:`plan_nodes`, so the
targets differ only in addresses and packaging.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time

from ..config import RunConfig
from .nodes import NodeSpec, coordinator_port, node_names, plan_nodes, tail_slice
from .runner import free_port

TARGETS = ("local", "processes", "compose", "k8s", "baremetal")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m torchslicer launch",
                                description="Start or render a TorchSlicer run for a target.")
    p.add_argument("script", help="training script that calls ts.run(...)")
    p.add_argument("-c", "--config", required=True, help="RunConfig YAML")
    p.add_argument("-t", "--target", choices=TARGETS, default="processes")
    p.add_argument("-o", "--out", default=None,
                   help="output directory (default: runs/deploy/<deploy.name>-<target>)")
    p.add_argument("--hosts", default="",
                   help="baremetal: comma-separated hosts, one per node, driving node first")
    p.add_argument("--workdir", default=None,
                   help="baremetal: directory on each host holding the script and config "
                        "(default: the current directory)")
    p.add_argument("--python", default=None,
                   help="interpreter to run the script with (default: this one for "
                        "local/processes/baremetal, python3 in containers)")
    p.add_argument("--timeout", type=float, default=0,
                   help="processes: kill the run after this many seconds (0 = no limit)")
    args = p.parse_args(argv)

    cfg = RunConfig.load(args.config)
    out = args.out or os.path.join("runs", "deploy", f"{cfg.deploy.name}-{args.target}")
    if args.target == "local":
        return _local(args, cfg)
    if args.target == "processes":
        return run_processes(args.script, args.config, cfg, out=out, python=args.python,
                             timeout=args.timeout)
    os.makedirs(out, exist_ok=True)
    if args.target == "compose":
        path = os.path.join(out, "docker-compose.yml")
        _write_yaml(path, render_compose(cfg, args.script, args.config, out=out,
                                         python=args.python or "python3"))
        driving = node_names(cfg)[0][0]
        print(f"wrote {path}\nrun:  docker compose -f {path} up --exit-code-from {driving}\n"
              f"stop: docker compose -f {path} down")
    elif args.target == "k8s":
        path = os.path.join(out, "k8s.yaml")
        with open(args.config) as f:
            config_text = f.read()
        _write_yaml(path, render_k8s(cfg, args.script, config_text,
                                     python=args.python or "python3"), many=True)
        print(f"wrote {path}\nrun:  kubectl apply -f {path}\n"
              f"wait: kubectl -n {cfg.deploy.namespace} wait --for=condition=complete "
              f"job/{_k8s_name(cfg, node_names(cfg)[0][0])} --timeout=24h\n"
              f"stop: kubectl delete -f {path}")
    else:
        hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
        paths = render_baremetal(cfg, args.script, args.config, hosts, out=out,
                                 workdir=args.workdir or os.getcwd(),
                                 python=args.python or "python3")
        print("wrote " + "\n      ".join(paths) +
              "\nrun each script on its host; start the workers/followers first")
    return 0


# ── local / processes ──────────────────────────────────────────────────────────

def _local(args, cfg: RunConfig) -> int:
    env = {**os.environ, "EXPERIMENT_CONFIG": args.config, "ROLE": "all"}
    return subprocess.call([args.python or sys.executable, args.script], env=env)


def run_processes(script: str, config_path: str, cfg: RunConfig, *, out: str,
                  python: str | None = None, timeout: float = 0,
                  extra_env: dict | None = None) -> int:
    """One OS process per node on this host (loopback, free ports). Workers start
    first; the driving node's output goes to stdout, the others' to ``out/<name>.log``.
    Returns the driving node's exit code (nonzero if a node crashed it)."""
    tcp = cfg.transport.tensor.strip().lower() == "tcp"
    offset = int(cfg.transport.tensor_port_offset) if tcp else 0
    names = node_names(cfg)
    ports = [free_port(offset) for _ in names]
    nodes = plan_nodes(cfg, ["127.0.0.1"] * len(names), ports, config_path=config_path,
                       tail_port=free_port(offset) if tail_slice(cfg) else None)
    if nodes[0].role == "coordinator":
        nodes[0].env["COORDINATOR_BIND_ADDRESS"] = nodes[0].address   # loopback only
    os.makedirs(out, exist_ok=True)
    python = python or sys.executable
    procs: list[tuple[NodeSpec, subprocess.Popen]] = []
    logs = []
    try:
        for nd in nodes[1:] + nodes[:1]:     # workers first
            env = {**os.environ, **(extra_env or {}), **nd.env}
            if nd.drives:
                stdout = None
            else:
                stdout = open(os.path.join(out, f"{nd.name}.log"), "w")
                logs.append(stdout)
            procs.append((nd, subprocess.Popen([python, script], env=env, stdout=stdout,
                                               stderr=subprocess.STDOUT if stdout else None)))
            print(f"[launch] {nd.name:<12} {nd.role:<11} {nd.address}  pid={procs[-1][1].pid}",
                  flush=True)
        lead = procs[-1][1]
        deadline = time.monotonic() + timeout if timeout else None
        while lead.poll() is None:
            if deadline and time.monotonic() > deadline:
                print(f"[launch] timeout after {timeout:.0f}s", flush=True)
                return 124
            for nd, pr in procs[:-1]:
                if pr.poll() not in (None, 0):
                    print(f"[launch] {nd.name} exited with {pr.returncode} "
                          f"(log: {os.path.join(out, nd.name + '.log')})", flush=True)
                    procs.remove((nd, pr))
            time.sleep(0.2)
        code = lead.returncode
        # the driving node shuts the workers down at the end of the run
        for nd, pr in procs[:-1]:
            try:
                pr.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print(f"[launch] {nd.name} still running after the run; stopping it", flush=True)
        print(f"[launch] done: {nodes[0].name} exited with {code}; worker logs in {out}",
              flush=True)
        return code
    finally:
        for _, pr in procs:
            if pr.poll() is None:
                pr.terminate()
                try:
                    pr.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pr.kill()
        for f in logs:
            f.close()


# ── compose ────────────────────────────────────────────────────────────────────

def render_compose(cfg: RunConfig, script: str, config_path: str, *, out: str = ".",
                   python: str = "python3", project_dir: str | None = None) -> dict:
    """A compose file with one service per node; service names are the hosts. The
    project directory (the cwd by default) is mounted at ``/workspace``, so ``script``
    and ``config_path`` are relative to it, as on the host."""
    names = node_names(cfg)
    port_w = int(cfg.network.worker_port)
    ports = [coordinator_port(cfg) if r == "coordinator" else port_w for _, r in names]
    nodes = plan_nodes(cfg, [n for n, _ in names], ports, config_path=config_path)
    root = os.path.relpath(os.path.abspath(project_dir or os.getcwd()), os.path.abspath(out))
    volumes = [f"{root}:/workspace"]
    if os.path.isdir(os.path.join(project_dir or os.getcwd(), "lib", "torchslicer")):
        # this repo: use the library source, as the dev compose files do
        volumes.append(f"{os.path.join(root, 'lib', 'torchslicer')}:/usr/local/lib/torchslicer")
    services = {}
    for nd in nodes:
        svc = {
            "image": cfg.deploy.image,
            "hostname": nd.host,
            "user": "${UID:-0}:${GID:-0}",   # `make _env` writes them: outputs owned by you
            "working_dir": "/workspace",
            "environment": dict(nd.env),
            "volumes": list(volumes),
            "command": [python, script],
            "networks": ["torchslicer"],
        }
        if cfg.deploy.gpus:
            svc["deploy"] = {"resources": {"reservations": {"devices": [
                {"driver": "nvidia", "count": int(cfg.deploy.gpus), "capabilities": ["gpu"]}]}}}
        if nd.drives:
            svc["depends_on"] = [x.name for x in nodes if not x.drives]
        services[nd.name] = svc
    return {"name": cfg.deploy.name, "services": services,
            "networks": {"torchslicer": {"driver": "bridge"}}}


# ── k8s ────────────────────────────────────────────────────────────────────────

_CONFIG_DIR = "/etc/torchslicer"


def _k8s_name(cfg: RunConfig, part: str) -> str:
    return f"{cfg.deploy.name}-{part}"


def render_k8s(cfg: RunConfig, script: str, config_text: str, *,
               python: str = "python3") -> list[dict]:
    """Manifests for one run. Workers/followers are an Indexed Job whose pods have
    stable DNS names ``<name>-workers-<i>.<name>-workers`` (headless Service), so
    discovery is static and the slice order is the pod index. The coordinator or
    driver is a Job behind a Service. Both Jobs complete when the run ends: the
    driving node shuts the workers down."""
    ns = cfg.deploy.namespace
    names = node_names(cfg)
    lead_name, lead_role = names[0]
    wjob = _k8s_name(cfg, "workers")
    lead = _k8s_name(cfg, lead_name)
    port_w = int(cfg.network.worker_port)
    lead_port = coordinator_port(cfg) if lead_role == "coordinator" else port_w
    hosts = [lead] + [f"{wjob}-{i}.{wjob}" for i in range(len(names) - 1)]
    ports = [lead_port] + [port_w] * (len(names) - 1)
    nodes = plan_nodes(cfg, hosts, ports, config_path=f"{_CONFIG_DIR}/run.yaml",
                       discovery="static")
    tcp = cfg.transport.tensor.strip().lower() == "tcp"
    offset = int(cfg.transport.tensor_port_offset)

    def ports_of(port, tail: int | None = None):
        ps = [{"name": "grpc", "port": port, "targetPort": port}]
        if tcp:
            ps.append({"name": "tensor", "port": port + offset, "targetPort": port + offset})
        if tail:   # u_shaped: the driving node's last slice (followers send to it)
            ps.append({"name": "tail", "port": tail, "targetPort": tail})
            if tcp:
                ps.append({"name": "tail-tensor", "port": tail + offset, "targetPort": tail + offset})
        return ps

    lead_tail = int(nodes[0].env["SL_TAIL_PORT"]) if "SL_TAIL_PORT" in nodes[0].env else None

    def container(nd: NodeSpec, env: dict) -> dict:
        c = {
            "name": "torchslicer",
            "image": cfg.deploy.image,
            "workingDir": "/workspace",
            "command": [python, script],
            "env": [{"name": k, "value": v} for k, v in env.items()],
            "ports": [{"containerPort": p["targetPort"], "name": p["name"]}
                      for p in ports_of(nd.port, lead_tail if nd is nodes[0] else None)],
            "volumeMounts": [{"name": "config", "mountPath": _CONFIG_DIR}],
        }
        if cfg.deploy.gpus:
            c["resources"] = {"limits": {"nvidia.com/gpu": int(cfg.deploy.gpus)}}
        return c

    def pod(labels: dict, nd: NodeSpec, env: dict, **extra) -> dict:
        return {
            "metadata": {"labels": labels},
            "spec": {
                "restartPolicy": "Never",
                **extra,
                "containers": [container(nd, env)],
                "volumes": [{"name": "config",
                             "configMap": {"name": _k8s_name(cfg, "config")}}],
            },
        }

    meta = lambda name: {"name": name, "namespace": ns,
                         "labels": {"app.kubernetes.io/name": cfg.deploy.name}}
    docs = [{
        "apiVersion": "v1", "kind": "ConfigMap", "metadata": meta(_k8s_name(cfg, "config")),
        "data": {"run.yaml": config_text},
    }]
    lead_labels = {"app.kubernetes.io/name": cfg.deploy.name, "torchslicer/node": lead_name}
    docs.append({
        "apiVersion": "v1", "kind": "Service", "metadata": meta(lead),
        "spec": {"selector": lead_labels, "ports": ports_of(lead_port, lead_tail)},
    })
    k = len(nodes) - 1
    if k:
        w_labels = {"app.kubernetes.io/name": cfg.deploy.name, "torchslicer/node": "workers"}
        docs.append({
            "apiVersion": "v1", "kind": "Service", "metadata": meta(wjob),
            "spec": {"clusterIP": "None", "publishNotReadyAddresses": True,
                     "selector": w_labels, "ports": ports_of(port_w)},
        })
        # one pod template for every worker: they differ only in their address,
        # which none of them needs (static discovery; the coordinator/driver dials them)
        wenv = {key: v for key, v in nodes[1].env.items() if key != "WORKER_ADDRESS"}
        docs.append({
            "apiVersion": "batch/v1", "kind": "Job", "metadata": meta(wjob),
            "spec": {
                "completionMode": "Indexed", "completions": k, "parallelism": k,
                "backoffLimit": 0,
                "template": pod(w_labels, nodes[1], wenv, subdomain=wjob),
            },
        })
    docs.append({
        "apiVersion": "batch/v1", "kind": "Job", "metadata": meta(lead),
        "spec": {"backoffLimit": 0, "template": pod(lead_labels, nodes[0], nodes[0].env)},
    })
    return docs


# ── bare metal ─────────────────────────────────────────────────────────────────

def render_baremetal(cfg: RunConfig, script: str, config_path: str, hosts: list[str], *,
                     out: str, workdir: str, python: str = "python3") -> list[str]:
    """One ``<node>.sh`` per node. ``hosts`` has one entry per node, the driving node
    first; nodes sharing a host get consecutive ports."""
    names = node_names(cfg)
    if len(hosts) != len(names):
        raise SystemExit(f"--hosts needs {len(names)} hosts ({', '.join(n for n, _ in names)}), "
                         f"got {len(hosts)}")
    tcp = cfg.transport.tensor.strip().lower() == "tcp"
    step = int(cfg.transport.tensor_port_offset) + 1 if tcp else 1
    used: dict[tuple, int] = {}
    ports = []
    for (_, role), h in zip(names, hosts):
        base = coordinator_port(cfg) if role == "coordinator" else int(cfg.network.worker_port)
        ports.append(base + step * used.get((h, base), 0))
        used[(h, base)] = used.get((h, base), 0) + 1
    tail = None
    if tail_slice(cfg) and not cfg.sl.tail_port:
        # the next free slot after every worker port on the driving node's host
        taken = {p + d for h, p in zip(hosts, ports) if h == hosts[0] for d in range(step)}
        tail = ports[0] + 2
        while any(tail + d in taken for d in range(step)):
            tail += step
    nodes = plan_nodes(cfg, hosts, ports, config_path=config_path, tail_port=tail)
    paths = []
    for nd in nodes:
        lines = ["#!/bin/sh", f"# {nd.name} ({nd.role}) on {nd.host}", "set -e",
                 f"cd {shlex.quote(workdir)}"]
        lines += [f"export {k}={shlex.quote(v)}" for k, v in nd.env.items()]
        lines.append(f"exec {python} {shlex.quote(script)}")
        path = os.path.join(out, f"{nd.name}.sh")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.chmod(path, 0o755)
        paths.append(path)
    return paths


def _write_yaml(path: str, data, many: bool = False) -> None:
    import yaml

    class _Dumper(yaml.SafeDumper):
        pass

    def _str(dumper, value):   # multi-line strings (the config in the ConfigMap) as blocks
        style = "|" if "\n" in value else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)

    _Dumper.add_representer(str, _str)
    with open(path, "w") as f:
        if many:
            yaml.dump_all(data, f, Dumper=_Dumper, sort_keys=False)
        else:
            yaml.dump(data, f, Dumper=_Dumper, sort_keys=False)
