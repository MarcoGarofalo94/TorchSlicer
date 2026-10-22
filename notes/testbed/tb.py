"""Lab-specific driver for the physical testbed (the owner's ansible inventory and .env under
/home/fcr/work/jetson/jetson); not shipped. See notes/validation.md "M14 start".

Run one TorchSlicer launch config across the physical testbed (baremetal target, each node in Docker).

    python tb.py run NAME SCRIPT CONFIG node1,node2,...  [--env K=V ...] [--node-env node:K=V ...] [--timeout S]
    python tb.py sh NODES 'command'          # run a shell command on nodes (parallel)
    python tb.py clean NODES                 # remove our ts-* containers

Nodes are inventory names (jon-00x, jor-001, dell-g3), driving node first, as in --hosts.
Only touches ~/ts-test on the nodes and containers named ts-*.
"""
import concurrent.futures as cf
import json
import os
import shlex
import subprocess
import sys
import time

JET = "/home/fcr/work/jetson/jetson"
REPO = "/home/fcr/work/TorchSlicer"
HERE = os.path.dirname(os.path.abspath(__file__))


def _env_file():
    out = {}
    for line in open(f"{JET}/.env"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k] = v.strip().strip('"').strip("'")
    return out


ENV = _env_file()
HOSTS = {}
for line in open(f"{JET}/inventory.ini"):
    p = line.split()
    if len(p) == 2 and p[1].startswith("ansible_host="):
        HOSTS[p[0]] = p[1].split("=", 1)[1]


def login(node):
    if node == "dell-g3":
        return "marcub", None
    if node == "jor-001":
        return ENV["ORIN_USER"], ENV["ORIN_PASS"]
    return ENV["NANO_USER"], ENV["NANO_PASS"]


def image(node):
    return "mesl-x86:latest" if node == "dell-g3" else "mesl-jetson:latest"


def mem(node):
    return {"jor-001": "48g", "dell-g3": "10g"}.get(node, "6g")


def sudo_pw(node):
    return ENV["DELL_PASS"] if node == "dell-g3" else login(node)[1]


def home(node):
    return f"/home/{login(node)[0]}"


def ssh(node, cmd, timeout=900, check=False, sudo=False):
    user, pw = login(node)
    stdin = None
    if sudo:
        cmd = "sudo -S -p '' sh -c " + shlex.quote(cmd.replace("$HOME", home(node)))
        stdin = sudo_pw(node) + "\n"
    base = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=10"]
    env = dict(os.environ)
    if pw:
        base = ["sshpass", "-e"] + base
        env["SSHPASS"] = pw
    else:
        base += ["-i", os.path.expanduser("~/.ssh/id_ed25519"), "-o", "BatchMode=yes"]
    r = subprocess.run(base + [f"{user}@{HOSTS[node]}", cmd], capture_output=True, text=True, env=env,
                       timeout=timeout, input=stdin)
    if check and r.returncode:
        raise RuntimeError(f"{node}: rc={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r


def scp(node, src, dst):
    user, pw = login(node)
    cmd = ["scp", "-q", "-r", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           "-o", "LogLevel=ERROR"]
    env = dict(os.environ)
    if pw:
        cmd = ["sshpass", "-e"] + cmd
        env["SSHPASS"] = pw
    else:
        cmd += ["-i", os.path.expanduser("~/.ssh/id_ed25519")]
    subprocess.run(cmd + [src, f"{user}@{HOSTS[node]}:{dst}"], check=True, env=env)


def par(nodes, fn):
    with cf.ThreadPoolExecutor(len(nodes)) as ex:
        return dict(zip(nodes, ex.map(fn, nodes)))


def docker_cmd(node, cname, script, envs):
    e = " ".join(f"-e {shlex.quote(k + '=' + v)}" for k, v in envs.items())
    return (f"docker rm -f {cname} >/dev/null 2>&1; docker run -d --name {cname} --network host --runtime nvidia "
            f"--memory {mem(node)} -v $HOME/ts-test:/t "
            f"-e PYTHONPATH=/t/TorchSlicer/lib/torchslicer:/t/TorchSlicer:/t/pylib -e PYTHONUNBUFFERED=1 {e} "
            f"{image(node)} sh {script}")


def run(name, script, config, nodes, envs, node_envs, timeout, extra_args=()):
    out = f"{REPO}/runs/deploy/tb-{name}"
    hosts = ",".join(HOSTS[n] for n in nodes)
    subprocess.run([f"{REPO}/.venv/bin/python", "-m", "torchslicer", "launch", script, "-c", config,
                    "--target", "baremetal", "--hosts", hosts, "--workdir", "/t/TorchSlicer", "--out", out,
                    *extra_args], check=True, cwd=REPO, stdout=subprocess.DEVNULL)
    scripts = sorted(f for f in os.listdir(out) if f.endswith(".sh"))
    # the scripts are in --hosts order only by name; map each to its node through its header line
    by_host = {}
    for f in scripts:
        host = open(f"{out}/{f}").read().splitlines()[1].rsplit(" on ", 1)[1]
        by_host.setdefault(host, []).append(f)
    used = sorted(set(nodes))
    par(used, lambda n: ssh(n, f"rm -rf ~/ts-test/runs/{name} && mkdir -p ~/ts-test/runs/{name}", check=True))
    par(used, lambda n: [scp(n, f"{out}/{f}", f"ts-test/runs/{name}/") for f in by_host[HOSTS[n]]])
    first = scripts_order(out, scripts)
    driving = first[0]
    plan = []          # (node, script) for every script, driving one last
    for f in first[1:] + [driving]:
        host = open(f"{out}/{f}").read().splitlines()[1].rsplit(" on ", 1)[1]
        node = next(n for n in nodes if HOSTS[n] == host)
        plan.append((node, f))
    cnames = []
    t0 = time.time()
    for node, f in plan:
        cname = f"ts-{name}-{f[:-3]}"
        env = dict(envs)
        for k, v in node_envs.get(node, {}).items():
            env[k] = v
        ssh(node, docker_cmd(node, cname, f"/t/runs/{name}/{f}", env), check=True, sudo=True)
        cnames.append((node, cname))
        if f is not driving:
            time.sleep(0.3)
    dnode, dname = cnames[-1]
    r = ssh(dnode, f"timeout {timeout} docker wait {dname}", timeout=timeout + 60, sudo=True)
    wall = time.time() - t0
    code = r.stdout.strip()
    logs = par_logs(cnames)
    os.makedirs(f"{HERE}/logs/{name}", exist_ok=True)
    for (node, cname), text in logs.items():
        open(f"{HERE}/logs/{name}/{cname}.log", "w").write(text)
    par(sorted({n for n, _ in cnames}), lambda n: ssh(n, "docker rm -f " + " ".join(
        c for nn, c in cnames if nn == n) + " >/dev/null 2>&1", sudo=True))
    for l in logs[(dnode, dname)].splitlines():
        if l.startswith("RESULT "):
            print(l, flush=True)
    hist = [l for l in logs[(dnode, dname)].splitlines() if l.startswith("HISTORY")]
    print(json.dumps({"name": name, "exit": code, "wall_s": round(wall, 1),
                      "history": hist[-1][8:] if hist else None}))
    if not hist or code != "0":
        print(logs[(dnode, dname)][-3000:])
        for (n, c), t in logs.items():
            if c != dname and ("Error" in t or "Traceback" in t):
                print(f"--- {c} ---\n{t[-1500:]}")


def scripts_order(out, scripts):
    """Driving node first: coordinator / driver / the first script launch lists."""
    meta = {f: open(f"{out}/{f}").read().splitlines()[1] for f in scripts}
    drive = [f for f in scripts if "(coordinator)" in meta[f] or "(driver)" in meta[f] or "(all)" in meta[f]]
    rest = [f for f in scripts if f not in drive]
    return drive + rest


def par_logs(cnames):
    return {nc: ssh(nc[0], f"docker logs {nc[1]} 2>&1 | tail -c 200000", sudo=True).stdout for nc in cnames}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sh":
        nodes = sys.argv[2].split(",")
        su = len(sys.argv) > 4 and sys.argv[4] == "sudo"
        for n, r in par(nodes, lambda n: ssh(n, sys.argv[3], sudo=su)).items():
            print(f"== {n} rc={r.returncode}\n{r.stdout}{r.stderr}", end="")
    elif cmd == "clean":
        nodes = sys.argv[2].split(",")
        par(nodes, lambda n: ssh(n, "docker ps -aq --filter name=^ts- | xargs -r docker rm -f", sudo=True))
    elif cmd == "run":
        name, script, config, nodes = sys.argv[2:6]
        envs, node_envs, timeout, extra = {}, {}, 900, []
        a = sys.argv[6:]
        i = 0
        while i < len(a):
            if a[i] == "--env":
                k, v = a[i + 1].split("=", 1); envs[k] = v; i += 2
            elif a[i] == "--node-env":
                n, kv = a[i + 1].split(":", 1); k, v = kv.split("=", 1)
                node_envs.setdefault(n, {})[k] = v; i += 2
            elif a[i] == "--timeout":
                timeout = int(a[i + 1]); i += 2
            elif a[i] == "--":
                extra = a[i + 1:]; break
            else:
                raise SystemExit(f"unknown arg {a[i]}")
        run(name, script, config, nodes.split(","), envs, node_envs, timeout, extra)
