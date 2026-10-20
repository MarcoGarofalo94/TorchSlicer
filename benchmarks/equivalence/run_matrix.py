"""Model x mode matrix: every mode must reproduce the unsplit-PyTorch loss trajectory.

usage: run_matrix.py [model ...]     logs in ./logs/<device>_<opt>/<model>_<mode>*.log
env:   MATRIX_DEVICE=cpu|cuda, MATRIX_OPT='{"name": ..., "params": {...}}', MATRIX_PORT_BASE,
       MATRIX_MODES=local,offload,... (a subset of the modes; only a full run writes results),
       MATRIX_NCCL=1 (add {central,p2p}-nccl: on CUDA they need one GPU per slice; on CPU they
       run the same code path on the gloo backend), MATRIX_PARTS (slices, default 3)
Two hosts: MATRIX_REMOTE='<ssh args> user@host' runs the odd-index workers (central) / P2P
followers on that host over ssh, from the same repo path there (MATRIX_REMOTE_REPO; sync the
code first, e.g. rsync). Nodes then talk over real addresses: MATRIX_HOST (this host's IP) and
MATRIX_REMOTE_HOST (the other's). Only the distributed modes run remotely.
A full run (no model args) writes results/results_<device>_<opt>.json.
"""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import tsmodels as M

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = str(REPO / ".venv/bin/python")
LOGS = HERE / "logs" / f"{M.DEVICE}_{M.OPTIMIZER['name']}"
LOGS.mkdir(parents=True, exist_ok=True)
GPU = M.DEVICE == "cuda"
BASE_ENV = {**os.environ, "PYTHONPATH": f"{HERE}:{REPO}", "PYTHONUNBUFFERED": "1",
            "DEVICE": M.DEVICE, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
if not GPU:
    BASE_ENV["CUDA_VISIBLE_DEVICES"] = ""
TIMEOUT = 240
# below Linux's ephemeral range (32768-60999): outgoing connections can't grab these
_port = [int(os.environ.get("MATRIX_PORT_BASE", 21000))]


def _ports(k):
    _port[0] += 20
    return [_port[0] + 2 * i for i in range(k)]   # step 2: tcp tensor server = port+1


REMOTE = os.environ.get("MATRIX_REMOTE", "")
REMOTE_REPO = os.environ.get("MATRIX_REMOTE_REPO", str(REPO))
LOCAL_HOST = os.environ.get("MATRIX_HOST", "127.0.0.1")
REMOTE_HOST = os.environ.get("MATRIX_REMOTE_HOST", "127.0.0.1")
if REMOTE and "127.0.0.1" in (LOCAL_HOST, REMOTE_HOST):
    sys.exit("MATRIX_REMOTE needs MATRIX_HOST and MATRIX_REMOTE_HOST (real addresses)")
_REMOTE_ENV = ("DEVICE", "CUBLAS_WORKSPACE_CONFIG", "TENSOR_TRANSPORT", "PORT", "WORKER_ADDRESS",
               "COORDINATOR_ADDRESS", "WORKER_PEERS", "IS_DRIVER", "WORKER_INDEX", "MATRIX_DEVICE",
               "MATRIX_OPT", "MATRIX_PARTS", "OMP_NUM_THREADS", "NCCL_SOCKET_IFNAME", "NCCL_DEBUG")


def _spawn(args, env, log, remote=False):
    if not remote:
        return subprocess.Popen([PY, *args], cwd=HERE, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    # The remote process carries a marker argument, so _stop can kill it by name there.
    here = f"{REMOTE_REPO}/benchmarks/equivalence"
    exports = " ".join(f"{k}={shlex.quote(env[k])}" for k in _REMOTE_ENV if k in env)
    cmd = (f"cd {shlex.quote(here)} && exec env PYTHONPATH={here}:{REMOTE_REPO} PYTHONUNBUFFERED=1 {exports} "
           f"timeout {TIMEOUT + 60} {REMOTE_REPO}/.venv/bin/python "
           + " ".join(shlex.quote(a) for a in args) + f" {_MARKER}")
    p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", *shlex.split(REMOTE), cmd],
                         stdout=open(log, "w"), stderr=subprocess.STDOUT)
    p.remote = True
    return p


_MARKER = f"tsmatrix-{os.getpid()}"   # tags this run's remote processes


def _json_losses(log):
    for line in reversed(Path(log).read_text().splitlines()):
        if line.startswith('{"losses"'):
            out = json.JSONDecoder().raw_decode(line)[0]   # another thread's log may follow on the line
            return "SKIP" if out.get("skip") else out["losses"]
    return None


def _stop(procs):
    for p in procs:
        if p.poll() is None:
            p.kill()
    for p in procs:
        p.wait()
    if any(getattr(p, "remote", False) for p in procs):
        _kill_remote()


def _host(i):
    """Address of worker ``i``: odd workers run on the remote host when there is one."""
    return REMOTE_HOST if REMOTE and i % 2 else LOCAL_HOST


def inproc(name, mode):
    log = LOGS / f"{name}_{mode}.log"
    p = _spawn(["run_inproc.py", name, mode], BASE_ENV, log)
    p.wait(TIMEOUT)
    return _json_losses(log)


# SIGUSR1 dumps every thread's stack to the log (to locate a hang)
_WORKER = ("import faulthandler, signal; faulthandler.register(signal.SIGUSR1, all_threads=True); "
           "import torch; torch.use_deterministic_algorithms(True, warn_only=True); "
           "import torchslicer as ts; ts.run_worker()")


def central(name, transport):
    *wports, cport = _ports(M.N_PARTS + 1)
    env = {**BASE_ENV, "TENSOR_TRANSPORT": transport}
    tag = f"{name}_central-{transport}"
    workers = [_spawn(["-c", _WORKER],
                      {**env, "PORT": str(p), "WORKER_ADDRESS": f"{_host(i)}:{p}",
                       "COORDINATOR_ADDRESS": f"{LOCAL_HOST}:{cport}"},
                      LOGS / f"{tag}_w{i}.log", remote=_host(i) != LOCAL_HOST) for i, p in enumerate(wports)]
    log = LOGS / f"{tag}_coord.log"
    coord = _spawn(["run_central.py", name, str(cport), LOCAL_HOST], env, log)
    try:
        coord.wait(TIMEOUT)
    finally:
        _stop([coord, *workers])
    return _json_losses(log)


def p2p(name, transport, nat=""):
    """``nat``: "" (every node reachable) | "driver" (the driver is behind NAT and
    opens no port) | "relay" (every node behind NAT, all calls through a relay process)."""
    ports = _ports(M.N_PARTS + 1)
    relay_port = ports.pop()
    peers = ",".join(f"{_host(i)}:{p}" for i, p in enumerate(ports))
    cfg_path = LOGS / "p2p.yaml"
    cfg_path.write_text(json.dumps({"training": {"epochs": M.EPOCHS, "optimizer": M.OPTIMIZER,
                                                 "criterion": M.CRITERION},
                                    "logging": {"enabled": False}}))   # JSON is valid YAML
    env = {**BASE_ENV, "TENSOR_TRANSPORT": transport, "WORKER_PEERS": peers,
           "EXPERIMENT_CONFIG": str(cfg_path)}
    tag = f"{name}_p2p-{transport}" + (f"-nat-{nat}" if nat else "")
    procs = []
    if nat == "relay":
        procs.append(_spawn(["-m", "torchslicer", "relay", "--port", str(relay_port)], BASE_ENV,
                            LOGS / f"{tag}_relay.log"))
        env.update(TRANSPORT_RELAY=f"127.0.0.1:{relay_port}", TRANSPORT_NAT="true")
    procs += [_spawn(["run_p2p.py", name, str(p)],
                     {**env, "IS_DRIVER": "false", "WORKER_INDEX": str(i),
                      "WORKER_ADDRESS": f"{_host(i)}:{p}"},
                     LOGS / f"{tag}_w{i}.log", remote=_host(i) != LOCAL_HOST)
              for i, p in enumerate(ports) if i > 0]
    time.sleep(2)
    log = LOGS / f"{tag}_driver.log"
    driver = _spawn(["run_p2p.py", name, str(ports[0])],
                    {**env, "IS_DRIVER": "true", "WORKER_INDEX": "0",
                     "WORKER_ADDRESS": f"{LOCAL_HOST}:{ports[0]}",
                     **({"TRANSPORT_NAT": "true"} if nat else {})}, log)
    try:
        driver.wait(TIMEOUT)
    finally:
        _stop([driver, *procs])
    return _json_losses(log)


MODES = {
    "local": lambda n: inproc(n, "local"),
    "gpipe": lambda n: inproc(n, "gpipe"),
    "central-grpc": lambda n: central(n, "grpc"),
    "central-tcp": lambda n: central(n, "tcp"),
    "p2p-grpc": lambda n: p2p(n, "grpc"),
    "p2p-tcp": lambda n: p2p(n, "tcp"),
    "p2p-nat": lambda n: p2p(n, "tcp", nat="driver"),      # M6: driver behind NAT
    "p2p-relay": lambda n: p2p(n, "grpc", nat="relay"),    # M6: every node behind NAT
    "central-ipc": lambda n: central(n, "ipc"),            # M9: CUDA IPC (TCP fallback on CPU)
    "p2p-ipc": lambda n: p2p(n, "ipc"),
    "offload": lambda n: inproc(n, "offload"),
    "offload-gpipe": lambda n: inproc(n, "offload-gpipe"),
    "offload-disk": lambda n: inproc(n, "offload-disk"),
}
if GPU:   # the CPU optimizer placement only differs from "offload" with a device
    MODES["offload-cpuopt"] = lambda n: inproc(n, "offload-cpuopt")
if os.environ.get("MATRIX_NCCL") == "1":   # M9: opt-in, see the module docstring
    MODES["central-nccl"] = lambda n: central(n, "nccl")
    MODES["p2p-nccl"] = lambda n: p2p(n, "nccl")
if os.environ.get("MATRIX_MODES"):   # e.g. MATRIX_MODES=local,offload for a quick check
    MODES = {k: v for k, v in MODES.items() if k in os.environ["MATRIX_MODES"].split(",")}


# modes compared against another reference than the plain unsplit run
REFERENCE = {"offload-cpuopt": "reference-cpuopt"}


def verdict(ref, got, tol):
    if got == "SKIP":
        return "SKIP (refused by design)"
    if not got or len(got) != len(ref):
        return "FAIL (no result)"
    err = max(abs(a - b) for a, b in zip(ref, got))
    return f"{'OK' if err <= tol else 'DIFF'} (max|Δ|={err:.1e})"


def _kill_remote():
    subprocess.run(["ssh", "-o", "BatchMode=yes", *shlex.split(REMOTE), f"pkill -f {_MARKER} || true"])


if __name__ == "__main__":
    if REMOTE:   # a killed run must not leave remote workers holding the ports
        import atexit
        import signal
        atexit.register(_kill_remote)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))   # runs the finally/atexit cleanup
    models = sys.argv[1:] or M.MODELS
    results = {}
    for name in models:
        ref = inproc(name, "reference")
        row = {"reference": ref}
        print(f"{name:9s} reference {ref}", flush=True)
        for mode, fn in MODES.items():
            t0 = time.time()
            try:
                got = fn(name)
            except subprocess.TimeoutExpired:
                got = None
            r = ref
            if mode in REFERENCE:
                key = REFERENCE[mode]
                if key not in row:
                    row[key] = inproc(name, key)
                r = row[key]
            v = verdict(r, got, 1e-5) if r else "FAIL (no reference)"
            row[mode] = {"losses": got, "verdict": v}
            print(f"{name:9s} {mode:13s} {v:24s} {time.time() - t0:5.1f}s  {got}", flush=True)
        results[name] = row
    if not sys.argv[1:] and not os.environ.get("MATRIX_MODES"):   # only a full run replaces the recorded results
        tag = f"{M.DEVICE}_{M.OPTIMIZER['name']}"
        (HERE / "results" / f"results_{tag}.json").write_text(json.dumps(results, indent=1))
