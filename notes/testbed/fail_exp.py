"""Real node failures on the testbed: kill one of our containers mid-run (docker kill, i.e. a
crashed process; nothing on the board changes), optionally restart it, and time what follows.

    python notes/testbed/fail_exp.py NAME --kill CONTAINER --after S [--restart S] [--servers N] [--env K=V ...]

CONTAINER is the launch script's name (client-3, worker-1, ...). S counts from the coordinator's
"client nodes count=" line (training starts). --restart S: `docker start` it again S seconds after
the kill (the same script: a new process registers and asks to join). --servers N: N server
workers on the AGX (slices 1..N). A memory watch polls every node; if any has less than
MIN_AVAIL_MB available it kills all our containers and the run fails.
Writes notes/testbed/results/fail_NAME.json.
"""
import datetime as dt
import json
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb  # noqa: E402

NANOS = ["jon-001", "jon-003", "jon-004", "jon-005", "jon-006", "jon-007", "jon-008", "jon-009"]
MIN_AVAIL_MB = 600
STAMP = re.compile(r"^(\d\d:\d\d:\d\d) ")



def avail_mb(node):
    r = tb.ssh(node, "awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo", timeout=20)
    return int(r.stdout.strip() or -1)


def main():
    name = sys.argv[1]
    a = sys.argv[2:]
    kill, after, restart, servers, envs = None, None, None, 1, {"OMP_NUM_THREADS": "4"}
    i = 0
    while i < len(a):
        k, v = a[i], a[i + 1]
        if k == "--kill":
            kill = v
        elif k == "--after":
            after = float(v)
        elif k == "--restart":
            restart = float(v)
        elif k == "--servers":
            servers = int(v)
        elif k == "--env":
            kk, vv = v.split("=", 1)
            envs[kk] = vv
        i += 2
    envs["N_WORKERS"] = str(1 + servers)
    os.environ["N_WORKERS"] = envs["N_WORKERS"]      # the launcher renders the node list here
    nodes = ["jor-001"] + ["jor-001"] * servers + NANOS
    # script name -> node: coordinator and worker-* on the AGX, client-k on NANOS[k]
    where = {"coordinator": "jor-001", **{f"worker-{j}": "jor-001" for j in range(servers)},
             **{f"client-{k}": n for k, n in enumerate(NANOS)}}
    cname = lambda s: f"ts-{name}-{s}"
    events, stop = {"mem_min_mb": {}}, threading.Event()

    def mem_watch():
        while not stop.is_set():
            for n, m in tb.par(sorted(set(nodes)), avail_mb).items():
                if m >= 0:
                    events["mem_min_mb"][n] = min(m, events["mem_min_mb"].get(n, m))
                if 0 <= m < MIN_AVAIL_MB:
                    events["mem_abort"] = f"{n} {m} MB"
                    tb.par(sorted(set(nodes)), lambda x: tb.ssh(
                        x, "docker ps -q --filter name=^ts- | xargs -r docker kill", sudo=True))
                    return
            stop.wait(5)

    def killer():
        coord = cname("coordinator")
        while not stop.is_set():          # wait for training to start
            r = tb.ssh("jor-001", f"docker logs {coord} 2>&1 | grep -m1 'client nodes count='", sudo=True, timeout=30)
            if r.stdout.strip():
                events["train_start"] = time.time()
                break
            stop.wait(1)
        if stop.wait(after):
            return
        node = where[kill]
        r = tb.ssh(node, f"docker kill {cname(kill)} >/dev/null && date +%s.%N", sudo=True, check=True)
        events["killed"] = float(r.stdout.strip())          # the node's clock (NTP, as the AGX's)
        print(f"killed {cname(kill)} on {node}", flush=True)
        if restart is not None and not stop.wait(restart):
            r = tb.ssh(node, f"docker start {cname(kill)} >/dev/null && date +%s.%N", sudo=True, check=True)
            events["restarted"] = float(r.stdout.strip())
            print(f"restarted {cname(kill)} on {node}", flush=True)

    threads = [threading.Thread(target=mem_watch, daemon=True)]
    if kill:
        threads.append(threading.Thread(target=killer, daemon=True))
    for t in threads:
        t.start()
    try:
        s = tb.run(name, "benchmarks/testbed/clients.py", "benchmarks/testbed/clients.yaml", nodes,
                   envs, {}, 1800, ["--python", "python"])
    finally:
        stop.set()
    log = s.pop("driver_log")
    s["results"] = [json.loads(l[7:]) for l in log.splitlines() if l.startswith("RESULT ")]
    s["events"] = events
    s["kill"], s["after"], s["restart"], s["servers"], s["envs"] = kill, after, restart, servers, envs
    # coordinator lines about the failure, with their wall-clock stamps
    keys = ("lost", "left", "joined", "recovery", "WorkerFailure", "failure", "retry")
    s["coordinator_events"] = [l[:300] for l in log.splitlines()
                               if STAMP.match(l) and any(k in l for k in keys) and "idle worker shutdown" not in l]
    # seconds from the kill to each line (container clocks are UTC; 1 s resolution)
    if "killed" in events:
        day = dt.datetime.fromtimestamp(events["killed"], dt.timezone.utc).date()
        for l in s["coordinator_events"]:
            t = dt.datetime.combine(day, dt.time.fromisoformat(STAMP.match(l).group(1)), dt.timezone.utc)
            s.setdefault("since_kill_s", []).append((round(t.timestamp() - events["killed"], 1), l[:160]))
    out = os.path.join(os.path.dirname(__file__), "results", f"fail_{name}.json")
    json.dump(s, open(out, "w"), indent=1)
    print(json.dumps({"name": name, "exit": s["exit"], "wall_s": s["wall_s"], "history": s["history"],
                      "epoch_s": [r.get("duration_s") for r in s["results"]],
                      "mem_min_mb": events["mem_min_mb"], "mem_abort": events.get("mem_abort")}))
    for dtk, l in s.get("since_kill_s", [(None, l) for l in s["coordinator_events"]]):
        print(f"  {dtk:>6}  {l}" if dtk is not None else "  " + l)


if __name__ == "__main__":
    main()
