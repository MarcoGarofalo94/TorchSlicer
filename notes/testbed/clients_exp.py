"""M14 client-node run on the testbed with per-client energy (tegrastats on every Nano).

    python notes/testbed/clients_exp.py NAME [--env K=V ...]

Coordinator + server worker on the AGX, clients on the 8 Nanos. Writes
notes/testbed/results/NAME.json: tb.run's summary (losses, per-epoch RESULT metrics, container
windows) + per client: idle W, mean W over its run, energy J, dynamic energy J (above idle).
"""
import datetime as dt
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb  # noqa: E402

NANOS = ["jon-001", "jon-003", "jon-004", "jon-005", "jon-006", "jon-007", "jon-008", "jon-009"]
NODES = ["jor-001", "jor-001"] + NANOS
LINE = re.compile(r"^(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d) .*VDD_IN (\d+)mW")


def parse(text, tz):
    """[(unix time, watts)]: tegrastats stamps whole seconds; spread a second's lines evenly."""
    rows = [(m.group(1), int(m.group(2))) for m in map(LINE.match, text.splitlines()) if m]
    out, i = [], 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j][0] == rows[i][0]:
            j += 1
        t0 = dt.datetime.strptime(rows[i][0] + " " + tz, "%m-%d-%Y %H:%M:%S %z").timestamp()
        for k in range(i, j):
            out.append((t0 + (k - i) / (j - i), rows[k][1] / 1000))
        i = j
    return out


def iso(s):
    """Docker's RFC 3339 (nanoseconds, Z) -> unix time; None for a container that never ran."""
    if not s or s.startswith("0001-"):
        return None
    head, _, frac = s.rstrip("Z").partition(".")
    return dt.datetime.fromisoformat(head + "+00:00").timestamp() + float("0." + (frac or "0"))


def energy(samples, start, end):
    inside = [(t, w) for t, w in samples if start <= t <= end]
    before = [w for t, w in samples if t < start - 1]
    e = sum(w * (t2 - t1) for (t1, w), (t2, _) in zip(inside, inside[1:]))
    idle = statistics.median(before) if before else float("nan")
    return {"idle_w": idle, "mean_w": statistics.mean(w for _, w in inside) if inside else None,
            "seconds": end - start, "energy_j": e, "dynamic_j": e - idle * (end - start)}


if __name__ == "__main__":
    name = sys.argv[1]
    envs = {"OMP_NUM_THREADS": "4"}
    a = sys.argv[2:]
    for i in range(0, len(a), 2):
        assert a[i] == "--env"
        k, v = a[i + 1].split("=", 1)
        envs[k] = v
    log = f"ts-test/out/teg_{name}.log"
    tz = tb.par(NANOS, lambda n: tb.ssh(n, "date +%z").stdout.strip())
    tb.par(NANOS, lambda n: tb.ssh(n, f"tegrastats --stop; rm -f $HOME/{log}; "
                                      f"tegrastats --start --interval 200 --logfile $HOME/{log}", sudo=True))
    time.sleep(10)                                   # idle baseline
    try:
        s = tb.run(name, "benchmarks/testbed/clients.py", "benchmarks/testbed/clients.yaml", NODES,
                   envs, {}, 1800, ["--python", "python"])
    finally:
        time.sleep(3)
        tb.par(NANOS, lambda n: tb.ssh(n, "tegrastats --stop", sudo=True))
    teg = tb.par(NANOS, lambda n: tb.ssh(n, f"cat $HOME/{log}", sudo=True).stdout)
    clients = {}
    for c, info in s["containers"].items():
        start, end = iso(info["started"]), iso(info["finished"])
        if info["node"] in NANOS and start and end:
            samples = parse(teg[info["node"]], tz[info["node"]])
            clients[c] = {"node": info["node"], **energy(samples, start, end)}
    s["clients"] = clients
    s["results"] = [json.loads(l[7:]) for l in s.pop("driver_log").splitlines() if l.startswith("RESULT ")]
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), "results", f"{name}.json")
    json.dump(s, open(out, "w"), indent=1)
    print(json.dumps({"name": name, "exit": s["exit"], "history": s["history"], "wall_s": s["wall_s"],
                      "epoch_s": [r.get("duration_s") for r in s["results"]],
                      "client_dynamic_j": {v["node"]: round(v["dynamic_j"], 1) for v in clients.values()},
                      "client_mean_w": {v["node"]: round(v["mean_w"] or 0, 2) for v in clients.values()}}))
