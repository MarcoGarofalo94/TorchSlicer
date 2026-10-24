"""Planning variants on Nano (driver) -> AGX -> Dell; one tb run each, plan + timing to results/plan_*.json."""
import json, re, statistics as st, sys
import tb

NODES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["jon-004", "jor-001", "dell-g3"]
TAG = sys.argv[2] if len(sys.argv) > 2 else "nad"
VARIANTS = {"A_uniform": {"PLAN_STRATEGY": "uniform", "PLANNING_HETEROGENEOUS": "0"},
            "B_flops_even": {"PLAN_STRATEGY": "flops", "PLANNING_HETEROGENEOUS": "0"},
            "C_flops_speeds": {"PLAN_STRATEGY": "flops", "PLANNING_HETEROGENEOUS": "1", "PLANNING_MEASURE_LINKS": "0"},
            "D_flops_links": {"PLAN_STRATEGY": "flops", "PLANNING_HETEROGENEOUS": "1", "PLANNING_MEASURE_LINKS": "1"}}
for name, env in VARIANTS.items():
    s = tb.run(f"plan-{TAG}-{name}", "benchmarks/testbed/plan.py", "benchmarks/testbed/plan_p2p.yaml", NODES,
               {"OMP_NUM_THREADS": "4", **env}, {}, 1500, ["--python", "python"])
    log = s.pop("driver_log")
    plan = re.findall(r"plan slice=(\d+) address=(\S+) device=(\S+) gflops=([\d.]+).*?nodes=(\S+) params=(\d+)", log)
    links = re.findall(r"link .*", log)[:4]
    res = [json.loads(l[7:]) for l in log.splitlines() if l.startswith("RESULT ")]
    steady = [r["ms_per_batch"] for r in res[1:]]
    out = {"variant": name, "nodes": NODES, "plan": plan, "results": res,
           "ms_per_batch": st.mean(steady) if steady else None, "exit": s["exit"]}
    json.dump(out, open(f"results/plan_{TAG}_{name}.json", "w"), indent=1)
    print(name, "exit", s["exit"], "ms/batch", round(out["ms_per_batch"] or -1, 1),
          "plan", [(p[0], p[2], p[3], p[4], p[5]) for p in plan],
          "cut MB/batch", [round(b / 20 / 1e6, 2) for b in (res[-1]["bytes_fwd"] if res else [])], flush=True)
