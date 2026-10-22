"""Run ab_clocks.sh (idle-gap benchmark: stock clocks vs jetson_clocks, always restored) on one Jetson.

    python3 notes/testbed/run_ab_clocks.py jor-001|jon-001..jon-009

The script stores the clocks with `jetson_clocks --store`, benchmarks, applies `jetson_clocks`,
benchmarks again, and restores the stored settings on exit (trap), printing them.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tb  # noqa: E402

node = sys.argv[1]
if node not in tb.HOSTS or node == "dell-g3":
    raise SystemExit(f"{node}: not a Jetson in the inventory")
tb.scp(node, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ab_clocks.sh"), "ts-test/ab_clocks.sh")
r = tb.ssh(node, f"HOME_DIR={tb.home(node)} sh {tb.home(node)}/ts-test/ab_clocks.sh", sudo=True, timeout=900)
print(f"== {node} rc={r.returncode}")
print(r.stdout.replace('"device": "Orin", ', ""))
if r.returncode:
    print(r.stderr[-2000:])
