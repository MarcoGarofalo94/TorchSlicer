#!/bin/sh
# A/B of benchmarks/testbed/idle_gap.py with stock clocks vs jetson_clocks; always restores.
# Run as root on a Jetson with HOME_DIR=<the user's home holding ts-test> (run_ab_clocks.py does).
BK=/root/ts_clocks_backup.conf
B="docker run --rm --runtime nvidia -v $HOME_DIR/ts-test:/t -w /t/TorchSlicer mesl-jetson:latest python benchmarks/testbed/idle_gap.py"
jetson_clocks --store $BK >/dev/null 2>&1 || { echo "store failed, not changing anything"; exit 1; }
trap 'jetson_clocks --restore $BK >/dev/null 2>&1; rm -f $BK; echo "== restored"; jetson_clocks --show | grep -E "^GPU|^EMC"' EXIT
echo "== stock"
$B --slice server 0 60 100 2>&1 | grep gap_ms
$B --slice client 0 60 200 2>&1 | grep gap_ms
jetson_clocks
echo "== jetson_clocks"
$B --slice server 0 60 100 2>&1 | grep gap_ms
$B --slice client 0 60 200 2>&1 | grep gap_ms
echo "temp (millidegrees C): $(cat /sys/class/thermal/thermal_zone0/temp)"
