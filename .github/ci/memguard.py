"""Run a command and kill it if the machine's memory use passes a limit.

    python .github/ci/memguard.py [--limit-gb 11] -- COMMAND ...

For self-hosted runners in containers, where going over the cgroup's memory limit can take
the whole container down: polls the cgroup's memory.current (or MemTotal - MemAvailable
outside a cgroup) every 0.2 s and kills the command's process group above the limit.
Exits with the command's exit code, or 137 when it was killed.
"""
import os
import signal
import subprocess
import sys
import time


def used_bytes() -> int:
    try:
        return int(open("/sys/fs/cgroup/memory.current").read())
    except OSError:
        info = {l.split(":")[0]: int(l.split()[1]) * 1024 for l in open("/proc/meminfo")}
        return info["MemTotal"] - info["MemAvailable"]


def main(argv) -> int:
    limit = 11.0
    if argv[:1] == ["--limit-gb"]:
        limit, argv = float(argv[1]), argv[2:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        print(__doc__)
        return 2
    proc = subprocess.Popen(argv, start_new_session=True)
    while proc.poll() is None:
        if used_bytes() > limit * 2**30:
            print(f"memguard: memory {used_bytes() / 2**30:.1f} GB > {limit} GB, killing {argv[0]}",
                  flush=True)
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return 137
        time.sleep(0.2)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
