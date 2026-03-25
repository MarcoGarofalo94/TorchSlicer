"""
Fault-tolerance test worker.

Identical to the standard worker except it supports deliberate crash injection
for testing DistributedExecutor's heartbeat + recovery path.

Environment variables (in addition to the standard worker vars):
  FAULT_KILL_EPOCH=N   Kill this process (SIGKILL) 2 s after responding to the
                       coordinator's save_checkpoint RPC for epoch N.  0 (default)
                       disables crash injection — worker behaves normally.

Why save_checkpoint and not get_stats?
  The coordinator only calls get_stats when profile.verbosity >= 1.  With the
  default verbosity=0, only the heartbeat pings get_stats (with epoch=-1), so
  the epoch number check never triggers.  save_checkpoint is called after EVERY
  epoch when fault_tolerance.enabled=True, and carries the real epoch number.

Usage:
  # Simulate crash after epoch 2:
  FAULT_KILL_EPOCH=2 python3 examples/train/centralized/GRPC/worker/main_ft_test.py 50051

  # No crash (normal worker):
  python3 examples/train/centralized/GRPC/worker/main_ft_test.py 50051
"""

import os
import signal
import socket
import sys
import threading
import time
from concurrent import futures

import grpc
import torch

from torchslicer.executors.worker import WorkerServicer, get_available_memory_mb, _GRPC_OPTS
from torchslicer.transport.grpc.worker import worker_service_pb2_grpc
from torchslicer.discovery import NodeInfo, announce_to_coordinator
from torchslicer.monitor import tracer as _tracer


_KILL_EPOCH = int(os.environ.get("FAULT_KILL_EPOCH", "0"))


class FaultTestWorkerServicer(WorkerServicer):
    """WorkerServicer with optional crash injection.

    Crash is triggered 2 s after responding to save_checkpoint for epoch >= KILL_EPOCH.
    The 2-second delay lets the RPC response (and checkpoint file) reach the
    coordinator before the process dies, so recovery has a clean checkpoint to
    load from for all workers including the one about to fail.
    """

    def save_checkpoint(self, request, context):
        result = super().save_checkpoint(request, context)
        if _KILL_EPOCH and request.epoch >= _KILL_EPOCH:
            print(
                f"[ft-test] SIMULATED CRASH — killing process 2 s after "
                f"epoch {request.epoch} checkpoint  (FAULT_KILL_EPOCH={_KILL_EPOCH})",
                flush=True,
            )
            threading.Thread(
                target=lambda: (time.sleep(2.0), os._exit(137)),
                daemon=True,
                name="fault-injector",
            ).start()
        return result


def serve():
    _tracer.auto_configure_if_env()

    port             = sys.argv[1] if len(sys.argv) > 1 else "50051"
    coordinator_addr = os.environ.get("COORDINATOR_ADDRESS", "coordinator:50054")
    hostname         = socket.gethostname()
    node_address     = os.environ.get("WORKER_ADDRESS", f"{hostname}:{port}")

    servicer = FaultTestWorkerServicer()
    server   = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=_GRPC_OPTS)
    worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    servicer.set_server(server)
    print(
        f"[worker] started on port {port}  hostname={hostname}  "
        f"FAULT_KILL_EPOCH={_KILL_EPOCH or 'disabled'}",
        flush=True,
    )

    device    = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    memory_mb = get_available_memory_mb(device)
    node_info = NodeInfo(
        node_id=hostname, address=node_address, device=device, memory_mb=memory_mb
    )

    print(f"[discovery] registering with coordinator at {coordinator_addr} ...", flush=True)
    try:
        result = announce_to_coordinator(coordinator_addr, node_info)
        print(
            f"[discovery] registered: run_id={result.run_id}  "
            f"worker_index={result.worker_index}",
            flush=True,
        )
    except RuntimeError as e:
        print(f"[discovery] FATAL: {e}", flush=True)
        server.stop(0)
        sys.exit(1)

    server.wait_for_termination()
    print(f"[worker] {hostname} terminated", flush=True)


if __name__ == "__main__":
    serve()
