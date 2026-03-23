import os
import sys
import socket

import grpc
import torch
from concurrent import futures

from torchslicer.executors.worker import (
    WorkerServicer,
    get_available_memory_mb,
    _GRPC_OPTS,
)
from torchslicer.transport.grpc.worker import worker_service_pb2_grpc
from torchslicer.discovery import NodeInfo, announce_to_coordinator
from torchslicer.monitor import tracer as _tracer


def serve():
    _tracer.auto_configure_if_env()

    port             = sys.argv[1] if len(sys.argv) > 1 else "50051"
    coordinator_addr = os.environ.get("COORDINATOR_ADDRESS", "coordinator:50054")
    hostname         = socket.gethostname()
    node_address     = os.environ.get("WORKER_ADDRESS", f"{hostname}:{port}")

    servicer = WorkerServicer()
    server   = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=_GRPC_OPTS)
    worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    servicer.set_server(server)
    print(f"[worker] started on port {port}  (hostname={hostname})")

    device    = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    memory_mb = get_available_memory_mb(device)
    node_info = NodeInfo(
        node_id=hostname, address=node_address, device=device, memory_mb=memory_mb
    )

    print(f"[discovery] registering with coordinator at {coordinator_addr} ...")
    try:
        result = announce_to_coordinator(coordinator_addr, node_info)
        print(f"[discovery] registered: run_id={result.run_id}, "
              f"worker_index={result.worker_index}")
    except RuntimeError as e:
        print(f"[discovery] FATAL: {e}")
        server.stop(0)
        sys.exit(1)

    server.wait_for_termination()
    print(f"[worker] {hostname} terminated cleanly")


if __name__ == '__main__':
    serve()
