"""
Internals for ts.simulate() — multi-process local split learning without Docker.

The subprocess entry-point (_run_worker_subprocess) must live at module level
so multiprocessing can pickle and spawn it.
"""

import os
import socket


def _find_free_port() -> int:
    """Bind to port 0, let the OS assign a free port, return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _run_worker_subprocess(port: int, coordinator_addr: str, device: str,
                            worker_index: int = 0,
                            tensor_transport: str = "grpc",
                            log_level: str = "WARNING") -> None:
    """Worker subprocess entry-point.

    Sets the required environment variables and blocks in ``run_worker()``
    until the coordinator signals shutdown.
    """
    os.environ["PORT"]                = str(port)
    os.environ["COORDINATOR_ADDRESS"] = coordinator_addr
    os.environ["DEVICE"]              = device
    os.environ["WORKER_ADDRESS"]      = f"127.0.0.1:{port}"
    os.environ["TENSOR_TRANSPORT"]    = tensor_transport
    os.environ["LOG_LEVEL"]           = log_level

    from torchslicer.executors.worker import run_worker
    import socket
    # Each subprocess needs a unique node_id so CoordinatorDiscovery treats
    # them as separate workers (default node_id is the hostname, which is the
    # same for all subprocesses on the same machine).
    node_id = f"{socket.gethostname()}-sim-{worker_index}"
    run_worker(
        port=port,
        coordinator_addr=coordinator_addr,
        worker_address=f"127.0.0.1:{port}",
        device=device,
        node_id=node_id,
    )
