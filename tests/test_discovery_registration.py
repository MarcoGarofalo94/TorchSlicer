from torchslicer.discovery.coordinator import CoordinatorDiscovery
from torchslicer.executors.worker import resolve_worker_address


def test_resolve_worker_address_defaults_to_hostname_and_port(monkeypatch):
    monkeypatch.delenv("WORKER_ADDRESS", raising=False)
    assert resolve_worker_address(50051, hostname="worker0") == "worker0:50051"


def test_resolve_worker_address_uses_env_override(monkeypatch):
    monkeypatch.setenv("WORKER_ADDRESS", "192.168.1.7:50061")
    assert resolve_worker_address(50051, hostname="worker0") == "192.168.1.7:50061"


def test_coordinator_discovery_registers_worker_advertised_address():
    discovery = CoordinatorDiscovery(run_id="run123")
    ok, worker_index, _ = discovery.handle_register(
        node_id="worker-a",
        address="192.168.1.7:50061",
        device="cpu",
        memory_mb=1024,
    )
    assert ok is True
    assert worker_index == 0

    nodes = discovery.discover(expected=1, timeout=0.1)
    assert len(nodes) == 1
    assert nodes[0].address == "192.168.1.7:50061"


def test_coordinator_discovery_reregistration_updates_address():
    discovery = CoordinatorDiscovery(run_id="run123")
    discovery.handle_register(
        node_id="worker-a",
        address="worker-a:50051",
        device="cpu",
        memory_mb=1024,
    )
    ok, worker_index, _ = discovery.handle_register(
        node_id="worker-a",
        address="192.168.1.7:50061",
        device="cpu",
        memory_mb=1024,
    )
    assert ok is True
    assert worker_index == 0

    nodes = discovery.discover(expected=1, timeout=0.1)
    assert nodes[0].address == "192.168.1.7:50061"
