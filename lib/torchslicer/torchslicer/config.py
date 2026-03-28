"""
RunConfig — single source of truth for all experiment parameters.

Load priority (highest wins):
    1. Python API kwargs — pass directly to SlicedModel.train() or executor
    2. YAML experiment file — --config path/to/run.yaml or EXPERIMENT_CONFIG env var
    3. Environment variables — standard env vars or .env file read by Docker Compose
    4. Defaults in code

Example::

    # Quick Python API — flat kwargs
    cfg = RunConfig.create(epochs=20, n_workers=4, transport="tcp", device="cuda")

    # Nested Python API
    cfg = RunConfig()
    cfg.training.epochs = 20

    # From YAML
    cfg = RunConfig.from_yaml("experiments/resnet18_4gpu.yaml")

    # From environment (Docker Compose / .env)
    cfg = RunConfig.from_env()

    # Merged (recommended in coordinator/main.py)
    cfg = RunConfig.load("experiments/resnet18_4gpu.yaml")  # YAML + env overrides

    # Save resolved config for reproducibility
    cfg.to_yaml("runs/my_run/resolved_config.yaml")
"""

import dataclasses
import datetime
import os
from dataclasses import dataclass, field
from typing import Optional


def _default_run_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


@dataclass
class TrainingConfig:
    epochs:           int  = 10
    optimizer:        dict = field(default_factory=lambda: {
        "name": "SGD",
        "params": {"lr": 0.05, "momentum": 0.9, "weight_decay": 5e-4},
    })
    criterion:        dict = field(default_factory=lambda: {
        "name": "CrossEntropyLoss", "params": {},
    })
    mixed_precision:  bool = False


@dataclass
class PipelineConfig:
    use_gpipe: bool = False
    n_micro:   int  = 4


@dataclass
class DiscoveryConfig:
    backend:    str   = "coordinator"  # "coordinator" | "static"
    n_workers:  int   = 2
    timeout:    float = 60.0
    peers:      list  = field(default_factory=list)   # for "static" backend: ["host:port", ...]
    tag_filter: list  = field(default_factory=list)   # only accept workers with ALL these tags
    registration_max_attempts:  int   = 30
    registration_delay_s:       float = 3.0
    registration_rpc_timeout_s: float = 5.0
    watchdog_interval_s:        float = 15.0

    def __post_init__(self):
        if self.peers is None:
            self.peers = []
        if self.tag_filter is None:
            self.tag_filter = []


@dataclass
class StartupConfig:
    worker_init_max_attempts:  int   = 20
    worker_init_delay_s:       float = 1.0
    worker_init_rpc_timeout_s: float = 5.0


@dataclass
class CheckpointConfig:
    enabled:    bool          = False
    dir:        str           = "./checkpoints"
    save_every: str           = "epoch"   # "epoch" | "never"
    resume:     Optional[str] = None      # path to run_state.json to resume from


@dataclass
class LoggingConfig:
    enabled: bool = True
    dir:     str  = "./runs"    # parent dir; each run writes to {dir}/{run_id}/
    level:   str  = "INFO"


@dataclass
class ProfileConfig:
    verbosity: int  = 0      # 0=off  1=epoch totals  2=phase breakdown  3=per-batch
    memory:    bool = False  # GPU memory snapshots per phase (adds ~CUDA sync per phase)


@dataclass
class TransportConfig:
    tensor:             str = "grpc"   # "grpc" | "tcp"
    tensor_port_offset: int = 1


@dataclass
class FaultToleranceConfig:
    enabled:              bool  = False
    heartbeat_interval_s: float = 5.0
    ping_timeout_s:       float = 3.0
    max_retries:          int   = 3


@dataclass
class NetworkConfig:
    """Infrastructure config: device, addresses, ports, keep-alive.

    All fields are configurable via YAML (``network:`` section) or env vars.
    In Docker deployments these are typically set via environment variables
    rather than committed YAML values.

    Fields:
        device:                   "auto" | "cuda" | "cpu" | "mps"  (env: DEVICE)
        coordinator_address:      address workers use to reach the coordinator
                                  (env: COORDINATOR_ADDRESS)
        coordinator_bind_address: local address the coordinator binds to
                                  (env: COORDINATOR_BIND_ADDRESS)
        worker_port:              gRPC port each worker listens on  (env: PORT)
        worker_address:           advertised address for this worker; None → auto
                                  (env: WORKER_ADDRESS)
        worker_tags:              capability tags for this worker   (env: WORKER_TAGS)
        keep_alive:               if False the coordinator exits after training
                                  (env: KEEP_ALIVE)
    """
    device:                   str           = "auto"
    coordinator_address:      Optional[str] = None             # None → no coordinator registration
    coordinator_bind_address: str           = "0.0.0.0:50054"
    worker_port:              int           = 50051
    worker_address:           Optional[str] = None             # None → auto hostname:worker_port
    worker_tags:              list          = field(default_factory=list)
    keep_alive:               bool          = True

    def __post_init__(self):
        if self.worker_tags is None:
            self.worker_tags = []


@dataclass
class RunConfig:
    run_id:          str                  = field(default_factory=_default_run_id)
    training:        TrainingConfig       = field(default_factory=TrainingConfig)
    pipeline:        PipelineConfig       = field(default_factory=PipelineConfig)
    discovery:       DiscoveryConfig      = field(default_factory=DiscoveryConfig)
    startup:         StartupConfig        = field(default_factory=StartupConfig)
    checkpoint:      CheckpointConfig     = field(default_factory=CheckpointConfig)
    logging:         LoggingConfig        = field(default_factory=LoggingConfig)
    profile:         ProfileConfig        = field(default_factory=ProfileConfig)
    transport:       TransportConfig      = field(default_factory=TransportConfig)
    fault_tolerance: FaultToleranceConfig = field(default_factory=FaultToleranceConfig)
    network:         NetworkConfig        = field(default_factory=NetworkConfig)

    # ── factory: flat Python API ──────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        *,
        run_id: str = None,
        # training
        epochs: int = None,
        optimizer: str = None,
        lr: float = None,
        criterion: str = None,
        mixed_precision: bool = None,
        # pipeline
        use_gpipe: bool = None,
        n_micro: int = None,
        # discovery
        n_workers: int = None,
        # transport
        transport: str = None,
        # network
        device: str = None,
        coordinator_address: str = None,
        worker_port: int = None,
        worker_tags: list = None,
        keep_alive: bool = None,
    ) -> "RunConfig":
        """Build a RunConfig from flat keyword arguments — the fastest Python API.

        Unspecified fields keep their defaults.  Nest into sub-configs directly
        for full control::

            cfg = RunConfig.create(epochs=10, n_workers=2, transport="tcp", device="cuda")
            cfg.checkpoint.enabled = True  # fine to mix with nested mutation
        """
        cfg = cls()
        if run_id is not None:              cfg.run_id = run_id
        if epochs is not None:              cfg.training.epochs = epochs
        if optimizer is not None:           cfg.training.optimizer["name"] = optimizer
        if lr is not None:                  cfg.training.optimizer["params"]["lr"] = lr
        if criterion is not None:           cfg.training.criterion["name"] = criterion
        if mixed_precision is not None:     cfg.training.mixed_precision = mixed_precision
        if use_gpipe is not None:           cfg.pipeline.use_gpipe = use_gpipe
        if n_micro is not None:             cfg.pipeline.n_micro = n_micro
        if n_workers is not None:           cfg.discovery.n_workers = n_workers
        if transport is not None:           cfg.transport.tensor = transport
        if device is not None:              cfg.network.device = device
        if coordinator_address is not None: cfg.network.coordinator_address = coordinator_address
        if worker_port is not None:         cfg.network.worker_port = worker_port
        if worker_tags is not None:         cfg.network.worker_tags = worker_tags
        if keep_alive is not None:          cfg.network.keep_alive = keep_alive
        return cfg

    # ── loaders ───────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        """Load from a YAML experiment config file."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config files. "
                "Install with: pip install pyyaml"
            )
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def from_env(cls) -> "RunConfig":
        """Load from environment variables (reads Docker Compose / .env vars)."""
        cfg = cls()
        _apply_env_overrides(cfg)
        return cfg

    @classmethod
    def load(cls, config_path: str = None) -> "RunConfig":
        """
        Load with merge priority: YAML > env vars > defaults.
        config_path overrides the EXPERIMENT_CONFIG env var if both are set.
        """
        path = config_path or os.environ.get("EXPERIMENT_CONFIG")
        if path:
            cfg = cls.from_yaml(path)
            _apply_env_overrides(cfg)
            return cfg
        return cls.from_env()

    # ── serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a plain dict representation of this config (deep copy)."""
        return dataclasses.asdict(self)

    def to_yaml(self, path: str) -> None:
        """Write the resolved config to a YAML file for reproducibility."""
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML serialization. "
                "Install with: pip install pyyaml"
            )
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    # ── internal ──────────────────────────────────────────────────────────────

    @classmethod
    def _from_dict(cls, data: dict) -> "RunConfig":
        cfg = cls()
        _merge_dict(cfg, data)
        return cfg


# ── dict → dataclass helper ───────────────────────────────────────────────────

def _merge_dict(target, data: dict) -> None:
    """Apply dict values onto a dataclass instance in-place, recursing into sub-dataclasses.

    Unknown keys are silently ignored.  ``null`` YAML values for list fields are
    coerced to empty lists so that ``peers: null`` never breaks list operations.
    """
    for key, val in data.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if dataclasses.is_dataclass(current) and isinstance(val, dict):
            _merge_dict(current, val)
        else:
            if val is None and isinstance(current, list):
                val = []
            setattr(target, key, val)


# ── env var parsing helpers ───────────────────────────────────────────────────

def _parse_bool(value: str) -> bool:
    return value.lower() in ("1", "true", "yes")


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


_EnvOverride = tuple[str, object]

_ENV_OVERRIDES: tuple[_EnvOverride, ...] = (
    # ── top level ──────────────────────────────────────────────────────────────
    ("RUN_ID",            lambda cfg, v: setattr(cfg,          "run_id",           v)),
    # ── training ───────────────────────────────────────────────────────────────
    ("EPOCHS",            lambda cfg, v: setattr(cfg.training, "epochs",           int(v))),
    ("MIXED_PRECISION",   lambda cfg, v: setattr(cfg.training, "mixed_precision",  _parse_bool(v))),
    # ── pipeline ───────────────────────────────────────────────────────────────
    ("USE_GPIPE",         lambda cfg, v: setattr(cfg.pipeline, "use_gpipe",        _parse_bool(v))),
    ("N_MICRO",           lambda cfg, v: setattr(cfg.pipeline, "n_micro",          int(v))),
    # ── discovery ──────────────────────────────────────────────────────────────
    ("N_WORKERS",                         lambda cfg, v: setattr(cfg.discovery, "n_workers",                  int(v))),
    ("DISCOVERY_BACKEND",                 lambda cfg, v: setattr(cfg.discovery, "backend",                    v)),
    ("DISCOVERY_TIMEOUT",                 lambda cfg, v: setattr(cfg.discovery, "timeout",                    float(v))),
    ("WORKER_PEERS",                      lambda cfg, v: setattr(cfg.discovery, "peers",                      _parse_csv(v))),
    ("WORKER_TAG_FILTER",                 lambda cfg, v: setattr(cfg.discovery, "tag_filter",                 _parse_csv(v))),
    ("DISCOVERY_REGISTRATION_MAX_ATTEMPTS", lambda cfg, v: setattr(cfg.discovery, "registration_max_attempts", int(v))),
    ("DISCOVERY_REGISTRATION_DELAY",      lambda cfg, v: setattr(cfg.discovery, "registration_delay_s",      float(v))),
    ("DISCOVERY_REGISTRATION_RPC_TIMEOUT",lambda cfg, v: setattr(cfg.discovery, "registration_rpc_timeout_s",float(v))),
    ("DISCOVERY_WATCHDOG_INTERVAL",       lambda cfg, v: setattr(cfg.discovery, "watchdog_interval_s",       float(v))),
    # ── startup ────────────────────────────────────────────────────────────────
    ("WORKER_INIT_MAX_ATTEMPTS", lambda cfg, v: setattr(cfg.startup, "worker_init_max_attempts",  int(v))),
    ("WORKER_INIT_DELAY",        lambda cfg, v: setattr(cfg.startup, "worker_init_delay_s",       float(v))),
    ("WORKER_INIT_RPC_TIMEOUT",  lambda cfg, v: setattr(cfg.startup, "worker_init_rpc_timeout_s", float(v))),
    # ── checkpoint ─────────────────────────────────────────────────────────────
    ("CHECKPOINT_ENABLED",    lambda cfg, v: setattr(cfg.checkpoint, "enabled",    _parse_bool(v))),
    ("CHECKPOINT_DIR",        lambda cfg, v: setattr(cfg.checkpoint, "dir",        v)),
    ("CHECKPOINT_SAVE_EVERY", lambda cfg, v: setattr(cfg.checkpoint, "save_every", v)),
    ("CHECKPOINT_RESUME",     lambda cfg, v: setattr(cfg.checkpoint, "resume",     v or None)),
    # ── logging ────────────────────────────────────────────────────────────────
    ("LOG_ENABLED", lambda cfg, v: setattr(cfg.logging, "enabled", _parse_bool(v))),
    ("LOG_DIR",     lambda cfg, v: setattr(cfg.logging, "dir",     v)),
    ("LOG_LEVEL",   lambda cfg, v: setattr(cfg.logging, "level",   v)),
    # ── profile ────────────────────────────────────────────────────────────────
    ("PROFILE_VERBOSITY", lambda cfg, v: setattr(cfg.profile, "verbosity", int(v))),
    ("PROFILE_MEMORY",    lambda cfg, v: setattr(cfg.profile, "memory",    _parse_bool(v))),
    # ── transport ──────────────────────────────────────────────────────────────
    ("TENSOR_TRANSPORT",   lambda cfg, v: setattr(cfg.transport, "tensor",             v.strip().lower())),
    ("TENSOR_PORT_OFFSET", lambda cfg, v: setattr(cfg.transport, "tensor_port_offset", int(v))),
    # ── fault tolerance ────────────────────────────────────────────────────────
    ("FAULT_TOLERANCE_ENABLED", lambda cfg, v: setattr(cfg.fault_tolerance, "enabled",              _parse_bool(v))),
    ("HEARTBEAT_INTERVAL",      lambda cfg, v: setattr(cfg.fault_tolerance, "heartbeat_interval_s", float(v))),
    ("HEARTBEAT_TIMEOUT",       lambda cfg, v: setattr(cfg.fault_tolerance, "ping_timeout_s",       float(v))),
    ("FAULT_MAX_RETRIES",       lambda cfg, v: setattr(cfg.fault_tolerance, "max_retries",          int(v))),
    # ── network ────────────────────────────────────────────────────────────────
    ("DEVICE",                   lambda cfg, v: setattr(cfg.network, "device",                   v)),
    ("COORDINATOR_ADDRESS",      lambda cfg, v: setattr(cfg.network, "coordinator_address",      v)),
    ("COORDINATOR_BIND_ADDRESS", lambda cfg, v: setattr(cfg.network, "coordinator_bind_address", v)),
    ("PORT",                     lambda cfg, v: setattr(cfg.network, "worker_port",              int(v))),
    ("WORKER_ADDRESS",           lambda cfg, v: setattr(cfg.network, "worker_address",           v)),
    ("WORKER_TAGS",              lambda cfg, v: setattr(cfg.network, "worker_tags",              _parse_csv(v))),
    ("KEEP_ALIVE",               lambda cfg, v: setattr(cfg.network, "keep_alive",               _parse_bool(v))),
)


def _apply_env_overrides(cfg: RunConfig) -> None:
    """Apply env vars on top of an existing config (env wins over lower-priority sources)."""
    for env_var, apply_value in _ENV_OVERRIDES:
        value = os.environ.get(env_var)
        if value:
            apply_value(cfg, value)
