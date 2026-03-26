"""
RunConfig — single source of truth for all experiment parameters.

Load priority (highest wins):
    1. Python API kwargs — pass directly to SlicedModel.train() or executor
    2. YAML experiment file — --config path/to/run.yaml or EXPERIMENT_CONFIG env var
    3. Environment variables — standard env vars or .env file read by Docker Compose
    4. Defaults in code

Example::

    # From code
    cfg = RunConfig()
    cfg.training.epochs = 20

    # From YAML
    cfg = RunConfig.from_yaml("experiments/resnet18_4gpu.yaml")

    # From environment (Docker Compose / .env)
    cfg = RunConfig.from_env()

    # Merged (recommended in coordinator/main.py)
    cfg = RunConfig.load("experiments/resnet18_4gpu.yaml")  # YAML + env overrides
"""

import datetime
import os
from dataclasses import dataclass, field
from typing import Callable, Optional


def _default_run_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


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
    registration_max_attempts: int = 30
    registration_delay_s: float = 3.0
    registration_rpc_timeout_s: float = 5.0
    watchdog_interval_s: float = 15.0


@dataclass
class StartupConfig:
    worker_init_max_attempts: int = 20
    worker_init_delay_s: float = 1.0
    worker_init_rpc_timeout_s: float = 5.0


@dataclass
class CheckpointConfig:
    enabled:    bool            = False
    dir:        str             = "./checkpoints"
    save_every: str             = "epoch"   # "epoch" | "never"
    resume:     Optional[str]   = None      # path to run_state.json to resume from


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
class FaultToleranceConfig:
    enabled:              bool  = False  # enable heartbeat monitoring and auto-recovery
    heartbeat_interval_s: float = 5.0   # seconds between pings
    ping_timeout_s:       float = 3.0   # gRPC call timeout for the liveness check
    max_retries:          int   = 3     # max epoch retries after a worker failure


@dataclass
class RunConfig:
    run_id:           str                  = field(default_factory=_default_run_id)
    training:         TrainingConfig       = field(default_factory=TrainingConfig)
    pipeline:         PipelineConfig       = field(default_factory=PipelineConfig)
    discovery:        DiscoveryConfig      = field(default_factory=DiscoveryConfig)
    startup:          StartupConfig        = field(default_factory=StartupConfig)
    checkpoint:       CheckpointConfig     = field(default_factory=CheckpointConfig)
    logging:          LoggingConfig        = field(default_factory=LoggingConfig)
    profile:          ProfileConfig        = field(default_factory=ProfileConfig)
    fault_tolerance:  FaultToleranceConfig = field(default_factory=FaultToleranceConfig)

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

    # ── internal ──────────────────────────────────────────────────────────────

    @classmethod
    def _from_dict(cls, data: dict) -> "RunConfig":
        cfg = cls()

        if v := data.get("run_id"):
            cfg.run_id = v

        if t := data.get("training"):
            cfg.training = TrainingConfig(
                epochs          = t.get("epochs",          cfg.training.epochs),
                optimizer       = t.get("optimizer",       cfg.training.optimizer),
                criterion       = t.get("criterion",       cfg.training.criterion),
                mixed_precision = t.get("mixed_precision", cfg.training.mixed_precision),
            )

        if p := data.get("pipeline"):
            cfg.pipeline = PipelineConfig(
                use_gpipe = p.get("use_gpipe", cfg.pipeline.use_gpipe),
                n_micro   = p.get("n_micro",   cfg.pipeline.n_micro),
            )

        if d := data.get("discovery"):
            cfg.discovery = DiscoveryConfig(
                backend    = d.get("backend",    cfg.discovery.backend),
                n_workers  = d.get("n_workers",  cfg.discovery.n_workers),
                timeout    = d.get("timeout",    cfg.discovery.timeout),
                peers      = d.get("peers",      cfg.discovery.peers) or [],
                tag_filter = d.get("tag_filter", cfg.discovery.tag_filter) or [],
                registration_max_attempts = d.get(
                    "registration_max_attempts", cfg.discovery.registration_max_attempts),
                registration_delay_s = d.get(
                    "registration_delay_s", cfg.discovery.registration_delay_s),
                registration_rpc_timeout_s = d.get(
                    "registration_rpc_timeout_s", cfg.discovery.registration_rpc_timeout_s),
                watchdog_interval_s = d.get(
                    "watchdog_interval_s", cfg.discovery.watchdog_interval_s),
            )

        if s := data.get("startup"):
            cfg.startup = StartupConfig(
                worker_init_max_attempts = s.get(
                    "worker_init_max_attempts", cfg.startup.worker_init_max_attempts),
                worker_init_delay_s = s.get(
                    "worker_init_delay_s", cfg.startup.worker_init_delay_s),
                worker_init_rpc_timeout_s = s.get(
                    "worker_init_rpc_timeout_s", cfg.startup.worker_init_rpc_timeout_s),
            )

        if c := data.get("checkpoint"):
            cfg.checkpoint = CheckpointConfig(
                enabled    = c.get("enabled",    cfg.checkpoint.enabled),
                dir        = c.get("dir",        cfg.checkpoint.dir),
                save_every = c.get("save_every", cfg.checkpoint.save_every),
                resume     = c.get("resume",     cfg.checkpoint.resume),
            )

        if lg := data.get("logging"):
            cfg.logging = LoggingConfig(
                enabled = lg.get("enabled", cfg.logging.enabled),
                dir     = lg.get("dir",     cfg.logging.dir),
                level   = lg.get("level",   cfg.logging.level),
            )

        if pr := data.get("profile"):
            cfg.profile = ProfileConfig(
                verbosity = pr.get("verbosity", cfg.profile.verbosity),
                memory    = pr.get("memory",    cfg.profile.memory),
            )

        if ft := data.get("fault_tolerance"):
            cfg.fault_tolerance = FaultToleranceConfig(
                enabled              = ft.get("enabled",              cfg.fault_tolerance.enabled),
                heartbeat_interval_s = ft.get("heartbeat_interval_s", cfg.fault_tolerance.heartbeat_interval_s),
                ping_timeout_s       = ft.get("ping_timeout_s",       cfg.fault_tolerance.ping_timeout_s),
                max_retries          = ft.get("max_retries",          cfg.fault_tolerance.max_retries),
            )

        return cfg


def _parse_bool(value: str) -> bool:
    return value.lower() in ("1", "true", "yes")


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


_EnvOverride = tuple[str, Callable[[RunConfig, str], None]]

_ENV_OVERRIDES: tuple[_EnvOverride, ...] = (
    ("RUN_ID", lambda cfg, value: setattr(cfg, "run_id", value)),
    ("EPOCHS", lambda cfg, value: setattr(cfg.training, "epochs", int(value))),
    ("MIXED_PRECISION", lambda cfg, value: setattr(cfg.training, "mixed_precision", _parse_bool(value))),
    ("USE_GPIPE", lambda cfg, value: setattr(cfg.pipeline, "use_gpipe", _parse_bool(value))),
    ("N_MICRO", lambda cfg, value: setattr(cfg.pipeline, "n_micro", int(value))),
    ("N_WORKERS", lambda cfg, value: setattr(cfg.discovery, "n_workers", int(value))),
    ("DISCOVERY_BACKEND", lambda cfg, value: setattr(cfg.discovery, "backend", value)),
    ("DISCOVERY_TIMEOUT", lambda cfg, value: setattr(cfg.discovery, "timeout", float(value))),
    ("WORKER_PEERS", lambda cfg, value: setattr(cfg.discovery, "peers", _parse_csv(value))),
    ("WORKER_TAG_FILTER", lambda cfg, value: setattr(cfg.discovery, "tag_filter", _parse_csv(value))),
    (
        "DISCOVERY_REGISTRATION_MAX_ATTEMPTS",
        lambda cfg, value: setattr(cfg.discovery, "registration_max_attempts", int(value)),
    ),
    (
        "DISCOVERY_REGISTRATION_DELAY",
        lambda cfg, value: setattr(cfg.discovery, "registration_delay_s", float(value)),
    ),
    (
        "DISCOVERY_REGISTRATION_RPC_TIMEOUT",
        lambda cfg, value: setattr(cfg.discovery, "registration_rpc_timeout_s", float(value)),
    ),
    (
        "DISCOVERY_WATCHDOG_INTERVAL",
        lambda cfg, value: setattr(cfg.discovery, "watchdog_interval_s", float(value)),
    ),
    (
        "WORKER_INIT_MAX_ATTEMPTS",
        lambda cfg, value: setattr(cfg.startup, "worker_init_max_attempts", int(value)),
    ),
    ("WORKER_INIT_DELAY", lambda cfg, value: setattr(cfg.startup, "worker_init_delay_s", float(value))),
    (
        "WORKER_INIT_RPC_TIMEOUT",
        lambda cfg, value: setattr(cfg.startup, "worker_init_rpc_timeout_s", float(value)),
    ),
    ("CHECKPOINT_ENABLED", lambda cfg, value: setattr(cfg.checkpoint, "enabled", _parse_bool(value))),
    ("CHECKPOINT_DIR", lambda cfg, value: setattr(cfg.checkpoint, "dir", value)),
    ("CHECKPOINT_SAVE_EVERY", lambda cfg, value: setattr(cfg.checkpoint, "save_every", value)),
    ("CHECKPOINT_RESUME", lambda cfg, value: setattr(cfg.checkpoint, "resume", value or None)),
    ("LOG_ENABLED", lambda cfg, value: setattr(cfg.logging, "enabled", _parse_bool(value))),
    ("LOG_DIR", lambda cfg, value: setattr(cfg.logging, "dir", value)),
    ("LOG_LEVEL", lambda cfg, value: setattr(cfg.logging, "level", value)),
    ("PROFILE_VERBOSITY", lambda cfg, value: setattr(cfg.profile, "verbosity", int(value))),
    ("PROFILE_MEMORY", lambda cfg, value: setattr(cfg.profile, "memory", _parse_bool(value))),
    (
        "FAULT_TOLERANCE_ENABLED",
        lambda cfg, value: setattr(cfg.fault_tolerance, "enabled", _parse_bool(value)),
    ),
    (
        "HEARTBEAT_INTERVAL",
        lambda cfg, value: setattr(cfg.fault_tolerance, "heartbeat_interval_s", float(value)),
    ),
    ("HEARTBEAT_TIMEOUT", lambda cfg, value: setattr(cfg.fault_tolerance, "ping_timeout_s", float(value))),
    ("FAULT_MAX_RETRIES", lambda cfg, value: setattr(cfg.fault_tolerance, "max_retries", int(value))),
)


def _apply_env_overrides(cfg: RunConfig) -> None:
    """Apply env vars on top of an existing config (env wins over lower-priority sources)."""
    for env_var, apply_value in _ENV_OVERRIDES:
        value = os.environ.get(env_var)
        if value:
            apply_value(cfg, value)
