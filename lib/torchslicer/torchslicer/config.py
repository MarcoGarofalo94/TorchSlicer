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
from typing import Optional


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
    backend:   str   = "coordinator"  # "coordinator" | "static"
    n_workers: int   = 2
    timeout:   float = 60.0
    peers:     list  = field(default_factory=list)  # for "static" backend: ["host:port", ...]


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


@dataclass
class ProfileConfig:
    verbosity: int  = 0      # 0=off  1=epoch totals  2=phase breakdown  3=per-batch
    memory:    bool = False  # GPU memory snapshots per phase (adds ~CUDA sync per phase)


@dataclass
class RunConfig:
    run_id:     str             = field(default_factory=_default_run_id)
    training:   TrainingConfig  = field(default_factory=TrainingConfig)
    pipeline:   PipelineConfig  = field(default_factory=PipelineConfig)
    discovery:  DiscoveryConfig = field(default_factory=DiscoveryConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging:    LoggingConfig   = field(default_factory=LoggingConfig)
    profile:    ProfileConfig   = field(default_factory=ProfileConfig)

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

        if v := os.environ.get("RUN_ID"):
            cfg.run_id = v

        t = cfg.training
        if v := os.environ.get("EPOCHS"):
            t.epochs = int(v)
        if v := os.environ.get("MIXED_PRECISION"):
            t.mixed_precision = v.lower() in ("1", "true", "yes")

        p = cfg.pipeline
        if v := os.environ.get("USE_GPIPE"):
            p.use_gpipe = v.lower() in ("1", "true", "yes")
        if v := os.environ.get("N_MICRO"):
            p.n_micro = int(v)

        d = cfg.discovery
        if v := os.environ.get("N_WORKERS"):
            d.n_workers = int(v)
        if v := os.environ.get("DISCOVERY_BACKEND"):
            d.backend = v
        if v := os.environ.get("DISCOVERY_TIMEOUT"):
            d.timeout = float(v)
        if v := os.environ.get("WORKER_PEERS"):
            d.peers = [p.strip() for p in v.split(",") if p.strip()]

        c = cfg.checkpoint
        if v := os.environ.get("CHECKPOINT_ENABLED"):
            c.enabled = v.lower() in ("1", "true", "yes")
        if v := os.environ.get("CHECKPOINT_DIR"):
            c.dir = v
        if v := os.environ.get("CHECKPOINT_SAVE_EVERY"):
            c.save_every = v
        if v := os.environ.get("CHECKPOINT_RESUME"):
            c.resume = v or None

        lg = cfg.logging
        if v := os.environ.get("LOG_ENABLED"):
            lg.enabled = v.lower() in ("1", "true", "yes")
        if v := os.environ.get("LOG_DIR"):
            lg.dir = v

        pr = cfg.profile
        if v := os.environ.get("PROFILE_VERBOSITY"):
            pr.verbosity = int(v)
        if v := os.environ.get("PROFILE_MEMORY"):
            pr.memory = v.lower() in ("1", "true", "yes")

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
                backend   = d.get("backend",   cfg.discovery.backend),
                n_workers = d.get("n_workers", cfg.discovery.n_workers),
                timeout   = d.get("timeout",   cfg.discovery.timeout),
                peers     = d.get("peers",     cfg.discovery.peers) or [],
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
            )

        if pr := data.get("profile"):
            cfg.profile = ProfileConfig(
                verbosity = pr.get("verbosity", cfg.profile.verbosity),
                memory    = pr.get("memory",    cfg.profile.memory),
            )

        return cfg


def _apply_env_overrides(cfg: RunConfig) -> None:
    """Apply env vars on top of a YAML-loaded config (env wins over YAML)."""
    env = os.environ.get

    if env("RUN_ID"):
        cfg.run_id = env("RUN_ID")
    if env("EPOCHS"):
        cfg.training.epochs = int(env("EPOCHS"))
    if env("MIXED_PRECISION"):
        cfg.training.mixed_precision = env("MIXED_PRECISION").lower() in ("1", "true", "yes")
    if env("USE_GPIPE"):
        cfg.pipeline.use_gpipe = env("USE_GPIPE").lower() in ("1", "true", "yes")
    if env("N_MICRO"):
        cfg.pipeline.n_micro = int(env("N_MICRO"))
    if env("N_WORKERS"):
        cfg.discovery.n_workers = int(env("N_WORKERS"))
    if env("DISCOVERY_BACKEND"):
        cfg.discovery.backend = env("DISCOVERY_BACKEND")
    if env("DISCOVERY_TIMEOUT"):
        cfg.discovery.timeout = float(env("DISCOVERY_TIMEOUT"))
    if env("WORKER_PEERS"):
        cfg.discovery.peers = [p.strip() for p in env("WORKER_PEERS").split(",") if p.strip()]
    if env("CHECKPOINT_ENABLED"):
        cfg.checkpoint.enabled = env("CHECKPOINT_ENABLED").lower() in ("1", "true", "yes")
    if env("CHECKPOINT_DIR"):
        cfg.checkpoint.dir = env("CHECKPOINT_DIR")
    if env("CHECKPOINT_SAVE_EVERY"):
        cfg.checkpoint.save_every = env("CHECKPOINT_SAVE_EVERY")
    if env("CHECKPOINT_RESUME"):
        cfg.checkpoint.resume = env("CHECKPOINT_RESUME") or None
    if env("LOG_ENABLED"):
        cfg.logging.enabled = env("LOG_ENABLED").lower() in ("1", "true", "yes")
    if env("LOG_DIR"):
        cfg.logging.dir = env("LOG_DIR")
    if env("PROFILE_VERBOSITY"):
        cfg.profile.verbosity = int(env("PROFILE_VERBOSITY"))
    if env("PROFILE_MEMORY"):
        cfg.profile.memory = env("PROFILE_MEMORY").lower() in ("1", "true", "yes")
