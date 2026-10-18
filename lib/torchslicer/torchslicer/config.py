"""
RunConfig — single source of truth for all experiment parameters.

Load priority (highest wins):
    1. Python API kwargs — pass directly to SlicedModel.train() or executor
    2. Environment variables — standard env vars or .env file read by Docker Compose;
       ``RunConfig.load()`` applies them on top of the YAML, so a per-node env var
       (ROLE, PORT, DEVICE, ...) overrides the file every node shares
    3. YAML experiment file — --config path/to/run.yaml or EXPERIMENT_CONFIG env var
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
    # covers the whole init: deserializing the slice (imports transformers/peft for HF
    # layers, GBs for an LLM slice) and moving it to the device. A worker that is not up
    # yet still fails fast (UNAVAILABLE) and is retried.
    worker_init_rpc_timeout_s: float = 300.0


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
    """How tensors and calls travel between nodes (see transport/tunnel.py and
    docs/tutorials/nat.md).

    Fields:
        tensor:             "grpc" | "tcp" | "ipc" | "nccl"
                            "grpc": all tensors via gRPC (default, works everywhere)
                            "tcp": raw TCP for activations/gradients between directly
                              reachable nodes; falls back to gRPC through tunnels
                            "ipc": CUDA IPC zero-copy for same-node GPU tensors; falls
                              back to TCP/gRPC for cross-machine or CPU tensors
                            "nccl": NCCL P2P (isend/irecv) via a ProcessGroup; one
                              GPU per slice (fails at init otherwise); gloo on CPU
                            (env: TENSOR_TRANSPORT)
        tensor_port_offset: TCP listener port = gRPC port + this  (env: TENSOR_PORT_OFFSET)
        nat:                this node is behind NAT: it opens no port and only dials
                            out. It opens a tunnel to every node that sends to it, and
                            their calls (gradients, U-shaped activations, metrics,
                            heartbeats) come back over it. A per-node setting
                            (env: TRANSPORT_NAT); with ``ROLE=all`` it applies to the
                            data owner (coordinator / driver) only.
        relay:              address of a relay (``python -m torchslicer relay``): every
                            node that sets it dials only the relay, which forwards each
                            call. Needed when two nodes that talk are both behind NAT
                            (env: TRANSPORT_RELAY); with ``ROLE=all`` any non-empty
                            value starts a relay on loopback.
        tunnel_keepalive_s: a node pings its idle tunnels this often, so NAT mappings
                            stay open (env: TUNNEL_KEEPALIVE)
    """
    tensor:             str   = "grpc"   # "grpc" | "tcp" | "ipc" | "nccl"
    tensor_port_offset: int   = 1
    nat:                bool  = False
    relay:              str   = ""
    tunnel_keepalive_s: float = 20.0


@dataclass
class FaultToleranceConfig:
    enabled:              bool  = False
    heartbeat_interval_s: float = 5.0
    ping_timeout_s:       float = 3.0
    max_retries:          int   = 3
    # A batch not completed within this many seconds while every worker is alive is
    # treated as lost (e.g. a dropped message) and raises.  0 disables the check.
    batch_timeout_s:      float = 0.0
    # Confirm a runtime-error report with the reported worker before recovering:
    # a reachable worker that reports no fatal error is kept (the report was
    # false, e.g. one worker naming another).  False = trust every report.
    verify_reports:       bool  = True
    # Heartbeats run even when ``enabled`` is False: a dead worker then fails the run
    # with WorkerFailureError instead of leaving the driver waiting forever.
    detect_failures:      bool  = True


@dataclass
class PlanningConfig:
    """Capability-aware planning: partitions are computed after discovery, from
    what the workers report and what the executor measures."""
    # False = plan as if every worker were identical (the partitions ts.slice made).
    heterogeneous:   bool = True
    measure_network: bool = True       # RTT + bandwidth executor → each worker
    measure_links:   bool = False      # RTT + bandwidth worker i → worker i+1
    probe_bytes:     int  = 1 << 20    # payload of each bandwidth probe


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
        worker_persistent:        if True a worker survives the end-of-run shutdown:
                                  it drops its slice and waits for the next run, and
                                  keeps retrying registration instead of exiting
                                  (env: WORKER_PERSISTENT)
        worker_compute_gflops:    declared speed in GFLOP/s, used instead of the
                                  startup micro-benchmark (env: WORKER_COMPUTE_GFLOPS)
        worker_max_memory_mb:     memory cap for this worker's slice
                                  (env: WORKER_MAX_MEMORY_MB)
        worker_first_slice:       this worker must hold slice 0, e.g. it owns the
                                  data (env: WORKER_FIRST_SLICE)
    """
    device:                   str           = "auto"
    coordinator_address:      Optional[str] = None             # None → no coordinator registration
    coordinator_bind_address: str           = "0.0.0.0:50054"
    worker_port:              int           = 50051
    worker_address:           Optional[str] = None             # None → auto hostname:worker_port
    worker_tags:              list          = field(default_factory=list)
    keep_alive:               bool          = True
    worker_persistent:        bool          = False
    # Declared capabilities (planning). 0 / False = measure / none.
    worker_compute_gflops:    float         = 0.0    # overrides the measured speed (e.g. simulation)
    worker_max_memory_mb:     int           = 0      # cap for this worker's slice
    worker_first_slice:       bool          = False  # this worker must hold slice 0

    def __post_init__(self):
        if self.worker_tags is None:
            self.worker_tags = []


@dataclass
class DeployConfig:
    """Where and how the run is deployed: the same YAML drives every target.

    ``topology`` and ``role`` are read by :func:`torchslicer.run` on every node;
    the rest is read by ``python -m torchslicer launch`` when it renders a target
    (processes, compose, k8s, bare metal). See ``docs/deploy/``.

    Fields:
        topology:  "local" (one process, ``LocalExecutor``) | "centralized"
                   (coordinator + workers) | "p2p" (driver + followers)  (env: TOPOLOGY)
        role:      this node's role: "all" (every node in this process, workers
                   on loopback) | "coordinator" | "worker" | "driver" | "follower".
                   "" = "all"  (env: ROLE)
        name:      prefix for compose services / k8s objects
        image:     container image for compose and k8s (must contain the script)
        namespace: k8s namespace
        gpus:      GPUs per node (compose device reservations, k8s nvidia.com/gpu)

    The coordinator listens on ``network.coordinator_bind_address`` and workers
    on ``network.worker_port``.
    """
    topology:         str = "local"
    role:             str = ""
    name:             str = "torchslicer"
    image:            str = "torchslicer:cpu"
    namespace:        str = "default"
    gpus:             int = 0


@dataclass
class SLConfig:
    """Split-learning setting (see ``torchslicer/sl/`` and docs/tutorials/sl_settings.md).

    Fields:
        setting:   "" (default: "pipeline" for centralized, else "vanilla") |
                   "pipeline" (the data owner hosts no slice; raw inputs and labels
                   leave it) | "vanilla" (it hosts slice 0; labels leave it) |
                   "u_shaped" (it hosts slice 0 and the last slice; nothing but cut
                   tensors leaves it)  (env: SL_SETTING)
        cut_hooks: ``[{cut: 0 | "all", name: "quantize", params: {bits: 8}}, ...]``
                   (built-in names: float16, bfloat16, quantize, topk,
                   gaussian_noise, recorder; ``ts.register_hook`` adds more)
        link_bandwidth_mbps, link_latency_ms: a link model for the simulated
                   communication time in the metrics (0 = off); nothing is slowed
                   down (env: SL_LINK_BANDWIDTH_MBPS, SL_LINK_LATENCY_MS)
        tail_port: u_shaped, distributed: port of the data owner's last slice
                   (0 = the data owner's port + 2; its TCP listener is at
                   tail_port + transport.tensor_port_offset)  (env: SL_TAIL_PORT)
        clients:   number of clients for ``ts.run``: > 1 means ``data_fn()``
                   returns one loader per client  (env: SL_CLIENTS)
        client_mode: "sequential" | "parallel"  (env: SL_CLIENT_MODE)
        handoff:   sequential sessions: "copy" | "keep"  (env: SL_HANDOFF)
        client_states: where a session keeps the clients' states: "cpu" | "gpu" |
                   "cuda:i" | "auto" (the slices' GPU if they fit, else host RAM). A run
                   whose states don't fit in 80 % of the free memory stops before
                   training  (env: SL_CLIENT_STATES)
    """
    setting:             str   = ""
    cut_hooks:           list  = field(default_factory=list)
    link_bandwidth_mbps: float = 0.0
    link_latency_ms:     float = 0.0
    tail_port:           int   = 0
    clients:             int   = 1
    client_mode:         str   = "sequential"
    handoff:             str   = "copy"
    algorithm:           str   = ""
    client_states:       str   = "cpu"

    def __post_init__(self):
        if self.cut_hooks is None:
            self.cut_hooks = []


@dataclass
class OffloadConfig:
    """Out-of-core training on one node (local topology; see torchslicer/offload/ and
    docs/tutorials/offload.md): slices live in a slower tier and are paged into the
    device one at a time.

    Fields:
        mode:      "none" | "cpu" (pinned host RAM) | "disk" (files in ``dir``)  (env: OFFLOAD)
        optimizer: where the optimizer step runs: "auto" (measured) | "gpu" (the state
                   is paged with the slice) | "cpu" (gradients go to the host)
                   (env: OFFLOAD_OPTIMIZER)
        dir:       disk mode: directory for the slice files  (env: OFFLOAD_DIR)
        slots:     device slots (>= 2: one computes while the next is prefetched)
                   (env: OFFLOAD_SLOTS)
        device:    compute device; "" = ``network.device``  (env: OFFLOAD_DEVICE)
        direct_io: disk mode: ``O_DIRECT`` file I/O, bypassing the page cache
                   (env: OFFLOAD_DIRECT_IO)
    """
    mode:      str = "none"
    optimizer: str = "auto"
    dir:       str = ""
    slots:     int = 2
    device:    str = ""
    direct_io: bool = False


@dataclass
class DataConfig:
    """The experiment's data, built by TorchSlicer (``torchslicer/data/``, M11). With
    ``dataset`` set, ``ts.run(model_fn)`` needs no ``data_fn``: the training set is
    split across ``sl.clients`` clients and the test set is evaluated.

    Fields:
        dataset:    "" (``data_fn`` decides) | mnist | fashion_mnist | cifar10 | cifar100 |
                    synthetic | npz  (env: DATA_DATASET)
        root:       download/cache directory  (env: DATA_ROOT)
        download:   download if missing  (env: DATA_DOWNLOAD)
        params:     dataset parameters (synthetic: shape/classes/n; npz: path/test_path)
        batch_size: training batch size per client  (env: DATA_BATCH_SIZE)
        test_batch_size: evaluation batch size
        partition:  how the training set is split across clients: iid | dirichlet |
                    shards | quantity (``ts.data.register_partitioner`` adds more)
                    (env: DATA_PARTITION)
        partition_params: e.g. ``{alpha: 0.5}`` (dirichlet), ``{shards_per_client: 2}``
        seed:       seed of the split, subsets and shuffling  (env: DATA_SEED)
        train_subset, test_subset: use N samples (seeded), 0 = all; for smoke tests
                    (env: DATA_TRAIN_SUBSET, DATA_TEST_SUBSET)
        eval_every: evaluate on the test set every N epochs, 0 = never  (env: DATA_EVAL_EVERY)
    """
    dataset:          str  = ""
    root:             str  = "data"
    download:         bool = True
    params:           dict = field(default_factory=dict)
    batch_size:       int  = 32
    test_batch_size:  int  = 256
    partition:        str  = "iid"
    partition_params: dict = field(default_factory=dict)
    seed:             int  = 0
    train_subset:     int  = 0
    test_subset:      int  = 0
    eval_every:       int  = 1

    def __post_init__(self):
        self.params = self.params or {}
        self.partition_params = self.partition_params or {}


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
    planning:        PlanningConfig       = field(default_factory=PlanningConfig)
    deploy:          DeployConfig         = field(default_factory=DeployConfig)
    sl:              SLConfig             = field(default_factory=SLConfig)
    offload:         OffloadConfig        = field(default_factory=OffloadConfig)
    data:            DataConfig           = field(default_factory=DataConfig)

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
        Load with merge priority: env vars > YAML > defaults.
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
    ("TRANSPORT_NAT",      lambda cfg, v: setattr(cfg.transport, "nat",                _parse_bool(v))),
    ("TRANSPORT_RELAY",    lambda cfg, v: setattr(cfg.transport, "relay",              v.strip())),
    ("TUNNEL_KEEPALIVE",   lambda cfg, v: setattr(cfg.transport, "tunnel_keepalive_s", float(v))),
    # ── fault tolerance ────────────────────────────────────────────────────────
    ("FAULT_TOLERANCE_ENABLED", lambda cfg, v: setattr(cfg.fault_tolerance, "enabled",              _parse_bool(v))),
    ("HEARTBEAT_INTERVAL",      lambda cfg, v: setattr(cfg.fault_tolerance, "heartbeat_interval_s", float(v))),
    ("HEARTBEAT_TIMEOUT",       lambda cfg, v: setattr(cfg.fault_tolerance, "ping_timeout_s",       float(v))),
    ("FAULT_MAX_RETRIES",       lambda cfg, v: setattr(cfg.fault_tolerance, "max_retries",          int(v))),
    ("VERIFY_FAILURE_REPORTS",  lambda cfg, v: setattr(cfg.fault_tolerance, "verify_reports",       _parse_bool(v))),
    ("DETECT_FAILURES",         lambda cfg, v: setattr(cfg.fault_tolerance, "detect_failures",      _parse_bool(v))),
    # ── network ────────────────────────────────────────────────────────────────
    ("DEVICE",                   lambda cfg, v: setattr(cfg.network, "device",                   v)),
    ("COORDINATOR_ADDRESS",      lambda cfg, v: setattr(cfg.network, "coordinator_address",      v)),
    ("COORDINATOR_BIND_ADDRESS", lambda cfg, v: setattr(cfg.network, "coordinator_bind_address", v)),
    ("PORT",                     lambda cfg, v: setattr(cfg.network, "worker_port",              int(v))),
    ("WORKER_ADDRESS",           lambda cfg, v: setattr(cfg.network, "worker_address",           v)),
    ("WORKER_TAGS",              lambda cfg, v: setattr(cfg.network, "worker_tags",              _parse_csv(v))),
    ("KEEP_ALIVE",               lambda cfg, v: setattr(cfg.network, "keep_alive",               _parse_bool(v))),
    ("WORKER_PERSISTENT",        lambda cfg, v: setattr(cfg.network, "worker_persistent",        _parse_bool(v))),
    ("WORKER_COMPUTE_GFLOPS",    lambda cfg, v: setattr(cfg.network, "worker_compute_gflops",    float(v))),
    ("WORKER_MAX_MEMORY_MB",     lambda cfg, v: setattr(cfg.network, "worker_max_memory_mb",     int(v))),
    ("WORKER_FIRST_SLICE",       lambda cfg, v: setattr(cfg.network, "worker_first_slice",       _parse_bool(v))),
    # ── planning ───────────────────────────────────────────────────────────────
    ("PLANNING_HETEROGENEOUS",   lambda cfg, v: setattr(cfg.planning, "heterogeneous",   _parse_bool(v))),
    ("PLANNING_MEASURE_NETWORK", lambda cfg, v: setattr(cfg.planning, "measure_network", _parse_bool(v))),
    ("PLANNING_MEASURE_LINKS",   lambda cfg, v: setattr(cfg.planning, "measure_links",   _parse_bool(v))),
    # ── deploy ─────────────────────────────────────────────────────────────────
    ("TOPOLOGY", lambda cfg, v: setattr(cfg.deploy, "topology", v.strip().lower())),
    ("ROLE",     lambda cfg, v: setattr(cfg.deploy, "role",     v.strip().lower())),
    # ── split learning ─────────────────────────────────────────────────────────
    ("SL_SETTING",             lambda cfg, v: setattr(cfg.sl, "setting",             v.strip().lower())),
    ("SL_LINK_BANDWIDTH_MBPS", lambda cfg, v: setattr(cfg.sl, "link_bandwidth_mbps", float(v))),
    ("SL_LINK_LATENCY_MS",     lambda cfg, v: setattr(cfg.sl, "link_latency_ms",     float(v))),
    ("SL_TAIL_PORT",           lambda cfg, v: setattr(cfg.sl, "tail_port",           int(v))),
    ("SL_CLIENTS",             lambda cfg, v: setattr(cfg.sl, "clients",             int(v))),
    ("SL_CLIENT_MODE",         lambda cfg, v: setattr(cfg.sl, "client_mode",         v.strip().lower())),
    # ── data ───────────────────────────────────────────────────────────────────
    ("DATA_DATASET",           lambda cfg, v: setattr(cfg.data, "dataset",           v.strip().lower())),
    ("DATA_ROOT",              lambda cfg, v: setattr(cfg.data, "root",              v)),
    ("DATA_DOWNLOAD",          lambda cfg, v: setattr(cfg.data, "download",          _parse_bool(v))),
    ("DATA_BATCH_SIZE",        lambda cfg, v: setattr(cfg.data, "batch_size",        int(v))),
    ("DATA_PARTITION",         lambda cfg, v: setattr(cfg.data, "partition",         v.strip().lower())),
    ("DATA_SEED",              lambda cfg, v: setattr(cfg.data, "seed",              int(v))),
    ("DATA_TRAIN_SUBSET",      lambda cfg, v: setattr(cfg.data, "train_subset",      int(v))),
    ("DATA_TEST_SUBSET",       lambda cfg, v: setattr(cfg.data, "test_subset",       int(v))),
    ("DATA_EVAL_EVERY",        lambda cfg, v: setattr(cfg.data, "eval_every",        int(v))),
    ("OFFLOAD",                lambda cfg, v: setattr(cfg.offload, "mode",           v.strip().lower())),
    ("OFFLOAD_OPTIMIZER",      lambda cfg, v: setattr(cfg.offload, "optimizer",      v.strip().lower())),
    ("OFFLOAD_DIR",            lambda cfg, v: setattr(cfg.offload, "dir",            v)),
    ("OFFLOAD_SLOTS",          lambda cfg, v: setattr(cfg.offload, "slots",          int(v))),
    ("OFFLOAD_DIRECT_IO",      lambda cfg, v: setattr(cfg.offload, "direct_io",      _parse_bool(v))),
    ("OFFLOAD_DEVICE",         lambda cfg, v: setattr(cfg.offload, "device",         v.strip().lower())),
    ("SL_CLIENT_STATES",       lambda cfg, v: setattr(cfg.sl, "client_states",       v.strip().lower())),
    ("SL_HANDOFF",             lambda cfg, v: setattr(cfg.sl, "handoff",             v.strip().lower())),
)


def _apply_env_overrides(cfg: RunConfig) -> None:
    """Apply env vars on top of an existing config (env wins over lower-priority sources)."""
    for env_var, apply_value in _ENV_OVERRIDES:
        value = os.environ.get(env_var)
        if value:
            apply_value(cfg, value)
