"""
RunLogger — per-run artifact writer and reader.

Writes two files into {run_dir}/{run_id}/:
  run_manifest.json   — static metadata: config, model, split layout, worker topology,
                        final training summary.  Written once at flush().
  metrics.jsonl       — append-only time-series: one JSON dict per line, one line per
                        logged event.  Trivially loadable with pandas for plotting.

All artifact files (checkpoints, etc.) produced by the run are stored in the same
directory so everything for a run lives in one place.

Usage (write)::

    logger = RunLogger(run_id="resnet18_4gpu", run_dir="./runs/resnet18_4gpu")
    logger.record_config(run_config)
    logger.record_model("ResNet", layer_names=["conv1", "bn1", ...])
    logger.record_split(partitions, layer_names)
    logger.record_workers(nodes)                 # List[NodeInfo]
    logger.record_executor("distributed")

    # inside the training loop:
    logger.log(step=0, epoch=0, loss=2.31, duration_s=14.1, phase="epoch")
    logger.log(step=0, epoch=0, worker=0, forward_avg_ms=4.2, phase="worker_epoch")

    logger.flush()   # writes run_manifest.json

Usage (read / plot)::

    from torchslicer.monitor import RunLogger
    import pandas as pd

    run = RunLogger.load("./runs/resnet18_4gpu")
    df  = run.to_dataframe()
    df[df.phase == "epoch"].plot(x="epoch", y="loss")
    df[df.phase == "worker_epoch"].pivot(index="epoch", columns="worker",
                                         values="forward_avg_ms").plot()
"""

import datetime
import json
import os
import time
from typing import List, Optional


class RunLogger:

    def __init__(self, run_id: str, run_dir: str):
        self.run_id  = run_id
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        self._log_history: List[dict] = []
        self._start_time = time.perf_counter()
        self._manifest: dict = {
            "run_id":     run_id,
            "status":     "running",
            "executor":   "unknown",
            "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "ended_at":   None,
            "duration_s": None,
        }

    # ── metadata recorders ──────────────────────────────────────────────────

    def record_config(self, run_config) -> None:
        """Store a RunConfig in the manifest (training/pipeline/discovery sections)."""
        cfg = run_config
        self._manifest["config"] = {
            "training": {
                "epochs":          cfg.training.epochs,
                "optimizer":       cfg.training.optimizer,
                "criterion":       cfg.training.criterion,
                "mixed_precision": cfg.training.mixed_precision,
            },
            "pipeline": {
                "use_gpipe": cfg.pipeline.use_gpipe,
                "n_micro":   cfg.pipeline.n_micro,
            },
            "discovery": {
                "backend":   cfg.discovery.backend,
                "n_workers": cfg.discovery.n_workers,
                "timeout":   cfg.discovery.timeout,
            },
            "profile": {
                "verbosity": cfg.profile.verbosity,
                "memory":    cfg.profile.memory,
            },
        }

    def record_model(self, model_name: str, layer_names: List[str]) -> None:
        """Store model class name, layer count, and layer type list."""
        self._manifest["model"] = {
            "class":        model_name,
            "total_layers": len(layer_names),
            "layer_types":  layer_names,
        }

    def record_split(self, partitions, layer_names: List[str],
                     strategy_name: str = "unknown") -> None:
        """Store the partition layout produced by the splitter."""
        parts = []
        for i, p in enumerate(partitions):
            names = [
                layer_names[j] if j < len(layer_names) else str(j)
                for j in p.layer_indices
            ]
            parts.append({
                "worker_index":  i,
                "layer_indices": list(p.layer_indices),
                "layer_names":   names,
                "n_layers":      len(p.layer_indices),
            })
        self._manifest["split"] = {
            "strategy":   strategy_name,
            "n_workers":  len(partitions),
            "partitions": parts,
        }

    def record_workers(self, nodes) -> None:
        """Store NodeInfo list from discovery (device, memory, address per worker)."""
        self._manifest["workers"] = [
            {
                "worker_index":  i,
                "node_id":       nd.node_id,
                "address":       nd.address,
                "device":        nd.device,
                "memory_mb_free": nd.memory_mb,
            }
            for i, nd in enumerate(nodes)
        ]

    def record_executor(self, name: str) -> None:
        self._manifest["executor"] = name

    def record_artifact(self, kind: str, name: str) -> None:
        """Register an artifact file produced by this run."""
        artifacts = self._manifest.setdefault("artifacts", {
            "dir":         self.run_dir,
            "checkpoints": [],
        })
        if kind == "checkpoint":
            artifacts["checkpoints"].append(name)
        else:
            artifacts[kind] = name

    # ── time-series logger ──────────────────────────────────────────────────

    def log(self, **metrics) -> None:
        """
        Append one entry to metrics.jsonl and in-memory log_history.

        Convention for the ``phase`` key (used for filtering in DataFrames):
          "epoch"          — coordinator epoch summary  (loss, duration_s, …)
          "batch"          — coordinator per-batch      (loss, batch_id, …)
          "worker_epoch"   — per-worker epoch aggregate (forward_avg_ms, …)
          "worker_batch"   — per-worker per-batch       (verbosity=3 only)
          "coordinator"    — coordinator overhead        (data_load_ms, wait_ms, …)
        """
        self._log_history.append(metrics)
        path = os.path.join(self.run_dir, "metrics.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    # ── finalise ────────────────────────────────────────────────────────────

    def flush(self, status: str = "completed") -> str:
        """Write run_manifest.json.  Returns the file path."""
        elapsed = time.perf_counter() - self._start_time
        self._manifest.update({
            "status":     status,
            "ended_at":   datetime.datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(elapsed, 3),
        })

        # Build training summary from epoch entries in log_history
        epoch_entries = [e for e in self._log_history if e.get("phase") == "epoch"]
        if epoch_entries:
            losses = [e["loss"] for e in epoch_entries if "loss" in e]
            self._manifest["training"] = {
                "epochs_completed": len(epoch_entries),
                "per_epoch":        epoch_entries,
                "final_loss":       round(losses[-1], 6) if losses else None,
            }

        artifacts = self._manifest.setdefault("artifacts", {
            "dir": self.run_dir, "checkpoints": [],
        })
        artifacts["run_manifest"] = "run_manifest.json"

        path = os.path.join(self.run_dir, "run_manifest.json")
        with open(path, "w") as f:
            json.dump(self._manifest, f, indent=2)
        print(f"[run_logger] manifest → {path}")
        return path

    # ── reader API ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, run_dir: str) -> "RunLogger":
        """
        Load a completed run directory for analysis.

        Parameters
        ----------
        run_dir:
            Path to the run directory (e.g. ``"./runs/resnet18_4gpu"``).

        Returns
        -------
        RunLogger
            Instance with ``manifest`` property and ``to_dataframe()`` method available.
        """
        manifest_path = os.path.join(run_dir, "run_manifest.json")
        metrics_path  = os.path.join(run_dir, "metrics.jsonl")

        instance = object.__new__(cls)
        instance.run_dir     = run_dir
        instance._start_time = 0.0

        with open(manifest_path) as f:
            instance._manifest = json.load(f)
        instance.run_id = instance._manifest.get("run_id", "")

        instance._log_history = []
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        instance._log_history.append(json.loads(line))

        return instance

    def to_dataframe(self):
        """
        Return all logged metrics as a pandas DataFrame.

        Each row is one entry from metrics.jsonl.  Filter by the ``phase`` column
        to get epoch summaries, worker stats, or batch-level data.

        Requires pandas (``pip install pandas``).
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required: pip install pandas")
        return pd.DataFrame(self._log_history)

    @property
    def manifest(self) -> dict:
        """The run_manifest.json content as a dict."""
        return self._manifest

    @property
    def log_history(self) -> List[dict]:
        """All logged entries (same data as metrics.jsonl)."""
        return list(self._log_history)
