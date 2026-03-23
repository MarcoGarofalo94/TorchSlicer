"""
RunLogger — per-run artifact writer and reader.

Writes into {run_dir}/:
  run_manifest.json    — static metadata written once at flush()
  metrics.jsonl        — epoch-level loss / duration
  coordinator.jsonl    — coordinator overhead (data_load, send, wait)
  worker_epoch.jsonl   — per-worker epoch aggregates (forward/backward/idle/mem)
  worker_batch.jsonl   — per-worker per-batch detail  (verbosity=3 only)
  partition_epoch.jsonl — local executor per-partition epoch aggregates
  partition_batch.jsonl — local executor per-partition per-batch (verbosity=3)

Each file is homogeneous: every line in a given file has the same schema,
so pd.read_json('worker_epoch.jsonl', lines=True) just works.

Usage (write)::

    logger = RunLogger(run_id="resnet18_4gpu", run_dir="./runs/resnet18_4gpu")
    logger.record_config(run_config)
    logger.record_model("ResNet", layer_names=["conv1", "bn1", ...])
    logger.record_split(partitions, layer_names)
    logger.record_workers(nodes)
    logger.record_executor("distributed")

    logger.log(step=1, epoch=1, loss=2.31, duration_s=14.1, phase="epoch")
    logger.log(step=1, epoch=1, worker=0, forward_avg_ms=4.2, phase="worker_epoch")

    logger.flush()

Usage (read / plot)::

    from torchslicer.monitor import RunLogger

    run = RunLogger.load("./runs/resnet18_4gpu")
    run.to_dataframe("epoch").plot(x="epoch", y="loss")
    run.to_dataframe("worker_epoch").pivot(
        index="epoch", columns="worker", values="forward_avg_ms").plot()
"""

import datetime
import json
import os
import time
from typing import Dict, List, Optional

# Maps phase name → jsonl filename (relative to run_dir)
_PHASE_FILES: Dict[str, str] = {
    "epoch":           "metrics.jsonl",
    "coordinator_epoch": "coordinator.jsonl",
    "worker_epoch":    "worker_epoch.jsonl",
    "worker_batch":    "worker_batch.jsonl",
    "partition_epoch": "partition_epoch.jsonl",
    "partition_batch": "partition_batch.jsonl",
}
_FALLBACK_FILE = "metrics.jsonl"


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
        self._manifest["model"] = {
            "class":        model_name,
            "total_layers": len(layer_names),
            "layer_types":  layer_names,
        }

    def record_split(self, partitions, layer_names: List[str],
                     strategy_name: str = "unknown") -> None:
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
        self._manifest["workers"] = [
            {
                "worker_index":   i,
                "node_id":        nd.node_id,
                "address":        nd.address,
                "device":         nd.device,
                "memory_mb_free": nd.memory_mb,
            }
            for i, nd in enumerate(nodes)
        ]

    def record_executor(self, name: str) -> None:
        self._manifest["executor"] = name

    def record_artifact(self, kind: str, name: str) -> None:
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
        Append one entry to the appropriate phase file and in-memory log_history.

        The ``phase`` key determines which file receives the entry:
          "epoch"              → metrics.jsonl
          "coordinator_epoch"  → coordinator.jsonl
          "worker_epoch"       → worker_epoch.jsonl
          "worker_batch"       → worker_batch.jsonl
          "partition_epoch"    → partition_epoch.jsonl
          "partition_batch"    → partition_batch.jsonl
        """
        self._log_history.append(metrics)
        filename = _PHASE_FILES.get(metrics.get("phase", ""), _FALLBACK_FILE)
        path = os.path.join(self.run_dir, filename)
        with open(path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    # ── finalise ────────────────────────────────────────────────────────────

    def flush(self, status: str = "completed") -> str:
        """Write run_manifest.json. Returns the file path."""
        elapsed = time.perf_counter() - self._start_time
        self._manifest.update({
            "status":     status,
            "ended_at":   datetime.datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(elapsed, 3),
        })

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

        # Record which phase files were actually written
        artifacts["metrics_files"] = [
            fname for fname in _PHASE_FILES.values()
            if os.path.exists(os.path.join(self.run_dir, fname))
        ]

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

        Reads run_manifest.json and all phase jsonl files found in the directory.
        """
        instance = object.__new__(cls)
        instance.run_dir     = run_dir
        instance._start_time = 0.0

        with open(os.path.join(run_dir, "run_manifest.json")) as f:
            instance._manifest = json.load(f)
        instance.run_id = instance._manifest.get("run_id", "")

        instance._log_history = []
        for fname in _PHASE_FILES.values():
            path = os.path.join(run_dir, fname)
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            instance._log_history.append(json.loads(line))

        return instance

    def to_dataframe(self, phase: Optional[str] = None):
        """
        Return logged metrics as a pandas DataFrame.

        Parameters
        ----------
        phase:
            If given, return only rows matching that phase (e.g. "epoch",
            "worker_epoch", "worker_batch", "coordinator_epoch").
            If None, return all rows combined.

        Alternatively, read a single file directly for maximum efficiency::

            import pandas as pd
            df = pd.read_json("runs/my_run/worker_epoch.jsonl", lines=True)
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required: pip install pandas")
        rows = self._log_history
        if phase is not None:
            rows = [r for r in rows if r.get("phase") == phase]
        return pd.DataFrame(rows)

    @property
    def manifest(self) -> dict:
        return self._manifest

    @property
    def log_history(self) -> List[dict]:
        return list(self._log_history)
