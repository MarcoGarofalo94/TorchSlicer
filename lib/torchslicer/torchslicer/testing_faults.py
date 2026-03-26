import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torchslicer as ts
from torchslicer.monitor.process_logger import get_logger


_LOG = get_logger("torchslicer.ft")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _env_text(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def schedule_process_exit(delay_s: float, exit_code: int, reason: str) -> None:
    _LOG.warning(
        "fault injection scheduled delay_s=%.2f exit_code=%s reason=%s",
        delay_s,
        exit_code,
        reason,
    )

    def _kill():
        time.sleep(delay_s)
        os._exit(exit_code)

    threading.Thread(target=_kill, daemon=True, name="fault-injector").start()


def _fault_state_dir() -> Path:
    root = Path(os.environ.get("FAULT_STATE_DIR", "/workspace/.torchslicer_faults"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _claim_fault_once(run_id: str, fault_key: str) -> bool:
    if not run_id:
        return True
    marker = _fault_state_dir() / f"{run_id}_{fault_key}.marker"
    try:
        fd = os.open(str(marker), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"{time.time():.6f}\n")
    return True


@dataclass
class WorkerFaultConfig:
    kill_epoch: int = 0
    kill_delay_s: float = 2.0
    kill_exit_code: int = 137
    runtime_phase: str = ""
    runtime_batch_id: int = -1
    runtime_worker_index: int = -1

    @classmethod
    def from_env(cls) -> "WorkerFaultConfig":
        return cls(
            kill_epoch=_env_int("FAULT_KILL_EPOCH", 0),
            kill_delay_s=_env_float("FAULT_KILL_DELAY_S", 2.0),
            kill_exit_code=_env_int("FAULT_KILL_EXIT_CODE", 137),
            runtime_phase=_env_text("FAULT_RAISE_PHASE").lower(),
            runtime_batch_id=_env_int("FAULT_RAISE_BATCH_ID", -1),
            runtime_worker_index=_env_int("FAULT_TARGET_WORKER_INDEX", -1),
        )

    def runtime_enabled(self) -> bool:
        return bool(self.runtime_phase)

    def matches_runtime(self, worker_index: int, phase: str, batch_id: int) -> bool:
        if not self.runtime_enabled():
            return False
        if self.runtime_phase != phase.lower():
            return False
        if self.runtime_batch_id >= 0 and self.runtime_batch_id != batch_id:
            return False
        if self.runtime_worker_index >= 0 and self.runtime_worker_index != worker_index:
            return False
        return True


class WorkerFaultController:
    def __init__(self, config: WorkerFaultConfig | None = None):
        self._config = config or WorkerFaultConfig.from_env()
        self._runtime_triggered = False
        self._kill_triggered = False

    def maybe_raise_runtime(self, worker_index: int, phase: str, batch_id: int, run_id: str = "") -> None:
        if self._runtime_triggered:
            return
        if not self._config.matches_runtime(worker_index, phase, batch_id):
            return
        fault_key = (
            f"worker_runtime_{self._config.runtime_phase}_"
            f"{self._config.runtime_batch_id}_{self._config.runtime_worker_index}"
        )
        if not _claim_fault_once(run_id, fault_key):
            self._runtime_triggered = True
            _LOG.info(
                "fault injection already consumed run_id=%s phase=%s batch_id=%s worker_index=%s",
                run_id,
                phase,
                batch_id,
                worker_index,
            )
            return
        self._runtime_triggered = True
        raise RuntimeError(
            "simulated worker failure "
            f"worker_index={worker_index} phase={phase} batch_id={batch_id}"
        )

    def maybe_schedule_checkpoint_kill(self, epoch: int) -> None:
        if self._kill_triggered or self._config.kill_epoch <= 0 or epoch < self._config.kill_epoch:
            return
        self._kill_triggered = True
        schedule_process_exit(
            delay_s=self._config.kill_delay_s,
            exit_code=self._config.kill_exit_code,
            reason=(
                "worker checkpoint crash "
                f"epoch={epoch} threshold={self._config.kill_epoch}"
            ),
        )


@dataclass
class CoordinatorFaultConfig:
    exit_epoch: int = 0
    exit_delay_s: float = 0.5
    exit_code: int = 137

    @classmethod
    def from_env(cls) -> "CoordinatorFaultConfig":
        return cls(
            exit_epoch=_env_int("FAULT_COORDINATOR_EXIT_EPOCH", 0),
            exit_delay_s=_env_float("FAULT_COORDINATOR_EXIT_DELAY_S", 0.5),
            exit_code=_env_int("FAULT_COORDINATOR_EXIT_CODE", 137),
        )

    def enabled(self) -> bool:
        return self.exit_epoch > 0


class CoordinatorCrashCallback(ts.TrainingCallback):
    def __init__(self, config: CoordinatorFaultConfig | None = None):
        self._config = config or CoordinatorFaultConfig.from_env()
        self._triggered = False

    @property
    def enabled(self) -> bool:
        return self._config.enabled()

    def on_epoch_end(self, epoch: int, metrics: dict) -> dict:
        if self._triggered or not self._config.enabled():
            return metrics
        if epoch < self._config.exit_epoch:
            return metrics
        self._triggered = True
        schedule_process_exit(
            delay_s=self._config.exit_delay_s,
            exit_code=self._config.exit_code,
            reason=(
                "coordinator crash "
                f"epoch={epoch} threshold={self._config.exit_epoch}"
            ),
        )
        return metrics
