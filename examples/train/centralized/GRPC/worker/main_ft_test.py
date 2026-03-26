import torchslicer as ts
from torchslicer.testing_faults import WorkerFaultController


class FaultTestWorkerServicer(ts.WorkerServicer):
    """WorkerServicer with env-driven runtime fault injection."""

    def __init__(self):
        super().__init__()
        self._faults = WorkerFaultController()

    def _before_runtime_phase(self, phase: str, batch_id: int) -> None:
        self._faults.maybe_raise_runtime(
            worker_index=self._worker_index,
            phase=phase,
            batch_id=batch_id,
            run_id=self._run_id,
        )

    def save_checkpoint(self, request, context):
        result = super().save_checkpoint(request, context)
        self._faults.maybe_schedule_checkpoint_kill(request.epoch)
        return result


ts.run_worker(servicer_class=FaultTestWorkerServicer)
