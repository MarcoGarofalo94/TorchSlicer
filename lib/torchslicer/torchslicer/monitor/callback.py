"""
TrainingCallback — base class for run lifecycle hooks.

Subclass and override only the methods you need.  All methods have no-op defaults so
subclasses never need to call super().

Usage::

    from torchslicer.monitor import TrainingCallback

    class MyCallback(TrainingCallback):
        def __init__(self, val_loader):
            self._val = val_loader

        def on_epoch_end(self, epoch: int, metrics: dict) -> dict:
            # compute validation accuracy and inject into the logged metrics
            metrics["val_accuracy"] = self._eval(self._val)
            return metrics

    sliced.train(train_loader, ..., callbacks=[MyCallback(val_loader)])

The augmented dict returned by on_epoch_end / on_batch_end is what gets written to
metrics.jsonl, so any key added here will be available for downstream plotting.
"""


class TrainingCallback:
    """
    Base class for training lifecycle callbacks.

    All methods receive the current metrics dict (which may already contain keys from
    earlier callbacks in the list) and should return a (possibly augmented) dict.
    Returning None is treated as "no change" — the original dict is kept.
    """

    def on_train_begin(self, run_id: str, config: dict) -> None:
        """Called once before the first epoch starts."""

    def on_epoch_begin(self, epoch: int) -> None:
        """Called at the start of each epoch, before any batches are processed."""

    def on_epoch_end(self, epoch: int, metrics: dict) -> dict:
        """
        Called after each epoch completes.

        Parameters
        ----------
        epoch:
            0-based epoch index.
        metrics:
            Dict containing at least ``loss`` and ``duration_s``.  Add custom keys
            here — they will be written to ``metrics.jsonl``.

        Returns
        -------
        dict
            The (possibly augmented) metrics dict.  Must return a dict.
        """
        return metrics

    def on_batch_end(self, epoch: int, batch: int, metrics: dict) -> dict:
        """
        Called after each batch completes (only when verbosity >= 1 or callbacks
        are registered).

        Parameters
        ----------
        epoch:
            Current epoch index.
        batch:
            0-based batch index within the epoch.
        metrics:
            Dict containing at least ``loss``.

        Returns
        -------
        dict
            The (possibly augmented) metrics dict.
        """
        return metrics

    def on_train_end(self, log_history: list) -> None:
        """
        Called once after teardown, with the full list of logged metric entries.

        Parameters
        ----------
        log_history:
            List of dicts, one per logged event (same data as metrics.jsonl).
        """
