"""
Lightning callback that makes anomalib's training legible to the platform.

anomalib reports progress through a rich progress bar and its metrics through
Lightning's logger machinery. The platform, in contrast, reads stdout line by
line (`src/lib/log-parsers/anomaly.ts`). This callback bridges the two by
printing the same shapes PaddleSeg prints:

    [ts] INFO: [TRAIN] epoch: 3, iter: 600/8000, loss: 0.1837, lr: 0.000100, ...
    [ts] INFO: [EVAL] #Images: 40 image_auroc: 0.9812 image_f1: 0.9231 ...
    [ts] INFO: [EVAL] The model with the best validation image_auroc (0.9812) was saved at iter 3000.

Three details that are easy to get wrong:

1. **`trainer.global_step` is useless for the memory-bank models.** Lightning
   increments it on optimizer steps, and PatchCore/PaDiM never step an optimizer
   (`configure_optimizers` returns None), so it stays at 0 for the whole run and
   the platform's progress bar would never move. The step counter here counts
   *batches consumed* instead.
2. **Metrics are read in `on_validation_end`, not `on_validation_epoch_end`.**
   The values are produced by anomalib's `Evaluator`, which is itself a callback;
   relying on callback ordering within the same hook would be a coin flip.
   `on_validation_end` runs after every `on_validation_epoch_end`.
3. **Metric names are not hard-coded.** Whatever the evaluator logged is printed,
   with the four names that have DB columns normalised to the lower-case spellings
   the parser maps onto columns. A metric nobody anticipated still shows up in the
   raw log and in `classMetrics`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from lightning.pytorch.callbacks import Callback

from .. import logger as L

#: Metric names that have dedicated `TrainingLog` columns, mapped to the keys
#: `log-parsers/anomaly.ts` looks for. Keys are compared case-insensitively.
_METRIC_ALIASES = {
    "image_auroc": "image_auroc",
    "image_f1score": "image_f1",
    "image_f1": "image_f1",
    "pixel_auroc": "pixel_auroc",
    "pixel_f1score": "pixel_f1",
    "pixel_f1": "pixel_f1",
}

#: Metrics logged by the model itself during training; they are already on the
#: `[TRAIN]` line and would be noise on the `[EVAL]` line.
_TRAIN_METRIC_KEYS = {"train_loss", "train_loss_step", "train_loss_epoch"}


def _as_float(value: Any) -> Optional[float]:
    """Best-effort float from a tensor / number, or None."""
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        value = float(value)
    except (TypeError, ValueError):
        return None
    # NaN is what anomalib's thresholds hold before the first validation pass.
    return None if value != value else value


class PaddleStyleLogger(Callback):
    """Print Paddle-compatible progress and metric lines.

    Args:
        log_iter: Emit a `[TRAIN]` line every N batches.
        best_metric: Metric monitored for the "best model" line. Must match the
            metric the `ModelCheckpoint` monitors, or the log would claim a
            checkpoint was saved that was not.
        total_steps_hint: `trainer.max_steps` when the caller knows it; the real
            total is recomputed at train start once the batch count is known.
    """

    def __init__(self, log_iter: int = 20, best_metric: str = "image_AUROC", total_steps_hint: int = 0) -> None:
        self.log_iter = max(1, int(log_iter))
        self.best_metric = best_metric
        self.total_steps_hint = int(total_steps_hint or 0)

        self._steps = 0
        self._total = 0
        self._epoch = 0
        self._batches_per_epoch = 0
        self._run_start = 0.0
        self._batch_start = 0.0
        self._batch_end = 0.0
        self._batch_cost = 0.0
        self._reader_cost = 0.0
        self._samples = 0
        self._best_value: Optional[float] = None
        self._best_step = 0
        self._eval_count = 0

    # -- helpers ----------------------------------------------------------

    @property
    def best(self) -> Dict[str, Any]:
        """Best metric seen, for the runner's final summary and metrics.json."""
        return {"metric": self.best_metric, "value": self._best_value, "step": self._best_step}

    @property
    def steps_done(self) -> int:
        return self._steps

    def _current_lr(self, trainer: Any) -> float:
        try:
            for optimizer in trainer.optimizers or []:
                for group in optimizer.param_groups:
                    return float(group.get("lr", 0.0))
        except Exception:  # noqa: BLE001 - logging must never break training
            pass
        # The memory-bank models have no optimizer at all; 0 is honest here.
        return 0.0

    def _memory_mb(self, trainer: Any):
        try:
            import torch

            from ..utils import memory_stats_mb

            device = getattr(trainer.strategy, "root_device", None)
            if device is None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            return memory_stats_mb(device)
        except Exception:  # noqa: BLE001
            return 0, 0

    # -- training ---------------------------------------------------------

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        self._run_start = time.time()
        self._batch_end = self._run_start
        try:
            self._batches_per_epoch = int(trainer.num_training_batches)
        except Exception:  # noqa: BLE001
            self._batches_per_epoch = 0

        max_steps = int(getattr(trainer, "max_steps", -1) or -1)
        max_epochs = int(getattr(trainer, "max_epochs", -1) or -1)
        # For a one-epoch model the honest total is "batches in the epoch": there
        # is no optimizer stepping, so max_steps would never be reached and the
        # progress bar would stall at a fraction of a percent.
        if max_epochs == 1 and self._batches_per_epoch > 0:
            self._total = self._batches_per_epoch
        elif max_steps > 0:
            self._total = max_steps
        elif max_epochs > 0 and self._batches_per_epoch > 0:
            self._total = max_epochs * self._batches_per_epoch
        else:
            self._total = self.total_steps_hint or 1

        L.log(
            "Training starts | steps: {} | batches/epoch: {} | monitored metric: {}".format(
                self._total, self._batches_per_epoch, self.best_metric
            )
        )

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._epoch = int(getattr(trainer, "current_epoch", 0)) + 1

    def on_train_batch_start(self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int) -> None:
        self._batch_start = time.time()
        # Time spent between batches is time the GPU waited for the dataloader.
        self._reader_cost = max(0.0, self._batch_start - self._batch_end)
        try:
            self._samples = int(batch.image.shape[0])
        except Exception:  # noqa: BLE001
            self._samples = 0

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        now = time.time()
        self._batch_end = now
        self._batch_cost = max(1e-9, now - self._batch_start)
        self._steps += 1

        if self._steps % self.log_iter != 0 and self._steps != self._total:
            return

        loss = None
        if isinstance(outputs, dict):
            loss = _as_float(outputs.get("loss"))
        else:
            loss = _as_float(outputs)
        # PatchCore/PaDiM return a constant 0 (or nothing); report 0.0 rather than
        # omitting the field, so the parser still produces a row and the chart has
        # a continuous x-axis.
        if loss is None:
            loss = 0.0

        remaining = max(0, self._total - self._steps)
        elapsed = now - self._run_start
        eta = remaining * (elapsed / self._steps) if self._steps else None
        reserved, allocated = self._memory_mb(trainer)

        L.seg_train(
            epoch=self._epoch,
            iters_done=self._steps,
            iters_total=self._total,
            loss=loss,
            lr=self._current_lr(trainer),
            batch_cost=self._batch_cost,
            reader_cost=self._reader_cost,
            ips=(self._samples / self._batch_cost) if self._samples else 0.0,
            mem_reserved_mb=reserved,
            mem_allocated_mb=allocated,
            eta_seconds=eta,
        )

    # -- validation -------------------------------------------------------

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        metrics = self.collect_metrics(trainer, pl_module)
        if not metrics:
            # Nothing to report usually means the validation split has no
            # anomalous images, so every metric is undefined. Say so once rather
            # than printing an empty EVAL line every interval.
            if self._eval_count == 0:
                L.log(
                    "WARNING: validation produced no metrics. An anomaly run needs "
                    "defect images in the validation split (dataset `abnormal_dir`) "
                    "to compute AUROC or pick a threshold."
                )
            self._eval_count += 1
            return

        num_images = self._val_image_count(trainer)
        self.emit_eval(metrics, num_images)
        self._eval_count += 1

        monitored = self._monitored_value(metrics)
        if monitored is not None and (self._best_value is None or monitored > self._best_value):
            self._best_value = monitored
            self._best_step = self._steps
        if self._best_value is not None:
            L.log(
                "[EVAL] The model with the best validation {} ({:.4f}) was saved at iter {}.".format(
                    self.best_metric.lower(), self._best_value, self._best_step
                )
            )

    def _monitored_value(self, metrics: Dict[str, float]) -> Optional[float]:
        wanted = self.best_metric.lower()
        for key, value in metrics.items():
            if key.lower() == wanted:
                return value
        # Fall back to the canonical alias, so monitoring `image_AUROC` still
        # works when the evaluator named it `image_auroc`.
        return metrics.get(_METRIC_ALIASES.get(wanted, wanted))

    def _val_image_count(self, trainer: Any) -> int:
        try:
            datamodule = trainer.datamodule
            data = getattr(datamodule, "val_data", None)
            if data is not None:
                return len(data)
        except Exception:  # noqa: BLE001
            pass
        return 0

    def collect_metrics(self, trainer: Any, pl_module: Any) -> Dict[str, float]:
        """Every scalar the evaluator logged, plus the adaptive thresholds."""
        out: Dict[str, float] = {}
        for source in (getattr(trainer, "callback_metrics", None), getattr(trainer, "logged_metrics", None)):
            if not source:
                continue
            for key, value in source.items():
                if key in _TRAIN_METRIC_KEYS:
                    continue
                number = _as_float(value)
                if number is None:
                    continue
                out[_METRIC_ALIASES.get(key.lower(), key.lower())] = number

        threshold = self.read_threshold(pl_module)
        if threshold is not None:
            out.setdefault("threshold", threshold)
        return out

    @staticmethod
    def read_threshold(pl_module: Any) -> Optional[float]:
        """Image-level anomaly-score cut-off chosen by anomalib.

        `OneClassPostProcessor` keeps it in a registered buffer that is NaN until
        the first validation pass has run.
        """
        post = getattr(pl_module, "post_processor", None)
        if post is None:
            return None
        for attr in ("image_threshold", "_image_threshold", "raw_image_threshold"):
            value = _as_float(getattr(post, attr, None))
            if value is not None:
                return value
        return None

    @staticmethod
    def emit_eval(metrics: Dict[str, float], num_images: int) -> None:
        """Print the `[EVAL]` metrics line.

        Column-backed metrics come first in a stable order so the line stays
        readable; anything else follows alphabetically and is preserved by the
        parser as `extraMetrics`.
        """
        preferred = ("image_auroc", "image_f1", "pixel_auroc", "pixel_f1", "threshold")
        parts = []
        for key in preferred:
            if key in metrics:
                parts.append("{}: {:.4f}".format(key, metrics[key]))
        for key in sorted(metrics):
            if key not in preferred:
                parts.append("{}: {:.4f}".format(key, metrics[key]))
        L.log("[EVAL] #Images: {} {}".format(num_images, " ".join(parts)))
