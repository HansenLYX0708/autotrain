"""
Validation metrics for an anomalib run.

anomalib's default evaluator registers **test metrics only**:

    # anomalib/models/components/base/anomalib_module.py
    @staticmethod
    def configure_evaluator() -> Evaluator:
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        ...
        return Evaluator(test_metrics=test_metrics)

That is fine for `anomalib test`, but it means a `fit()` run logs nothing during
validation. Consequences for this platform, all of them silent:

* the monitoring page's AUROC chart stays empty;
* `ModelCheckpoint(monitor="image_AUROC")` has nothing to monitor, so it either
  errors or keeps the last epoch instead of the best one;
* the "best model" line the log parser reads is never printed.

So the runner replaces the model's evaluator with one that registers the same
metrics for **both** validation and test.

`strict=False` on the pixel metrics matters: it is what makes them skip
themselves when the batch carries no `gt_mask`, which is exactly the case for a
dataset whose defect images have no masks yet. With `strict=True` the run dies
mid-validation.
"""

from __future__ import annotations

from typing import Any, List


def build_evaluator(include_pixel: bool = True) -> Any:
    """An `Evaluator` with image (and optionally pixel) AUROC/F1 on val and test.

    Args:
        include_pixel: Register pixel-level metrics. They are non-strict, so
            leaving this on for a mask-less dataset costs nothing but a pair of
            metrics that never report.
    """
    from anomalib.metrics import AUROC, Evaluator, F1Score

    def metrics() -> List[Any]:
        out = [
            AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
            F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
        ]
        if include_pixel:
            out += [
                AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
                F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
            ]
        return out

    # Separate instances per stage: a torchmetrics object accumulates state, and
    # sharing one between val and test would mix the two sets of predictions.
    return Evaluator(val_metrics=metrics(), test_metrics=metrics())


def attach(model: Any, include_pixel: bool = True) -> Any:
    """Give `model` an evaluator that reports during validation.

    Assigning the attribute is enough, and it must NOT also be passed to the
    trainer: `Evaluator` is both an `nn.Module` and a `Callback`, and
    `AnomalibModule.configure_callbacks()` returns whatever `self.evaluator`
    holds *at the time Lightning calls it* — which is during `fit()` setup, after
    this function has run. Registering it a second time would make every metric
    see each batch twice.
    """
    evaluator = build_evaluator(include_pixel=include_pixel)
    model.evaluator = evaluator
    return evaluator
