"""
Paddle-compatible stdout logging.

The web app parses training stdout with `src/lib/log-parsers/*` to build the
`TrainingLog` rows that drive the monitoring charts. Those parsers were written
against PaddleSeg / PaddleDetection output, so this module reproduces those log
lines **exactly** rather than inventing a new format. That is what lets a torch
job reuse the whole monitoring / progress / best-checkpoint pipeline unchanged.

Formats reproduced
------------------
PaddleSeg TRAIN (single line, parsed by `log-parsers/segmentation.ts`):

    [2026/08/18 11:08:59] INFO: [TRAIN] epoch: 12, iter: 200/2000, loss: 0.5440,
      lr: 0.004445, batch_cost: 0.1866, reader_cost: 0.0662,
      ips: 21.4352 samples/sec, max_mem_reserved: 3988 MB,
      max_mem_allocated: 3542 MB | ETA 00:13:51

PaddleSeg EVAL (5-line block; the parser accumulates it into one row):

    [ts] INFO: [EVAL] #Images: 10 mIoU: 0.5696 Acc: 0.9900 Kappa: 0.4986 Dice: 0.6021
    [ts] INFO: [EVAL] Class IoU:
    [0.9952 0.3447 0.6496]
    [ts] INFO: [EVAL] Class Precision:
    [0.9976 0.4916 0.764 ]
    [ts] INFO: [EVAL] Class Recall:
    [0.9976 0.5358 0.8128]
    [ts] INFO: [EVAL] The model with the best validation mIoU (0.6675) was saved at iter 5000.

PaddleDetection TRAIN (single line, parsed by `log-parsers/detection.ts`):

    [2026/08/18 10:20:46] INFO: Epoch: [8] [60/79] learning_rate: 0.000996
      loss: 4.193813 loss_cls: 1.671748 loss_box: 0.352000 eta: 0:02:24
      batch_cost: 0.3400 data_cost: 0.0100 ips: 5.0500 images/s
      max_mem_reserved: 3988 max_mem_allocated: 3542

Two subtleties that are easy to get wrong:

* The class-metric arrays are printed **bare, on their own line** (no timestamp
  prefix), because the parser only accepts a line that starts with `[` and ends
  with `]` as an array payload. This is what numpy's `print(arr)` produces and
  what PaddleSeg emits.
* The timestamp must not contain a `[digits/digits]` substring, or
  `log-parsers/detection.ts` would mistake it for the `[iter/total]` marker.
  `[YYYY/MM/DD HH:MM:SS]` is safe (the space breaks the pattern).
"""

from __future__ import annotations

import datetime
import sys
from typing import Dict, Optional, Sequence


def _now() -> str:
    return datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")


def log(message: str) -> None:
    """Emit one `[ts] INFO: <message>` line and flush immediately.

    Flushing matters: the runner reads stdout incrementally to update progress,
    and a buffered stream makes a live job look frozen.
    """
    sys.stdout.write("[{}] INFO: {}\n".format(_now(), message))
    sys.stdout.flush()


def log_raw(message: str) -> None:
    """Emit a line with no prefix (used for the bare `[a b c]` arrays)."""
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def format_eta(seconds: Optional[float], pad_hours: bool = True) -> str:
    """`H:MM:SS`, optionally zero-padded to `HH:MM:SS`.

    PaddleSeg pads the hours (`ETA 00:13:51`); PaddleDetection does not
    (`eta: 0:02:24`) and switches to `1 day, 20:03:57` past 24h. Both parsers
    accept either, but matching each framework keeps the output familiar.
    """
    if seconds is None or seconds != seconds or seconds < 0:  # NaN-safe
        return "--:--:--" if pad_hours else "0:00:00"
    seconds = int(seconds)
    if not pad_hours:
        return str(datetime.timedelta(seconds=seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours + days * 24, minutes, secs)


def format_float_array(values: Sequence[float], precision: int = 4) -> str:
    """Render a float array the way numpy does, e.g. `[0.9952 0.3447 0.6496]`.

    Trailing zeros are stripped per element (numpy prints `0.764 ` not
    `0.7640`), which is cosmetic — the parser splits on whitespace either way.
    """
    parts = []
    for value in values:
        text = "{:.{p}f}".format(float(value), p=precision)
        parts.append(text.rstrip("0").ljust(len(text), " ") if "." in text else text)
    return "[" + " ".join(parts) + "]"


# ---------------------------------------------------------------------------
# Segmentation (PaddleSeg-compatible)
# ---------------------------------------------------------------------------


def seg_train(
    epoch: int,
    iters_done: int,
    iters_total: int,
    loss: float,
    lr: float,
    batch_cost: float,
    reader_cost: float,
    ips: float,
    mem_reserved_mb: int,
    mem_allocated_mb: int,
    eta_seconds: Optional[float],
) -> None:
    log(
        "[TRAIN] epoch: {epoch}, iter: {done}/{total}, loss: {loss:.4f}, "
        "lr: {lr:.6f}, batch_cost: {bc:.4f}, reader_cost: {rc:.5f}, "
        "ips: {ips:.4f} samples/sec, max_mem_reserved: {mr} MB, "
        "max_mem_allocated: {ma} MB | ETA {eta}".format(
            epoch=epoch,
            done=iters_done,
            total=iters_total,
            loss=loss,
            lr=lr,
            bc=batch_cost,
            rc=reader_cost,
            ips=ips,
            mr=int(mem_reserved_mb),
            ma=int(mem_allocated_mb),
            eta=format_eta(eta_seconds),
        )
    )


def seg_eval(
    num_images: int,
    miou: float,
    acc: float,
    kappa: float,
    dice: float,
    class_iou: Sequence[float],
    class_precision: Sequence[float],
    class_recall: Sequence[float],
) -> None:
    """Emit the 5-line PaddleSeg EVAL block (without the trailing best line)."""
    log(
        "[EVAL] #Images: {n} mIoU: {miou:.4f} Acc: {acc:.4f} "
        "Kappa: {kappa:.4f} Dice: {dice:.4f}".format(
            n=num_images, miou=miou, acc=acc, kappa=kappa, dice=dice
        )
    )
    log("[EVAL] Class IoU: ")
    log_raw(format_float_array(class_iou))
    log("[EVAL] Class Precision: ")
    log_raw(format_float_array(class_precision))
    log("[EVAL] Class Recall: ")
    log_raw(format_float_array(class_recall))


def seg_best(best_miou: float, best_iter: int) -> None:
    """Closing line of the EVAL block; this is what flushes the parser's row.

    The parser reads `bestMetric` / `bestIter` from here and mirrors them onto
    `TrainingJob.bestMetric` / `bestIter`, which the UI shows as "best model".
    """
    log(
        "[EVAL] The model with the best validation mIoU ({:.4f}) "
        "was saved at iter {}.".format(best_miou, best_iter)
    )


# ---------------------------------------------------------------------------
# Detection (PaddleDetection-compatible)
# ---------------------------------------------------------------------------

# Only these loss components have dedicated columns in the `TrainingLog` table.
# Anything else a model reports is folded into the raw log line only.
_DET_LOSS_KEYS = ("loss_cls", "loss_iou", "loss_dfl", "loss_l1")


def det_train(
    epoch: int,
    step: int,
    steps_per_epoch: int,
    lr: float,
    loss: float,
    losses: Optional[Dict[str, float]],
    eta_seconds: Optional[float],
    batch_cost: float,
    data_cost: float,
    ips: float,
    mem_reserved_mb: int,
    mem_allocated_mb: int,
) -> None:
    parts = [
        "Epoch: [{}]".format(epoch),
        "[{}/{}]".format(step, steps_per_epoch),
        "learning_rate: {:.6f}".format(lr),
        "loss: {:.6f}".format(loss),
    ]
    losses = losses or {}
    # Emit the four parsed keys first, in a stable order, then the remainder so
    # the raw log still carries everything the model reported.
    for key in _DET_LOSS_KEYS:
        if key in losses:
            parts.append("{}: {:.6f}".format(key, float(losses[key])))
    for key in sorted(losses):
        if key not in _DET_LOSS_KEYS:
            parts.append("{}: {:.6f}".format(key, float(losses[key])))
    parts.append("eta: {}".format(format_eta(eta_seconds, pad_hours=False)))
    parts.append("batch_cost: {:.4f}".format(batch_cost))
    parts.append("data_cost: {:.4f}".format(data_cost))
    parts.append("ips: {:.4f} images/s".format(ips))
    parts.append("max_mem_reserved: {}".format(int(mem_reserved_mb)))
    parts.append("max_mem_allocated: {}".format(int(mem_allocated_mb)))
    log(" ".join(parts))


# pycocotools' exact `summarize()` format string. Reproduced verbatim because
# `src/app/api/validation-jobs/route.ts` greps for these literals to extract
# mAP/AR from an eval run.
_COCO_ROWS = (
    ("Average Precision", "(AP)", "0.50:0.95", "all", 100),
    ("Average Precision", "(AP)", "0.50", "all", 100),
    ("Average Precision", "(AP)", "0.75", "all", 100),
    ("Average Precision", "(AP)", "0.50:0.95", "small", 100),
    ("Average Precision", "(AP)", "0.50:0.95", "medium", 100),
    ("Average Precision", "(AP)", "0.50:0.95", "large", 100),
    ("Average Recall", "(AR)", "0.50:0.95", "all", 1),
    ("Average Recall", "(AR)", "0.50:0.95", "all", 10),
    ("Average Recall", "(AR)", "0.50:0.95", "all", 100),
    ("Average Recall", "(AR)", "0.50:0.95", "small", 100),
    ("Average Recall", "(AR)", "0.50:0.95", "medium", 100),
    ("Average Recall", "(AR)", "0.50:0.95", "large", 100),
)


def det_coco_stats(stats: Sequence[float], anno_type: str = "bbox") -> None:
    """Print the 12-row COCO summary block, byte-compatible with pycocotools.

    `stats` must be the standard 12-element COCOeval vector in the canonical
    order (AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl).
    """
    log_raw("Evaluate annotation type *{}*".format(anno_type))
    for i, (title, kind, iou, area, max_dets) in enumerate(_COCO_ROWS):
        value = float(stats[i]) if i < len(stats) else -1.0
        log_raw(
            " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}".format(
                title, kind, iou, area, max_dets, value
            )
        )
