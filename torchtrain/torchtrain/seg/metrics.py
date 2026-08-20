"""
Segmentation metrics, computed exactly the way PaddleSeg computes them.

PaddleSeg accumulates three per-class area vectors over the whole validation set
and derives every metric from them:

    intersect_area[c] = |pred == c AND label == c|      (true positives)
    pred_area[c]      = |pred == c|
    label_area[c]     = |label == c|

    IoU[c]        = intersect / (pred + label - intersect)
    mIoU          = mean over classes that appear in pred or label
    Acc (overall) = sum(intersect) / sum(pred)
    Precision[c]  = intersect[c] / pred[c]
    Recall[c]     = intersect[c] / label[c]
    Kappa         = (po - pe) / (1 - pe), po = sum(intersect)/sum(pred),
                    pe = sum(pred * label) / sum(pred)^2
    Dice[c]       = 2 * intersect / (pred + label)

Reproducing this (rather than reaching for a generic confusion-matrix metric) is
deliberate: the numbers a user sees for a torch run must be directly comparable
to the PaddleSeg run they are migrating from, and `Acc`/`Kappa` in particular
are easy to define differently and end up subtly off.

`ignore_index` pixels are excluded from all three vectors, so padding introduced
by the variable-size collate can never move a metric.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from .transforms import IGNORE_INDEX

_EPS = 1e-12


class SegMetric:
    """Accumulator for the three area vectors."""

    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.intersect = np.zeros(self.num_classes, dtype=np.float64)
        self.pred = np.zeros(self.num_classes, dtype=np.float64)
        self.label = np.zeros(self.num_classes, dtype=np.float64)
        self.num_images = 0

    def reset(self) -> None:
        self.intersect[:] = 0
        self.pred[:] = 0
        self.label[:] = 0
        self.num_images = 0

    def update(self, pred: torch.Tensor, label: torch.Tensor) -> None:
        """`pred` and `label` are `[N, H, W]` int tensors of class ids."""
        pred_np = pred.detach().to("cpu").reshape(-1).numpy()
        label_np = label.detach().to("cpu").reshape(-1).numpy()
        valid = label_np != self.ignore_index
        pred_np, label_np = pred_np[valid], label_np[valid]
        if pred_np.size == 0:
            self.num_images += int(pred.shape[0]) if pred.dim() == 3 else 1
            return

        bins = np.arange(self.num_classes + 1)
        self.pred += np.histogram(pred_np, bins=bins)[0]
        self.label += np.histogram(label_np, bins=bins)[0]
        hit = pred_np[pred_np == label_np]
        if hit.size:
            self.intersect += np.histogram(hit, bins=bins)[0]
        self.num_images += int(pred.shape[0]) if pred.dim() == 3 else 1

    # -- derived metrics ---------------------------------------------------

    def class_iou(self) -> Tuple[List[float], float]:
        union = self.pred + self.label - self.intersect
        iou = np.divide(self.intersect, union, out=np.zeros_like(union), where=union > 0)
        # PaddleSeg averages only over classes that actually occur, so an unused
        # class in a 20-class label map cannot drag mIoU to 0.
        present = union > 0
        miou = float(iou[present].mean()) if bool(present.any()) else 0.0
        return [float(v) for v in iou], miou

    def class_measurement(self) -> Tuple[float, List[float], List[float]]:
        total_pred = float(self.pred.sum())
        acc = float(self.intersect.sum() / (total_pred + _EPS))
        precision = np.divide(self.intersect, self.pred, out=np.zeros_like(self.pred), where=self.pred > 0)
        recall = np.divide(self.intersect, self.label, out=np.zeros_like(self.label), where=self.label > 0)
        return acc, [float(v) for v in precision], [float(v) for v in recall]

    def kappa(self) -> float:
        total = float(self.pred.sum())
        if total <= 0:
            return 0.0
        po = float(self.intersect.sum()) / total
        pe = float((self.pred * self.label).sum()) / (total * total)
        if abs(1.0 - pe) < _EPS:
            return 0.0
        return float((po - pe) / (1.0 - pe))

    def class_dice(self) -> Tuple[List[float], float]:
        denom = self.pred + self.label
        dice = np.divide(2.0 * self.intersect, denom, out=np.zeros_like(denom), where=denom > 0)
        present = denom > 0
        mdice = float(dice[present].mean()) if bool(present.any()) else 0.0
        return [float(v) for v in dice], mdice

    def summary(self) -> Dict[str, object]:
        class_iou, miou = self.class_iou()
        acc, precision, recall = self.class_measurement()
        class_dice, mdice = self.class_dice()
        return {
            "num_images": self.num_images,
            "mIoU": miou,
            "acc": acc,
            "kappa": self.kappa(),
            "dice": mdice,
            "class_iou": class_iou,
            "class_precision": precision,
            "class_recall": recall,
            "class_dice": class_dice,
        }
