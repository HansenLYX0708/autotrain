"""
COCO detection metrics.

Two implementations, same output vector:

1. `pycocotools` when it is importable — the reference implementation, so numbers
   are bit-comparable with PaddleDetection's.
2. A self-contained NumPy evaluator otherwise. `pycocotools` needs a C extension
   and is a recurring install problem on Windows/Python 3.8, and a detection
   framework that cannot report mAP unless an optional compiler toolchain is
   present is not much use. The algorithm below follows COCOeval exactly:
   greedy score-ordered matching per IoU threshold, crowd regions matched by
   intersection-over-detection-area and never counted as false positives,
   monotone-decreasing precision interpolated at 101 recall points.

The result is the canonical 12-element `stats` vector:

    [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]

`torchtrain.logger.det_coco_stats` prints it in pycocotools' exact text format,
which is what `src/app/api/validation-jobs/route.ts` parses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..logger import log

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
RECALL_THRESHOLDS = np.linspace(0.0, 1.00, 101)
# (label, [min_area, max_area]) in COCO's canonical order.
AREA_RANGES: Sequence[Tuple[str, Tuple[float, float]]] = (
    ("all", (0.0, 1e10)),
    ("small", (0.0, 32.0 ** 2)),
    ("medium", (32.0 ** 2, 96.0 ** 2)),
    ("large", (96.0 ** 2, 1e10)),
)
MAX_DETS = (1, 10, 100)


class Detection(dict):
    """One predicted box: `image_id`, `clsid`, `score`, `bbox_xyxy`."""


def _iou_matrix(dt_boxes: np.ndarray, gt_boxes: np.ndarray, iscrowd: np.ndarray) -> np.ndarray:
    """IoU, except crowd ground truth uses intersection / detection area.

    That asymmetry is COCO's: a detection landing inside a crowd region should
    neither be rewarded nor punished, so it must be able to "match" the crowd
    regardless of how large the crowd box is.
    """
    if len(dt_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(dt_boxes), len(gt_boxes)), dtype=np.float64)

    dt = dt_boxes[:, None, :]
    gt = gt_boxes[None, :, :]
    inter_w = np.minimum(dt[..., 2], gt[..., 2]) - np.maximum(dt[..., 0], gt[..., 0])
    inter_h = np.minimum(dt[..., 3], gt[..., 3]) - np.maximum(dt[..., 1], gt[..., 1])
    inter = np.clip(inter_w, 0, None) * np.clip(inter_h, 0, None)

    dt_area = (dt_boxes[:, 2] - dt_boxes[:, 0]) * (dt_boxes[:, 3] - dt_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = dt_area[:, None] + gt_area[None, :] - inter
    denom = np.where(iscrowd[None, :] > 0, np.repeat(dt_area[:, None], len(gt_boxes), axis=1), union)
    return np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)


class CocoDetectionEvaluator:
    """NumPy re-implementation of `COCOeval` for bounding boxes."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.gt: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        self.dt: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        self.image_ids: List[int] = []

    def add_ground_truth(self, records: Sequence[Dict[str, Any]]) -> None:
        """`records` are `CocoData.images` entries."""
        for record in records:
            image_id = int(record["id"])
            self.image_ids.append(image_id)
            for anno in record["annotations"]:
                key = (image_id, int(anno["clsid"]))
                self.gt.setdefault(key, []).append(
                    {
                        "bbox": np.asarray(anno["bbox_xyxy"], dtype=np.float64),
                        "area": float(anno["area"]),
                        "iscrowd": int(anno["iscrowd"]),
                    }
                )

    def add_detections(self, detections: Sequence[Dict[str, Any]]) -> None:
        for det in detections:
            key = (int(det["image_id"]), int(det["clsid"]))
            self.dt.setdefault(key, []).append(
                {
                    "bbox": np.asarray(det["bbox_xyxy"], dtype=np.float64),
                    "score": float(det["score"]),
                }
            )

    # -- core -------------------------------------------------------------

    def _evaluate_image(
        self, image_id: int, clsid: int, area_range: Tuple[float, float], max_det: int
    ) -> Optional[Dict[str, np.ndarray]]:
        gt = self.gt.get((image_id, clsid), [])
        dt = self.dt.get((image_id, clsid), [])
        if not gt and not dt:
            return None

        # Ignored GT (crowd, or outside the area range) is sorted last so the
        # matcher prefers a real box when both are candidates.
        gt_ignore = np.array(
            [1 if (g["iscrowd"] or g["area"] < area_range[0] or g["area"] > area_range[1]) else 0 for g in gt],
            dtype=np.int64,
        )
        order = np.argsort(gt_ignore, kind="mergesort")
        gt = [gt[i] for i in order]
        gt_ignore = gt_ignore[order]

        dt = sorted(dt, key=lambda d: -d["score"])[:max_det]
        dt_scores = np.array([d["score"] for d in dt], dtype=np.float64)
        gt_boxes = np.asarray([g["bbox"] for g in gt], dtype=np.float64).reshape(-1, 4)
        dt_boxes = np.asarray([d["bbox"] for d in dt], dtype=np.float64).reshape(-1, 4)
        iscrowd = np.array([g["iscrowd"] for g in gt], dtype=np.int64)
        ious = _iou_matrix(dt_boxes, gt_boxes, iscrowd)

        num_t, num_g, num_d = len(IOU_THRESHOLDS), len(gt), len(dt)
        gt_matched = np.zeros((num_t, num_g), dtype=np.int64)
        dt_matched = np.zeros((num_t, num_d), dtype=np.int64)
        dt_ignore = np.zeros((num_t, num_d), dtype=np.int64)

        for ti, threshold in enumerate(IOU_THRESHOLDS):
            for di in range(num_d):
                best_iou = min(threshold, 1 - 1e-10)
                match = -1
                for gi in range(num_g):
                    if gt_matched[ti, gi] > 0 and not iscrowd[gi]:
                        continue
                    # Once a non-ignored GT is matched, stop before the ignored
                    # tail (which is sorted last) -- COCOeval's early break.
                    if match > -1 and gt_ignore[match] == 0 and gt_ignore[gi] == 1:
                        break
                    if ious[di, gi] < best_iou:
                        continue
                    best_iou = ious[di, gi]
                    match = gi
                if match == -1:
                    continue
                dt_ignore[ti, di] = gt_ignore[match]
                dt_matched[ti, di] = 1
                gt_matched[ti, match] = 1

        if num_d:
            dt_area = (dt_boxes[:, 2] - dt_boxes[:, 0]) * (dt_boxes[:, 3] - dt_boxes[:, 1])
            out_of_range = ((dt_area < area_range[0]) | (dt_area > area_range[1])).astype(np.int64)
            dt_ignore = np.logical_or(dt_ignore, np.logical_and(dt_matched == 0, out_of_range[None, :])).astype(np.int64)

        return {
            "dt_scores": dt_scores,
            "dt_matched": dt_matched,
            "dt_ignore": dt_ignore,
            "gt_ignore": gt_ignore,
        }

    def accumulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """`(precision[T,R,K,A,M], recall[T,K,A,M])`, -1 where undefined."""
        image_ids = sorted(set(self.image_ids))
        num_t, num_r = len(IOU_THRESHOLDS), len(RECALL_THRESHOLDS)
        num_k, num_a, num_m = self.num_classes, len(AREA_RANGES), len(MAX_DETS)
        precision = -np.ones((num_t, num_r, num_k, num_a, num_m))
        recall = -np.ones((num_t, num_k, num_a, num_m))
        max_det = max(MAX_DETS)

        for ai, (_label, area_range) in enumerate(AREA_RANGES):
            for ki in range(num_k):
                # Evaluate once at the largest maxDets, then truncate per-image
                # for the smaller ones -- exactly what COCOeval does.
                per_image = [self._evaluate_image(iid, ki, area_range, max_det) for iid in image_ids]
                per_image = [e for e in per_image if e is not None]
                if not per_image:
                    continue
                for mi, limit in enumerate(MAX_DETS):
                    scores = np.concatenate([e["dt_scores"][:limit] for e in per_image]) if per_image else np.zeros(0)
                    if scores.size == 0:
                        gt_ignore = np.concatenate([e["gt_ignore"] for e in per_image])
                        if np.count_nonzero(gt_ignore == 0) > 0:
                            recall[:, ki, ai, mi] = 0
                            precision[:, :, ki, ai, mi] = 0
                        continue
                    order = np.argsort(-scores, kind="mergesort")
                    matched = np.concatenate([e["dt_matched"][:, :limit] for e in per_image], axis=1)[:, order]
                    ignored = np.concatenate([e["dt_ignore"][:, :limit] for e in per_image], axis=1)[:, order]
                    gt_ignore = np.concatenate([e["gt_ignore"] for e in per_image])
                    num_positives = np.count_nonzero(gt_ignore == 0)
                    if num_positives == 0:
                        continue

                    tps = np.logical_and(matched, np.logical_not(ignored))
                    fps = np.logical_and(np.logical_not(matched), np.logical_not(ignored))
                    tp_sum = np.cumsum(tps, axis=1).astype(np.float64)
                    fp_sum = np.cumsum(fps, axis=1).astype(np.float64)

                    for ti in range(num_t):
                        tp, fp = tp_sum[ti], fp_sum[ti]
                        num_dets = len(tp)
                        rc = tp / num_positives
                        pr = tp / (fp + tp + np.spacing(1))
                        recall[ti, ki, ai, mi] = rc[-1] if num_dets else 0

                        # Make precision monotonically decreasing so the
                        # interpolated curve matches COCO's definition.
                        pr = pr.copy()
                        for i in range(num_dets - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]
                        indices = np.searchsorted(rc, RECALL_THRESHOLDS, side="left")
                        q = np.zeros(num_r)
                        for ri, pi in enumerate(indices):
                            if pi < num_dets:
                                q[ri] = pr[pi]
                        precision[ti, :, ki, ai, mi] = q

        return precision, recall

    def summarize(self) -> List[float]:
        precision, recall = self.accumulate()

        def mean_precision(iou: Optional[float], area: str, max_det: int) -> float:
            ai = [i for i, (label, _) in enumerate(AREA_RANGES) if label == area][0]
            mi = MAX_DETS.index(max_det)
            values = precision[:, :, :, ai, mi]
            if iou is not None:
                ti = int(np.argmin(np.abs(IOU_THRESHOLDS - iou)))
                values = values[ti : ti + 1]
            valid = values[values > -1]
            return float(valid.mean()) if valid.size else -1.0

        def mean_recall(area: str, max_det: int) -> float:
            ai = [i for i, (label, _) in enumerate(AREA_RANGES) if label == area][0]
            mi = MAX_DETS.index(max_det)
            values = recall[:, :, ai, mi]
            valid = values[values > -1]
            return float(valid.mean()) if valid.size else -1.0

        return [
            mean_precision(None, "all", 100),
            mean_precision(0.5, "all", 100),
            mean_precision(0.75, "all", 100),
            mean_precision(None, "small", 100),
            mean_precision(None, "medium", 100),
            mean_precision(None, "large", 100),
            mean_recall("all", 1),
            mean_recall("all", 10),
            mean_recall("all", 100),
            mean_recall("small", 100),
            mean_recall("medium", 100),
            mean_recall("large", 100),
        ]

    def per_class_ap50(self) -> List[float]:
        """AP@0.50 per class, for the human-readable per-class table."""
        precision, _ = self.accumulate()
        ti = int(np.argmin(np.abs(IOU_THRESHOLDS - 0.5)))
        ai, mi = 0, MAX_DETS.index(100)
        out: List[float] = []
        for ki in range(self.num_classes):
            values = precision[ti, :, ki, ai, mi]
            valid = values[values > -1]
            out.append(float(valid.mean()) if valid.size else -1.0)
        return out


def _evaluate_with_pycocotools(
    data: Any, detections: Sequence[Dict[str, Any]]
) -> Optional[Tuple[List[float], List[float]]]:
    """Reference evaluation via pycocotools, or None when it is unavailable."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return None

    import contextlib
    import io

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(data.anno_path)
            results = [
                {
                    "image_id": int(d["image_id"]),
                    "category_id": int(data.cat_ids[int(d["clsid"])]),
                    "bbox": [
                        float(d["bbox_xyxy"][0]),
                        float(d["bbox_xyxy"][1]),
                        float(d["bbox_xyxy"][2] - d["bbox_xyxy"][0]),
                        float(d["bbox_xyxy"][3] - d["bbox_xyxy"][1]),
                    ],
                    "score": float(d["score"]),
                }
                for d in detections
            ]
            if not results:
                return [0.0] * 12, [0.0] * data.num_classes
            coco_dt = coco_gt.loadRes(results)
            evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        stats = [float(v) for v in evaluator.stats]

        precisions = evaluator.eval["precision"]  # [T, R, K, A, M]
        per_class: List[float] = []
        for ki in range(data.num_classes):
            values = precisions[0, :, ki, 0, 2]
            valid = values[values > -1]
            per_class.append(float(valid.mean()) if valid.size else -1.0)
        return stats, per_class
    except Exception as exc:  # noqa: BLE001 - never let the reference path break a run
        log("WARNING: pycocotools evaluation failed ({}); using the built-in evaluator.".format(exc))
        return None


def evaluate_detections(
    data: Any,
    detections: Sequence[Dict[str, Any]],
    records: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Evaluate `detections` against `data`'s ground truth.

    Prefers `pycocotools` for exactness and falls back to the built-in evaluator.
    Returns `{stats, per_class_ap50, class_names, engine, num_detections}`.
    """
    reference = _evaluate_with_pycocotools(data, detections)
    if reference is not None:
        stats, per_class = reference
        engine = "pycocotools"
    else:
        evaluator = CocoDetectionEvaluator(data.num_classes)
        evaluator.add_ground_truth(records if records is not None else data.images)
        evaluator.add_detections(detections)
        stats = evaluator.summarize()
        per_class = evaluator.per_class_ap50()
        engine = "torchtrain"

    return {
        "stats": stats,
        "per_class_ap50": per_class,
        "class_names": list(data.class_names),
        "engine": engine,
        "num_detections": len(detections),
    }
