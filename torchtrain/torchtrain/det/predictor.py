"""
Detection inference.

Output layout mirrors PaddleDetection's `tools/infer.py`:

    <save_dir>/<name>.jpg     # the image with boxes + labels drawn on it
    <save_dir>/bbox.json      # COCO-format results for downstream tooling

The rendered images are what the platform's validation page shows (it scans the
output directory for viewable images), and `bbox.json` is what a script would
consume.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from .. import logger as L
from ..seg.dataset import read_image
from ..seg.predictor import IMAGE_EXTS, collect_images, color_map


def _draw_boxes(
    image: np.ndarray,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    scores: Sequence[float],
    class_names: Sequence[str],
) -> np.ndarray:
    """Draw boxes with class-coloured outlines and a `name score` caption."""
    import cv2

    canvas = np.clip(image, 0, 255).astype(np.uint8).copy()
    if canvas.ndim == 2:
        canvas = np.repeat(canvas[:, :, None], 3, axis=2)
    palette = np.asarray(color_map(max(1, len(class_names)))[: max(1, len(class_names)) * 3], dtype=np.uint8)
    palette = palette.reshape(-1, 3)

    height = canvas.shape[0]
    thickness = max(1, int(round(height / 500.0)))
    font_scale = max(0.35, height / 1600.0)

    for box, label, score in zip(boxes, labels, scores):
        clsid = int(label)
        colour = tuple(int(c) for c in palette[clsid % len(palette)][::-1])  # RGB -> BGR
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)
        name = class_names[clsid] if clsid < len(class_names) else str(clsid)
        caption = "{} {:.2f}".format(name, float(score))
        (text_w, text_h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = max(0, y1 - text_h - 4)
        cv2.rectangle(canvas, (x1, top), (x1 + text_w + 2, top + text_h + 4), colour, -1)
        cv2.putText(
            canvas, caption, (x1 + 1, top + text_h + 1),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return canvas


@torch.no_grad()
def predict_from_weights(cfg: Dict[str, Any], args: Any, weights: str, save_dir: str) -> Dict[str, Any]:
    """`tools/predict.py` entrypoint for detection."""
    from . import trainer as det_trainer

    model, setup, meta, device = det_trainer.load_model_for_inference(cfg, args, weights, need_dataset=False)
    threshold = float(getattr(args, "score_threshold", 0.5) or 0.5)
    class_names = _class_names(cfg, int(meta["num_classes"]))
    image_paths = collect_images(args.image_path)
    os.makedirs(save_dir, exist_ok=True)

    L.log("Number of predict images = {}".format(len(image_paths)))
    L.log("Score threshold = {}".format(threshold))

    results: List[Dict[str, Any]] = []
    written: List[str] = []
    total_boxes = 0

    for index, path in enumerate(image_paths, start=1):
        original = read_image(path)
        tensor = torch.from_numpy(np.ascontiguousarray(original.transpose(2, 0, 1))).to(device) / 255.0
        output = model([tensor])[0]

        boxes = output["boxes"].detach().to("cpu").numpy()
        scores = output["scores"].detach().to("cpu").numpy()
        labels = output["labels"].detach().to("cpu").numpy() - 1  # torchvision -> Paddle clsid
        keep = (scores >= threshold) & (labels >= 0)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        total_boxes += int(len(boxes))

        rendered = _draw_boxes(original, boxes, labels, scores, class_names)
        target = os.path.join(save_dir, os.path.splitext(os.path.basename(path))[0] + ".jpg")
        import cv2

        cv2.imwrite(target, rendered[:, :, ::-1])  # RGB -> BGR for imwrite
        written.append(target)

        for box, score, label in zip(boxes, scores, labels):
            results.append(
                {
                    "image_file": os.path.basename(path),
                    "category_id": int(label),
                    "category": class_names[int(label)] if int(label) < len(class_names) else str(int(label)),
                    "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                    "score": float(score),
                }
            )

        L.log("Detection results: {} boxes -> {}".format(len(boxes), target))
        if index % 20 == 0:
            L.log("Predicting [{}/{}]".format(index, len(image_paths)))

    json_path = os.path.join(save_dir, "bbox.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    L.log("Total {} detections above threshold across {} images.".format(total_boxes, len(image_paths)))
    L.log("Inference results saved to {}".format(os.path.abspath(save_dir)))
    del setup
    return {"save_dir": save_dir, "images": written, "imageCount": len(image_paths), "bbox_json": json_path}


def _class_names(cfg: Dict[str, Any], num_classes: int) -> List[str]:
    """Read class names from the eval/test annotation file when reachable."""
    from . import dataset as dsmod

    for key in ("EvalDataset", "TrainDataset", "TestDataset"):
        block = cfg.get(key)
        if not isinstance(block, dict):
            continue
        try:
            anno_path, image_dir = dsmod.resolve_dataset_block(block, key)
            return dsmod.CocoData(anno_path, image_dir).class_names
        except (FileNotFoundError, ValueError, KeyError):
            continue
    return ["class_{}".format(i) for i in range(num_classes)]


__all__ = ["predict_from_weights", "IMAGE_EXTS"]
