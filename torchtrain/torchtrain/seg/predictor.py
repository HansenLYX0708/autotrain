"""
Segmentation inference.

The output layout mirrors PaddleSeg's `predict.py` exactly:

    <save_dir>/pseudo_color_prediction/<name>.png   # paletted mask, pixel == class id
    <save_dir>/added_prediction/<name>.png          # 50/50 blend over the input

Both matter to the platform:
* `pseudo_color_prediction` is a *paletted* PNG whose palette is PaddleSeg's
  VOC-derived color map, which `src/lib/seg-colors.ts` reproduces. The
  annotation/analysis tooling (`tools/tem_analysis`) reads these masks back as
  class indices, so writing an RGB PNG here would break it.
* `added_prediction` is what the validation page displays, because
  `findInferenceImages()` scans the output directory for viewable images.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .. import logger as L
from . import dataset as dsmod
from . import transforms as T

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def color_map(num_classes: int) -> List[int]:
    """PaddleSeg's `get_color_map_list(n)[3:]` flattened to a PNG palette.

    Must stay byte-identical to `getSegColorMap` in `src/lib/seg-colors.ts`, or
    the UI legend will label regions with the wrong class.
    """
    n = max(1, num_classes) + 1
    cm = [0] * (n * 3)
    for i in range(n):
        lab, j = i, 0
        while lab:
            cm[i * 3] |= ((lab >> 0) & 1) << (7 - j)
            cm[i * 3 + 1] |= ((lab >> 1) & 1) << (7 - j)
            cm[i * 3 + 2] |= ((lab >> 2) & 1) << (7 - j)
            j += 1
            lab >>= 3
        # `lab` is consumed above; loop exits once every bit has been placed.
    shifted = cm[3:]
    palette = shifted[: num_classes * 3]
    # PNG palettes hold 256 entries; pad the tail with black.
    return palette + [0] * (768 - len(palette))


def collect_images(image_path: str) -> List[str]:
    """Expand a file or directory into a sorted list of image paths."""
    if not image_path:
        raise ValueError("--image_path is required")
    if os.path.isfile(image_path):
        return [image_path]
    if not os.path.isdir(image_path):
        raise FileNotFoundError("Input path does not exist: {}".format(image_path))
    found: List[str] = []
    for root, _dirs, files in os.walk(image_path):
        for name in sorted(files):
            if name.lower().endswith(IMAGE_EXTS):
                found.append(os.path.join(root, name))
    if not found:
        raise ValueError("No images found under {} (looked for {}).".format(image_path, ", ".join(IMAGE_EXTS)))
    return found


def save_pseudo_color(mask: np.ndarray, path: str, num_classes: int) -> None:
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = Image.fromarray(mask.astype(np.uint8), mode="P")
    image.putpalette(color_map(num_classes))
    image.save(path)


def save_blend(image: np.ndarray, mask: np.ndarray, path: str, num_classes: int, weight: float = 0.6) -> None:
    """Write the `added_prediction` overlay (input dimmed, mask coloured)."""
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    palette = np.asarray(color_map(num_classes)[: num_classes * 3], dtype=np.uint8).reshape(-1, 3)
    colored = palette[np.clip(mask, 0, num_classes - 1)]
    base = np.clip(image, 0, 255).astype(np.float32)
    if base.ndim == 2:
        base = np.repeat(base[:, :, None], 3, axis=2)
    blended = base * weight + colored.astype(np.float32) * (1.0 - weight)
    Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)).save(path)


@torch.no_grad()
def predict(
    model: Any,
    setup: Any,
    device: torch.device,
    image_paths: Sequence[str],
    save_dir: str,
    transforms: Optional[T.Compose] = None,
) -> Dict[str, Any]:
    """Run inference over `image_paths`, writing PaddleSeg-style outputs."""
    model.eval()
    num_classes = int(setup.num_classes)
    # Reuse the validation transforms so inference preprocessing matches eval.
    if transforms is None:
        val_block = setup.cfg.get("val_dataset") or setup.cfg.get("train_dataset") or {}
        transforms = T.build_transforms(
            [op for op in (val_block.get("transforms") or []) if _is_eval_safe(op)],
            default_size=setup.default_size,
        )

    pseudo_dir = os.path.join(save_dir, "pseudo_color_prediction")
    added_dir = os.path.join(save_dir, "added_prediction")
    os.makedirs(pseudo_dir, exist_ok=True)
    os.makedirs(added_dir, exist_ok=True)

    L.log("Number of predict images = {}".format(len(image_paths)))
    written: List[str] = []
    class_pixels = np.zeros(num_classes, dtype=np.int64)

    for index, path in enumerate(image_paths, start=1):
        original = dsmod.read_image(path)
        im, _ = transforms(original.copy(), None)
        tensor = torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))[None].to(device)

        logits = model(tensor)
        # Predict at the input's native resolution so the mask can be overlaid
        # on (and measured against) the original image without rescaling.
        logit = torch.nn.functional.interpolate(
            logits[0], size=original.shape[:2], mode="bilinear", align_corners=setup.align_corners
        )
        mask = logit.argmax(dim=1)[0].to("cpu").numpy().astype(np.uint8)
        class_pixels += np.histogram(mask, bins=np.arange(num_classes + 1))[0]

        stem = os.path.splitext(os.path.basename(path))[0]
        pseudo_path = os.path.join(pseudo_dir, stem + ".png")
        added_path = os.path.join(added_dir, stem + ".png")
        save_pseudo_color(mask, pseudo_path, num_classes)
        save_blend(original, mask, added_path, num_classes)
        written.extend([pseudo_path, added_path])

        if index % 10 == 0 or index == len(image_paths):
            L.log("Predicting [{}/{}] {}".format(index, len(image_paths), os.path.basename(path)))

    total = int(class_pixels.sum()) or 1
    L.log(
        "Class pixel ratio: {}".format(
            ", ".join("{}={:.4f}".format(i, class_pixels[i] / total) for i in range(num_classes))
        )
    )
    L.log("Prediction results saved to {}".format(os.path.abspath(save_dir)))
    return {
        "save_dir": save_dir,
        "pseudo_color_prediction": pseudo_dir,
        "added_prediction": added_dir,
        "images": written,
        "imageCount": len(image_paths),
        "classPixels": [int(v) for v in class_pixels],
    }


def _is_eval_safe(op: Any) -> bool:
    """Drop random augmentations from a transform list reused for inference."""
    name = ""
    if isinstance(op, str):
        name = op
    elif isinstance(op, dict):
        name = str(op.get("type") or op.get("name") or "")
    return not name.startswith("Random")
