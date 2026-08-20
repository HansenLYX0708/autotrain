"""
Detection transforms, addressed by PaddleDetection's op names.

Scope is deliberately narrower than PaddleDetection's: torchvision detectors
carry their own `GeneralizedRCNNTransform`, which resizes to a `[min_size,
max_size]` range and normalises with ImageNet statistics *inside* the model. So:

* `NormalizeImage` / `Permute` / `Decode` / `PadGT` are accepted and ignored —
  they describe work the torchvision model already does. Ignoring them is
  correct, not lazy: applying them here would double-normalise the input.
* `Resize` / `BatchRandomResize` are read only to derive the model's
  `min_size` / `max_size`, which is where torchvision wants that information.
* The geometric/photometric augmentations that genuinely change the sample
  (`RandomFlip`, `RandomDistort`, `RandomExpand`, `RandomCrop`) are implemented.

Unknown ops are skipped with a warning so a config referencing a
PaddleDetection-only augmentation still trains.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..logger import log

# Ops that describe preprocessing torchvision performs internally.
_NOOP_OPS = {
    "Decode",
    "NormalizeImage",
    "Permute",
    "PadGT",
    "NormalizeBox",
    "BboxXYXY2XYWH",
    "Gt2YoloTarget",
    "PadBatch",
    "Pad",
}

# Ops whose only useful information is the target resolution.
_SIZE_OPS = {"Resize", "BatchRandomResize", "RandomResize", "LetterBoxResize", "KeepRatioResize"}

Sample = Tuple[np.ndarray, Dict[str, np.ndarray]]


class DetTransform:
    def __call__(self, im: np.ndarray, target: Dict[str, np.ndarray]) -> Sample:
        return self.apply(im, target)

    def apply(self, im: np.ndarray, target: Dict[str, np.ndarray]) -> Sample:  # pragma: no cover
        raise NotImplementedError


class DetCompose(DetTransform):
    def __init__(self, transforms: Sequence[DetTransform]) -> None:
        self.transforms = list(transforms)

    def apply(self, im: np.ndarray, target: Dict[str, np.ndarray]) -> Sample:
        for t in self.transforms:
            im, target = t(im, target)
        return im, target

    def __len__(self) -> int:
        return len(self.transforms)


class RandomFlip(DetTransform):
    """Horizontal flip; boxes are mirrored in absolute xyxy coordinates."""

    def __init__(self, prob: float = 0.5, **_: Any) -> None:
        self.prob = float(prob)

    def apply(self, im: np.ndarray, target: Dict[str, np.ndarray]) -> Sample:
        if random.random() >= self.prob:
            return im, target
        width = im.shape[1]
        im = im[:, ::-1, :]
        boxes = target.get("boxes")
        if boxes is not None and len(boxes):
            boxes = boxes.copy()
            x1 = boxes[:, 0].copy()
            boxes[:, 0] = width - boxes[:, 2]
            boxes[:, 2] = width - x1
            target["boxes"] = boxes
        return np.ascontiguousarray(im), target


class RandomVerticalFlip(DetTransform):
    def __init__(self, prob: float = 0.5, **_: Any) -> None:
        self.prob = float(prob)

    def apply(self, im: np.ndarray, target: Dict[str, np.ndarray]) -> Sample:
        if random.random() >= self.prob:
            return im, target
        height = im.shape[0]
        im = im[::-1, :, :]
        boxes = target.get("boxes")
        if boxes is not None and len(boxes):
            boxes = boxes.copy()
            y1 = boxes[:, 1].copy()
            boxes[:, 1] = height - boxes[:, 3]
            boxes[:, 3] = height - y1
            target["boxes"] = boxes
        return np.ascontiguousarray(im), target


class RandomDistort(DetTransform):
    def __init__(
        self,
        brightness: Any = None,
        contrast: Any = None,
        saturation: Any = None,
        prob: float = 0.5,
        **_: Any,
    ) -> None:
        self.brightness = _range(brightness, 0.25)
        self.contrast = _range(contrast, 0.25)
        self.saturation = _range(saturation, 0.25)
        self.prob = float(prob)

    def apply(self, im: np.ndarray, target: Dict[str, np.ndarray]) -> Sample:
        if random.random() >= self.prob:
            return im, target
        out = im.astype(np.float32)
        out = out + random.uniform(-self.brightness, self.brightness) * 255.0
        mean = out.mean()
        out = (out - mean) * (1.0 + random.uniform(-self.contrast, self.contrast)) + mean
        if out.ndim == 3 and out.shape[2] == 3:
            gray = out.mean(axis=2, keepdims=True)
            out = (out - gray) * (1.0 + random.uniform(-self.saturation, self.saturation)) + gray
        return np.clip(out, 0, 255), target


class RandomExpand(DetTransform):
    """Zoom out by pasting the image into a larger canvas (small-object aug)."""

    def __init__(self, ratio: float = 2.0, prob: float = 0.5, fill_value: Any = (127.5, 127.5, 127.5), **_: Any) -> None:
        self.ratio = float(ratio)
        self.prob = float(prob)
        self.fill_value = _triple(fill_value, 127.5)

    def apply(self, im: np.ndarray, target: Dict[str, np.ndarray]) -> Sample:
        if self.ratio <= 1.0 or random.random() >= self.prob:
            return im, target
        h, w = im.shape[:2]
        scale = random.uniform(1.0, self.ratio)
        new_h, new_w = int(h * scale), int(w * scale)
        top = random.randint(0, new_h - h)
        left = random.randint(0, new_w - w)
        canvas = np.zeros((new_h, new_w, im.shape[2]), dtype=np.float32)
        canvas[:, :] = np.asarray(self.fill_value[: im.shape[2]], dtype=np.float32)
        canvas[top : top + h, left : left + w] = im
        boxes = target.get("boxes")
        if boxes is not None and len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top
            target["boxes"] = boxes
        return canvas, target


class RandomCrop(DetTransform):
    """Crop a random window, dropping boxes whose centre falls outside it."""

    def __init__(self, prob: float = 0.5, min_scale: float = 0.6, num_attempts: int = 10, **_: Any) -> None:
        self.prob = float(prob)
        self.min_scale = float(min_scale)
        self.num_attempts = int(num_attempts)

    def apply(self, im: np.ndarray, target: Dict[str, np.ndarray]) -> Sample:
        boxes = target.get("boxes")
        if boxes is None or not len(boxes) or random.random() >= self.prob:
            return im, target
        h, w = im.shape[:2]
        for _ in range(self.num_attempts):
            scale = random.uniform(self.min_scale, 1.0)
            crop_h, crop_w = int(h * scale), int(w * scale)
            if crop_h < 8 or crop_w < 8:
                continue
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
            cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
            keep = (cx >= left) & (cx < left + crop_w) & (cy >= top) & (cy < top + crop_h)
            if not bool(keep.any()):
                continue  # never produce an empty target: it teaches nothing
            cropped = boxes[keep].copy()
            cropped[:, [0, 2]] = np.clip(cropped[:, [0, 2]] - left, 0, crop_w - 1)
            cropped[:, [1, 3]] = np.clip(cropped[:, [1, 3]] - top, 0, crop_h - 1)
            valid = (cropped[:, 2] > cropped[:, 0] + 1) & (cropped[:, 3] > cropped[:, 1] + 1)
            if not bool(valid.any()):
                continue
            new_target = dict(target)
            new_target["boxes"] = cropped[valid]
            for key in ("labels", "iscrowd"):
                if key in target and target[key] is not None and len(target[key]):
                    new_target[key] = target[key][keep][valid]
            return np.ascontiguousarray(im[top : top + crop_h, left : left + crop_w]), new_target
        return im, target


DET_TRANSFORMS: Dict[str, Any] = {
    "RandomFlip": RandomFlip,
    "RandomHorizontalFlip": RandomFlip,
    "RandomVerticalFlip": RandomVerticalFlip,
    "RandomDistort": RandomDistort,
    "RandomExpand": RandomExpand,
    "RandomCrop": RandomCrop,
}


def parse_transform_specs(specs: Optional[Sequence[Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Normalise PaddleDetection's `- OpName: {params}` list into pairs.

    PaddleDetection writes each op as a single-key mapping; PaddleSeg-style
    `- type: OpName` is also accepted so a hand-written config works either way.
    """
    out: List[Tuple[str, Dict[str, Any]]] = []
    for spec in specs or []:
        if isinstance(spec, str):
            out.append((spec, {}))
        elif isinstance(spec, dict):
            if "type" in spec or "name" in spec or "__tag__" in spec:
                name = str(spec.get("type") or spec.get("name") or spec.get("__tag__"))
                out.append((name, {k: v for k, v in spec.items() if k not in ("type", "name", "__tag__")}))
            else:
                for name, params in spec.items():
                    out.append((str(name), params if isinstance(params, dict) else {}))
    return out


def build_transforms(specs: Optional[Sequence[Any]]) -> DetCompose:
    ops: List[DetTransform] = []
    for name, params in parse_transform_specs(specs):
        if name in _NOOP_OPS or name in _SIZE_OPS:
            continue
        cls = DET_TRANSFORMS.get(name)
        if cls is None:
            log("WARNING: unsupported detection transform '{}' skipped.".format(name))
            continue
        try:
            ops.append(cls(**params))
        except TypeError:
            ops.append(cls())
    return DetCompose(ops)


def infer_input_size(*spec_lists: Optional[Sequence[Any]]) -> Optional[Tuple[int, int]]:
    """Derive `(min_size, max_size)` for torchvision from Resize-style ops.

    A single square target (`[640, 640]`) becomes `min_size=640, max_size=640`;
    a multi-scale list (`[320, ..., 768]`) becomes `min(list), max(list)`, which
    is how torchvision expresses scale jitter.
    """
    sizes: List[int] = []
    for specs in spec_lists:
        for name, params in parse_transform_specs(specs):
            if name not in _SIZE_OPS:
                continue
            target = params.get("target_size", params.get("size"))
            sizes.extend(_flatten_ints(target))
    sizes = [s for s in sizes if s >= 32]
    if not sizes:
        return None
    return min(sizes), max(sizes)


def _flatten_ints(value: Any) -> List[int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        out: List[int] = []
        for item in value:
            out.extend(_flatten_ints(item))
        return out
    return []


def _range(value: Any, default: float) -> float:
    """PaddleDetection writes distortion strength as a scalar or a `[lo, hi]`."""
    if isinstance(value, (int, float)):
        return abs(float(value))
    if isinstance(value, (list, tuple)) and value:
        nums = [float(v) for v in value if isinstance(v, (int, float))]
        if len(nums) >= 2:
            return abs(nums[1] - nums[0]) / 2.0
        if nums:
            return abs(nums[0])
    if isinstance(value, dict):
        return _range(value.get("range", value.get("prob")), default)
    return default


def _triple(value: Any, default: float) -> List[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * 3
    if isinstance(value, (list, tuple)) and value:
        nums = [float(v) for v in value if isinstance(v, (int, float))]
        if len(nums) >= 3:
            return nums[:3]
        if nums:
            return (nums * 3)[:3]
    return [default] * 3
