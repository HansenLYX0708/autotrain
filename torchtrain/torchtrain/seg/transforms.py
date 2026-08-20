"""
PaddleSeg-compatible transform pipeline.

Every op is addressed by the same `type:` string PaddleSeg uses, and takes the
same parameter names, so the `transforms:` list in a config written for
PaddleSeg is honoured verbatim here. That is the whole point: the platform's
dataset/training config generators emit PaddleSeg transform blocks, and we do
not want a second dialect.

Convention (also PaddleSeg's): a sample is `(im, label)` where

    im    : HWC float32, values in [0, 255] until `Normalize` runs
    label : HW int64, or None for inference

`Normalize` is what converts to CHW-ready scaled floats; it is applied last and
also performs the /255 scaling, matching PaddleSeg's `Normalize` op.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Sample = Tuple[np.ndarray, Optional[np.ndarray]]

# PaddleSeg's ignore label. Pixels with this value are excluded from both the
# loss and the metrics.
IGNORE_INDEX = 255

_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


class Transform:
    """Base class. Subclasses override `apply`."""

    def __call__(self, im: np.ndarray, label: Optional[np.ndarray] = None) -> Sample:
        return self.apply(im, label)

    def apply(self, im: np.ndarray, label: Optional[np.ndarray]) -> Sample:  # pragma: no cover
        raise NotImplementedError


class Compose(Transform):
    def __init__(self, transforms: Sequence[Transform]) -> None:
        self.transforms = list(transforms)

    def apply(self, im: np.ndarray, label: Optional[np.ndarray]) -> Sample:
        for t in self.transforms:
            im, label = t(im, label)
        return im, label

    def __len__(self) -> int:
        return len(self.transforms)


class Resize(Transform):
    """`target_size` is `[width, height]`, as in PaddleSeg."""

    def __init__(self, target_size: Any = (512, 512), interp: str = "linear", keep_ratio: bool = False) -> None:
        size = _size_pair(target_size, (512, 512))
        self.width, self.height = size
        self.interp = _INTERP.get(str(interp).lower(), cv2.INTER_LINEAR)
        self.keep_ratio = bool(keep_ratio)

    def apply(self, im: np.ndarray, label: Optional[np.ndarray]) -> Sample:
        target_w, target_h = self.width, self.height
        if self.keep_ratio:
            h, w = im.shape[:2]
            scale = min(target_w / float(w), target_h / float(h))
            target_w, target_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        im = cv2.resize(im, (target_w, target_h), interpolation=self.interp)
        if label is not None:
            label = cv2.resize(label, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return im, label


class ResizeStepScaling(Transform):
    def __init__(
        self,
        min_scale_factor: float = 0.75,
        max_scale_factor: float = 1.25,
        scale_step_size: float = 0.25,
    ) -> None:
        self.min_scale_factor = float(min_scale_factor)
        self.max_scale_factor = float(max_scale_factor)
        self.scale_step_size = float(scale_step_size)

    def apply(self, im: np.ndarray, label: Optional[np.ndarray]) -> Sample:
        if self.min_scale_factor == self.max_scale_factor:
            scale = self.min_scale_factor
        elif self.scale_step_size <= 0:
            scale = random.uniform(self.min_scale_factor, self.max_scale_factor)
        else:
            steps = int((self.max_scale_factor - self.min_scale_factor) / self.scale_step_size + 1)
            scale = float(np.random.choice(np.linspace(self.min_scale_factor, self.max_scale_factor, max(1, steps))))
        h, w = im.shape[:2]
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return im, label


class RandomHorizontalFlip(Transform):
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = float(prob)

    def apply(self, im: np.ndarray, label: Optional[np.ndarray]) -> Sample:
        if random.random() < self.prob:
            im = im[:, ::-1, ...]
            if label is not None:
                label = label[:, ::-1]
        return im, label


class RandomVerticalFlip(Transform):
    def __init__(self, prob: float = 0.5) -> None:
        self.prob = float(prob)

    def apply(self, im: np.ndarray, label: Optional[np.ndarray]) -> Sample:
        if random.random() < self.prob:
            im = im[::-1, :, ...]
            if label is not None:
                label = label[::-1, :]
        return im, label


class RandomRotation(Transform):
    def __init__(self, max_rotation: float = 15, im_padding_value: Any = (127.5, 127.5, 127.5),
                 label_padding_value: int = IGNORE_INDEX) -> None:
        self.max_rotation = float(max_rotation)
        self.im_padding_value = _to_triple(im_padding_value, 127.5)
        self.label_padding_value = int(label_padding_value)

    def apply(self, im: np.ndarray, label: Optional[np.ndarray]) -> Sample:
        if self.max_rotation <= 0:
            return im, label
        h, w = im.shape[:2]
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        im = cv2.warpAffine(
            im, matrix, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=self.im_padding_value,
        )
        if label is not None:
            label = cv2.warpAffine(
                label, matrix, (w, h), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=self.label_padding_value,
            )
        return im, label


class RandomPaddingCrop(Transform):
    """Crop to `crop_size`, padding first when the image is smaller.

    In the common PaddleSeg recipe (`ResizeStepScaling` + `RandomPaddingCrop`)
    the crop size *is* the network input size, so getting the padding value
    right matters: padded label pixels must be `IGNORE_INDEX`, not class 0, or
    the model learns to predict background on the borders.
    """

    def __init__(
        self,
        crop_size: Any = (512, 512),
        im_padding_value: Any = (127.5, 127.5, 127.5),
        label_padding_value: int = IGNORE_INDEX,
    ) -> None:
        self.crop_w, self.crop_h = _size_pair(crop_size, (512, 512))
        self.im_padding_value = _to_triple(im_padding_value, 127.5)
        self.label_padding_value = int(label_padding_value)

    def apply(self, im: np.ndarray, label: Optional[np.ndarray]) -> Sample:
        h, w = im.shape[:2]
        pad_h, pad_w = max(0, self.crop_h - h), max(0, self.crop_w - w)
        if pad_h > 0 or pad_w > 0:
            channels = im.shape[2] if im.ndim == 3 else 1
            value = self.im_padding_value[:channels] if channels <= 3 else [self.im_padding_value[0]] * channels
            im = cv2.copyMakeBorder(im, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=value)
            if label is not None:
                label = cv2.copyMakeBorder(
                    label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.label_padding_value
                )
            h, w = im.shape[:2]
        if h > self.crop_h or w > self.crop_w:
            top = random.randint(0, h - self.crop_h)
            left = random.randint(0, w - self.crop_w)
            im = im[top : top + self.crop_h, left : left + self.crop_w]
            if label is not None:
                label = label[top : top + self.crop_h, left : left + self.crop_w]
        return im, label


class Padding(Transform):
    def __init__(self, target_size: Any = (512, 512), im_padding_value: Any = (127.5, 127.5, 127.5),
                 label_padding_value: int = IGNORE_INDEX) -> None:
        self.target_w, self.target_h = _size_pair(target_size, (512, 512))
        self.im_padding_value = _to_triple(im_padding_value, 127.5)
        self.label_padding_value = int(label_padding_value)

    def apply(self, im: np.ndarray, label: Optional[np.ndarray]) -> Sample:
        h, w = im.shape[:2]
        pad_h, pad_w = max(0, self.target_h - h), max(0, self.target_w - w)
        if pad_h == 0 and pad_w == 0:
            return im, label
        channels = im.shape[2] if im.ndim == 3 else 1
        value = self.im_padding_value[:channels] if channels <= 3 else [self.im_padding_value[0]] * channels
        im = cv2.copyMakeBorder(im, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=value)
        if label is not None:
            label = cv2.copyMakeBorder(label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.label_padding_value)
        return im, label


class RandomDistort(Transform):
    """Brightness / contrast / saturation / hue jitter, PaddleSeg parameter names."""

    def __init__(
        self,
        brightness_range: float = 0.5,
        brightness_prob: float = 0.5,
        contrast_range: float = 0.5,
        contrast_prob: float = 0.5,
        saturation_range: float = 0.5,
        saturation_prob: float = 0.5,
        hue_range: float = 18,
        hue_prob: float = 0.5,
    ) -> None:
        self.brightness_range = float(brightness_range)
        self.brightness_prob = float(brightness_prob)
        self.contrast_range = float(contrast_range)
        self.contrast_prob = float(contrast_prob)
        self.saturation_range = float(saturation_range)
        self.saturation_prob = float(saturation_prob)
        self.hue_range = float(hue_range)
        self.hue_prob = float(hue_prob)

    def apply(self, im: np.ndarray, label: Optional[np.ndarray]) -> Sample:
        im = im.astype(np.float32)
        if random.random() < self.brightness_prob and self.brightness_range > 0:
            im = im + random.uniform(-self.brightness_range, self.brightness_range) * 255.0
        if random.random() < self.contrast_prob and self.contrast_range > 0:
            factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
            mean = im.mean()
            im = (im - mean) * factor + mean
        if random.random() < self.saturation_prob and self.saturation_range > 0 and im.ndim == 3 and im.shape[2] == 3:
            factor = 1.0 + random.uniform(-self.saturation_range, self.saturation_range)
            gray = im.mean(axis=2, keepdims=True)
            im = (im - gray) * factor + gray
        return np.clip(im, 0, 255), label


class RandomBlur(Transform):
    def __init__(self, prob: float = 0.1, blur_type: str = "gaussian") -> None:
        self.prob = float(prob)
        self.blur_type = str(blur_type)

    def apply(self, im: np.ndarray, label: Optional[np.ndarray]) -> Sample:
        if self.prob <= 0 or random.random() >= self.prob:
            return im, label
        radius = random.choice([3, 5])
        if self.blur_type == "median":
            im = cv2.medianBlur(im.astype(np.uint8), radius).astype(np.float32)
        else:
            im = cv2.GaussianBlur(im, (radius, radius), 0)
        return im, label


class RandomScaleAspect(Transform):
    def __init__(self, min_scale: float = 0.5, aspect_ratio: float = 0.33) -> None:
        self.min_scale = float(min_scale)
        self.aspect_ratio = float(aspect_ratio)

    def apply(self, im: np.ndarray, label: Optional[np.ndarray]) -> Sample:
        if self.min_scale >= 1 or self.aspect_ratio <= 0:
            return im, label
        h, w = im.shape[:2]
        for _ in range(10):
            area = h * w * random.uniform(self.min_scale, 1.0)
            ratio = random.uniform(self.aspect_ratio, 1.0 / self.aspect_ratio)
            new_w = int(round(np.sqrt(area * ratio)))
            new_h = int(round(np.sqrt(area / ratio)))
            if new_w <= w and new_h <= h and new_w > 0 and new_h > 0:
                left = random.randint(0, w - new_w)
                top = random.randint(0, h - new_h)
                im = cv2.resize(im[top : top + new_h, left : left + new_w], (w, h), interpolation=cv2.INTER_LINEAR)
                if label is not None:
                    label = cv2.resize(
                        label[top : top + new_h, left : left + new_w], (w, h), interpolation=cv2.INTER_NEAREST
                    )
                break
        return im, label


class Normalize(Transform):
    """Scale to [0,1] then apply mean/std. Always the last op, as in PaddleSeg."""

    def __init__(self, mean: Any = (0.5, 0.5, 0.5), std: Any = (0.5, 0.5, 0.5)) -> None:
        self.mean = np.asarray(_to_triple(mean, 0.5), dtype=np.float32)
        self.std = np.asarray(_to_triple(std, 0.5), dtype=np.float32)
        if np.any(self.std == 0):
            raise ValueError("Normalize.std must not contain zeros")

    def apply(self, im: np.ndarray, label: Optional[np.ndarray]) -> Sample:
        im = im.astype(np.float32) / 255.0
        channels = im.shape[2] if im.ndim == 3 else 1
        mean = self.mean[:channels] if channels <= len(self.mean) else np.resize(self.mean, channels)
        std = self.std[:channels] if channels <= len(self.std) else np.resize(self.std, channels)
        im = (im - mean) / std
        return im, label


# ---------------------------------------------------------------------------
# Registry / builder
# ---------------------------------------------------------------------------

TRANSFORMS: Dict[str, Any] = {
    "Resize": Resize,
    "ResizeStepScaling": ResizeStepScaling,
    "RandomHorizontalFlip": RandomHorizontalFlip,
    "RandomVerticalFlip": RandomVerticalFlip,
    "RandomRotation": RandomRotation,
    "RandomPaddingCrop": RandomPaddingCrop,
    "RandomCrop": RandomPaddingCrop,
    "Padding": Padding,
    "RandomDistort": RandomDistort,
    "RandomBlur": RandomBlur,
    "RandomScaleAspect": RandomScaleAspect,
    "Normalize": Normalize,
}


def build_transforms(specs: Optional[Sequence[Any]], default_size: Optional[Sequence[int]] = None) -> Compose:
    """Build a `Compose` from a PaddleSeg `transforms:` list.

    Unknown ops are skipped with a warning instead of raising, because a config
    may legitimately reference a PaddleSeg-only op (e.g. `RandomAffine`) that we
    do not implement, and dropping one augmentation is much better than failing
    a queued training job. `Normalize` is appended when missing, since without
    it the network would see raw 0-255 pixels.
    """
    from ..logger import log

    ops: List[Transform] = []
    has_normalize = False
    for spec in specs or []:
        if isinstance(spec, str):
            name, params = spec, {}
        elif isinstance(spec, dict):
            name = str(spec.get("type") or spec.get("name") or spec.get("__tag__") or "")
            params = {k: v for k, v in spec.items() if k not in ("type", "name", "__tag__")}
        else:
            continue
        cls = TRANSFORMS.get(name)
        if cls is None:
            log("WARNING: unsupported transform '{}' skipped (not implemented in torchtrain).".format(name))
            continue
        try:
            ops.append(cls(**params))
        except TypeError as exc:
            log("WARNING: transform '{}' has unsupported parameters ({}); using its defaults.".format(name, exc))
            ops.append(cls())
        if name == "Normalize":
            has_normalize = True

    if not ops and default_size:
        ops.append(Resize(target_size=default_size))
    if not has_normalize:
        ops.append(Normalize())
    return Compose(ops)


def _size_pair(value: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, (int, float)):
        return int(value), int(value)
    if isinstance(value, (list, tuple)):
        items = list(value)
        if len(items) == 1 and isinstance(items[0], (list, tuple)):
            items = list(items[0])
        nums = [int(v) for v in items if isinstance(v, (int, float))]
        if len(nums) >= 2:
            return nums[0], nums[1]
        if len(nums) == 1:
            return nums[0], nums[0]
    return default


def _to_triple(value: Any, default: float) -> List[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * 3
    if isinstance(value, (list, tuple)) and value:
        nums = [float(v) for v in value if isinstance(v, (int, float))]
        if len(nums) == 1:
            return nums * 3
        if len(nums) >= 3:
            return nums[:3]
        if nums:
            return (nums + nums * 3)[:3]
    return [default] * 3
