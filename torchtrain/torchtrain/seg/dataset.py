"""
PaddleSeg list-file dataset.

Layout produced by the platform's `/api/datasets/labelme-to-paddleseg` converter
(and what `train_dataset.train_path` points at):

    <dataset_root>/
      JPEGImages/<name>.<ext>          # any format PIL or OpenCV can read
      Annotations/<name>.png           # 8-bit *paletted* PNG, pixel == class id
      train.txt                        # "JPEGImages/a.tif Annotations/a.png"
      val.txt
      class_names.txt                  # one name per line, background first

Two details that are easy to get wrong and worth stating:

* The mask must be read as **palette indices, not RGB**. `PIL` gives mode `"P"`
  arrays whose values are already the class ids; converting to RGB first (or
  letting OpenCV expand the palette) turns class 1 into a colour triple and
  every pixel becomes "wrong class".
* The platform's real TEM data is single-channel 16-bit-capable `.tif`.
  Grayscale is expanded to 3 channels so the ImageNet-pretrained backbones see
  the input shape they expect.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from ..logger import log
from . import transforms as T

_IMAGE_READ_ERRORS = (OSError, ValueError)


def read_image(path: str) -> np.ndarray:
    """Read an image as HWC float32 RGB in [0, 255].

    PIL first (it handles paletted and 16-bit TIFFs predictably), OpenCV as a
    fallback for the formats PIL refuses.
    """
    array: Optional[np.ndarray] = None
    try:
        from PIL import Image

        with Image.open(path) as handle:
            handle.load()
            if handle.mode in ("P", "1"):
                handle = handle.convert("L")
            array = np.asarray(handle)
    except _IMAGE_READ_ERRORS:
        array = None
    except ImportError:
        array = None

    if array is None:
        import cv2

        array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if array is None:
            raise FileNotFoundError("Could not read image: {}".format(path))
        if array.ndim == 3 and array.shape[2] >= 3:
            array = array[:, :, ::-1]  # BGR -> RGB

    array = np.asarray(array)
    if array.dtype == np.uint16:
        array = (array.astype(np.float32) / 257.0)  # 16-bit -> 8-bit range
    elif array.dtype != np.float32:
        array = array.astype(np.float32)

    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 4:
        array = array[:, :, :3]
    return np.ascontiguousarray(array, dtype=np.float32)


def read_label(path: str) -> np.ndarray:
    """Read a segmentation mask as HW int64 class ids."""
    try:
        from PIL import Image

        with Image.open(path) as handle:
            handle.load()
            # Mode "P" arrays already hold the class index; do NOT convert().
            array = np.asarray(handle)
    except (ImportError, *_IMAGE_READ_ERRORS):
        import cv2

        array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if array is None:
            raise FileNotFoundError("Could not read label: {}".format(path))

    array = np.asarray(array)
    if array.ndim == 3:
        # An RGB mask means the palette was expanded somewhere upstream; the
        # first channel is the best guess and beats crashing.
        array = array[:, :, 0]
    return np.ascontiguousarray(array.astype(np.int64))


def parse_list_file(list_path: str, dataset_root: str) -> List[Tuple[str, Optional[str]]]:
    """Parse a `train.txt`/`val.txt` into absolute `(image, label)` pairs.

    Lines are whitespace-separated; a single column means "image only"
    (inference lists). Paths are resolved against `dataset_root` when relative.
    Missing files are reported once, in bulk, so a broken dataset produces one
    actionable error instead of a wall of them.
    """
    if not os.path.isfile(list_path):
        raise FileNotFoundError("Dataset list file not found: {}".format(list_path))

    pairs: List[Tuple[str, Optional[str]]] = []
    missing: List[str] = []
    with open(list_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            image_rel = parts[0]
            label_rel = parts[1] if len(parts) > 1 else None
            image = image_rel if os.path.isabs(image_rel) else os.path.join(dataset_root, image_rel)
            label = None
            if label_rel:
                label = label_rel if os.path.isabs(label_rel) else os.path.join(dataset_root, label_rel)
            if not os.path.isfile(image):
                missing.append(image)
                continue
            if label and not os.path.isfile(label):
                missing.append(label)
                continue
            pairs.append((image, label))

    if missing:
        preview = "\n  ".join(missing[:5])
        log(
            "WARNING: {} entries in {} reference files that do not exist and were skipped:\n  {}".format(
                len(missing), list_path, preview
            )
        )
    if not pairs:
        raise ValueError(
            "No usable samples in {}. Check that dataset_root ({}) is correct.".format(list_path, dataset_root)
        )
    return pairs


class SegDataset(TorchDataset):
    """Dataset built from a PaddleSeg-style config block.

    `mode` follows PaddleSeg: `train` shuffles + augments, `val`/`test` do not.
    In val mode the sample is returned at its original resolution unless the
    config's transforms resize it, matching PaddleSeg's behaviour of evaluating
    at native size when only `Normalize` is configured.
    """

    def __init__(
        self,
        dataset_root: str,
        list_path: str,
        num_classes: int,
        mode: str = "train",
        transforms: Optional[T.Compose] = None,
        ignore_index: int = T.IGNORE_INDEX,
    ) -> None:
        self.dataset_root = dataset_root or ""
        self.mode = mode
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.transforms = transforms or T.build_transforms(None)
        self.samples = parse_list_file(list_path, self.dataset_root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label_path = self.samples[index]
        im = read_image(image_path)
        label = read_label(label_path) if label_path else None

        if label is not None:
            # Out-of-range ids would index past the logits and crash in the loss
            # with an unhelpful CUDA assert; map them to `ignore_index` instead.
            invalid = (label >= self.num_classes) & (label != self.ignore_index)
            if bool(invalid.any()):
                label = np.where(invalid, self.ignore_index, label)

        im, label = self.transforms(im, label)
        tensor = torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))
        if label is None:
            return tensor, image_path
        return tensor, torch.from_numpy(np.ascontiguousarray(label)).long()


def build_dataset(
    block: Dict[str, Any],
    mode: str,
    fallback_num_classes: Optional[int] = None,
    default_size: Optional[Sequence[int]] = None,
) -> SegDataset:
    """Build a `SegDataset` from a `train_dataset:` / `val_dataset:` block."""
    if not isinstance(block, dict):
        raise ValueError("Expected a mapping for the {}_dataset config block".format(mode))

    dataset_root = str(block.get("dataset_root") or "")
    list_key = "train_path" if mode == "train" else "val_path"
    list_path = block.get(list_key) or block.get("train_path") or block.get("val_path")
    if not list_path:
        raise ValueError(
            "{}_dataset is missing `{}` (path to the train.txt/val.txt list file).".format(mode, list_key)
        )
    list_path = str(list_path)
    if not os.path.isabs(list_path) and dataset_root:
        list_path = os.path.join(dataset_root, list_path)

    num_classes = block.get("num_classes", fallback_num_classes)
    if not num_classes:
        raise ValueError("{}_dataset is missing `num_classes`.".format(mode))

    return SegDataset(
        dataset_root=dataset_root,
        list_path=list_path,
        num_classes=int(num_classes),
        mode=mode,
        transforms=T.build_transforms(block.get("transforms"), default_size=default_size),
        ignore_index=int(block.get("ignore_index", T.IGNORE_INDEX)),
    )


def read_class_names(dataset_root: str, num_classes: int) -> List[str]:
    """Read `class_names.txt` / `labels.txt`, or synthesise generic names."""
    for candidate in ("class_names.txt", "labels.txt"):
        path = os.path.join(dataset_root or "", candidate)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                names = [line.strip() for line in handle if line.strip()]
            if names:
                if len(names) < num_classes:
                    names += ["class_{}".format(i) for i in range(len(names), num_classes)]
                return names[:num_classes]
    return ["class_{}".format(i) for i in range(num_classes)]


def collate_variable_size(batch):
    """Collate that tolerates differing H/W by padding to the batch maximum.

    Needed because evaluation commonly runs at native resolution (only
    `Normalize` in `val_dataset.transforms`), and a TEM dataset can legitimately
    mix image sizes. Padded regions are labelled `IGNORE_INDEX` so they cannot
    affect the metrics.
    """
    images = [item[0] for item in batch]
    seconds = [item[1] for item in batch]
    max_h = max(int(img.shape[1]) for img in images)
    max_w = max(int(img.shape[2]) for img in images)

    padded_images = []
    for img in images:
        pad_h, pad_w = max_h - int(img.shape[1]), max_w - int(img.shape[2])
        padded_images.append(
            torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0.0) if (pad_h or pad_w) else img
        )
    image_batch = torch.stack(padded_images, dim=0)

    if isinstance(seconds[0], torch.Tensor):
        padded_labels = []
        for label in seconds:
            pad_h, pad_w = max_h - int(label.shape[0]), max_w - int(label.shape[1])
            padded_labels.append(
                torch.nn.functional.pad(label, (0, pad_w, 0, pad_h), value=T.IGNORE_INDEX)
                if (pad_h or pad_w)
                else label
            )
        return image_batch, torch.stack(padded_labels, dim=0)
    return image_batch, list(seconds)
