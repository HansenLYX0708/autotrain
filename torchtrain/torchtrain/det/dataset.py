"""
COCO detection dataset.

Reads the COCO JSON directly rather than through `pycocotools`, which keeps the
package installable on Windows/Python 3.8 without a compiler. The parsed
structure is also what the built-in evaluator consumes, so ground truth is read
exactly once.

Label convention (the one place Paddle and torchvision genuinely disagree):

    PaddleDetection `num_classes`  = foreground classes only
    torchvision     `num_classes`  = foreground + background

So a category id from the JSON maps to `clsid` in `[0, K)` (PaddleDetection's
`catid2clsid`) and to `clsid + 1` for torchvision, leaving 0 free for background.
`class_names` is stored in PaddleDetection order so predictions can be labelled
consistently with the rest of the platform.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from ..logger import log
from ..seg.dataset import read_image
from . import transforms as T


class CocoData:
    """Parsed COCO annotations, shared by the dataset and the evaluator."""

    def __init__(self, anno_path: str, image_dir: str) -> None:
        if not os.path.isfile(anno_path):
            raise FileNotFoundError("COCO annotation file not found: {}".format(anno_path))
        with open(anno_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)

        self.anno_path = anno_path
        self.image_dir = image_dir

        categories = sorted(raw.get("categories") or [], key=lambda c: int(c["id"]))
        if not categories:
            raise ValueError("COCO file has no `categories`: {}".format(anno_path))
        self.cat_ids: List[int] = [int(c["id"]) for c in categories]
        self.class_names: List[str] = [str(c.get("name", c["id"])) for c in categories]
        self.catid2clsid: Dict[int, int] = {cid: i for i, cid in enumerate(self.cat_ids)}

        self.images: List[Dict[str, Any]] = []
        by_id: Dict[int, Dict[str, Any]] = {}
        for item in raw.get("images") or []:
            record = {
                "id": int(item["id"]),
                "file_name": str(item.get("file_name", "")),
                "width": int(item.get("width", 0) or 0),
                "height": int(item.get("height", 0) or 0),
                "annotations": [],
            }
            by_id[record["id"]] = record
            self.images.append(record)

        dropped = 0
        for anno in raw.get("annotations") or []:
            image = by_id.get(int(anno.get("image_id", -1)))
            if image is None:
                dropped += 1
                continue
            bbox = anno.get("bbox")
            if not bbox or len(bbox) < 4:
                dropped += 1
                continue
            x, y, w, h = [float(v) for v in bbox[:4]]
            if w <= 0 or h <= 0:
                dropped += 1
                continue
            cat_id = int(anno.get("category_id", -1))
            if cat_id not in self.catid2clsid:
                dropped += 1
                continue
            image["annotations"].append(
                {
                    "id": int(anno.get("id", 0)),
                    "clsid": self.catid2clsid[cat_id],
                    "category_id": cat_id,
                    "bbox_xyxy": [x, y, x + w, y + h],
                    "area": float(anno.get("area", w * h) or (w * h)),
                    "iscrowd": int(anno.get("iscrowd", 0) or 0),
                }
            )
        if dropped:
            log("WARNING: skipped {} malformed/unmatched annotations in {}.".format(dropped, os.path.basename(anno_path)))

    @property
    def num_classes(self) -> int:
        """Foreground classes (PaddleDetection convention)."""
        return len(self.cat_ids)

    def image_path(self, record: Dict[str, Any]) -> str:
        return os.path.join(self.image_dir, record["file_name"])

    def with_annotations(self) -> List[Dict[str, Any]]:
        return [r for r in self.images if r["annotations"]]


def resolve_dataset_block(block: Dict[str, Any], key_hint: str) -> Tuple[str, str]:
    """Resolve `(anno_path, image_dir)` from a `TrainDataset:` style block."""
    if not isinstance(block, dict):
        raise ValueError("Expected a mapping for the {} config block".format(key_hint))
    dataset_dir = str(block.get("dataset_dir") or "")
    anno_path = block.get("anno_path")
    if not anno_path:
        raise ValueError("{} is missing `anno_path` (path to the COCO JSON).".format(key_hint))
    anno_path = str(anno_path)
    if not os.path.isabs(anno_path) and dataset_dir:
        anno_path = os.path.join(dataset_dir, anno_path)

    image_dir = str(block.get("image_dir") or "")
    if image_dir and not os.path.isabs(image_dir) and dataset_dir:
        image_dir = os.path.join(dataset_dir, image_dir)
    elif not image_dir:
        # `ImageFolder`-style blocks omit image_dir; images sit next to the JSON.
        image_dir = dataset_dir or os.path.dirname(anno_path)
    return anno_path, image_dir


class CocoDetectionDataset(TorchDataset):
    """Yields `(image_tensor, target)` in the format torchvision detectors expect.

    * image: float32 CHW in `[0, 1]` (the model's own transform normalises it)
    * target: `{boxes: [N,4] xyxy float32, labels: [N] int64, image_id, area, iscrowd}`
    """

    def __init__(
        self,
        data: CocoData,
        mode: str = "train",
        transforms: Optional[T.DetCompose] = None,
        allow_empty: bool = False,
    ) -> None:
        self.data = data
        self.mode = mode
        self.transforms = transforms or T.DetCompose([])
        # Training on images with no boxes teaches the model nothing and trips
        # older torchvision versions, so they are dropped unless asked for.
        self.records = data.images if (allow_empty or mode != "train") else data.with_annotations()
        if not self.records:
            raise ValueError(
                "No usable images in {}. Every image lacks annotations; set allow_empty: true to keep them.".format(
                    data.anno_path
                )
            )
        missing = [r for r in self.records[:50] if not os.path.isfile(data.image_path(r))]
        if missing:
            log(
                "WARNING: {} of the first {} images are missing on disk, e.g. {}. Check image_dir/dataset_dir.".format(
                    len(missing), min(50, len(self.records)), data.image_path(missing[0])
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = self.data.image_path(record)
        im = read_image(path)

        annos = record["annotations"]
        target: Dict[str, Any] = {
            "boxes": np.asarray([a["bbox_xyxy"] for a in annos], dtype=np.float32).reshape(-1, 4),
            "labels": np.asarray([a["clsid"] + 1 for a in annos], dtype=np.int64),
            "iscrowd": np.asarray([a["iscrowd"] for a in annos], dtype=np.int64),
        }
        if self.mode == "train":
            im, target = self.transforms(im, target)

        boxes = np.asarray(target["boxes"], dtype=np.float32).reshape(-1, 4)
        height, width = im.shape[:2]
        if len(boxes):
            boxes[:, 0::2] = boxes[:, 0::2].clip(0, width - 1)
            boxes[:, 1::2] = boxes[:, 1::2].clip(0, height - 1)
            keep = (boxes[:, 2] > boxes[:, 0] + 1e-3) & (boxes[:, 3] > boxes[:, 1] + 1e-3)
            boxes = boxes[keep]
            labels = np.asarray(target["labels"], dtype=np.int64)[keep]
            iscrowd = np.asarray(target["iscrowd"], dtype=np.int64)[keep]
        else:
            labels = np.zeros((0,), dtype=np.int64)
            iscrowd = np.zeros((0,), dtype=np.int64)

        tensor = torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))) / 255.0
        torch_target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "iscrowd": torch.from_numpy(iscrowd),
            "area": torch.from_numpy(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).astype(np.float32)),
            "image_id": torch.tensor([int(record["id"])]),
            # Kept so the evaluator can map predictions back to native pixels
            # even if a transform changed the resolution.
            "orig_size": torch.tensor([int(record["height"] or height), int(record["width"] or width)]),
            "path": path,
        }
        return tensor, torch_target


def collate_detection(batch):
    """Detection batches are lists, not stacked tensors (variable image sizes)."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def build_datasets(
    cfg: Dict[str, Any], mode: str
) -> Tuple[CocoDetectionDataset, CocoData]:
    """Build the train or eval dataset from a PaddleDetection-shaped config."""
    key = "TrainDataset" if mode == "train" else "EvalDataset"
    block = cfg.get(key) or cfg.get("EvalDataset") or cfg.get("TrainDataset")
    if not block:
        raise ValueError("Config has no `{}:` block.".format(key))
    anno_path, image_dir = resolve_dataset_block(block, key)
    data = CocoData(anno_path, image_dir)

    reader_key = "TrainReader" if mode == "train" else "EvalReader"
    reader = cfg.get(reader_key) or {}
    transforms = T.build_transforms(
        list(reader.get("sample_transforms") or []) + list(reader.get("batch_transforms") or [])
    )
    dataset = CocoDetectionDataset(
        data,
        mode=mode,
        transforms=transforms,
        allow_empty=bool(block.get("allow_empty", mode != "train")),
    )
    return dataset, data
