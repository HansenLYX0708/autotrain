"""Pipeline test: InscribedRotation inside a real Ultralytics OBB train dataset.

Copies a few images/labels from the real slider dataset into a temp dataset,
builds YOLODataset(augment=True), injects the transform, and verifies output
angles/corners against the core, survival across close_mosaic rebuilds, and
that a val dataset stays untouched.

    D:/pythonEnvs/torchEnv/Scripts/python.exe torchtrain/tests/test_inscribed_pipeline.py
"""
import math
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchtrain.inscribed_rotation import (  # noqa: E402
    InscribedRotation, apply_matrix, inscribed_matrix, install_inscribed_rotation, rotate_image_inscribed,
)

SRC = "H:/Autotrain_user_database/Hansen Liu/localization/slider-gen5-corner-obb-d4f84092/obb"
DST = "D:/_work/projects/autoTraining/Autotrain_g/temp/inscribed-pipeline/obb"
N_IMAGES = 4


def build_dataset(images):
    import yaml
    for split in ("train", "val"):
        os.makedirs(os.path.join(DST, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DST, "labels", split), exist_ok=True)
    for name in images:
        for split in ("train", "val"):
            shutil.copy2(os.path.join(SRC, "images", "train", name),
                         os.path.join(DST, "images", split, name))
            shutil.copy2(os.path.join(SRC, "labels", "train", name.replace(".jpg", ".txt")),
                         os.path.join(DST, "labels", split, name.replace(".jpg", ".txt")))
    data = {"path": DST.replace(os.sep, "/"), "train": "images/train", "val": "images/val",
            "names": {0: "slider_abs"}}
    with open(os.path.join(DST, "data.yaml"), "w") as f:
        yaml.safe_dump(data, f)
    return data


def make_hyp():
    from ultralytics.utils import IterableSimpleNamespace
    keys = ("mosaic copy_paste copy_paste_mode mixup cutmix degrees translate scale shear "
            "perspective hsv_h hsv_s hsv_v flipud fliplr bgr mask_ratio overlap_mask erasing "
            "fraction rect single_cls cache imgsz task classes close_mosaic").split()
    hyp = IterableSimpleNamespace(**{k: 0.0 for k in keys})
    hyp.imgsz = 1440
    hyp.fraction = 1.0
    hyp.task = "obb"
    hyp.copy_paste_mode = "flip"
    return hyp


def segments_angle_stats(segs):
    return [math.degrees(math.atan2((s[1] - s[0])[1], (s[1] - s[0])[0])) for s in segs]


def test_pipeline():
    from ultralytics.data.build import build_yolo_dataset
    import cv2

    hyp = make_hyp()
    images = sorted(os.listdir(os.path.join(SRC, "images", "train")))[:N_IMAGES]
    data = build_dataset(images)
    train = build_yolo_dataset(hyp, os.path.join(DST, "images", "train"), 4, data, mode="train")
    val = build_yolo_dataset(hyp, os.path.join(DST, "images", "val"), 4, data, mode="val")

    import random as _random
    _random.seed(123)
    transform = install_inscribed_rotation(train, degrees=10, p=1.0)
    assert not hasattr(val, "_autotrain_inscribed")

    # Capture the angle/segments the transform actually uses (other transforms
    # may consume random draws, so reproducing the sequence is fragile).
    # get_image_and_label() runs the transform chain, so read the raw label
    # store instead — it holds the pre-transform segments the transform sees.
    label0 = train.labels[0]
    h, w = cv2.imread(os.path.join(DST, "images", "train", images[0])).shape[:2]
    src_segs = np.asarray(label0["segments"]).copy()
    src_was_norm = bool(label0.get("normalized", True))
    captured = {}
    import torchtrain.inscribed_rotation as mod
    orig = mod.rotate_image_inscribed
    def spy(*a, **k):
        result = orig(*a, **k)
        captured["angle"] = a[2]
        captured["keep"] = result[2]
        captured["segs"] = result[1]
        return result
    mod.rotate_image_inscribed = spy
    try:
        out = train[0]
    finally:
        mod.rotate_image_inscribed = orig
    angle = captured["angle"]
    keep_mask = captured["keep"]
    expected_segs = captured["segs"]
    bboxes = out["bboxes"]  # (N,5) normalized xywhr, letterboxed to imgsz
    assert bboxes.shape[0] == expected_segs.shape[0] == int(keep_mask.sum()), (bboxes.shape, keep_mask)
    # LetterBox (uniform scale + pad) preserves rect angles; compare fitted angle mod 90.
    s = min(1440 / w, 1440 / h)
    pad = np.array([(1440 - w * s) / 2, (1440 - h * s) / 2])
    for got, exp in zip(bboxes.numpy(), expected_segs):
        exp_angle = cv2.minAreaRect((exp * s + pad).astype(np.float32))[2]
        got_angle = math.degrees(float(got[4]))
        delta = min(abs(((got_angle - exp_angle - k) + 90) % 180 - 90) for k in (0, 90))
        assert delta < 1.0, (got_angle, exp_angle)
    # Independent check: each kept box angle equals its source rect angle
    # transformed by the same matrix applied to the image (M's point map is the
    # inverse of the getRotationMatrix2D angle convention, so use apply_matrix).
    src_px = src_segs * np.array([w, h]) if src_was_norm else src_segs
    M = mod.inscribed_matrix(w, h, angle)
    kept_src = [s for s, ok in zip(src_px, keep_mask) if ok]
    for seg, exp in zip(kept_src, expected_segs):
        if seg.shape[0] > 4:
            seg = cv2.boxPoints(cv2.minAreaRect(seg.astype(np.float32)))
        moved = mod.apply_matrix(seg.reshape(-1, 2), M).reshape(-1, 4, 2)
        src_angle = cv2.minAreaRect(moved.astype(np.float32))[2]
        exp_angle = cv2.minAreaRect(exp.astype(np.float32))[2]
        delta = min(abs(((src_angle - exp_angle - k) + 90) % 180 - 90) for k in (0, 90))
        assert delta < 1.0, (src_angle, exp_angle)
    assert transform.stats["images"] == 1

    # close_mosaic rebuild must keep the transform.
    hyp2 = make_hyp()
    train.close_mosaic(hyp2)
    def count_inscribed(ds):
        from ultralytics.data.augment import Compose
        found = []
        def walk(c):
            if isinstance(c, Compose):
                for t in c.transforms:
                    if isinstance(t, InscribedRotation):
                        found.append(t)
                    walk(t)
        walk(ds.transforms)
        return found
    assert len(count_inscribed(train)) == 1
    out2 = train[1]
    assert transform.stats["images"] == 2
    assert len(count_inscribed(val)) == 0
    _ = val[0]
    print("transform stats:", transform.stats)


if __name__ == "__main__":
    test_pipeline()
    print("PASS test_pipeline")
