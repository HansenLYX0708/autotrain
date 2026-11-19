"""Offline inscribed-rotation augmentation for managed localization datasets.

Only the train split is augmented: each train image gets N rotated copies named
``<stem>__rot<copy index>_<signed angle, 2 decimals><ext>`` that inherit the source image's
group and split. A project.json snapshot in the annotation schema is written
with the rotated (and cropped) boxes, then the existing ``prepare`` path
produces the YOLO task folder — the result is indistinguishable from an import.
"""
import json
import math
import shutil
import tempfile
import uuid
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from .inscribed_rotation import rotate_image_inscribed
from .localization_data import corners, prepare, read_project, validate


def rect_from_corners(quad):
    """Inverse of localization_data.corners: (cx, cy, width, height, angle deg)."""
    edge_w = quad[1] - quad[0]
    edge_h = quad[3] - quad[0]
    return {
        "cx": float(quad[:, 0].mean()), "cy": float(quad[:, 1].mean()),
        "width": float(np.hypot(*edge_w)), "height": float(np.hypot(*edge_h)),
        "angle": float(math.degrees(math.atan2(edge_w[1], edge_w[0]))),
    }


def rotate_augment(source, save_dir, task, degrees, copies, seed):
    source = Path(source).resolve()
    target = Path(save_dir).resolve()
    task_dir = source / task
    if task != "obb":
        raise ValueError("rotate-augment currently supports --task obb only")
    if not (0 < degrees <= 45):
        raise ValueError("--degrees must be in (0, 45]")
    if not (1 <= copies <= 20):
        raise ValueError("--copies must be in [1, 20]")
    if not (source / "project.json").is_file() or not (source / "manifest.json").is_file():
        raise ValueError("Source must be a managed localization dataset root (project.json + manifest.json)")
    schema_file = task_dir / "schema.json"
    if not schema_file.is_file():
        raise ValueError("Missing {}".format(schema_file))
    schema = json.loads(schema_file.read_text(encoding="utf-8-sig"))
    if schema.get("task") != task:
        raise ValueError("Source schema task {} disagrees with --task {}".format(schema.get("task"), task))

    project = read_project(source / "project.json")
    validate(project, pose=False)
    manifest = {m["source"]: m for m in json.loads((source / "manifest.json").read_text(encoding="utf-8-sig"))}
    rng = np.random.default_rng(int(seed))

    images_root = Path(tempfile.mkdtemp(prefix=".rotate-images-", dir=str(target.parent)))
    stage = Path(tempfile.mkdtemp(prefix=".rotate-stage-", dir=str(target.parent)))
    report = {
        "source": str(source), "task": task, "degrees": degrees, "copies": copies, "seed": seed,
        "images": [], "totals": {"copies": 0, "skippedEmpty": 0, "kept": 0, "shrunk": 0, "dropped": 0},
    }
    try:
        new_images = []
        for image in project["images"]:
            entry = manifest.get(image["name"])
            if entry is None:
                raise ValueError("manifest.json has no entry for " + image["name"])
            source_file = task_dir / "images" / entry["split"] / entry["file"]
            if not source_file.is_file():
                raise ValueError("Missing image file " + str(source_file))
            dest = images_root / PurePosixPath(image["name"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source_file), str(dest))
            new_images.append(image)
            if image["split"] != "train" or not image["objects"]:
                continue
            original = cv2.imread(str(source_file))
            if original is None:
                raise ValueError("Cannot read " + str(source_file))
            segments = np.array([corners(obj) for obj in image["objects"]], dtype=np.float64)
            stem, ext = PurePosixPath(image["name"]).stem, PurePosixPath(image["name"]).suffix
            parent = str(PurePosixPath(image["name"]).parent)
            parent = "" if parent == "." else parent + "/"
            for index in range(copies):
                angle = float(rng.uniform(-degrees, degrees))
                out_img, out_segments, keep, stats = rotate_image_inscribed(original, segments, math.radians(angle))
                if not any(keep):
                    # Rotated copy has no labels left; skip it rather than
                    # shipping an unlabeled image in the augmented dataset.
                    report["totals"]["skippedEmpty"] += 1
                    report["images"].append({"name": None, "source": image["name"], "angle": angle,
                                             "objects": 0, "skipped": "empty", **stats})
                    continue
                # Copy index keeps names unique when two draws round to the same angle.
                new_name = "{}{}__rot{}_{:+.2f}{}".format(parent, stem, index, angle, ext)
                out_dest = images_root / PurePosixPath(new_name)
                out_dest.parent.mkdir(parents=True, exist_ok=True)
                params = [cv2.IMWRITE_JPEG_QUALITY, 100] if ext.lower() in (".jpg", ".jpeg") else []
                if not cv2.imwrite(str(out_dest), out_img, params):
                    raise ValueError("Cannot write " + str(out_dest))
                objects = []
                kept = iter(out_segments)
                for obj, ok in zip(image["objects"], keep):
                    if not ok:
                        continue
                    quad = next(kept)
                    rect = rect_from_corners(np.asarray(quad))
                    objects.append({
                        **obj, "id": "{}-r{}".format(obj["id"], index),
                        "kind": "obb" if abs(rect["angle"]) > 1e-9 else obj["kind"],
                        "cx": rect["cx"], "cy": rect["cy"],
                        "width": rect["width"], "height": rect["height"], "angle": rect["angle"],
                    })
                copy_record = dict(image)
                copy_record.update({"name": new_name, "objects": objects})
                new_images.append(copy_record)
                for key in ("kept", "shrunk", "dropped"):
                    report["totals"][key] += stats[key]
                report["totals"]["copies"] += 1
                report["images"].append({"name": new_name, "source": image["name"], "angle": angle,
                                         "objects": len(objects), **stats})
        if not report["totals"]["copies"]:
            raise ValueError("No train images to rotate")
        snapshot = dict(project)
        snapshot["id"] = str(uuid.uuid4())
        snapshot["images"] = new_images
        project_file = stage / "project.json"
        project_file.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        prepare(str(project_file), str(images_root), str(target), task)
        (target / "augment-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return target
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(images_root, ignore_errors=True)
        shutil.rmtree(stage, ignore_errors=True)
