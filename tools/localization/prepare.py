"""Prepare reviewed AutoTrain localization projects for YOLO OBB/Pose.

Uses Pillow only; does not install frameworks, download weights or start training.
Output is created atomically and must not already exist.
"""
import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def corners(obj):
    angle = math.radians(obj["angle"])
    c, s = math.cos(angle), math.sin(angle)
    return [
        (obj["cx"] + x * obj["width"] / 2 * c - y * obj["height"] / 2 * s,
         obj["cy"] + x * obj["width"] / 2 * s + y * obj["height"] / 2 * c)
        for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    ]


def in_image(x, y, image):
    return -1e-6 <= x <= image["width"] + 1e-6 and -1e-6 <= y <= image["height"] + 1e-6


def read_project(filename):
    with Path(filename).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def validate(project, pose=True):
    """Independent CLI validation: exported JSON is not assumed to be trusted."""
    errors = []
    if not isinstance(project, dict) or project.get("format") != "autotrain-localization" or project.get("version") != 1:
        raise ValueError("Expected an autotrain-localization version 1 project")
    for field in ("labels", "keypointNames"):
        values = project.get(field)
        if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() == v and v for v in values):
            raise ValueError("{} must be an array of non-empty names".format(field))
        if len(values) != len(set(values)):
            errors.append("{} contains duplicate names".format(field))
    labels, names = project["labels"], project["keypointNames"]
    if not labels or (pose and not names):
        errors.append("Class names and (for Pose) keypoint names must not be empty")
    images = project.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Project has no images")
    seen, groups = set(), {}
    counts = {split: Counter() for split in ("train", "val", "test")}
    for image in images:
        name = image.get("name", "")
        if (not isinstance(name, str) or not name or "\\" in name or ":" in name
                or any(ord(c) < 32 for c in name)
                or any(p in ("", ".", "..") for p in name.split("/"))):
            raise ValueError("Unsafe relative image path: {!r}".format(name))
        if name.lower() in seen:
            errors.append("Duplicate image: " + name)
        seen.add(name.lower())
        if not all(isinstance(image.get(k), int) and not isinstance(image[k], bool) and image[k] > 0 for k in ("width", "height")):
            raise ValueError("{}: invalid image dimensions".format(name))
        split, group = image.get("split"), image.get("group")
        if split not in counts:
            errors.append("{}: assign train/val/test".format(name))
        if not isinstance(group, str) or not group.strip():
            errors.append("{}: missing capture group".format(name))
        elif group in groups and groups[group] != split:
            errors.append("Capture group {} crosses splits".format(group))
        else:
            groups[group] = split
        if image.get("reviewed") is not True:
            errors.append("{}: annotation/negative image has not been reviewed".format(name))
        objects = image.get("objects")
        if not isinstance(objects, list):
            raise ValueError("{}: objects must be an array".format(name))
        ids = set()
        for obj in objects:
            prefix = "{} / {}".format(name, obj.get("id", "?"))
            if not isinstance(obj.get("id"), str) or not obj["id"] or obj["id"] in ids:
                errors.append(prefix + ": missing/duplicate instance ID")
            ids.add(obj.get("id"))
            if obj.get("label") not in labels:
                errors.append(prefix + ": unknown class")
            elif split in counts:
                counts[split][obj["label"]] += 1
            if (not all(number(obj.get(k)) for k in ("cx", "cy", "width", "height", "angle"))
                    or obj["width"] < 1 or obj["height"] < 1 or not -180 <= obj["angle"] <= 180):
                raise ValueError(prefix + ": invalid rectangle geometry")
            if obj.get("kind") not in ("obb", "rectangle") or (obj["kind"] == "rectangle" and obj["angle"] != 0):
                errors.append(prefix + ": invalid rectangle kind/angle")
            if not all(in_image(x, y, image) for x, y in corners(obj)):
                errors.append(prefix + ": box lies outside image")
            keypoints = obj.get("keypoints")
            if not isinstance(keypoints, dict):
                raise ValueError(prefix + ": keypoints must be an object")
            if set(keypoints) - set(names):
                errors.append(prefix + ": unknown keypoint names")
            for key in names:
                point = keypoints.get(key)
                if point is None:
                    if pose:
                        errors.append(prefix + ": unprocessed keypoint " + key)
                    continue
                if (not isinstance(point, dict) or not number(point.get("x")) or not number(point.get("y"))
                        or not number(point.get("visibility")) or point["visibility"] not in (0, 1, 2)):
                    raise ValueError(prefix + ": invalid keypoint " + key)
                if point["visibility"] and not in_image(point["x"], point["y"], image):
                    errors.append(prefix + ": keypoint outside image: " + key)
            if pose and not any(p["visibility"] for p in keypoints.values()):
                errors.append(prefix + ": no supervised keypoints")
    for split in ("train", "val"):
        for label in labels:
            if not counts[split][label]:
                errors.append("{} has no instances of {}".format(split, label))
    if errors:
        raise ValueError("Data validation failed:\n" + "\n".join(errors[:100]))
    return counts


def label_line(obj, image, labels, keypoint_names, task):
    """YOLO uses normalized coordinates; OBB corners retain cyclic ordering."""
    polygon = corners(obj)
    w, h = image["width"], image["height"]
    # Clamp only floating-point epsilon after the full geometric validation.
    norm = lambda v, size: min(1.0, max(0.0, v / size))
    if task == "obb":
        values = [v for x, y in polygon for v in (norm(x, w), norm(y, h))]
    else:
        xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        values = [norm((x1 + x2) / 2, w), norm((y1 + y2) / 2, h), norm(x2 - x1, w), norm(y2 - y1, h)]
        for name in keypoint_names:
            point = obj["keypoints"][name]
            values.extend([norm(point["x"], w), norm(point["y"], h), point["visibility"]] if point["visibility"] else [0, 0, 0])
    return str(labels.index(obj["label"])) + " " + " ".join("{:.9g}".format(v) for v in values)


def json_write(filename, value):
    Path(filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(project_file, images_root, output, task="both"):
    from PIL import Image

    project = read_project(project_file)
    validate(project, pose=task != "obb")
    root, target = Path(images_root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise ValueError("Images root is not a directory")
    if target.exists():
        raise ValueError("Output already exists; choose a new versioned output directory")
    if not target.parent.is_dir():
        raise ValueError("Output parent directory must already exist")
    tasks = ["obb", "pose"] if task == "both" else [task]
    prepared, hashes = [], {}
    # Finish all validation before creating any output directory.
    for index, image in enumerate(project["images"]):
        source = (root / PurePosixPath(image["name"])).resolve()
        try:
            source.relative_to(root)
        except ValueError:
            raise ValueError("Image escapes root: " + image["name"])
        if source.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp") or not source.is_file():
            raise ValueError("Missing/unsupported image: " + str(source))
        with Image.open(source) as checked:
            checked.verify()
        with Image.open(source) as opened:
            if opened.getexif().get(274, 1) != 1:
                raise ValueError("{}: normalize EXIF orientation before annotating".format(image["name"]))
            if opened.size != (image["width"], image["height"]):
                raise ValueError("{}: dimensions differ from annotation".format(image["name"]))
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        sha = digest.hexdigest()
        if sha in hashes and hashes[sha] != image["split"]:
            raise ValueError("Identical image content crosses splits: " + image["name"])
        hashes[sha] = image["split"]
        filename = "{:06d}_{}{}".format(index, sha[:12], source.suffix.lower())
        prepared.append((image, source, filename, sha))
    stage = Path(tempfile.mkdtemp(prefix=".localization-", dir=str(target.parent)))
    try:
        for task_name in tasks:
            base = stage / task_name
            for split in ("train", "val", "test"):
                (base / "images" / split).mkdir(parents=True)
                (base / "labels" / split).mkdir(parents=True)
            config = {
                "path": (target / task_name).as_posix(),
                "train": "images/train", "val": "images/val",
                "names": dict(enumerate(project["labels"])),
            }
            if any(i["split"] == "test" for i in project["images"]):
                config["test"] = "images/test"
            if task_name == "pose":
                config["kpt_shape"] = [len(project["keypointNames"]), 3]
                # No flip_idx: unknown symmetry must not silently swap semantic landmarks.
            # JSON syntax is also valid YAML; no additional YAML dependency is needed.
            json_write(base / "data.yaml", config)
            json_write(base / "schema.json", {
                "task": task_name, "labels": project["labels"],
                "keypointNames": project["keypointNames"] if task_name == "pose" else [],
                "coordinateSystem": "original-image pixels; x right, y down",
            })
            for image, source, filename, _ in prepared:
                shutil.copy2(str(source), str(base / "images" / image["split"] / filename))
                lines = [label_line(o, image, project["labels"], project["keypointNames"], task_name) for o in image["objects"]]
                (base / "labels" / image["split"] / (Path(filename).stem + ".txt")).write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
        json_write(stage / "project.json", project)
        json_write(stage / "manifest.json", [
            {"source": image["name"], "file": filename, "split": image["split"], "group": image["group"], "sha256": sha}
            for image, _, filename, sha in prepared
        ])
        # Destination must remain absent; never merge with a previous dataset.
        if target.exists():
            raise ValueError("Output was created by another process")
        stage.rename(target)
    except Exception:
        shutil.rmtree(stage)
        raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--images", required=True, help="Original image root selected in the browser")
    parser.add_argument("--output", required=True, help="New directory; parent must exist")
    parser.add_argument("--task", choices=["obb", "pose", "both"], default="both")
    args = parser.parse_args()
    try:
        print("Prepared:", prepare(args.project, args.images, args.output, args.task))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
