"""Read-only COCO image/defect audit. Python >=3.8; no pretrained weights."""
import argparse
import collections
import concurrent.futures
import datetime
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

VERSION = 1
# Two 79-dimensional descriptors, eight geometric and eight photometric features.
BLOCKS = [(0, 79, .45), (79, 158, .25), (158, 166, .15), (166, 174, .15)]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def descriptor(patch):
    """Fixed-size *local* descriptor; physical scale is retained separately."""
    f = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    z = (f - f.mean()) / max(float(f.std()), .02)
    dct = cv2.dct(z)[:4, :4].ravel()[1:] / 64
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy)
    bins = np.minimum((angle * (8 / (2 * np.pi))).astype(int), 7)
    hog = []
    for yy in (0, 32):
        for xx in (0, 32):
            h = np.bincount(bins[yy:yy+32, xx:xx+32].ravel(),
                            weights=mag[yy:yy+32, xx:xx+32].ravel(), minlength=8)
            hog.extend(h / max(float(h.sum()), 1e-8))
    center = f[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.uint8)
    for bit, (dy, dx) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, 1),
                                    (1, 1), (1, 0), (1, -1), (0, -1)]):
        lbp |= ((f[1+dy:63+dy, 1+dx:63+dx] >= center).astype(np.uint8) << bit)
    hist_lbp = np.bincount((lbp // 16).ravel(), minlength=16) / lbp.size
    hist_gray = np.histogram(f, bins=16, range=(0, 1))[0] / f.size
    return np.concatenate([dct, hog, hist_lbp, hist_gray])


def local_features(gray, bbox):
    x, y, w, h = bbox
    H, W = gray.shape
    left, top = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
    right, bottom = min(W, int(math.ceil(x+w))), min(H, int(math.ceil(y+h)))
    patch = gray[top:bottom, left:right]
    pad = max(12, int(math.ceil(.5 * max(w, h))))
    l, t = max(0, left-pad), max(0, top-pad)
    r, b = min(W, right+pad), min(H, bottom+pad)
    context = gray[t:b, l:r]
    mask = np.ones(context.shape, bool)
    mask[top-t:bottom-t, left-l:right-l] = False
    ring = context[mask].astype(np.float32)
    if not ring.size:
        ring = context.ravel().astype(np.float32)
    pf = patch.astype(np.float32)
    p10, p50, p90 = np.percentile(pf, [10, 50, 90]) / 255
    contrast = float((pf.mean() - ring.mean()) / max(float(ring.std()), 5))
    geometry = [(x+w/2)/W, (y+h/2)/H, math.log(w/W), math.log(h/H),
                math.log(w/h), math.log(w*h/(W*H)), math.log(min(w, h)),
                min((x+w/2)/W, 1-(x+w/2)/W, (y+h/2)/H, 1-(y+h/2)/H)]
    photo = [float(pf.mean()/255), float(pf.std()/255), p10, p50, p90,
             float(ring.mean()/255), float(ring.std()/255), contrast]
    vector = np.concatenate([descriptor(patch), descriptor(context), geometry, photo])
    assert vector.shape == (174,) and np.isfinite(vector).all()
    return vector.astype(np.float32), {
        "cx": geometry[0], "cy": geometry[1], "area_fraction": w*h/(W*H),
        "short_px": min(w, h), "aspect": w/h, "contrast": contrast,
        "gray_mean": photo[0], "gray_std": photo[1],
    }


def inspect_image(task):
    image, annotations, path = task
    image = dict(image)
    problems, records, features = [], [], []
    image["issues"] = []
    image["readable"] = False
    p = Path(path)
    if not p.is_file():
        image["issues"].append("missing_image")
        return image, records, features, problems
    try:
        raw = p.read_bytes()
        image["bytes"] = len(raw)
        image["file_sha256"] = sha(raw)
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError("OpenCV could not decode image")
        image["pixel_sha256"] = sha(str((decoded.shape, str(decoded.dtype))).encode()
                                    + decoded.tobytes())
        if decoded.dtype != np.uint8:
            raise ValueError("Only 8-bit images supported by these descriptors")
        gray = decoded if decoded.ndim == 2 else cv2.cvtColor(
            decoded, cv2.COLOR_BGRA2GRAY if decoded.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        image.update(readable=True, actual_width=W, actual_height=H)
        if (image["width"], image["height"]) != (W, H):
            image["issues"].append("dimension_mismatch")
        for a in annotations:
            base = {"key": image["key"] + ":" + str(a["id"]),
                    "image_key": image["key"], "annotation_id": a["id"],
                    "category_id": a["category_id"], "split": image["split"],
                    "source_group": image["source_group"], "file_name": image["file_name"],
                    "bbox": a.get("bbox"), "issues": []}
            try:
                x, y, w, h = map(float, a["bbox"])
                if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
                    raise ValueError("Non-finite or nonpositive bbox")
                if x < -1e-6 or y < -1e-6 or x+w > W+1e-6 or y+h > H+1e-6:
                    base["issues"].append("bbox_outside_image")
                if x+w <= 0 or y+h <= 0 or x >= W or y >= H:
                    raise ValueError("No image intersection")
                vector, values = local_features(gray, (x, y, w, h))
                base.update(values)
                records.append(base)
                features.append(vector)
                if base["issues"]:
                    problems.append(dict(base, error="bbox_outside_image"))
            except (ValueError, TypeError, KeyError, cv2.error, OverflowError) as e:
                problems.append(dict(base, error="invalid_bbox", detail=str(e)))
        image["annotation_count"] = len(annotations)
    except (OSError, ValueError, cv2.error) as e:
        image["issues"].append("decode_or_format_error")
        image["error"] = str(e)
    return image, records, features, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", default="annotations3_20261102")
    parser.add_argument("--train", default="instance_train_step8.json")
    parser.add_argument("--val", default="instance_val_step6.json")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.cache.mkdir(parents=True, exist_ok=True)
    tasks, datasets, provenance, structural = [], {}, {}, []
    categories = None
    for split, name in [("train", args.train), ("val", args.val)]:
        p = args.root / args.annotations / name
        raw = p.read_bytes()
        d = json.loads(raw.decode("utf-8-sig"))
        datasets[split] = d
        provenance[split] = {"path": str(p.resolve()), "sha256": sha(raw),
                             "mtime": datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                             "images": len(d["images"]), "annotations": len(d["annotations"])}
        current_categories = {c["id"]: c["name"] for c in d["categories"]}
        if categories is not None and current_categories != categories:
            raise ValueError("Category ID/name mapping differs between splits")
        categories = current_categories
        for field in ("images", "annotations", "categories"):
            ids = [o["id"] for o in d[field]]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate %s IDs in %s; resolve before audit" % (field, split))
        image_ids = {i["id"] for i in d["images"]}
        by_image = collections.defaultdict(list)
        for a in d["annotations"]:
            if a["image_id"] not in image_ids or a["category_id"] not in categories:
                structural.append({"split": split, "annotation_id": a["id"],
                                   "error": "unknown_image_or_category"})
                continue
            by_image[a["image_id"]].append(a)
        for i in d["images"]:
            m = re.search(r"([LR]-\d+)", i["file_name"])
            identity = re.search(r"([LR]-\d+.*)", Path(i["file_name"]).stem)
            source = m.group(1) if m else "UNKNOWN:" + split + ":" + str(i["id"])
            image = dict(i, key=split + ":" + str(i["id"]), split=split,
                         source_group=source, canonical_object=identity.group(1).rstrip("-") if identity else "",
                         prefix=i["file_name"][:m.start()].rstrip("_") if m else "")
            path = (args.root / split / i["file_name"]).resolve()
            if args.root.resolve() not in path.parents:
                raise ValueError("Image path outside supplied dataset root")
            tasks.append((image, by_image[i["id"]], str(path)))
    print("Scanning %d images; %d workers; no input files are changed." %
          (len(tasks), args.workers), flush=True)
    images, records, features, problems = [], [], [], list(structural)
    start = time.monotonic()
    # map preserves input order, so feature rows and JSON records are reproducible.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (image, recs, vecs, errs) in enumerate(pool.map(inspect_image, tasks)):
            images.append(image)
            records.extend(recs)
            features.extend(vecs)
            problems.extend(errs)
            if (index+1) % 500 == 0 or index+1 == len(tasks):
                print("%d/%d images; %d defects; %.1fs" %
                      (index+1, len(tasks), len(records), time.monotonic()-start), flush=True)
    result = {"version": VERSION, "created_at": datetime.datetime.now().isoformat(),
              "root": str(args.root.resolve()), "provenance": provenance,
              "categories": categories, "images": images, "records": records,
              "problems": problems,
              "feature_blocks": BLOCKS,
              "feature_method": "local/context DCT15+HOG32+LBP16+gray_hist16; geometry8; photo8",
              "versions": {"python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__}}
    write_json(args.cache / "audit.json", result)
    np.savez_compressed(str(args.cache / "features.npz"), features=np.stack(features))
    print("Extraction complete: %s" % args.cache, flush=True)


if __name__ == "__main__":
    main()
