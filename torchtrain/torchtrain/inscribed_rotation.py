"""Inscribed rotation augmentation for rotated (OBB) boxes.

Rotates the image about its centre, then crops the largest centred axis-aligned
rectangle (same aspect ratio) that fits inside the rotated image and rescales
it to the original size — a single cv2.warpAffine, no padding, no gray corners.

Labels are rotated by the same affine. A rotated box that the crop would cut is
replaced by the largest-area rectangle that keeps the box's *exact* orientation,
is contained in the transformed box, and has all four corners inside the image.
Ultralytics' own geometric augmentation (RandomPerspective) cannot do this: it
clips segment points to the canvas and refits with cv2.minAreaRect, which gives
cut boxes a wrong angle. See docs/LOCALIZATION_WORKFLOW.md.

Pure numpy + cv2; no shapely. Shared by the online training transform
(install_inscribed_rotation below, injected in torchtrain/localization.py) and
the offline `rotate-augment` dataset tool (torchtrain/tools/localization.py).
"""
import math
import random

import cv2
import numpy as np

TOLERANCE = 1e-6
MIN_SIDE_PX = 2.0


def inscribed_scale(w, h, angle_rad):
    """Scale of the largest centred axis-aligned w:h rectangle inside the image rotated by angle."""
    c, s = abs(math.cos(angle_rad)), abs(math.sin(angle_rad))
    return min(1.0 / (c + (h / w) * s), 1.0 / ((w / h) * s + c))


def inscribed_matrix(w, h, angle_rad):
    """2x3 affine: rotate by angle about the centre, then zoom the inscribed crop back to w x h."""
    k = inscribed_scale(w, h, angle_rad)
    return cv2.getRotationMatrix2D((w / 2.0, h / 2.0), math.degrees(angle_rad), 1.0 / k)


def apply_matrix(points, M):
    """(N,2) points through a 2x3 affine matrix."""
    pts = np.asarray(points, dtype=np.float64)
    return pts[:, :2] @ M[:, :2].T + M[:, 2]


def _clip_halfplane(poly, a, b, bound):
    """Keep the part of a convex polygon where a*x + b*y <= bound (Sutherland-Hodgman)."""
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        fp, fq = a * p[0] + b * p[1] - bound, a * q[0] + b * q[1] - bound
        if fp <= 0:
            out.append(p)
        if (fp <= 0) != (fq <= 0):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _center_feasible(half_p, half_q, edges):
    """Is there a center offset (p,q) inside [-half_p,half_p]x[-half_q,half_q]
    satisfying every edge constraint alpha*p + beta*q <= bound?"""
    poly = [(-half_p, -half_q), (half_p, -half_q), (half_p, half_q), (-half_p, half_q)]
    for alpha, beta, bound in edges:
        poly = _clip_halfplane(poly, alpha, beta, bound)
        if not poly:
            return False
    return True


def _center_polygon(half_p, half_q, edges):
    poly = [(-half_p, -half_q), (half_p, -half_q), (half_p, half_q), (-half_p, half_q)]
    for alpha, beta, bound in edges:
        poly = _clip_halfplane(poly, alpha, beta, bound)
        if not poly:
            return None
    return poly


def shrink_rotated_box(corners, w, h, min_side=MIN_SIDE_PX, margin=TOLERANCE):
    """Largest same-orientation rectangle inside `corners` with all corners in the image.

    `corners` is a (4,2) cyclic rectangle (already transformed). The returned
    rectangle shares the box's center axes and orientation; it is shortened
    along its own axes, never re-fit. Returns the 4 corners in the same cyclic
    order, or None when nothing of at least min_side fits.

    Solve in box-local coordinates: a candidate rectangle has centre offset
    (p, q) and half-extents (x, y). "Corner inside image" is one linear
    constraint per image edge, alpha*p + beta*q + |alpha|*x + |beta|*y <= d.
    For fixed (x, y) the feasible (p, q) region is a convex polygon; the
    feasible (x, y) region is convex, so x * ymax(x) is a concave 1-D function
    and a golden-section search finds the maximum.
    """
    c = corners.mean(axis=0)
    edge_u = corners[1] - corners[0]
    edge_v = corners[3] - corners[0]
    a, b = np.hypot(*edge_u) / 2.0, np.hypot(*edge_v) / 2.0
    if a <= 0 or b <= 0:
        return None
    u, v = edge_u / (2 * a), edge_v / (2 * b)
    # Image edges in local coordinates: constraint on corner = c + lam*u + mu*v.
    edges = [
        (-u[0], -v[0], c[0] - margin),          # world x >= 0
        (u[0], v[0], (w - margin) - c[0]),      # world x <= w
        (-u[1], -v[1], c[1] - margin),          # world y >= 0
        (u[1], v[1], (h - margin) - c[1]),      # world y <= h
    ]

    def constraints(x, y):
        return [(alpha, beta, d - abs(alpha) * x - abs(beta) * y) for alpha, beta, d in edges]

    def feasible(x, y):
        if x <= 0 or y <= 0 or x > a + 1e-9 or y > b + 1e-9:
            return False
        return _center_feasible(a - x, b - y, constraints(x, y))

    def y_max(x):
        lo, hi = 0.0, b
        if not feasible(x, 1e-12):
            return 0.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if feasible(x, mid):
                lo = mid
            else:
                hi = mid
        return lo

    # x * y_max(x) is concave in x on (0, a]; golden section finds the peak.
    gr = (math.sqrt(5) - 1) / 2
    lo, hi = 0.0, a
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = x1 * y_max(x1), x2 * y_max(x2)
    for _ in range(200):
        if hi - lo < 1e-9:
            break
        if f1 < f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = x2 * y_max(x2)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = x1 * y_max(x1)
    x = (lo + hi) / 2
    y = y_max(x)
    if 2 * x < min_side or 2 * y < min_side:
        return None
    poly = _center_polygon(a - x, b - y, constraints(x, y))
    if not poly:
        return None
    p = sum(pt[0] for pt in poly) / len(poly)
    q = sum(pt[1] for pt in poly) / len(poly)
    out = np.array(
        [c + (p + sx * x) * u + (q + sy * y) * v for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]],
        dtype=np.float64,
    )
    return np.clip(out, 0.0, [w - 0.0, h - 0.0])


def rotate_image_inscribed(image, segments, angle_rad):
    """Rotate `image` and (N,4,2) pixel `segments` by `angle_rad` with inscribed crop.

    Returns (out_image, out_segments, keep_mask, stats). Dropped boxes are
    removed from out_segments; stats has kept/shrunk/dropped counts.
    """
    h, w = image.shape[:2]
    M = inscribed_matrix(w, h, angle_rad)
    out_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    segments = np.asarray(segments, dtype=np.float64)
    stats = {"kept": 0, "shrunk": 0, "dropped": 0}
    if segments.size == 0:
        return out_image, segments.reshape(0, 4, 2), np.zeros(0, dtype=bool), stats
    moved = apply_matrix(segments.reshape(-1, 2), M).reshape(-1, 4, 2)
    inside = (moved[..., 0] >= -TOLERANCE) & (moved[..., 0] <= w + TOLERANCE) & \
             (moved[..., 1] >= -TOLERANCE) & (moved[..., 1] <= h + TOLERANCE)
    new_segments, keep = [], []
    for i, quad in enumerate(moved):
        if inside[i].all():
            new_segments.append(quad)
            keep.append(True)
            stats["kept"] += 1
            continue
        shrunk = shrink_rotated_box(quad, w, h)
        if shrunk is None:
            keep.append(False)
            stats["dropped"] += 1
        else:
            new_segments.append(shrunk)
            keep.append(True)
            stats["shrunk"] += 1
    out = np.array(new_segments, dtype=np.float64).reshape(-1, 4, 2) if new_segments else np.zeros((0, 4, 2))
    return out_image, out, np.array(keep, dtype=bool), stats


def sample_angle(degrees, rng):
    return math.radians(float(rng.uniform(-degrees, degrees)))


class InscribedRotation:
    """Picklable Ultralytics train transform (top-level class for spawn workers).

    Runs before RandomPerspective inside the train dataset's pre_transform, on
    the real image before LetterBox, so `labels['instances'].segments` holds the
    4 OBB corners (normalized). The transform denormalizes, applies the shared
    core, then normalizes back so downstream transforms see a consistent state.

    Uses the `random` module: Ultralytics' worker seeding reseeds it per
    worker, so each worker draws an independent angle sequence (a private RNG
    would be identical in every worker after pickling).
    """

    def __init__(self, degrees, p=0.5):
        self.degrees = float(degrees)
        self.p = float(p)
        self.stats = {"images": 0, "rotated": 0, "kept": 0, "shrunk": 0, "dropped": 0}

    def __call__(self, labels):
        if self.p <= 0 or self.degrees <= 0 or random.random() >= self.p:
            return labels
        from ultralytics.utils.instance import Instances

        img = labels["img"]
        h, w = img.shape[:2]
        angle = math.radians(random.uniform(-self.degrees, self.degrees))
        instances = labels["instances"]
        was_normalized = instances.normalized
        if was_normalized:
            instances.denormalize(w, h)
        bbox_format = instances._bboxes.format
        segments = np.asarray(instances.segments, dtype=np.float64)
        if segments.ndim == 3 and segments.shape[1] > 4:
            # Ultralytics resamples OBB segments to ~100 boundary points. They
            # are unclipped here, so minAreaRect recovers the exact rectangle;
            # reduce to its 4 corners before transforming.
            segments = np.stack([
                cv2.boxPoints(cv2.minAreaRect(seg.astype(np.float32))).astype(np.float64)
                for seg in segments
            ]) if len(segments) else segments.reshape(0, 4, 2)
        out_img, new_segments, keep, stats = rotate_image_inscribed(img, segments, angle)
        mins = new_segments.min(axis=1) if len(new_segments) else np.zeros((0, 2))
        maxs = new_segments.max(axis=1) if len(new_segments) else np.zeros((0, 2))
        xyxy = np.concatenate([mins, maxs], axis=1) if len(new_segments) else np.zeros((0, 4))
        new_instances = Instances(
            bboxes=xyxy, segments=new_segments.astype(np.float32),
            keypoints=instances.keypoints[keep] if instances.keypoints is not None else None,
            bbox_format="xyxy", normalized=False,
        )
        new_instances.convert_bbox(format=bbox_format)
        if was_normalized:
            new_instances.normalize(w, h)
        labels["instances"] = new_instances
        labels["cls"] = np.asarray(labels["cls"])[keep]
        labels["img"] = out_img
        labels["resized_shape"] = out_img.shape[:2]
        for key, value in stats.items():
            self.stats[key] += value
        self.stats["images"] += 1
        self.stats["rotated"] += 1
        return labels

    def __repr__(self):
        return "InscribedRotation(degrees={}, p={})".format(self.degrees, self.p)


def _compose_with_perspective(compose):
    """The nested Compose holding a RandomPerspective (v8 pre_transform), or None."""
    from ultralytics.data.augment import Compose, RandomPerspective
    for item in compose.transforms:
        if isinstance(item, RandomPerspective):
            return compose
        if isinstance(item, Compose):
            found = _compose_with_perspective(item)
            if found is not None:
                return found
    return None


def inject_into_compose(outer, transform):
    """Insert `transform` immediately before RandomPerspective inside a Compose."""
    from ultralytics.data.augment import Compose
    if not isinstance(outer, Compose):
        raise ValueError("Train dataset has no transform Compose to inject into")
    inner = _compose_with_perspective(outer)
    if inner is None:
        raise ValueError("Train pipeline has no RandomPerspective; inscribed rotation has no anchor")
    if not any(isinstance(t, InscribedRotation) for t in inner.transforms):
        index = next(i for i, t in enumerate(inner.transforms) if t.__class__.__name__ == "RandomPerspective")
        inner.transforms.insert(index, transform)


def inject_transform(dataset, transform):
    """Insert `transform` immediately before RandomPerspective in dataset.transforms."""
    inject_into_compose(getattr(dataset, "transforms", None), transform)


def install_inscribed_rotation(dataset, degrees, p):
    """Enable inscribed rotation on a YOLO train dataset, surviving close_mosaic.

    YOLODataset.build_transforms is monkey-patched once at class level (plain
    top-level function, so the dataset still pickles for spawn workers); each
    rebuild re-injects the transform stored on the instance as
    `_autotrain_inscribed`. The val dataset never gets that attribute.
    """
    from ultralytics.data.dataset import YOLODataset

    if getattr(YOLODataset.build_transforms, "__name__", "") != "_autotrain_build_transforms":
        original = YOLODataset.build_transforms

        def _autotrain_build_transforms(self, hyp=None):
            transforms = original(self, hyp)
            transform = getattr(self, "_autotrain_inscribed", None)
            if transform is not None and getattr(self, "augment", False):
                inject_into_compose(transforms, transform)
            return transforms

        YOLODataset.build_transforms = _autotrain_build_transforms
    transform = InscribedRotation(degrees=degrees, p=p)
    dataset._autotrain_inscribed = transform
    inject_transform(dataset, transform)
    return transform
