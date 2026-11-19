"""Unit tests for torchtrain.inscribed_rotation. Plain script: run directly.

    D:/pythonEnvs/torchEnv/Scripts/python.exe torchtrain/tests/test_inscribed_rotation.py
"""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchtrain.inscribed_rotation import (  # noqa: E402
    InscribedRotation, apply_matrix, inscribed_matrix, inscribed_scale,
    rotate_image_inscribed, shrink_rotated_box,
)


def box_corners(cx, cy, w, h, angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([(cx + x * w / 2 * c - y * h / 2 * s, cy + x * w / 2 * s + y * h / 2 * c)
                     for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])


def rect_of(corners):
    edge = corners[1] - corners[0]
    return np.hypot(*edge), math.degrees(math.atan2(edge[1], edge[0]))


def angle_of(corners):
    return math.degrees(math.atan2(*(corners[1] - corners[0])[::-1][:: -1]))


def test_scale():
    assert abs(inscribed_scale(100, 100, 0) - 1.0) < 1e-12
    k = inscribed_scale(100, 100, math.radians(45))
    assert abs(k - math.sqrt(2) / 2) < 1e-9
    # Rotated image's inscribed rect (kw,kh) must fit: corners inside rotated rect
    for deg in (1, 7, 13, 30, 44.9):
        a = math.radians(deg)
        w, h = 320, 240
        k = inscribed_scale(w, h, a)
        c, s = math.cos(a), math.sin(a)
        # corner (kw/2,kh/2) inside rotated frame: |x c + y s| <= w/2 and |-x s + y c| <= h/2
        x, y = k * w / 2, k * h / 2
        assert x * c + y * s <= w / 2 + 1e-9
        assert x * s + y * c <= h / 2 + 1e-9


def test_marker_pixel_maps_to_matrix():
    rng = np.random.default_rng(0)
    img = np.zeros((240, 320, 3), np.uint8)
    px, py = 137, 91
    img[py, px] = (255, 255, 255)
    a = math.radians(8)
    M = inscribed_matrix(320, 240, a)
    out = cv2.warpAffine(img, M, (320, 240), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    q = apply_matrix(np.array([[px, py]]), M)[0]
    iy, ix = np.unravel_index(np.argmax(out[..., 0]), out.shape[:2])
    assert abs(ix - q[0]) <= 1 and abs(iy - q[1]) <= 1, (q, ix, iy)


def brute_force_max(shrunk_target, quad, w, h, min_side=2.0):
    """Independent search: same-orientation rects on a fine (p,q,x,y) grid."""
    c = quad.mean(axis=0)
    u = (quad[1] - quad[0]) / np.hypot(*(quad[1] - quad[0]))
    v = (quad[3] - quad[0]) / np.hypot(*(quad[3] - quad[0]))
    a = np.hypot(*(quad[1] - quad[0])) / 2
    b = np.hypot(*(quad[3] - quad[0])) / 2
    best = 0.0
    for tx in np.linspace(0.02, 1.0, 60):
        for ty in np.linspace(0.02, 1.0, 60):
            x, y = a * tx, b * ty
            if 2 * x < min_side or 2 * y < min_side:
                continue
            for px in np.linspace(-(a - x), a - x, 11):
                for qy in np.linspace(-(b - y), b - y, 11):
                    pts = np.array([c + (px + sx * x) * u + (qy + sy * y) * v
                                    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
                    if (pts >= -1e-6).all() and (pts <= [w + 1e-6, h + 1e-6]).all():
                        best = max(best, 4 * x * y)
    return best


def test_shrink_vs_bruteforce():
    rng = np.random.default_rng(7)
    w, h = 320, 240
    cases = []
    # boxes cut by 1 and 2 edges, near 0/90, small and large rotation
    cases.append(box_corners(160, 120, 200, 40, 5))
    cases.append(box_corners(20, 30, 60, 30, 40))
    cases.append(box_corners(300, 220, 80, 40, -25))
    cases.append(box_corners(160, 120, 10, 90, 89))
    cases.append(box_corners(60, 200, 100, 60, 12))
    for i in range(12):
        cx, cy = rng.uniform(0, w, 2)
        cases.append(box_corners(cx, cy, rng.uniform(10, 200), rng.uniform(10, 120), rng.uniform(-90, 90)))
    for quad in cases:
        got = shrink_rotated_box(quad, w, h)
        src_area = abs(np.cross(quad[1] - quad[0], quad[3] - quad[0]))
        if got is None:
            assert brute_force_max(src_area, quad, w, h) < 4.0
            continue
        assert (got >= -1e-6).all() and (got <= [w + 1e-6, h + 1e-6]).all(), got
        opt = brute_force_max(src_area, quad, w, h)
        got_area = abs(np.cross(got[1] - got[0], got[3] - got[0]))
        assert got_area >= 0.99 * opt - 1e-6, (got_area, opt)
        # orientation preserved: first edge parallel to source first edge
        e1, e2 = got[1] - got[0], quad[1] - quad[0]
        cos = abs(e1 @ e2 / (np.hypot(*e1) * np.hypot(*e2)))
        assert cos > 1 - 1e-6


def test_end_to_end_angles():
    w, h = 320, 240
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    for angle_deg in (0.1, -0.1, 15, -15):
        segs = np.stack([box_corners(160, 120, 60, 20, 30), box_corners(60, 60, 30, 30, -10)])
        out_img, out_segs, keep, stats = rotate_image_inscribed(img, segs, math.radians(angle_deg))
        assert out_img.shape == img.shape
        assert keep.all()
        M = inscribed_matrix(w, h, math.radians(angle_deg))
        expected = apply_matrix(segs.reshape(-1, 2), M).reshape(-1, 4, 2)
        for exp, dst in zip(expected, out_segs):
            a_exp = math.degrees(math.atan2((exp[1] - exp[0])[1], (exp[1] - exp[0])[0]))
            a_dst = math.degrees(math.atan2((dst[1] - dst[0])[1], (dst[1] - dst[0])[0]))
            delta = ((a_dst - a_exp) + 180) % 360 - 180
            assert abs(delta) < 1e-6, (a_exp, a_dst, angle_deg)
        assert (out_segs >= -1e-6).all() and (out_segs <= [w + 1e-6, h + 1e-6]).all()


def test_transform_p0_identity():
    class I:
        normalized = True
        segments = np.zeros((0, 4, 2))
        keypoints = None

        class _B:
            format = "xywh"

        _bboxes = _B()

        def denormalize(self, w, h):
            pass

    t = InscribedRotation(degrees=10, p=0.0)
    labels = {"img": np.zeros((10, 10, 3), np.uint8), "cls": np.zeros(0), "instances": I()}
    assert t(labels) is labels


def test_workers_draw_different_angles():
    """Two simulated workers reseeded differently must not share the sequence."""
    import random
    angles = []
    for seed in (11, 22):
        random.seed(seed)
        t = InscribedRotation(degrees=10, p=1.0)
        label = {"img": np.zeros((10, 10, 3), np.uint8), "cls": np.zeros(0),
                 "instances": _FakeInstances()}
        # patch: capture angle via monkey-replacing rotate_image_inscribed
        import torchtrain.inscribed_rotation as mod
        seen = []
        orig = mod.rotate_image_inscribed
        mod.rotate_image_inscribed = lambda *a, **k: (seen.append(a[2]), orig(*a, **k))[1]
        try:
            t(label)
        finally:
            mod.rotate_image_inscribed = orig
        angles.append(seen[0])
    assert angles[0] != angles[1], angles


class _FakeInstances:
    normalized = True
    segments = np.zeros((0, 4, 2))
    keypoints = None

    class _B:
        format = "xywh"

    _bboxes = _B()

    def denormalize(self, w, h):
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("ALL PASS")
