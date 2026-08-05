"""Guo-Hall thinning, spur pruning and ordered centerline polylines.

Why pruning is not optional: Guo-Hall (like every morphological thinning) leaves
short spurs wherever the region boundary bulges, and a spur tip is a degree-1
pixel just like a real end of the band. Taking "the rightmost skeleton pixel" on a
raw skeleton therefore picks up noise. We reduce the skeleton to its longest
geodesic path first, which is the actual centerline of a band-shaped region.

Known bias, deliberately kept: a thinning skeleton retracts from the tip of a
region by roughly half the local band width, so the reported endpoint sits
slightly inside the true tip. `extend_to_region()` implements the opt-in
correction (`--endpoint-extend region`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

# Neighbour offsets in OpenCV's thinningGuoHall naming: p2 = north, then clockwise.
_PAD = 2


def _guo_hall_numpy(mask: np.ndarray) -> np.ndarray:
    """Reference Guo-Hall thinning, used when cv2.ximgproc is unavailable.

    Mirrors OpenCV's `thinningGuoHallIteration` exactly (same condition table,
    same simultaneous deletion per sub-iteration), just vectorised over shifted
    neighbour planes instead of a pixel loop.
    """
    im = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for it in (0, 1):
            a = im
            p2 = a[:-2, 1:-1]
            p3 = a[:-2, 2:]
            p4 = a[1:-1, 2:]
            p5 = a[2:, 2:]
            p6 = a[2:, 1:-1]
            p7 = a[2:, :-2]
            p8 = a[1:-1, :-2]
            p9 = a[:-2, :-2]
            c = (
                ((1 - p2) & (p3 | p4))
                + ((1 - p4) & (p5 | p6))
                + ((1 - p6) & (p7 | p8))
                + ((1 - p8) & (p9 | p2))
            )
            n1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8)
            n2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9)
            n = np.minimum(n1, n2)
            m = ((p6 | p7 | (1 - p9)) & p8) if it == 0 else ((p2 | p3 | (1 - p5)) & p4)
            marker = (c == 1) & (n >= 2) & (n <= 3) & (m == 0) & (a[1:-1, 1:-1] > 0)
            if marker.any():
                im[1:-1, 1:-1] = np.where(marker, 0, im[1:-1, 1:-1])
                changed = True
        if not changed:
            break
    return im[1:-1, 1:-1].astype(bool)


def guo_hall(mask: np.ndarray, backend: str = "auto") -> np.ndarray:
    """Thin `mask` to a 1-pixel skeleton with the Guo-Hall algorithm.

    The mask is padded with background before thinning so that a region touching
    the image border is thinned like any other region. Without the pad, OpenCV
    never touches the outermost pixel ring and leaves a full-width stub there --
    and several layers in these TEM masks do run off the left/right edge.
    """
    mask = np.ascontiguousarray(mask.astype(bool))
    if not mask.any():
        return np.zeros_like(mask)
    padded = np.pad(mask, _PAD)
    out = None
    if backend in ("auto", "cv2"):
        try:
            import cv2

            out = cv2.ximgproc.thinning(
                (padded.astype(np.uint8) * 255),
                thinningType=cv2.ximgproc.THINNING_GUOHALL,
            ).astype(bool)
        except Exception:
            if backend == "cv2":
                raise
            out = None
    if out is None:
        out = _guo_hall_numpy(padded)
    return out[_PAD:-_PAD, _PAD:-_PAD]


# --------------------------------------------------------------------------- #
# Graph reduction: skeleton pixels -> one ordered polyline
# --------------------------------------------------------------------------- #

_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _build_graph(skel: np.ndarray):
    coords = np.argwhere(skel)  # (row, col)
    if len(coords) == 0:
        return coords, None
    index = -np.ones(skel.shape, dtype=np.int64)
    index[coords[:, 0], coords[:, 1]] = np.arange(len(coords))
    rows, cols, data = [], [], []
    h, w = skel.shape
    for dy, dx in _OFFSETS:
        ry = coords[:, 0] + dy
        rx = coords[:, 1] + dx
        ok = (ry >= 0) & (ry < h) & (rx >= 0) & (rx < w)
        src = np.nonzero(ok)[0]
        dst = index[ry[ok], rx[ok]]
        good = dst >= 0
        cost = np.hypot(dy, dx)
        rows.append(src[good])
        cols.append(dst[good])
        data.append(np.full(good.sum(), cost))
    g = coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(coords), len(coords)),
    ).tocsr()
    return coords, g


def longest_path_pixels(skel: np.ndarray) -> np.ndarray:
    """Longest geodesic path through the skeleton, as ordered (row, col) pixels.

    Double sweep with Dijkstra (edge cost 1 or sqrt(2)): exact for trees, which is
    what a pruned band skeleton is; for the rare skeleton with a loop it still
    returns a sensible longest traversal.
    """
    coords, g = _build_graph(skel)
    if g is None:
        return np.empty((0, 2), dtype=int)
    if len(coords) == 1:
        return coords
    n_comp, labels = connected_components(g, directed=False)
    if n_comp > 1:
        biggest = np.argmax(np.bincount(labels))
        keep = labels == biggest
        sub = np.zeros_like(skel)
        sub[coords[keep, 0], coords[keep, 1]] = True
        coords, g = _build_graph(sub)
    d0 = dijkstra(g, directed=False, indices=0)
    src = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    d1, pred = dijkstra(g, directed=False, indices=src, return_predecessors=True)
    dst = int(np.argmax(np.where(np.isfinite(d1), d1, -1)))
    path = [dst]
    while path[-1] != src and path[-1] >= 0:
        path.append(int(pred[path[-1]]))
    path.reverse()
    return coords[np.array([p for p in path if p >= 0])]


@dataclass
class Polyline:
    """Ordered centerline. `pts` is (N, 2) in (x, y) pixel coordinates."""

    pts: np.ndarray
    s: np.ndarray  # cumulative arclength in pixels, s[0] == 0

    @classmethod
    def from_pixels(cls, pixels: np.ndarray) -> "Polyline":
        pts = np.column_stack([pixels[:, 1], pixels[:, 0]]).astype(float)
        return cls.from_points(pts)

    @classmethod
    def from_points(cls, pts: np.ndarray) -> "Polyline":
        pts = np.asarray(pts, dtype=float)
        if len(pts) == 0:
            return cls(pts.reshape(0, 2), np.zeros(0))
        step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return cls(pts, np.concatenate([[0.0], np.cumsum(step)]))

    def __len__(self) -> int:
        return len(self.pts)

    @property
    def length(self) -> float:
        return float(self.s[-1]) if len(self.s) else 0.0

    @property
    def x(self) -> np.ndarray:
        return self.pts[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.pts[:, 1]

    def oriented(self, mode: str) -> "Polyline":
        """Return the same polyline ordered so that index 0 is the `mode` end."""
        first, last = self.pts[0], self.pts[-1]
        axis = 0 if mode in ("xmin", "xmax") else 1
        want_min = mode in ("xmin", "ymin")
        flip = (first[axis] > last[axis]) if want_min else (first[axis] < last[axis])
        return Polyline.from_points(self.pts[::-1]) if flip else self

    def endpoint(self, mode: str) -> np.ndarray:
        """Endpoint of the *path* (not the extreme pixel of the region)."""
        return self.oriented(mode).pts[0]

    def window_from(self, mode: str, max_s: float) -> "Polyline":
        """Points within `max_s` pixels of arclength from the `mode` endpoint."""
        p = self.oriented(mode)
        keep = p.s <= max_s
        if keep.sum() < 2:
            keep[:2] = True
        return Polyline(p.pts[keep], p.s[keep])

    def tangent(self, index: int, half: int = 6) -> np.ndarray:
        """Unit tangent from a local PCA, oriented along increasing index."""
        lo, hi = max(0, index - half), min(len(self.pts), index + half + 1)
        seg = self.pts[lo:hi]
        if len(seg) < 2:
            return np.array([1.0, 0.0])
        centred = seg - seg.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        d = vt[0]
        if np.dot(d, seg[-1] - seg[0]) < 0:
            d = -d
        return d / max(np.linalg.norm(d), 1e-12)

    def tangents(self, half: int = 6) -> np.ndarray:
        return np.array([self.tangent(i, half) for i in range(len(self.pts))])


def centerline(
    mask: np.ndarray, backend: str = "auto", min_length: int = 3
) -> Optional[Polyline]:
    """Guo-Hall skeleton -> pruned longest path -> ordered polyline."""
    skel = guo_hall(mask, backend=backend)
    pixels = longest_path_pixels(skel)
    if len(pixels) < min_length:
        return None
    return Polyline.from_pixels(pixels)


def extend_to_region(
    line: Polyline,
    mask: np.ndarray,
    mode: str,
    max_extra_px: float = 60.0,
    tangent_half: int = 14,
) -> Tuple[Polyline, float]:
    """Extrapolate the `mode` end along its tangent out to the region's extreme.

    Compensates the tip retraction of morphological thinning. Returns the extended
    polyline and how far (px) it was extended; 0.0 when nothing was added.

    `tangent_half` is deliberately wider than the default used elsewhere: the tip of
    a skeleton is its raggedest part, and a short PCA window there produces a tangent
    whose error is multiplied by the extension length.
    """
    p = line.oriented(mode)
    axis = 0 if mode in ("xmin", "xmax") else 1
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return line, 0.0
    coord = xs if axis == 0 else ys
    target = coord.min() if mode in ("xmin", "ymin") else coord.max()
    tip = p.pts[0]
    t = -p.tangent(0, half=tangent_half)  # outward
    if abs(t[axis]) < 1e-6:
        return line, 0.0
    dist = (target - tip[axis]) / t[axis]
    if dist <= 0.5:
        return line, 0.0
    dist = min(dist, max_extra_px)
    step = np.arange(1.0, dist + 1e-9)
    extra = tip + np.outer(step, t)
    h, w = mask.shape
    inside = (
        (extra[:, 0] >= 0) & (extra[:, 0] < w) & (extra[:, 1] >= 0) & (extra[:, 1] < h)
    )
    extra = extra[inside]
    if len(extra) == 0:
        return line, 0.0
    return Polyline.from_points(np.vstack([extra[::-1], p.pts])), float(len(extra))
