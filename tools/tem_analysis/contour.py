"""Ordered outer-boundary extraction.

The Milling measurement fits a spline along the region edge, which needs the
boundary pixels *in traversal order*. An erosion difference (`mask & ~eroded`)
gives the right pixel set but no ordering, so we trace instead: Moore-neighbour
tracing with Jacob's stopping criterion, which is the same contour OpenCV's
`findContours(RETR_EXTERNAL, CHAIN_APPROX_NONE)` returns. Implemented here so the
tool has no hard OpenCV dependency.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# Clockwise from north-west, in (row, col) offsets.
_DIRS: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
)
_DIR_INDEX = {d: i for i, d in enumerate(_DIRS)}


def trace_outer_contour(mask: np.ndarray) -> np.ndarray:
    """Return the outer boundary of the largest blob as ordered (x, y) points."""
    m = np.pad(np.asarray(mask, dtype=bool), 1)
    filled = np.argwhere(m)
    if len(filled) == 0:
        return np.empty((0, 2), dtype=float)
    start = (int(filled[0][0]), int(filled[0][1]))  # topmost, then leftmost
    if not m[start[0], start[1] - 1] and m.sum() == 1:
        return np.array([[start[1] - 1.0, start[0] - 1.0]])

    contour: List[Tuple[int, int]] = [start]
    b = start
    back = (start[0], start[1] - 1)  # west of the topmost-leftmost pixel: background
    first_step = None
    # A closed 8-connected boundary visits at most ~2x its pixel count.
    for _ in range(8 * int(m.sum()) + 8):
        d0 = _DIR_INDEX[(back[0] - b[0], back[1] - b[1])]
        nxt = None
        for k in range(1, 9):
            dy, dx = _DIRS[(d0 + k) % 8]
            cand = (b[0] + dy, b[1] + dx)
            if m[cand[0], cand[1]]:
                py, px = _DIRS[(d0 + k - 1) % 8]
                back = (b[0] + py, b[1] + px)
                nxt = cand
                break
        if nxt is None:
            break  # isolated pixel
        if first_step is None:
            first_step = (b, nxt)
        elif (b, nxt) == first_step:
            break
        contour.append(nxt)
        b = nxt
    # Jacob's criterion stops one step after re-entering the start pixel, so the
    # start is in the list twice; keep the ring free of a duplicate.
    if len(contour) > 1 and contour[-1] == contour[0]:
        contour.pop()
    pts = np.array(contour, dtype=float) - 1.0
    return np.column_stack([pts[:, 1], pts[:, 0]])  # -> (x, y)


def longest_run(keep: np.ndarray, closed: bool = True) -> np.ndarray:
    """Indices of the longest contiguous True run in `keep`.

    The contour is a cycle, so a run may wrap past the last index; with
    `closed=True` the wrap-around is handled by rotating the array.
    """
    keep = np.asarray(keep, dtype=bool)
    n = len(keep)
    if n == 0 or not keep.any():
        return np.empty(0, dtype=int)
    if keep.all():
        return np.arange(n)
    shift = 0
    if closed and keep[0] and keep[-1]:
        shift = int(np.argmin(keep))  # rotate so the array starts on a False
    rolled = np.roll(keep, -shift)
    edges = np.diff(np.concatenate([[0], rolled.view(np.int8), [0]]))
    starts = np.nonzero(edges == 1)[0]
    ends = np.nonzero(edges == -1)[0]
    best = int(np.argmax(ends - starts))
    return (np.arange(starts[best], ends[best]) + shift) % n
