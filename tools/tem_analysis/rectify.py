"""Rotate a label map so the Leveling layer becomes horizontal.

`scipy.ndimage.rotate` would do the resampling, but its angle sign and its output
centring are easy to get subtly wrong, and every measured point has to be mappable
back to original-image coordinates. So the affine transform is built explicitly
here and kept alongside its inverse.

Interpolation is nearest-neighbour (`order=0`): the array holds class ids, and any
averaging would invent classes that were never predicted. Padding is filled with
IGNORE_INDEX rather than 0 so that "outside the original frame" stays
distinguishable from "predicted as background".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from labelmap import IGNORE_INDEX


def _rot(theta_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


@dataclass
class Rectifier:
    """Maps original (x, y) <-> rectified (x, y). Identity when angle is None."""

    angle_deg: Optional[float] = None
    in_shape: Tuple[int, int] = (0, 0)
    out_shape: Tuple[int, int] = (0, 0)
    _fwd: np.ndarray = field(default_factory=lambda: np.eye(2))
    _inv: np.ndarray = field(default_factory=lambda: np.eye(2))
    _c_in: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _c_out: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @property
    def applied(self) -> bool:
        return self.angle_deg is not None

    @classmethod
    def identity(cls, shape: Tuple[int, int]) -> "Rectifier":
        return cls(None, shape, shape)

    @classmethod
    def from_angle(cls, shape: Tuple[int, int], line_angle_deg: float) -> "Rectifier":
        """Build the transform that rotates `line_angle_deg` down to horizontal."""
        theta = -float(line_angle_deg)
        fwd = _rot(theta)
        h, w = shape
        corners = np.array(
            [[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0]]
        )
        c_in = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
        rotated = (corners - c_in) @ fwd.T
        span = rotated.max(axis=0) - rotated.min(axis=0)
        out_w, out_h = int(math.ceil(span[0])) + 1, int(math.ceil(span[1])) + 1
        c_out = np.array([(out_w - 1) / 2.0, (out_h - 1) / 2.0])
        return cls(theta, shape, (out_h, out_w), fwd, _rot(-theta), c_in, c_out)

    def apply(self, ids: np.ndarray) -> np.ndarray:
        if not self.applied:
            return ids
        # affine_transform maps output -> input, in (row, col) order.
        m = self._inv
        m_rc = np.array([[m[1, 1], m[1, 0]], [m[0, 1], m[0, 0]]])
        c_in_rc = self._c_in[::-1]
        c_out_rc = self._c_out[::-1]
        return ndimage.affine_transform(
            ids,
            m_rc,
            offset=c_in_rc - m_rc @ c_out_rc,
            output_shape=self.out_shape,
            order=0,
            mode="constant",
            cval=IGNORE_INDEX,
            output=np.uint8,
        )

    def to_orig(self, pts) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        if not self.applied:
            return pts
        return (pts - self._c_out) @ self._inv.T + self._c_in

    def to_rect(self, pts) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        if not self.applied:
            return pts
        return (pts - self._c_in) @ self._fwd.T + self._c_out

    def matrix(self):
        """Forward 3x3 homogeneous matrix in (x, y), for the JSON report."""
        if not self.applied:
            return np.eye(3).tolist()
        m = np.eye(3)
        m[:2, :2] = self._fwd
        m[:2, 2] = self._c_out - self._fwd @ self._c_in
        return m.tolist()
