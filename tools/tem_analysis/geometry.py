"""Line fits, angles, R2, arclength splines and analytic curvature.

Angle convention, spelled out because image y grows downward:
  angle_deg_image  atan2(dy, dx) straight from pixel coordinates -> clockwise
                   rotation is positive
  angle_deg_math   atan2(-dy, dx) -> the usual counter-clockwise-positive angle
Both are reported everywhere so no caller has to guess.

Two line fits are kept on purpose:
  fit_line_tls  total least squares (PCA). Orientation-unbiased, used for every
                reported *angle* and for rotation.
  fit_line_ols  ordinary least squares y~x. Biased for steep lines, but it is
                what "fit R2" conventionally means, so it is used for R2/RMSE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import splev, splprep


def norm_angle_deg(deg: float) -> float:
    """Fold an undirected line angle into (-90, 90]."""
    a = (float(deg) + 90.0) % 180.0 - 90.0
    return 90.0 if a == -90.0 else a


@dataclass
class Line:
    point: np.ndarray  # a point on the line, (x, y)
    direction: np.ndarray  # unit direction, (x, y)

    @property
    def angle_deg_image(self) -> float:
        return norm_angle_deg(
            math.degrees(math.atan2(self.direction[1], self.direction[0]))
        )

    @property
    def angle_deg_math(self) -> float:
        return norm_angle_deg(-self.angle_deg_image)

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])

    def project(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        t = (pts - self.point) @ self.direction
        return self.point + np.outer(t, self.direction)

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(pts) - self.point) @ self.normal

    def endpoints_spanning(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The two projections of `pts` that are farthest apart along the line."""
        proj = self.project(pts)
        t = (proj - self.point) @ self.direction
        return proj[int(np.argmin(t))], proj[int(np.argmax(t))]

    def y_at(self, x: float) -> float:
        if abs(self.direction[0]) < 1e-12:
            return float("nan")
        t = (x - self.point[0]) / self.direction[0]
        return float(self.point[1] + t * self.direction[1])


def fit_line_tls(pts: np.ndarray) -> Line:
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        raise ValueError("need at least 2 points for a line fit")
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    d = vt[0]
    if d[0] < 0 or (abs(d[0]) < 1e-12 and d[1] < 0):
        d = -d
    return Line(centre, d / np.linalg.norm(d))


def fit_line_ols(pts: np.ndarray, axis: Optional[str] = None) -> Dict[str, float]:
    """Least-squares fit with a conventional R2.

    `axis` defaults to the better-conditioned regression: y~x for shallow point
    clouds, x~y for steep ones. R2 of y~x on a near-vertical layer is meaningless,
    so the choice is reported back to the caller.
    """
    pts = np.asarray(pts, dtype=float)
    if axis is None:
        line = fit_line_tls(pts)
        axis = "y~x" if abs(line.direction[0]) >= abs(line.direction[1]) else "x~y"
    u, v = (pts[:, 0], pts[:, 1]) if axis == "y~x" else (pts[:, 1], pts[:, 0])
    slope, intercept = np.polyfit(u, v, 1)
    pred = slope * u + intercept
    resid = v - pred
    ss_tot = float(((v - v.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt((resid**2).mean()))
    # R2 = 1 - SSres/SStot collapses when the dependent variable barely varies, i.e.
    # for a nearly axis-aligned line: a Leveling layer 1000 px long and 2 px wavy
    # scores R2 ~ 0.02 despite being an excellent straight fit. Flag it so callers
    # judge straightness by rmse instead of by R2.
    degenerate = bool(float(v.std()) < 3.0 * rmse)
    return {
        "axis": axis,
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "r2_degenerate": degenerate,
        "r2_note": (
            "the line is nearly axis-aligned, so R2 is dominated by the tiny "
            "variance of the dependent variable; use rmse to judge straightness"
            if degenerate
            else ""
        ),
        "rmse_px": rmse,
        "max_residual_px": float(np.abs(resid).max()),
        "residuals_px": resid,
    }


def trimmed_line_fit(
    pts: np.ndarray, sigma: float = 2.0, iters: int = 5, min_keep: int = 8
) -> Tuple[Line, np.ndarray]:
    """TLS fit with iterative sigma-clipping on the perpendicular residual."""
    pts = np.asarray(pts, dtype=float)
    keep = np.ones(len(pts), dtype=bool)
    line = fit_line_tls(pts)
    for _ in range(max(0, iters)):
        resid = line.signed_distance(pts[keep])
        scale = resid.std()
        if scale < 1e-9:
            break
        new = keep.copy()
        new[np.nonzero(keep)[0][np.abs(resid) > sigma * scale]] = False
        if new.sum() < max(min_keep, 2) or new.sum() == keep.sum():
            break
        keep = new
        line = fit_line_tls(pts[keep])
    return line, keep


def fit_circle(pts: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Algebraic circle fit. Returns (centre, radius, rms radial residual).

    The points are centred first. Without that, the normal equations for a nearly
    collinear arc are so ill-conditioned that the fit happily returns a 10 px radius
    with a 5 px residual -- which is exactly the failure mode that makes a naive
    "smallest local radius wins" search pick a straight stretch of edge. The rms is
    returned so callers can reject a fit that did not converge on a real arc.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        raise ValueError("need at least 3 points for a circle fit")
    origin = pts.mean(axis=0)
    p = pts - origin
    a = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
    rhs = p[:, 0] ** 2 + p[:, 1] ** 2
    sol, *_ = np.linalg.lstsq(a, rhs, rcond=None)
    centre = np.array([sol[0] / 2.0, sol[1] / 2.0])
    radius = float(np.sqrt(max(sol[2] + centre @ centre, 1e-12)))
    rms = float(np.sqrt(((np.linalg.norm(p - centre, axis=1) - radius) ** 2).mean()))
    return centre + origin, radius, rms


def dedupe_consecutive(pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """splprep needs a strictly increasing parameter; drop repeated points."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.vstack([pts[:1], pts[1:][step > tol]])


@dataclass
class Spline:
    """Cubic B-spline through an ordered point sequence, parametrised by arclength.

    Curvature is taken from the spline derivatives analytically rather than by
    finite-differencing the pixel chain, which would be dominated by the
    +/-1 px quantisation of the mask boundary.
    """

    tck: tuple
    length_px: float

    @classmethod
    def fit(cls, pts: np.ndarray, smooth: Optional[float] = None) -> "Spline":
        pts = dedupe_consecutive(pts)
        if len(pts) < 4:
            raise ValueError("need at least 4 distinct points for a cubic spline")
        step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(step)])
        total = float(s[-1])
        if smooth is None:
            smooth = float(len(pts))  # ~1 px of boundary noise per point
        tck, _ = splprep([pts[:, 0], pts[:, 1]], u=s / total, s=smooth, k=3)
        return cls(tck, total)

    def eval(self, u, der: int = 0) -> np.ndarray:
        out = splev(np.atleast_1d(u), self.tck, der=der)
        return np.column_stack([out[0], out[1]])

    def curvature(self, u) -> np.ndarray:
        """Signed curvature in 1/px: (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)."""
        d1 = self.eval(u, 1)
        d2 = self.eval(u, 2)
        num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        den = np.power(d1[:, 0] ** 2 + d1[:, 1] ** 2, 1.5)
        return num / np.maximum(den, 1e-12)

    def sample(self, n: int = 800) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (u, points, arclength in px) on a dense uniform-u grid."""
        u = np.linspace(0.0, 1.0, n)
        pts = self.eval(u)
        step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return u, pts, np.concatenate([[0.0], np.cumsum(step)])

    def rmse_to(self, pts: np.ndarray, n: int = 2000) -> float:
        """RMS distance from the original points to the smoothed curve."""
        _, dense, _ = self.sample(n)
        from scipy.spatial import cKDTree

        d, _ = cKDTree(dense).query(np.asarray(pts, dtype=float))
        return float(np.sqrt((d**2).mean()))
