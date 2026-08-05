"""Requirement 1: Leveling skeleton -> straight line -> tilt angle.

The angle this module measures on the *original* map is what defines the
rectification for every other measurement, so `fit_leveling()` is called by
`analyze.py` before the rotation exists. `measure()` then re-runs on the rectified
map; the residual angle there should be ~0 and is a cheap self-check on the whole
rotate/inverse-transform path.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import labelmap
from context import AnalysisContext, MissingClass
from geometry import fit_line_ols, fit_line_tls
from skeleton import centerline

CLASS = "Leveling"
RESIDUAL_TOL_DEG = 0.05
ANGLE_ERR_WARN_DEG = 0.05


def fit_leveling(ids: np.ndarray, classes: List[str], params) -> Dict[str, Any]:
    """Fit the Leveling layer on an arbitrary label map. Raises MissingClass."""
    if CLASS not in classes:
        raise MissingClass("class {!r} is not in the class list".format(CLASS))
    mask, dropped, _ = labelmap.class_mask(
        ids,
        classes.index(CLASS),
        min_area=params.min_area,
        largest_only=not params.keep_all_components,
    )
    if not mask.any():
        raise MissingClass("class {!r} has no pixels".format(CLASS))
    line = centerline(mask, backend=params.thinning_backend)
    if line is None:
        raise MissingClass("class {!r} skeleton is too short".format(CLASS))
    tls = fit_line_tls(line.pts)
    ols = fit_line_ols(line.pts, axis="y~x")
    p1, p2 = tls.endpoints_spanning(line.pts)
    return {
        "line": line,
        "tls": tls,
        "ols": ols,
        "p1": p1,
        "p2": p2,
        "dropped": dropped,
        "angle_deg_image": tls.angle_deg_image,
    }


def _summary(ctx: AnalysisContext, fit: Dict[str, Any]) -> Dict[str, Any]:
    tls, ols = fit["tls"], fit["ols"]
    return {
        "skeleton_points": int(len(fit["line"])),
        "skeleton_length": ctx.dist(fit["line"].length),
        "angle_deg_image": tls.angle_deg_image,
        "angle_deg_math": tls.angle_deg_math,
        "fit_r2": ols["r2"],
        "fit_r2_axis": ols["axis"],
        "fit_r2_degenerate": ols["r2_degenerate"],
        "fit_r2_note": ols["r2_note"],
        "rmse": ctx.dist(ols["rmse_px"]),
        "max_residual": ctx.dist(ols["max_residual_px"]),
        "p1": ctx.point(fit["p1"]),
        "p2": ctx.point(fit["p2"]),
    }


def angle_stderr_deg(pts: np.ndarray, rmse_px: float) -> float:
    """Standard error of the fitted tilt angle, in degrees.

    Judging the rectification by R2 does not work: a near-horizontal line always
    scores a poor R2 (see geometry.fit_line_ols). What matters is the uncertainty of
    the slope, whose textbook standard error is sigma / sqrt(sum (x - xbar)^2).

    This assumes independent residuals. Real layer waviness is correlated along x,
    so treat the number as a lower bound on the angle uncertainty; the residual
    angle measured after rectification is the empirical counterpart.
    """
    x = np.asarray(pts, dtype=float)[:, 0]
    sxx = float(((x - x.mean()) ** 2).sum())
    if sxx <= 0:
        return float("nan")
    return float(np.degrees(np.arctan(rmse_px / np.sqrt(sxx))))


def _straightness_warning(ctx: AnalysisContext, fit: Dict[str, Any]) -> float:
    se = angle_stderr_deg(fit["line"].pts, fit["ols"]["rmse_px"])
    if np.isfinite(se) and se > ANGLE_ERR_WARN_DEG:
        ctx.warn(
            "Leveling is {:.0f} px long with {:.2f} px rmse; the rectification angle "
            "has a standard error of {:.3f} deg".format(
                fit["line"].length, fit["ols"]["rmse_px"], se
            )
        )
    return se


def measure(ctx: AnalysisContext) -> Dict[str, Any]:
    pre = ctx.pre.get("leveling")
    if pre is None:
        raise MissingClass("Leveling was not measurable on the original image")

    out: Dict[str, Any] = {
        "angle_deg_image": float(pre["tls"].angle_deg_image),
        "angle_deg_math": float(pre["tls"].angle_deg_math),
        "fit_r2": pre["ols"]["r2"],
        "fit_r2_axis": pre["ols"]["axis"],
        "fit_r2_degenerate": pre["ols"]["r2_degenerate"],
        "fit_r2_note": pre["ols"]["r2_note"],
        "rmse": ctx.dist(pre["ols"]["rmse_px"]),
        "max_residual": ctx.dist(pre["ols"]["max_residual_px"]),
        "skeleton_points": int(len(pre["line"])),
        "skeleton_length": ctx.dist(pre["line"].length),
        "rotation_applied": ctx.rect.applied,
        "rotation_angle_deg": ctx.rect.angle_deg,
    }
    out["angle_stderr_deg"] = _straightness_warning(ctx, pre)

    # Self-check on the rectified map: the layer should now be horizontal.
    if ctx.rect.applied:
        after = fit_leveling(ctx.ids, ctx.classes, ctx.params)
        residual = after["tls"].angle_deg_image
        out["residual_angle_deg_after_rotation"] = residual
        out["rectified"] = _summary(ctx, after)
        if abs(residual) > RESIDUAL_TOL_DEG:
            ctx.warn(
                "Leveling is still tilted by {:.3f} deg after rectification "
                "(tolerance {:.2f})".format(residual, RESIDUAL_TOL_DEG)
            )
    return out
