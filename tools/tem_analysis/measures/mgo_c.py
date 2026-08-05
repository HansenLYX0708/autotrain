"""Requirement 4: fit a line to the MgO_C centerline, report fit R2 and angle.

R2 comes from an ordinary least-squares y~x fit, which is what "fit R2" normally
means. The reported angle comes from a total-least-squares fit instead: OLS
systematically flattens the slope when the residuals are comparable to the x-span,
while TLS is orientation-unbiased. Both are in the output so the difference is
visible rather than hidden.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from context import AnalysisContext
from geometry import fit_line_ols, fit_line_tls

CLASS = "MgO_C"


def measure(ctx: AnalysisContext) -> Dict[str, Any]:
    # Same centerline as measures/interfaces.py, so the fitted segment spans exactly
    # b1..b2 instead of a differently-retracted pair of endpoints.
    line = ctx.centerline(CLASS, extend=("xmin", "xmax"))
    tls = fit_line_tls(line.pts)
    ols = fit_line_ols(line.pts)
    if ols["axis"] != "y~x":
        ctx.warn(
            "{} is closer to vertical than horizontal; R2 was computed on x~y "
            "instead of y~x".format(CLASS)
        )
    # No warning for a degenerate R2: MgO_C is horizontal by construction, so it is
    # degenerate on essentially every image and a warning would be pure noise. The
    # flag and the explanation travel with the result instead.
    p1, p2 = tls.endpoints_spanning(line.pts)
    resid = tls.signed_distance(line.pts)
    return {
        "skeleton_points": int(len(line)),
        "skeleton_length": ctx.dist(line.length),
        "angle_deg_image": tls.angle_deg_image,
        "angle_deg_math": tls.angle_deg_math,
        "fit_r2": ols["r2"],
        "fit_r2_axis": ols["axis"],
        "fit_r2_degenerate": ols["r2_degenerate"],
        "fit_r2_note": ols["r2_note"],
        "fit_slope": ols["slope"],
        "fit_intercept": ols["intercept"],
        "rmse": ctx.dist(ols["rmse_px"]),
        "max_residual": ctx.dist(ols["max_residual_px"]),
        "perp_rmse": ctx.dist(float(np.sqrt((resid**2).mean()))),
        "fit_p1": ctx.point(p1),
        "fit_p2": ctx.point(p2),
        "fit_length": ctx.dist(float(np.linalg.norm(p2 - p1))),
        "residual_profile": [
            [ctx.nm(float(s)), ctx.nm(float(r))] for s, r in zip(line.s, resid)
        ],
    }
