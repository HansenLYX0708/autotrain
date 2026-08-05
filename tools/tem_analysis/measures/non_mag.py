"""Requirement 6: fit a line to the Non_mag edge away from Block_D, and its endpoints.

Which of the two horizontal edges to use is decided by adjacency rather than
hard-coded as "top" or "bottom": the wanted edge is the one *not* touching Block_D.
Counting Block_D contacts on both candidate edges makes the choice survive a sample
mounted the other way up, or a relabelled stack, instead of silently measuring the
wrong interface. `--nonmag-edge top|bottom` forces it.

Each edge is taken as the extreme foreground pixel per column. For a blob whose top
and bottom are single-valued functions of x -- which Non_mag is -- that *is* the
edge, with no side-selection heuristic to get wrong.

The fit is sigma-clipped: the left/right corners round off and would otherwise drag
the line, and a stray predicted pixel would dominate a plain least-squares fit.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy import ndimage

from context import AnalysisContext, MissingClass
from geometry import fit_line_ols, fit_line_tls, trimmed_line_fit

CLASS = "Non_mag"
AWAY_FROM = "Block_D"
_CONN8 = np.ones((3, 3), dtype=bool)


def edge_columns(mask: np.ndarray, side: str) -> np.ndarray:
    """Extreme foreground pixel per column, as (x, y) ordered by x."""
    cols = np.nonzero(mask.any(axis=0))[0]
    if len(cols) == 0:
        raise MissingClass("{} has no pixels".format(CLASS))
    if side == "top":
        rows = np.argmax(mask[:, cols], axis=0)
    else:
        rows = mask.shape[0] - 1 - np.argmax(mask[::-1, cols], axis=0)
    return np.column_stack([cols.astype(float), rows.astype(float)])


def _pick_side(ctx: AnalysisContext, mask: np.ndarray) -> Tuple[str, Dict[str, int]]:
    """Choose the edge that is not against Block_D, and report both contact counts."""
    forced = ctx.params.nonmag_edge
    counts = {"top": 0, "bottom": 0}
    try:
        near = ndimage.binary_dilation(ctx.mask(AWAY_FROM), structure=_CONN8)
    except MissingClass:
        fallback = forced if forced != "auto" else "bottom"
        ctx.warn(
            "{} is missing, so the {} edge could not be chosen by adjacency; using "
            "the {} edge".format(AWAY_FROM, CLASS, fallback)
        )
        return fallback, counts
    for side in ("top", "bottom"):
        pts = np.round(edge_columns(mask, side)).astype(int)
        counts[side] = int(near[pts[:, 1], pts[:, 0]].sum())
    if forced != "auto":
        return forced, counts
    side = "top" if counts["top"] < counts["bottom"] else "bottom"
    if counts[side] > 0.2 * len(np.nonzero(mask.any(axis=0))[0]):
        ctx.warn(
            "{}: both edges touch {} ({} top / {} bottom columns); the '{}' edge was "
            "picked but check the overlay".format(
                CLASS, AWAY_FROM, counts["top"], counts["bottom"], side
            )
        )
    return side, counts


def measure(ctx: AnalysisContext) -> Dict[str, Any]:
    # Anchored on Block_D rather than taking the biggest blob: one prediction in the
    # sample batch grew a spurious Non_mag band across the bottom of the frame that
    # was larger than the real box.
    mask = ctx.mask(CLASS, prefer_near=AWAY_FROM)
    side, contacts = _pick_side(ctx, mask)
    edge = edge_columns(mask, side)
    if len(edge) < 8:
        raise MissingClass("{} is only {} columns wide".format(CLASS, len(edge)))

    if ctx.params.nonmag_trim:
        line, keep = trimmed_line_fit(edge, sigma=float(ctx.params.nonmag_sigma))
    else:
        line, keep = fit_line_tls(edge), np.ones(len(edge), dtype=bool)
    used = edge[keep]
    ols = fit_line_ols(used, axis="y~x")

    # Endpoints are the fitted line evaluated over the surviving x-range, not raw
    # pixels, so corner rounding cannot shift them.
    x1, x2 = float(used[:, 0].min()), float(used[:, 0].max())
    p1 = np.array([x1, line.y_at(x1)])
    p2 = np.array([x2, line.y_at(x2)])
    dropped = int((~keep).sum())
    if dropped > 0.25 * len(edge):
        ctx.warn(
            "{}: sigma-clipping dropped {}/{} columns; the edge may not be "
            "straight".format(CLASS, dropped, len(edge))
        )
    return {
        "edge_side": side,
        "edge_side_rule": "the horizontal edge not adjacent to {}".format(AWAY_FROM),
        "{}_contact_columns".format(AWAY_FROM.lower()): contacts,
        "edge_columns": int(len(edge)),
        "columns_used": int(keep.sum()),
        "columns_dropped": dropped,
        "Non_mag1": ctx.point(p1),
        "Non_mag2": ctx.point(p2),
        "angle_deg_image": line.angle_deg_image,
        "angle_deg_math": line.angle_deg_math,
        "length": ctx.dist(float(np.linalg.norm(p2 - p1))),
        "fit_r2": ols["r2"],
        "fit_r2_degenerate": ols["r2_degenerate"],
        "fit_r2_note": ols["r2_note"],
        "rmse": ctx.dist(ols["rmse_px"]),
        "max_residual": ctx.dist(ols["max_residual_px"]),
        "edge_polyline": [[float(x), float(y)] for x, y in edge],
    }
