"""Requirement 5: the milled Milling flank, and its point of maximum curvature.

A Milling band has two flanks: one against the device stack (MgO_L / MgO_R /
Block_*), one against vacuum. The measured one is the vacuum-facing flank -- the
surface the ion mill actually produced.

Three selection methods, all working on the ordered outer boundary (a spline needs
the boundary in traversal order, which an erosion difference cannot provide):

`outer` (default) -- keep boundary pixels that touch *no* other class, i.e. only
    background, then keep the longest contiguous run. This is the milled surface.

`inner` -- the mirror image: keep pixels that do touch another class. Useful for
    measuring the interface against the stack rather than the milled surface.

`skeleton` -- the originally proposed method: split the boundary with the skeleton
    and keep the side whose signed distance along the skeleton normal is positive.
    Parameter-free and needs no neighbouring class, but on this geometry it flips
    partway along the band: measurement on the sample mask shows MgO_L on the
    negative-normal side low down and Block_U on the positive-normal side higher up,
    so no single sign holds one flank for the whole length. Kept for comparison
    only.

The edge is line - curve - line. Instead of fitting that piecewise model, the whole
edge is smoothed with a cubic B-spline and the curvature is taken analytically; the
straight tails come out at kappa ~ 0 and cannot win the maximum. Only a small
arclength margin at each end is excluded, to reject spline end artifacts --
note the bend is *not* at the arclength midpoint (on the sample mask it is at
~85% of the arclength), so a "central fraction" window is off by default.

Straightness of the two tails is reported as a check on the line-curve-line model.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

import labelmap
from context import AnalysisContext, MissingClass
from contour import longest_run, trace_outer_contour
from geometry import Spline, dedupe_consecutive, fit_circle, fit_line_ols, fit_line_tls

REFERENCE = "Non_mag"
DENSE = 2000
# A real milling bend is not a circular arc, so 1-2 px of rms over the fitting
# window is normal. This threshold only has to catch the ill-conditioned fits, which
# land at 5-17 px, so it is set well above the honest residuals seen in a 26-image
# batch (max 2.4 px) and well below the failures.
CIRCLE_RMS_TOL_PX = 3.5
_CONN8 = np.ones((3, 3), dtype=bool)


def _touches_other_class(
    ctx: AnalysisContext, mask: np.ndarray, boundary: np.ndarray
) -> np.ndarray:
    """Per boundary pixel: does it touch some class other than background/self?"""
    other = (ctx.ids != 0) & (ctx.ids != labelmap.IGNORE_INDEX) & ~mask
    near = ndimage.binary_dilation(other, structure=_CONN8)
    bi = np.round(boundary).astype(int)
    return near[bi[:, 1], bi[:, 0]]


def _on_frame(ctx: AnalysisContext, boundary: np.ndarray) -> np.ndarray:
    """Per boundary pixel: is it on the image crop rather than a real surface?

    The Milling bands run off the left/right edge of the field of view, so the outer
    flank's boundary includes the straight cut made by the crop, plus the sharp
    corners where the flank meets it. Those corners are the highest curvature on the
    whole edge and would win the maximum -- `m2` landed on the right image border
    before this was excluded. Rectification turns that straight cut into a slanted,
    nearest-neighbour-jagged line bordering the IGNORE padding, so both the frame and
    the padding are excluded, dilated by `--milling-frame-margin-px` to clear the
    corner itself rather than only the cut.
    """
    h, w = ctx.ids.shape
    invalid = ctx.ids == labelmap.IGNORE_INDEX
    invalid[0, :] = invalid[-1, :] = True
    invalid[:, 0] = invalid[:, -1] = True
    margin = max(1, int(ctx.params.milling_frame_margin_px))
    invalid = ndimage.binary_dilation(invalid, structure=_CONN8, iterations=margin)
    bi = np.round(boundary).astype(int)
    return invalid[np.clip(bi[:, 1], 0, h - 1), np.clip(bi[:, 0], 0, w - 1)]


def _keep_by_skeleton(ctx: AnalysisContext, cls: str, boundary: np.ndarray):
    """Positive side of the skeleton normal, oriented toward the Non_mag centroid."""
    line = ctx.centerline(cls)
    tangents = line.tangents(half=ctx.params.milling_tangent_half)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    try:
        target = ctx.centroid(REFERENCE)
    except MissingClass:
        target = np.array([ctx.ids.shape[1] / 2.0, ctx.ids.shape[0] / 2.0])
        ctx.warn(
            "{} missing; used the image centre to orient the {} edge "
            "normal".format(REFERENCE, cls)
        )
    if float(np.sum(np.einsum("ij,ij->i", normals, target - line.pts))) < 0:
        normals = -normals
    _, idx = cKDTree(line.pts).query(boundary)
    return np.einsum("ij,ij->i", boundary - line.pts[idx], normals[idx]) > 0


def _inner_edge(ctx: AnalysisContext, cls: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    mask = ctx.mask(cls)
    boundary = trace_outer_contour(mask)
    if len(boundary) < 16:
        raise MissingClass("{} boundary is too short to fit".format(cls))

    method = ctx.params.milling_edge_method
    if method == "skeleton":
        keep = _keep_by_skeleton(ctx, cls, boundary)
    else:
        touching = _touches_other_class(ctx, mask, boundary)
        keep = ~touching if method == "outer" else touching
    keep = keep & ~_on_frame(ctx, boundary)
    run = longest_run(keep, closed=True)
    if len(run) < 16:
        raise MissingClass(
            "{} {} edge has only {} points".format(cls, method, len(run))
        )
    if len(run) < 0.5 * int(keep.sum()):
        ctx.warn(
            "{}: the {} boundary is fragmented ({} of {} selected pixels are in the "
            "longest run); the edge may be interrupted by another class".format(
                cls, method, len(run), int(keep.sum())
            )
        )
    edge = boundary[run]
    # Traverse left-to-right so tails and m1/m2 are comparable across images.
    if edge[0, 0] > edge[-1, 0]:
        edge = edge[::-1]
    info = {
        "edge_method": method,
        "boundary_points": int(len(boundary)),
        "selected_points": int(keep.sum()),
        "edge_points": int(len(edge)),
    }
    return edge, info


def _line_fit(ctx: AnalysisContext, pts: np.ndarray) -> Optional[Dict[str, Any]]:
    if len(pts) < 8:
        return None
    tls = fit_line_tls(pts)
    ols = fit_line_ols(pts)
    p1, p2 = tls.endpoints_spanning(pts)
    return {
        "points": int(len(pts)),
        "angle_deg_image": tls.angle_deg_image,
        "angle_deg_math": tls.angle_deg_math,
        "fit_r2": ols["r2"],
        "fit_r2_axis": ols["axis"],
        "fit_r2_degenerate": ols["r2_degenerate"],
        "rmse": ctx.dist(ols["rmse_px"]),
        "max_residual": ctx.dist(ols["max_residual_px"]),
        "p1": ctx.point(p1),
        "p2": ctx.point(p2),
    }


def _local_circle(
    ctx: AnalysisContext, cls: str, edge: np.ndarray, at: np.ndarray, radius_px: float
) -> Optional[Dict[str, Any]]:
    """Fit a circle to the raw edge points around the located bend.

    Why this exists: the spline's *location* of the bend is reliable, but its
    curvature *magnitude* is not. splprep reduces the knot count until the residual
    matches `s`, and a few cubic segments cannot hold a constant curvature -- on a
    synthetic mask arc of exactly 200 px the spline reports a 168 px radius (-16%)
    while this circle fit reports 190 px (-5%). Evaluating the circle only at the
    already-located bend avoids the other trap: a global "smallest local radius"
    scan is won by ill-conditioned fits on the straight tails.
    """
    step = np.linalg.norm(np.diff(edge, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(step)])
    i = int(cKDTree(edge).query(at)[1])
    win = float(
        np.clip(ctx.params.milling_circle_window_frac * radius_px, 12.0, 0.25 * s[-1])
    )
    sel = np.abs(s - s[i]) <= win
    if sel.sum() < 8:
        return None
    centre, radius, rms = fit_circle(edge[sel])
    reliable = rms <= CIRCLE_RMS_TOL_PX
    if not reliable:
        ctx.warn(
            "{}: the local circle fit has a {:.2f} px rms residual (> {:.1f}), so the "
            "bend is not a single arc at this scale".format(cls, rms, CIRCLE_RMS_TOL_PX)
        )
    return {
        "centre": ctx.point(centre),
        "radius": ctx.dist(radius),
        "kappa_per_nm": (1.0 / radius) / ctx.scale_nm,
        "window": ctx.dist(win, ""),
        "points": int(sel.sum()),
        "rms": ctx.dist(rms),
        "reliable": bool(reliable),
    }


def _measure(ctx: AnalysisContext, cls: str) -> Dict[str, Any]:
    edge, info = _inner_edge(ctx, cls)
    edge = dedupe_consecutive(edge)
    sigma = float(ctx.params.milling_smooth)
    spline = Spline.fit(edge, smooth=len(edge) * sigma * sigma)
    u, pts, s = spline.sample(DENSE)
    kappa = spline.curvature(u)  # signed, 1/px
    total = float(s[-1])

    margin = ctx.px(float(ctx.params.milling_edge_margin_nm))
    frac = float(ctx.params.milling_middle_frac)
    lo = max(margin, (0.5 - frac / 2.0) * total)
    hi = min(total - margin, (0.5 + frac / 2.0) * total)
    band = (s >= lo) & (s <= hi)
    if band.sum() < 3:
        ctx.warn(
            "{}: curvature search band is empty (edge {:.1f} nm, margin {:.1f} nm, "
            "middle-frac {:.2f}); searched the whole edge instead".format(
                cls, ctx.nm(total), ctx.params.milling_edge_margin_nm, frac
            )
        )
        band = np.ones_like(s, dtype=bool)
        lo, hi = 0.0, total

    i = int(np.nonzero(band)[0][int(np.argmax(np.abs(kappa[band])))])
    kappa_px = float(kappa[i])
    if abs(s[i] - lo) < 1e-6 or abs(s[i] - hi) < 1e-6:
        ctx.warn(
            "{}: the curvature maximum sits on the edge of the search band; widen "
            "--milling-middle-frac or shrink --milling-edge-margin-nm".format(cls)
        )

    tail = float(ctx.params.milling_tail_frac)
    radius_px = 1.0 / max(abs(kappa_px), 1e-12)
    return {
        **info,
        "edge_length": ctx.dist(total),
        "spline_sigma_px": sigma,
        "spline_rmse": ctx.dist(spline.rmse_to(edge)),
        "search_band_nm": [ctx.nm(lo), ctx.nm(hi)],
        "max_curvature_point": ctx.point(pts[i]),
        "max_curvature_arclength": ctx.dist(float(s[i])),
        "kappa_per_px": kappa_px,
        "kappa_per_nm": kappa_px / ctx.scale_nm,
        "kappa_abs_per_nm": abs(kappa_px) / ctx.scale_nm,
        "radius": ctx.dist(radius_px),
        "local_circle": _local_circle(ctx, cls, edge, pts[i], radius_px),
        "magnitude_note": (
            "'radius'/'kappa_*' come from the spline derivatives, which locate the "
            "bend well but underestimate the radius by roughly 15% on a smooth arc "
            "(measured on synthetic ground truth). 'local_circle.radius' is a "
            "circle fitted to the raw edge points around the same point and is "
            "accurate to about 5%; prefer it for magnitudes"
        ),
        "sign_note": "kappa>0 turns clockwise in image coordinates (y down)",
        "tail_left": _line_fit(ctx, pts[s <= tail * total]),
        "tail_right": _line_fit(ctx, pts[s >= (1.0 - tail) * total]),
        "edge_polyline": [[float(x), float(y)] for x, y in pts[::10]],
        "curvature_profile": [
            [ctx.nm(float(a)), float(k) / ctx.scale_nm]
            for a, k in zip(s[::10], kappa[::10])
        ],
    }


def measure_left(ctx: AnalysisContext) -> Dict[str, Any]:
    return {"point_name": "m1", **_measure(ctx, "Milling_L")}


def measure_right(ctx: AnalysisContext) -> Dict[str, Any]:
    return {"point_name": "m2", **_measure(ctx, "Milling_R")}
