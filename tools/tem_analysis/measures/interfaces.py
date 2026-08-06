"""Requirements 2 and 3: SAF_Ru <-> MgO_C endpoint offsets, and the inward dip.

Requirement 2  a1 = right end of the SAF_Ru_L centerline
               a2 = left  end of the SAF_Ru_R centerline
               b1, b2 = left / right ends of the MgO_C centerline (raw skeleton tips)
               b3, b4 = the same ends of the MgO_C *fitted* segment
               report dx, dy for (a1, b1), (a1, b3), (a2, b2) and (a2, b4)

Requirement 3  walk `--window-nm` (5 nm) back along the SAF_Ru_L centerline from
               a1 and report the largest *downward* excursion. The baseline is the
               endpoint itself, i.e. dev = y - y(a1), positive = lower in the
               image. The window is measured along arclength, not along x: on a
               layer that bends near its tip an x-window would cover a different
               amount of actual layer.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from context import AnalysisContext, MissingClass


def _dip(ctx: AnalysisContext, line, end: str, window_nm: float) -> Dict[str, Any]:
    """Largest downward deviation from the endpoint within `window_nm` of arclength."""
    win = line.window_from(end, ctx.px(window_nm))
    origin = win.pts[0]
    dev = win.pts[:, 1] - origin[1]
    i = int(np.argmax(dev))
    j = int(np.argmax(np.abs(dev)))
    return {
        "baseline": "horizontal line through the endpoint",
        "window": {"requested_nm": float(window_nm), **ctx.dist(win.length)},
        "window_points": int(len(win)),
        "window_start": ctx.point(origin),
        "window_end": ctx.point(win.pts[-1]),
        # The traversed skeleton segment, so the overlay can shade the exact span
        # that was searched rather than an approximation of it.
        "window_polyline": [[float(x), float(y)] for x, y in win.pts],
        "max_downward_deviation": ctx.dist(float(dev[i])),
        "max_downward_point": ctx.point(win.pts[i]),
        "max_downward_arclength": ctx.dist(float(win.s[i])),
        "max_abs_deviation": ctx.dist(float(dev[j])),
        "max_abs_point": ctx.point(win.pts[j]),
        "deviation_profile": [
            [ctx.nm(float(s)), ctx.nm(float(d))] for s, d in zip(win.s, dev)
        ],
    }


def measure(ctx: AnalysisContext) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    window_nm = float(ctx.params.window_nm)

    # Both ends of MgO_C are measured, so both are eligible for the tip-retraction
    # correction; SAF_Ru only needs the end that faces the stack.
    mgo_c = ctx.centerline("MgO_C", extend=("xmin", "xmax"))
    b1 = mgo_c.endpoint("xmin")
    b2 = mgo_c.endpoint("xmax")
    out["b1"] = ctx.point(b1)
    out["b2"] = ctx.point(b2)
    out["mgo_c_skeleton_length"] = ctx.dist(mgo_c.length)

    # b3/b4 are the ends of the MgO_C *fitted* segment, computed by measures/mgo_c.py
    # (which the registry runs first). Reported alongside b1/b2 so the offset can be
    # read either against the raw skeleton tip or against the straight-line model.
    b3 = ctx.measured_point("mgo_c", "b3")
    b4 = ctx.measured_point("mgo_c", "b4")
    if b3 is None or b4 is None:
        ctx.warn("MgO_C fitted segment unavailable, so a1-b3 / a2-b4 were skipped")
    out["b3"] = ctx.point(b3) if b3 is not None else None
    out["b4"] = ctx.point(b4) if b4 is not None else None

    for tag, cls, end, raw, fitted, raw_tag, fit_tag in (
        ("a1", "SAF_Ru_L", "xmax", b1, b3, "b1", "b3"),
        ("a2", "SAF_Ru_R", "xmin", b2, b4, "b2", "b4"),
    ):
        try:
            line = ctx.centerline(cls, extend=(end,))
        except MissingClass as exc:
            ctx.warn("{}: {}".format(tag, exc))
            out[tag] = None
            out["{}_{}".format(tag, raw_tag)] = None
            out["{}_{}".format(tag, fit_tag)] = None
            out["{}_dip".format(cls.lower())] = None
            continue
        a = line.endpoint(end)
        out[tag] = ctx.point(a)
        out["{}_{}".format(tag, raw_tag)] = ctx.offset(a, raw)
        out["{}_{}".format(tag, fit_tag)] = (
            ctx.offset(a, fitted) if fitted is not None else None
        )
        out["{}_skeleton_length".format(cls.lower())] = ctx.dist(line.length)
        out["{}_dip".format(cls.lower())] = _dip(ctx, line, end, window_nm)

    for key in ("a1_b1", "a2_b2"):
        if out.get(key) is None:
            ctx.warn("{} offset unavailable".format(key))
    return out
