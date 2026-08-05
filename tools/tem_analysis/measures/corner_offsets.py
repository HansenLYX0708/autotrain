"""Offsets between the Non_mag edge endpoints and the Milling curvature points.

Non_mag1 <-> m1 on the left, Non_mag2 <-> m2 on the right. This is the only
measurement that combines two others, so it reads them back from `ctx.results`
instead of recomputing: it must run after `non_mag`, `milling_l` and `milling_r`,
which the ordering in `measures/__init__.py` guarantees.

Either side degrades to `null` on its own if the measurement it depends on failed,
so a mask that lost one Milling band still yields the other offset.
"""

from __future__ import annotations

from typing import Any, Dict

from context import AnalysisContext, MissingClass

PAIRS = (
    ("non_mag1_m1", "Non_mag1", "milling_l", "m1"),
    ("non_mag2_m2", "Non_mag2", "milling_r", "m2"),
)


def measure(ctx: AnalysisContext) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    found = 0
    for key, corner_name, milling_key, point_name in PAIRS:
        corner = ctx.measured_point("non_mag", corner_name)
        milling = ctx.measured_point(milling_key, "max_curvature_point")
        if corner is None or milling is None:
            missing = corner_name if corner is None else point_name
            ctx.warn("{}: {} is unavailable".format(key, missing))
            out[key] = None
            continue
        out[key] = {
            "from": corner_name,
            "to": point_name,
            "from_point": ctx.point(corner),
            "to_point": ctx.point(milling),
            **ctx.offset(corner, milling),
        }
        found += 1
    if found == 0:
        raise MissingClass("neither Non_mag endpoint could be paired with a Milling point")
    return out
