"""Measurement registry: one entry per requirement.

Every callable takes an `AnalysisContext` and returns a JSON-serialisable dict.
`analyze.py` runs them in order and isolates failures, so a class missing from one
prediction degrades that single entry to `null` instead of losing the whole image.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

from measures import corner_offsets, interfaces, leveling, mgo_c, milling, non_mag

# Order matters. Measurements that combine others read them back from ctx.results, so
# they must run later: interfaces needs mgo_c's b3/b4, and corner_offsets needs
# non_mag plus both milling results.
MEASURES: "OrderedDict[str, Callable]" = OrderedDict(
    (
        ("leveling", leveling.measure),
        ("mgo_c", mgo_c.measure),
        ("interfaces", interfaces.measure),
        ("milling_l", milling.measure_left),
        ("milling_r", milling.measure_right),
        ("non_mag", non_mag.measure),
        ("corner_offsets", corner_offsets.measure),
    )
)

__all__ = [
    "MEASURES",
    "corner_offsets",
    "interfaces",
    "leveling",
    "mgo_c",
    "milling",
    "non_mag",
]
