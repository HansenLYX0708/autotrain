"""Measurement registry: one entry per requirement.

Every callable takes an `AnalysisContext` and returns a JSON-serialisable dict.
`analyze.py` runs them in order and isolates failures, so a class missing from one
prediction degrades that single entry to `null` instead of losing the whole image.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

from measures import corner_offsets, interfaces, leveling, mgo_c, milling, non_mag

# Order matters: corner_offsets combines the non_mag and milling results, so it runs
# after them and reads them back from ctx.results.
MEASURES: "OrderedDict[str, Callable]" = OrderedDict(
    (
        ("leveling", leveling.measure),
        ("interfaces", interfaces.measure),
        ("mgo_c", mgo_c.measure),
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
