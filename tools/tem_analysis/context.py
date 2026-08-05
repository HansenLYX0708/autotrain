"""Shared state handed to every measurement module.

Each measurement in `measures/` is a `measure(ctx) -> dict`. Masks and centerlines
are cached here because several requirements need the same skeleton (SAF_Ru_L is
used both for the a1/b1 offset and for the 5 nm dip), and Guo-Hall thinning is the
expensive step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import labelmap
from rectify import Rectifier
from skeleton import Polyline, centerline, extend_to_region


class MissingClass(Exception):
    """A class the measurement needs is absent (or too small) in this mask."""


@dataclass
class AnalysisContext:
    ids: np.ndarray  # rectified label ids
    classes: List[str]
    scale_nm: float
    params: Any  # argparse.Namespace
    rect: Rectifier
    warnings: List[str] = field(default_factory=list)
    # Results computed on the *original* map before rectification (the Leveling
    # angle has to be known before the rotation it defines can be applied).
    pre: Dict[str, Any] = field(default_factory=dict)
    # Finished measurements, filled in by analyze.py in registry order, so a later
    # measurement can combine earlier ones (corner_offsets needs both non_mag and
    # milling). Entries are absent when the measurement failed.
    results: Dict[str, Any] = field(default_factory=dict)
    _masks: Dict[str, np.ndarray] = field(default_factory=dict)
    _lines: Dict[str, Optional[Polyline]] = field(default_factory=dict)

    # ---- units -------------------------------------------------------------- #

    def nm(self, px: float) -> float:
        return float(px) * self.scale_nm

    def px(self, nm: float) -> float:
        return float(nm) / self.scale_nm

    def dist(self, px: float, prefix: str = "") -> Dict[str, float]:
        key = "{}_".format(prefix) if prefix else ""
        return {key + "px": float(px), key + "nm": self.nm(px)}

    # ---- geometry ----------------------------------------------------------- #

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def class_id(self, name: str) -> int:
        try:
            return self.classes.index(name)
        except ValueError:
            raise MissingClass("class {!r} is not in the class list".format(name))

    def mask(self, name: str, prefer_near: Optional[str] = None) -> np.ndarray:
        """Cleaned boolean mask for one class, cached.

        `prefer_near` names a class the wanted component must touch; it breaks the
        "largest component wins" rule, which a spurious band larger than the real
        region can otherwise defeat (see labelmap.class_mask).
        """
        key = "{}|{}".format(name, prefer_near or "")
        if key not in self._masks:
            anchor = None
            if prefer_near:
                try:
                    anchor = self.mask(prefer_near)
                except MissingClass:
                    self.warn(
                        "{}: {} is unavailable, so the component was chosen by size "
                        "alone".format(name, prefer_near)
                    )
            m, dropped, anchored = labelmap.class_mask(
                self.ids,
                self.class_id(name),
                min_area=self.params.min_area,
                largest_only=not self.params.keep_all_components,
                prefer_near=anchor,
            )
            if not m.any():
                raise MissingClass("class {!r} has no pixels in this mask".format(name))
            if dropped:
                self.warn(
                    "{}: dropped {} component(s), largest {} px -- the layer may be "
                    "split in the prediction{}".format(
                        name,
                        len(dropped),
                        dropped[0],
                        "; kept the component touching {}".format(prefer_near)
                        if anchored
                        else "",
                    )
                )
            self._masks[key] = m
        return self._masks[key]

    def centerline(self, name: str, extend: Sequence[str] = ()) -> Polyline:
        """Pruned Guo-Hall centerline of a class, cached.

        `extend` lists `skeleton.Polyline.oriented` modes ("xmin"/"xmax"/...) and is
        only honoured when `--endpoint-extend region` is active; it pushes those
        ends of the skeleton back out to the region tips that thinning eroded away.
        """
        extend = tuple(extend)
        key = "{}|{}".format(name, ",".join(extend))
        if key not in self._lines:
            line = centerline(self.mask(name), backend=self.params.thinning_backend)
            if line is None:
                raise MissingClass(
                    "class {!r} skeleton is too short to measure".format(name)
                )
            if extend and self.params.endpoint_extend == "region":
                for end in extend:
                    line, added = extend_to_region(line, self.mask(name), end)
                    if added:
                        self.warn(
                            "{}: {} end extended by {:.0f} px to the region tip "
                            "(--endpoint-extend region)".format(name, end, added)
                        )
            self._lines[key] = line
        return self._lines[key]

    def centroid(self, name: str) -> np.ndarray:
        ys, xs = np.nonzero(self.mask(name))
        return np.array([xs.mean(), ys.mean()])

    # ---- reporting ---------------------------------------------------------- #

    def point(self, xy: Sequence[float]) -> Dict[str, float]:
        """A point in both rectified and original-image coordinates."""
        p = np.asarray(xy, dtype=float).reshape(2)
        o = self.rect.to_orig(p)[0]
        return {
            "x_px": float(p[0]),
            "y_px": float(p[1]),
            "x_orig_px": float(o[0]),
            "y_orig_px": float(o[1]),
        }

    def offset(self, a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
        """Signed b - a offset in px and nm, with the sign convention spelled out."""
        d = np.asarray(b, dtype=float).reshape(2) - np.asarray(a, dtype=float).reshape(2)
        return {
            "dx": self.dist(float(d[0])),
            "dy": self.dist(float(d[1])),
            "distance": self.dist(float(np.hypot(d[0], d[1]))),
            "convention": "dx>0: b is right of a; dy>0: b is lower than a in the image",
        }

    def measured_point(self, key: str, *path: str) -> Optional[np.ndarray]:
        """Fetch a point from an earlier measurement, or None if it is unavailable."""
        node: Any = self.results.get(key)
        for part in path:
            if not isinstance(node, dict):
                return None
            node = node.get(part)
        if not isinstance(node, dict) or "x_px" not in node:
            return None
        return np.array([float(node["x_px"]), float(node["y_px"])])

    def polyline_out(self, line: Polyline, step: int = 1) -> List[List[float]]:
        pts = line.pts[::step]
        return [[float(x), float(y)] for x, y in pts]
