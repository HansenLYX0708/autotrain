"""Serialise results: JSON per image, one CSV row per image, annotated overlay PNG.

The JSON is the complete record (including the curvature and deviation profiles, so
parameters can be re-tuned without re-running). The CSV is a fixed, explicitly
declared set of scalars -- a CSV whose columns are "whatever keys this batch
happened to produce" is useless for comparing runs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import labelmap
from context import AnalysisContext

# Dotted paths into the JSON `results` block, plus a few top-level fields.
CSV_COLUMNS: List[str] = [
    "image",
    "fov_nm",
    "image_width_px",
    "scale_nm_per_px",
    "scale_source",
    "rotation.applied",
    "rotation.angle_deg_image",
    "results.leveling.angle_deg_image",
    "results.leveling.angle_stderr_deg",
    # R2 is degenerate for near-horizontal layers (see geometry.fit_line_ols), so the
    # rmse columns are the ones to judge straightness by.
    "results.leveling.fit_r2",
    "results.leveling.rmse.nm",
    "results.leveling.residual_angle_deg_after_rotation",
    "results.interfaces.a1.x_px",
    "results.interfaces.a1.y_px",
    "results.interfaces.b1.x_px",
    "results.interfaces.b1.y_px",
    "results.interfaces.a1_b1.dx.nm",
    "results.interfaces.a1_b1.dy.nm",
    "results.interfaces.b3.x_px",
    "results.interfaces.b3.y_px",
    "results.interfaces.a1_b3.dx.nm",
    "results.interfaces.a1_b3.dy.nm",
    "results.interfaces.a2.x_px",
    "results.interfaces.a2.y_px",
    "results.interfaces.b2.x_px",
    "results.interfaces.b2.y_px",
    "results.interfaces.a2_b2.dx.nm",
    "results.interfaces.a2_b2.dy.nm",
    "results.interfaces.b4.x_px",
    "results.interfaces.b4.y_px",
    "results.interfaces.a2_b4.dx.nm",
    "results.interfaces.a2_b4.dy.nm",
    "results.interfaces.saf_ru_l_dip.max_downward_deviation.nm",
    "results.interfaces.saf_ru_l_dip.max_downward_arclength.nm",
    "results.interfaces.saf_ru_r_dip.max_downward_deviation.nm",
    "results.interfaces.saf_ru_r_dip.max_downward_arclength.nm",
    "results.mgo_c.angle_deg_image",
    "results.mgo_c.fit_r2",
    "results.mgo_c.rmse.nm",
    "results.mgo_c.fit_length.nm",
    "results.milling_l.edge_method",
    "results.milling_l.max_curvature_point.x_px",
    "results.milling_l.max_curvature_point.y_px",
    "results.milling_l.kappa_abs_per_nm",
    "results.milling_l.radius.nm",
    "results.milling_l.local_circle.radius.nm",
    "results.milling_l.local_circle.rms.px",
    "results.milling_l.local_circle.reliable",
    "results.milling_l.tail_left.angle_deg_image",
    "results.milling_l.tail_right.angle_deg_image",
    "results.milling_r.max_curvature_point.x_px",
    "results.milling_r.max_curvature_point.y_px",
    "results.milling_r.kappa_abs_per_nm",
    "results.milling_r.radius.nm",
    "results.milling_r.local_circle.radius.nm",
    "results.milling_r.local_circle.rms.px",
    "results.milling_r.local_circle.reliable",
    "results.milling_r.tail_left.angle_deg_image",
    "results.milling_r.tail_right.angle_deg_image",
    "results.non_mag.edge_side",
    "results.non_mag.Non_mag1.x_px",
    "results.non_mag.Non_mag1.y_px",
    "results.non_mag.Non_mag2.x_px",
    "results.non_mag.Non_mag2.y_px",
    "results.non_mag.angle_deg_image",
    "results.non_mag.length.nm",
    "results.non_mag.fit_r2",
    "results.non_mag.rmse.nm",
    "results.corner_offsets.non_mag1_m1.dx.nm",
    "results.corner_offsets.non_mag1_m1.dy.nm",
    "results.corner_offsets.non_mag1_m1.distance.nm",
    "results.corner_offsets.non_mag2_m2.dx.nm",
    "results.corner_offsets.non_mag2_m2.dy.nm",
    "results.corner_offsets.non_mag2_m2.distance.nm",
    "n_warnings",
    "warnings",
]


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def dig(record: Dict[str, Any], dotted: str) -> Any:
    node: Any = record
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if not isinstance(node, (dict, list)) else None


def write_json(record: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(record), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def csv_row(record: Dict[str, Any]) -> Dict[str, Any]:
    row = {c: dig(record, c) for c in CSV_COLUMNS}
    row["image"] = record.get("image")
    row["n_warnings"] = len(record.get("warnings") or [])
    row["warnings"] = " | ".join(record.get("warnings") or [])
    return row


def write_csv(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(csv_row(rec))


# --------------------------------------------------------------------------- #
# Overlay
# --------------------------------------------------------------------------- #

_SKELETON_CLASSES = (
    "Leveling",
    "SAF_Ru_L",
    "SAF_Ru_R",
    "MgO_C",
    "Milling_L",
    "Milling_R",
)


def pseudo_color(ids: np.ndarray, num_classes: int) -> np.ndarray:
    return labelmap.palette(num_classes)[ids]


def _pt(node: Optional[Dict[str, Any]]):
    if not node:
        return None
    return float(node["x_px"]), float(node["y_px"])


def render_overlay(
    ctx: AnalysisContext,
    record: Dict[str, Any],
    path: Path,
    background: Optional[np.ndarray] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = ctx.ids.shape
    img = background if background is not None else pseudo_color(ctx.ids, len(ctx.classes))
    fig, ax = plt.subplots(figsize=(w / 100.0, h / 100.0), dpi=140)
    ax.imshow(img, interpolation="nearest")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)

    for cls in _SKELETON_CLASSES:
        try:
            line = ctx.centerline(cls)
        except Exception:
            continue
        ax.plot(line.x, line.y, color="white", lw=0.8, alpha=0.85)

    res = record.get("results", {})

    def line_between(a, b, **kw):
        if a and b:
            ax.plot([a[0], b[0]], [a[1], b[1]], **kw)

    mgo = res.get("mgo_c") or {}
    line_between(_pt(mgo.get("b3")), _pt(mgo.get("b4")), color="cyan", ls="--", lw=1.2)

    nmg = res.get("non_mag") or {}
    if nmg.get("edge_polyline"):
        arr = np.asarray(nmg["edge_polyline"], dtype=float)
        ax.plot(arr[:, 0], arr[:, 1], color="yellow", lw=0.8, alpha=0.55)
    line_between(
        _pt(nmg.get("Non_mag1")),
        _pt(nmg.get("Non_mag2")),
        color="yellow",
        ls="--",
        lw=1.2,
    )

    # Non_mag1 -> m1 and Non_mag2 -> m2, drawn as an L so dx and dy are separable.
    for key in ("non_mag1_m1", "non_mag2_m2"):
        node = (res.get("corner_offsets") or {}).get(key) or {}
        a, b = _pt(node.get("from_point")), _pt(node.get("to_point"))
        if a is None or b is None:
            continue
        ax.plot([a[0], b[0], b[0]], [a[1], a[1], b[1]], color="springgreen", lw=1.0, ls=":")
        ax.annotate(
            "d=({:+.2f}, {:+.2f}) nm".format(
                (node.get("dx") or {}).get("nm", 0.0), (node.get("dy") or {}).get("nm", 0.0)
            ),
            ((a[0] + b[0]) / 2.0, a[1]),
            textcoords="offset points",
            xytext=(0, -6),
            ha="center",
            va="top",
            color="springgreen",
            fontsize=6.5,
        )

    lev = res.get("leveling") or {}
    rect_lev = lev.get("rectified") or {}
    line_between(
        _pt(rect_lev.get("p1")), _pt(rect_lev.get("p2")), color="lime", ls="--", lw=1.0
    )

    for key in ("milling_l", "milling_r"):
        node = res.get(key) or {}
        poly = node.get("edge_polyline")
        if poly:
            arr = np.asarray(poly, dtype=float)
            ax.plot(arr[:, 0], arr[:, 1], color="orange", lw=1.2, alpha=0.9)
        # The fitted osculating circle makes an over- or under-estimated radius
        # obvious at a glance, which no printed number does.
        circle = node.get("local_circle") or {}
        centre = _pt(circle.get("centre"))
        radius = (circle.get("radius") or {}).get("px")
        if centre and radius and radius < 4 * max(h, w):
            ax.add_patch(
                plt.Circle(
                    centre,
                    radius,
                    fill=False,
                    color="deepskyblue",
                    lw=0.9,
                    ls=":",
                    alpha=0.9,
                )
            )

    # a1/b1/blockD1 sit within a few pixels of each other, so the label offsets are
    # staggered by hand rather than all placed up-and-right.
    labelled = [
        ("a1", (res.get("interfaces") or {}).get("a1"), "red", (7, 7)),
        ("a2", (res.get("interfaces") or {}).get("a2"), "red", (7, 7)),
        ("b1", (res.get("interfaces") or {}).get("b1"), "cyan", (-4, 9)),
        ("b2", (res.get("interfaces") or {}).get("b2"), "cyan", (4, 9)),
        ("b3", (res.get("interfaces") or {}).get("b3"), "deepskyblue", (-4, -13)),
        ("b4", (res.get("interfaces") or {}).get("b4"), "deepskyblue", (4, -13)),
        ("m1", (res.get("milling_l") or {}).get("max_curvature_point"), "orange", (8, -3)),
        ("m2", (res.get("milling_r") or {}).get("max_curvature_point"), "orange", (8, -3)),
        # Above the point: m1/m2 sit just below these two and would collide.
        ("Non_mag1", (res.get("non_mag") or {}).get("Non_mag1"), "yellow", (6, 8)),
        ("Non_mag2", (res.get("non_mag") or {}).get("Non_mag2"), "yellow", (6, 8)),
    ]
    for name, node, color, offset in labelled:
        p = _pt(node)
        if p is None:
            continue
        ax.plot([p[0]], [p[1]], marker="o", ms=5, mfc="none", mec=color, mew=1.4)
        ax.annotate(
            name,
            p,
            textcoords="offset points",
            xytext=offset,
            color=color,
            fontsize=7,
            weight="bold",
        )

    inter = res.get("interfaces") or {}
    for dip_key, anchor_key in (("saf_ru_l_dip", "a1"), ("saf_ru_r_dip", "a2")):
        dip = inter.get(dip_key) or {}
        a = _pt(inter.get(anchor_key))
        if a is None:
            continue
        # Shade the searched window: the skeleton span, its two end ticks and the
        # baseline the deviation is measured from, so the number can be read off.
        span = dip.get("window_polyline")
        if span:
            arr = np.asarray(span, dtype=float)
            ax.plot(arr[:, 0], arr[:, 1], color="magenta", lw=2.2, alpha=0.85)
            x0, x1 = float(arr[0, 0]), float(arr[-1, 0])
            ax.plot([x0, x1], [a[1], a[1]], color="magenta", lw=0.7, ls="--", alpha=0.8)
            for xt in (x0, x1):
                ax.plot([xt, xt], [a[1] - 9, a[1] + 9], color="magenta", lw=0.9)
            ax.annotate(
                "{:.3g} nm".format((dip.get("window") or {}).get("requested_nm", 0.0)),
                ((x0 + x1) / 2.0, a[1] - 11),
                ha="center",
                va="bottom",
                color="magenta",
                fontsize=6.5,
                weight="bold",
            )
        p = _pt(dip.get("max_downward_point"))
        if p is None:
            continue
        ax.annotate(
            "",
            xy=(p[0], p[1]),
            xytext=(p[0], a[1]),
            arrowprops=dict(arrowstyle="->", color="magenta", lw=1.1),
        )
        ax.annotate(
            "{:+.3f} nm".format((dip.get("max_downward_deviation") or {}).get("nm", 0.0)),
            (p[0], p[1]),
            textcoords="offset points",
            xytext=(0, -13),
            ha="center",
            va="top",
            color="magenta",
            fontsize=6.5,
        )

    ax.text(
        0.995,
        0.005,
        _legend_text(record),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="white",
        family="monospace",
        bbox=dict(facecolor="black", alpha=0.55, pad=4, edgecolor="none"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=140, facecolor="black")
    plt.close(fig)


def _fmt(value: Any, spec: str = "{:.3f}") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def _legend_text(record: Dict[str, Any]) -> str:
    g = lambda p: dig(record, p)  # noqa: E731
    rows = [
        "scale            {} nm/px (FOV {} nm)".format(
            _fmt(g("scale_nm_per_px"), "{:.5f}"), _fmt(g("fov_nm"), "{:.2f}")
        ),
        "leveling angle   {} deg".format(_fmt(g("results.leveling.angle_deg_image"))),
        "MgO_C angle/R2   {} deg / {}".format(
            _fmt(g("results.mgo_c.angle_deg_image")), _fmt(g("results.mgo_c.fit_r2"), "{:.4f}")
        ),
        "a1-b1 dx,dy      {}, {} nm".format(
            _fmt(g("results.interfaces.a1_b1.dx.nm")),
            _fmt(g("results.interfaces.a1_b1.dy.nm")),
        ),
        "a1-b3 dx,dy      {}, {} nm".format(
            _fmt(g("results.interfaces.a1_b3.dx.nm")),
            _fmt(g("results.interfaces.a1_b3.dy.nm")),
        ),
        "a2-b2 dx,dy      {}, {} nm".format(
            _fmt(g("results.interfaces.a2_b2.dx.nm")),
            _fmt(g("results.interfaces.a2_b2.dy.nm")),
        ),
        "a2-b4 dx,dy      {}, {} nm".format(
            _fmt(g("results.interfaces.a2_b4.dx.nm")),
            _fmt(g("results.interfaces.a2_b4.dy.nm")),
        ),
        "L dip / R dip    {} / {} nm".format(
            _fmt(g("results.interfaces.saf_ru_l_dip.max_downward_deviation.nm")),
            _fmt(g("results.interfaces.saf_ru_r_dip.max_downward_deviation.nm")),
        ),
        "m1 R spl/circle  {} / {} nm".format(
            _fmt(g("results.milling_l.radius.nm"), "{:.2f}"),
            _fmt(g("results.milling_l.local_circle.radius.nm"), "{:.2f}"),
        ),
        "m2 R spl/circle  {} / {} nm".format(
            _fmt(g("results.milling_r.radius.nm"), "{:.2f}"),
            _fmt(g("results.milling_r.local_circle.radius.nm"), "{:.2f}"),
        ),
        "Non_mag angle    {} deg (rmse {} nm)".format(
            _fmt(g("results.non_mag.angle_deg_image")),
            _fmt(g("results.non_mag.rmse.nm"), "{:.4f}"),
        ),
        "Nm1-m1 dx,dy     {}, {} nm".format(
            _fmt(g("results.corner_offsets.non_mag1_m1.dx.nm")),
            _fmt(g("results.corner_offsets.non_mag1_m1.dy.nm")),
        ),
        "Nm2-m2 dx,dy     {}, {} nm".format(
            _fmt(g("results.corner_offsets.non_mag2_m2.dx.nm")),
            _fmt(g("results.corner_offsets.non_mag2_m2.dy.nm")),
        ),
    ]
    return "\n".join(rows)
