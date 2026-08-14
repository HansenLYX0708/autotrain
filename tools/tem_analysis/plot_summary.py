"""Plot a batch summary.csv produced by analyze.py.

`analyze.py` deliberately emits numbers only: a fixed-column CSV is the thing you
can diff between batches, and a plot is not. This script is the other half --
it turns one summary.csv (or several, for a batch-to-batch comparison) into the
figures you actually look at when triaging a run.

Four figures, each answering a different question:

    distributions   "what is the spread, and which images are outliers?"
                    One panel per unit-compatible group of metrics, box + jitter,
                    with outliers marked and labelled by image index.

    trend           "is anything drifting across the batch?"
                    Metric against image order, with a median line and a robust
                    +/-3.5 MAD band. Acquisition order is usually stage order, so
                    a monotonic drift here is a real signal, not noise. With
                    several CSVs, batch boundaries are drawn and labelled so a
                    step between runs is not mistaken for drift within one.

    symmetry        "are the left and right sides consistent?"
                    Left against right for the pairs physics says should mirror,
                    with the identity diagonal. A per-image asymmetry shows up
                    here and in no single-metric box plot.

    quality         "which images should I open the overlay for?"
                    Warnings per image, the most common warning texts, the
                    local_circle reliability flags and the Non_mag edge-side
                    tally -- the things that decide whether a row is trustworthy.

`--slide-figure` adds a fifth: a 16:9, four-panel, large-type version for a deck.
The full distributions figure has eight panels and 7 pt annotations, which is
unreadable once it is scaled into a slide.

Outliers use the median/MAD rule, not mean/sigma: with 26 rows a single bad
prediction moves the mean and inflates sigma enough to hide itself. The modified
z-score is 0.6745*(x - median)/MAD and |z| > 3.5 is the usual threshold
(Iglewicz & Hoaglin). MAD == 0 (a metric that is constant) disables the test
rather than flagging every row that differs by one ulp.

Usage:
    python plot_summary.py --csv <analysis>/summary.csv
    python plot_summary.py --csv <analysis>/summary.csv --out <dir> --stats-csv
    python plot_summary.py --csv runA/summary.csv --csv runB/summary.csv --labels A,B
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Agg before pyplot: this runs headless in the same environments analyze.py does.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Every label here is ASCII on purpose. matplotlib's default font stack has no
# CJK glyphs, and a report full of tofu boxes is worse than English axis labels.
MAD_TO_SIGMA = 0.6745
OUTLIER_Z = 3.5

# `vert=False` was deprecated in matplotlib 3.11 in favour of `orientation`, which
# older versions do not accept. Pick once at import so the script is quiet on both.
_HORIZONTAL = (
    {"orientation": "horizontal"}
    if tuple(int(p) for p in matplotlib.__version__.split(".")[:2]) >= (3, 9)
    else {"vert": False}
)

FG = "#1a1d21"
GRID = "#d8dde3"
ACCENT = "#2b7fd4"
BAD = "#d1493f"
OK = "#3f9c78"


class Metric:
    """One plotted quantity: a CSV column plus how to present it."""

    def __init__(self, column: str, label: str, unit: str, invert: bool = False):
        self.column = column
        self.label = label
        self.unit = unit
        # `invert` mirrors a right-hand metric so it can share a panel with its
        # left-hand twin: a2_b2.dx is about -5.3 nm where a1_b1.dx is +5.3 nm, and
        # plotting them on one axis without the flip wastes the whole panel on the
        # gap between two clusters that are supposed to be compared.
        self.invert = invert

    def values(self, rows: Sequence[Dict[str, str]]) -> np.ndarray:
        out = []
        for r in rows:
            v = to_float(r.get(self.column))
            out.append(-v if (self.invert and v is not None) else v)
        return np.array([np.nan if v is None else v for v in out], dtype=float)


# Panels are grouped by unit so a shared x axis is meaningful.
GROUPS: List[Tuple[str, List[Metric]]] = [
    ("Interface offsets, lateral (nm)", [
        Metric("results.interfaces.a1_b1.dx.nm", "a1-b1 dx", "nm"),
        Metric("results.interfaces.a1_b3.dx.nm", "a1-b3 dx", "nm"),
        Metric("results.interfaces.a2_b2.dx.nm", "a2-b2 dx (flipped)", "nm", invert=True),
        Metric("results.interfaces.a2_b4.dx.nm", "a2-b4 dx (flipped)", "nm", invert=True),
    ]),
    ("Interface offsets, vertical (nm)", [
        Metric("results.interfaces.a1_b1.dy.nm", "a1-b1 dy", "nm"),
        Metric("results.interfaces.a1_b3.dy.nm", "a1-b3 dy", "nm"),
        Metric("results.interfaces.a2_b2.dy.nm", "a2-b2 dy", "nm"),
        Metric("results.interfaces.a2_b4.dy.nm", "a2-b4 dy", "nm"),
    ]),
    ("Endpoint dip within 5 nm (nm)", [
        Metric("results.interfaces.saf_ru_l_dip.max_downward_deviation.nm", "left dip", "nm"),
        Metric("results.interfaces.saf_ru_r_dip.max_downward_deviation.nm", "right dip", "nm"),
    ]),
    ("Milling bend radius (nm)", [
        Metric("results.milling_l.local_circle.radius.nm", "m1 circle", "nm"),
        Metric("results.milling_l.radius.nm", "m1 spline", "nm"),
        Metric("results.milling_r.local_circle.radius.nm", "m2 circle", "nm"),
        Metric("results.milling_r.radius.nm", "m2 spline", "nm"),
    ]),
    ("Corner offsets (nm)", [
        Metric("results.corner_offsets.non_mag1_m1.dx.nm", "Nm1-m1 dx", "nm"),
        Metric("results.corner_offsets.non_mag2_m2.dx.nm", "Nm2-m2 dx (flipped)", "nm", invert=True),
        Metric("results.corner_offsets.non_mag1_m1.dy.nm", "Nm1-m1 dy", "nm"),
        Metric("results.corner_offsets.non_mag2_m2.dy.nm", "Nm2-m2 dy", "nm"),
    ]),
    ("Lengths (nm)", [
        Metric("results.non_mag.length.nm", "Non_mag edge", "nm"),
        Metric("results.mgo_c.fit_length.nm", "MgO_C fit", "nm"),
    ]),
    ("Angles (deg, image frame)", [
        Metric("results.leveling.angle_deg_image", "Leveling tilt", "deg"),
        Metric("results.mgo_c.angle_deg_image", "MgO_C tilt", "deg"),
        Metric("results.non_mag.angle_deg_image", "Non_mag tilt", "deg"),
    ]),
    # R2 is degenerate for near-horizontal layers, so straightness is judged by
    # rmse here -- same rule the JSON's fit_r2_note states.
    ("Fit rmse (nm) -- judge straightness by this, not R2", [
        Metric("results.leveling.rmse.nm", "Leveling", "nm"),
        Metric("results.mgo_c.rmse.nm", "MgO_C", "nm"),
        Metric("results.non_mag.rmse.nm", "Non_mag", "nm"),
    ]),
]

# The handful worth watching against acquisition order.
TREND: List[Metric] = [
    Metric("results.interfaces.a1_b1.dx.nm", "a1-b1 dx", "nm"),
    Metric("results.interfaces.a2_b2.dx.nm", "a2-b2 dx (flipped)", "nm", invert=True),
    Metric("results.interfaces.saf_ru_l_dip.max_downward_deviation.nm", "left 5 nm dip", "nm"),
    Metric("results.milling_l.local_circle.radius.nm", "m1 circle radius", "nm"),
    Metric("results.milling_r.local_circle.radius.nm", "m2 circle radius", "nm"),
    Metric("results.non_mag.length.nm", "Non_mag edge length", "nm"),
    Metric("results.leveling.angle_deg_image", "Leveling tilt", "deg"),
    Metric("results.mgo_c.angle_deg_image", "MgO_C tilt", "deg"),
]

# Left/right pairs that physics says should mirror each other. A point far off
# the diagonal is a per-image asymmetry, which no single-metric box plot shows.
SYMMETRY: List[Tuple[str, Metric, Metric]] = [
    ("Interface dx", Metric("results.interfaces.a1_b1.dx.nm", "left  a1-b1 dx", "nm"),
     Metric("results.interfaces.a2_b2.dx.nm", "right  a2-b2 dx (flipped)", "nm", invert=True)),
    ("Bend radius", Metric("results.milling_l.local_circle.radius.nm", "left  m1 circle", "nm"),
     Metric("results.milling_r.local_circle.radius.nm", "right  m2 circle", "nm")),
    ("Corner dx", Metric("results.corner_offsets.non_mag1_m1.dx.nm", "left  Nm1-m1 dx", "nm"),
     Metric("results.corner_offsets.non_mag2_m2.dx.nm", "right  Nm2-m2 dx (flipped)", "nm", invert=True)),
]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def to_float(text: Any) -> Optional[float]:
    """CSV cell -> float, or None. Blank means the measurement failed."""
    if text is None:
        return None
    s = str(text).strip()
    if not s or s.lower() in ("none", "nan", "null"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def read_summary(path: Path) -> List[Dict[str, str]]:
    # utf-8-sig: write_csv emits a BOM so Excel opens it correctly, and without
    # this the first column name would come back as "\ufeffimage".
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("{}: no data rows".format(path))
    if "image" not in rows[0]:
        raise SystemExit(
            "{}: no 'image' column -- is this a summary.csv from analyze.py?".format(path)
        )
    return rows


def robust_stats(v: np.ndarray) -> Dict[str, float]:
    ok = v[np.isfinite(v)]
    if ok.size == 0:
        return {"n": 0, "median": np.nan, "mad": np.nan, "sd": np.nan,
                "min": np.nan, "max": np.nan}
    med = float(np.median(ok))
    return {
        "n": int(ok.size),
        "median": med,
        "mad": float(np.median(np.abs(ok - med))),
        "sd": float(np.std(ok)),
        "min": float(ok.min()),
        "max": float(ok.max()),
    }


def outlier_mask(v: np.ndarray) -> np.ndarray:
    """Modified z-score > 3.5. A zero MAD disables the test (see module docstring)."""
    m = np.zeros(v.shape, dtype=bool)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return m
    med = np.median(v[ok])
    mad = np.median(np.abs(v[ok] - med))
    if mad <= 0:
        return m
    z = np.zeros_like(v)
    z[ok] = MAD_TO_SIGMA * (v[ok] - med) / mad
    m[ok] = np.abs(z[ok]) > OUTLIER_Z
    return m


def short_name(image: str, width: int = 28) -> str:
    stem = re.sub(r"\.(png|bmp|tiff?|jpe?g)$", "", str(image), flags=re.I)
    return stem if len(stem) <= width else stem[: width - 1] + "\u2026"


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #

def style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#aab2bd",
        "axes.labelcolor": FG,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": FG,
        "ytick.color": FG,
        "text.color": FG,
        "grid.color": GRID,
        "legend.fontsize": 8,
        "font.family": "DejaVu Sans",
    })


def jitter(n: int, spread: float, seed: int) -> np.ndarray:
    # Deterministic: the same CSV must produce a byte-identical figure, otherwise
    # two runs of this script look like two different batches.
    return np.random.default_rng(seed).uniform(-spread, spread, n)


def box_panel(ax, metrics: List[Metric], rows: Sequence[Dict[str, str]], title: str,
              stat_fontsize: float = 7.2, dot_size: float = 11):
    present = [(m, m.values(rows)) for m in metrics]
    present = [(m, v) for m, v in present if np.isfinite(v).any()]
    if not present:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no data", ha="center", va="center", color="#96a0ac")
        return

    data = [v[np.isfinite(v)] for _, v in present]
    pos = list(range(len(present), 0, -1))  # first metric on top
    bp = ax.boxplot(data, positions=pos, widths=0.55, showfliers=False,
                    patch_artist=True, medianprops=dict(color=BAD, lw=1.6),
                    **_HORIZONTAL)
    for patch in bp["boxes"]:
        patch.set(facecolor="#e8eef6", edgecolor="#7f8b9a", lw=0.9)
    for key in ("whiskers", "caps"):
        for art in bp[key]:
            art.set(color="#7f8b9a", lw=0.9)

    for k, (m, v) in enumerate(present):
        y = pos[k]
        ok = np.isfinite(v)
        bad = outlier_mask(v)
        yj = y + jitter(int(ok.sum()), 0.13, seed=abs(hash(m.column)) % (2 ** 31))
        ax.scatter(v[ok], yj, s=dot_size, color=ACCENT, alpha=.75, zorder=3, linewidths=0)
        if bad.any():
            # Re-jitter with the same generator so flagged points keep their offset.
            yb = y + jitter(int(ok.sum()), 0.13, seed=abs(hash(m.column)) % (2 ** 31))[bad[ok]]
            ax.scatter(v[bad], yb, s=dot_size * 3, facecolors="none", edgecolors=BAD,
                       linewidths=1.3, zorder=4)
        st = robust_stats(v)
        ax.text(1.005, y, "  {:.3f}  ({:.3f})  n={}".format(st["median"], st["sd"], st["n"]),
                transform=ax.get_yaxis_transform(), va="center", fontsize=stat_fontsize,
                family="monospace", color="#55606d")

    ax.set_yticks(pos)
    ax.set_yticklabels([m.label for m, _ in present])
    ax.set_xlabel(present[0][0].unit)
    ax.set_title(title, loc="left")
    ax.grid(axis="x", lw=.6, alpha=.8)
    ax.set_axisbelow(True)
    ax.margins(x=.08)


def trend_panel(ax, m: Metric, rows: Sequence[Dict[str, str]], show_x: bool):
    v = m.values(rows)
    x = np.arange(len(v))
    ok = np.isfinite(v)
    if not ok.any():
        ax.set_axis_off()
        return
    st = robust_stats(v)
    band = st["mad"] * OUTLIER_Z / MAD_TO_SIGMA if st["mad"] > 0 else 0.0
    if band > 0:
        ax.axhspan(st["median"] - band, st["median"] + band, color="#eef3f9", zorder=0)
    ax.axhline(st["median"], color="#9aa4b2", lw=.9, ls="--", zorder=1)
    ax.plot(x[ok], v[ok], "-", color=ACCENT, lw=1.0, alpha=.85, zorder=2)
    ax.plot(x[ok], v[ok], "o", ms=3.4, color=ACCENT, zorder=3)
    bad = outlier_mask(v)
    for i in np.nonzero(bad)[0]:
        ax.plot(i, v[i], "o", ms=7, mfc="none", mec=BAD, mew=1.4, zorder=4)
        ax.annotate("#{}".format(i + 1), (i, v[i]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7, color=BAD)
    # Missing values are a result too: a gap here means that measurement failed.
    for i in np.nonzero(~ok)[0]:
        ax.axvline(i, color="#e0b24c", lw=1.0, alpha=.7, zorder=1)
    ax.set_ylabel("{}\n[{}]".format(m.label, m.unit), fontsize=8)
    ax.grid(axis="y", lw=.6, alpha=.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, len(v) - 0.4)
    if show_x:
        ax.set_xlabel("image index (CSV order)")
    else:
        # tick_params, not set_xticklabels([]): with sharex=True the latter clears
        # the labels on the shared axis, so every panel including the bottom one
        # loses its numbers.
        ax.tick_params(labelbottom=False)


def quality_figure(rows: Sequence[Dict[str, str]], out: Path, title: str) -> Path:
    fig = plt.figure(figsize=(13, 7.2), dpi=140)
    # Two gridspecs rather than one: the bottom row's y labels are whole warning
    # sentences and need a wide left margin, while the top row would waste that
    # space. A single gridspec can only have one `left`.
    gs_top = fig.add_gridspec(2, 1, height_ratios=[1, 1.25], hspace=.42,
                              left=.06, right=.98, top=.90, bottom=.09)
    gs_bot = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=.42, wspace=.30,
                              left=.20, right=.98, top=.90, bottom=.09)

    # (a) warnings per image
    ax = fig.add_subplot(gs_top[0, 0])
    nw = np.array([to_float(r.get("n_warnings")) or 0 for r in rows])
    colors = [BAD if v > 0 else "#c3ccd6" for v in nw]
    ax.bar(np.arange(len(nw)), nw, color=colors, width=.72)
    ax.set_title("(a) warnings per image  --  open the overlay for every red bar", loc="left")
    ax.set_xlabel("image index (CSV order)")
    ax.set_ylabel("count")
    ax.set_xlim(-0.6, len(nw) - 0.4)
    ax.grid(axis="y", lw=.6, alpha=.8)
    ax.set_axisbelow(True)
    if nw.max() <= 0:
        ax.text(.5, .5, "no warnings in this batch", transform=ax.transAxes,
                ha="center", va="center", color=OK, fontsize=11, weight="bold")
    else:
        ax.set_yticks(np.arange(0, nw.max() + 1))

    # (b) most common warning texts, collapsed to their stable prefix
    ax = fig.add_subplot(gs_bot[1, 0])
    counter: Counter = Counter()
    for r in rows:
        for w in (r.get("warnings") or "").split(" | "):
            w = w.strip()
            if not w:
                continue
            # Numbers inside a warning differ per image ("dropped 6 components,
            # largest 4786 px"), so bucket by the text with numbers masked.
            counter[re.sub(r"\d+(\.\d+)?", "#", w)] += 1
    if counter:
        items = counter.most_common(8)[::-1]
        # The informative part of these warnings is the prefix (class name + what
        # happened); the tail is boilerplate. Truncate, then wrap to two lines.
        labels = [textwrap(clip(k, 58), 30) for k, _ in items]
        ax.barh(np.arange(len(items)), [c for _, c in items], color="#e0b24c", height=.66)
        ax.set_yticks(np.arange(len(items)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("images affected")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(axis="x", lw=.6, alpha=.8)
        ax.set_axisbelow(True)
    else:
        ax.set_axis_off()
        ax.text(.5, .5, "no warnings", ha="center", va="center", color=OK, weight="bold")
    ax.set_title("(b) warning types (digits masked)", loc="left")

    # (c) categorical flags that decide whether a row is usable
    ax = fig.add_subplot(gs_bot[1, 1])
    cats: List[Tuple[str, Counter]] = []
    for col, name in (("results.milling_l.local_circle.reliable", "m1 circle reliable"),
                      ("results.milling_r.local_circle.reliable", "m2 circle reliable"),
                      ("results.non_mag.edge_side", "Non_mag edge side"),
                      ("scale_source", "scale source")):
        c = Counter((r.get(col) or "-").strip() for r in rows)
        if c and set(c) != {"-"}:
            cats.append((name, c))
    if cats:
        y, ticks, labels = 0, [], []
        palette = {"True": OK, "true": OK, "False": BAD, "false": BAD}
        for name, c in cats:
            left = 0.0
            for key, cnt in sorted(c.items()):
                ax.barh(y, cnt, left=left, height=.62,
                        color=palette.get(key, ACCENT), edgecolor="white", lw=.8)
                ax.text(left + cnt / 2, y, "{} ({})".format(key, cnt), ha="center",
                        va="center", fontsize=7.2, color="white", weight="bold")
                left += cnt
            ticks.append(y)
            labels.append(name)
            y += 1
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("images")
        ax.grid(axis="x", lw=.6, alpha=.8)
        ax.set_axisbelow(True)
    else:
        ax.set_axis_off()
    ax.set_title("(c) per-image flags", loc="left")

    fig.suptitle(title, x=.06, ha="left", fontsize=13, weight="bold")
    return save(fig, out)


def clip(s: str, width: int) -> str:
    return s if len(s) <= width else s[: width - 1].rstrip() + "\u2026"


def textwrap(s: str, width: int) -> str:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines[:3]) + ("\u2026" if len(lines) > 3 else "")


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), facecolor="white")
    plt.close(fig)
    print("wrote {}".format(path))
    return path


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def distributions_figure(rows, out: Path, title: str) -> Path:
    groups = [(t, ms) for t, ms in GROUPS
              if any(np.isfinite(m.values(rows)).any() for m in ms)]
    ncol = 2
    nrow = math.ceil(len(groups) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.05 * nrow + 1.1), dpi=140)
    axes = np.atleast_1d(axes).ravel()
    for ax, (t, ms) in zip(axes, groups):
        box_panel(ax, ms, rows, t)
    for ax in axes[len(groups):]:
        ax.set_axis_off()
    fig.suptitle(title, x=.055, ha="left", fontsize=13, weight="bold")
    fig.text(.055, .012,
             "box = quartiles, red line = median, dots = images, red ring = "
             "outlier (modified z > {:.1f}).  Right of each row: median (sd) n."
             .format(OUTLIER_Z), fontsize=7.8, color="#55606d")
    fig.tight_layout(rect=(0, .028, 1, .965))
    return save(fig, out)


# A slide is 16:9 and gets looked at from three metres away. The full
# `distributions` figure has eight panels and 7 pt annotations; scaled into a
# deck it is unreadable, so the presentation version is authored separately:
# four panels, larger type, and the group that matters most first.
SLIDE_GROUPS = ("Interface offsets, lateral (nm)", "Endpoint dip within 5 nm (nm)",
                "Milling bend radius (nm)", "Angles (deg, image frame)")


def slide_figure(rows, out: Path, title: str) -> Path:
    groups = [(t, ms) for t, ms in GROUPS if t in SLIDE_GROUPS
              and any(np.isfinite(m.values(rows)).any() for m in ms)]
    if not groups:
        return out
    with plt.rc_context({"axes.titlesize": 13, "axes.labelsize": 11,
                         "xtick.labelsize": 11, "ytick.labelsize": 11.5}):
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 6.0), dpi=150)
        axes = axes.ravel()
        for ax, (t, ms) in zip(axes, groups):
            box_panel(ax, ms, rows, t, stat_fontsize=9.5, dot_size=22)
        for ax in axes[len(groups):]:
            ax.set_axis_off()
        fig.suptitle(title, x=.045, ha="left", fontsize=16, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, .945))
    return save(fig, out)


def batch_bounds(rows) -> List[Tuple[int, int, str]]:
    """(start, end, label) for each run of consecutive rows from the same CSV."""
    out: List[Tuple[int, int, str]] = []
    for i, r in enumerate(rows):
        label = r.get("_batch", "")
        if out and out[-1][2] == label:
            out[-1] = (out[-1][0], i, label)
        else:
            out.append((i, i, label))
    return out


def trend_figure(rows, out: Path, title: str) -> Path:
    ms = [m for m in TREND if np.isfinite(m.values(rows)).any()]
    if not ms:
        raise SystemExit("no plottable trend metrics in this CSV")
    fig, axes = plt.subplots(len(ms), 1, figsize=(13, 1.28 * len(ms) + 1.2),
                             dpi=140, sharex=True)
    axes = np.atleast_1d(axes)
    for k, m in enumerate(ms):
        trend_panel(axes[k], m, rows, show_x=(k == len(ms) - 1))
    # With more than one CSV the rows are concatenated, so mark where each batch
    # starts -- otherwise a step at a batch boundary reads as a drift.
    spans = batch_bounds(rows)
    if len(spans) > 1:
        for ax in axes:
            for start, _, _ in spans[1:]:
                ax.axvline(start - 0.5, color="#5b6674", lw=1.2, ls="-", alpha=.8)
        for start, end, label in spans:
            axes[0].annotate(label, ((start + end) / 2.0, 1.02), xycoords=("data", "axes fraction"),
                             ha="center", va="bottom", fontsize=8.5, weight="bold",
                             color="#55606d")
    fig.suptitle(title, x=.055, ha="left", fontsize=13, weight="bold")
    fig.text(.055, .012,
             "dashed = median, shaded = robust +/-{:.1f} MAD band, red ring = "
             "outlier, amber line = measurement missing on that image."
             .format(OUTLIER_Z), fontsize=7.8, color="#55606d")
    fig.tight_layout(rect=(0, .03, 1, .965))
    return save(fig, out)


def symmetry_figure(rows, out: Path, title: str) -> Path:
    pairs = [(t, a, b) for t, a, b in SYMMETRY
             if np.isfinite(a.values(rows)).any() and np.isfinite(b.values(rows)).any()]
    if not pairs:
        return out
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.3 * len(pairs), 4.9), dpi=140)
    axes = np.atleast_1d(axes)
    for ax, (t, a, b) in zip(axes, pairs):
        x, y = a.values(rows), b.values(rows)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=26, color=ACCENT, alpha=.8, linewidths=0)
        lo = float(min(x[ok].min(), y[ok].min()))
        hi = float(max(x[ok].max(), y[ok].max()))
        pad = .08 * (hi - lo or 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#9aa4b2", lw=1, ls="--")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        d = y[ok] - x[ok]
        ax.set_title("{}\nmean L-R gap {:+.3f} {}".format(t, -float(d.mean()), a.unit),
                     loc="left", fontsize=9.5)
        ax.set_xlabel("{} [{}]".format(a.label, a.unit))
        ax.set_ylabel("{} [{}]".format(b.label, b.unit))
        ax.grid(lw=.6, alpha=.8)
        ax.set_axisbelow(True)
    fig.suptitle(title, x=.03, ha="left", fontsize=13, weight="bold")
    fig.text(.03, .018,
             "dashed = perfect left/right symmetry; a point off the diagonal is a "
             "per-image asymmetry.", fontsize=7.8, color="#55606d")
    fig.tight_layout(rect=(0, .065, 1, .935))
    return save(fig, out)


def write_stats_csv(rows, path: Path) -> None:
    """The same table the report quotes, so nobody re-types medians by hand."""
    metrics = [m for _, ms in GROUPS for m in ms]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "column", "unit", "n", "median", "sd", "mad",
                    "min", "max", "n_outliers", "outlier_images"])
        for m in metrics:
            v = m.values(rows)
            st = robust_stats(v)
            if st["n"] == 0:
                continue
            bad = outlier_mask(v)
            w.writerow([
                m.label, m.column + (" (flipped)" if m.invert else ""), m.unit,
                st["n"], round(st["median"], 6), round(st["sd"], 6),
                round(st["mad"], 6), round(st["min"], 6), round(st["max"], 6),
                int(bad.sum()),
                " | ".join(short_name(rows[i].get("image", "?"), 60)
                           for i in np.nonzero(bad)[0]),
            ])
    print("wrote {}".format(path))


def print_stats(rows) -> None:
    print("\n{:<26} {:>4} {:>10} {:>9} {:>10} {:>10} {:>4}".format(
        "metric", "n", "median", "sd", "min", "max", "out"))
    print("-" * 78)
    for title, ms in GROUPS:
        shown = False
        for m in ms:
            v = m.values(rows)
            st = robust_stats(v)
            if st["n"] == 0:
                continue
            if not shown:
                print("[{}]".format(title))
                shown = True
            print("{:<26} {:>4d} {:>10.3f} {:>9.3f} {:>10.3f} {:>10.3f} {:>4d}".format(
                m.label[:26], st["n"], st["median"], st["sd"], st["min"], st["max"],
                int(outlier_mask(v).sum())))


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, action="append", required=True,
                   help="a summary.csv from analyze.py; repeat to compare batches")
    p.add_argument("--labels",
                   help="comma separated names for the CSVs (default: parent folder)")
    p.add_argument("--out", type=Path,
                   help="output directory (default: next to the first CSV)")
    p.add_argument("--prefix", default="summary",
                   help="output filename prefix (default: summary)")
    p.add_argument("--stats-csv", action="store_true",
                   help="also write <prefix>_stats.csv with median/sd/outliers")
    p.add_argument("--no-trend", action="store_true", help="skip the trend figure")
    p.add_argument("--no-symmetry", action="store_true", help="skip the symmetry figure")
    p.add_argument("--slide-figure", action="store_true",
                   help="also write <prefix>_slide.png: a 16:9, four-panel, "
                        "large-type version for a presentation")
    p.add_argument("--dpi", type=int, default=140)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    style()
    plt.rcParams["figure.dpi"] = a.dpi

    csvs: List[Path] = a.csv
    for c in csvs:
        if not c.is_file():
            raise SystemExit("not a file: {}".format(c))
    labels = ([s.strip() for s in a.labels.split(",")] if a.labels
              else [c.parent.name or c.stem for c in csvs])
    if len(labels) != len(csvs):
        raise SystemExit("--labels has {} entries but {} CSVs were given".format(
            len(labels), len(csvs)))

    out = a.out or csvs[0].parent
    out.mkdir(parents=True, exist_ok=True)

    # Several CSVs are concatenated into one row list and tagged with `_batch`;
    # the trend figure draws a separator at each boundary so a step between runs
    # is not mistaken for drift within one.
    all_rows: List[Dict[str, str]] = []
    for path, label in zip(csvs, labels):
        rows = read_summary(path)
        for r in rows:
            r["_batch"] = label
        all_rows.extend(rows)
        print("{}: {} rows  ({})".format(label, len(rows), path))

    tag = labels[0] if len(csvs) == 1 else " + ".join(labels)
    n = len(all_rows)

    distributions_figure(all_rows, out / "{}_distributions.png".format(a.prefix),
                         "Batch distributions  --  {}  ({} images)".format(tag, n))
    if not a.no_trend:
        trend_figure(all_rows, out / "{}_trend.png".format(a.prefix),
                     "Across the batch  --  {}  ({} images)".format(tag, n))
    if not a.no_symmetry:
        symmetry_figure(all_rows, out / "{}_symmetry.png".format(a.prefix),
                        "Left / right symmetry  --  {}".format(tag))
    quality_figure(all_rows, out / "{}_quality.png".format(a.prefix),
                   "Run quality  --  {}  ({} images)".format(tag, n))
    if a.slide_figure:
        slide_figure(all_rows, out / "{}_slide.png".format(a.prefix),
                     "Batch distributions  --  {} images".format(n))
    if a.stats_csv:
        write_stats_csv(all_rows, out / "{}_stats.csv".format(a.prefix))
    print_stats(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
