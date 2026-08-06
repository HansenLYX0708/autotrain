"""Regression checks against synthetic geometry with known ground truth.

Run as a script; no pytest needed (this tools/ tree has no test framework).

    python tools/tem_analysis/selftest.py

The end-to-end case builds an ideal scene (horizontal layers, a circular Milling
arc of known radius, a flat Block_D), rotates the label array by a known angle with
nearest-neighbour interpolation, and then asks the tool to recover the original
numbers. That exercises thinning, pruning, rectification, the inverse coordinate
transform, spline curvature and every measurement in one shot.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze  # noqa: E402
import labelmap  # noqa: E402
from contour import longest_run, trace_outer_contour  # noqa: E402
from geometry import Spline, fit_line_ols, fit_line_tls  # noqa: E402
from rectify import Rectifier  # noqa: E402
from skeleton import guo_hall  # noqa: E402

TILT_DEG = 3.0
# The Milling annuli span [ARC_RADIUS, ARC_RADIUS + MILL_THICK): the inner flank is a
# circle of radius ARC_RADIUS (shared with MgO_L/MgO_R just inside it) and the outer,
# vacuum-facing flank -- the one measured by default -- is one pixel short of the
# outer limit. Both radii are exact, so either edge method can be checked.
ARC_RADIUS = 200.0
MILL_THICK = 16.0
ARC_OUTER = ARC_RADIUS + MILL_THICK - 1.0
SCALE_NM = 0.0755

_failures: List[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print("  {} {}{}".format("PASS" if ok else "FAIL", name, "  " + detail if detail else ""))
    if not ok:
        _failures.append(name)


def close(name: str, got: float, want: float, tol: float, unit: str = "") -> None:
    ok = got is not None and np.isfinite(got) and abs(got - want) <= tol
    check(
        name,
        bool(ok),
        "got {:.4f}{} want {:.4f} +/- {:.4f}".format(
            float("nan") if got is None else got, unit, want, tol
        ),
    )


# --------------------------------------------------------------------------- #
# Unit-level checks
# --------------------------------------------------------------------------- #

def test_guo_hall_backends() -> None:
    print("guo_hall: cv2.ximgproc vs the numpy fallback")
    if importlib.util.find_spec("cv2") is None:
        check("cv2 available", False, "skipped: opencv-contrib not installed")
        return
    rng = np.random.default_rng(0)
    shapes = []
    band = np.zeros((200, 300), dtype=bool)
    band[90:105, 20:280] = True
    shapes.append(("straight band", band))
    elbow = np.zeros((200, 300), dtype=bool)
    elbow[150:165, 20:200] = True
    elbow[40:165, 185:200] = True
    shapes.append(("elbow band", elbow))
    blob = rng.random((160, 160)) > 0.55
    from scipy import ndimage

    blob = ndimage.binary_closing(blob, np.ones((7, 7), bool))
    shapes.append(("random blob", blob))
    for label, mask in shapes:
        a = guo_hall(mask, backend="cv2")
        b = guo_hall(mask, backend="numpy")
        diff = int((a != b).sum())
        check(
            "  {} identical".format(label),
            diff == 0,
            "{} differing pixels ({} skeleton px)".format(diff, int(a.sum())),
        )


def test_rotation_roundtrip() -> None:
    print("rectify: forward/inverse coordinate transform")
    rect = Rectifier.from_angle((512, 640), 7.25)
    pts = np.array([[0.0, 0.0], [639.0, 0.0], [320.0, 256.0], [10.0, 500.0]])
    back = rect.to_orig(rect.to_rect(pts))
    err = float(np.abs(back - pts).max())
    close("  roundtrip error", err, 0.0, 1e-9, " px")
    # A direction at the fitted angle must come out horizontal.
    d = np.array([math.cos(math.radians(7.25)), math.sin(math.radians(7.25))])
    moved = rect.to_rect(np.array([[0.0, 0.0], d * 100.0]))
    ang = math.degrees(math.atan2(*(moved[1] - moved[0])[::-1]))
    close("  rectified direction", ang, 0.0, 1e-6, " deg")


def test_line_fit() -> None:
    print("geometry: line fit on an exact line")
    x = np.arange(0.0, 500.0)
    for want in (0.0, 7.5, -30.0, 80.0):
        y = 100.0 + math.tan(math.radians(want)) * x
        line = fit_line_tls(np.column_stack([x, y]))
        close("  tls angle {:>5.1f} deg".format(want), line.angle_deg_image, want, 1e-6)
    ols = fit_line_ols(np.column_stack([x, 100.0 + 0.5 * x]))
    close("  r2 on an exact line", ols["r2"], 1.0, 1e-9)
    check("  r2 not flagged degenerate", not ols["r2_degenerate"])
    flat = fit_line_ols(np.column_stack([x, 100.0 + np.sin(x / 7.0)]))
    check(
        "  r2 flagged degenerate on a flat wavy line",
        flat["r2_degenerate"],
        "r2={:.4f}".format(flat["r2"]),
    )


def test_spline_curvature() -> None:
    print("geometry: analytic curvature of a circle")
    for radius in (50.0, 200.0, 600.0):
        t = np.linspace(0.35, 2.1, 900)
        pts = np.column_stack([radius * np.cos(t), radius * np.sin(t)])
        spline = Spline.fit(pts, smooth=0.0)
        u, _, s = spline.sample(1200)
        mid = (s > 0.15 * s[-1]) & (s < 0.85 * s[-1])
        got = 1.0 / abs(float(np.median(spline.curvature(u)[mid])))
        close("  radius {:>5.0f} px".format(radius), got, radius, 0.01 * radius, " px")


def test_contour() -> None:
    print("contour: tracing and run selection")
    mask = np.zeros((60, 80), dtype=bool)
    mask[10:40, 20:70] = True
    pts = trace_outer_contour(mask)
    # A filled rectangle's 8-connected boundary ring is 2*(w-1) + 2*(h-1) pixels.
    want = 2 * (50 - 1) + 2 * (30 - 1)
    check("  rectangle ring length", len(pts) == want, "got {} want {}".format(len(pts), want))
    inside = mask[np.round(pts[:, 1]).astype(int), np.round(pts[:, 0]).astype(int)]
    check("  all contour pixels are foreground", bool(inside.all()))
    # Runs are [0,1], [3,4] and [6,7]; wrapping merges the last with the first into
    # a run of 4, which must beat both interior runs of 2.
    keep = np.array([1, 1, 0, 1, 1, 0, 1, 1], dtype=bool)
    run = longest_run(keep, closed=True)
    check(
        "  wrap-around run",
        sorted(run.tolist()) == [0, 1, 6, 7],
        "got {}".format(sorted(run.tolist())),
    )
    check(
        "  open run",
        sorted(longest_run(keep, closed=False).tolist()) == [0, 1],
        "got {}".format(sorted(longest_run(keep, closed=False).tolist())),
    )


# --------------------------------------------------------------------------- #
# End-to-end check on a synthetic scene
# --------------------------------------------------------------------------- #

# Ideal (pre-tilt) geometry. Layer y-ranges are half-open [lo, hi).
#
# The Milling arcs reproduce the real topology rather than a bare annulus sector:
#   - MgO_L / MgO_R span a wider angle than the Milling band and reach the same outer
#     radius, so they wrap around the Milling band's radial end caps. Without that,
#     the caps face background, count as "outer", and their 90 deg corners are the
#     sharpest thing on the edge -- the curvature maximum lands on a cap instead of
#     the arc.
#   - the left arc runs off the left border, so the frame-cut exclusion is exercised;
#     the right arc stays inside, so the cap wrapping is exercised.
IDEAL = {
    "size": (1200, 1024),
    "leveling_y": (1100, 1110),
    "saf_l": ((400, 413), (0, 431)),  # (y range, x range) -> a1 = (430, 406)
    "saf_r": ((400, 413), (660, 1024)),  # a2 = (660, 406)
    "mgo_c": ((398, 409), (470, 641)),  # b1 = (470, 403), b2 = (640, 403)
    # Same stacking order as the real masks: Non_mag sits directly under Block_D and
    # touches it, so the Non_mag edge "not adjacent to Block_D" is the bottom one,
    # y = 590, spanning x 464..646.
    "block_d": ((480, 541), (460, 651)),
    "non_mag": ((541, 591), (464, 647)),
    "arc_l": ((150.0, 640.0), (90.0, 180.0)),  # (centre, theta range in degrees)
    "arc_r": ((700.0, 640.0), (0.0, 90.0)),
}
MGO_WRAP_DEG = 10.0


def _fill(ids: np.ndarray, name: str, yr, xr) -> None:
    ids[yr[0] : yr[1], xr[0] : xr[1]] = labelmap.DEFAULT_CLASSES.index(name)


def _annulus(
    ids: np.ndarray, name: str, centre, r0: float, r1: float, t0: float, t1: float
) -> None:
    """Annulus sector; theta is measured in image coordinates, so y grows downward."""
    h, w = ids.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx = xx - centre[0]
    dy = yy - centre[1]
    r = np.hypot(dx, dy)
    theta = np.degrees(np.arctan2(dy, dx))
    sel = (r >= r0) & (r < r1) & (theta >= t0) & (theta <= t1)
    ids[sel] = labelmap.DEFAULT_CLASSES.index(name)


def build_scene() -> np.ndarray:
    h, w = IDEAL["size"]
    ids = np.zeros((h, w), dtype=np.uint8)
    _fill(ids, "SAF_Ru_L", *IDEAL["saf_l"])
    _fill(ids, "SAF_Ru_R", *IDEAL["saf_r"])
    _fill(ids, "MgO_C", *IDEAL["mgo_c"])
    _fill(ids, "Block_D", *IDEAL["block_d"])
    _fill(ids, "Non_mag", *IDEAL["non_mag"])
    _fill(ids, "Leveling", IDEAL["leveling_y"], (0, w))
    for tag, key in (("L", "arc_l"), ("R", "arc_r")):
        centre, (t0, t1) = IDEAL[key]
        _annulus(
            ids,
            "MgO_" + tag,
            centre,
            ARC_RADIUS - MILL_THICK,
            ARC_RADIUS + MILL_THICK,
            t0 - MGO_WRAP_DEG,
            t1 + MGO_WRAP_DEG,
        )
        _annulus(
            ids, "Milling_" + tag, centre, ARC_RADIUS, ARC_RADIUS + MILL_THICK, t0, t1
        )
    return ids


def write_scene(path: Path, tilt_deg: float) -> np.ndarray:
    ids = build_scene()
    if tilt_deg:
        # The padding is deliberately left as IGNORE rather than flattened to
        # background. That mirrors reality: a band cut by the field of view keeps a
        # straight artificial edge, and the only thing marking it as artificial is the
        # region outside the original frame. Flattening it to background hides that,
        # and the curvature maximum then lands on the crop corner.
        ids = Rectifier.from_angle(ids.shape, -tilt_deg).apply(ids)
    labelmap._annotate_module().save_mask(
        path, ids, labelmap.palette(len(labelmap.DEFAULT_CLASSES))
    )
    return ids


def run_pipeline(path: Path, extra: List[str]) -> Dict[str, Any]:
    argv = [
        "--mask", str(path),
        "--out", str(path.parent),
        "--scale-nm", str(SCALE_NM),
        "--no-overlay",
    ] + extra
    params = analyze.build_parser().parse_args(argv)
    record = analyze.analyze_one(path, params)
    record.pop("_ctx", None)
    return record


def test_untilted_absolute() -> None:
    """Absolute coordinates, on the untilted scene with rectification off.

    Rotation grows the canvas, so a tilted-then-rectified point differs from the
    ideal one by a constant translation. Absolute positions are therefore checked
    here, where the canvas is untouched, and the tilted case below checks only
    canvas-invariant quantities.
    """
    print("end-to-end: untilted scene, --no-rotate (absolute positions)")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "flat.png"
        write_scene(path, 0.0)
        rec = run_pipeline(path, ["--no-rotate", "--endpoint-extend", "region"])
        res = rec["results"]
        ctx_w = rec["image_size_px"][0]
        for w in rec["warnings"]:
            print("      (warning) {}".format(w))

        it = res["interfaces"]
        close("  a1", it["a1"]["x_px"], 430.0, 1.5, " px")
        close("  a1 y", it["a1"]["y_px"], 406.0, 1.0, " px")
        close("  a2", it["a2"]["x_px"], 660.0, 1.5, " px")
        close("  b1", it["b1"]["x_px"], 470.0, 1.5, " px")
        close("  b1 y", it["b1"]["y_px"], 403.0, 1.0, " px")
        close("  b2", it["b2"]["x_px"], 640.0, 1.5, " px")
        close("  a1->b1 dx", it["a1_b1"]["dx"]["px"], 40.0, 1.5, " px")
        close("  a1->b1 dy", it["a1_b1"]["dy"]["px"], -3.0, 1.0, " px")
        close("  a2->b2 dx", it["a2_b2"]["dx"]["px"], -20.0, 1.5, " px")
        close("  a2->b2 dy", it["a2_b2"]["dy"]["px"], -3.0, 1.0, " px")
        # MgO_C is a perfectly straight rectangle here, so the fitted segment ends
        # (b3/b4) must coincide with the raw skeleton tips (b1/b2), and the offsets
        # measured against them must match.
        close("  b3 x", it["b3"]["x_px"], 470.0, 1.5, " px")
        close("  b3 y", it["b3"]["y_px"], 403.0, 1.0, " px")
        close("  b4 x", it["b4"]["x_px"], 640.0, 1.5, " px")
        close("  b4 y", it["b4"]["y_px"], 403.0, 1.0, " px")
        close("  a1->b3 dx", it["a1_b3"]["dx"]["px"], 40.0, 1.5, " px")
        close("  a1->b3 dy", it["a1_b3"]["dy"]["px"], -3.0, 1.0, " px")
        close("  a2->b4 dx", it["a2_b4"]["dx"]["px"], -20.0, 1.5, " px")
        close("  a2->b4 dy", it["a2_b4"]["dy"]["px"], -3.0, 1.0, " px")
        close(
            "  flat layer has no dip",
            it["saf_ru_l_dip"]["max_downward_deviation"]["px"],
            0.0,
            0.5,
            " px",
        )
        check(
            "  point coordinates are unmapped without rotation",
            it["a1"]["x_px"] == it["a1"]["x_orig_px"],
        )

        close("  leveling angle", res["leveling"]["angle_deg_image"], 0.0, 0.01, " deg")
        close("  MgO_C angle", res["mgo_c"]["angle_deg_image"], 0.0, 0.01, " deg")
        close("  MgO_C rmse", res["mgo_c"]["rmse"]["px"], 0.0, 0.05, " px")
        close("  MgO_C fit length", res["mgo_c"]["fit_length"]["px"], 170.0, 2.0, " px")

        nmg = res["non_mag"]
        # Block_D is directly above Non_mag, so the auto rule must take the bottom edge.
        check(
            "  Non_mag picked the edge away from Block_D",
            nmg["edge_side"] == "bottom",
            "side={} contacts={}".format(nmg["edge_side"], nmg["block_d_contact_columns"]),
        )
        close("  Non_mag1 x", nmg["Non_mag1"]["x_px"], 464.0, 1.5, " px")
        close("  Non_mag1 y", nmg["Non_mag1"]["y_px"], 590.0, 1.0, " px")
        close("  Non_mag2 x", nmg["Non_mag2"]["x_px"], 646.0, 1.5, " px")
        close("  Non_mag angle", nmg["angle_deg_image"], 0.0, 0.02, " deg")

        # The spline curvature locates the bend but underestimates the radius; the
        # local circle fit is the accurate one. Both bounds are asserted so a
        # regression in either estimator is caught.
        for key, name in (("milling_l", "m1"), ("milling_r", "m2")):
            node = res[key]
            check(
                "  {} measures the outer flank".format(name),
                node["edge_method"] == "outer",
                node["edge_method"],
            )
            close(
                "  {} radius (spline)".format(name),
                node["radius"]["px"],
                ARC_OUTER,
                0.20 * ARC_OUTER,
                " px",
            )
            close(
                "  {} radius (local circle)".format(name),
                node["local_circle"]["radius"]["px"],
                ARC_OUTER,
                0.08 * ARC_OUTER,
                " px",
            )
            check(
                "  {} circle fit is reliable".format(name),
                node["local_circle"]["reliable"],
                "rms {:.2f} px".format(node["local_circle"]["rms"]["px"]),
            )
            # This synthetic edge is a plain arc with no straight tails, so the tails
            # cannot be checked for straightness (the real masks do have them). What
            # is checked is that both tails were extracted and that they come from
            # opposite ends, i.e. their chord directions differ appreciably.
            spread = abs(
                node["tail_right"]["angle_deg_image"]
                - node["tail_left"]["angle_deg_image"]
            )
            check(
                "  {} tails come from opposite ends".format(name),
                15.0 < spread < 100.0,
                "chord directions differ by {:.1f} deg".format(spread),
            )
            check(
                "  {} edge avoids the image frame".format(name),
                min(x for x, _ in node["edge_polyline"]) > 0.5
                and max(x for x, _ in node["edge_polyline"]) < ctx_w - 1.5,
                "x range {:.0f}..{:.0f}".format(
                    min(x for x, _ in node["edge_polyline"]),
                    max(x for x, _ in node["edge_polyline"]),
                ),
            )

        # corner_offsets combines two other measurements, so what is checked is that it
        # reports exactly the difference between the two points it names. The absolute
        # value cannot be predicted: on a constant-curvature arc the location of the
        # curvature maximum is set by pixel noise, not by geometry.
        for key, corner, milling in (
            ("non_mag1_m1", "Non_mag1", "milling_l"),
            ("non_mag2_m2", "Non_mag2", "milling_r"),
        ):
            off = res["corner_offsets"][key]
            want_dx = res[milling]["max_curvature_point"]["x_px"] - nmg[corner]["x_px"]
            want_dy = res[milling]["max_curvature_point"]["y_px"] - nmg[corner]["y_px"]
            close("  {} dx".format(key), off["dx"]["px"], want_dx, 1e-9, " px")
            close("  {} dy".format(key), off["dy"]["px"], want_dy, 1e-9, " px")
            close(
                "  {} nm scaling".format(key),
                off["dx"]["nm"],
                want_dx * SCALE_NM,
                1e-9,
                " nm",
            )

        # Side selection is checked on the edge geometry, not on its curvature: the
        # distance from the known arc centre to the selected polyline says directly
        # which flank was taken. (Curvature cannot be used for the inner flank here --
        # MgO wraps the band's end caps, so the `inner` run includes those caps and
        # their corners, which is a genuine limitation of the method on a band whose
        # ends are capped by the same neighbouring class.)
        inner = run_pipeline(path, ["--no-rotate", "--milling-edge-method", "inner"])
        for key, name, arc_key in (
            ("milling_l", "m1", "arc_l"),
            ("milling_r", "m2", "arc_r"),
        ):
            centre = np.array(IDEAL[arc_key][0])
            for tag, record, want in (
                ("outer", res, ARC_OUTER),
                ("inner", inner["results"], ARC_RADIUS),
            ):
                poly = np.asarray(record[key]["edge_polyline"], dtype=float)
                got = float(np.median(np.linalg.norm(poly - centre, axis=1)))
                close(
                    "  {} {} flank distance from arc centre".format(name, tag),
                    got,
                    want,
                    3.0,
                    " px",
                )

        # Without the correction, thinning must retract the tip inward (smaller x).
        plain = run_pipeline(path, ["--no-rotate"])
        retraction = 430.0 - plain["results"]["interfaces"]["a1"]["x_px"]
        check(
            "  endpoint retraction without --endpoint-extend",
            3.0 < retraction < 12.0,
            "{:.1f} px inward (SAF_Ru_L half-width is 6.5 px)".format(retraction),
        )


def test_tilted_invariants() -> None:
    print("end-to-end: scene tilted by {:.1f} deg, rectified".format(TILT_DEG))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tilted.png"
        write_scene(path, TILT_DEG)
        rec = run_pipeline(path, ["--endpoint-extend", "region"])
        res = rec["results"]
        for w in rec["warnings"]:
            print("      (warning) {}".format(w))

        lev = res["leveling"]
        close("  recovered tilt", lev["angle_deg_image"], TILT_DEG, 0.05, " deg")
        close(
            "  residual tilt after rectification",
            lev["residual_angle_deg_after_rotation"],
            0.0,
            0.05,
            " deg",
        )
        # Two nearest-neighbour rotations leave the layers a couple of pixels ragged,
        # so these tolerances are resampling noise, not measurement error.
        it = res["interfaces"]
        close("  a1->b1 dx", it["a1_b1"]["dx"]["px"], 40.0, 3.0, " px")
        close("  a1->b1 dy", it["a1_b1"]["dy"]["px"], -3.0, 6.0, " px")
        close("  a2->b2 dx", it["a2_b2"]["dx"]["px"], -20.0, 3.0, " px")
        close("  a2->b2 dy", it["a2_b2"]["dy"]["px"], -3.0, 6.0, " px")
        close("  MgO_C angle", res["mgo_c"]["angle_deg_image"], 0.0, 0.35, " deg")
        close("  MgO_C rmse", res["mgo_c"]["rmse"]["px"], 0.0, 1.0, " px")
        close("  Non_mag angle", res["non_mag"]["angle_deg_image"], 0.0, 0.2, " deg")
        for key, name in (("milling_l", "m1"), ("milling_r", "m2")):
            close(
                "  {} radius (local circle)".format(name),
                res[key]["local_circle"]["radius"]["px"],
                ARC_OUTER,
                0.10 * ARC_OUTER,
                " px",
            )

        # --no-rotate must leave every angle tilted by the scene tilt.
        raw = run_pipeline(path, ["--no-rotate", "--endpoint-extend", "region"])
        close(
            "  --no-rotate MgO_C angle",
            raw["results"]["mgo_c"]["angle_deg_image"],
            TILT_DEG,
            0.35,
            " deg",
        )
        check(
            "  --no-rotate leaves the map untouched",
            raw["rotation"]["applied"] is False,
        )


def test_scale_from_filename() -> None:
    print("scale: field of view parsed out of the filename")
    cases = [
        ("751845_RMF 24(12,15)_1300kx_FOV_76.37nm.png", 76.37),
        ("a_FOV_120nm.png", 120.0),
        ("a_fov-8.5 nm.tif", 8.5),
        ("no_scale_here.png", None),
        ("1300kx_only.png", None),
    ]
    for name, want in cases:
        got = labelmap.parse_fov_nm(name)
        check(
            "  {:44s} -> {}".format(name, want),
            got == want,
            "got {}".format(got),
        )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic_1300kx_FOV_76.37nm.png"
        write_scene(path, 0.0)
        params = analyze.build_parser().parse_args(
            ["--mask", str(path), "--out", tmp, "--no-overlay", "--no-rotate"]
        )
        rec = analyze.analyze_one(path, params)
        rec.pop("_ctx", None)
        width = rec["image_width_px"]
        close("  FOV", rec["fov_nm"], 76.37, 1e-9, " nm")
        close("  nm/px", rec["scale_nm_per_px"], 76.37 / width, 1e-12)
        check("  source", rec["scale_source"] == "filename", rec["scale_source"])
        # A distance in nm must be exactly the pixel distance times the scale.
        blk = rec["results"]["non_mag"]
        close(
            "  nm conversion",
            blk["length"]["nm"],
            blk["length"]["px"] * 76.37 / width,
            1e-9,
            " nm",
        )


def test_decoy_component() -> None:
    """A spurious Non_mag band larger than the real box must not win.

    Regression test for a real prediction in the sample batch: a Non_mag band across
    the bottom of the frame, bigger than the actual Non_mag box, so "keep the largest
    component" measured the artefact and reported a 34 nm edge instead of ~10 nm.
    The component is now anchored on Block_D.
    """
    print("robustness: a decoy Non_mag component bigger than the real one")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "decoy.png"
        ids = build_scene()
        real = int((ids == labelmap.DEFAULT_CLASSES.index("Non_mag")).sum())
        _fill(ids, "Non_mag", (950, 1010), (50, 1000))  # nowhere near Block_D
        decoy = int((ids == labelmap.DEFAULT_CLASSES.index("Non_mag")).sum()) - real
        check("  decoy is the larger blob", decoy > real, "{} vs {} px".format(decoy, real))
        labelmap._annotate_module().save_mask(
            path, ids, labelmap.palette(len(labelmap.DEFAULT_CLASSES))
        )
        res = run_pipeline(path, ["--no-rotate"])["results"]
        close("  Non_mag1 x", res["non_mag"]["Non_mag1"]["x_px"], 464.0, 1.5, " px")
        close("  Non_mag1 y", res["non_mag"]["Non_mag1"]["y_px"], 590.0, 1.0, " px")
        close("  Non_mag2 x", res["non_mag"]["Non_mag2"]["x_px"], 646.0, 1.5, " px")
        check("  edge side", res["non_mag"]["edge_side"] == "bottom", res["non_mag"]["edge_side"])


def test_numpy_backend_end_to_end() -> None:
    print("end-to-end: the numpy thinning fallback gives the same answers")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.png"
        write_scene(path, TILT_DEG)
        a = run_pipeline(path, ["--thinning-backend", "cv2"])
        b = run_pipeline(path, ["--thinning-backend", "numpy"])
        for key in ("leveling", "mgo_c", "non_mag"):
            close(
                "  {} angle matches".format(key),
                b["results"][key]["angle_deg_image"],
                a["results"][key]["angle_deg_image"],
                1e-9,
                " deg",
            )


TESTS: List[Callable[[], None]] = [
    test_guo_hall_backends,
    test_rotation_roundtrip,
    test_line_fit,
    test_spline_curvature,
    test_contour,
    test_untilted_absolute,
    test_tilted_invariants,
    test_scale_from_filename,
    test_decoy_component,
    test_numpy_backend_end_to_end,
]


def main() -> int:
    for fn in TESTS:
        fn()
        print()
    if _failures:
        print("{} FAILED: {}".format(len(_failures), ", ".join(_failures)))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
