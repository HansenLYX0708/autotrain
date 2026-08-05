"""Geometric analysis of PaddleSeg pseudo-color masks of TEM cross-sections.

Reads an 8-bit paletted label map (PaddleSeg's `pseudo_color_prediction/`),
optionally rectifies it so the Leveling layer is horizontal, then runs one module
per measurement and writes JSON + an annotated overlay PNG, plus a summary CSV in
batch mode.

Measurements (see measures/ for the algorithm notes):
    leveling        Leveling skeleton -> line -> tilt angle (drives the rectification)
    interfaces      a1/a2/b1/b2 endpoints, their dx/dy, and the 5 nm inward dip
    mgo_c           MgO_C fitted line, fit R2, angle
    milling_l/r     milled (vacuum-facing) flank, spline curvature, m1/m2
    non_mag         Non_mag edge away from Block_D, Non_mag1/Non_mag2
    corner_offsets  dx/dy for Non_mag1<->m1 and Non_mag2<->m2

Usage:
    python analyze.py --mask <mask.png> --out <dir>
    python analyze.py --mask-dir <pseudo_color_prediction> --out <dir> --csv summary.csv
    python analyze.py --mask <mask.png> --out <dir> --no-rotate
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import labelmap  # noqa: E402
import report  # noqa: E402
from context import AnalysisContext, MissingClass  # noqa: E402
from measures import MEASURES, leveling  # noqa: E402
from rectify import Rectifier  # noqa: E402

VERSION = "1.0.0"
MASK_EXTS = {".png", ".bmp", ".tif", ".tiff"}
RAW_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mask", type=Path, help="a single segmentation PNG")
    src.add_argument(
        "--mask-dir", type=Path, help="directory of segmentation PNGs (batch mode)"
    )
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument(
        "--scale-nm",
        type=float,
        default=0.0755,
        help="nm per pixel (default 0.0755). Note 76.37 nm / 1024 px would be "
             "0.0746, so double-check the calibration for your FOV",
    )
    p.add_argument(
        "--classes",
        help="comma separated class names, in class-id order; '_background_' is "
             "prepended automatically. Defaults to a sibling class_names.txt, then "
             "to the built-in TEM list",
    )
    p.add_argument(
        "--no-rotate",
        action="store_true",
        help="do not rectify with Leveling; measure on the raw mask",
    )
    p.add_argument(
        "--endpoint-extend",
        choices=("none", "region"),
        default="none",
        help="thinning retracts a skeleton from the tip of its region by about half "
             "the band width. 'region' extrapolates a1/a2 back out to the region "
             "tip along the local tangent (default: none, faithful Guo-Hall)",
    )
    p.add_argument(
        "--window-nm",
        type=float,
        default=5.0,
        help="arclength window for the inward dip measurement (default 5 nm)",
    )
    p.add_argument(
        "--min-area",
        type=int,
        default=30,
        help="drop connected components smaller than this many pixels (default 30)",
    )
    p.add_argument(
        "--keep-all-components",
        action="store_true",
        help="keep every component of a class instead of only the largest",
    )
    p.add_argument(
        "--thinning-backend",
        choices=("auto", "cv2", "numpy"),
        default="auto",
        help="Guo-Hall implementation (default auto: cv2.ximgproc, else numpy)",
    )
    p.add_argument(
        "--milling-edge-method",
        choices=("outer", "inner", "skeleton"),
        default="outer",
        help="which flank of a Milling band to measure. 'outer' (default) is the "
             "milled, vacuum-facing surface: boundary pixels touching no other "
             "class. 'inner' is the flank against the stack. 'skeleton' keeps the "
             "positive side of the skeleton normal (see measures/milling.py -- it "
             "flips partway along the band on this geometry)",
    )
    p.add_argument(
        "--milling-frame-margin-px",
        type=int,
        default=4,
        help="discard Milling boundary pixels within this many pixels of the image "
             "frame or the rotation padding: the crop is not a real surface and its "
             "corner would otherwise win the curvature maximum (default 4)",
    )
    p.add_argument(
        "--milling-smooth",
        type=float,
        default=1.0,
        help="assumed edge noise in pixels; sets the spline smoothing (default 1.0)",
    )
    p.add_argument(
        "--milling-middle-frac",
        type=float,
        default=1.0,
        help="central fraction of the edge arclength eligible for the curvature "
             "maximum. Default 1.0 (no restriction): the straight tails have "
             "kappa ~ 0 anyway, and the bend is not at the arclength midpoint",
    )
    p.add_argument(
        "--milling-edge-margin-nm",
        type=float,
        default=2.0,
        help="arclength excluded at both ends of the edge (spline end artifacts)",
    )
    p.add_argument(
        "--milling-circle-window-frac",
        type=float,
        default=0.8,
        help="half-window of the local circle cross-check, as a fraction of the "
             "spline radius at the bend (default 0.8; floor 12 px)",
    )
    p.add_argument(
        "--milling-tail-frac",
        type=float,
        default=0.25,
        help="fraction of arclength at each end fitted with a line, as a check on "
             "the line-curve-line shape (default 0.25)",
    )
    p.add_argument(
        "--milling-tangent-half",
        type=int,
        default=6,
        help="half-window (px) of the local PCA used for skeleton tangents",
    )
    p.add_argument(
        "--nonmag-edge",
        choices=("auto", "top", "bottom"),
        default="auto",
        help="which horizontal Non_mag edge to fit. 'auto' (default) picks the one "
             "not adjacent to Block_D, by counting contacts on both",
    )
    p.add_argument(
        "--nonmag-sigma",
        type=float,
        default=2.0,
        help="sigma-clipping threshold for the Non_mag edge fit",
    )
    p.add_argument(
        "--no-nonmag-trim",
        dest="nonmag_trim",
        action="store_false",
        help="disable sigma-clipping in the Non_mag fit",
    )
    p.add_argument("--csv", help="summary CSV filename, written inside --out")
    p.add_argument("--no-overlay", action="store_true", help="skip the overlay PNG")
    p.add_argument(
        "--overlay-on",
        type=Path,
        help="directory of source images; draw the overlay on the matching raw "
             "image (matched by filename stem) instead of the color mask",
    )
    p.set_defaults(nonmag_trim=True)
    return p


def _raw_background(ctx: AnalysisContext, stem: str, folder: Path) -> Optional[np.ndarray]:
    """The source image for this mask, rectified with the same transform.

    The stem is matched against image files only: annotation folders routinely hold
    a sidecar `<stem>.json` next to `<stem>.jpg`, and matching on the stem alone
    would hand a JSON file to the image loader.
    """
    for cand in sorted(folder.iterdir()):
        if not (cand.is_file() and cand.stem == stem):
            continue
        if cand.suffix.lower() not in RAW_EXTS:
            continue
        arr = np.array(Image.open(str(cand)).convert("RGB"))
        if tuple(arr.shape[:2]) != tuple(ctx.rect.in_shape):
            return None
        if not ctx.rect.applied:
            return arr
        return np.stack([ctx.rect.apply(arr[:, :, c]) for c in range(3)], axis=-1)
    return None


def analyze_one(path: Path, params: argparse.Namespace) -> Dict[str, Any]:
    classes = labelmap.resolve_classes(params.classes, path)
    ids, note = labelmap.load_label_map(path, len(classes))
    warnings: List[str] = []

    pre: Dict[str, Any] = {}
    rect = Rectifier.identity(ids.shape)
    try:
        pre["leveling"] = leveling.fit_leveling(ids, classes, params)
        if not params.no_rotate:
            rect = Rectifier.from_angle(ids.shape, pre["leveling"]["angle_deg_image"])
    except MissingClass as exc:
        warnings.append("rectification skipped: {}".format(exc))

    ctx = AnalysisContext(
        ids=rect.apply(ids),
        classes=classes,
        scale_nm=params.scale_nm,
        params=params,
        rect=rect,
        warnings=warnings,
        pre=pre,
    )

    record: Dict[str, Any] = {
        "image": path.name,
        "image_path": str(path),
        "image_size_px": [int(ids.shape[1]), int(ids.shape[0])],
        "mask_note": note,
        "classes": classes,
        "scale_nm_per_px": float(params.scale_nm),
        "tool_version": VERSION,
        "params": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in sorted(vars(params).items())
            if k not in ("mask", "mask_dir", "out")
        },
        "rotation": {
            "applied": rect.applied,
            "angle_deg_image": rect.angle_deg,
            "matrix": rect.matrix(),
            "output_size_px": [int(ctx.ids.shape[1]), int(ctx.ids.shape[0])],
        },
        "results": {},
        "warnings": warnings,
    }

    for name, fn in MEASURES.items():
        try:
            # Published to ctx.results as we go, so a later measurement can combine
            # earlier ones (corner_offsets needs non_mag plus both milling results).
            record["results"][name] = ctx.results[name] = fn(ctx)
        except MissingClass as exc:
            record["results"][name] = None
            ctx.warn("{}: {}".format(name, exc))
        except Exception as exc:  # keep the batch alive, but say what broke
            record["results"][name] = None
            ctx.warn("{}: {}: {}".format(name, type(exc).__name__, exc))
            traceback.print_exc(file=sys.stderr)

    record["_ctx"] = ctx
    return record


def main(argv: Optional[List[str]] = None) -> int:
    params = build_parser().parse_args(argv)
    if params.mask_dir:
        if not params.mask_dir.is_dir():
            raise SystemExit("--mask-dir is not a directory: {}".format(params.mask_dir))
        masks = sorted(
            p
            for p in params.mask_dir.iterdir()
            if p.is_file() and p.suffix.lower() in MASK_EXTS
        )
        if not masks:
            raise SystemExit("no mask images in {}".format(params.mask_dir))
    else:
        if not params.mask.is_file():
            raise SystemExit("--mask is not a file: {}".format(params.mask))
        masks = [params.mask]

    params.out.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    failures = 0

    for i, path in enumerate(masks, 1):
        print("[{}/{}] {}".format(i, len(masks), path.name), flush=True)
        try:
            record = analyze_one(path, params)
        except Exception as exc:
            failures += 1
            print("    FAILED: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            continue
        ctx = record.pop("_ctx")
        stem = path.stem
        if not params.no_overlay:
            try:
                bg = None
                if params.overlay_on and params.overlay_on.is_dir():
                    bg = _raw_background(ctx, stem, params.overlay_on)
                    if bg is None:
                        ctx.warn(
                            "no matching raw image of the same size for "
                            "--overlay-on; drew on the mask instead"
                        )
                report.render_overlay(
                    ctx, record, params.out / "{}_overlay.png".format(stem), bg
                )
            except Exception as exc:
                ctx.warn("overlay failed: {}: {}".format(type(exc).__name__, exc))
                traceback.print_exc(file=sys.stderr)
        # Written last so that overlay warnings make it into the record.
        report.write_json(record, params.out / "{}.json".format(stem))
        for w in record["warnings"]:
            print("    ! {}".format(w))
        records.append(record)

    if params.csv:
        csv_path = params.out / params.csv
        report.write_csv(records, csv_path)
        print("wrote {} ({} rows)".format(csv_path, len(records)))

    print(
        "done: {} analysed, {} failed, output in {}".format(
            len(records), failures, params.out
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
