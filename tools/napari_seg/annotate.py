"""napari-based segmentation annotator that writes PaddleSeg-ready datasets.

Why napari: the annotation target is a 2-D integer label array, so a pixel can
only ever belong to one class. Adjacent regions therefore share their boundary
by construction -- no gaps, no overlaps, no anti-aliased in-between pixels. That
is the problem polygon tools (labelme / X-AnyLabeling) cannot solve without
post-processing.

Output layout matches `src/app/api/datasets/labelme-to-paddleseg/route.ts` so the
result can be registered in the platform directly:

    <out>/
      JPEGImages/       source images, copied on first save, filename sanitized
      Annotations/      8-bit paletted PNG, palette index == class id
      train.txt         "JPEGImages/xxx.jpg Annotations/xxx.png" per line
      val.txt
      class_names.txt   first line is always "_background_"

The palette is the same VOC pseudo-color map as `src/lib/seg-colors.ts`
(`getSegColorMap`), which keeps the platform's dataset preview and pixel
statistics in sync with training.

Filenames are rewritten on the way in: anything outside ASCII [A-Za-z0-9._-]
becomes '_', because PaddleSeg splits list-file lines on whitespace, so a name
like "cam 1.jpg" would be read as two tokens and break training. `--print-names`
shows the mapping.

Usage:
    python annotate.py --images D:/data/raw --out D:/data/out/data \
        --classes partA,partB
    # resume later (class names are read back from class_names.txt)
    python annotate.py --images D:/data/raw --out D:/data/out/data
    # show how filenames are rewritten inside the dataset
    python annotate.py --images D:/data/raw --out D:/data/out/data --print-names --no-gui
    # pseudo-labelling: pull in model predictions, then review/fix them by hand
    python annotate.py --images D:/data/new_raw --out D:/data/out/data \
        --import-masks D:/pd/PaddleSeg/output/pseudo_color_prediction

Growing the dataset with model predictions:
    `paddleseg/core/predict.py` writes two directories. Only one is a label map:
      added_prediction/        RGB image blended with the mask -- a preview, NOT
                               a label; importing it is refused
      pseudo_color_prediction/ 8-bit paletted PNG, index == class id -- this one
    PaddleSeg builds that palette with `get_color_map_list()`, which drops the
    leading black entry exactly like `src/lib/seg-colors.ts` does, so prediction
    masks and masks painted here are byte-compatible.

Keys:
    Ctrl+Right / Ctrl+Left   next / previous image (auto-saves)
    Ctrl+Shift+S             save current mask
    Ctrl+= / Ctrl+-          zoom in / out       (mouse wheel also zooms)
    Ctrl+0                   fit image to window
    hold Space + drag        pan without leaving the paint tool
    [ / ]                    smaller / larger brush
    2 / 3 / 4 / 5            napari's own paint / fill / pick / erase tools
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

BACKGROUND = "_background_"
IGNORE_INDEX = 255
IGNORE_NAME = "__ignore__"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# How far an RGB pixel may sit from a palette color before it counts as "not a
# label map", and the share of such pixels that makes us reject the file.
RGB_SNAP_TOL = 30.0
RGB_SNAP_MAX_OFF = 0.05


# --------------------------------------------------------------------------- #
# Pure helpers (importable and testable without napari)
# --------------------------------------------------------------------------- #

def voc_colormap(num_classes: int) -> np.ndarray:
    """Return an (num_classes, 3) uint8 palette.

    Mirror of `getSegColorMap()` in src/lib/seg-colors.ts: the PASCAL VOC color
    map with the leading black entry dropped, so class 0 -> (128, 0, 0),
    class 1 -> (0, 128, 0), class 2 -> (128, 128, 0), ...
    """
    count = max(1, num_classes)
    n = count + 1
    cm = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        lab, j = i, 0
        while lab:
            cm[i, 0] |= ((lab >> 0) & 1) << (7 - j)
            cm[i, 1] |= ((lab >> 1) & 1) << (7 - j)
            cm[i, 2] |= ((lab >> 2) & 1) << (7 - j)
            j += 1
            lab >>= 3
    return cm[1:]


def sanitize(name: str) -> str:
    """Make a filename safe for PaddleSeg list files (whitespace-separated).

    Mirror of `sanitizeFilename()` in the labelme-to-paddleseg route: anything
    outside ASCII [A-Za-z0-9._-] becomes '_', runs collapse, edges are trimmed.
    """
    stem, ext = os.path.splitext(name)
    clean = "".join(
        c if (c.isascii() and (c.isalnum() or c in "._-")) else "_" for c in stem
    )
    while "__" in clean:
        clean = clean.replace("__", "_")
    clean = clean.strip("_")
    return "{}{}".format(clean or "image", ext)


def build_palette(num_classes: int) -> np.ndarray:
    """Full 256-entry palette: classes get VOC colors, 255 (ignore) gets white."""
    pal = np.zeros((256, 3), dtype=np.uint8)
    colors = voc_colormap(num_classes)
    pal[: len(colors)] = colors
    pal[IGNORE_INDEX] = (255, 255, 255)
    return pal


def save_mask(path: Path, labels: np.ndarray, palette: np.ndarray) -> None:
    """Write an 8-bit paletted PNG whose palette index equals the class id."""
    arr = np.ascontiguousarray(labels, dtype=np.uint8)
    # `Image.fromarray(..., mode="P")` is deprecated (removed in Pillow 13);
    # frombytes keeps the raw values as palette indices without any remapping.
    img = Image.frombytes("P", (arr.shape[1], arr.shape[0]), arr.tobytes())
    img.putpalette(palette.reshape(-1).tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), optimize=True)


def load_mask(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    """Read an existing mask, or return a zero (all-background) array."""
    if not path.exists():
        return np.zeros(shape, dtype=np.uint8)
    arr = np.array(Image.open(str(path)))
    if arr.ndim != 2:
        raise ValueError("{} is RGB; expected a paletted/grayscale mask".format(path))
    if arr.shape != shape:
        raise ValueError(
            "{} is {} but the image is {}".format(path, arr.shape, shape)
        )
    return arr.astype(np.uint8)


def read_image(path: Path) -> np.ndarray:
    im = Image.open(str(path))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    return np.array(im)


def list_images(images_dir: Path) -> List[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def resolve_class_names(out_dir: Path, classes_arg: Optional[str]) -> List[str]:
    """Class order defines the ids; index 0 is always the background class."""
    persisted = out_dir / "class_names.txt"
    if classes_arg:
        names = [n.strip() for n in classes_arg.split(",") if n.strip()]
    elif persisted.exists():
        names = [
            line.strip()
            for line in persisted.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raise SystemExit(
            "no classes: pass --classes a,b,c or point --out at a directory "
            "that already has class_names.txt"
        )
    if not names or names[0] != BACKGROUND:
        names.insert(0, BACKGROUND)
    if len(names) != len(set(names)):
        raise SystemExit("duplicate class names: {}".format(names))
    if len(names) > 255:
        raise SystemExit("at most 255 classes are supported (255 is ignore_index)")
    return names


def dataset_names(src: Path) -> Tuple[str, str]:
    """Return the (image, mask) names `src` gets inside the dataset."""
    safe = sanitize(src.name)
    return safe, Path(safe).stem + ".png"


def build_name_map(images: Sequence[Path]) -> Dict[Path, Tuple[str, str]]:
    """Map every source image to its (image, mask) name, keeping them unique.

    `sanitize()` is lossy: "工位1.png" and "产线1.png" both collapse to "1.png",
    and a fully non-ASCII name collapses to "image.png". Applying it blindly
    would make one image silently overwrite another, so colliding names get a
    numeric suffix. Uniqueness is decided over the whole (sorted) image list, so
    the result is stable as long as the source directory does not change.
    """
    used: Dict[str, Path] = {}
    mapping: Dict[Path, Tuple[str, str]] = {}
    for src in images:
        safe, _ = dataset_names(src)
        base, ext = os.path.splitext(safe)
        stem, n = base, 2
        while stem in used:
            stem = "{}_{}".format(base, n)
            n += 1
        used[stem] = src
        mapping[src] = (stem + ext, stem + ".png")
    return mapping


def rgb_to_indices(
    rgb: np.ndarray, palette: np.ndarray, valid_ids: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Snap every RGB pixel to the closest palette entry. Returns (ids, distance)."""
    a = rgb.astype(np.int32)
    best = np.full(a.shape[:2], np.iinfo(np.int32).max, dtype=np.int32)
    idx = np.zeros(a.shape[:2], dtype=np.uint8)
    for cid in valid_ids:
        d = ((a - palette[cid].astype(np.int32)) ** 2).sum(-1)
        m = d < best
        best[m] = d[m]
        idx[m] = cid
    return idx, np.sqrt(best.astype(np.float64))


def read_external_mask(
    path: Path,
    shape: Tuple[int, int],
    num_classes: int,
    palette: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """Read a mask produced elsewhere (e.g. PaddleSeg's pseudo_color_prediction).

    Paletted / grayscale PNGs are taken at face value: the stored value already is
    the class id, which is what `paddleseg/core/predict.py` writes. RGB files are
    snapped to the nearest palette color, which rescues hand-made color masks but
    rejects blended visualisations such as `added_prediction/`.
    """
    im = Image.open(str(path))
    valid = list(range(num_classes)) + [IGNORE_INDEX]
    note = im.mode

    if im.mode in ("P", "L"):
        arr = np.array(im).astype(np.uint8)
    elif im.mode in ("RGB", "RGBA"):
        arr, dist = rgb_to_indices(np.array(im.convert("RGB")), palette, valid)
        off = float((dist > RGB_SNAP_TOL).mean())
        if off > RGB_SNAP_MAX_OFF:
            raise ValueError(
                "{:.0%} of pixels are not PaddleSeg palette colors -- this looks "
                "like a blended preview (added_prediction/), not a label map; use "
                "pseudo_color_prediction/ instead".format(off)
            )
        note = "RGB snapped to palette ({:.2%} off-palette)".format(off)
    else:
        raise ValueError("unsupported image mode {}".format(im.mode))

    if arr.ndim != 2:
        raise ValueError("expected a 2-D mask, got shape {}".format(arr.shape))
    if arr.shape != shape:
        raise ValueError("mask is {} but the image is {}".format(arr.shape, shape))
    unknown = [int(v) for v in np.unique(arr) if v not in valid]
    if unknown:
        raise ValueError(
            "class ids {} are outside 0..{} (num_classes={})".format(
                unknown, num_classes - 1, num_classes
            )
        )
    return arr, note


def index_masks_by_stem(mask_dir: Path) -> Dict[str, Path]:
    """Map stem -> mask path, searching recursively.

    PaddleSeg keeps the input sub-directory structure under
    `pseudo_color_prediction/`, so a flat listing is not enough.
    """
    found: Dict[str, Path] = {}
    for path in sorted(mask_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            found.setdefault(path.stem, path)
    return found


def import_masks(
    images: Sequence[Path],
    mask_dir: Path,
    out_dir: Path,
    class_names: Sequence[str],
    overwrite: bool = False,
) -> Dict[str, List[str]]:
    """Fold externally produced masks (model predictions) into the dataset.

    Both the image and the mask are stored under their sanitized names, so a
    later `annotate.py` run picks them up for review.
    """
    palette = build_palette(len(class_names))
    available = index_masks_by_stem(mask_dir)
    names = build_name_map(images)
    report: Dict[str, List[str]] = {
        "imported": [], "skipped": [], "missing": [], "rejected": []
    }

    (out_dir / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (out_dir / "Annotations").mkdir(parents=True, exist_ok=True)

    for src in images:
        img_name, mask_name = names[src]
        dst_mask = out_dir / "Annotations" / mask_name
        source = available.get(src.stem)
        if source is None:
            report["missing"].append(src.name)
            continue
        if dst_mask.exists() and not overwrite:
            report["skipped"].append("{} (already annotated)".format(src.name))
            continue
        try:
            image = read_image(src)
            labels, note = read_external_mask(
                source, (image.shape[0], image.shape[1]), len(class_names), palette
            )
        except (ValueError, OSError) as exc:
            report["rejected"].append("{}: {}".format(src.name, exc))
            continue

        save_mask(dst_mask, labels, palette)
        dst_img = out_dir / "JPEGImages" / img_name
        if not dst_img.exists():
            shutil.copy2(str(src), str(dst_img))
        report["imported"].append("{} -> {} [{}]".format(src.name, mask_name, note))

    return report


def write_lists(
    out_dir: Path,
    images: Sequence[Path],
    train_ratio: float,
    seed: int,
) -> Tuple[int, int]:
    """Write train.txt / val.txt for every image that already has a mask."""
    annotations = out_dir / "Annotations"
    names = build_name_map(images)
    lines: List[str] = []
    for src in images:
        img_name, mask_name = names[src]
        if (annotations / mask_name).exists():
            lines.append("JPEGImages/{} Annotations/{}".format(img_name, mask_name))

    random.Random(seed).shuffle(lines)
    cut = int(len(lines) * train_ratio)
    if len(lines) > 1:
        cut = min(max(cut, 1), len(lines) - 1)  # never leave a split empty
    train, val = lines[:cut], lines[cut:]

    for name, rows in (("train.txt", train), ("val.txt", val)):
        text = "\n".join(rows)
        (out_dir / name).write_text(text + "\n" if text else "", encoding="utf-8")
    return len(train), len(val)


# --------------------------------------------------------------------------- #
# napari application
# --------------------------------------------------------------------------- #

class Annotator:
    def __init__(
        self,
        images_dir: Path,
        out_dir: Path,
        class_names: List[str],
        train_ratio: float = 0.8,
        seed: int = 0,
        brush_size: int = 12,
    ) -> None:
        import napari

        self.images = list_images(images_dir)
        if not self.images:
            raise SystemExit("no images found in {}".format(images_dir))

        self.out = out_dir
        self.names = build_name_map(self.images)
        self.class_names = class_names
        self.palette = build_palette(len(class_names))
        self.train_ratio = train_ratio
        self.seed = seed
        self.index = 0
        self._syncing = False
        self._shape: Optional[Tuple[int, int]] = None

        (self.out / "JPEGImages").mkdir(parents=True, exist_ok=True)
        (self.out / "Annotations").mkdir(parents=True, exist_ok=True)
        (self.out / "class_names.txt").write_text(
            "\n".join(self.class_names) + "\n", encoding="utf-8"
        )

        self.viewer = napari.Viewer(title="PaddleSeg annotator")
        self.image_layer = None
        self.labels_layer = None
        self._build_dock(brush_size)
        self._bind_keys()
        self.viewer.camera.events.zoom.connect(self._on_camera_zoom)
        self.goto(0, save=False)

    # -- paths ------------------------------------------------------------- #

    def image_dst(self, src: Path) -> Path:
        return self.out / "JPEGImages" / self.names[src][0]

    def mask_dst(self, src: Path) -> Path:
        return self.out / "Annotations" / self.names[src][1]

    # -- widgets ----------------------------------------------------------- #

    def _build_dock(self, brush_size: int) -> None:
        from magicgui.widgets import (
            CheckBox,
            ComboBox,
            Container,
            FloatSpinBox,
            Label,
            PushButton,
            SpinBox,
        )

        choices: List[Tuple[str, int]] = [
            ("{}: {}".format(i, n), i) for i, n in enumerate(self.class_names)
        ]
        choices.append(("{}: {}".format(IGNORE_INDEX, IGNORE_NAME), IGNORE_INDEX))

        self.w_progress = Label(value="")
        self.w_class = ComboBox(label="class", choices=choices, value=min(1, len(self.class_names) - 1))
        self.w_brush = SpinBox(label="brush", min=1, max=500, value=brush_size)
        self.w_contour = CheckBox(label="outline only", value=False)
        self.w_zoom = Label(value="zoom -")
        self.w_zoom_in = PushButton(text="zoom in  (Ctrl+=)")
        self.w_zoom_out = PushButton(text="zoom out  (Ctrl+-)")
        self.w_fit = PushButton(text="fit to window  (Ctrl+0)")
        self.w_keep_view = CheckBox(label="keep zoom when switching", value=True)
        self.w_prev = PushButton(text="< prev  (Ctrl+Left)")
        self.w_next = PushButton(text="next >  (Ctrl+Right)")
        self.w_save = PushButton(text="save  (Ctrl+Shift+S)")
        self.w_ratio = FloatSpinBox(label="train ratio", min=0.1, max=0.95, step=0.05, value=self.train_ratio)
        self.w_export = PushButton(text="write train.txt / val.txt")
        self.w_stats = PushButton(text="pixel stats of current mask")
        self.w_status = Label(value="")
        self.w_hints = Label(
            value=(
                "wheel = zoom   |   Space+drag = pan\n"
                "[ ] = brush size   |   2 paint  3 fill  4 pick  5 erase\n"
                "paint over a neighbour to move the shared edge"
            )
        )

        self.w_class.changed.connect(self._on_class_changed)
        self.w_brush.changed.connect(
            lambda v: setattr(self.labels_layer, "brush_size", int(v))
        )
        self.w_contour.changed.connect(
            lambda v: setattr(self.labels_layer, "contour", 1 if v else 0)
        )
        self.w_zoom_in.changed.connect(lambda _=None: self.zoom_by(1.25))
        self.w_zoom_out.changed.connect(lambda _=None: self.zoom_by(1 / 1.25))
        self.w_fit.changed.connect(lambda _=None: self.fit_view())
        self.w_prev.changed.connect(lambda _=None: self.goto(self.index - 1))
        self.w_next.changed.connect(lambda _=None: self.goto(self.index + 1))
        self.w_save.changed.connect(lambda _=None: self.save_current(verbose=True))
        self.w_export.changed.connect(lambda _=None: self.export())
        self.w_stats.changed.connect(lambda _=None: self.show_stats())

        container = Container(
            widgets=[
                self.w_progress,
                self.w_class,
                self.w_brush,
                self.w_contour,
                self.w_zoom,
                self.w_zoom_in,
                self.w_zoom_out,
                self.w_fit,
                self.w_keep_view,
                self.w_prev,
                self.w_next,
                self.w_save,
                self.w_ratio,
                self.w_export,
                self.w_stats,
                self.w_status,
                self.w_hints,
            ],
            labels=True,
        )
        self.viewer.window.add_dock_widget(container, area="right", name="annotate")

    def _bind_keys(self) -> None:
        bindings = {
            "Control-Right": lambda _v: self.goto(self.index + 1),
            "Control-Left": lambda _v: self.goto(self.index - 1),
            "Control-Shift-S": lambda _v: self.save_current(verbose=True),
            "Control-=": lambda _v: self.zoom_by(1.25),
            "Control-Shift-=": lambda _v: self.zoom_by(1.25),
            "Control--": lambda _v: self.zoom_by(1 / 1.25),
            "Control-0": lambda _v: self.fit_view(),
        }
        for key, func in bindings.items():
            try:
                self.viewer.bind_key(key, func, overwrite=True)
            except Exception as exc:  # napari version differences
                print("could not bind {}: {}".format(key, exc), file=sys.stderr)

    # -- layers ------------------------------------------------------------ #

    def _label_colors(self) -> Dict[Optional[int], Tuple[float, float, float, float]]:
        colors: Dict[Optional[int], Tuple[float, float, float, float]] = {}
        for i in range(len(self.class_names)):
            r, g, b = self.palette[i]
            # Class 0 is the background: leave it transparent so the underlying
            # image stays visible. Unpainted pixels are 0 too, which is exactly
            # what PaddleSeg expects for `_background_`.
            alpha = 0.0 if i == 0 else 1.0
            colors[i] = (r / 255.0, g / 255.0, b / 255.0, alpha)
        colors[IGNORE_INDEX] = (1.0, 1.0, 1.0, 1.0)
        colors[None] = (0.0, 0.0, 0.0, 0.0)
        return colors

    def _apply_colors(self) -> None:
        """Match the platform palette so previews agree with training."""
        colors = self._label_colors()
        try:
            from napari.utils.colormaps import DirectLabelColormap

            self.labels_layer.colormap = DirectLabelColormap(color_dict=colors)
            return
        except Exception:
            pass
        try:  # napari < 0.5
            self.labels_layer.color = {k: v for k, v in colors.items() if k is not None}
        except Exception as exc:
            print("could not apply palette: {}".format(exc), file=sys.stderr)

    def _on_class_changed(self, value: int) -> None:
        if self._syncing or self.labels_layer is None:
            return
        self._syncing = True
        try:
            self.labels_layer.selected_label = int(value)
        finally:
            self._syncing = False

    def _on_selected_label(self, event) -> None:
        if self._syncing:
            return
        value = int(self.labels_layer.selected_label)
        if not (0 <= value < len(self.class_names) or value == IGNORE_INDEX):
            return
        self._syncing = True
        try:
            self.w_class.value = value
        finally:
            self._syncing = False

    # -- view -------------------------------------------------------------- #

    def zoom_by(self, factor: float) -> None:
        self.viewer.camera.zoom = float(self.viewer.camera.zoom) * factor

    def fit_view(self) -> None:
        self.viewer.reset_view()

    def _on_camera_zoom(self, event=None) -> None:
        self.w_zoom.value = "zoom {:.2f}x".format(float(self.viewer.camera.zoom))

    # -- navigation / io --------------------------------------------------- #

    def goto(self, index: int, save: bool = True) -> None:
        if save:
            self.save_current()
        self.index = index % len(self.images)
        src = self.images[self.index]
        image = read_image(src)
        shape = (image.shape[0], image.shape[1])
        # Keep the camera where it was when stepping through same-sized images,
        # otherwise a zoomed-in workflow would reset on every switch.
        keep_view = (
            self.labels_layer is not None
            and bool(self.w_keep_view.value)
            and self._shape == shape
        )
        camera = (
            (float(self.viewer.camera.zoom), tuple(self.viewer.camera.center))
            if keep_view else None
        )
        try:
            labels = load_mask(self.mask_dst(src), shape)
        except ValueError as exc:
            self._set_status("mask ignored: {}".format(exc))
            labels = np.zeros(shape, dtype=np.uint8)

        if self.labels_layer is None:
            self.image_layer = self.viewer.add_image(image, name="image")
            self.labels_layer = self.viewer.add_labels(labels, name="label", opacity=0.5)
            self._apply_colors()
            self.labels_layer.mode = "paint"
            self.labels_layer.brush_size = int(self.w_brush.value)
            # Painting must overwrite neighbouring classes, otherwise adjacent
            # regions cannot share an edge.
            for attr, value in (("preserve_labels", False), ("n_edit_dimensions", 2)):
                try:
                    setattr(self.labels_layer, attr, value)
                except Exception:
                    pass
            self.labels_layer.selected_label = int(self.w_class.value)
            self.labels_layer.events.selected_label.connect(self._on_selected_label)
        else:
            self.image_layer.data = image
            self.labels_layer.data = labels

        self._shape = shape
        if camera is None:
            self.viewer.reset_view()
        else:
            self.viewer.camera.zoom, self.viewer.camera.center = camera
        self._on_camera_zoom()
        title = "[{}/{}] {}".format(self.index + 1, len(self.images), src.name)
        self.viewer.title = title
        done = sum(1 for p in self.images if self.mask_dst(p).exists())
        self.w_progress.value = "{}   |   saved {}/{}".format(
            title, done, len(self.images)
        )

    def save_current(self, verbose: bool = False) -> None:
        if self.labels_layer is None:
            return
        src = self.images[self.index]
        data = np.asarray(self.labels_layer.data, dtype=np.uint8)

        bad = np.unique(data)
        bad = [
            int(v) for v in bad
            if v != IGNORE_INDEX and v >= len(self.class_names)
        ]
        if bad:
            self._set_status("refused to save: unknown class ids {}".format(bad))
            return

        save_mask(self.mask_dst(src), data, self.palette)
        dst = self.image_dst(src)
        if not dst.exists():
            shutil.copy2(str(src), str(dst))
        if verbose:
            self._set_status("saved {}".format(self.mask_dst(src).name))

    def export(self) -> None:
        self.save_current()
        n_train, n_val = write_lists(
            self.out, self.images, float(self.w_ratio.value), self.seed
        )
        self._set_status(
            "train.txt {} / val.txt {}   num_classes={}".format(
                n_train, n_val, len(self.class_names)
            )
        )

    def show_stats(self) -> None:
        data = np.asarray(self.labels_layer.data, dtype=np.uint8)
        total = data.size
        values, counts = np.unique(data, return_counts=True)
        parts = []
        for value, count in zip(values.tolist(), counts.tolist()):
            name = (
                IGNORE_NAME if value == IGNORE_INDEX
                else self.class_names[value] if value < len(self.class_names)
                else "?{}".format(value)
            )
            parts.append("{} {:.1f}%".format(name, 100.0 * count / total))
        self._set_status("  ".join(parts))

    def _set_status(self, text: str) -> None:
        self.w_status.value = text
        print(text)

    def run(self) -> None:
        import napari

        napari.run()
        self.save_current()  # window closed -> flush the last edits
        write_lists(self.out, self.images, self.train_ratio, self.seed)


# --------------------------------------------------------------------------- #

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paint segmentation masks in napari, export a PaddleSeg dataset.",
    )
    parser.add_argument("--images", required=True, type=Path, help="source image directory")
    parser.add_argument("--out", required=True, type=Path, help="PaddleSeg data directory to write")
    parser.add_argument(
        "--classes",
        help="comma separated foreground class names; '_background_' is prepended "
             "automatically. Omit to reuse <out>/class_names.txt",
    )
    parser.add_argument(
        "--import-masks",
        type=Path,
        metavar="DIR",
        help="fold existing masks into the dataset before annotating, matching "
             "them to --images by filename stem (searched recursively). Point "
             "this at PaddleSeg's output/pseudo_color_prediction to review and "
             "correct model predictions instead of labelling from scratch",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="let --import-masks replace masks that already exist",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="only run --import-masks / --print-names, do not open napari",
    )
    parser.add_argument(
        "--print-names",
        action="store_true",
        help="print how each source filename is rewritten inside the dataset",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--brush-size", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.images.is_dir():
        raise SystemExit("--images is not a directory: {}".format(args.images))
    args.out.mkdir(parents=True, exist_ok=True)

    class_names = resolve_class_names(args.out, args.classes)
    print("classes: {}".format(", ".join(
        "{}={}".format(i, n) for i, n in enumerate(class_names)
    )))
    print("num_classes for the training config: {}".format(len(class_names)))

    images = list_images(args.images)
    if not images:
        raise SystemExit("no images found in {}".format(args.images))

    if args.print_names:
        names = build_name_map(images)
        print("\nsource name -> dataset name (anything outside ASCII "
              "[A-Za-z0-9._-] becomes '_', because PaddleSeg list files split "
              "lines on whitespace)")
        renamed = deduped = 0
        for src in images:
            img_name, mask_name = names[src]
            tags = []
            if img_name != src.name:
                tags.append("renamed")
                renamed += 1
            if img_name != dataset_names(src)[0]:
                tags.append("de-duplicated: sanitizing collided with another file")
                deduped += 1
            print("  {}\n    JPEGImages/{}\n    Annotations/{}{}".format(
                src.name, img_name, mask_name,
                "\n    ^ " + "; ".join(tags) if tags else ""))
        print("{} of {} renamed, {} de-duplicated".format(renamed, len(images), deduped))

    if args.import_masks:
        if not args.import_masks.is_dir():
            raise SystemExit("--import-masks is not a directory: {}".format(args.import_masks))
        (args.out / "class_names.txt").write_text(
            "\n".join(class_names) + "\n", encoding="utf-8"
        )
        report = import_masks(
            images, args.import_masks, args.out, class_names, overwrite=args.overwrite
        )
        print("\nimported {} / skipped {} / missing {} / rejected {}".format(
            *(len(report[k]) for k in ("imported", "skipped", "missing", "rejected"))
        ))
        for key in ("rejected", "missing", "skipped", "imported"):
            for line in report[key]:
                print("  [{}] {}".format(key, line))
        n_train, n_val = write_lists(args.out, images, args.train_ratio, args.seed)
        print("train.txt {} / val.txt {}".format(n_train, n_val))

    if args.no_gui:
        return 0

    try:
        annotator = Annotator(
            images_dir=args.images,
            out_dir=args.out,
            class_names=class_names,
            train_ratio=args.train_ratio,
            seed=args.seed,
            brush_size=args.brush_size,
        )
    except ImportError as exc:
        raise SystemExit(
            "napari is not available ({}). Install it with:\n"
            "    pip install -r {}".format(exc, Path(__file__).with_name("requirements.txt"))
        )
    annotator.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
