"""Read PaddleSeg pseudo-color masks as class-id arrays and per-class masks.

The palette is not arbitrary: `tools/napari_seg/annotate.py::voc_colormap()` is the
PASCAL VOC color map with the leading black entry dropped, so `class_id ==
palette index` and class 0 is (128, 0, 0) rather than black. That module is loaded
by path here instead of being re-implemented, so the two tools can never drift
apart. It only imports napari inside function bodies, so importing it is cheap.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

IGNORE_INDEX = 255
BACKGROUND = "_background_"

# Class order for this project's TEM stack; the index is the PaddleSeg class id.
DEFAULT_CLASSES: List[str] = [
    BACKGROUND,
    "SAF_Ru_L",
    "SAF_Ru_R",
    "MgO_L",
    "MgO_R",
    "MgO_C",
    "Non_mag",
    "Milling_L",
    "Milling_R",
    "Leveling",
    "Block_U",
    "Block_D",
]

# Share of pixels allowed to sit far from any palette color before we conclude the
# file is a blended preview (`added_prediction/`) rather than a label map.
RGB_SNAP_TOL = 30.0
RGB_SNAP_MAX_OFF = 0.05

_ANNOTATE_PY = Path(__file__).resolve().parent.parent / "napari_seg" / "annotate.py"
_annotate = None


def _annotate_module():
    global _annotate
    if _annotate is None:
        if not _ANNOTATE_PY.exists():
            raise RuntimeError("cannot find {}".format(_ANNOTATE_PY))
        spec = importlib.util.spec_from_file_location("_tem_annotate", _ANNOTATE_PY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _annotate = module
    return _annotate


def palette(num_classes: int) -> np.ndarray:
    """256x3 uint8 palette; index == class id, 255 (ignore) is white."""
    return _annotate_module().build_palette(num_classes)


def resolve_classes(explicit: Optional[str], mask_path: Optional[Path]) -> List[str]:
    """Class order defines the ids. `--classes` wins, then a sibling
    class_names.txt, then the built-in TEM list."""
    if explicit:
        names = [n.strip() for n in explicit.split(",") if n.strip()]
    else:
        names = None
        for base in _candidate_dirs(mask_path):
            persisted = base / "class_names.txt"
            if persisted.exists():
                names = [
                    line.strip()
                    for line in persisted.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                break
        if not names:
            return list(DEFAULT_CLASSES)
    if not names or names[0] != BACKGROUND:
        names.insert(0, BACKGROUND)
    if len(names) != len(set(names)):
        raise ValueError("duplicate class names: {}".format(names))
    return names


def _candidate_dirs(mask_path: Optional[Path]) -> List[Path]:
    if mask_path is None:
        return []
    d = mask_path.parent
    return [d, d.parent, d.parent.parent] if d.parent != d else [d]


def load_label_map(path: Path, num_classes: int) -> Tuple[np.ndarray, str]:
    """Return (class ids, a note about how the file was interpreted)."""
    im = Image.open(str(path))
    if im.mode in ("P", "L"):
        ids = np.array(im).astype(np.uint8)
        note = "{} (index == class id)".format(im.mode)
    elif im.mode in ("RGB", "RGBA"):
        ann = _annotate_module()
        palette = ann.build_palette(num_classes)
        valid = list(range(num_classes)) + [IGNORE_INDEX]
        ids, dist = ann.rgb_to_indices(np.array(im.convert("RGB")), palette, valid)
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
    if ids.ndim != 2:
        raise ValueError("expected a 2-D label map, got shape {}".format(ids.shape))
    return ids, note


_CONN8 = np.ones((3, 3), dtype=bool)


def class_mask(
    ids: np.ndarray,
    class_id: int,
    min_area: int = 30,
    largest_only: bool = True,
    prefer_near: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[int], bool]:
    """Boolean mask for one class, cleaned of prediction specks.

    Returns (mask, areas of the dropped components, whether `prefer_near` decided
    it). Reporting the dropped areas matters: a large dropped component means the
    segmentation split a layer in two, and every endpoint measured afterwards is
    suspect.

    `prefer_near` restricts the choice to components touching that mask before
    falling back to sheer size. "Largest" alone is not a safe proxy for "the right
    one": one prediction in a 26-image batch grew a spurious Non_mag band across the
    bottom of the frame that was *bigger* than the real Non_mag box, so the size rule
    picked the artefact. Anchoring on the neighbour that the measurement already
    depends on (Block_D for Non_mag) resolves it from information we have anyway.
    """
    raw = ids == class_id
    if not raw.any():
        return raw, [], False
    lab, n = ndimage.label(raw, structure=_CONN8)
    if n <= 1 and min_area <= 0:
        return raw, [], False
    areas = ndimage.sum_labels(raw, lab, index=np.arange(1, n + 1)).astype(int)
    keep = areas >= max(1, min_area)
    anchored = False
    if largest_only and keep.any():
        candidates = keep.copy()
        if prefer_near is not None and prefer_near.any():
            touching = np.zeros(n, dtype=bool)
            near = ndimage.binary_dilation(prefer_near, structure=_CONN8)
            hit = np.unique(lab[near])
            touching[hit[hit > 0] - 1] = True
            if (candidates & touching).any():
                candidates &= touching
                anchored = True
        best = int(np.argmax(np.where(candidates, areas, -1)))
        keep = np.zeros_like(keep)
        keep[best] = True
    dropped = sorted((int(a) for a, k in zip(areas, keep) if not k), reverse=True)
    lut = np.zeros(n + 1, dtype=bool)
    lut[1:] = keep
    return lut[lab], dropped, anchored
