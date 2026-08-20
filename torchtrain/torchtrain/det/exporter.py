"""
Detection model export.

torchvision detectors take a *list* of variable-size tensors and return a list of
dicts, which neither `torch.jit.trace` nor ONNX export handles cleanly (and
scripting them requires upstream changes). Rather than ship a broken artifact,
the bundle contains the checkpoint plus everything needed to reload it:

    <save_dir>/
      model.pt        # the same state dict + architecture metadata
      infer_cfg.yml   # architecture, backbone, class names, input size, threshold

`torchtrain.det.trainer.load_model_for_inference` reads exactly this metadata, so
the bundle is self-describing and can be loaded without the original job config.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict

import yaml

from .. import logger as L
from ..utils import load_checkpoint


def export(cfg: Dict[str, Any], args: Any, weights: str, save_dir: str) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)
    payload = load_checkpoint(weights)

    target = os.path.join(save_dir, "model.pt")
    if os.path.abspath(weights) != os.path.abspath(target):
        shutil.copyfile(weights, target)

    from .predictor import _class_names

    num_classes = int(payload.get("num_classes") or cfg.get("num_classes") or 1)
    infer_cfg = {
        "framework": "TorchDet",
        "task": "detection",
        "format": "torch-checkpoint",
        "model_file": "model.pt",
        "architecture": payload.get("architecture") or cfg.get("architecture") or "FasterRCNN",
        "backbone": payload.get("backbone") or "",
        "num_classes": num_classes,
        "class_names": _class_names(cfg, num_classes),
        "input_size": payload.get("input_size"),
        "score_threshold": float(getattr(args, "score_threshold", 0.5) or 0.5),
        "preprocess": "torchvision GeneralizedRCNNTransform (resize + ImageNet normalisation, applied inside the model)",
        "note": (
            "Load with torchtrain.det.trainer.load_model_for_inference, or rebuild the "
            "architecture above via torchvision and load state_dict from model.pt."
        ),
    }
    with open(os.path.join(save_dir, "infer_cfg.yml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(infer_cfg, handle, sort_keys=False, allow_unicode=True)

    if getattr(args, "format", "torchscript") == "onnx":
        L.log(
            "NOTE: ONNX export is not supported for torchvision detectors "
            "(variable-size list inputs). Wrote a torch checkpoint bundle instead."
        )

    L.log("Exported detection bundle to {}".format(os.path.abspath(save_dir)))
    return {"dir": os.path.abspath(save_dir), "format": "torch-checkpoint"}
