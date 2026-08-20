"""
Detection architectures, backed by `torchvision.models.detection`.

Architecture names reuse PaddleDetection's where a faithful torchvision
equivalent exists (`FasterRCNN`, `RetinaNet`, `FCOS`, `SSD`). PaddleDetection
families with no torchvision counterpart (`YOLOv3`/PP-YOLOE, `PicoDet`, `DETR`)
raise a clear error listing what *is* available, rather than silently training a
different network than the config asked for.

Transfer learning
-----------------
`pretrain_weights` decides where the initial weights come from:

    COCO      (default) COCO-pretrained detector, classification head replaced
                        to match `num_classes`. By far the best option for the
                        few-hundred-image datasets this platform is used with.
    ImageNet            ImageNet backbone only, detector heads random.
    <path>.pt           A checkpoint produced by this trainer.
    "" / Null / false   Fully random initialisation.

Head replacement is per-architecture surgery (torchvision has no generic API for
it), so a family we cannot rewire falls back to ImageNet with a warning instead
of loading a 91-class head that cannot be trained against this dataset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..logger import log

# architecture -> {backbone label: torchvision builder name}
DET_ARCHITECTURES: Dict[str, Dict[str, str]] = {
    "FasterRCNN": {
        "ResNet50-FPN": "fasterrcnn_resnet50_fpn",
        "ResNet50-FPN-v2": "fasterrcnn_resnet50_fpn_v2",
        "MobileNetV3-Large-FPN": "fasterrcnn_mobilenet_v3_large_fpn",
        "MobileNetV3-Large-320-FPN": "fasterrcnn_mobilenet_v3_large_320_fpn",
    },
    "RetinaNet": {
        "ResNet50-FPN": "retinanet_resnet50_fpn",
        "ResNet50-FPN-v2": "retinanet_resnet50_fpn_v2",
    },
    "FCOS": {
        "ResNet50-FPN": "fcos_resnet50_fpn",
    },
    "SSD": {
        "VGG16": "ssd300_vgg16",
        "MobileNetV3-Large": "ssdlite320_mobilenet_v3_large",
    },
}

# PaddleDetection backbone spellings that map onto a torchvision option, so a
# config written for Paddle still resolves to something sensible.
_BACKBONE_ALIASES: Dict[str, str] = {
    "ResNet": "ResNet50-FPN",
    "ResNet50": "ResNet50-FPN",
    "ResNet50_vd": "ResNet50-FPN",
    "ResNet101": "ResNet50-FPN",
    "ResNeXt": "ResNet50-FPN",
    "MobileNetV3": "MobileNetV3-Large-FPN",
    "MobileNetV1": "MobileNetV3-Large-FPN",
    "VGG": "VGG16",
}

_UNSUPPORTED_HINTS = {
    "YOLOv3": "PP-YOLOE / YOLOv3 are PaddleDetection-only. Use FCOS or RetinaNet for a comparable anchor-free/one-stage model.",
    "PicoDet": "PicoDet is PaddleDetection-only. Use SSD with the MobileNetV3-Large backbone for a comparable lightweight model.",
    "DETR": "DETR/RT-DETR are not in torchvision. Use FasterRCNN (ResNet50-FPN-v2) for comparable accuracy.",
    "CenterNet": "CenterNet is not in torchvision. Use FCOS, which is also anchor-free.",
}


def _resolve_names(cfg: Dict[str, Any]) -> Tuple[str, str, str]:
    """`(architecture, backbone_label, builder_name)` from a merged config."""
    arch = str(cfg.get("architecture") or "FasterRCNN")
    if arch not in DET_ARCHITECTURES:
        hint = _UNSUPPORTED_HINTS.get(arch, "")
        raise ValueError(
            "Architecture '{}' is not supported by TorchDet. Supported: {}.{}".format(
                arch, ", ".join(DET_ARCHITECTURES), " " + hint if hint else ""
            )
        )

    # PaddleDetection nests components under a block named after the arch.
    arch_block = cfg.get(arch) if isinstance(cfg.get(arch), dict) else {}
    raw_backbone = str(arch_block.get("backbone") or cfg.get("backbone") or "")
    options = DET_ARCHITECTURES[arch]
    default_backbone = next(iter(options))

    backbone = raw_backbone if raw_backbone in options else _BACKBONE_ALIASES.get(raw_backbone, "")
    if backbone not in options:
        if raw_backbone:
            log(
                "WARNING: backbone '{}' is not available for {}; using {}.".format(
                    raw_backbone, arch, default_backbone
                )
            )
        backbone = default_backbone
    return arch, backbone, options[backbone]


def _weights_backbone(builder_name: str):
    """ImageNet weights enum for a builder's backbone, or None if unavailable."""
    try:
        if "resnet50" in builder_name:
            from torchvision.models import ResNet50_Weights

            return ResNet50_Weights.IMAGENET1K_V1
        if "mobilenet_v3_large" in builder_name:
            from torchvision.models import MobileNet_V3_Large_Weights

            return MobileNet_V3_Large_Weights.IMAGENET1K_V1
        if "vgg16" in builder_name:
            from torchvision.models import VGG16_Weights

            return VGG16_Weights.IMAGENET1K_FEATURES
    except Exception as exc:  # noqa: BLE001
        log("WARNING: could not resolve ImageNet backbone weights ({}).".format(exc))
    return None


def _replace_classification_head(net: nn.Module, arch: str, num_classes_with_bg: int) -> bool:
    """Rewire a COCO-pretrained detector's classifier for `num_classes_with_bg`.

    Returns False when the architecture has no supported surgery, so the caller
    can fall back to ImageNet initialisation.
    """
    try:
        if arch == "FasterRCNN":
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

            in_features = net.roi_heads.box_predictor.cls_score.in_features
            net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
            return True
        if arch == "RetinaNet":
            from torchvision.models.detection.retinanet import RetinaNetClassificationHead

            head = net.head.classification_head
            net.head.classification_head = RetinaNetClassificationHead(
                in_channels=net.backbone.out_channels,
                num_anchors=head.num_anchors,
                num_classes=num_classes_with_bg,
            )
            return True
        if arch == "FCOS":
            from torchvision.models.detection.fcos import FCOSClassificationHead

            head = net.head.classification_head
            net.head.classification_head = FCOSClassificationHead(
                in_channels=net.backbone.out_channels,
                num_anchors=head.num_anchors,
                num_classes=num_classes_with_bg,
            )
            return True
    except Exception as exc:  # noqa: BLE001 - torchvision internals vary by version
        log("WARNING: could not replace the {} classification head ({}).".format(arch, exc))
        return False
    return False


def _coco_weights(builder_name: str):
    try:
        from torchvision.models import get_model_weights

        enum = get_model_weights(builder_name)
        return enum.DEFAULT
    except Exception as exc:  # noqa: BLE001
        log("WARNING: could not resolve COCO-pretrained weights for {} ({}).".format(builder_name, exc))
        return None


def build_model(
    cfg: Dict[str, Any],
    num_classes: int,
    input_size: Optional[Tuple[int, int]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build a detection model. `num_classes` excludes background (Paddle style).

    Returns `(model, meta)` where `meta` is embedded in checkpoints so
    `val.py` / `predict.py` can rebuild the identical network from the weights.
    """
    arch, backbone, builder_name = _resolve_names(cfg)
    num_classes = int(num_classes)
    if num_classes < 1:
        raise ValueError("`num_classes` must be >= 1 (foreground classes, background excluded)")
    num_classes_with_bg = num_classes + 1

    pretrain = cfg.get("pretrain_weights", "COCO")
    if pretrain is None or pretrain is False:
        pretrain = ""
    pretrain = str(pretrain)
    mode = pretrain.strip().lower()

    import torchvision.models.detection as tvdet

    builder = getattr(tvdet, builder_name)
    kwargs: Dict[str, Any] = {}
    if input_size:
        kwargs["min_size"], kwargs["max_size"] = int(input_size[0]), int(input_size[1])

    net: Optional[nn.Module] = None
    strategy = "random"

    if mode in ("coco", "default", "true", "pretrained", "coco_pretrained"):
        weights = _coco_weights(builder_name)
        if weights is not None:
            try:
                # Build with COCO's own class count so the pretrained head loads,
                # then swap the classifier for ours.
                net = builder(weights=weights, **kwargs)
                if _replace_classification_head(net, arch, num_classes_with_bg):
                    strategy = "COCO (head replaced)"
                else:
                    log(
                        "WARNING: {} does not support head replacement here; "
                        "falling back to ImageNet backbone weights.".format(arch)
                    )
                    net = None
            except Exception as exc:  # noqa: BLE001 - typically a download failure
                log("WARNING: loading COCO weights for {} failed ({}); trying ImageNet.".format(arch, exc))
                net = None
        if net is None:
            mode = "imagenet"

    if net is None and mode in ("imagenet", "backbone", "imagenet1k"):
        weights_backbone = _weights_backbone(builder_name)
        try:
            net = builder(weights=None, weights_backbone=weights_backbone, num_classes=num_classes_with_bg, **kwargs)
            strategy = "ImageNet backbone" if weights_backbone is not None else "random"
        except Exception as exc:  # noqa: BLE001
            log("WARNING: ImageNet init failed ({}); using random initialisation.".format(exc))
            net = None

    if net is None:
        net = builder(weights=None, weights_backbone=None, num_classes=num_classes_with_bg, **kwargs)
        strategy = "random"

    meta = {
        "task": "det",
        "architecture": arch,
        "backbone": backbone,
        "builder": builder_name,
        "num_classes": num_classes,
        "input_size": list(input_size) if input_size else None,
        "init": strategy,
    }

    # A checkpoint path overrides everything above: it is a full state dict for
    # this exact architecture.
    if pretrain.endswith((".pt", ".pth", ".tar")):
        import os

        if os.path.isfile(pretrain):
            from ..utils import load_checkpoint

            payload = load_checkpoint(pretrain)
            missing, unexpected = net.load_state_dict(payload["state_dict"], strict=False)
            meta["init"] = "checkpoint: {}".format(pretrain)
            log(
                "Loaded pretrained weights from {} ({} missing, {} unexpected keys).".format(
                    pretrain, len(missing), len(unexpected)
                )
            )
        else:
            log("WARNING: pretrain_weights '{}' not found; keeping {} initialisation.".format(pretrain, strategy))

    log("Model: {} / {} ({}), init: {}".format(arch, backbone, builder_name, meta["init"]))
    return net, meta


def rebuild_from_meta(meta: Dict[str, Any], cfg: Dict[str, Any]) -> nn.Module:
    """Rebuild the architecture recorded in a checkpoint, with random weights."""
    rebuilt_cfg = dict(cfg)
    rebuilt_cfg["architecture"] = meta.get("architecture", cfg.get("architecture", "FasterRCNN"))
    rebuilt_cfg[rebuilt_cfg["architecture"]] = {"backbone": meta.get("backbone", "")}
    rebuilt_cfg["pretrain_weights"] = ""
    input_size = meta.get("input_size")
    model, _ = build_model(
        rebuilt_cfg,
        int(meta.get("num_classes", 1)),
        tuple(input_size) if input_size else None,
    )
    return model


# Loss-component names torchvision reports, mapped onto the four loss columns the
# platform's `TrainingLog` table has. The mapping is semantic, not arbitrary:
# classification -> loss_cls, box regression -> loss_l1, objectness/centerness ->
# loss_iou, RPN box regression -> loss_dfl.
_LOSS_ALIASES = {
    "loss_classifier": "loss_cls",
    "classification": "loss_cls",
    "bbox_regression": "loss_l1",
    "loss_box_reg": "loss_l1",
    "loss_objectness": "loss_iou",
    "bbox_ctrness": "loss_iou",
    "loss_rpn_box_reg": "loss_dfl",
}


def normalize_loss_dict(losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Rename torchvision loss keys to the platform's column names."""
    out: Dict[str, float] = {}
    for key, value in losses.items():
        out[_LOSS_ALIASES.get(key, key)] = float(value.detach())
    return out


def architecture_options() -> List[str]:
    return list(DET_ARCHITECTURES)
