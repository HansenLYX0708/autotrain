"""
Segmentation architectures.

Every model returns a **list of logits** (main head first, then auxiliary heads),
which is exactly PaddleSeg's contract. That is what makes PaddleSeg's rule

    len(loss.types) == number of logits the model emits

apply here unchanged, and it is why `SEG_ARCHITECTURES[].logits` in
`src/lib/model-yaml.ts` can keep validating torch models with the same code it
uses for Paddle ones.

Architectures are backed by `torchvision.models.segmentation` where a faithful
equivalent exists, plus a hand-written UNet (torchvision has none).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..logger import log

# name -> (torchvision builder, default backbone, supports aux head)
_TV_BUILDERS: Dict[str, Tuple[str, str, bool]] = {
    "DeepLabV3P": ("deeplabv3", "ResNet50", True),
    "DeepLabV3": ("deeplabv3", "ResNet50", True),
    "FCN": ("fcn", "ResNet50", True),
    "LRASPP": ("lraspp", "MobileNetV3-Large", False),
}

# backbone label (as shown in the UI) -> torchvision function suffix
_BACKBONES: Dict[str, str] = {
    "ResNet50": "resnet50",
    "ResNet50_vd": "resnet50",
    "ResNet101": "resnet101",
    "ResNet101_vd": "resnet101",
    "MobileNetV3-Large": "mobilenet_v3_large",
    "MobileNetV3_large": "mobilenet_v3_large",
}


class SegModel(nn.Module):
    """Uniform wrapper: `forward(x) -> List[Tensor]`, each `[N, C, H, W]`."""

    def __init__(self, num_classes: int, align_corners: bool = False) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.align_corners = bool(align_corners)

    def _resize(self, logit: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if logit.shape[-2:] == size:
            return logit
        return F.interpolate(logit, size=size, mode="bilinear", align_corners=self.align_corners)


class _TorchvisionSeg(SegModel):
    def __init__(self, net: nn.Module, num_classes: int, align_corners: bool, has_aux: bool) -> None:
        super().__init__(num_classes, align_corners)
        self.net = net
        self.has_aux = has_aux

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        size = (int(x.shape[-2]), int(x.shape[-1]))
        out = self.net(x)
        logits = [self._resize(out["out"], size)]
        if self.has_aux and "aux" in out:
            logits.append(self._resize(out["aux"], size))
        return logits


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(SegModel):
    """Classic UNet (single logit), matching PaddleSeg's `UNet` channel widths.

    Kept dependency-free because it is the sane default for the small,
    single-channel microscopy datasets this platform is used with: it trains from
    scratch on a few dozen images where an ImageNet ResNet backbone overfits.
    """

    def __init__(
        self,
        num_classes: int,
        align_corners: bool = False,
        base_channels: int = 64,
        depth: int = 4,
        **_: Any,
    ) -> None:
        super().__init__(num_classes, align_corners)
        depth = max(1, int(depth))
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.downs = nn.ModuleList()
        in_ch = 3
        for ch in channels[:-1]:
            self.downs.append(_double_conv(in_ch, ch))
            in_ch = ch
        self.bottleneck = _double_conv(channels[-2], channels[-1])

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(nn.ConvTranspose2d(channels[i], channels[i - 1], 2, stride=2))
            self.up_convs.append(_double_conv(channels[i - 1] * 2, channels[i - 1]))

        self.classifier = nn.Conv2d(channels[0], self.num_classes, 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        size = (int(x.shape[-2]), int(x.shape[-1]))
        skips = []
        for block in self.downs:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                # Odd input sizes make the transposed conv land one pixel short.
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=self.align_corners)
            x = conv(torch.cat([skip, x], dim=1))
        return [self._resize(self.classifier(x), size)]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

SEG_ARCHITECTURES = ("UNet", "DeepLabV3P", "DeepLabV3", "FCN", "LRASPP")


def _load_backbone_weights(backbone_key: str, pretrained: Any) -> Any:
    """Resolve `weights_backbone` for a torchvision segmentation builder.

    `pretrained` is whatever the config said: `True`/`"imagenet"` asks for
    ImageNet weights (a download on first use), a path is loaded manually later,
    and anything falsy means random init. A download failure is downgraded to a
    warning: an offline machine should still be able to train from scratch
    rather than have the job die at startup.
    """
    if not pretrained or isinstance(pretrained, str) and pretrained.lower() in ("null", "none", "false", ""):
        return None
    if isinstance(pretrained, str) and ("/" in pretrained or "\\" in pretrained or pretrained.endswith(".pt")):
        return None  # a checkpoint path; handled by `load_pretrained`
    try:
        from torchvision.models import get_model_weights

        if backbone_key.startswith("resnet"):
            from torchvision.models import ResNet50_Weights, ResNet101_Weights

            return ResNet50_Weights.IMAGENET1K_V1 if backbone_key == "resnet50" else ResNet101_Weights.IMAGENET1K_V1
        if backbone_key.startswith("mobilenet"):
            from torchvision.models import MobileNet_V3_Large_Weights

            return MobileNet_V3_Large_Weights.IMAGENET1K_V1
        del get_model_weights
    except Exception as exc:  # noqa: BLE001 - any torchvision/version issue
        log("WARNING: could not resolve pretrained backbone weights ({}); training from scratch.".format(exc))
    return None


def build_model(
    cfg: Dict[str, Any],
    num_classes: Optional[int] = None,
    num_logits: Optional[int] = None,
) -> SegModel:
    """Build a segmentation model from a PaddleSeg `model:` block.

    `num_logits` (derived from `len(loss.types)`) decides whether an auxiliary
    head is attached, so the model and the loss can never disagree about how
    many outputs there are.
    """
    if not isinstance(cfg, dict):
        raise ValueError("`model:` must be a mapping")

    arch = str(cfg.get("type") or cfg.get("name") or cfg.get("__tag__") or "UNet")
    classes = int(cfg.get("num_classes") or num_classes or 0)
    if classes < 1:
        raise ValueError("`model.num_classes` must be >= 1")
    align_corners = bool(cfg.get("align_corners", False))

    backbone_cfg = cfg.get("backbone")
    backbone_name = ""
    backbone_pretrained: Any = None
    if isinstance(backbone_cfg, dict):
        backbone_name = str(backbone_cfg.get("type") or backbone_cfg.get("name") or "")
        backbone_pretrained = backbone_cfg.get("pretrained")
    elif isinstance(backbone_cfg, str):
        backbone_name = backbone_cfg
    pretrained = cfg.get("pretrained", backbone_pretrained)

    if arch == "UNet":
        model = UNet(
            classes,
            align_corners=align_corners,
            base_channels=int(cfg.get("base_channels", 64) or 64),
            depth=int(cfg.get("depth", 4) or 4),
        )
        if num_logits and num_logits != 1:
            log(
                "WARNING: UNet emits 1 logit but the loss config declares {}. "
                "Extra loss entries will be ignored.".format(num_logits)
            )
        return _maybe_load_pretrained(model, pretrained)

    if arch not in _TV_BUILDERS:
        raise ValueError(
            "Unsupported segmentation architecture '{}'. Supported: {}".format(arch, ", ".join(SEG_ARCHITECTURES))
        )

    builder_prefix, default_backbone, supports_aux = _TV_BUILDERS[arch]
    backbone_label = backbone_name or default_backbone
    backbone_key = _BACKBONES.get(backbone_label)
    if backbone_key is None:
        log(
            "WARNING: backbone '{}' is not available for {}; using {}.".format(
                backbone_label, arch, default_backbone
            )
        )
        backbone_key = _BACKBONES[default_backbone]

    aux_loss = bool(supports_aux and (num_logits is None or num_logits >= 2))
    fn_name = "{}_{}".format(builder_prefix, backbone_key)

    import torchvision.models.segmentation as tvseg

    builder = getattr(tvseg, fn_name, None)
    if builder is None:
        # e.g. `fcn_mobilenet_v3_large` does not exist upstream.
        fallback = "{}_{}".format(builder_prefix, _BACKBONES[default_backbone])
        log("WARNING: torchvision has no '{}'; falling back to '{}'.".format(fn_name, fallback))
        builder = getattr(tvseg, fallback)
        backbone_key = _BACKBONES[default_backbone]

    kwargs: Dict[str, Any] = {"weights": None, "num_classes": classes}
    if builder_prefix != "lraspp":
        kwargs["aux_loss"] = aux_loss
    weights_backbone = _load_backbone_weights(backbone_key, pretrained)
    try:
        net = builder(weights_backbone=weights_backbone, **kwargs)
    except Exception as exc:  # noqa: BLE001 - most often a weights download failure
        log("WARNING: building {} with pretrained backbone failed ({}); retrying from scratch.".format(arch, exc))
        net = builder(weights_backbone=None, **kwargs)

    model = _TorchvisionSeg(net, classes, align_corners, has_aux=aux_loss and builder_prefix != "lraspp")
    return _maybe_load_pretrained(model, pretrained)


def _maybe_load_pretrained(model: SegModel, pretrained: Any) -> SegModel:
    """Load a whole-model checkpoint when `pretrained` is a filesystem path."""
    if not isinstance(pretrained, str):
        return model
    if not (pretrained.endswith(".pt") or pretrained.endswith(".pth") or pretrained.endswith(".tar")):
        return model
    import os

    if not os.path.isfile(pretrained):
        log("WARNING: pretrained weights '{}' not found; training from scratch.".format(pretrained))
        return model
    from ..utils import load_checkpoint

    payload = load_checkpoint(pretrained)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    log(
        "Loaded pretrained weights from {} ({} missing, {} unexpected keys).".format(
            pretrained, len(missing), len(unexpected)
        )
    )
    return model


def model_meta(cfg: Dict[str, Any], num_classes: int, num_logits: int) -> Dict[str, Any]:
    """Snapshot of what was built, embedded in checkpoints for later rebuilds."""
    return {
        "task": "seg",
        "model_cfg": cfg,
        "num_classes": int(num_classes),
        "num_logits": int(num_logits),
    }
