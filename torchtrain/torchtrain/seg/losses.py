"""
Segmentation losses, addressed by PaddleSeg's `loss.types[].type` names.

PaddleSeg's contract, reproduced here:

    loss:
      types:
        - type: CrossEntropyLoss      # one entry per logit the model emits
        - type: CrossEntropyLoss
      coef: [1, 0.4]                  # same length as `types`

The total loss is `sum(coef[i] * loss[i](logits[i], label))`. A mismatch between
`len(types)` and the number of logits is the single most common PaddleSeg
configuration error, so `build_losses` raises with the same explanatory message
the platform's UI validation uses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms import IGNORE_INDEX


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = IGNORE_INDEX, weight: Optional[Sequence[float]] = None,
                 top_k_percent_pixels: float = 1.0, **_: Any) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.weight = torch.tensor(list(weight), dtype=torch.float32) if weight else None
        self.top_k_percent_pixels = float(top_k_percent_pixels)

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(logit.device) if self.weight is not None else None
        if self.top_k_percent_pixels >= 1.0:
            return F.cross_entropy(logit, label, weight=weight, ignore_index=self.ignore_index)
        per_pixel = F.cross_entropy(
            logit, label, weight=weight, ignore_index=self.ignore_index, reduction="none"
        ).reshape(-1)
        keep = max(1, int(per_pixel.numel() * self.top_k_percent_pixels))
        top_k, _ = torch.topk(per_pixel, keep)
        return top_k.mean()


class OhemCrossEntropyLoss(nn.Module):
    """Online hard example mining CE (PaddleSeg's `OhemCrossEntropyLoss`)."""

    def __init__(self, thresh: float = 0.7, min_kept: int = 10000, ignore_index: int = IGNORE_INDEX, **_: Any) -> None:
        super().__init__()
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.ignore_index = int(ignore_index)

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        per_pixel = F.cross_entropy(logit, label, ignore_index=self.ignore_index, reduction="none").reshape(-1)
        valid = per_pixel[label.reshape(-1) != self.ignore_index]
        if valid.numel() == 0:
            return logit.sum() * 0.0
        threshold_loss = -torch.log(torch.tensor(self.thresh, device=logit.device).clamp(min=1e-8))
        hard = valid[valid > threshold_loss]
        if hard.numel() < min(self.min_kept, valid.numel()):
            keep = min(self.min_kept, valid.numel())
            hard, _ = torch.topk(valid, keep)
        return hard.mean()


class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int = IGNORE_INDEX, smooth: float = 1.0, **_: Any) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.smooth = float(smooth)

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        num_classes = logit.shape[1]
        probs = F.softmax(logit, dim=1)
        mask = (label != self.ignore_index).unsqueeze(1).float()
        target = F.one_hot(label.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
        probs, target = probs * mask, target * mask
        dims = (0, 2, 3)
        intersection = (probs * target).sum(dims)
        cardinality = probs.sum(dims) + target.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, ignore_index: int = IGNORE_INDEX, **_: Any) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.ignore_index = int(ignore_index)

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logit, label, ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * ce
        valid = label != self.ignore_index
        return loss[valid].mean() if bool(valid.any()) else logit.sum() * 0.0


class BCELoss(nn.Module):
    """Binary CE for 1- or 2-channel logits (PaddleSeg's `BCELoss`)."""

    def __init__(self, ignore_index: int = IGNORE_INDEX, **_: Any) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if logit.shape[1] == 1:
            scores = logit[:, 0]
        else:
            scores = logit[:, 1] - logit[:, 0]
        valid = label != self.ignore_index
        if not bool(valid.any()):
            return logit.sum() * 0.0
        target = (label == 1).float()
        return F.binary_cross_entropy_with_logits(scores[valid], target[valid])


class LovaszSoftmaxLoss(nn.Module):
    """Lovász-Softmax (Berman et al.), the multi-class IoU surrogate."""

    def __init__(self, ignore_index: int = IGNORE_INDEX, **_: Any) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)

    @staticmethod
    def _grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        p = gt_sorted.numel()
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0 : p - 1]
        return jaccard

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        num_classes = logit.shape[1]
        probs = F.softmax(logit, dim=1).permute(0, 2, 3, 1).reshape(-1, num_classes)
        labels = label.reshape(-1)
        valid = labels != self.ignore_index
        probs, labels = probs[valid], labels[valid]
        if labels.numel() == 0:
            return logit.sum() * 0.0
        losses = []
        for c in range(num_classes):
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            errors = (fg - probs[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            losses.append(torch.dot(errors_sorted, self._grad(fg[perm])))
        return torch.stack(losses).mean() if losses else logit.sum() * 0.0


class MixedLoss(nn.Module):
    """`MixedLoss` with nested `losses:` / `coef:`, as PaddleSeg defines it."""

    def __init__(self, losses: Optional[Sequence[Any]] = None, coef: Optional[Sequence[float]] = None,
                 ignore_index: int = IGNORE_INDEX, **_: Any) -> None:
        super().__init__()
        specs = list(losses or [{"type": "CrossEntropyLoss"}, {"type": "DiceLoss"}])
        self.losses = nn.ModuleList([build_loss(spec, ignore_index) for spec in specs])
        weights = list(coef or [1.0] * len(self.losses))
        if len(weights) != len(self.losses):
            weights = (weights + [1.0] * len(self.losses))[: len(self.losses)]
        self.coef = weights

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        total = None
        for weight, loss_fn in zip(self.coef, self.losses):
            value = float(weight) * loss_fn(logit, label)
            total = value if total is None else total + value
        return total


LOSSES: Dict[str, Any] = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "OhemCrossEntropyLoss": OhemCrossEntropyLoss,
    "DiceLoss": DiceLoss,
    "FocalLoss": FocalLoss,
    "BCELoss": BCELoss,
    "LovaszSoftmaxLoss": LovaszSoftmaxLoss,
    "MixedLoss": MixedLoss,
}


def build_loss(spec: Any, ignore_index: int = IGNORE_INDEX) -> nn.Module:
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict):
        name = str(spec.get("type") or spec.get("name") or spec.get("__tag__") or "CrossEntropyLoss")
        params = {k: v for k, v in spec.items() if k not in ("type", "name", "__tag__")}
    else:
        name, params = "CrossEntropyLoss", {}
    cls = LOSSES.get(name)
    if cls is None:
        from ..logger import log

        log("WARNING: unknown loss '{}', using CrossEntropyLoss instead.".format(name))
        cls = CrossEntropyLoss
    params.setdefault("ignore_index", ignore_index)
    return cls(**params)


class SegCriterion(nn.Module):
    """Weighted sum over the model's logits, plus a per-logit breakdown.

    The breakdown is what feeds `loss_cls` / `loss_iou` style columns in the log
    line, so the monitoring page can chart the auxiliary head separately.
    """

    def __init__(self, losses: Sequence[nn.Module], coef: Sequence[float]) -> None:
        super().__init__()
        self.losses = nn.ModuleList(list(losses))
        self.coef = [float(c) for c in coef]

    def forward(self, logits: List[torch.Tensor], label: torch.Tensor):
        total: Optional[torch.Tensor] = None
        parts: Dict[str, float] = {}
        count = min(len(logits), len(self.losses))
        for i in range(count):
            value = self.losses[i](logits[i], label)
            weighted = self.coef[i] * value
            total = weighted if total is None else total + weighted
            parts["loss_aux{}".format(i)] = float(value.detach())
        # `loss_cls` is the name the log parser stores in a dedicated column;
        # map the main head onto it so the chart has a meaningful series.
        if "loss_aux0" in parts:
            parts["loss_cls"] = parts.pop("loss_aux0")
        return total, parts


def build_losses(cfg: Optional[Dict[str, Any]], num_logits: int, ignore_index: int = IGNORE_INDEX) -> SegCriterion:
    """Build the criterion from a `loss:` block, validating the logits count."""
    cfg = cfg or {}
    specs = cfg.get("types") or [{"type": "CrossEntropyLoss"}]
    if isinstance(specs, dict):
        specs = [specs]
    coef = cfg.get("coef") or [1.0] * len(specs)
    if isinstance(coef, (int, float)):
        coef = [float(coef)]

    if len(specs) != num_logits:
        raise ValueError(
            "The length of logits_list should equal to the types of loss config: "
            "{} != {}. The model emits {} logit(s), so loss.types must have exactly "
            "{} entr{}.".format(
                num_logits, len(specs), num_logits, num_logits, "y" if num_logits == 1 else "ies"
            )
        )
    if len(coef) != len(specs):
        raise ValueError(
            "loss.coef must have the same length as loss.types ({} != {}).".format(len(coef), len(specs))
        )

    return SegCriterion([build_loss(spec, ignore_index) for spec in specs], coef)


def count_logits(cfg: Optional[Dict[str, Any]]) -> int:
    """How many logits the `loss:` block implies. Used to configure aux heads."""
    cfg = cfg or {}
    specs = cfg.get("types")
    if isinstance(specs, dict):
        return 1
    if isinstance(specs, (list, tuple)) and specs:
        return len(specs)
    return 1
