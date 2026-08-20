"""
Runtime helpers shared by the segmentation and detection trainers.
"""

from __future__ import annotations

import contextlib
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .logger import log


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(use_gpu: bool = True) -> torch.device:
    """Pick the device, reporting *why* when a GPU request cannot be honoured.

    The runner sets `CUDA_VISIBLE_DEVICES` before spawning us, so device 0 here
    is always the GPU the user selected in the UI.
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
        log("Using GPU: {} (CUDA_VISIBLE_DEVICES={})".format(name, visible))
        return device
    if use_gpu:
        log("WARNING: use_gpu is true but torch.cuda.is_available() is False - falling back to CPU.")
    return torch.device("cpu")


def memory_stats_mb(device: torch.device) -> Tuple[int, int]:
    """`(max_reserved, max_allocated)` in MB; zeros on CPU."""
    if device.type != "cuda":
        return 0, 0
    mb = 1024 * 1024
    return (
        int(torch.cuda.max_memory_reserved(device) / mb),
        int(torch.cuda.max_memory_allocated(device) / mb),
    )


class AverageMeter:
    """Windowed mean, used for the smoothed `batch_cost` / `loss` we report."""

    def __init__(self, window: int = 20) -> None:
        self.window = max(1, window)
        self.values: List[float] = []

    def update(self, value: float) -> None:
        self.values.append(float(value))
        if len(self.values) > self.window:
            self.values = self.values[-self.window :]

    @property
    def avg(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def reset(self) -> None:
        self.values = []


class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start

    def reset(self) -> None:
        self.start = time.time()


def build_optimizer(
    params: Any,
    cfg: Dict[str, Any],
    base_lr: float,
) -> torch.optim.Optimizer:
    """Build a torch optimizer from a Paddle-style `optimizer:` block.

    Accepted `type` values mirror the options the web app offers:
    `SGD`, `Momentum` (SGD + momentum, Paddle's name), `Adam`, `AdamW`,
    `RMSProp`. An unknown type falls back to SGD with a warning rather than
    aborting a queued job.
    """
    opt_type = str(cfg.get("type") or cfg.get("name") or "SGD")
    momentum = float(cfg.get("momentum", 0.9) or 0.9)
    weight_decay = cfg.get("weight_decay")
    if weight_decay is None:
        weight_decay = cfg.get("regularizer", {}).get("factor") if isinstance(cfg.get("regularizer"), dict) else None
    weight_decay = float(weight_decay if weight_decay is not None else 0.0)

    key = opt_type.lower()
    if key in ("sgd", "momentum"):
        return torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=momentum if key == "momentum" else float(cfg.get("momentum", 0.0) or 0.0),
            weight_decay=weight_decay,
            nesterov=bool(cfg.get("nesterov", False)),
        )
    if key == "adam":
        return torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    if key in ("rmsprop", "rms_prop"):
        return torch.optim.RMSprop(params, lr=base_lr, momentum=momentum, weight_decay=weight_decay)

    log("WARNING: unknown optimizer type '{}', falling back to SGD(momentum=0.9).".format(opt_type))
    return torch.optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=weight_decay)


class LrScheduler:
    """Step-wise LR schedule with optional linear warmup.

    Implemented by hand instead of via `torch.optim.lr_scheduler` because the
    Paddle configs express the schedule in *training steps* (iterations for
    segmentation, epochs for detection) and mix a warmup with the decay policy
    in one block. Computing the factor directly is both shorter and easier to
    keep faithful to the source config.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        total_steps: int,
        policy: str = "PolynomialDecay",
        power: float = 0.9,
        end_lr: float = 0.0,
        gamma: float = 0.1,
        milestones: Optional[List[int]] = None,
        warmup_steps: int = 0,
        warmup_start_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.total_steps = max(1, int(total_steps))
        self.policy = policy or "PolynomialDecay"
        self.power = float(power)
        self.end_lr = float(end_lr)
        self.gamma = float(gamma)
        self.milestones = sorted(int(m) for m in (milestones or []))
        self.warmup_steps = max(0, int(warmup_steps))
        self.warmup_start_lr = float(warmup_start_lr)
        self.last_lr = self.base_lr

    def lr_at(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            ratio = step / float(self.warmup_steps)
            return self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * ratio

        progress_step = step - self.warmup_steps
        progress_total = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress_step / float(progress_total)))

        policy = self.policy
        if policy in ("PolynomialDecay", "Polynomial", "PolyDecay"):
            return (self.base_lr - self.end_lr) * ((1.0 - progress) ** self.power) + self.end_lr
        if policy in ("CosineAnnealingDecay", "CosineDecay", "Cosine"):
            import math

            return self.end_lr + (self.base_lr - self.end_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        if policy in ("PiecewiseDecay", "Piecewise", "MultiStepDecay", "StepDecay"):
            decays = sum(1 for m in self.milestones if step >= m)
            return self.base_lr * (self.gamma ** decays)
        if policy in ("ExponentialDecay", "ExpDecay"):
            return self.base_lr * (self.gamma ** progress)
        if policy in ("ConstLR", "Constant"):
            return self.base_lr
        # Unknown policy: hold the base LR rather than silently decaying to 0.
        return self.base_lr

    def step(self, step: int) -> float:
        lr = self.lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_lr = lr
        return lr


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    base_lr: float,
    total_steps: int,
) -> LrScheduler:
    """Build an `LrScheduler` from a Paddle-style `lr_scheduler:` block.

    Note the two frameworks disagree about what "warmup start" means, and both
    spellings appear in configs this platform generates:
      * PaddleSeg's `warmup_start_lr` is an **absolute** learning rate.
      * PaddleDetection's `LinearWarmup.start_factor` is a **fraction** of
        `base_lr`.
    Treating them identically makes a detection warmup start ~`base_lr` times too
    high, so they are read from distinct keys.
    """
    start_lr = cfg.get("warmup_start_lr")
    if start_lr is None and cfg.get("start_factor") is not None:
        start_lr = float(cfg["start_factor"]) * float(base_lr)

    return LrScheduler(
        optimizer,
        base_lr=base_lr,
        total_steps=total_steps,
        policy=str(cfg.get("type") or cfg.get("name") or "PolynomialDecay"),
        power=float(cfg.get("power", 0.9) or 0.9),
        end_lr=float(cfg.get("end_lr", cfg.get("eta_min", 0.0)) or 0.0),
        gamma=float(cfg.get("gamma", 0.1) or 0.1),
        milestones=cfg.get("milestones") or cfg.get("boundaries") or cfg.get("decay_epochs"),
        warmup_steps=int(cfg.get("warmup_iters", cfg.get("warmup_steps", cfg.get("warmup_epochs", 0))) or 0),
        warmup_start_lr=float(start_lr or 0.0),
    )


def amp_context(enabled: bool, device: torch.device):
    """`autocast` context that works across the torch 1.x / 2.4 API split."""
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except (AttributeError, TypeError):  # torch < 2.0
        return torch.cuda.amp.autocast(enabled=True)


def amp_scaler(enabled: bool, device: torch.device):
    """`GradScaler` that works across the torch 1.x / 2.4 API split."""
    active = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=active)
    except (AttributeError, TypeError):  # torch < 2.0
        return torch.cuda.amp.GradScaler(enabled=active)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

# The platform's checkpoint browser (`/api/checkpoints`) discovers weights by
# looking for `<save_dir>/<subdir>/model.pt`, mirroring PaddleSeg's
# `<save_dir>/best_model/model.pdparams` layout. Keep this constant in sync
# with `TORCH_WEIGHT_FILE` in `src/lib/frameworks.ts`.
WEIGHT_FILE = "model.pt"
OPT_FILE = "optimizer.pt"


def save_checkpoint(
    save_dir: str,
    subdir: str,
    model: torch.nn.Module,
    meta: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> str:
    """Write `<save_dir>/<subdir>/model.pt` and return its path.

    `meta` is embedded in the checkpoint so `val.py` / `predict.py` / `export.py`
    can rebuild the exact same network from the weights file alone, without
    having to trust that the YAML next to it was never edited.
    """
    target_dir = os.path.join(save_dir, subdir)
    os.makedirs(target_dir, exist_ok=True)
    weights_path = os.path.join(target_dir, WEIGHT_FILE)
    payload = {"state_dict": model.state_dict()}
    payload.update(meta)
    torch.save(payload, weights_path)
    if optimizer is not None:
        torch.save({"state_dict": optimizer.state_dict()}, os.path.join(target_dir, OPT_FILE))
    return weights_path


def load_checkpoint(path: str, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load a checkpoint written by `save_checkpoint`, or a bare state_dict.

    Accepting a bare `state_dict` keeps hand-exported or third-party weights
    usable, which matters because users paste arbitrary paths into the
    validation UI.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("Weights file not found: {}".format(path))
    # `weights_only=True` is the safe default but only exists on torch >= 1.13
    # and rejects the plain dicts we store; try it first, then fall back.
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload
    return {"state_dict": payload}


def prune_checkpoints(save_dir: str, keep: int, protect: Tuple[str, ...] = ("best_model",)) -> None:
    """Keep only the newest `keep` step checkpoints (never touching `protect`).

    Segmentation runs can save every few hundred iterations; without this a long
    job silently fills the user's storage quota.
    """
    if keep <= 0 or not os.path.isdir(save_dir):
        return
    entries = []
    for name in os.listdir(save_dir):
        if name in protect:
            continue
        full = os.path.join(save_dir, name)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, WEIGHT_FILE)):
            entries.append((os.path.getmtime(full), full))
    entries.sort(reverse=True)
    for _, stale in entries[keep:]:
        try:
            for child in os.listdir(stale):
                os.remove(os.path.join(stale, child))
            os.rmdir(stale)
        except OSError as exc:
            log("WARNING: could not prune checkpoint {}: {}".format(stale, exc))
