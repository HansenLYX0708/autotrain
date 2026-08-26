"""
Config translation between the platform's merged YAML and anomalib.

The merged document the platform produces is an anomalib config plus one extra
top-level block:

    model:      { class_path: anomalib.models.Patchcore, init_args: {...} }
    data:       { class_path: anomalib.data.Folder,      init_args: {...} }
    trainer:    { max_steps: 8000, ... }
    autotrain:  { image_size: [256, 256], log_iter: 20, val_interval: 500, ... }

`autotrain:` exists because a few things are either awkward or dangerous to
express in anomalib's own schema:

* **image_size** would require nesting a `torchvision.transforms.v2.Compose`
  through jsonargparse's `class_path` plumbing. Worse, it would let a user add a
  `Normalize` step, which makes EfficientAd refuse to train
  (`Transforms for EfficientAd should not contain Normalize`). Calling the model
  class's own `configure_pre_processor(image_size=...)` cannot get this wrong.
* **val_interval** cannot be mapped onto Lightning's `val_check_interval`
  without knowing the batch count: an int larger than the number of training
  batches raises unless `check_val_every_n_epoch=None`, and setting that to None
  disables epoch-end validation — which is the only validation a one-epoch
  memory-bank model (PatchCore, PaDiM) would ever get. See `resolve_val_args`.

`Engine.from_config` takes a *path*, so the sanitised config is written back to
disk next to the job's outputs; keeping it also makes a failed run reproducible
by hand.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import yaml

#: Top-level key holding platform-only settings.
PLATFORM_KEY = "autotrain"

#: File the sanitised anomalib config is written to, inside `save_dir`.
RESOLVED_CONFIG_NAME = "anomalib_config.yaml"


def split_platform_block(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return `(anomalib_cfg, autotrain_cfg)`.

    The input is not mutated: the caller still has the merged document for
    logging and error messages.
    """
    platform = cfg.get(PLATFORM_KEY) or {}
    if not isinstance(platform, dict):
        raise ValueError(
            "`autotrain:` must be a mapping of platform settings, got {}".format(type(platform).__name__)
        )
    anomalib_cfg = {k: v for k, v in cfg.items() if k != PLATFORM_KEY}
    return anomalib_cfg, dict(platform)


def validate(anomalib_cfg: Dict[str, Any]) -> None:
    """Fail early, with a message that names the config block at fault.

    Without this, a missing `data:` block surfaces as a jsonargparse usage dump
    several hundred lines long, which is what the user would see in the job log.
    """
    model = anomalib_cfg.get("model")
    if not isinstance(model, dict) or not model.get("class_path"):
        raise ValueError(
            "The model config must provide `model.class_path` (e.g. "
            "`anomalib.models.Patchcore`). Check the model configuration attached "
            "to this job."
        )
    data = anomalib_cfg.get("data")
    if not isinstance(data, dict) or not data.get("class_path"):
        raise ValueError(
            "The dataset config must provide `data.class_path` (e.g. "
            "`anomalib.data.Folder`). Check the dataset attached to this job."
        )
    init_args = data.get("init_args") or {}
    if not init_args.get("normal_dir"):
        raise ValueError(
            "`data.init_args.normal_dir` is required: an anomaly-detection job "
            "trains on normal images only, so it must know where they are."
        )
    root = init_args.get("root")
    if root and not os.path.isdir(str(root)):
        raise FileNotFoundError("Dataset root does not exist: {}".format(root))


def apply_cli_overrides(anomalib_cfg: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Fold the platform's CLI flags into the trainer block.

    The runner is invoked with the same flags as the other frameworks
    (`--save_dir`, `--amp`, `--use_vdl`, `--batch_size`, ...), and those must win
    over the YAML — the platform's job dialog is the more specific statement of
    intent.
    """
    trainer = dict(anomalib_cfg.get("trainer") or {})

    if getattr(args, "cpu", False):
        trainer["accelerator"] = "cpu"
    if getattr(args, "amp", False):
        trainer["precision"] = "16-mixed"
    iters = getattr(args, "iters", None)
    if iters:
        trainer["max_steps"] = int(iters)
    epochs = getattr(args, "epochs", None)
    if epochs:
        trainer["max_epochs"] = int(epochs)

    # A rich progress bar writes carriage returns, which the platform's stdout
    # parser cannot use and which bloat the stored log. Never allow it on.
    trainer["enable_progress_bar"] = False
    # Sanity-checking validation before training would emit an [EVAL] row with
    # meaningless numbers at iteration 0.
    trainer.setdefault("num_sanity_val_steps", 0)

    anomalib_cfg["trainer"] = trainer

    batch_size = getattr(args, "batch_size", None)
    num_workers = getattr(args, "num_workers", None)
    if batch_size or num_workers is not None:
        data = dict(anomalib_cfg.get("data") or {})
        init_args = dict(data.get("init_args") or {})
        if batch_size:
            init_args["train_batch_size"] = int(batch_size)
        if num_workers is not None:
            init_args["num_workers"] = int(num_workers)
        data["init_args"] = init_args
        anomalib_cfg["data"] = data

    return anomalib_cfg


def write_resolved(anomalib_cfg: Dict[str, Any], save_dir: str) -> str:
    """Write the sanitised config and return its path."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, RESOLVED_CONFIG_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(anomalib_cfg, handle, sort_keys=False, allow_unicode=True)
    return path


def image_size(platform_cfg: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """`autotrain.image_size` as `(height, width)`, or None when unset.

    Note the order flip: the platform stores `[width, height]` (matching the rest
    of its config generators), while every torchvision transform takes
    `(height, width)`.
    """
    raw = platform_cfg.get("image_size")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return (int(raw), int(raw))
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return (int(raw[1]), int(raw[0]))
    if isinstance(raw, (list, tuple)) and len(raw) == 1:
        return (int(raw[0]), int(raw[0]))
    raise ValueError("autotrain.image_size must be a number or [width, height], got {!r}".format(raw))


def center_crop_size(platform_cfg: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    raw = platform_cfg.get("center_crop_size")
    if raw in (None, 0):
        return None
    if isinstance(raw, (int, float)):
        return (int(raw), int(raw))
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return (int(raw[1]), int(raw[0]))
    raise ValueError("autotrain.center_crop_size must be a number or [width, height], got {!r}".format(raw))


def resolve_val_args(
    platform_cfg: Dict[str, Any],
    batches_per_epoch: int,
    model_max_epochs: Optional[int],
) -> Dict[str, Any]:
    """Turn `autotrain.val_interval` into a *valid* pair of Trainer arguments.

    Three cases, in order of precedence:

    1. **One-epoch models** (PatchCore, PaDiM declare `max_epochs = 1` in their
       `trainer_arguments`). Step-based validation can never fire, so validate at
       the end of the single epoch.
    2. **Interval fits inside an epoch.** Use it directly and keep epoch-end
       validation on.
    3. **Interval spans several epochs** — common here, because a normal-image
       training set can be a few dozen images, i.e. a handful of batches.
       Lightning only permits that with `check_val_every_n_epoch=None`, which
       also disables the epoch-end check; that is fine, because step-based
       validation will fire.
    """
    interval = int(platform_cfg.get("val_interval") or 0)

    if model_max_epochs is not None and model_max_epochs <= 1:
        return {"check_val_every_n_epoch": 1}

    if interval <= 0:
        return {"check_val_every_n_epoch": 1}

    if batches_per_epoch > 0 and interval <= batches_per_epoch:
        return {"val_check_interval": interval, "check_val_every_n_epoch": 1}

    return {"val_check_interval": interval, "check_val_every_n_epoch": None}


def best_metric(platform_cfg: Dict[str, Any]) -> str:
    """Metric the checkpoint callback monitors. Defaults to image-level AUROC."""
    value = platform_cfg.get("best_metric")
    return str(value) if value else "image_AUROC"


def log_iter(platform_cfg: Dict[str, Any]) -> int:
    value = int(platform_cfg.get("log_iter") or 0)
    return value if value > 0 else 20


def tiling_enabled(anomalib_cfg: Dict[str, Any]) -> bool:
    """Whether the trainer callbacks include an enabled tiler.

    Used to produce an actionable error *before* training starts when the chosen
    model has no `tiler` attribute, instead of anomalib's bare
    "Model does not support tiling."
    """
    callbacks = anomalib_cfg.get("trainer", {}).get("callbacks")
    entries: List[Any] = callbacks if isinstance(callbacks, list) else ([callbacks] if callbacks else [])
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if "TilerConfiguration" in str(entry.get("class_path", "")):
            init_args = entry.get("init_args") or {}
            return bool(init_args.get("enable", False))
    return False
