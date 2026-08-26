"""
Train / evaluate / predict / export for `framework: TorchAnomaly`.

Everything model-related is anomalib's; this module owns the seams between
anomalib and the platform:

* `Engine.from_config` is used instead of the `anomalib` CLI, so we control the
  callbacks, the checkpoint location and stdout.
* The model's own `configure_pre_processor(image_size=...)` provides the resize.
  Building a transform by hand here would be how EfficientAd ends up with the
  `Normalize` step it refuses to train with.
* `Engine.fit()` puts its outputs in `<default_root_dir>/<Model>/<dataset>/<category>/vN`,
  so the checkpoint the platform expects at `<save_dir>/best_model/model.ckpt`
  is copied there afterwards. `bestWeightsPath()` in `src/lib/job-commands.ts` is
  the other half of that contract.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from .. import logger as L
from . import config as adcfg
from . import evaluator as adeval

# `.logger` is NOT imported here: it subclasses a Lightning `Callback`, so a
# missing anomalib install would surface as `ModuleNotFoundError: lightning`
# raised from an import statement, instead of the actionable message
# `_import_anomalib` produces. Every function imports it after that check.

#: Subdirectory of `save_dir` holding the checkpoint the platform reads.
BEST_DIR = "best_model"
#: Must match `ANOMALY_WEIGHT_FILE` in `src/lib/frameworks.ts`.
WEIGHT_FILE = "model.ckpt"
#: Metrics summary written next to the checkpoint.
METRICS_FILE = "metrics.json"


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def _import_anomalib() -> None:
    """Fail with an actionable message when anomalib is missing.

    This is the single most likely first-run error: the TorchAnomaly framework
    needs its own environment, because anomalib pins Lightning and jsonargparse
    and a plain torch env (the one TorchSeg/TorchDet use) has none of it.
    """
    try:
        import anomalib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "anomalib is not installed in this interpreter. TorchAnomaly needs a "
            "dedicated environment:\n"
            "    pip install anomalib==2.6.*\n"
            "then point Settings -> Framework Python Environments -> TorchAnomaly "
            "at that interpreter."
        ) from exc


def _model_max_epochs(model: Any) -> Optional[int]:
    """`max_epochs` the model forces on the trainer, if any.

    PatchCore and PaDiM publish `{'max_epochs': 1, ...}`, which is how a
    memory-bank model announces "one pass over the data, no optimisation". The
    validation cadence and the progress total both depend on knowing this.
    """
    try:
        args = model.trainer_arguments or {}
        value = args.get("max_epochs")
        return int(value) if value is not None else None
    except Exception:  # noqa: BLE001
        return None


def _batches_per_epoch(datamodule: Any) -> int:
    try:
        datamodule.setup("fit")
        return len(datamodule.train_dataloader())
    except Exception as exc:  # noqa: BLE001
        L.log("WARNING: could not determine the number of training batches ({}).".format(exc))
        return 0


def _apply_pre_processor(model: Any, platform_cfg: Dict[str, Any]) -> None:
    """Set the input size through the model's own factory."""
    size = adcfg.image_size(platform_cfg)
    if size is None:
        return
    crop = adcfg.center_crop_size(platform_cfg)
    factory = type(model).configure_pre_processor
    try:
        # Only PatchCore takes a centre crop (the paper's 256 -> 224 recipe).
        model.pre_processor = (
            factory(image_size=size, center_crop_size=crop) if crop else factory(image_size=size)
        )
    except TypeError:
        # A model whose factory does not accept center_crop_size.
        if crop:
            L.log(
                "WARNING: {} does not support center_crop_size; ignoring it.".format(type(model).__name__)
            )
        model.pre_processor = factory(image_size=size)
    L.log("Input size: {}x{} (h x w){}".format(size[0], size[1], " crop {}".format(crop) if crop else ""))


def _check_tiling(anomalib_cfg: Dict[str, Any], model: Any) -> None:
    """Refuse an impossible tiling request with a message naming alternatives.

    anomalib's own error is `ValueError: Model does not support tiling.`, which
    does not say which models do.
    """
    if not adcfg.tiling_enabled(anomalib_cfg):
        return
    if hasattr(getattr(model, "model", None), "tiler"):
        return
    raise ValueError(
        "{} does not support input tiling. Only PaDiM, PatchCore, "
        "ReverseDistillation and STFPM do. Either disable tiling in the training "
        "config or switch model.".format(type(model).__name__)
    )


def _prepare(cfg: Dict[str, Any], args: Any, save_dir: str) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """Build `(engine, model, datamodule, platform_cfg)` from a merged config."""
    _import_anomalib()
    from anomalib.engine import Engine

    anomalib_cfg, platform_cfg = adcfg.split_platform_block(cfg)
    adcfg.validate(anomalib_cfg)
    anomalib_cfg = adcfg.apply_cli_overrides(anomalib_cfg, args)
    resolved_path = adcfg.write_resolved(anomalib_cfg, save_dir)
    L.log("Resolved anomalib config: {}".format(resolved_path))

    engine, model, datamodule = Engine.from_config(config_path=resolved_path)
    _apply_pre_processor(model, platform_cfg)
    _check_tiling(anomalib_cfg, model)
    return engine, model, datamodule, platform_cfg


def _has_masks(datamodule: Any) -> bool:
    """Whether the dataset declares a mask directory (i.e. pixel metrics exist)."""
    return bool(getattr(datamodule, "mask_dir", None))


def _trainer_args(engine: Any) -> Dict[str, Any]:
    """The Trainer keyword arguments an `Engine` will use, for in-place edits.

    anomalib builds its `Trainer` lazily, from `Engine._cache.args`, at the start
    of `fit()`. There is no public API to add a callback or change the validation
    cadence to an `Engine` that `from_config` handed back, and both have to be
    decided *after* the datamodule exists (the cadence depends on the batch
    count). Poking the cache is therefore deliberate, and isolated here so a
    future anomalib release breaks in exactly one place with a clear message.
    """
    cache = getattr(engine, "_cache", None)
    args = getattr(cache, "args", None)
    if not isinstance(args, dict):
        raise RuntimeError(
            "This anomalib release does not expose Engine._cache.args, which the "
            "adapter uses to register its logging callback and validation "
            "cadence. Pin anomalib==2.6.* or update torchtrain/torchtrain/ad/runner.py."
        )
    return args


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train(cfg: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Train an anomaly-detection model and mirror the best checkpoint."""
    save_dir = args.save_dir or "output/anomaly"
    os.makedirs(save_dir, exist_ok=True)

    engine, model, datamodule, platform_cfg = _prepare(cfg, args, save_dir)

    # anomalib's subclass, not Lightning's: `Engine._setup_anomalib_callbacks`
    # checks `isinstance(c, anomalib...ModelCheckpoint)` before adding its own, so
    # passing Lightning's plain class would give the trainer two checkpoint
    # callbacks writing to two different directories.
    from anomalib.callbacks.checkpoint import ModelCheckpoint

    from .logger import PaddleStyleLogger

    include_pixel = _has_masks(datamodule)
    adeval.attach(model, include_pixel=include_pixel)
    if not include_pixel:
        L.log(
            "No mask_dir in the dataset config: pixel_auroc / pixel_f1 will not be "
            "reported. Add masks for the defect images to get localisation metrics."
        )

    best_metric = adcfg.best_metric(platform_cfg)
    if not include_pixel and best_metric.lower().startswith("pixel"):
        L.log(
            "WARNING: best_metric is {} but the dataset has no masks; falling back "
            "to image_AUROC.".format(best_metric)
        )
        best_metric = "image_AUROC"

    # Validation cadence has to be computed, not configured: see
    # `config.resolve_val_args`.
    trainer_args = _trainer_args(engine)
    batches = _batches_per_epoch(datamodule)
    val_args = adcfg.resolve_val_args(platform_cfg, batches, _model_max_epochs(model))
    trainer_args.update(val_args)
    L.log("Validation cadence: {}".format(val_args))

    progress = PaddleStyleLogger(
        log_iter=adcfg.log_iter(platform_cfg),
        best_metric=best_metric,
        total_steps_hint=int(trainer_args.get("max_steps") or 0),
    )
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(save_dir, BEST_DIR),
        filename="model",
        monitor=best_metric,
        mode="max",
        save_top_k=1,
        save_last=False,
        # Keeps the filename literally `model.ckpt` instead of embedding the
        # metric name and value.
        auto_insert_metric_name=False,
    )
    trainer_args["callbacks"] = list(trainer_args.get("callbacks") or []) + [progress, checkpoint]

    engine.fit(model=model, datamodule=datamodule)

    weights = _finalize_checkpoint(save_dir, checkpoint)
    summary: Dict[str, Any] = {
        "framework": "TorchAnomaly",
        "model": type(model).__name__,
        "best_metric": best_metric,
        "best_value": progress.best["value"],
        "best_step": progress.best["step"],
        "steps": progress.steps_done,
        "weights": weights,
    }

    # A final test pass on the held-out half of the test split gives the numbers
    # the validation page shows, and prints one last [EVAL] line.
    if getattr(args, "do_eval", False):
        try:
            results = engine.test(model=model, datamodule=datamodule, verbose=False)
            metrics = _flatten_test_results(results)
            metrics.setdefault("threshold", PaddleStyleLogger.read_threshold(model) or float("nan"))
            PaddleStyleLogger.emit_eval(metrics, _test_image_count(datamodule))
            summary["test_metrics"] = metrics
        except Exception as exc:  # noqa: BLE001 - a failed test must not fail the run
            L.log("WARNING: final evaluation failed: {}".format(exc))

    _write_metrics(save_dir, summary)
    L.log(
        "Best {} = {} at iter {}.".format(
            best_metric,
            "{:.4f}".format(progress.best["value"]) if progress.best["value"] is not None else "n/a",
            progress.best["step"] or "n/a",
        )
    )
    L.log("Final weights: {}".format(weights or "(none written)"))
    return summary


def _finalize_checkpoint(save_dir: str, checkpoint: Any) -> Optional[str]:
    """Put the best checkpoint exactly where the platform looks for it.

    `ModelCheckpoint` appends `.ckpt` and, depending on version and metric
    availability, may write `model-v1.ckpt` or nothing at all. The platform
    resolves a single fixed path, so normalise here rather than teaching the
    web app about Lightning's naming.
    """
    best_dir = os.path.join(save_dir, BEST_DIR)
    target = os.path.join(best_dir, WEIGHT_FILE)
    source = getattr(checkpoint, "best_model_path", "") or getattr(checkpoint, "last_model_path", "")

    if source and os.path.isfile(source):
        if os.path.abspath(source) != os.path.abspath(target):
            shutil.copyfile(source, target)
        return target

    # No monitored metric ever appeared (e.g. a dataset with no defect images).
    # A run with no checkpoint at all is useless, so fall back to any .ckpt the
    # callback wrote, and say what happened.
    if os.path.isdir(best_dir):
        candidates = sorted(f for f in os.listdir(best_dir) if f.endswith(".ckpt"))
        if candidates:
            fallback = os.path.join(best_dir, candidates[-1])
            if os.path.abspath(fallback) != os.path.abspath(target):
                shutil.copyfile(fallback, target)
            L.log("WARNING: no monitored metric was recorded; kept {}.".format(candidates[-1]))
            return target
    L.log(
        "WARNING: no checkpoint was written. This usually means validation never "
        "ran, so the monitored metric never existed."
    )
    return None


def _write_metrics(save_dir: str, summary: Dict[str, Any]) -> None:
    path = os.path.join(save_dir, BEST_DIR, METRICS_FILE)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
    except OSError as exc:
        L.log("WARNING: could not write {}: {}".format(path, exc))


def _flatten_test_results(results: Any) -> Dict[str, float]:
    """`engine.test` returns a list of dicts; normalise to our metric names."""
    from .logger import _METRIC_ALIASES, _as_float  # noqa: PLC0415 - internal reuse

    out: Dict[str, float] = {}
    entries = results if isinstance(results, list) else [results]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            number = _as_float(value)
            if number is not None:
                out[_METRIC_ALIASES.get(str(key).lower(), str(key).lower())] = number
    return out


def _test_image_count(datamodule: Any) -> int:
    try:
        return len(datamodule.test_data)
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


def evaluate(cfg: Dict[str, Any], args: Any, weights: str) -> Dict[str, float]:
    """Evaluate a checkpoint and print the `[EVAL]` line the platform parses."""
    save_dir = os.path.dirname(os.path.dirname(os.path.abspath(weights))) or "."
    engine, model, datamodule, platform_cfg = _prepare(cfg, args, save_dir)

    from .logger import PaddleStyleLogger

    include_pixel = _has_masks(datamodule)
    adeval.attach(model, include_pixel=include_pixel)

    results = engine.test(model=model, datamodule=datamodule, ckpt_path=weights, verbose=False)
    metrics = _flatten_test_results(results)
    threshold = PaddleStyleLogger.read_threshold(model)
    if threshold is not None:
        metrics.setdefault("threshold", threshold)
    PaddleStyleLogger.emit_eval(metrics, _test_image_count(datamodule))
    return metrics


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def collect_images(image_path: Optional[str]) -> List[str]:
    if not image_path:
        raise ValueError("--image_path is required for prediction.")
    if os.path.isfile(image_path):
        return [image_path]
    if not os.path.isdir(image_path):
        raise FileNotFoundError("No such file or directory: {}".format(image_path))
    files = [
        os.path.join(image_path, name)
        for name in sorted(os.listdir(image_path))
        if name.lower().endswith(IMAGE_EXTS)
    ]
    if not files:
        raise FileNotFoundError("No images found in {}".format(image_path))
    return files


def predict(cfg: Dict[str, Any], args: Any, weights: str, save_dir: str) -> Dict[str, Any]:
    """Score images and write heatmaps plus a per-image `scores.json`.

    The platform's inference viewer lists images found up to one directory deep
    under the output path (`findInferenceImages` in
    `src/app/api/validation-jobs/route.ts`), so the visualiser is pointed at
    `<save_dir>/heatmaps`. Left alone it would write to
    `<save_dir>/<Model>/<dataset>/<category>/latest/images`, five levels down,
    and the UI would show "no results" for a run that worked.

    `scores.json` carries what a heatmap cannot: the per-image anomaly score, the
    threshold it was compared against, and the verdict.
    """
    os.makedirs(save_dir, exist_ok=True)
    engine, model, _datamodule, _platform = _prepare(cfg, args, save_dir)

    from .logger import PaddleStyleLogger

    image_path = getattr(args, "image_path", None)
    images = collect_images(image_path)
    L.log("Predicting {} image(s)".format(len(images)))

    heatmap_dir = os.path.join(save_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    if getattr(model, "visualizer", None) is not None:
        model.visualizer.output_dir = heatmap_dir

    # `data_path` makes anomalib build the PredictDataset and a batch-size-1
    # dataloader itself, which also side-steps the collate error that a folder of
    # differently sized images would otherwise cause.
    predictions = engine.predict(
        model=model, data_path=image_path, ckpt_path=weights, return_predictions=True
    )

    threshold = PaddleStyleLogger.read_threshold(model)
    records: List[Dict[str, Any]] = []
    for batch in predictions or []:
        records.extend(_records_from_batch(batch, threshold))

    payload = {
        "model": type(model).__name__,
        "threshold": threshold,
        "count": len(records),
        "anomalous": sum(1 for r in records if r.get("pred_label")),
        "images": records,
    }
    scores_path = os.path.join(save_dir, "scores.json")
    with open(scores_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    L.log(
        "Flagged {}/{} image(s) as anomalous (threshold {}). Scores: {}".format(
            payload["anomalous"],
            payload["count"],
            "{:.4f}".format(threshold) if threshold is not None else "n/a",
            scores_path,
        )
    )
    return payload


def _records_from_batch(batch: Any, threshold: Optional[float]) -> List[Dict[str, Any]]:
    """Per-image rows from one predict batch, tolerant of shape differences."""
    from .logger import _as_float  # noqa: PLC0415 - internal reuse

    paths = getattr(batch, "image_path", None) or []
    if isinstance(paths, str):
        paths = [paths]
    scores = getattr(batch, "pred_score", None)
    labels = getattr(batch, "pred_label", None)

    out: List[Dict[str, Any]] = []
    for i, path in enumerate(paths):
        score = _as_float(scores[i]) if scores is not None and len(scores) > i else None
        label = labels[i] if labels is not None and len(labels) > i else None
        out.append(
            {
                "image": os.path.basename(str(path)),
                "path": str(path),
                "score": score,
                "threshold": threshold,
                "pred_label": bool(_as_float(label)) if label is not None else None,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export(cfg: Dict[str, Any], args: Any, weights: str, save_dir: str) -> Optional[str]:
    """Export to TorchScript / ONNX / OpenVINO through anomalib's exporter."""
    _import_anomalib()
    from anomalib.deploy import ExportType

    requested = (getattr(args, "format", None) or "torch").lower()
    mapping = {
        "torch": ExportType.TORCH,
        "torchscript": ExportType.TORCH,
        "pt": ExportType.TORCH,
        "onnx": ExportType.ONNX,
        "openvino": ExportType.OPENVINO,
    }
    if requested not in mapping:
        raise ValueError(
            "Unsupported export format {!r}. Use torch, onnx or openvino.".format(requested)
        )

    engine, model, datamodule, _platform = _prepare(cfg, args, save_dir)
    path = engine.export(
        model=model,
        export_type=mapping[requested],
        export_root=save_dir,
        ckpt_path=weights,
        datamodule=datamodule if mapping[requested] == ExportType.OPENVINO else None,
    )
    L.log("Exported to {}".format(path))
    return str(path) if path else None
