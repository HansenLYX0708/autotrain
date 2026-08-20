"""
Segmentation training / evaluation loop.

Semantics follow PaddleSeg, because the platform's progress bar, best-checkpoint
tracking and log parsing were all built against it:

* Training length is measured in **iterations** (`iters`), not epochs. The
  reported `epoch` is derived (`iter // iters_per_epoch + 1`) purely for display.
* Evaluation runs every `save_interval` iterations when `--do_eval` is passed.
* Checkpoints land in `<save_dir>/iter_<N>/model.pt`, and the best-scoring one is
  mirrored to `<save_dir>/best_model/model.pt` — the same layout the platform's
  `/api/checkpoints` endpoint walks for PaddleSeg.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .. import config as cfgmod
from .. import logger as L
from ..utils import (
    AverageMeter,
    Timer,
    amp_context,
    amp_scaler,
    build_lr_scheduler,
    build_optimizer,
    load_checkpoint,
    memory_stats_mb,
    prune_checkpoints,
    resolve_device,
    save_checkpoint,
    set_seed,
)
from . import dataset as dsmod
from . import losses as lossmod
from . import models as modelmod
from .metrics import SegMetric
from .transforms import IGNORE_INDEX


class SegRunSetup:
    """Everything derived from a config, in one place.

    Built once and shared by `train`, `evaluate` and `predict` so the three
    entrypoints can never disagree about how a config maps onto a model.
    """

    def __init__(self, cfg: Dict[str, Any], args: Any) -> None:
        self.cfg = cfg
        self.model_cfg = cfg.get("model") or {}
        self.loss_cfg = cfg.get("loss") or {}

        self.num_logits = lossmod.count_logits(self.loss_cfg)
        self.num_classes = int(
            self.model_cfg.get("num_classes")
            or (cfg.get("train_dataset") or {}).get("num_classes")
            or (cfg.get("val_dataset") or {}).get("num_classes")
            or cfg.get("num_classes")
            or 0
        )
        if self.num_classes < 1:
            raise ValueError(
                "num_classes could not be determined. Set it under `model:` or in the dataset block."
            )

        self.iters = int(getattr(args, "iters", None) or cfgmod.get_int(cfg, "iters", default=1000) or 1000)
        self.batch_size = int(
            getattr(args, "batch_size", None) or cfgmod.get_int(cfg, "batch_size", default=2) or 2
        )
        self.num_workers = getattr(args, "num_workers", None)
        if self.num_workers is None:
            self.num_workers = cfgmod.get_int(cfg, "num_workers", "worker_num", default=0) or 0
        self.num_workers = max(0, int(self.num_workers))

        # PaddleSeg keeps these three on the CLI, not in the YAML; accept both so
        # a hand-edited config still works.
        self.log_iters = int(
            getattr(args, "log_iters", None) or cfgmod.get_int(cfg, "log_iters", "log_iter", default=10) or 10
        )
        self.save_interval = int(
            getattr(args, "save_interval", None)
            or cfgmod.get_int(cfg, "save_interval", default=max(1, self.iters // 10))
            or max(1, self.iters // 10)
        )
        self.use_gpu = cfgmod.get_bool(cfg, "use_gpu", default=True)
        self.align_corners = bool(self.model_cfg.get("align_corners", False))
        self.default_size = cfgmod.size_pair(
            (cfg.get("train_dataset") or {}).get("target_size"), default=[512, 512]
        )

    def build_model(self) -> modelmod.SegModel:
        return modelmod.build_model(self.model_cfg, self.num_classes, self.num_logits)

    def build_criterion(self) -> lossmod.SegCriterion:
        return lossmod.build_losses(self.loss_cfg, self.num_logits, IGNORE_INDEX)

    def build_train_dataset(self) -> dsmod.SegDataset:
        return dsmod.build_dataset(
            self.cfg.get("train_dataset") or {}, "train", self.num_classes, self.default_size
        )

    def build_val_dataset(self) -> Optional[dsmod.SegDataset]:
        block = self.cfg.get("val_dataset")
        if not block:
            return None
        return dsmod.build_dataset(block, "val", self.num_classes, self.default_size)

    def class_names(self) -> List[str]:
        root = (self.cfg.get("val_dataset") or self.cfg.get("train_dataset") or {}).get("dataset_root") or ""
        return dsmod.read_class_names(str(root), self.num_classes)

    def meta(self) -> Dict[str, Any]:
        return modelmod.model_meta(self.model_cfg, self.num_classes, self.num_logits)


def _make_loader(dataset: dsmod.SegDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle and len(dataset) > batch_size,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dsmod.collate_variable_size,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def evaluate(
    model: modelmod.SegModel,
    dataset: dsmod.SegDataset,
    device: torch.device,
    num_classes: int,
    num_workers: int = 0,
    print_detail: bool = True,
) -> Dict[str, Any]:
    """Run validation and (optionally) print the PaddleSeg `[EVAL]` block."""
    model.eval()
    metric = SegMetric(num_classes)
    # Batch size 1: validation images are commonly at native resolution and can
    # differ in size, and a per-image pass keeps peak memory predictable.
    loader = _make_loader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    if print_detail:
        L.log("Start evaluating (total_samples: {}, total_iters: {})...".format(len(dataset), len(loader)))

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        pred = logits[0].argmax(dim=1)
        metric.update(pred, labels)

    summary = metric.summary()
    if print_detail:
        L.seg_eval(
            num_images=len(dataset),
            miou=float(summary["mIoU"]),
            acc=float(summary["acc"]),
            kappa=float(summary["kappa"]),
            dice=float(summary["dice"]),
            class_iou=summary["class_iou"],  # type: ignore[arg-type]
            class_precision=summary["class_precision"],  # type: ignore[arg-type]
            class_recall=summary["class_recall"],  # type: ignore[arg-type]
        )
    model.train()
    return summary


def train(cfg: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Train a segmentation model. Returns the best evaluation summary."""
    set_seed(int(getattr(args, "seed", 1234) or 1234))
    setup = SegRunSetup(cfg, args)
    device = resolve_device(setup.use_gpu)

    save_dir = getattr(args, "save_dir", None) or cfg.get("save_dir") or "output"
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = setup.build_train_dataset()
    val_dataset = setup.build_val_dataset() if getattr(args, "do_eval", False) else None
    class_names = setup.class_names()

    model = setup.build_model().to(device)
    criterion = setup.build_criterion().to(device)

    base_lr = float(
        cfgmod.get_num(cfg.get("lr_scheduler") or {}, "learning_rate", "base_lr", default=0.01) or 0.01
    )
    optimizer = build_optimizer(model.parameters(), cfg.get("optimizer") or {}, base_lr)
    scheduler = build_lr_scheduler(optimizer, cfg.get("lr_scheduler") or {}, base_lr, setup.iters)

    use_amp = bool(getattr(args, "amp", False) or cfgmod.get_bool(cfg, "use_amp", default=False))
    scaler = amp_scaler(use_amp, device)

    iters_per_epoch = max(1, len(train_dataset) // max(1, setup.batch_size))
    L.log("---------------- Configuration ----------------")
    L.log("architecture     : {}".format(setup.model_cfg.get("type", "UNet")))
    L.log("num_classes      : {} ({})".format(setup.num_classes, ", ".join(class_names)))
    L.log("num_logits       : {}".format(setup.num_logits))
    L.log("iters            : {}".format(setup.iters))
    L.log("batch_size       : {}".format(setup.batch_size))
    L.log("num_workers      : {}".format(setup.num_workers))
    L.log("base_lr          : {}".format(base_lr))
    L.log("lr_scheduler     : {}".format(scheduler.policy))
    L.log("optimizer        : {}".format((cfg.get("optimizer") or {}).get("type", "SGD")))
    L.log("use_amp          : {}".format(use_amp))
    L.log("train samples    : {} ({} iters/epoch)".format(len(train_dataset), iters_per_epoch))
    L.log("val samples      : {}".format(len(val_dataset) if val_dataset else 0))
    L.log("save_dir         : {}".format(os.path.abspath(save_dir)))
    L.log("save_interval    : {}".format(setup.save_interval))
    L.log("-----------------------------------------------")

    writer = _maybe_tensorboard(getattr(args, "use_vdl", False), save_dir)

    start_iter = 0
    if getattr(args, "resume_model", None):
        start_iter = _resume(args.resume_model, model, optimizer, device)

    loader = _make_loader(train_dataset, setup.batch_size, shuffle=True, num_workers=setup.num_workers)
    loss_meter = AverageMeter(setup.log_iters)
    batch_meter = AverageMeter(setup.log_iters)
    reader_meter = AverageMeter(setup.log_iters)

    best_miou = -1.0
    best_iter = -1
    best_summary: Dict[str, Any] = {}
    current_iter = start_iter
    total_timer = Timer()
    batch_timer = Timer()
    iterator = iter(loader)

    model.train()
    while current_iter < setup.iters:
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels = next(iterator)
        reader_meter.update(batch_timer.elapsed())

        current_iter += 1
        lr = scheduler.step(current_iter - 1)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_context(use_amp, device):
            logits = model(images)
            loss, parts = criterion(logits, labels)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_value = float(loss.detach())
        loss_meter.update(loss_value)
        batch_meter.update(batch_timer.elapsed())
        batch_timer.reset()

        if current_iter % setup.log_iters == 0 or current_iter == setup.iters:
            avg_batch = batch_meter.avg or 1e-6
            remaining = (setup.iters - current_iter) * avg_batch
            reserved, allocated = memory_stats_mb(device)
            L.seg_train(
                epoch=(current_iter - 1) // iters_per_epoch + 1,
                iters_done=current_iter,
                iters_total=setup.iters,
                loss=loss_meter.avg,
                lr=lr,
                batch_cost=avg_batch,
                reader_cost=reader_meter.avg,
                ips=setup.batch_size / avg_batch,
                mem_reserved_mb=reserved,
                mem_allocated_mb=allocated,
                eta_seconds=remaining,
            )
            if writer is not None:
                writer.add_scalar("Train/loss", loss_meter.avg, current_iter)
                writer.add_scalar("Train/lr", lr, current_iter)
                for key, value in parts.items():
                    writer.add_scalar("Train/{}".format(key), value, current_iter)

        should_save = current_iter % setup.save_interval == 0 or current_iter == setup.iters
        if should_save:
            save_checkpoint(
                save_dir, "iter_{}".format(current_iter), model,
                dict(setup.meta(), iter=current_iter, loss=loss_meter.avg), optimizer,
            )
            prune_checkpoints(save_dir, int(getattr(args, "keep_checkpoint_max", 5) or 5))

            if val_dataset is not None:
                summary = evaluate(
                    model, val_dataset, device, setup.num_classes, num_workers=0, print_detail=True
                )
                miou = float(summary["mIoU"])
                if miou > best_miou:
                    best_miou, best_iter, best_summary = miou, current_iter, summary
                    save_checkpoint(
                        save_dir, "best_model", model,
                        dict(setup.meta(), iter=current_iter, mIoU=miou), None,
                    )
                # Always print the closing line: it is what terminates the EVAL
                # block for the log parser and updates the job's best metric.
                L.seg_best(best_miou, best_iter)
                if writer is not None:
                    writer.add_scalar("Eval/mIoU", miou, current_iter)
                    writer.add_scalar("Eval/Acc", float(summary["acc"]), current_iter)

    if writer is not None:
        writer.close()

    L.log(
        "Training finished in {}. Best mIoU: {} at iter {}.".format(
            L.format_eta(total_timer.elapsed()),
            "{:.4f}".format(best_miou) if best_miou >= 0 else "n/a",
            best_iter if best_iter > 0 else "n/a",
        )
    )
    L.log("Final weights: {}".format(os.path.join(save_dir, "iter_{}".format(setup.iters), "model.pt")))
    return best_summary


def _resume(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    payload = load_checkpoint(path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    start_iter = int(payload.get("iter", 0) or 0)
    opt_path = os.path.join(os.path.dirname(path), "optimizer.pt")
    if os.path.isfile(opt_path):
        try:
            optimizer.load_state_dict(load_checkpoint(opt_path, map_location=device)["state_dict"])
        except (RuntimeError, KeyError, ValueError) as exc:
            L.log("WARNING: could not restore optimizer state ({}); continuing with a fresh optimizer.".format(exc))
    L.log("Resumed from {} at iter {}.".format(path, start_iter))
    return start_iter


def _maybe_tensorboard(enabled: bool, save_dir: str):
    """TensorBoard writer when `--use_vdl` is passed and the package exists.

    `use_vdl` is Paddle's VisualDL flag; the platform sends it for every
    framework, so we accept it and quietly no-op when TensorBoard is not
    installed rather than failing the run over a logging nicety.
    """
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        L.log("WARNING: --use_vdl requested but TensorBoard is not installed (pip install tensorboard); skipping.")
        return None
    log_dir = os.path.join(save_dir, "vdl")
    os.makedirs(log_dir, exist_ok=True)
    L.log("TensorBoard logs: {}".format(log_dir))
    return SummaryWriter(log_dir)


def load_model_for_inference(
    cfg: Dict[str, Any], args: Any, weights_path: str
) -> Tuple[modelmod.SegModel, SegRunSetup, torch.device]:
    """Rebuild a model and load weights, for `val.py` / `predict.py` / `export.py`.

    The checkpoint's embedded `model_cfg` wins over the YAML when present, so
    evaluating a checkpoint cannot silently use a different architecture than the
    one it was trained with (e.g. because the config was edited afterwards).
    """
    setup = SegRunSetup(cfg, args)
    device = resolve_device(cfgmod.get_bool(cfg, "use_gpu", default=True) and not getattr(args, "cpu", False))
    payload = load_checkpoint(weights_path, map_location=device)

    embedded = payload.get("model_cfg")
    if isinstance(embedded, dict) and embedded:
        if embedded != setup.model_cfg:
            L.log("Using the architecture recorded in the checkpoint ({}).".format(embedded.get("type")))
        setup.model_cfg = embedded
        setup.num_classes = int(embedded.get("num_classes") or setup.num_classes)
    if payload.get("num_logits"):
        setup.num_logits = int(payload["num_logits"])

    model = setup.build_model().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    L.log("Loaded weights from {}".format(weights_path))
    return model, setup, device
