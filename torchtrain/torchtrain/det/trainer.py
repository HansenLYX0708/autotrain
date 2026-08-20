"""
Detection training / evaluation loop.

Semantics follow PaddleDetection:

* Training length is measured in **epochs** (`epoch:`), and the per-step log line
  is `Epoch: [E] [step/steps_per_epoch]`, which is what
  `log-parsers/detection.ts` expects.
* `LearningRate.schedulers` is expressed in epochs; it is converted to steps here
  so warmup and decay land where the config says they should.
* Evaluation runs every `snapshot_epoch` epochs and prints the pycocotools-format
  COCO table.
* Checkpoints land in `<save_dir>/epoch_<N>/model.pt`, with the best mAP mirrored
  to `<save_dir>/best_model/model.pt` and the last epoch to
  `<save_dir>/model_final/model.pt`.
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
    build_optimizer,
    load_checkpoint,
    memory_stats_mb,
    prune_checkpoints,
    resolve_device,
    save_checkpoint,
    set_seed,
    LrScheduler,
)
from . import dataset as dsmod
from . import metrics as metricmod
from . import models as modelmod
from . import transforms as T


class DetRunSetup:
    """Everything derived from a detection config, in one place."""

    def __init__(self, cfg: Dict[str, Any], args: Any) -> None:
        self.cfg = cfg
        self.train_reader = cfg.get("TrainReader") or {}
        self.eval_reader = cfg.get("EvalReader") or {}

        self.epochs = int(getattr(args, "epochs", None) or cfgmod.get_int(cfg, "epoch", "epochs", default=12) or 12)
        self.batch_size = int(
            getattr(args, "batch_size", None) or cfgmod.get_int(self.train_reader, "batch_size", default=2) or 2
        )
        self.eval_batch_size = int(cfgmod.get_int(self.eval_reader, "batch_size", default=1) or 1)

        workers = getattr(args, "num_workers", None)
        if workers is None:
            workers = cfgmod.get_int(cfg, "worker_num", "num_workers", default=0) or 0
        self.num_workers = max(0, int(workers))

        self.log_iter = int(
            getattr(args, "log_iters", None) or cfgmod.get_int(cfg, "log_iter", "log_iters", default=20) or 20
        )
        self.snapshot_epoch = int(
            getattr(args, "save_interval", None) or cfgmod.get_int(cfg, "snapshot_epoch", default=1) or 1
        )
        self.use_gpu = cfgmod.get_bool(cfg, "use_gpu", default=True)
        self.input_size = T.infer_input_size(
            self.eval_reader.get("sample_transforms"),
            self.train_reader.get("batch_transforms"),
            self.train_reader.get("sample_transforms"),
        )
        # `num_classes` excludes background (PaddleDetection convention). The
        # dataset's own category count wins when the two disagree, because the
        # annotations are the ground truth and a stale config is common.
        self.num_classes = int(cfgmod.get_int(cfg, "num_classes", default=0) or 0)

    def resolve_num_classes(self, data: dsmod.CocoData) -> int:
        if self.num_classes and self.num_classes != data.num_classes:
            L.log(
                "WARNING: config says num_classes={} but the annotation file defines {} categories. "
                "Using {} (the dataset).".format(self.num_classes, data.num_classes, data.num_classes)
            )
        self.num_classes = data.num_classes
        return self.num_classes

    def meta(self, model_meta: Dict[str, Any]) -> Dict[str, Any]:
        return dict(model_meta, det_cfg={"architecture": self.cfg.get("architecture")})


def _lr_config(cfg: Dict[str, Any], steps_per_epoch: int, total_epochs: int) -> Tuple[float, Dict[str, Any]]:
    """Translate `LearningRate.schedulers` (epochs) into a step-based config.

    PaddleDetection encodes the schedule as an ordered list whose entries are
    identified by YAML tag: the decay policy first, then an optional
    `!LinearWarmup`. `torchtrain.config` keeps the tag as `__tag__`.
    """
    block = cfg.get("LearningRate") or {}
    base_lr = float(cfgmod.get_num(block, "base_lr", "learning_rate", default=0.01) or 0.01)

    out: Dict[str, Any] = {"type": "CosineDecay"}
    for entry in cfgmod.as_list(block.get("schedulers")):
        if not isinstance(entry, dict):
            continue
        name = cfgmod.tag_name(entry)
        if name == "LinearWarmup" or "start_factor" in entry:
            warmup_epochs = float(cfgmod.get_num(entry, "epochs", "steps", default=0) or 0)
            # `steps` is already in iterations; `epochs` needs converting.
            out["warmup_steps"] = int(
                entry["steps"] if "steps" in entry else round(warmup_epochs * steps_per_epoch)
            )
            if "start_factor" in entry:
                out["start_factor"] = float(entry["start_factor"])
            continue
        if name:
            out["type"] = name
        if "max_epochs" in entry:
            out["max_epochs"] = float(entry["max_epochs"])
        if "gamma" in entry:
            out["gamma"] = float(entry["gamma"])
        if "milestones" in entry:
            out["milestones"] = [int(round(float(m) * steps_per_epoch)) for m in cfgmod.as_list(entry["milestones"])]

    del total_epochs
    return base_lr, out


def _make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        collate_fn=dsmod.collate_detection,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: dsmod.CocoDetectionDataset,
    data: dsmod.CocoData,
    device: torch.device,
    num_workers: int = 0,
    score_threshold: float = 0.0,
    print_detail: bool = True,
) -> Dict[str, Any]:
    """Run COCO evaluation and print the pycocotools-format summary table."""
    model.eval()
    loader = _make_loader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    detections: List[Dict[str, Any]] = []

    if print_detail:
        L.log("Eval iter: 0")
    for index, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)
        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].detach().to("cpu").numpy()
            scores = output["scores"].detach().to("cpu").numpy()
            labels = output["labels"].detach().to("cpu").numpy()
            for box, score, label in zip(boxes, scores, labels):
                if score < score_threshold:
                    continue
                clsid = int(label) - 1  # torchvision label -> Paddle clsid
                if clsid < 0 or clsid >= data.num_classes:
                    continue
                detections.append(
                    {"image_id": image_id, "clsid": clsid, "score": float(score), "bbox_xyxy": [float(v) for v in box]}
                )
        if print_detail and index % 100 == 0:
            L.log("Eval iter: {}".format(index))

    if print_detail:
        L.log("Total sample number: {}, average FPS: 0.0".format(len(dataset)))

    result = metricmod.evaluate_detections(data, detections, records=dataset.records)
    if print_detail:
        L.det_coco_stats(result["stats"], anno_type="bbox")
        L.log("mAP(0.50:0.95) = {:.4f} | mAP(0.50) = {:.4f} | evaluator: {}".format(
            result["stats"][0], result["stats"][1], result["engine"]
        ))
        for name, ap in zip(result["class_names"], result["per_class_ap50"]):
            L.log("  class AP50  {:<24} {:.4f}".format(name, ap))
    model.train()
    return result


def train(cfg: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """Train a detection model. Returns the best evaluation result."""
    set_seed(int(getattr(args, "seed", 1234) or 1234))
    setup = DetRunSetup(cfg, args)
    device = resolve_device(setup.use_gpu and not getattr(args, "cpu", False))

    save_dir = getattr(args, "save_dir", None) or cfg.get("save_dir") or "output"
    os.makedirs(save_dir, exist_ok=True)

    train_dataset, train_data = dsmod.build_datasets(cfg, "train")
    num_classes = setup.resolve_num_classes(train_data)
    val_dataset: Optional[dsmod.CocoDetectionDataset] = None
    val_data: Optional[dsmod.CocoData] = None
    if getattr(args, "do_eval", False):
        try:
            val_dataset, val_data = dsmod.build_datasets(cfg, "eval")
        except (ValueError, FileNotFoundError) as exc:
            L.log("WARNING: evaluation disabled - {}".format(exc))

    model, model_meta = modelmod.build_model(cfg, num_classes, setup.input_size)
    model = model.to(device)

    steps_per_epoch = max(1, (len(train_dataset) + setup.batch_size - 1) // setup.batch_size)
    total_steps = steps_per_epoch * setup.epochs
    base_lr, lr_cfg = _lr_config(cfg, steps_per_epoch, setup.epochs)

    opt_builder = cfg.get("OptimizerBuilder") or {}
    opt_cfg = dict(opt_builder.get("optimizer") or {})
    if opt_builder.get("regularizer"):
        opt_cfg["regularizer"] = opt_builder["regularizer"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(params, opt_cfg, base_lr)
    scheduler = LrScheduler(
        optimizer,
        base_lr=base_lr,
        total_steps=total_steps,
        policy=str(lr_cfg.get("type", "CosineDecay")),
        gamma=float(lr_cfg.get("gamma", 0.1)),
        milestones=lr_cfg.get("milestones"),
        warmup_steps=int(lr_cfg.get("warmup_steps", 0) or 0),
        warmup_start_lr=float(lr_cfg.get("start_factor", 0.0) or 0.0) * base_lr,
    )
    clip_grad = cfgmod.get_num(opt_builder, "clip_grad_by_norm", default=None)

    use_amp = bool(getattr(args, "amp", False) or cfgmod.get_bool(cfg, "use_amp", default=False))
    scaler = amp_scaler(use_amp, device)

    L.log("---------------- Configuration ----------------")
    L.log("architecture     : {} / {}".format(model_meta["architecture"], model_meta["backbone"]))
    L.log("init             : {}".format(model_meta["init"]))
    L.log("num_classes      : {} ({})".format(num_classes, ", ".join(train_data.class_names)))
    L.log("epochs           : {} ({} steps/epoch, {} total)".format(setup.epochs, steps_per_epoch, total_steps))
    L.log("batch_size       : {}".format(setup.batch_size))
    L.log("num_workers      : {}".format(setup.num_workers))
    L.log("base_lr          : {}".format(base_lr))
    L.log("lr_scheduler     : {} (warmup {} steps)".format(lr_cfg.get("type"), lr_cfg.get("warmup_steps", 0)))
    L.log("input_size       : {}".format(setup.input_size or "torchvision default (800/1333)"))
    L.log("use_amp          : {}".format(use_amp))
    L.log("train samples    : {}".format(len(train_dataset)))
    L.log("val samples      : {}".format(len(val_dataset) if val_dataset else 0))
    L.log("save_dir         : {}".format(os.path.abspath(save_dir)))
    L.log("snapshot_epoch   : {}".format(setup.snapshot_epoch))
    L.log("-----------------------------------------------")

    writer = _maybe_tensorboard(getattr(args, "use_vdl", False), save_dir)

    start_epoch = 0
    if getattr(args, "resume_model", None):
        start_epoch = _resume(args.resume_model, model, optimizer, device)

    loader = _make_loader(train_dataset, setup.batch_size, shuffle=True, num_workers=setup.num_workers)
    loss_meter = AverageMeter(setup.log_iter)
    batch_meter = AverageMeter(setup.log_iter)
    data_meter = AverageMeter(setup.log_iter)

    best_map = -1.0
    best_epoch = -1
    best_result: Dict[str, Any] = {}
    global_step = start_epoch * steps_per_epoch
    total_timer = Timer()
    model.train()

    for epoch in range(start_epoch, setup.epochs):
        batch_timer = Timer()
        for step, (images, targets) in enumerate(loader, start=1):
            data_meter.update(batch_timer.elapsed())
            lr = scheduler.step(global_step)
            global_step += 1

            images = [img.to(device, non_blocking=True) for img in images]
            device_targets = [
                {
                    "boxes": t["boxes"].to(device, non_blocking=True),
                    "labels": t["labels"].to(device, non_blocking=True),
                }
                for t in targets
            ]

            optimizer.zero_grad(set_to_none=True)
            with amp_context(use_amp, device):
                loss_dict = model(images, device_targets)
                loss = sum(loss_dict.values())

            if not torch.isfinite(loss):
                # A non-finite loss poisons every subsequent weight; skipping the
                # step is what torchvision's reference training script does too.
                L.log("WARNING: non-finite loss at step {} ({}); skipping this batch.".format(global_step, float(loss)))
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, float(clip_grad))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(params, float(clip_grad))
                optimizer.step()

            loss_meter.update(float(loss.detach()))
            batch_meter.update(batch_timer.elapsed())
            batch_timer.reset()

            if step % setup.log_iter == 0 or step == steps_per_epoch:
                avg_batch = batch_meter.avg or 1e-6
                remaining = (total_steps - global_step) * avg_batch
                reserved, allocated = memory_stats_mb(device)
                L.det_train(
                    epoch=epoch,
                    step=step,
                    steps_per_epoch=steps_per_epoch,
                    lr=lr,
                    loss=loss_meter.avg,
                    losses=modelmod.normalize_loss_dict(loss_dict),
                    eta_seconds=remaining,
                    batch_cost=avg_batch,
                    data_cost=data_meter.avg,
                    ips=setup.batch_size / avg_batch,
                    mem_reserved_mb=reserved,
                    mem_allocated_mb=allocated,
                )
                if writer is not None:
                    writer.add_scalar("Train/loss", loss_meter.avg, global_step)
                    writer.add_scalar("Train/lr", lr, global_step)

        is_last = epoch == setup.epochs - 1
        if (epoch + 1) % setup.snapshot_epoch == 0 or is_last:
            save_checkpoint(
                save_dir, "epoch_{}".format(epoch), model,
                dict(setup.meta(model_meta), epoch=epoch, loss=loss_meter.avg), optimizer,
            )
            prune_checkpoints(
                save_dir, int(getattr(args, "keep_checkpoint_max", 5) or 5),
                protect=("best_model", "model_final"),
            )
            if val_dataset is not None and val_data is not None:
                result = evaluate(model, val_dataset, val_data, device, num_workers=0, print_detail=True)
                current = float(result["stats"][0])
                if current > best_map:
                    best_map, best_epoch, best_result = current, epoch, result
                    save_checkpoint(
                        save_dir, "best_model", model,
                        dict(setup.meta(model_meta), epoch=epoch, mAP=current), None,
                    )
                L.log("Best mAP(0.50:0.95) = {:.4f} at epoch {}.".format(best_map, best_epoch))
                if writer is not None:
                    writer.add_scalar("Eval/mAP", current, global_step)
                    writer.add_scalar("Eval/mAP50", float(result["stats"][1]), global_step)
        if is_last:
            save_checkpoint(
                save_dir, "model_final", model,
                dict(setup.meta(model_meta), epoch=epoch, loss=loss_meter.avg), None,
            )

    if writer is not None:
        writer.close()

    L.log(
        "Training finished in {}. Best mAP: {} at epoch {}.".format(
            L.format_eta(total_timer.elapsed()),
            "{:.4f}".format(best_map) if best_map >= 0 else "n/a",
            best_epoch if best_epoch >= 0 else "n/a",
        )
    )
    L.log("Final weights: {}".format(os.path.join(save_dir, "model_final", "model.pt")))
    return best_result


def evaluate_from_weights(cfg: Dict[str, Any], args: Any, weights: str) -> Dict[str, Any]:
    """`tools/val.py` entrypoint for detection."""
    model, setup, data, dataset, device = load_model_for_inference(cfg, args, weights, need_dataset=True)
    del setup
    return evaluate(
        model, dataset, data, device,
        num_workers=int(getattr(args, "num_workers", 0) or 0),
        print_detail=True,
    )


def load_model_for_inference(
    cfg: Dict[str, Any], args: Any, weights: str, need_dataset: bool = False
):
    """Rebuild a detection model from a checkpoint, optionally with its dataset.

    The checkpoint's embedded architecture/backbone/num_classes win over the
    YAML, so evaluating a checkpoint can never accidentally use a different
    network than the one that produced it.
    """
    setup = DetRunSetup(cfg, args)
    device = resolve_device(setup.use_gpu and not getattr(args, "cpu", False))
    payload = load_checkpoint(weights, map_location=device)

    dataset = None
    data = None
    if need_dataset:
        dataset, data = dsmod.build_datasets(cfg, "eval")

    meta = {
        "architecture": payload.get("architecture") or cfg.get("architecture") or "FasterRCNN",
        "backbone": payload.get("backbone") or "",
        "num_classes": int(payload.get("num_classes") or (data.num_classes if data else setup.num_classes) or 1),
        "input_size": payload.get("input_size") or (list(setup.input_size) if setup.input_size else None),
    }
    model = modelmod.rebuild_from_meta(meta, cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    L.log("Loaded weights from {} ({} / {}, {} classes)".format(
        weights, meta["architecture"], meta["backbone"] or "default", meta["num_classes"]
    ))

    if need_dataset:
        return model, setup, data, dataset, device
    return model, setup, meta, device


def _resume(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    payload = load_checkpoint(path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    start_epoch = int(payload.get("epoch", -1) or -1) + 1
    opt_path = os.path.join(os.path.dirname(path), "optimizer.pt")
    if os.path.isfile(opt_path):
        try:
            optimizer.load_state_dict(load_checkpoint(opt_path, map_location=device)["state_dict"])
        except (RuntimeError, KeyError, ValueError) as exc:
            L.log("WARNING: could not restore optimizer state ({}).".format(exc))
    L.log("Resumed from {} at epoch {}.".format(path, start_epoch))
    return start_epoch


def _maybe_tensorboard(enabled: bool, save_dir: str):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        L.log("WARNING: --use_vdl requested but TensorBoard is not installed; skipping.")
        return None
    log_dir = os.path.join(save_dir, "vdl")
    os.makedirs(log_dir, exist_ok=True)
    L.log("TensorBoard logs: {}".format(log_dir))
    return SummaryWriter(log_dir)
