"""Ultralytics adapter for the platform's three-config/job/validation contracts."""
import argparse
import json
import math
import os
import shutil
import sys
from uuid import uuid4
from pathlib import Path

import yaml

PREFIX = "AUTOTRAIN_LOCALIZATION "


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def emit(**data):
    data["nonFinite"] = [key for key, value in data.items() if isinstance(value, float) and not math.isfinite(value)]
    data["nonFinite"] += [key for key, value in data.get("extraMetrics", {}).items()
                          if isinstance(value, float) and not math.isfinite(value)]
    print(PREFIX + json.dumps(clean(data), ensure_ascii=False), flush=True)


def load_config(filename):
    with open(filename, encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    block = config.get("localization", {})
    if block.get("task") not in ("obb", "pose"):
        raise ValueError("Merged YAML must declare localization.task: obb or pose")
    return config, block


def dataset_schema(block):
    data_file = Path(block["data"]).resolve()
    schema_file = data_file.with_name("schema.json")
    data = yaml.safe_load(data_file.read_text(encoding="utf-8-sig"))
    schema = json.loads(schema_file.read_text(encoding="utf-8-sig"))
    if schema["task"] != block["task"]:
        raise ValueError("Dataset and model localization tasks disagree")
    names = data["names"]
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(names))]
    if names != schema["labels"]:
        raise ValueError("Dataset class order disagrees with schema.json")
    if block["task"] == "pose" and data.get("kpt_shape") != [len(schema["keypointNames"]), 3]:
        raise ValueError("Dataset keypoint shape disagrees with schema.json")
    return data, schema


def metrics_record(metrics, task):
    metrics = {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float)) or hasattr(v, "item")}
    suffix = "P" if task == "pose" else "B"
    return {
        "mAP": metrics.get("metrics/mAP50-95({})".format(suffix)),
        "mAP50": metrics.get("metrics/mAP50({})".format(suffix)),
        "extraMetrics": metrics,
    }


def long_edge(polygon):
    a, b, c, _ = polygon
    edges = [(b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])]
    lengths = [math.hypot(*edge) for edge in edges]
    edge = edges[0 if lengths[0] >= lengths[1] else 1]
    return {
        "center_px": [sum(p[0] for p in polygon) / 4, sum(p[1] for p in polygon) / 4],
        "long_side_px": max(lengths), "short_side_px": min(lengths),
        "angle_deg_mod180": math.degrees(math.atan2(edge[1], edge[0])) % 180,
    }


def prediction_record(result, task, schema):
    instances = []
    if task == "obb" and result.obb is not None:
        for box in result.obb:
            polygon = box.xyxyxyxy[0].cpu().tolist()
            instances.append({
                "label": result.names[int(box.cls[0])], "class_id": int(box.cls[0]),
                "confidence": float(box.conf[0]), "corners_px": polygon, **long_edge(polygon),
            })
    elif task == "pose" and result.boxes is not None and result.keypoints is not None:
        points = result.keypoints.data.cpu().tolist()
        for i, box in enumerate(result.boxes):
            if len(points[i]) != len(schema["keypointNames"]):
                raise ValueError("Checkpoint keypoint count differs from schema")
            instances.append({
                "label": result.names[int(box.cls[0])], "class_id": int(box.cls[0]),
                "confidence": float(box.conf[0]), "bbox_xyxy_px": box.xyxy[0].cpu().tolist(),
                "keypoints": {name: {"x": p[0], "y": p[1], "confidence": p[2] if len(p) > 2 else None}
                              for name, p in zip(schema["keypointNames"], points[i])},
            })
    return {"source": str(result.path), "width": result.orig_shape[1], "height": result.orig_shape[0], "instances": instances}


def geometry_evaluation(model, data, split, task, schema, common):
    """Pixel errors on class-matched detections; misses remain explicit in the report."""
    import cv2
    import numpy as np

    root = Path(data["path"])
    source = root / data[split]
    distances, angles, point_errors = [], [], {name: [] for name in schema["keypointNames"]}
    counts = {"targets": 0, "predictions": 0, "matched": 0}
    point_targets = {name: 0 for name in schema["keypointNames"]}
    image_records = []
    for result in model.predict(source=str(source), stream=True, save=False, verbose=False, conf=0.25, **common):
        record = prediction_record(result, task, schema)
        image_records.append(record)
        width, height = record["width"], record["height"]
        label_path = root / "labels" / split / (Path(result.path).stem + ".txt")
        targets = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            values = list(map(float, line.split()))
            if task == "obb":
                polygon = [[values[i] * width, values[i + 1] * height] for i in range(1, 9, 2)]
                targets.append({"class_id": int(values[0]), "corners_px": polygon, **long_edge(polygon)})
            else:
                _, cx, cy, bw, bh = values[:5]
                targets.append({
                    "class_id": int(values[0]),
                    "bbox_xyxy_px": [(cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height],
                    "keypoints": {name: {"x": values[5+3*i]*width, "y": values[6+3*i]*height, "visibility": int(values[7+3*i])}
                                  for i, name in enumerate(schema["keypointNames"])},
                })
        predictions = record["instances"]
        counts["targets"] += len(targets)
        if task == "pose":
            for target in targets:
                for name, point in target["keypoints"].items():
                    if point["visibility"] > 0:
                        point_targets[name] += 1
        counts["predictions"] += len(predictions)

        def polygon(instance):
            if task == "obb":
                return np.array(instance["corners_px"], dtype=np.float32)
            x1, y1, x2, y2 = instance["bbox_xyxy_px"]
            return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        pairs = []
        for i, target in enumerate(targets):
            for j, predicted in enumerate(predictions):
                if target["class_id"] != predicted["class_id"]:
                    continue
                a, b = polygon(target), polygon(predicted)
                intersection = cv2.intersectConvexConvex(a, b)[0]
                union = abs(cv2.contourArea(a)) + abs(cv2.contourArea(b)) - intersection
                iou = intersection / max(union, 1e-9)
                if iou >= 0.5:
                    pairs.append((iou, i, j))
        used_targets, used_predictions = set(), set()
        for _, i, j in sorted(pairs, reverse=True):
            if i in used_targets or j in used_predictions:
                continue
            used_targets.add(i)
            used_predictions.add(j)
            counts["matched"] += 1
            target, predicted = targets[i], predictions[j]
            if task == "obb":
                distances.append(math.hypot(*(a-b for a, b in zip(target["center_px"], predicted["center_px"]))))
                if target["long_side_px"] / target["short_side_px"] >= 1.1:
                    difference = abs(target["angle_deg_mod180"] - predicted["angle_deg_mod180"]) % 180
                    angles.append(min(difference, 180-difference))
            else:
                for name, point in target["keypoints"].items():
                    if point["visibility"] > 0:
                        guess = predicted["keypoints"][name]
                        if guess["confidence"] is not None and guess["confidence"] >= 0.5:
                            point_errors[name].append(math.hypot(point["x"]-guess["x"], point["y"]-guess["y"]))

    def summary(values):
        return {"count": len(values), "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
                "p95": float(np.percentile(values, 95)) if values else None}

    report = {
        "confidenceThreshold": 0.25, "matchingIoU": 0.5, "images": len(image_records),
        **counts, "missedTargets": counts["targets"]-counts["matched"],
        "falsePositives": counts["predictions"]-counts["matched"],
    }
    if task == "obb":
        report.update(centerErrorPx=summary(distances), angleErrorDegMod180=summary(angles))
    else:
        report.update(keypointConfidenceThreshold=0.5, keypointErrorPx={
            name: {**summary(values), "targets": point_targets[name], "missed": point_targets[name]-len(values)}
            for name, values in point_errors.items()
        })
    return report, image_records


def run(args):
    import torch
    import ultralytics
    from ultralytics import YOLO

    config, block = load_config(args.config)
    task = block["task"]
    train = dict(block.get("train", {}))
    data, schema = dataset_schema(block)
    use_gpu = config.get("use_gpu", True)
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("This job requests GPU training, but this Python environment cannot use CUDA")
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0") if use_gpu else "cpu"
    if use_gpu and "," in device:
        raise ValueError("TorchOBB/TorchPose currently require exactly one selected GPU")
    if args.save_dir:
        output = Path(args.save_dir).resolve()
    elif args.action == "train":
        output = Path(config.get("save_dir", "output/localization")).resolve()
    else:
        base = Path(config.get("save_dir", Path(args.config).with_suffix(""))).resolve()
        output = base / "validation" / ("{}-{}".format(args.action, uuid4().hex[:12]))
    output.mkdir(parents=True, exist_ok=True)
    common = dict(device=device, imgsz=int(train.get("imgsz", 1024)), project=str(output.parent), name=output.name, exist_ok=True)
    architecture = block.get("model", {}).get("architecture")
    weights = args.model_path or block.get("model", {}).get("weights")
    if weights and not Path(weights).is_file():
        raise ValueError("Weights file does not exist: {}. Configure a local pretrained checkpoint in the model YAML.".format(weights))
    if args.action != "train" and not weights:
        raise ValueError("Evaluation/inference/export requires a trained checkpoint")
    model = YOLO(str(weights) if weights else "{}.yaml".format(architecture))
    if model.task != task:
        raise ValueError("Checkpoint task {} differs from configured {}".format(model.task, task))
    if args.action != "train" and list(model.names.values()) != schema["labels"]:
        raise ValueError("Checkpoint classes/order differ from the dataset")
    if args.action != "train" and task == "pose" and model.model.yaml.get("kpt_shape") != [len(schema["keypointNames"]), 3]:
        raise ValueError("Checkpoint keypoint shape differs from the prepared dataset schema")
    write_json(output / "schema.json", schema)
    write_json(output / "runtime.json", {
        "task": task, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda, "device": device, "gpu": torch.cuda.get_device_name(0) if use_gpu else None,
        "config": config, "action": args.action,
    })
    print("AutoTrain {} {} on {} ({})".format(task, args.action, device, torch.cuda.get_device_name(0) if use_gpu else "CPU"), flush=True)

    if args.action == "train":
        interval = max(1, int(train.pop("log_interval", 20)))
        state = {"batch": 0, "epoch": None, "best": -1.0, "fit_epoch": None}

        def on_batch(trainer):
            if state["epoch"] != trainer.epoch:
                state["epoch"], state["batch"] = trainer.epoch, 0
            state["batch"] += 1
            if state["batch"] % interval and state["batch"] != len(trainer.train_loader):
                return
            losses = {k: float(v) for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items()}
            emit(kind="train", epoch=trainer.epoch + 1, iteration=state["batch"], totalIter=len(trainer.train_loader),
                 loss=sum(losses.values()), learningRate=trainer.optimizer.param_groups[0]["lr"],
                 lossL1=losses.get("train/box_loss"), lossCls=losses.get("train/cls_loss"),
                 lossDfl=losses.get("train/dfl_loss"), extraMetrics=losses,
                 memAllocated=round(torch.cuda.memory_allocated() / 1024 ** 2) if use_gpu else 0,
                 memReserved=round(torch.cuda.memory_reserved() / 1024 ** 2) if use_gpu else 0)

        def on_epoch(trainer):
            # Ultralytics invokes this again after validating its best checkpoint.
            # That second result must never be paired with the last epoch's weights.
            if state["fit_epoch"] == trainer.epoch:
                return
            state["fit_epoch"] = trainer.epoch
            metrics = metrics_record(trainer.metrics or {}, task)
            score = metrics["mAP"]
            best = score is not None and math.isfinite(score) and score >= state["best"]
            if best and Path(trainer.last).is_file():
                destination = output / "best_model" / "model.pt"
                destination.parent.mkdir(exist_ok=True)
                shutil.copy2(str(trainer.last), str(destination))
                state["best"] = score
            period = int(train.get("save_period", -1))
            if period > 0 and (trainer.epoch + 1) % period == 0 and Path(trainer.last).is_file():
                checkpoint = output / ("epoch_{}".format(trainer.epoch + 1)) / "model.pt"
                checkpoint.parent.mkdir(exist_ok=True)
                shutil.copy2(str(trainer.last), str(checkpoint))
            emit(kind="eval", epoch=trainer.epoch + 1, iteration=0, totalIter=0, **metrics,
                 bestMetric=score if best else None, bestIter=trainer.epoch + 1 if best else None,
                 bestMetricName="pose_mAP50_95" if task == "pose" else "obb_mAP50_95")
            write_json(output / "metrics.json", {"taskKind": task, "epoch": trainer.epoch + 1, **metrics})

        model.add_callback("on_train_batch_end", on_batch)
        model.add_callback("on_fit_epoch_end", on_epoch)
        if not args.do_eval:
            raise ValueError("Localization jobs require validation to select a checkpoint")
        train.update(common)
        train.update(data=block["data"], val=True, plots=False, save=True, verbose=False)
        if args.amp:
            train["amp"] = True
        model.train(**train)
        if Path(model.trainer.last).is_file():
            dest = output / "last_model" / "model.pt"
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(str(model.trainer.last), str(dest))
        if not (output / "best_model/model.pt").is_file():
            raise RuntimeError("Training ended without a validated best checkpoint")
    elif args.action == "eval":
        split = args.split or block.get("evaluation", {}).get("split", "val")
        if split not in ("val", "test") or not data.get(split):
            raise ValueError("Selected validation/test split is absent from the dataset")
        result = model.val(data=block["data"], split=split, batch=int(train.get("batch", 4)),
                           workers=int(train.get("workers", 0)), plots=False, verbose=False, **common)
        metrics = metrics_record(result.results_dict, task)
        geometry, predictions = geometry_evaluation(model, data, split, task, schema, common)
        record = {"taskKind": task, "split": split, "outputDir": str(output), "samplesCount": geometry["images"], "geometry": geometry, **metrics}
        write_json(output / "eval-predictions.json", predictions)
        write_json(output / "metrics.json", record)
        print("AUTOTRAIN_RESULT " + json.dumps(clean(record)), flush=True)
    elif args.action == "infer":
        if not args.image_path or not Path(args.image_path).exists():
            raise ValueError("Inference input does not exist")
        predictions = [
            prediction_record(result, task, schema)
            for result in model.predict(source=args.image_path, stream=True, save=True, verbose=False,
                                        conf=float(block.get("inference", {}).get("confidence", 0.25)), **common)
        ]
        write_json(output / "predictions.json", {"taskKind": task, "coordinateSystem": "original-image pixels", "images": predictions})
    else:
        if args.format not in ("onnx", "torchscript"):
            raise ValueError("Export supports onnx or torchscript")
        package = output / ("{}_{}".format(task, args.format))
        package.mkdir(exist_ok=True)
        staged = package / "source.pt"
        shutil.copy2(str(weights), str(staged))
        try:
            exported = YOLO(str(staged)).export(format=args.format, imgsz=common["imgsz"], device=device, simplify=False)
            destination = package / ("model" + Path(exported).suffix)
            Path(exported).replace(destination)
        finally:
            staged.unlink(missing_ok=True)
        write_json(package / "schema.json", schema)
        write_json(package / "export.json", {
            "format": args.format, "model": destination.name, "task": task,
            "schema": schema, "inputSize": common["imgsz"],
            "postprocessing": "Ultralytics raw head output; implement task-specific decoding, NMS and letterbox reversal",
        })


def main(action=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "eval", "infer", "export", "prepare"])
    parser.add_argument("--config")
    parser.add_argument("--model_path")
    parser.add_argument("--save_dir")
    parser.add_argument("--image_path")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use_vdl", action="store_true")
    parser.add_argument("--format", default="torchscript")
    parser.add_argument("--split", choices=["val", "test"])
    parser.add_argument("--project")
    parser.add_argument("--images")
    parser.add_argument("--task", choices=["obb", "pose"])
    args = parser.parse_args(([action] if action else []) + sys.argv[1:])
    if args.action == "prepare":
        from .localization_data import prepare
        if not all([args.project, args.images, args.save_dir, args.task]):
            parser.error("prepare requires --project, --images, --save_dir, --task")
        print(prepare(args.project, args.images, args.save_dir, args.task), flush=True)
    else:
        if not args.config:
            parser.error("--config is required")
        run(args)


if __name__ == "__main__":
    main()
