"""Explicit CLI training/validation/inference for prepared OBB and Pose datasets.

This entry point is separate from AutoTrain Training Jobs. Only trusted local
weights are accepted, so offline runs never implicitly fetch a checkpoint.
"""
import argparse
import json
import math
from pathlib import Path


def json_write(filename, value):
    Path(filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def long_edge_pose(corners):
    """Stable long-edge orientation modulo 180 degrees; never a head/tail heading."""
    a, b, c, _ = corners
    e1, e2 = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
    l1, l2 = math.hypot(*e1), math.hypot(*e2)
    edge = e1 if l1 >= l2 else e2
    return {
        "center_px": [sum(p[0] for p in corners) / 4, sum(p[1] for p in corners) / 4],
        "long_side_px": max(l1, l2), "short_side_px": min(l1, l2),
        "angle_deg_mod180": math.degrees(math.atan2(edge[1], edge[0])) % 180,
    }


def serialize_result(result, task, keypoint_names):
    records = []
    if task == "obb" and result.obb is not None:
        for box in result.obb:
            polygon = box.xyxyxyxy[0].cpu().tolist()
            records.append({
                "class_id": int(box.cls[0]), "label": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]), "corners_px": polygon, **long_edge_pose(polygon),
            })
    elif task == "pose" and result.boxes is not None and result.keypoints is not None:
        raw = result.keypoints.data.cpu().tolist()
        if raw and len(raw[0]) != len(keypoint_names):
            raise ValueError("Checkpoint keypoint count does not match schema.json")
        for index, box in enumerate(result.boxes):
            records.append({
                "class_id": int(box.cls[0]), "label": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]), "bbox_xyxy_px": box.xyxy[0].cpu().tolist(),
                "keypoints": {
                    name: {"x": point[0], "y": point[1], "confidence": point[2] if len(point) > 2 else None}
                    for name, point in zip(keypoint_names, raw[index])
                },
            })
    return {"source": str(result.path), "height": result.orig_shape[0], "width": result.orig_shape[1], "instances": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "val", "predict", "export"])
    parser.add_argument("--task", choices=["obb", "pose"], required=True)
    parser.add_argument("--model", required=True, help="Trusted local .pt checkpoint (pretrained for train)")
    parser.add_argument("--data", help="Prepared data.yaml (train/val)")
    parser.add_argument("--schema", help="Prepared schema.json; required for pose predict")
    parser.add_argument("--source", help="Image or image directory (predict)")
    parser.add_argument("--output", required=True, help="New run directory")
    parser.add_argument("--device", default="0", help="CUDA device index or cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0, help="0 is a safe Windows starting point")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--format", choices=["onnx", "torchscript"], default="onnx")
    args = parser.parse_args()
    model_path, output = Path(args.model).resolve(), Path(args.output).resolve()
    if not model_path.is_file() or model_path.suffix.lower() != ".pt":
        parser.error("--model must be an existing local .pt file")
    if output.exists():
        parser.error("--output already exists; select a new run directory")
    if not output.parent.is_dir():
        parser.error("--output parent must exist")
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1 or args.workers < 0 or not 0 <= args.conf <= 1:
        parser.error("Invalid epochs/imgsz/batch/workers/conf")
    schema = None
    data = None
    if args.action in ("train", "val"):
        if not args.data or not Path(args.data).is_file():
            parser.error("train/val requires prepared --data data.yaml")
        data = json.loads(Path(args.data).read_text(encoding="utf-8-sig"))
        schema = json.loads(Path(args.data).with_name("schema.json").read_text(encoding="utf-8"))
        if args.action == "val" and args.split not in data:
            parser.error("Dataset has no {} split".format(args.split))
    elif args.schema:
        schema = json.loads(Path(args.schema).read_text(encoding="utf-8-sig"))
    if schema and schema["task"] != args.task:
        parser.error("Dataset schema task differs from --task")
    if args.action == "predict":
        if not args.source or not Path(args.source).exists():
            parser.error("predict requires an existing local --source")
        if args.task == "pose" and not schema:
            parser.error("pose predict requires --schema from the trained dataset")
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError:
        parser.error("Install tools/localization/requirements.txt in a separate Python environment")
    model = YOLO(str(model_path))
    if model.task != args.task:
        parser.error("Checkpoint is a {} model, not {}".format(model.task, args.task))
    if args.action != "train" and schema:
        if list(model.names.values()) != schema["labels"]:
            parser.error("Checkpoint classes/order differ from schema.json")
    common = dict(device=args.device, imgsz=args.imgsz, project=str(output.parent), name=output.name, exist_ok=False)
    if args.action == "train":
        model.train(
            data=str(Path(args.data).resolve()), epochs=args.epochs, batch=args.batch,
            workers=args.workers, seed=args.seed, deterministic=True, patience=30, amp=False, plots=False,
            # No unknown mirror mapping; geometry augmentation moves keypoints with pixels.
            fliplr=0.0, flipud=0.0, degrees=180.0, translate=0.05, scale=0.2,
            mosaic=0.0, mixup=0.0, perspective=0.0, **common
        )
    elif args.action == "val":
        model.val(data=str(Path(args.data).resolve()), split=args.split, batch=args.batch, workers=args.workers, plots=False, **common)
    elif args.action == "predict":
        predictions = []
        for result in model.predict(source=str(Path(args.source).resolve()), conf=args.conf, save=True, stream=True, **common):
            predictions.append(serialize_result(result, args.task, schema["keypointNames"] if schema else []))
        output.mkdir(parents=True, exist_ok=True)
        json_write(output / "predictions.json", {
            "task": args.task, "coordinateSystem": "original-image pixels; x right, y down",
            "images": predictions,
        })
    else:
        # Ultralytics exports next to the source model; stage a copy to preserve the original.
        import shutil
        output.mkdir()
        staged = output / model_path.name
        shutil.copy2(str(model_path), str(staged))
        YOLO(str(staged)).export(format=args.format, imgsz=args.imgsz, device=args.device)
    output.mkdir(parents=True, exist_ok=True)
    if schema:
        json_write(output / "schema.json", schema)
    json_write(output / "run.json", {
        "arguments": vars(args), "ultralytics": ultralytics.__version__,
        "keypointNote": "confidence is not the annotation visibility code; apply production thresholds separately",
    })
    print("Results:", output)


if __name__ == "__main__":
    main()
