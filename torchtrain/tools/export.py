#!/usr/bin/env python
"""
Model export entrypoint.

    python tools/export.py --config <merged.yml> --model_path <w>.pt \
        --save_dir <out> --format torchscript

Writes a deployment bundle mirroring PaddleDetection's `export_model` layout, so
the platform's "exported model" browser (which lists sub-folders of
`export_model/`) works unchanged:

    <save_dir>/
      model.pt / model.onnx     # the serialized graph
      infer_cfg.yml             # preprocessing + class names needed at inference

TorchScript is the default because it has no extra dependency; ONNX requires the
`onnx` package and is offered for TensorRT pipelines.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import yaml  # noqa: E402

from torchtrain import cli  # noqa: E402
from torchtrain import config as cfgmod  # noqa: E402
from torchtrain import logger as L  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained PyTorch model for deployment.")
    cli.add_common_args(parser)
    cli.add_weights_args(parser)
    parser.add_argument("--save_dir", "--output_dir", dest="save_dir", default="export_model",
                        help="Directory to write the exported bundle into.")
    parser.add_argument("--format", dest="format", default="torchscript",
                        choices=["torchscript", "onnx"], help="Serialization format.")
    parser.add_argument("--input_shape", dest="input_shape", default=None,
                        help="Tracing shape as H,W (default: the config's eval size).")
    parser.add_argument("--opset", dest="opset", type=int, default=16, help="ONNX opset version.")
    return parser.parse_args()


class _SegExportWrapper(torch.nn.Module):
    """Return a single tensor instead of a list, so the traced graph is simple."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[0]


def main() -> None:
    args = parse_args()
    cfg = cli.apply_opts(cfgmod.load_config(args.config), args.opt)
    task = args.task or cfgmod.detect_task(cfg)
    weights = cli.resolve_weights(args, cfg)
    save_dir = args.save_dir or "export_model"
    os.makedirs(save_dir, exist_ok=True)

    if task != cfgmod.SEG:
        # Detection models take a list of variable-size tensors and return a list
        # of dicts, which neither TorchScript tracing nor ONNX export handles
        # cleanly. Shipping the checkpoint plus its inference config is honest
        # and still usable; scripting torchvision detectors is a separate project.
        from torchtrain.det import exporter as det_exporter

        det_exporter.export(cfg, args, weights, save_dir)
        return

    from torchtrain.seg import trainer

    model, setup, device = trainer.load_model_for_inference(cfg, args, weights)
    model = model.to("cpu").eval()

    if args.input_shape:
        parts = [int(p) for p in str(args.input_shape).replace("x", ",").split(",") if p.strip()]
        height, width = (parts + parts)[:2]
    else:
        width, height = setup.default_size or [512, 512]
    example = torch.randn(1, 3, int(height), int(width))

    wrapper = _SegExportWrapper(model)
    if args.format == "onnx":
        target = os.path.join(save_dir, "model.onnx")
        torch.onnx.export(
            wrapper, example, target,
            input_names=["image"], output_names=["logits"],
            dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                          "logits": {0: "batch", 2: "height", 3: "width"}},
            opset_version=int(args.opset),
        )
    else:
        target = os.path.join(save_dir, "model.pt")
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, example, strict=False)
        traced.save(target)

    val_block = cfg.get("val_dataset") or cfg.get("train_dataset") or {}
    infer_cfg = {
        "framework": "TorchSeg",
        "task": "semantic_segmentation",
        "format": args.format,
        "model_file": os.path.basename(target),
        "architecture": setup.model_cfg.get("type", "UNet"),
        "num_classes": setup.num_classes,
        "class_names": setup.class_names(),
        "input_shape": [3, int(height), int(width)],
        "preprocess": [op for op in (val_block.get("transforms") or []) if _is_eval_safe(op)],
        "align_corners": setup.align_corners,
    }
    with open(os.path.join(save_dir, "infer_cfg.yml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(infer_cfg, handle, sort_keys=False, allow_unicode=True)

    L.log("Exported {} model to {}".format(args.format, os.path.abspath(target)))
    L.log("Inference config: {}".format(os.path.join(os.path.abspath(save_dir), "infer_cfg.yml")))
    L.log("export summary: {}".format(json.dumps({"dir": os.path.abspath(save_dir), "format": args.format})))
    del device


def _is_eval_safe(op) -> bool:
    name = op if isinstance(op, str) else str((op or {}).get("type") or "")
    return not name.startswith("Random")


if __name__ == "__main__":
    cli.run(main)
