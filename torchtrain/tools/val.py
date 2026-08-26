#!/usr/bin/env python
"""
Evaluation entrypoint.

    python tools/val.py --config <merged.yml> --model_path <save_dir>/best_model/model.pt

Segmentation prints the PaddleSeg `[EVAL]` block; detection prints the
pycocotools-format COCO summary table; anomaly detection prints a single
`[EVAL] #Images: N image_auroc: ... threshold: ...` line. All three are what
`src/app/api/validation-jobs/route.ts` greps for when it fills in a validation
job's metrics, so the output format is part of the contract.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchtrain import cli  # noqa: E402
from torchtrain import config as cfgmod  # noqa: E402
from torchtrain import logger as L  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PyTorch model.")
    cli.add_common_args(parser)
    cli.add_weights_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = cli.apply_opts(cfgmod.load_config(args.config), args.opt)
    task = args.task or cfgmod.detect_task(cfg)
    weights = cli.resolve_weights(args, cfg)

    L.log("torchtrain val | task={} | weights={}".format(task, weights))

    if task == cfgmod.AD:
        from torchtrain.ad import runner as ad_runner

        ad_runner.evaluate(cfg, args, weights)
    elif task == cfgmod.SEG:
        from torchtrain.seg import trainer

        model, setup, device = trainer.load_model_for_inference(cfg, args, weights)
        val_dataset = setup.build_val_dataset()
        if val_dataset is None:
            raise ValueError("The config has no `val_dataset:` block, so there is nothing to evaluate.")
        trainer.evaluate(
            model, val_dataset, device, setup.num_classes,
            num_workers=int(args.num_workers or 0), print_detail=True,
        )
    else:
        from torchtrain.det import trainer as det_trainer

        det_trainer.evaluate_from_weights(cfg, args, weights)


if __name__ == "__main__":
    cli.run(main)
