#!/usr/bin/env python
"""
Inference entrypoint.

    python tools/predict.py --config <merged.yml> --model_path <w>.pt \
        --image_path <file-or-dir> --save_dir <out>

Unlike PaddleSeg's `predict.py`, TIFF inputs are accepted directly, so the
platform's TIFF-to-PNG staging step is unnecessary for torch jobs.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchtrain import cli  # noqa: E402
from torchtrain import config as cfgmod  # noqa: E402
from torchtrain import logger as L  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained PyTorch model.")
    cli.add_common_args(parser)
    cli.add_weights_args(parser)
    cli.add_predict_args(parser)
    parser.add_argument("--score_threshold", dest="score_threshold", type=float, default=0.5,
                        help="Detection only: minimum score for a box to be drawn/saved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = cli.apply_opts(cfgmod.load_config(args.config), args.opt)
    task = args.task or cfgmod.detect_task(cfg)
    weights = cli.resolve_weights(args, cfg)
    save_dir = args.save_dir or "output/predict_results"

    L.log("torchtrain predict | task={} | input={} | save_dir={}".format(task, args.image_path, save_dir))

    if task == cfgmod.AD:
        from torchtrain.ad import runner as ad_runner

        ad_runner.predict(cfg, args, weights, save_dir)
    elif task == cfgmod.SEG:
        from torchtrain.seg import predictor, trainer

        model, setup, device = trainer.load_model_for_inference(cfg, args, weights)
        images = predictor.collect_images(args.image_path)
        predictor.predict(model, setup, device, images, save_dir)
    else:
        from torchtrain.det import predictor as det_predictor

        det_predictor.predict_from_weights(cfg, args, weights, save_dir)


if __name__ == "__main__":
    cli.run(main)
