#!/usr/bin/env python
"""
Training entrypoint.

    python tools/train.py --config <merged.yml> --save_dir <out> --do_eval
    python tools/train.py -c <merged.yml> --amp --use_vdl

The task (segmentation vs detection) is inferred from the config keys, so one
entrypoint serves both `TorchSeg` and `TorchDet` jobs — exactly like PaddleSeg's
and PaddleDetection's `tools/train.py` do for their own frameworks.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchtrain import cli  # noqa: E402
from torchtrain import config as cfgmod  # noqa: E402
from torchtrain import logger as L  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch model from a merged AutoTrain config.")
    cli.add_common_args(parser)
    cli.add_train_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = cli.apply_opts(cfgmod.load_config(args.config), args.opt)
    task = args.task or cfgmod.detect_task(cfg)

    L.log("torchtrain train | task={} | config={}".format(task, os.path.abspath(args.config)))

    if task == cfgmod.SEG:
        from torchtrain.seg import trainer

        trainer.train(cfg, args)
    else:
        from torchtrain.det import trainer as det_trainer

        det_trainer.train(cfg, args)


if __name__ == "__main__":
    cli.run(main)
