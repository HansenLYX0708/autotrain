"""
Shared CLI plumbing for `tools/*.py`.

The flag names deliberately match the Paddle repositories, because the web app
generates the command string and we want that generator to stay boring:

    tools/train.py   --config X [--save_dir Y] [--do_eval] [--amp] [--use_vdl]
    tools/val.py     --config X --model_path W
    tools/predict.py --config X --model_path W --image_path I --save_dir O
    tools/export.py  --config X --model_path W --save_dir O

PaddleDetection's spellings are accepted as aliases (`-c`, `--infer_img`,
`--infer_dir`, `--output_dir`, `-o key=value`) so a command copied from a
detection workflow, or hand-edited by a user, still runs.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional


def bootstrap_path() -> None:
    """Make `import torchtrain` work when invoked as `python tools/train.py`.

    The repo root (the parent of `tools/`) is prepended to `sys.path`. Without
    this the script only works when the package is pip-installed, and the
    platform runs it straight from a checked-out folder.
    """
    tools_dir = os.path.dirname(os.path.abspath(sys.argv[0] or __file__))
    repo_root = os.path.dirname(tools_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    return str(value).strip().lower() in ("1", "true", "yes", "on", "t", "y")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c", "--config", dest="config", required=True,
        help="Path to the merged job config YAML.",
    )
    parser.add_argument(
        "--task", dest="task", default=None, choices=["seg", "det", "ad"],
        help="Override task detection (normally inferred from the config keys).",
    )
    parser.add_argument(
        "-o", "--opt", dest="opt", nargs="+", default=None, metavar="KEY=VALUE",
        help="Override top-level config keys, e.g. -o weights=/path/model.pt iters=100.",
    )
    parser.add_argument(
        "--num_workers", dest="num_workers", type=int, default=None,
        help="DataLoader workers (default: the config's num_workers).",
    )
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="Force CPU even if a GPU is visible.")


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--save_dir", dest="save_dir", default=None, help="Directory for checkpoints and logs.")
    parser.add_argument("--do_eval", dest="do_eval", action="store_true", help="Evaluate every save_interval.")
    parser.add_argument("--amp", "--use_amp", dest="amp", nargs="?", const=True, type=str2bool, default=False,
                        help="Enable mixed-precision training.")
    parser.add_argument("--use_vdl", dest="use_vdl", nargs="?", const=True, type=str2bool, default=False,
                        help="Write TensorBoard logs (Paddle's VisualDL flag name).")
    parser.add_argument("--vdl_log_dir", dest="vdl_log_dir", default=None, help="Accepted for CLI compatibility.")
    parser.add_argument("--iters", dest="iters", type=int, default=None, help="Override iters (segmentation).")
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=None,
                        help="Override epochs (detection).")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--log_iters", "--log_iter", dest="log_iters", type=int, default=None,
                        help="Log every N steps.")
    parser.add_argument("--save_interval", "--snapshot_epoch", dest="save_interval", type=int, default=None,
                        help="Checkpoint/eval cadence (iters for seg, epochs for det).")
    parser.add_argument("--keep_checkpoint_max", dest="keep_checkpoint_max", type=int, default=5,
                        help="How many step checkpoints to retain (best_model is never pruned).")
    parser.add_argument("--resume_model", "--resume", dest="resume_model", default=None,
                        help="Checkpoint to resume from.")
    parser.add_argument("--seed", dest="seed", type=int, default=1234, help="Random seed.")


def add_weights_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_path", "--weights", dest="model_path", default=None,
                        help="Path to model.pt (or pass -o weights=...).")


def add_predict_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image_path", "--infer_img", "--infer_dir", dest="image_path", default=None,
                        help="Image file or directory to run inference on.")
    parser.add_argument("--save_dir", "--output_dir", dest="save_dir", default="output/predict_results",
                        help="Directory for the prediction outputs.")


def apply_opts(cfg: Dict[str, Any], opts: Optional[List[str]]) -> Dict[str, Any]:
    """Apply `-o key=value` / `-o a.b=value` overrides onto a loaded config."""
    for item in opts or []:
        if "=" not in item:
            continue
        key, _, raw = item.partition("=")
        value: Any = raw
        lowered = raw.strip().lower()
        if lowered in ("true", "false"):
            value = lowered == "true"
        elif lowered in ("null", "none", "~"):
            value = None
        else:
            try:
                value = int(raw) if raw.strip().lstrip("+-").isdigit() else float(raw)
            except ValueError:
                value = raw
        target = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            nxt = target.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                target[part] = nxt
            target = nxt
        target[parts[-1]] = value
    return cfg


def resolve_weights(args: Any, cfg: Dict[str, Any]) -> str:
    """Find the weights path from `--model_path`, `-o weights=`, or the config."""
    for candidate in (getattr(args, "model_path", None), cfg.get("weights"), cfg.get("model_path")):
        if candidate:
            return str(candidate)
    raise ValueError(
        "No weights specified. Pass --model_path /path/to/model.pt (or -o weights=/path/to/model.pt)."
    )


def run(main: Callable[[], Any]) -> None:
    """Invoke `main`, turning exceptions into a clean non-zero exit.

    The runner marks a job failed on a non-zero exit code and shows the last
    stderr lines, so a full traceback plus a one-line summary is exactly what is
    most useful in the UI.
    """
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted by user.\n")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 - top-level guard for a CLI
        traceback.print_exc()
        sys.stderr.write("\nERROR: {}: {}\n".format(type(exc).__name__, exc))
        sys.stderr.flush()
        sys.exit(1)
