"""
AutoTrain PyTorch training framework.

This package is the PyTorch counterpart to the PaddleSeg / PaddleDetection
repositories the platform drives. It deliberately mirrors their *shape* so the
web app needs as little framework-specific code as possible:

  * The repository root looks like a Paddle repo:
        torchtrain/            <- this importable package (cf. `paddleseg`, `ppdet`)
        tools/train.py         <- entrypoints with Paddle-compatible CLI flags
        tools/val.py
        tools/predict.py
        tools/export.py

  * The YAML config schema is the **same schema the platform already generates**
    for PaddleSeg (semantic segmentation) and PaddleDetection (detection).
    See `torchtrain.config` for the exact keys that are honoured.

  * stdout is byte-compatible with the Paddle log formats, so the existing
    `src/lib/log-parsers` implementation parses torch runs without changes.

Supported frameworks (as named by `project.framework` in the web app):
    TorchSeg  -> semantic segmentation  (PaddleSeg-shaped config)
    TorchDet  -> object detection       (PaddleDetection-shaped config)
"""

__version__ = "1.0.0"

# Keep the import surface tiny: `tools/*.py` import submodules explicitly so a
# missing optional dependency in one task never breaks the other.
__all__ = ["__version__"]
