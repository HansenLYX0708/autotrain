"""
Config loading for the AutoTrain PyTorch trainer.

The documents fed to this loader are produced by the web app by deep-merging
three YAML files (dataset -> training -> model). They use the *Paddle* schemas
verbatim, which has two consequences:

1. **Custom YAML tags must not be fatal.** PaddleDetection identifies learning
   rate schedulers and dataset classes by tag (`- !CosineDecay`, `!COCODataSet`)
   rather than by a `name:` key. `yaml.safe_load` raises
   `ConstructorError: could not determine a constructor for the tag '!CosineDecay'`
   on those documents. We register a catch-all multi-constructor that turns
   `!Foo {a: 1}` into `{'__tag__': 'Foo', 'a': 1}` so the tag survives as data.

2. **Two schemas share one loader.** `train_dataset:` / `model:` / `iters:` is
   the PaddleSeg shape; `TrainDataset:` / `architecture:` / `epoch:` is the
   PaddleDetection shape. `detect_task()` decides which one a document is, so
   the CLI can keep a single entrypoint like Paddle does.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import yaml

SEG = "seg"
DET = "det"


class _TagTolerantLoader(yaml.SafeLoader):
    """SafeLoader that keeps unknown `!Tag`s as data instead of raising."""


def _construct_tagged(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> Any:
    """Turn `!Foo {...}` / `!Foo [...]` / `!Foo bar` into plain Python data.

    Mappings gain a `__tag__` key so callers can recover the tag; sequences and
    scalars are returned as-is because no Paddle config relies on their tag.
    """
    if isinstance(node, yaml.MappingNode):
        value = loader.construct_mapping(node, deep=True)
        value["__tag__"] = tag_suffix
        return value
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_TagTolerantLoader.add_multi_constructor("!", _construct_tagged)
# `1.0e-4` is a float in YAML 1.2 but a *string* under PyYAML's YAML 1.1
# resolver unless the mantissa has a dot. Paddle configs contain both spellings
# (`base_lr: 1e-4`), so widen the float resolver rather than making every
# caller coerce.
_TagTolerantLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?[0-9][0-9_]*(?:[eE][-+]?[0-9]+)
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def load_config(path: str) -> Dict[str, Any]:
    """Load a merged job config. Raises a readable error for a bad path."""
    if not path:
        raise ValueError("--config/-c is required")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=_TagTolerantLoader)
    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping, got {type(cfg).__name__}: {path}")
    return cfg


def detect_task(cfg: Dict[str, Any]) -> str:
    """Infer whether a config describes segmentation or detection.

    Segmentation markers win when both are present because a Seg config can
    legitimately mention `num_classes` at the top level (a Detection marker),
    while a Detection config never carries `train_dataset:`.
    """
    seg_markers = ("train_dataset", "val_dataset", "iters", "lr_scheduler")
    det_markers = ("TrainDataset", "EvalDataset", "architecture", "TrainReader", "LearningRate")
    if any(k in cfg for k in seg_markers):
        return SEG
    if any(k in cfg for k in det_markers):
        return DET
    # A `model:` block alone is ambiguous; segmentation is the safer default
    # because its config is the smaller, stricter of the two.
    if "model" in cfg:
        return SEG
    raise ValueError(
        "Could not determine the task from the config. Expected either "
        "segmentation keys (train_dataset / iters / model) or detection keys "
        "(TrainDataset / architecture / LearningRate). Pass --task seg|det to override."
    )


# ---------------------------------------------------------------------------
# Small readers shared by both tasks
# ---------------------------------------------------------------------------


def tag_name(node: Any, default: str = "") -> str:
    """Recover a component's name from either `name:`, `type:` or its YAML tag."""
    if isinstance(node, dict):
        for key in ("name", "type", "__tag__"):
            value = node.get(key)
            if isinstance(value, str) and value:
                return value
    elif isinstance(node, str) and node:
        return node
    return default


def get_num(cfg: Dict[str, Any], *keys: str, default: Optional[float] = None) -> Optional[float]:
    """First key that holds something numeric, else `default`."""
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return default


def get_int(cfg: Dict[str, Any], *keys: str, default: Optional[int] = None) -> Optional[int]:
    value = get_num(cfg, *keys, default=None)
    return default if value is None else int(value)


def get_bool(cfg: Dict[str, Any], *keys: str, default: bool = False) -> bool:
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "yes", "no", "on", "off"):
            return value.strip().lower() in ("true", "yes", "on")
    return default


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def size_pair(value: Any, default: Any = None) -> Optional[List[int]]:
    """Normalise `512`, `[512, 512]` and `[[512, 512]]` to `[w, h]`."""
    if value is None:
        return None if default is None else size_pair(default)
    if isinstance(value, (int, float)):
        return [int(value), int(value)]
    items = as_list(value)
    if len(items) == 1 and isinstance(items[0], (list, tuple)):
        items = list(items[0])
    numbers = [int(v) for v in items if isinstance(v, (int, float))]
    if len(numbers) >= 2:
        return [numbers[0], numbers[1]]
    if len(numbers) == 1:
        return [numbers[0], numbers[0]]
    return None if default is None else size_pair(default)
