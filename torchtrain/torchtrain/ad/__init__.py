"""
Unsupervised anomaly detection (`framework: TorchAnomaly`), backed by anomalib.

This package is deliberately thin. Unlike `torchtrain.seg` and `torchtrain.det`,
which implement their models, everything here delegates to
`anomalib <https://github.com/open-edge-platform/anomalib>`_ and only adds the
four things the platform needs and anomalib does not provide:

1. **Config translation** (`config.py`). The platform merges three YAML files
   into one document; the `autotrain:` block in it is ours and anomalib would
   reject it, so it is split off and the remainder is written out as a pure
   anomalib config for `Engine.from_config`.
2. **Paddle-compatible logging** (`logger.py`). anomalib logs through a rich
   progress bar, which is unparsable. The callback here prints the same
   `[TRAIN]` / `[EVAL]` lines PaddleSeg does, so `src/lib/log-parsers/anomaly.ts`
   and the monitoring charts work with no anomalib-specific code.
3. **Validation metrics during training** (`evaluator.py`). anomalib's default
   `Evaluator` registers *test* metrics only, so a `fit()` run would log nothing
   at all and the platform's curves would be empty.
4. **The platform's checkpoint layout** (`runner.py`). `Engine.fit()` writes to a
   versioned directory of its own choosing; the platform expects
   `<save_dir>/best_model/model.ckpt`.

anomalib is imported lazily inside the functions that need it, so a plain
PyTorch environment can still run TorchSeg / TorchDet jobs from this same
repository.

Nothing is imported here on purpose: `logger.py` subclasses a Lightning
`Callback`, so importing it eagerly would make `torchtrain.ad.config` — which is
plain YAML handling and is unit-testable on its own — depend on Lightning being
installed.
"""

__all__ = ["config", "evaluator", "logger", "runner"]
