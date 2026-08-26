# torchtrain — PyTorch trainer for AutoTrain

The PyTorch counterpart to the PaddleSeg / PaddleDetection repositories the
platform drives. It is bundled with the app rather than cloned separately, and it
deliberately mirrors the *shape* of the Paddle repos so the web app needs as
little framework-specific code as possible.

```
torchtrain/                 <- repository root (SystemConfig.torchPath)
  torchtrain/               <- importable package (cf. `paddleseg`, `ppdet`)
    config.py               YAML loading (tolerates PaddleDetection's !Tags)
    logger.py               Paddle-identical stdout formats
    utils.py                optimizer / LR schedule / AMP / checkpoints
    cli.py                  shared argparse + Paddle-compatible flag aliases
    seg/                    semantic segmentation  (framework: TorchSeg)
    det/                    object detection       (framework: TorchDet)
    ad/                     anomaly detection      (framework: TorchAnomaly)
  tools/
    train.py  val.py  predict.py  export.py
    eval.py  infer.py       aliases for the PaddleDetection spellings
  requirements.txt
  requirements-ad.txt       anomalib, for TorchAnomaly only (separate venv!)
```

## Frameworks

| `project.framework` | Task | Config schema | Trains by |
|---|---|---|---|
| `TorchSeg` | semantic segmentation | PaddleSeg's | iterations (`iters`) |
| `TorchDet` | object detection | PaddleDetection's | epochs (`epoch`) |
| `TorchAnomaly` | unsupervised anomaly detection | anomalib's | steps (`trainer.max_steps`) |

`TorchAnomaly` is the odd one out and deliberately so: there is no Paddle
anomaly-detection framework to mirror, so it consumes anomalib's own
`model:` / `data:` / `trainer:` schema. See [`torchtrain/ad/`](torchtrain/ad/) and
`docs/ANOMALY_DETECTION_DESIGN.md`. It is a **thin adapter**, not an
implementation: anomalib does the modelling, and this package only translates the
config, reproduces the Paddle log lines, restores validation metrics that
anomalib's default evaluator omits, and puts the checkpoint where the platform
looks for it.

**It needs its own virtualenv.** anomalib pins Lightning and jsonargparse; do not
install it into the environment TorchSeg/TorchDet use.

## Why the Paddle schemas and log formats are reused

This is the single most important design decision in the package, and it is
intentional rather than lazy:

* **Config schema.** `TorchSeg` consumes PaddleSeg-shaped YAML and `TorchDet`
  consumes PaddleDetection-shaped YAML. The platform's dataset/training/model
  config generators, the deep merge in `src/lib/yaml-merge.ts`, and the
  create/edit dialogs are therefore shared, and a project can be moved between
  runtimes by changing `project.framework` alone.
* **Log format.** stdout is byte-compatible with the Paddle formats, so
  `src/lib/log-parsers/*` parses torch runs unchanged, and the monitoring charts,
  progress bar and best-checkpoint tracking work with no extra code. See
  `torchtrain/logger.py` for the exact strings that are reproduced.
* **Checkpoint layout.** `<save_dir>/best_model/model.pt` mirrors PaddleSeg's
  `<save_dir>/best_model/model.pdparams`, so `/api/checkpoints` walks one layout
  for three frameworks.

Where the runtimes genuinely differ, the difference is documented rather than
papered over — see "Deliberate differences" below.

## CLI

Flag names match the Paddle repos. `-c`/`--config` and the PaddleDetection
spellings (`-o key=value`, `--infer_img`, `--infer_dir`, `--output_dir`) are
accepted as aliases.

```bash
# Train (task inferred from the config's keys; --task seg|det|ad overrides)
python tools/train.py --config merged.yml --save_dir out --do_eval [--amp] [--use_vdl]

# Evaluate
python tools/val.py --config merged.yml --model_path out/best_model/model.pt

# Predict (a file or a directory; TIFF input is supported directly)
python tools/predict.py --config merged.yml --model_path out/best_model/model.pt \
    --image_path images/ --save_dir predictions/

# Export
python tools/export.py --config merged.yml --model_path out/best_model/model.pt \
    --save_dir export_model/ [--format torchscript|onnx]
```

Anomaly jobs use the same flags, with `model.ckpt` instead of `model.pt`:

```bash
python tools/train.py   --config merged.yml --save_dir out --do_eval
python tools/val.py     --config merged.yml --model_path out/best_model/model.ckpt
python tools/predict.py --config merged.yml --model_path out/best_model/model.ckpt \
    --image_path images/ --save_dir predictions/     # writes heatmaps/ + scores.json
python tools/export.py  --config merged.yml --model_path out/best_model/model.ckpt \
    --save_dir export_model/ --format onnx           # or torchscript / openvino
```

The GPU comes from `CUDA_VISIBLE_DEVICES` (the platform sets it), so there is no
`--gpus` flag and no distributed launcher: this trainer is single-process.

## Supported architectures

Keep these tables in sync with `SEG_ARCHITECTURES` / `TORCH_SEG_ARCHITECTURES` /
`TORCH_DET_PRESETS` in `src/lib/model-yaml.ts` — the UI validates against them.

### TorchSeg (`torchtrain/seg/models.py`)

| `model.type` | Backbones | Logits | Notes |
|---|---|---|---|
| `UNet` | — | 1 | Hand-written; the best default for a few dozen images |
| `DeepLabV3P` | ResNet50 / ResNet101 / MobileNetV3-Large | 2 | torchvision, auxiliary FCN head |
| `FCN` | ResNet50 / ResNet101 | 2 | torchvision, auxiliary FCN head |
| `LRASPP` | MobileNetV3-Large | 1 | torchvision, lightweight |

`len(loss.types)` must equal the logits count — the same rule PaddleSeg enforces.
For the torchvision models, two loss entries is what *enables* the auxiliary head,
so the model and the loss can never disagree.

Losses: `CrossEntropyLoss`, `OhemCrossEntropyLoss`, `DiceLoss`, `FocalLoss`,
`BCELoss`, `LovaszSoftmaxLoss`, `MixedLoss`.

### TorchDet (`torchtrain/det/models.py`)

| `architecture` | Backbones |
|---|---|
| `FasterRCNN` | ResNet50-FPN, ResNet50-FPN-v2, MobileNetV3-Large-FPN, MobileNetV3-Large-320-FPN |
| `RetinaNet` | ResNet50-FPN, ResNet50-FPN-v2 |
| `FCOS` | ResNet50-FPN |
| `SSD` | VGG16, MobileNetV3-Large |

`pretrain_weights` selects the initialisation:

| Value | Meaning |
|---|---|
| `COCO` (default) | COCO-pretrained detector, classifier replaced to match `num_classes`. Biggest accuracy win on small datasets. Not supported for `SSD`, which falls back to ImageNet. |
| `ImageNet` | ImageNet backbone only |
| `<path>.pt` | A checkpoint produced by this trainer |
| empty | Random initialisation |

PaddleDetection families with no torchvision equivalent (`YOLOv3`/PP-YOLOE,
`PicoDet`, `DETR`/RT-DETR, `CenterNet`) raise an error naming a comparable
alternative instead of silently training a different network.

### TorchAnomaly (`torchtrain/ad/`, models from anomalib)

Keep in sync with `ANOMALY_PRESETS` in `src/lib/model-yaml.ts`.

| `model.class_path` | Tiling | Loss curve | Notes |
|---|---|---|---|
| `anomalib.models.Patchcore` | yes | no | Memory bank; one pass, no backprop. Best few-shot baseline. |
| `anomalib.models.Padim` | yes | no | Per-position Gaussians — needs a rigidly aligned part. |
| `anomalib.models.Stfpm` | yes | yes | Student-teacher; optimizer fixed inside anomalib (SGD, lr 0.4). |
| `anomalib.models.EfficientAd` | **no** | yes | Best accuracy/latency. Requires `train_batch_size: 1`, forbids `Normalize` in the transform, and downloads a teacher checkpoint + ImageNette on first run. |
| `anomalib.models.Supersimplenet` | no | yes | Has a `supervised` flag upstream; anomalib wires only the unsupervised path. |

Only PaDiM / PatchCore / ReverseDistillation / STFPM implement a `tiler`, so the
adapter refuses a tiling request for anything else instead of letting anomalib
raise a bare `ValueError: Model does not support tiling.`

## Deliberate differences from the Paddle frameworks

| Topic | Behaviour |
|---|---|
| Distributed training | Single-process. `--gpus` and `paddle.distributed.launch` do not apply; the platform never wraps a torch job in a launcher. |
| Detection normalisation | `NormalizeImage` / `Permute` / `PadGT` describe work torchvision does *inside* the model (`GeneralizedRCNNTransform`), so they are accepted and ignored. Applying them here would double-normalise the input. |
| Detection input size | `Resize` / `BatchRandomResize` target sizes become the model's `min_size` / `max_size`, which is how torchvision expresses scale jitter. |
| `regularizer.type` | torch optimizers only implement L2 weight decay, so an `L1` request is ignored (and the UI does not offer it). |
| Instance segmentation | Not supported: the wired torchvision detectors are box-only. Use PaddleDetection for polygon masks. |
| TIFF inference input | Read directly, so the platform's PaddleSeg TIFF-to-PNG staging step is skipped for torch jobs. |
| Detection ONNX export | Not supported (variable-size list inputs); `tools/export.py` writes a self-describing checkpoint bundle instead. Segmentation supports both TorchScript and ONNX. |

## Metrics

* **Segmentation** reproduces PaddleSeg's definitions exactly (`torchtrain/seg/metrics.py`),
  so numbers are directly comparable to a PaddleSeg run you are migrating from:
  mIoU / overall Acc / Kappa / Dice plus per-class IoU, precision and recall.
* **Detection** uses `pycocotools` when it is importable, and otherwise a
  self-contained NumPy COCOeval re-implementation (`torchtrain/det/metrics.py`).
  The two agree to 0.0 on all 12 stats and on per-class AP50; the fallback exists
  because `pycocotools` needs a C toolchain and is a recurring install problem on
  Windows.
* **Anomaly detection** reports image and pixel AUROC/F1 from anomalib's metrics,
  plus the adaptive score threshold. Two things to know:
  * pixel metrics are non-strict, so they simply do not appear when the defect
    images have no masks;
  * anomalib's default evaluator registers *test* metrics only, so
    `torchtrain/ad/evaluator.py` installs an evaluator that also reports during
    validation. Without it a `fit()` run logs no metrics at all, the AUROC chart
    stays empty and `ModelCheckpoint` has nothing to monitor.

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Optional: `pycocotools` (reference COCO metrics), `tensorboard` (`--use_vdl`),
`onnx` (`tools/export.py --format onnx`). All degrade gracefully when absent.

For `TorchAnomaly`, in a **separate** virtualenv:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-ad.txt      # anomalib==2.6.*
```

Then in the app's **Settings**:

1. **Framework Paths → PyTorch Trainer Path**: this folder (pre-filled by default).
   All three torch frameworks share it.
2. **Framework Python Environments**: add `TorchSeg` and `TorchDet` entries
   pointing at the interpreter that has PyTorch, and a separate `TorchAnomaly`
   entry pointing at the anomalib venv. This is required — the per-GPU mapping
   normally points at PaddlePaddle environments, which cannot run torch jobs, and
   a TorchAnomaly job in a plain torch env fails with an actionable
   "anomalib is not installed" message.
