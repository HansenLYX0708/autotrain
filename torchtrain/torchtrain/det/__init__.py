"""Object detection task (framework name: `TorchDet`).

The config schema is PaddleDetection's, so a config authored for
PaddleDetection in the web app trains here without edits:

    metric: COCO
    num_classes: 6            # foreground classes, background excluded

    TrainDataset:
      name: COCODataSet
      dataset_dir: /path/to/COCO/my_dataset
      image_dir: data/train
      anno_path: data/annotations/instance_train.json

    EvalDataset: {...}

    epoch: 24
    architecture: FasterRCNN
    FasterRCNN:
      backbone: ResNet
      neck: FPN

    LearningRate:
      base_lr: 0.01
      schedulers:
        - !CosineDecay
          max_epochs: 24
        - !LinearWarmup
          start_factor: 0.001
          epochs: 1

    OptimizerBuilder:
      optimizer: {type: Momentum, momentum: 0.9}
      regularizer: {type: L2, factor: 0.0001}

    TrainReader:
      batch_size: 2
      sample_transforms:
        - Decode: {}
        - RandomFlip: {}

Two schema details that matter:

* `num_classes` counts **foreground classes only** (PaddleDetection's
  convention). torchvision expects background included, so the model is built
  with `num_classes + 1` and dataset labels are shifted to `1..N`.
* Learning-rate schedulers are identified by YAML *tag* (`- !CosineDecay`), which
  `torchtrain.config` preserves as a `__tag__` key.
"""

__all__ = ["dataset", "transforms", "models", "metrics", "trainer", "predictor", "exporter"]
