"""Semantic segmentation task (framework name: `TorchSeg`).

The config schema is PaddleSeg's, so a config authored for PaddleSeg in the web
app trains here without edits:

    batch_size: 4
    iters: 2000

    train_dataset:
      type: Dataset
      dataset_root: /path/to/data
      train_path: /path/to/data/train.txt
      num_classes: 3
      mode: train
      transforms:
        - type: Resize
          target_size: [512, 512]
        - type: RandomHorizontalFlip
        - type: Normalize

    val_dataset: {...}

    optimizer:
      type: SGD
      momentum: 0.9
      weight_decay: 0.0005

    lr_scheduler:
      type: PolynomialDecay
      learning_rate: 0.01
      power: 0.9

    model:
      type: UNet
      num_classes: 3

    loss:
      types:
        - type: CrossEntropyLoss
      coef: [1]
"""

__all__ = ["dataset", "transforms", "models", "losses", "metrics", "trainer", "predictor"]
