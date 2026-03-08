# Auto Training project

## Requirements

Auto training platform 是一款基于typescript开发的web应用，用于将PaddleDetection PaddleClas 复杂的配置功能可视化为便捷的人机交互。同时包含设备管理，标注数据，导入数据库，训练参数生成，模型训练与验证，训练日志追踪，数据验证等功能

功能需求如下：

- 设备管理
  - 监控GPU和CPU的内存占用，usage，温度，系统环境等
- 标注数据
  - 支持标注object detection和classification数据，并能转化为coco格式的数据集
- 导入数据库
  - 支持导入自定义的COCO数据集，可以显示当前数据集的统计信息，比如类别数量，标注数量，总训练数量等
- 生成训练参数
  - 以PaddleDetection或者PaddleClas为base训练工具，生成完整的训练参数并保存
- 模型训练与验证
  - 支持训练模型，验证模型等功能
- 训练日志追踪
  - 支持把PaddleDetection或者PaddleClas输出的日志解析并可视化。
- 数据验证
  - 支持用户使用自己的图片测试，支持单文件测试和文件夹测试，输出结果能生成summary report

具体页面设计和功能如下：

使用全英文设计，每个功能页面相互独立，使用database保存运行数据

- 主界面 dashboard：
  
- 展示目前有多少个项目。多少个数据集，多少模型，多少个训练参数，正在运行多少个等等信息，图表显示GPU的利用率和显存占用温度，当前运行环境等,能指定当前python运行环境和Paddle Detection路径或PaddleClas等训练框架的路径
  
- 项目界面：
  
- 创建项目是流程起点，创建后可以在项目内创建数据集，模型，训练参数等，该页面会展示目前存在的所有项目，以及项目的增删查改功能，选择某个项目后，显示项目详细内容
  
- 数据集界面：

  - 可以导入自定义的COCO数据集，可以指定training和eval的json文件和image文件夹，要能展示当前选择数据集的统计信息，图标展示类别数量，标注数量，总训练数量等，并提供下载统计图功能

- 模型界面：

  - PaddleDetection使用YAML文件构建模型的结构，该页面提供多种模型选择，每种模型映射到具体的YAML文件，根据配置重组到一个YAML文件中并保存为一个完整的YAML文件，格式请依据下面给出的实例，同时包括Advance功能，可以修改完整的YAML文件

  - 完整模型和训练参数

    ```yaml
    metric: COCO
    num_classes: 9
    
    TrainDataset:
      name: COCODataSet
      image_dir: train
      anno_path: annotations/instance_train.json
      dataset_dir: G:/datasets/_test/coco
      data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']
    
    EvalDataset:
      name: COCODataSet
      image_dir: val
      anno_path: annotations/instance_val.json
      dataset_dir: G:/datasets/_test/coco
      allow_empty: true
    
    TestDataset:
      name: ImageFolder
      anno_path: G:/datasets/_test/coco/annotations/instance_val.json # also support txt (like VOC's label_list.txt)
      dataset_dir: dataset/coco # if set, anno_path will be 'dataset_dir/anno_path'
    
    # ##################################################################runtime
    use_gpu: true
    use_xpu: false
    use_mlu: false
    use_npu: false
    use_gcu: false
    log_iter: 20
    save_dir: D:/_work/projects/autoTraining/runningWeb/mini-services/training-service/outputs/cmm5nuccm0001ui7osczyl9nn
    snapshot_epoch: 1
    print_flops: false
    print_params: false
    
    # Exporting the model
    export:
      post_process: True  # Whether post-processing is included in the network when export model.
      nms: True           # Whether NMS is included in the network when export model.
      benchmark: False    # It is used to testing model performance, if set `True`, post-process and NMS will not be exported.
      fuse_conv_bn: False
    
    # ################################################################## opt
    epoch: 10
    
    LearningRate:
      base_lr: 0.001
      schedulers:
        - name: CosineDecay
          max_epochs: 96
        - name: LinearWarmup
          start_factor: 0.
          epochs: 5
    
    OptimizerBuilder:
      optimizer:
        momentum: 0.9
        type: Momentum
      regularizer:
        factor: 0.0005
        type: L2
    
    # ########################################################## model
    architecture: YOLOv3
    norm_type: sync_bn
    use_ema: true
    ema_decay: 0.9998
    ema_black_list: ['proj_conv.weight']
    custom_black_list: ['reduce_mean']
    
    YOLOv3:
      backbone: CSPResNet
      neck: CustomCSPPAN
      yolo_head: PPYOLOEHead
      post_process: ~
    
    CSPResNet:
      layers: [3, 6, 6, 3]
      channels: [64, 128, 256, 512, 1024]
      return_idx: [1, 2, 3]
      use_large_stem: True
      use_alpha: True
    
    CustomCSPPAN:
      out_channels: [768, 384, 192]
      stage_num: 1
      block_num: 3
      act: 'swish'
      spp: true
    
    PPYOLOEHead:
      fpn_strides: [32, 16, 8]
      grid_cell_scale: 5.0
      grid_cell_offset: 0.5
      static_assigner_epoch: 30
      use_varifocal_loss: True
      loss_weight: {class: 1.0, iou: 2.5, dfl: 0.5}
      static_assigner:
        name: ATSSAssigner
        topk: 9
      assigner:
        name: TaskAlignedAssigner
        topk: 13
        alpha: 1.0
        beta: 6.0
      nms:
        name: MultiClassNMS
        nms_top_k: 1000
        keep_top_k: 300
        score_threshold: 0.01
        nms_threshold: 0.7
    
    # #################################################### reader
    worker_num: 4
    eval_height: &eval_height 640
    eval_width: &eval_width 640
    eval_size: &eval_size [*eval_height, *eval_width]
    
    TrainReader:
      sample_transforms:
        - Decode: {}
        - RandomDistort: {}
        - RandomExpand: {fill_value: [123.675, 116.28, 103.53]}
        - RandomCrop: {}
        - RandomFlip: {}
      batch_transforms:
        - BatchRandomResize: {target_size: [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768], random_size: True, random_interp: True, keep_ratio: False}
        - NormalizeImage: {mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}
        - Permute: {}
        - PadGT: {}
      batch_size: 8
      shuffle: true
      drop_last: true
      use_shared_memory: true
      collate_batch: true
    
    EvalReader:
      sample_transforms:
        - Decode: {}
        - Resize: {target_size: *eval_size, keep_ratio: False, interp: 2}
        - NormalizeImage: {mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}
        - Permute: {}
      batch_size: 2
    
    TestReader:
      inputs_def:
        image_shape: [3, *eval_height, *eval_width]
      sample_transforms:
        - Decode: {}
        - Resize: {target_size: *eval_size, keep_ratio: False, interp: 2}
        - NormalizeImage: {mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}
        - Permute: {}
      batch_size: 1
    ##################################################################
    
    output_dir: D:/_work/projects/autoTraining/runningWeb/mini-services/training-service/outputs/cmm5nuccm0001ui7osczyl9nn
    weights: output/ppyoloe_crn_s_300e_coco/model_final
    
    pretrain_weights: https://paddledet.bj.bcebos.com/models/pretrained/CSPResNetb_s_pretrained.pdparams
    depth_mult: 0.33
    width_mult: 0.50
    
    ```

    

- 训练页面：

  - 该页面能指定使用CPU或GPU训练，是否使用AMP，训练时同时验证，开启vdl等，生成的指令示例如下示例
  - 开启训练的cmd命令示例如下：

```cmd
python -m paddle.distributed.launch --gpus 0 tools/train.py -c configs/ppyoloe/Gen1/ppyoloe_plus_crn_s_80e_PHO_abs_step4.yml --amp --use_vdl=true --vdl_log_dir=output/Gen1/ppyoloe_plus_crn_s_80e_PHO_abs_step4/vdl
```

- 监控界面：

  - 将PaddleDetection正在训练的日志记录可视化，PaddleDetection的日志示例如下：

  - ```txt
    [03/03 10:20:46] ppdet.engine.callbacks INFO: Epoch: [0] [100/827] learning_rate: 0.000024 loss: 4.534513 loss_cls: 2.415748 loss_iou: 0.530099 loss_dfl: 1.626410 loss_l1: 1.266741 eta: 1 day, 20:03:57 batch_cost: 2.3920 data_cost: 2.1271 ips: 4.1807 images/s, max_mem_reserved: 14321 MB, max_mem_allocated: 12036 MB
    ```

- 验证界面：

  - 提供验证已训练模型的精度，使用eval.py单独验证某个数据集的mAP指标等，可以测试单个图片的结果，批量测试文件夹的结果，将指定模型文件转化为TensorRT的格式等

  - 测试图片的指令如下：

    ```cmd
    python tools/infer.py -c configs/ppyoloe/PoleTrainScripts/ppyoloe_plus_crn_s_80e_Pole.yml --infer_dir=G:\\data\\gen4\\capCyc\\L-H34847- -o weights=G:\\models\\hawkeye\\gen4\\PoletipWindow\\training\\ppyoloe_plus_crn_s_80e_Pole\best_model.pdparams
    ```

  - 转化导出部署到TensorRT的指令如下：

    ```cmd
    python tools/export_model.py -c configs/ppyoloe/PoleTrainScripts/ppyoloe_plus_crn_s_80e_Pole.yml -o weights=output/Pole/ppyoloe_plus_crn_s_80e_Pole/best_model.pdparams trt=True --output_dir output/Pole/ppyoloe_plus_crn_s_80e_Pole
    ```






