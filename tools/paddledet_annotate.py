"""Axis-aligned detection pre-labeling for the annotation page.

Runs inside the PaddleDetection environment (cwd = the PaddleDetection repo).
Produces the same JSON shape the annotation predict API returns for the
torch localization trainer, with task "detect".
"""

import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppdet_root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--confidence', type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, os.path.abspath(args.ppdet_root))

    import warnings
    warnings.filterwarnings('ignore')

    import paddle
    from PIL import Image, ImageOps
    from ppdet.core.workspace import load_config, merge_config
    from ppdet.engine import Trainer
    from ppdet.metrics import get_infer_results
    from ppdet.data.source.category import get_categories
    from ppdet.utils.check import check_gpu, check_config, check_version

    cfg = load_config(args.config)
    merge_config({'weights': args.weights})
    for device in ('use_npu', 'use_xpu', 'use_gpu', 'use_mlu', 'use_gcu'):
        if device not in cfg:
            setattr(cfg, device, False)
    paddle.set_device('gpu' if cfg.use_gpu else 'cpu')
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    trainer = Trainer(cfg, mode='test')
    anno_file = trainer.dataset.get_anno()
    if not anno_file or not os.path.isfile(anno_file):
        print("Category file not found: %s; the job's dataset annotation file "
              "is needed to name classes" % anno_file, file=sys.stderr)
        sys.exit(2)
    trainer.load_weights(cfg.weights)

    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = sorted(
        entry.path for entry in os.scandir(args.image_dir)
        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in extensions)
    if not images:
        print('No images found in %s' % args.image_dir, file=sys.stderr)
        sys.exit(2)

    predict_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)), 'predict-tmp')
    outs = trainer.predict(images, output_dir=predict_dir,
                           save_results=False, visualize=False)

    imid2path = {int(k): v for k, v in trainer.dataset.get_imid2path().items()}
    clsid2catid, catid2name = get_categories(cfg.metric, anno_file=anno_file)
    labels = [catid2name[clsid2catid[i]] for i in sorted(clsid2catid)]

    per_image = {}
    for batch in outs:
        for row in get_infer_results(batch, clsid2catid).get('bbox', []):
            per_image.setdefault(int(row['image_id']), []).append(row)

    result_images = []
    total = 0
    for image_path in images:
        im_id = None
        for key, value in imid2path.items():
            if os.path.abspath(value) == os.path.abspath(image_path):
                im_id = key
                break
        if im_id is None:
            continue
        with Image.open(image_path) as handle:
            handle = ImageOps.exif_transpose(handle)
            width, height = handle.size
        instances = []
        for row in per_image.get(im_id, []):
            if row['score'] < args.confidence:
                continue
            x, y, w, h = row['bbox']
            x1 = min(max(x, 0.0), float(width))
            y1 = min(max(y, 0.0), float(height))
            x2 = min(max(x + w, 0.0), float(width))
            y2 = min(max(y + h, 0.0), float(height))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            instances.append({
                'label': catid2name[row['category_id']],
                'confidence': round(float(row['score']), 4),
                'bbox_xyxy_px': [x1, y1, x2, y2],
            })
        total += len(instances)
        result_images.append({
            'source': os.path.basename(image_path),
            'width': width, 'height': height, 'instances': instances,
        })

    payload = {'task': 'detect', 'labels': labels, 'keypointNames': [],
               'images': result_images}
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False)
    print('wrote %s: %d images, %d boxes, %d classes'
          % (args.output, len(result_images), total, len(labels)))


if __name__ == '__main__':
    main()
