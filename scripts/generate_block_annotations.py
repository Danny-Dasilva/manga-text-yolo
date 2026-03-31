#!/usr/bin/env python3
"""Generate block annotations using reference model."""

import argparse
import json
import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def nms(boxes: list, iou_thresh: float = 0.5) -> list:
    """Apply non-maximum suppression to remove overlapping boxes."""
    if not boxes:
        return []

    # Convert to numpy
    bboxes = np.array([b['bbox'] for b in boxes])
    scores = np.array([b['confidence'] for b in boxes])

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # Sort by confidence
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # Compute IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU below threshold
        mask = iou <= iou_thresh
        order = order[1:][mask]

    return [boxes[i] for i in keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='data/comic-text-detector.onnx', help='Reference ONNX model')
    parser.add_argument('--input-dir', default='data/train/images', help='Input image directory')
    parser.add_argument('--output-dir', default='data/train/annotations_v2', help='Output annotation directory')
    parser.add_argument('--conf-thresh', type=float, default=0.25)
    parser.add_argument('--skip-existing', action='store_true', help='Skip images with existing annotations')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sess = ort.InferenceSession(args.model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    image_paths = list(Path(args.input_dir).glob('*.jpg'))
    print(f"Processing {len(image_paths)} images...")

    skipped = 0
    for img_path in tqdm(image_paths):
        # Skip if annotation exists
        ann_path = output_dir / f'{img_path.stem}.json'
        if args.skip_existing and ann_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        orig_h, orig_w = img.shape[:2]

        # Preprocess
        img_resized = cv2.resize(img, (1024, 1024))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        # Run inference
        outputs = sess.run(None, {'images': tensor})
        blk = outputs[0][0]  # [num_detections, 7]

        # Filter by confidence and convert
        text_blocks = []
        for box in blk:
            x, y, w, h, conf = box[:5]
            if conf > args.conf_thresh:
                # Reference model uses pixel coords (0-1024)
                # Scale to original image size
                scale_x = orig_w / 1024
                scale_y = orig_h / 1024

                x1 = int((x - w/2) * scale_x)
                y1 = int((y - h/2) * scale_y)
                x2 = int((x + w/2) * scale_x)
                y2 = int((y + h/2) * scale_y)

                text_blocks.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf)
                })

        # Apply NMS to remove duplicates
        text_blocks = nms(text_blocks, iou_thresh=0.5)

        # Save annotation
        ann_path = output_dir / f'{img_path.stem}.json'
        with open(ann_path, 'w') as f:
            json.dump({
                'text_blocks': text_blocks,
                'original_size': [orig_w, orig_h]
            }, f, indent=2)

    print(f"Generated annotations for {len(image_paths) - skipped} images in {output_dir} (skipped {skipped} existing)")

if __name__ == '__main__':
    main()
