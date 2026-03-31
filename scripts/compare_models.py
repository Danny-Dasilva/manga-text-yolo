#!/usr/bin/env python3
"""
Compare YOLO26 trained model against reference comic-text-detector.

Compares:
- Detection quality (IoU, precision, recall if GT available)
- Inference speed (ms per image)
- Output formats and counts

Usage:
    python scripts/compare_models.py --limit 10
    python scripts/compare_models.py --image-dir data/merged_val/images --limit 10
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch


def load_yolo26_model(checkpoint_path: str, device: str = 'cuda', conf_threshold: float = 0.1):
    """Load YOLO26 trained model using inference detector."""
    from src.inference.detector import TextDetector

    detector = TextDetector(
        model_path=checkpoint_path,
        input_size=1024,
        device=device,
        conf_threshold=conf_threshold,
        backend='pytorch'
    )
    return detector


def load_reference_model(onnx_path: str, device: str = 'cuda'):
    """Load reference comic-text-detector ONNX model."""
    import onnxruntime as ort

    # Select providers based on device
    if 'cuda' in device:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f"Reference model loaded with provider: {session.get_providers()[0]}")

    # Get input/output info
    input_info = session.get_inputs()[0]
    output_info = [o.name for o in session.get_outputs()]
    print(f"  Input: {input_info.name} shape={input_info.shape}")
    print(f"  Outputs: {output_info}")

    return session


def preprocess_for_reference(image: np.ndarray, input_size: int = 1024) -> np.ndarray:
    """Preprocess image for reference ONNX model (same as original CTD)."""
    h, w = image.shape[:2]

    # Resize maintaining aspect ratio
    scale = min(input_size / h, input_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    pad_h = (input_size - new_h) // 2
    pad_w = (input_size - new_w) // 2
    padded = np.full((input_size, input_size, 3), 127, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    # Normalize and convert to NCHW
    tensor = padded.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW
    tensor = np.expand_dims(tensor, 0)  # Add batch dim

    return tensor, {'scale': scale, 'pad': (pad_h, pad_w), 'original_size': (h, w)}


def run_reference_inference(session, image: np.ndarray, input_size: int = 1024):
    """Run inference with reference ONNX model."""
    tensor, meta = preprocess_for_reference(image, input_size)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})

    return outputs, meta


def extract_blocks_from_reference(outputs, meta, conf_threshold: float = 0.3):
    """Extract text blocks from reference model outputs.

    Reference model (comic-text-detector) outputs:
    - blk: [1, N, 7] with [cx, cy, w, h, obj, cls0, cls1]
      - cx, cy: center coordinates in 1024px space
      - w, h: box width/height in pixels
      - obj: objectness score
      - cls0, cls1: class probabilities (text, non-text)
    - seg: [1, 1, H, W] segmentation mask
    - det: [1, 2, H, W] DBNet output
    """
    h, w = meta['original_size']
    scale = meta['scale']
    pad_h, pad_w = meta['pad']

    # Output order: blk, seg, det
    if len(outputs) >= 3:
        blk, seg, det = outputs[0], outputs[1], outputs[2]
    else:
        # Fallback for 2-output models
        seg, det = outputs[0], outputs[1]
        blk = None

    blocks = []

    if blk is not None and blk.size > 0:
        # blk format: [1, N, 7] with [cx, cy, w, h, obj, cls0, cls1]
        blk = blk.squeeze(0)  # [N, 7]

        for box in blk:
            # Parse box format
            cx, cy, bw, bh = box[0], box[1], box[2], box[3]
            obj = box[4]
            cls0 = box[5]  # text class probability

            # Final confidence = objectness * text_class_prob
            conf = obj * cls0

            if conf < conf_threshold:
                continue

            # Convert from center to corner format (in 1024 space)
            x1_px = cx - bw / 2
            y1_px = cy - bh / 2
            x2_px = cx + bw / 2
            y2_px = cy + bh / 2

            # Remove padding and scale back to original coordinates
            x1 = (x1_px - pad_w) / scale
            y1 = (y1_px - pad_h) / scale
            x2 = (x2_px - pad_w) / scale
            y2 = (y2_px - pad_h) / scale

            # Clip to bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            box_w = x2 - x1
            box_h = y2 - y1
            if box_w > 5 and box_h > 5:
                blocks.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf)
                })

    # Sort by confidence (highest first)
    blocks.sort(key=lambda x: x['confidence'], reverse=True)

    return blocks, seg, det


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def match_detections(boxes1, boxes2, iou_threshold: float = 0.5):
    """Match detections between two sets of boxes."""
    matched = 0
    used = set()

    for b1 in boxes1:
        best_iou = 0
        best_idx = -1
        for i, b2 in enumerate(boxes2):
            if i in used:
                continue
            iou = compute_iou(b1['bbox'], b2['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_threshold and best_idx >= 0:
            matched += 1
            used.add(best_idx)

    return matched


def draw_comparison(image: np.ndarray, yolo26_blocks, ref_blocks, output_path: str):
    """Draw both models' detections on same image for comparison."""
    vis = image.copy()

    # Draw reference blocks in RED
    for block in ref_blocks:
        x1, y1, x2, y2 = block['bbox']
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"REF:{block.get('confidence', 1):.2f}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw YOLO26 blocks in GREEN
    for block in yolo26_blocks:
        x1, y1, x2, y2 = block['bbox']
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"Y26:{block.get('confidence', 1):.2f}", (x1, y2+12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imwrite(output_path, vis)
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Compare YOLO26 vs reference CTD model')
    parser.add_argument('--yolo26-checkpoint', default='runs/block_yolo26/best.pt',
                       help='Path to YOLO26 checkpoint')
    parser.add_argument('--reference-model', default='models/comic-text-detector.onnx',
                       help='Path to reference ONNX model')
    parser.add_argument('--image-dir', default='data/merged_val/images',
                       help='Directory with test images')
    parser.add_argument('--output-dir', default='outputs/comparison',
                       help='Output directory for visualizations')
    parser.add_argument('--limit', type=int, default=10,
                       help='Number of images to test')
    parser.add_argument('--conf-threshold', type=float, default=0.1,
                       help='Confidence threshold for YOLO26')
    parser.add_argument('--ref-conf-threshold', type=float, default=0.3,
                       help='Confidence threshold for reference model')
    parser.add_argument('--device', default='cuda',
                       help='Device for inference')
    parser.add_argument('--warmup', type=int, default=3,
                       help='Warmup iterations for timing')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("=" * 60)
    print("Loading YOLO26 model...")
    yolo26 = load_yolo26_model(args.yolo26_checkpoint, args.device, args.conf_threshold)

    print("\nLoading reference model...")
    reference = load_reference_model(args.reference_model, args.device)

    # Get test images
    image_dir = Path(args.image_dir)
    image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    image_paths = image_paths[:args.limit]
    print(f"\nTesting on {len(image_paths)} images")
    print("=" * 60)

    # Warmup
    print("\nWarming up...")
    warmup_img = cv2.imread(str(image_paths[0]))
    for _ in range(args.warmup):
        _ = yolo26(warmup_img)
        tensor, _ = preprocess_for_reference(warmup_img)
        _ = reference.run(None, {reference.get_inputs()[0].name: tensor})

    # Run comparison
    results = {
        'yolo26_times': [],
        'reference_times': [],
        'yolo26_counts': [],
        'reference_counts': [],
        'matched_counts': [],
    }

    print("\nRunning comparison...")
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # YOLO26 inference
        t0 = time.perf_counter()
        yolo26_result = yolo26(img)
        yolo26_time = (time.perf_counter() - t0) * 1000

        yolo26_blocks = yolo26_result.text_blocks

        # Reference inference
        t0 = time.perf_counter()
        ref_outputs, ref_meta = run_reference_inference(reference, img)
        ref_time = (time.perf_counter() - t0) * 1000

        ref_blocks, _, _ = extract_blocks_from_reference(
            ref_outputs, ref_meta, args.ref_conf_threshold
        )

        # Match detections
        matched = match_detections(yolo26_blocks, ref_blocks, iou_threshold=0.5)

        # Store results
        results['yolo26_times'].append(yolo26_time)
        results['reference_times'].append(ref_time)
        results['yolo26_counts'].append(len(yolo26_blocks))
        results['reference_counts'].append(len(ref_blocks))
        results['matched_counts'].append(matched)

        # Draw comparison visualization
        vis_path = output_dir / f"compare_{img_path.stem}.jpg"
        draw_comparison(img, yolo26_blocks, ref_blocks, str(vis_path))

        print(f"  [{i+1}/{len(image_paths)}] {img_path.name}: "
              f"YOLO26={len(yolo26_blocks)} ({yolo26_time:.1f}ms) | "
              f"REF={len(ref_blocks)} ({ref_time:.1f}ms) | "
              f"Matched={matched}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    avg_yolo26_time = np.mean(results['yolo26_times'])
    avg_ref_time = np.mean(results['reference_times'])
    avg_yolo26_count = np.mean(results['yolo26_counts'])
    avg_ref_count = np.mean(results['reference_counts'])
    avg_matched = np.mean(results['matched_counts'])

    print(f"\n{'Metric':<25} {'YOLO26':<15} {'Reference':<15}")
    print("-" * 55)
    print(f"{'Avg inference time':<25} {avg_yolo26_time:.1f}ms{'':<8} {avg_ref_time:.1f}ms")
    print(f"{'Avg detections/image':<25} {avg_yolo26_count:.1f}{'':<12} {avg_ref_count:.1f}")
    print(f"{'Speed ratio':<25} {avg_ref_time/avg_yolo26_time:.2f}x faster{'':<5} 1.00x")

    print(f"\n{'Cross-model agreement:':<25}")
    print(f"  Avg matched detections: {avg_matched:.1f}")
    if avg_yolo26_count > 0:
        print(f"  YOLO26 recall vs REF: {100*avg_matched/avg_yolo26_count:.1f}%")
    if avg_ref_count > 0:
        print(f"  REF recall vs YOLO26: {100*avg_matched/avg_ref_count:.1f}%")

    print(f"\nVisualizations saved to: {output_dir}")
    print("  GREEN boxes = YOLO26 detections")
    print("  RED boxes = Reference model detections")

    return results


if __name__ == '__main__':
    main()
