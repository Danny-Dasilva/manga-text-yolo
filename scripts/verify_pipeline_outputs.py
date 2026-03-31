#!/usr/bin/env python3
"""Verify that our pipeline produces outputs matching the original comic-text-detector model.

Compares:
1. Block detection (blk) - speech bubble boxes
2. Segmentation (seg) - text mask
3. Text line detection (det) - DBNet probability/threshold maps

Usage:
    python scripts/verify_pipeline_outputs.py --image test.jpg --original-model data/comic-text-detector.onnx
"""

import argparse
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
import json


def load_and_preprocess(image_path: str, size: int = 1024) -> tuple[np.ndarray, tuple[int, int]]:
    """Load and preprocess image for model input."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, (size, size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    return tensor, (orig_h, orig_w)


def run_original_model(model_path: str, tensor: np.ndarray) -> dict:
    """Run original comic-text-detector ONNX model."""
    sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # Get input name
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: tensor})

    # Original model outputs: blk, seg, det
    output_names = [o.name for o in sess.get_outputs()]
    print(f"Original model outputs: {output_names}")
    print(f"  blk shape: {outputs[0].shape}")
    print(f"  seg shape: {outputs[1].shape}")
    print(f"  det shape: {outputs[2].shape}")

    return {
        'blk': outputs[0],  # [1, N, 7] - block detections
        'seg': outputs[1],  # [1, 1, 1024, 1024] - segmentation mask
        'det': outputs[2],  # [1, 2, 1024, 1024] - DBNet prob/threshold
    }


def run_our_pipeline(model_path: str, tensor: np.ndarray) -> dict:
    """Run our trained pipeline ONNX model."""
    sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: tensor})

    output_names = [o.name for o in sess.get_outputs()]
    print(f"Our pipeline outputs: {output_names}")
    for i, (name, out) in enumerate(zip(output_names, outputs)):
        print(f"  {name} shape: {out.shape}")

    # Map to same structure as original
    result = {}
    for name, out in zip(output_names, outputs):
        result[name] = out

    return result


def compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> float:
    """Compute mean IoU between two sets of boxes."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return 0.0

    # Simple box matching by proximity
    ious = []
    for box1 in boxes1:
        x1_1, y1_1, x2_1, y2_1 = box1[:4]
        best_iou = 0
        for box2 in boxes2:
            x1_2, y1_2, x2_2, y2_2 = box2[:4]

            # Intersection
            xi1 = max(x1_1, x1_2)
            yi1 = max(y1_1, y1_2)
            xi2 = min(x2_1, x2_2)
            yi2 = min(y2_1, y2_2)

            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

            # Union
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - inter_area

            iou = inter_area / union_area if union_area > 0 else 0
            best_iou = max(best_iou, iou)

        ious.append(best_iou)

    return np.mean(ious) if ious else 0.0


def compute_mask_similarity(mask1: np.ndarray, mask2: np.ndarray) -> dict:
    """Compute similarity metrics between two masks."""
    # Binarize
    m1 = (mask1 > 0.5).astype(np.float32)
    m2 = (mask2 > 0.5).astype(np.float32)

    intersection = np.sum(m1 * m2)
    union = np.sum(m1) + np.sum(m2) - intersection
    iou = intersection / union if union > 0 else 0

    # Dice coefficient
    dice = (2 * intersection) / (np.sum(m1) + np.sum(m2)) if (np.sum(m1) + np.sum(m2)) > 0 else 0

    # MSE
    mse = np.mean((mask1 - mask2) ** 2)

    return {
        'iou': float(iou),
        'dice': float(dice),
        'mse': float(mse),
    }


def extract_boxes_from_blk(blk: np.ndarray, conf_thresh: float = 0.25) -> np.ndarray:
    """Extract bounding boxes from blk output."""
    boxes = []
    for det in blk[0]:  # Remove batch dimension
        x, y, w, h, conf = det[:5]
        if conf > conf_thresh:
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            boxes.append([x1, y1, x2, y2, conf])
    return np.array(boxes) if boxes else np.array([]).reshape(0, 5)


def visualize_comparison(image_path: str, original: dict, ours: dict, output_dir: str):
    """Create visualization comparing outputs."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Draw block detections
    img_blocks = img.copy()

    # Original boxes in green
    orig_boxes = extract_boxes_from_blk(original['blk'])
    for box in orig_boxes:
        x1, y1, x2, y2 = (box[:4] * np.array([w/1024, h/1024, w/1024, h/1024])).astype(int)
        cv2.rectangle(img_blocks, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Our boxes in blue (if available)
    if 'blk' in ours:
        our_boxes = extract_boxes_from_blk(ours['blk'])
        for box in our_boxes:
            x1, y1, x2, y2 = (box[:4] * np.array([w/1024, h/1024, w/1024, h/1024])).astype(int)
            cv2.rectangle(img_blocks, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imwrite(str(output_dir / 'blocks_comparison.jpg'), img_blocks)

    # Segmentation masks
    if 'seg' in original:
        orig_seg = (original['seg'][0, 0] * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / 'seg_original.png'), orig_seg)

    if 'seg' in ours:
        our_seg = (ours['seg'][0, 0] * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / 'seg_ours.png'), our_seg)

    # DBNet detection maps
    if 'det' in original:
        orig_det_prob = (original['det'][0, 0] * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / 'det_prob_original.png'), orig_det_prob)

    if 'det' in ours:
        our_det_prob = (ours['det'][0, 0] * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / 'det_prob_ours.png'), our_det_prob)

    print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Verify pipeline outputs match original model')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--original-model', default='data/comic-text-detector.onnx', help='Original ONNX model')
    parser.add_argument('--our-model', default=None, help='Our trained ONNX model (optional)')
    parser.add_argument('--output-dir', default='outputs/verification', help='Output directory for visualizations')
    parser.add_argument('--conf-thresh', type=float, default=0.25, help='Confidence threshold for detections')
    args = parser.parse_args()

    print(f"Loading image: {args.image}")
    tensor, orig_size = load_and_preprocess(args.image)

    print(f"\nRunning original model: {args.original_model}")
    original_outputs = run_original_model(args.original_model, tensor)

    if args.our_model:
        print(f"\nRunning our model: {args.our_model}")
        our_outputs = run_our_pipeline(args.our_model, tensor)

        # Compare outputs
        print("\n=== Comparison Results ===")

        # Block detection comparison
        if 'blk' in original_outputs and 'blk' in our_outputs:
            orig_boxes = extract_boxes_from_blk(original_outputs['blk'], args.conf_thresh)
            our_boxes = extract_boxes_from_blk(our_outputs['blk'], args.conf_thresh)
            box_iou = compute_iou(orig_boxes, our_boxes)
            print(f"Block Detection:")
            print(f"  Original boxes: {len(orig_boxes)}")
            print(f"  Our boxes: {len(our_boxes)}")
            print(f"  Mean IoU: {box_iou:.4f}")

        # Segmentation comparison
        if 'seg' in original_outputs and 'seg' in our_outputs:
            seg_metrics = compute_mask_similarity(
                original_outputs['seg'][0, 0],
                our_outputs['seg'][0, 0]
            )
            print(f"Segmentation:")
            print(f"  IoU: {seg_metrics['iou']:.4f}")
            print(f"  Dice: {seg_metrics['dice']:.4f}")
            print(f"  MSE: {seg_metrics['mse']:.6f}")

        # DBNet detection comparison
        if 'det' in original_outputs and 'det' in our_outputs:
            det_metrics = compute_mask_similarity(
                original_outputs['det'][0, 0],
                our_outputs['det'][0, 0]
            )
            print(f"Text Line Detection (DBNet):")
            print(f"  IoU: {det_metrics['iou']:.4f}")
            print(f"  Dice: {det_metrics['dice']:.4f}")
            print(f"  MSE: {det_metrics['mse']:.6f}")

        # Visualize
        visualize_comparison(args.image, original_outputs, our_outputs, args.output_dir)
    else:
        print("\nNo comparison model provided. Showing original model outputs only.")
        visualize_comparison(args.image, original_outputs, {}, args.output_dir)

        # Print summary of original model output structure
        print("\n=== Original Model Output Structure ===")
        print("This is what our pipeline should match:")
        print(f"  blk: {original_outputs['blk'].shape} - Block/bubble detections")
        print(f"  seg: {original_outputs['seg'].shape} - Segmentation mask")
        print(f"  det: {original_outputs['det'].shape} - DBNet text line detection")


if __name__ == '__main__':
    main()
