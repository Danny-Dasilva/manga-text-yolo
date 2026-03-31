#!/usr/bin/env python3
"""
Evaluate model detections against ground truth.

Compares:
- Block detections vs block_annotations
- Computes precision, recall, IoU metrics
- Visualizes GT vs predictions

Usage:
    python scripts/evaluate_vs_gt.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np


def load_ground_truth(ann_path: Path) -> List[Dict]:
    """Load ground truth annotations from JSON."""
    if not ann_path.exists():
        return []
    with open(ann_path) as f:
        data = json.load(f)
    return data.get('text_blocks', [])


def compute_iou(box1: List[int], box2: List[int]) -> float:
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


def match_predictions_to_gt(
    predictions: List[Dict],
    gt_boxes: List[Dict],
    iou_threshold: float = 0.5
) -> Tuple[int, int, int, List[float]]:
    """
    Match predictions to ground truth.

    Returns:
        (true_positives, false_positives, false_negatives, matched_ious)
    """
    if not gt_boxes:
        return 0, len(predictions), 0, []

    if not predictions:
        return 0, 0, len(gt_boxes), []

    # Track which GT boxes have been matched
    gt_matched = [False] * len(gt_boxes)
    matched_ious = []

    tp = 0
    fp = 0

    for pred in predictions:
        pred_box = pred['bbox']
        best_iou = 0
        best_gt_idx = -1

        for i, gt in enumerate(gt_boxes):
            if gt_matched[i]:
                continue
            gt_box = gt['bbox']
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)

    return tp, fp, fn, matched_ious


def draw_evaluation(
    image: np.ndarray,
    predictions: List[Dict],
    gt_boxes: List[Dict],
    output_path: str
):
    """Draw GT (blue) and predictions (green) on image."""
    vis = image.copy()

    # Draw GT boxes in BLUE
    for gt in gt_boxes:
        x1, y1, x2, y2 = gt['bbox']
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw predictions in GREEN
    for pred in predictions:
        x1, y1, x2, y2 = pred['bbox']
        conf = pred.get('confidence', 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{conf:.2f}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imwrite(output_path, vis)
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate against ground truth')
    parser.add_argument('--checkpoint', default='runs/block_yolo26/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--image-dir', default='data/merged_val/images',
                       help='Directory with test images')
    parser.add_argument('--block-ann-dir', default='data/merged_val/block_annotations',
                       help='Directory with block annotations')
    parser.add_argument('--output-dir', default='outputs/evaluation',
                       help='Output directory for visualizations')
    parser.add_argument('--limit', type=int, default=10,
                       help='Number of images to evaluate')
    parser.add_argument('--conf-threshold', type=float, default=0.1,
                       help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for matching')
    parser.add_argument('--device', default='cuda',
                       help='Device for inference')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    from src.inference.detector import TextDetector

    print("Loading model...")
    detector = TextDetector(
        model_path=args.checkpoint,
        input_size=1024,
        device=args.device,
        conf_threshold=args.conf_threshold,
        backend='pytorch'
    )

    # Get test images
    image_dir = Path(args.image_dir)
    block_ann_dir = Path(args.block_ann_dir)

    image_paths = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
    image_paths = image_paths[:args.limit]

    print(f"Evaluating on {len(image_paths)} images...")
    print("=" * 60)

    # Metrics accumulators
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []

    for img_path in image_paths:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Get predictions
        result = detector(img)
        predictions = result.text_blocks

        # Load ground truth
        gt_path = block_ann_dir / f"{img_path.stem}.json"
        gt_boxes = load_ground_truth(gt_path)

        # Match and compute metrics
        tp, fp, fn, ious = match_predictions_to_gt(
            predictions, gt_boxes, args.iou_threshold
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(ious)

        # Draw visualization
        vis_path = output_dir / f"eval_{img_path.stem}.jpg"
        draw_evaluation(img, predictions, gt_boxes, str(vis_path))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"{img_path.name}: GT={len(gt_boxes)} Pred={len(predictions)} "
              f"TP={tp} FP={fp} FN={fn} P={precision:.2f} R={recall:.2f}")

    # Overall metrics
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = np.mean(all_ious) if all_ious else 0

    print(f"\nTotal: TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"Avg IoU (matched): {avg_iou:.3f}")

    print(f"\nVisualizations saved to: {output_dir}")
    print("  BLUE boxes = Ground Truth")
    print("  GREEN boxes = Predictions")


if __name__ == '__main__':
    main()
