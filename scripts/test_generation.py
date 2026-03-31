#!/usr/bin/env python3
"""
Test Generation Script: Visualize model predictions vs ground truth.

Generates side-by-side comparisons of:
- Original image
- Ground truth mask
- Model predicted mask

Usage:
    python scripts/test_generation.py --checkpoint runs/segmentation_yolo26/best.pt \
        --img-dir data/merged_val/images --mask-dir data/merged_val/annotations \
        --output outputs/test_generation --num-samples 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.detector import create_text_detector
from models.heads import TEXTDET_MASK


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get backbone name from checkpoint
    backbone_name = 'yolo26s.pt'
    if 'config' in checkpoint and checkpoint['config']:
        backbone_name = checkpoint['config'].get('backbone', backbone_name)

    # Create model
    model = create_text_detector(
        backbone_name=backbone_name,
        pretrained_backbone=False,
        freeze_backbone=True,
        device=device
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Handle torch.compile prefix
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.forward_mode = TEXTDET_MASK
    return model


def preprocess_image(image: np.ndarray, target_size: int = 1024) -> tuple:
    """Preprocess image for model input."""
    h, w = image.shape[:2]

    # Resize maintaining aspect ratio
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    # Convert to tensor
    tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)

    return tensor, (h, w), (new_h, new_w)


def postprocess_mask(mask: torch.Tensor, original_size: tuple, padded_size: tuple) -> np.ndarray:
    """Postprocess model output mask."""
    mask = mask.squeeze().cpu().numpy()

    # Crop to padded size
    new_h, new_w = padded_size
    mask = mask[:new_h, :new_w]

    # Resize to original
    h, w = original_size
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # Threshold
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask


def create_comparison_image(
    original: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    title: str = ""
) -> np.ndarray:
    """Create side-by-side comparison image."""
    h, w = original.shape[:2]

    # Resize masks to match original
    if gt_mask.shape[:2] != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if pred_mask.shape[:2] != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Create overlay versions
    gt_overlay = original.copy()
    pred_overlay = original.copy()

    # Apply colored masks
    gt_color = np.zeros_like(original)
    gt_color[:, :, 1] = gt_mask  # Green for ground truth
    pred_color = np.zeros_like(original)
    pred_color[:, :, 2] = pred_mask  # Red for prediction

    gt_overlay = cv2.addWeighted(gt_overlay, 0.7, gt_color, 0.3, 0)
    pred_overlay = cv2.addWeighted(pred_overlay, 0.7, pred_color, 0.3, 0)

    # Stack horizontally: Original | GT | Prediction
    comparison = np.hstack([original, gt_overlay, pred_overlay])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, 'Ground Truth (Green)', (w + 10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, 'Prediction (Red)', (2*w + 10, 30), font, 1, (0, 0, 255), 2)

    if title:
        cv2.putText(comparison, title, (10, h - 10), font, 0.6, (255, 255, 255), 1)

    return comparison


def calculate_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    """Calculate IoU and other metrics."""
    gt_binary = (gt_mask > 127).astype(np.float32)
    pred_binary = (pred_mask > 127).astype(np.float32)

    intersection = np.sum(gt_binary * pred_binary)
    union = np.sum(gt_binary) + np.sum(pred_binary) - intersection

    iou = intersection / (union + 1e-6)
    precision = intersection / (np.sum(pred_binary) + 1e-6)
    recall = intersection / (np.sum(gt_binary) + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    parser = argparse.ArgumentParser(description='Generate test visualizations')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--img-dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--mask-dir', type=str, required=True,
                        help='Directory containing ground truth masks')
    parser.add_argument('--output', type=str, default='outputs/test_generation',
                        help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sample selection')
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, device)
    print(f"Model loaded on {device}")

    # Get image files
    img_dir = Path(args.img_dir)
    mask_dir = Path(args.mask_dir)

    image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    print(f"Found {len(image_files)} images")

    # Sample images
    if len(image_files) > args.num_samples:
        image_files = random.sample(image_files, args.num_samples)

    # Process images
    all_metrics = []

    for img_path in tqdm(image_files, desc="Generating visualizations"):
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Load ground truth mask
        mask_path = mask_dir / f"mask-{img_path.stem}.png"
        if not mask_path.exists():
            mask_path = mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            print(f"Mask not found for {img_path.name}, skipping...")
            continue

        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue

        # Preprocess
        tensor, orig_size, padded_size = preprocess_image(image)
        tensor = tensor.to(device)

        # Run inference
        with torch.no_grad():
            pred = model(tensor)
            # Model output is already in [0,1] range, no need for sigmoid
            # Only apply sigmoid if output is unbounded (logits)
            if pred.min() < 0 or pred.max() > 1:
                pred = torch.sigmoid(pred)

        # Postprocess
        pred_mask = postprocess_mask(pred, orig_size, padded_size)

        # Calculate metrics
        metrics = calculate_metrics(gt_mask, pred_mask)
        all_metrics.append(metrics)

        # Create comparison
        title = f"{img_path.name} | IoU: {metrics['iou']:.3f} | F1: {metrics['f1']:.3f}"
        comparison = create_comparison_image(image, gt_mask, pred_mask, title)

        # Save
        output_path = output_dir / f"comparison_{img_path.stem}.png"
        cv2.imwrite(str(output_path), comparison)

    # Print summary
    if all_metrics:
        avg_iou = np.mean([m['iou'] for m in all_metrics])
        avg_f1 = np.mean([m['f1'] for m in all_metrics])
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])

        print("\n" + "="*50)
        print("Summary Metrics")
        print("="*50)
        print(f"Samples: {len(all_metrics)}")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average F1: {avg_f1:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print("="*50)
        print(f"\nVisualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
