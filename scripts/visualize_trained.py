#!/usr/bin/env python3
"""Visualize outputs from trained model checkpoint."""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.inference.detector import TextDetector


def visualize_result(img, result, output_path: str):
    """Draw detection results on image."""
    vis_img = img.copy()
    h, w = img.shape[:2]

    # Draw segmentation mask (red overlay)
    if result.mask is not None:
        mask = result.mask
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask, (w, h))
        mask_color = np.zeros_like(vis_img)
        mask_color[:, :, 2] = mask_resized  # Red channel
        vis_img = cv2.addWeighted(vis_img, 0.7, mask_color, 0.3, 0)

    # Draw text blocks (green boxes)
    if result.text_blocks:
        for block in result.text_blocks:
            bbox = block.get('bbox') or block.get('xyxy')
            if bbox:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                conf = block.get('confidence', block.get('score', 0))
                if conf:
                    cv2.putText(vis_img, f"{conf:.2f}", (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw text lines within block (blue polygons)
            lines = block.get('lines', [])
            for line in lines:
                if isinstance(line, np.ndarray):
                    pts = line.reshape(-1, 2).astype(np.int32)
                    cv2.polylines(vis_img, [pts], True, (255, 0, 0), 1)

    cv2.imwrite(output_path, vis_img)
    print(f"Saved: {output_path}")
    return vis_img


def main():
    parser = argparse.ArgumentParser(description='Visualize trained model outputs')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--image', help='Single image path')
    parser.add_argument('--image-dir', help='Directory of images')
    parser.add_argument('--output', default='outputs/visualizations', help='Output directory')
    parser.add_argument('--limit', type=int, default=10, help='Max images to process')
    parser.add_argument('--conf-thresh', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--device', default='cuda', help='Device')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model using inference detector
    print("Loading model...")
    detector = TextDetector(
        model_path=args.checkpoint,
        input_size=1024,
        device=args.device,
        conf_threshold=args.conf_thresh,
        backend='pytorch'
    )

    # Get images
    if args.image:
        img_paths = [Path(args.image)]
    elif args.image_dir:
        img_paths = list(Path(args.image_dir).glob('*.jpg'))[:args.limit]
        img_paths += list(Path(args.image_dir).glob('*.png'))[:args.limit - len(img_paths)]
    else:
        # Default to validation images
        img_paths = list(Path('data/merged_val/images').glob('*.jpg'))[:args.limit]

    print(f"Processing {len(img_paths)} images...")

    for img_path in img_paths:
        print(f"\nProcessing: {img_path.name}")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Could not load image, skipping")
            continue

        # Run detection
        result = detector(img)

        # Print result info
        print(f"  Mask shape: {result.mask.shape if result.mask is not None else 'None'}")
        print(f"  Text blocks: {len(result.text_blocks)}")

        # Visualize
        output_path = output_dir / f"vis_{img_path.stem}.jpg"
        visualize_result(img, result, str(output_path))

    print(f"\nDone! Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
