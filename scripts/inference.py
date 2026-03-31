#!/usr/bin/env python3
"""
Inference Script for Comic Text Detection

Usage:
    # Single image
    python scripts/inference.py --model runs/train/best.pt --input image.jpg --output result.png

    # Directory of images
    python scripts/inference.py --model runs/train/best.pt --input images/ --output results/

    # With custom settings
    python scripts/inference.py --model runs/train/best.pt --input image.jpg --output result.png \
        --input-size 1024 --conf-threshold 0.5 --half
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Run comic text detection')

    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint or ONNX file')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output image or directory')

    parser.add_argument('--input-size', type=int, default=1024,
                        help='Model input size')
    parser.add_argument('--conf-threshold', type=float, default=0.4,
                        help='Confidence threshold')
    parser.add_argument('--mask-threshold', type=float, default=0.3,
                        help='Mask binarization threshold')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for processing')

    parser.add_argument('--half', action='store_true',
                        help='Use FP16 inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference')
    parser.add_argument('--backend', type=str, default='pytorch',
                        choices=['pytorch', 'onnx'],
                        help='Inference backend')

    parser.add_argument('--save-mask', action='store_true', default=True,
                        help='Save segmentation masks')
    parser.add_argument('--save-json', action='store_true', default=True,
                        help='Save detection JSON')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization with detections overlaid')

    return parser.parse_args()


def visualize_result(image: np.ndarray, result) -> np.ndarray:
    """Draw detection results on image."""
    vis = image.copy()

    # Draw mask overlay
    if result.mask is not None:
        mask_colored = np.zeros_like(vis)
        mask_colored[:, :, 1] = result.mask  # Green channel
        vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)

    # Draw text blocks
    for block in result.text_blocks:
        # Draw polygon
        pts = np.array(block['polygon'], dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

        # Draw confidence
        x, y = pts[0]
        conf = block['confidence']
        cv2.putText(vis, f"{conf:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return vis


def main():
    args = parse_args()

    from src.inference import TextDetector, BatchTextDetector

    # Check device
    import torch
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    # Initialize detector
    print(f"Loading model: {args.model}")
    detector = TextDetector(
        model_path=args.model,
        input_size=args.input_size,
        device=device,
        half=args.half,
        backend=args.backend,
        conf_threshold=args.conf_threshold,
        mask_threshold=args.mask_threshold,
    )

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Check if input is directory or single file
    if input_path.is_dir():
        # Process directory
        output_path.mkdir(parents=True, exist_ok=True)

        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            image_paths.extend(input_path.glob(ext))

        print(f"Processing {len(image_paths)} images...")

        for img_path in tqdm(image_paths):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not read {img_path}")
                continue

            result = detector(image)

            stem = img_path.stem

            if args.save_mask:
                mask_path = output_path / f"{stem}_mask.png"
                cv2.imwrite(str(mask_path), result.mask)

            if args.save_json:
                import json
                json_path = output_path / f"{stem}.json"
                with open(json_path, 'w') as f:
                    json.dump({
                        'text_blocks': result.text_blocks,
                        'original_size': result.original_size,
                    }, f, indent=2)

            if args.visualize:
                vis = visualize_result(image, result)
                vis_path = output_path / f"{stem}_vis.png"
                cv2.imwrite(str(vis_path), vis)

    else:
        # Process single file
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"Error: Could not read {input_path}")
            return

        print("Running detection...")
        result = detector(image)

        print(f"Found {len(result.text_blocks)} text blocks")

        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.save_mask:
            if output_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                cv2.imwrite(str(output_path), result.mask)
            else:
                mask_path = output_path.with_suffix('.png')
                cv2.imwrite(str(mask_path), result.mask)

        if args.save_json:
            import json
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump({
                    'text_blocks': result.text_blocks,
                    'original_size': result.original_size,
                }, f, indent=2)

        if args.visualize:
            vis = visualize_result(image, result)
            vis_path = output_path.with_name(f"{output_path.stem}_vis.png")
            cv2.imwrite(str(vis_path), vis)
            print(f"Saved visualization to: {vis_path}")

    print("Done!")


if __name__ == '__main__':
    main()
