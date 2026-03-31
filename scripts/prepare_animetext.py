#!/usr/bin/env python3
"""
Prepare AnimeText dataset for comic-text-detector training.

Downloads and converts AnimeText from HuggingFace to all 3 annotation formats:
1. Segmentation masks (mask-{name}.png) - Binary PNG masks
2. Line annotations (line-{name}.txt) - Polygon coordinates for DBNet
3. Block annotations ({name}.txt) - YOLO format for block detection
4. JSON annotations ({name}.json) - For unified detection loader

AnimeText format:
- bbox: [x_center, y_center, width, height] normalized 0-1
- category: 0 or 1 (text block types)

Usage:
    python scripts/prepare_animetext.py --num-samples 50000 --output-dir data/animetext
    python scripts/prepare_animetext.py --num-samples 10000 --split val --output-dir data/animetext_val
"""

import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Lazy imports for faster startup
def get_pil():
    from PIL import Image, ImageDraw
    return Image, ImageDraw

def get_datasets():
    from datasets import load_dataset
    return load_dataset


def normalized_to_xyxy(bbox, img_width, img_height):
    """Convert normalized [xcen, ycen, w, h] to pixel [x1, y1, x2, y2]."""
    xcen, ycen, w, h = bbox
    x1 = int((xcen - w/2) * img_width)
    y1 = int((ycen - h/2) * img_height)
    x2 = int((xcen + w/2) * img_width)
    y2 = int((ycen + h/2) * img_height)
    # Clamp to image bounds
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(0, min(x2, img_width - 1))
    y2 = max(0, min(y2, img_height - 1))
    return [x1, y1, x2, y2]


def xyxy_to_polygon(bbox):
    """Convert [x1, y1, x2, y2] to polygon string "x1 y1 x2 y1 x2 y2 x1 y2"."""
    x1, y1, x2, y2 = bbox
    return f"{x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"


def process_sample(sample, output_dir, img_dir, ann_dir):
    """Process a single sample and save all annotation formats."""
    Image, ImageDraw = get_pil()

    try:
        image = sample['image']
        image_id = sample['image_id']
        objects = sample['objects']

        img_width, img_height = image.size
        bboxes_norm = objects['bbox']  # List of [xcen, ycen, w, h]
        categories = objects.get('category', [1] * len(bboxes_norm))

        if not bboxes_norm:
            return image_id, 0, "no_objects"

        # Create filename
        filename = f"animetext_{image_id}"

        # 1. Save image
        img_path = img_dir / f"{filename}.jpg"
        image.convert('RGB').save(img_path, quality=95)

        # Convert all bboxes to pixel coords
        bboxes_xyxy = [normalized_to_xyxy(b, img_width, img_height) for b in bboxes_norm]

        # 2. Generate YOLO txt (block detection) - keep normalized format
        yolo_lines = []
        for bbox, cat in zip(bboxes_norm, categories):
            xcen, ycen, w, h = bbox
            # Use category 1 for all text (or keep original category)
            yolo_lines.append(f"1 {xcen:.6f} {ycen:.6f} {w:.6f} {h:.6f}")

        yolo_path = ann_dir / f"{filename}.txt"
        with open(yolo_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        # 3. Generate JSON (unified detection)
        json_data = {
            "text_blocks": [
                {
                    "bbox": bbox,
                    "confidence": 1.0,
                    "area": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                }
                for bbox in bboxes_xyxy
            ],
            "original_size": [img_width, img_height]
        }

        json_path = ann_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f)

        # 4. Generate line annotations (polygon format for DBNet)
        polygon_lines = [xyxy_to_polygon(bbox) for bbox in bboxes_xyxy]

        line_path = ann_dir / f"line-{filename}.txt"
        with open(line_path, 'w') as f:
            f.write('\n'.join(polygon_lines))

        # 5. Generate mask (segmentation) using PIL
        mask = Image.new('L', (img_width, img_height), 0)
        draw = ImageDraw.Draw(mask)
        for bbox in bboxes_xyxy:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], fill=255)

        mask_path = ann_dir / f"mask-{filename}.png"
        mask.save(mask_path)

        return image_id, len(bboxes_xyxy), "success"

    except Exception as e:
        return sample.get('image_id', 'unknown'), 0, str(e)


def main():
    parser = argparse.ArgumentParser(description='Prepare AnimeText dataset')
    parser.add_argument('--num-samples', type=int, default=50000,
                        help='Number of samples to download (default: 50000)')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to use')
    parser.add_argument('--output-dir', type=str, default='data/animetext',
                        help='Output directory')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Starting index for sampling')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    img_dir = output_dir / 'images'
    ann_dir = output_dir / 'annotations'

    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Map split names (dataset uses 'valid' not 'validation')
    split_map = {'validation': 'valid', 'train': 'train', 'test': 'test'}
    actual_split = split_map.get(args.split, args.split)

    print(f"Loading AnimeText dataset (split={actual_split}, streaming=True)...")
    load_dataset = get_datasets()

    # Use streaming to avoid downloading entire dataset
    ds = load_dataset('deepghs/AnimeText', split=actual_split, streaming=True)

    # Skip to start index and take num_samples
    if args.start_idx > 0:
        ds = ds.skip(args.start_idx)
    ds = ds.take(args.num_samples)

    print(f"Processing {args.num_samples} samples...")
    print(f"Output: {output_dir}")

    results = {'success': 0, 'no_objects': 0, 'error': 0}
    total_annotations = 0

    # Process samples (can't easily parallelize streaming dataset)
    for sample in tqdm(ds, total=args.num_samples, desc="Processing"):
        image_id, count, status = process_sample(sample, output_dir, img_dir, ann_dir)

        if status == "success":
            results['success'] += 1
            total_annotations += count
        elif status == "no_objects":
            results['no_objects'] += 1
        else:
            results['error'] += 1
            if results['error'] <= 5:
                print(f"Error processing {image_id}: {status}")

    print(f"\n=== Results ===")
    print(f"Success: {results['success']} images")
    print(f"No objects (skipped): {results['no_objects']} images")
    print(f"Errors: {results['error']} images")
    print(f"Total annotations: {total_annotations}")

    # Verify output
    print(f"\n=== Output verification ===")
    print(f"Images: {len(list(img_dir.glob('*.jpg')))}")
    print(f"YOLO txt: {len(list(ann_dir.glob('animetext_*.txt')))}")
    print(f"JSON: {len(list(ann_dir.glob('*.json')))}")
    print(f"Line txt: {len(list(ann_dir.glob('line-*.txt')))}")
    print(f"Masks: {len(list(ann_dir.glob('mask-*.png')))}")

    # Create symlinks in standard format for training
    print(f"\n=== Creating training symlinks ===")

    # Create symlinks for easier access (optional)
    for img_path in img_dir.glob('*.jpg'):
        stem = img_path.stem
        # PNG symlink (for segmentation - points to mask)
        png_link = ann_dir / f"{stem}.png"
        if not png_link.exists():
            mask_path = ann_dir / f"mask-{stem}.png"
            if mask_path.exists():
                png_link.symlink_to(mask_path.name)

    print("Done! Dataset is ready for training.")
    print(f"\nTo use this data:")
    print(f"  - Images: {img_dir}")
    print(f"  - Annotations: {ann_dir}")


if __name__ == '__main__':
    main()
