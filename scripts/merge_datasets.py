#!/usr/bin/env python3
"""
Merge multiple datasets for training.

Creates a unified training directory by creating symlinks to images and annotations
from multiple source directories.

Usage:
    python scripts/merge_datasets.py \
        --sources data/train data/animetext \
        --output data/merged_train \
        --val-sources data/val data/animetext_val \
        --val-output data/merged_val
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm


def create_symlinks(source_dir: Path, output_img_dir: Path, output_ann_dir: Path, prefix: str = ""):
    """Create symlinks for all images and annotations from source to output."""

    # Find image and annotation directories
    if (source_dir / 'images').exists():
        img_dir = source_dir / 'images'
        ann_dir = source_dir / 'annotations'
    else:
        # Assume flat structure (images and annotations in same directory)
        img_dir = source_dir
        ann_dir = source_dir

    created = {'images': 0, 'annotations': 0}

    # Link images
    for img_path in img_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
            target = output_img_dir / f"{prefix}{img_path.name}"
            if not target.exists():
                rel_path = os.path.relpath(img_path, output_img_dir)
                os.symlink(rel_path, target)
                created['images'] += 1

    # Link annotations (all types)
    for ann_path in ann_dir.glob('*'):
        if ann_path.suffix.lower() in ['.txt', '.json', '.png']:
            target = output_ann_dir / f"{prefix}{ann_path.name}"
            if not target.exists():
                rel_path = os.path.relpath(ann_path, output_ann_dir)
                os.symlink(rel_path, target)
                created['annotations'] += 1

    return created


def verify_annotations(ann_dir: Path):
    """Verify all annotation types are present."""
    counts = {
        'images': len(list(ann_dir.parent.glob('images/*'))),
        'json': len(list(ann_dir.glob('*.json'))),
        'yolo_txt': len([f for f in ann_dir.glob('*.txt') if not f.name.startswith('line-')]),
        'line_txt': len(list(ann_dir.glob('line-*.txt'))),
        'masks': len(list(ann_dir.glob('mask-*.png'))) + len(list(ann_dir.glob('*.png'))),
    }
    return counts


def main():
    parser = argparse.ArgumentParser(description='Merge datasets')
    parser.add_argument('--sources', nargs='+', required=True,
                        help='Source directories to merge')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for merged training data')
    parser.add_argument('--val-sources', nargs='*', default=[],
                        help='Validation source directories')
    parser.add_argument('--val-output', type=str, default=None,
                        help='Output directory for merged validation data')
    args = parser.parse_args()

    # Setup output directories
    output_dir = Path(args.output)
    output_img_dir = output_dir / 'images'
    output_ann_dir = output_dir / 'annotations'
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_ann_dir.mkdir(parents=True, exist_ok=True)

    print(f"Merging {len(args.sources)} sources into {output_dir}")

    total_created = {'images': 0, 'annotations': 0}

    for i, source in enumerate(args.sources):
        source_path = Path(source)
        print(f"\nProcessing source {i+1}: {source_path}")

        if not source_path.exists():
            print(f"  WARNING: Source does not exist, skipping")
            continue

        # Use source name as prefix to avoid collisions
        prefix = "" if i == 0 else f"{source_path.name}_"

        created = create_symlinks(source_path, output_img_dir, output_ann_dir, prefix)
        total_created['images'] += created['images']
        total_created['annotations'] += created['annotations']
        print(f"  Created {created['images']} image links, {created['annotations']} annotation links")

    print(f"\n=== Training Dataset Summary ===")
    print(f"Total images linked: {total_created['images']}")
    print(f"Total annotations linked: {total_created['annotations']}")

    counts = verify_annotations(output_ann_dir)
    print(f"\nAnnotation breakdown:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # Process validation if specified
    if args.val_sources and args.val_output:
        val_output = Path(args.val_output)
        val_img_dir = val_output / 'images'
        val_ann_dir = val_output / 'annotations'
        val_img_dir.mkdir(parents=True, exist_ok=True)
        val_ann_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Merging Validation Data ===")
        val_total = {'images': 0, 'annotations': 0}

        for i, source in enumerate(args.val_sources):
            source_path = Path(source)
            print(f"Processing val source: {source_path}")

            if not source_path.exists():
                print(f"  WARNING: Source does not exist, skipping")
                continue

            prefix = "" if i == 0 else f"{source_path.name}_"
            created = create_symlinks(source_path, val_img_dir, val_ann_dir, prefix)
            val_total['images'] += created['images']
            val_total['annotations'] += created['annotations']

        print(f"Validation total: {val_total['images']} images, {val_total['annotations']} annotations")

    print("\nDone! Merged dataset ready for training.")


if __name__ == '__main__':
    main()
