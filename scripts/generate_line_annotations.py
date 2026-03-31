#!/usr/bin/env python3
"""
Generate line-*.txt polygon annotations from existing JSON bounding box annotations.

The db_dataset.py expects line-{name}.txt files containing space-separated polygon coordinates:
    x1 y1 x2 y2 x3 y3 x4 y4  (for each text region, one per line)

This script converts the xyxy bboxes from JSON to 4-point polygon format.
"""

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse


def bbox_to_polygon(bbox: list) -> str:
    """Convert [x1, y1, x2, y2] bbox to polygon string "x1 y1 x2 y1 x2 y2 x1 y2"."""
    x1, y1, x2, y2 = bbox
    # 4 corners: top-left, top-right, bottom-right, bottom-left
    polygon = f"{x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"
    return polygon


def process_json_file(json_path: Path, output_dir: Path, min_confidence: float = 0.0) -> tuple:
    """Process a single JSON file and create corresponding line-*.txt file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        text_blocks = data.get('text_blocks', [])

        # Filter by confidence if specified
        if min_confidence > 0:
            text_blocks = [b for b in text_blocks if b.get('confidence', 1.0) >= min_confidence]

        if not text_blocks:
            return json_path.name, 0, "no_blocks"

        # Generate polygon lines
        lines = []
        for block in text_blocks:
            bbox = block['bbox']
            polygon_str = bbox_to_polygon(bbox)
            lines.append(polygon_str)

        # Write to line-{name}.txt
        stem = json_path.stem
        output_path = output_dir / f"line-{stem}.txt"

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        return json_path.name, len(lines), "success"

    except Exception as e:
        return json_path.name, 0, str(e)


def main():
    parser = argparse.ArgumentParser(description='Generate line annotations from JSON')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing JSON annotation files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for line-*.txt files (default: same as input)')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                        help='Minimum confidence threshold (0.0 = keep all)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = list(input_dir.glob('*.json'))
    print(f"Found {len(json_files)} JSON files in {input_dir}")

    # Check how many line-*.txt already exist
    existing_lines = list(output_dir.glob('line-*.txt'))
    print(f"Found {len(existing_lines)} existing line-*.txt files")

    # Process files in parallel
    results = {'success': 0, 'no_blocks': 0, 'error': 0}
    total_polygons = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_json_file, jp, output_dir, args.min_confidence): jp
            for jp in json_files
        }

        for future in tqdm(as_completed(futures), total=len(json_files), desc="Generating line annotations"):
            name, count, status = future.result()
            if status == "success":
                results['success'] += 1
                total_polygons += count
            elif status == "no_blocks":
                results['no_blocks'] += 1
            else:
                results['error'] += 1

    print(f"\nResults:")
    print(f"  Success: {results['success']} files")
    print(f"  No blocks (skipped): {results['no_blocks']} files")
    print(f"  Errors: {results['error']} files")
    print(f"  Total polygons written: {total_polygons}")

    # Verify output
    new_lines = list(output_dir.glob('line-*.txt'))
    print(f"\nTotal line-*.txt files now: {len(new_lines)}")


if __name__ == '__main__':
    main()
