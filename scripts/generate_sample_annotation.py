#!/usr/bin/env python3
"""
Generate sample annotation visualization for comic text detection.

Creates a 3-panel visualization similar to dmMaze/comic-text-detector homepage:
- Left panel: Original image
- Center panel: Segmentation mask (white text on black)
- Right panel: Detection overlay with boxes and lines

Usage:
    python scripts/generate_sample_annotation.py \
        --image path/to/manga.jpg \
        --output outputs/sample_annotation.png \
        [--dbnet-model runs/detection_v5/best.pt] \
        [--yolo-model runs/yolo_blocks/yolov10s_blocks/weights/best.pt]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch

# Colors (BGR format for OpenCV) - matching original comic-text-detector
COLORS = {
    'block_box': (127, 255, 127),      # Light green - block bounding boxes
    'line_polygon': (0, 127, 255),     # Orange - text line polygons
    'line_rect': (127, 127, 0),        # Cyan - min area rectangles (optional)
}


def draw_textblocks(canvas: np.ndarray, text_blocks: list) -> None:
    """
    Draw text blocks and their lines on canvas.

    Args:
        canvas: Image to draw on (modified in-place)
        text_blocks: List of TextBlock dicts with 'bbox', 'lines', etc.
    """
    # Dynamic line width based on image size
    lw = max(round(sum(canvas.shape) / 2 * 0.003), 2)

    # Draw block bounding boxes
    for block in text_blocks:
        bbox = block.get('bbox', block.get('xyxy', []))
        if len(bbox) >= 4:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLORS['block_box'], lw)

        # Draw line polygons within this block
        lines = block.get('lines', [])
        for line in lines:
            if isinstance(line, np.ndarray):
                pts = line.reshape((-1, 1, 2)).astype(np.int32)
            else:
                pts = np.array(line, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], True, COLORS['line_polygon'], lw)


def draw_raw_lines(canvas: np.ndarray, lines: list) -> None:
    """
    Draw raw line polygons (without block grouping).

    Args:
        canvas: Image to draw on (modified in-place)
        lines: List of line polygons (numpy arrays)
    """
    lw = max(round(sum(canvas.shape) / 2 * 0.003), 2)

    for line in lines:
        if isinstance(line, np.ndarray):
            pts = line.reshape((-1, 1, 2)).astype(np.int32)
        else:
            pts = np.array(line, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], True, COLORS['line_polygon'], lw)


def create_3panel_visualization(
    image: np.ndarray,
    mask: np.ndarray,
    text_blocks: list,
    panel_width: int = 400,
    add_labels: bool = True,
) -> np.ndarray:
    """
    Create 3-panel visualization like CTD homepage.

    Args:
        image: Original BGR image
        mask: Binary segmentation mask (grayscale)
        text_blocks: List of TextBlock dicts from detection
        panel_width: Width of each panel
        add_labels: Whether to add text labels

    Returns:
        Combined image with [Original | Mask | Detections] panels
    """
    h, w = image.shape[:2]
    aspect = h / w
    panel_h = int(panel_width * aspect)

    # Resize inputs to panel size
    img_resized = cv2.resize(image, (panel_width, panel_h))
    mask_resized = cv2.resize(mask, (panel_width, panel_h))

    # Panel 1: Original
    panel1 = img_resized.copy()

    # Panel 2: Mask (convert to 3-channel for stacking)
    panel2 = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

    # Panel 3: Detections overlay
    panel3 = img_resized.copy()

    # Scale coordinates to panel size
    scale_x = panel_width / w
    scale_y = panel_h / h

    scaled_blocks = []
    for block in text_blocks:
        scaled_block = block.copy()

        # Scale bounding box
        bbox = block.get('bbox', block.get('xyxy', []))
        if len(bbox) >= 4:
            scaled_block['bbox'] = [
                int(bbox[0] * scale_x),
                int(bbox[1] * scale_y),
                int(bbox[2] * scale_x),
                int(bbox[3] * scale_y),
            ]

        # Scale line polygons
        lines = block.get('lines', [])
        scaled_lines = []
        for line in lines:
            if isinstance(line, np.ndarray):
                scaled_line = line.copy().astype(np.float32)
                scaled_line[:, 0] *= scale_x
                scaled_line[:, 1] *= scale_y
                scaled_lines.append(scaled_line.astype(np.int32))
            elif isinstance(line, list):
                scaled_line = [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in line]
                scaled_lines.append(np.array(scaled_line, dtype=np.int32))
        scaled_block['lines'] = scaled_lines

        scaled_blocks.append(scaled_block)

    draw_textblocks(panel3, scaled_blocks)

    # Add labels
    if add_labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        label_color = (255, 255, 255)
        shadow_color = (0, 0, 0)

        labels = [
            (panel1, 'Original'),
            (panel2, 'Segmentation'),
            (panel3, 'Detection'),
        ]

        for panel, label in labels:
            # Shadow for readability
            cv2.putText(panel, label, (12, 27), font, font_scale, shadow_color, font_thickness + 1)
            cv2.putText(panel, label, (10, 25), font, font_scale, label_color, font_thickness)

    # Combine horizontally with small gap
    gap = 3
    total_width = panel_width * 3 + gap * 2
    combined = np.zeros((panel_h, total_width, 3), dtype=np.uint8)
    combined[:, :panel_width] = panel1
    combined[:, panel_width + gap:panel_width * 2 + gap] = panel2
    combined[:, panel_width * 2 + gap * 2:] = panel3

    return combined


def main():
    parser = argparse.ArgumentParser(
        description='Generate sample annotation visualization for comic text detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with validation image
    python scripts/generate_sample_annotation.py \\
        --image data/animetext/images/animetext_1000041.jpg

    # Custom output path
    python scripts/generate_sample_annotation.py \\
        --image my_manga.jpg \\
        --output my_visualization.png

    # Specify model paths
    python scripts/generate_sample_annotation.py \\
        --image test.jpg \\
        --dbnet-model runs/detection_v5/best.pt \\
        --yolo-model runs/yolo_blocks/yolov10s_blocks/weights/best.pt
        """,
    )
    parser.add_argument('--image', required=True, help='Input manga/comic image')
    parser.add_argument('--output', default='outputs/sample_annotation.png',
                        help='Output visualization path')
    parser.add_argument('--dbnet-model', default='runs/detection_v5/best.pt',
                        help='Path to DBNet checkpoint')
    parser.add_argument('--yolo-model', default='runs/yolo_blocks/yolov10s_blocks/weights/best.pt',
                        help='Path to YOLOv10s block detector')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Inference device')
    parser.add_argument('--panel-width', type=int, default=400,
                        help='Width of each panel in visualization')
    parser.add_argument('--no-labels', action='store_true',
                        help='Disable panel labels')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--mask-threshold', type=float, default=0.3,
                        help='Threshold for binary mask')

    args = parser.parse_args()

    # Validate input
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Resolve model paths relative to project root
    dbnet_path = project_root / args.dbnet_model if not Path(args.dbnet_model).is_absolute() else Path(args.dbnet_model)
    yolo_path = project_root / args.yolo_model if not Path(args.yolo_model).is_absolute() else Path(args.yolo_model)

    if not dbnet_path.exists():
        print(f"Error: DBNet model not found: {dbnet_path}")
        sys.exit(1)

    if not yolo_path.exists():
        print(f"Error: YOLO model not found: {yolo_path}")
        sys.exit(1)

    print(f"Loading combined model...")
    print(f"  DBNet: {dbnet_path}")
    print(f"  YOLO:  {yolo_path}")

    # Load combined detector
    from src.inference.detector import CombinedTextDetector

    detector = CombinedTextDetector(
        dbnet_model_path=str(dbnet_path),
        yolo_model_path=str(yolo_path),
        input_size=1024,
        device=args.device,
        conf_threshold=args.conf_threshold,
        mask_threshold=args.mask_threshold,
    )

    # Load and process image
    print(f"Processing image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        sys.exit(1)

    # Run detection
    result = detector(image)

    print(f"Detection results:")
    print(f"  Text blocks: {len(result.text_blocks)}")
    total_lines = sum(len(blk.get('lines', [])) for blk in result.text_blocks)
    print(f"  Total lines: {total_lines}")

    # Create visualization
    vis = create_3panel_visualization(
        image,
        result.mask,
        result.text_blocks,
        panel_width=args.panel_width,
        add_labels=not args.no_labels,
    )

    # Save output
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)
    print(f"Saved visualization to: {output_path}")

    # Also save individual components for reference
    components_dir = output_path.parent / f"{output_path.stem}_components"
    components_dir.mkdir(exist_ok=True)

    cv2.imwrite(str(components_dir / "original.png"), image)
    cv2.imwrite(str(components_dir / "mask.png"), result.mask)

    # Detection overlay only
    detection_overlay = image.copy()
    draw_textblocks(detection_overlay, result.text_blocks)
    cv2.imwrite(str(components_dir / "detection.png"), detection_overlay)

    print(f"Saved individual components to: {components_dir}")


if __name__ == '__main__':
    main()
