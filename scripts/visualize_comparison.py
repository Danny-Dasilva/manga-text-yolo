#!/usr/bin/env python3
"""
Visualization Comparison Script: AnimeText_yolo vs mayocream/comic-text-detector

Compares two HuggingFace models:
- deepghs/AnimeText_yolo: YOLO-based text block detection
- mayocream/comic-text-detector-onnx: Multi-output detector (blocks, mask, lines)

Usage:
    python scripts/visualize_comparison.py --image test.jpg --output results/
    python scripts/visualize_comparison.py --image-dir datasets/test/ --output results/ --limit 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available")


# Colors for visualization (BGR format)
COLORS = {
    'animetext_box': (0, 255, 0),      # Green - AnimeText_yolo boxes
    'ctd_box': (255, 0, 0),            # Blue - mayocream CTD block boxes
    'ctd_mask': (0, 0, 255),           # Red - CTD mask overlay
    'ctd_lines': (255, 255, 0),        # Cyan - CTD lines
}


def ensure_animetext_model(model_path: Path) -> Path:
    """Ensure AnimeText_yolo model is available, download if missing."""
    if model_path.exists():
        return model_path

    print(f"Downloading deepghs/AnimeText_yolo...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="deepghs/AnimeText_yolo",
            filename="yolo12s_animetext/model.onnx",
            local_dir=model_path.parent.parent
        )
        return Path(downloaded)
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        raise


def ensure_ctd_model(model_path: Path) -> Path:
    """Ensure mayocream/comic-text-detector-onnx is available, download if missing."""
    if model_path.exists():
        return model_path

    print(f"Downloading mayocream/comic-text-detector-onnx...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="mayocream/comic-text-detector-onnx",
            filename="comic-text-detector.onnx",
            local_dir=model_path.parent
        )
        return Path(downloaded)
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        raise


def preprocess_image(image: np.ndarray, target_size: int) -> Tuple[np.ndarray, Tuple[int, int], float]:
    """
    Preprocess image for model input.

    Returns:
        - preprocessed tensor [1, 3, H, W]
        - original size (H, W)
        - scale factor
    """
    orig_h, orig_w = image.shape[:2]

    # Resize maintaining aspect ratio
    scale = target_size / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    # Convert to float and normalize
    tensor = padded.astype(np.float32) / 255.0

    # HWC -> CHW -> NCHW
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, 0)

    return tensor, (orig_h, orig_w), scale


def postprocess_yolo_boxes(
    output: np.ndarray,
    orig_size: Tuple[int, int],
    input_size: int,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Post-process AnimeText YOLO output to get bounding boxes.

    Args:
        output: Raw YOLO output [1, 5, num_anchors] or [1, num_anchors, 5]
        orig_size: Original image size (H, W)
        input_size: Model input size
        conf_threshold: Confidence threshold
        nms_threshold: NMS IoU threshold

    Returns:
        List of (x1, y1, x2, y2, confidence) tuples
    """
    # Handle different output formats
    if output.shape[1] == 5:
        # [1, 5, num_anchors] -> transpose to [num_anchors, 5]
        output = output[0].T
    elif output.shape[2] == 5:
        # [1, num_anchors, 5]
        output = output[0]
    else:
        # Assume [1, num_anchors, 5+num_classes]
        output = output[0]

    # Extract x, y, w, h, confidence
    x_center = output[:, 0]
    y_center = output[:, 1]
    width = output[:, 2]
    height = output[:, 3]
    confidence = output[:, 4]

    # Filter by confidence
    mask = confidence > conf_threshold
    if not np.any(mask):
        return []

    x_center = x_center[mask]
    y_center = y_center[mask]
    width = width[mask]
    height = height[mask]
    confidence = confidence[mask]

    # Convert to x1, y1, x2, y2
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    # Scale to original image size
    orig_h, orig_w = orig_size
    scale = max(orig_h, orig_w) / input_size

    x1 = (x1 * scale).astype(int)
    y1 = (y1 * scale).astype(int)
    x2 = (x2 * scale).astype(int)
    y2 = (y2 * scale).astype(int)

    # Clip to image bounds
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    # Apply NMS
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(float)
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        confidence.tolist(),
        conf_threshold,
        nms_threshold
    )

    if len(indices) == 0:
        return []

    indices = indices.flatten()

    results = []
    for i in indices:
        results.append((int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]), float(confidence[i])))

    return results


def postprocess_mayocream_blocks(
    output: np.ndarray,
    orig_size: Tuple[int, int],
    input_size: int = 1024,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Post-process mayocream CTD block output.

    The mayocream/comic-text-detector-onnx model outputs blocks in format:
        [1, 64512, 7] where each row is:
        [x_center, y_center, width, height, confidence, class1, class2]

    Args:
        output: Raw block output [1, 64512, 7]
        orig_size: Original image size (H, W)
        input_size: Model input size (1024)
        conf_threshold: Confidence threshold
        nms_threshold: NMS IoU threshold

    Returns:
        List of (x1, y1, x2, y2, confidence) tuples
    """
    output = output[0]  # Remove batch -> [64512, 7]

    # Extract coordinates and confidence
    x_center = output[:, 0]
    y_center = output[:, 1]
    width = output[:, 2]
    height = output[:, 3]
    confidence = output[:, 4]
    # class1, class2 = output[:, 5], output[:, 6]  # Not used for visualization

    # Filter by confidence
    mask = confidence > conf_threshold
    if not np.any(mask):
        return []

    x_center = x_center[mask]
    y_center = y_center[mask]
    width = width[mask]
    height = height[mask]
    confidence = confidence[mask]

    # Convert to x1, y1, x2, y2
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    # Scale to original image size (mayocream uses 1024x1024 input)
    orig_h, orig_w = orig_size
    scale = max(orig_h, orig_w) / input_size

    x1 = (x1 * scale).astype(int)
    y1 = (y1 * scale).astype(int)
    x2 = (x2 * scale).astype(int)
    y2 = (y2 * scale).astype(int)

    # Clip to image bounds
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    # Apply NMS
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(float)
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        confidence.tolist(),
        conf_threshold,
        nms_threshold
    )

    if len(indices) == 0:
        return []

    indices = indices.flatten()

    results = []
    for i in indices:
        results.append((int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]), float(confidence[i])))

    return results


def draw_boxes(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int, float]],
    color: Tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on image."""
    result = image.copy()

    for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Draw label with confidence
        text = f"{label}: {conf:.2f}"
        font_scale = 0.5
        font_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Background for text
        cv2.rectangle(result, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(result, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return result


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.4,
    threshold: float = 0.5,
) -> np.ndarray:
    """Draw mask overlay on image."""
    result = image.copy()

    # Resize mask to image size if needed
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Create binary mask
    binary_mask = (mask > threshold).astype(np.uint8)

    # Create colored overlay
    overlay = np.zeros_like(image)
    overlay[binary_mask == 1] = color

    # Blend with original
    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)

    # Also draw contours for clarity
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)

    return result


def draw_lines_overlay(
    image: np.ndarray,
    lines: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.3,
    threshold: float = 0.3,
) -> np.ndarray:
    """Draw text lines overlay on image."""
    result = image.copy()

    # lines shape: [2, H, W] - probability and threshold maps
    if lines.ndim == 3 and lines.shape[0] == 2:
        prob_map = lines[0]
    else:
        prob_map = lines

    # Resize to image size
    if prob_map.shape[:2] != image.shape[:2]:
        prob_map = cv2.resize(prob_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Create binary mask
    binary_mask = (prob_map > threshold).astype(np.uint8)

    # Create colored overlay
    overlay = np.zeros_like(image)
    overlay[binary_mask == 1] = color

    # Blend with original
    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)

    return result


def create_comparison_visualization(
    image: np.ndarray,
    animetext_boxes: List[Tuple[int, int, int, int, float]],
    ctd_boxes: List[Tuple[int, int, int, int, float]],
    ctd_mask: Optional[np.ndarray],
    ctd_lines: Optional[np.ndarray],
) -> np.ndarray:
    """Create a comparison visualization with all annotations."""
    h, w = image.shape[:2]

    # 1. AnimeText_yolo visualization (boxes only)
    vis_animetext = image.copy()
    vis_animetext = draw_boxes(vis_animetext, animetext_boxes, COLORS['animetext_box'], "text")
    cv2.putText(vis_animetext, "AnimeText_yolo (deepghs)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis_animetext, f"Boxes: {len(animetext_boxes)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 2. CTD blocks visualization
    vis_ctd_blocks = image.copy()
    vis_ctd_blocks = draw_boxes(vis_ctd_blocks, ctd_boxes, COLORS['ctd_box'], "block")
    cv2.putText(vis_ctd_blocks, "Comic Text Detector (mayocream)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis_ctd_blocks, f"Blocks: {len(ctd_boxes)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['ctd_box'], 2)

    # 3. CTD mask + lines visualization
    vis_ctd_segmentation = image.copy()
    if ctd_mask is not None:
        vis_ctd_segmentation = draw_mask_overlay(vis_ctd_segmentation, ctd_mask, COLORS['ctd_mask'], alpha=0.3)
    if ctd_lines is not None:
        vis_ctd_segmentation = draw_lines_overlay(vis_ctd_segmentation, ctd_lines, COLORS['ctd_lines'], alpha=0.3)
    cv2.putText(vis_ctd_segmentation, "CTD Mask (red) + Lines (cyan)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 4. Original image
    vis_original = image.copy()
    cv2.putText(vis_original, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Create 2x2 grid
    top_row = np.hstack([vis_original, vis_animetext])
    bottom_row = np.hstack([vis_ctd_blocks, vis_ctd_segmentation])
    result = np.vstack([top_row, bottom_row])

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare AnimeText_yolo vs mayocream/comic-text-detector"
    )

    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image-dir", type=str, help="Directory of images")
    parser.add_argument("--output", type=str, default="outputs/comparison", help="Output directory")
    parser.add_argument("--limit", type=int, default=5, help="Max images to process")

    parser.add_argument("--animetext-model", type=str,
                        default="models/animetext-yolo/yolo12s_animetext/model.onnx",
                        help="AnimeText_yolo ONNX model path (auto-downloads if missing)")
    parser.add_argument("--ctd-model", type=str,
                        default="models/comic-text-detector.onnx",
                        help="mayocream/comic-text-detector ONNX model path (auto-downloads if missing)")

    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")

    args = parser.parse_args()

    if not ONNX_AVAILABLE:
        print("Error: ONNX Runtime is required")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up providers
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"]

    # Load models
    print("Loading models...")

    # AnimeText_yolo (640x640 input)
    animetext_path = Path(args.animetext_model)
    if not animetext_path.exists():
        animetext_path = Path(__file__).parent.parent / args.animetext_model

    animetext_path = ensure_animetext_model(animetext_path)
    print(f"  AnimeText_yolo: {animetext_path}")
    animetext_session = ort.InferenceSession(str(animetext_path), providers=providers)
    animetext_input_name = animetext_session.get_inputs()[0].name

    # mayocream/comic-text-detector (1024x1024 input)
    ctd_path = Path(args.ctd_model)
    if not ctd_path.exists():
        ctd_path = Path(__file__).parent.parent / args.ctd_model

    ctd_path = ensure_ctd_model(ctd_path)
    print(f"  Comic Text Detector: {ctd_path}")
    ctd_session = ort.InferenceSession(str(ctd_path), providers=providers)
    ctd_input_name = ctd_session.get_inputs()[0].name
    ctd_outputs = [o.name for o in ctd_session.get_outputs()]
    print(f"    Outputs: {ctd_outputs}")

    # Collect images
    image_paths = []
    if args.image:
        image_paths = [Path(args.image)]
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            image_paths.extend(img_dir.glob(ext))
        image_paths = image_paths[:args.limit]
    else:
        # Use default test images
        test_dir = Path("/home/danny/Documents/personal/extension/backend/training/datasets/manga-bubble-detection-evening/test/images")
        if test_dir.exists():
            image_paths = list(test_dir.glob("*.jpg"))[:args.limit]
        else:
            print("Error: No images specified. Use --image or --image-dir")
            sys.exit(1)

    if not image_paths:
        print("Error: No images found")
        sys.exit(1)

    print(f"\nProcessing {len(image_paths)} images...")
    print(f"Comparing:")
    print(f"  - deepghs/AnimeText_yolo (YOLO12, 640x640)")
    print(f"  - mayocream/comic-text-detector (3 outputs, 1024x1024)")

    for img_path in image_paths:
        print(f"\n  Processing: {img_path.name}")

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"    Error: Could not read image")
            continue

        orig_h, orig_w = image.shape[:2]

        # 1. Run AnimeText_yolo (640x640)
        animetext_input, orig_size, scale = preprocess_image(image, 640)
        animetext_output = animetext_session.run(None, {animetext_input_name: animetext_input})
        animetext_boxes = postprocess_yolo_boxes(
            animetext_output[0], (orig_h, orig_w), 640, args.conf_threshold
        )
        print(f"    AnimeText_yolo: {len(animetext_boxes)} boxes")

        # 2. Run mayocream/comic-text-detector (1024x1024)
        ctd_input, _, _ = preprocess_image(image, 1024)
        ctd_outputs_raw = ctd_session.run(None, {ctd_input_name: ctd_input})

        # mayocream model has 3 outputs: blk [1,64512,7], seg [1,1,1024,1024], det [1,2,1024,1024]
        ctd_blocks = []
        ctd_mask = None
        ctd_lines = None

        if len(ctd_outputs_raw) >= 3:
            # blk, seg, det
            ctd_blocks = postprocess_mayocream_blocks(
                ctd_outputs_raw[0], (orig_h, orig_w), 1024, args.conf_threshold
            )
            ctd_mask = ctd_outputs_raw[1][0, 0]  # [1, 1, H, W] -> [H, W]
            ctd_lines = ctd_outputs_raw[2][0]    # [1, 2, H, W] -> [2, H, W]
            print(f"    CTD: {len(ctd_blocks)} blocks, mask, lines")
        elif len(ctd_outputs_raw) == 2:
            # Fallback: mask, lines only (shouldn't happen with mayocream model)
            ctd_mask = ctd_outputs_raw[0][0, 0]
            ctd_lines = ctd_outputs_raw[1][0]
            print(f"    CTD: mask, lines (no blocks)")

        # Resize mask/lines to original image size
        if ctd_mask is not None:
            ctd_mask = cv2.resize(ctd_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        if ctd_lines is not None:
            ctd_lines_resized = np.zeros((2, orig_h, orig_w), dtype=ctd_lines.dtype)
            for i in range(2):
                ctd_lines_resized[i] = cv2.resize(ctd_lines[i], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            ctd_lines = ctd_lines_resized

        # Create comparison visualization
        comparison = create_comparison_visualization(
            image,
            animetext_boxes,
            ctd_blocks,
            ctd_mask,
            ctd_lines,
        )

        # Save result
        output_path = output_dir / f"{img_path.stem}_comparison.jpg"
        cv2.imwrite(str(output_path), comparison)
        print(f"    Saved: {output_path}")

        # Also save individual visualizations
        # AnimeText only
        vis_animetext = draw_boxes(image.copy(), animetext_boxes, COLORS['animetext_box'], "text")
        cv2.imwrite(str(output_dir / f"{img_path.stem}_animetext.jpg"), vis_animetext)

        # CTD with all annotations
        vis_ctd = image.copy()
        if ctd_mask is not None:
            vis_ctd = draw_mask_overlay(vis_ctd, ctd_mask, COLORS['ctd_mask'], alpha=0.3)
        if ctd_lines is not None:
            vis_ctd = draw_lines_overlay(vis_ctd, ctd_lines, COLORS['ctd_lines'], alpha=0.3)
        if ctd_blocks:
            vis_ctd = draw_boxes(vis_ctd, ctd_blocks, COLORS['ctd_box'], "block")
        cv2.imwrite(str(output_dir / f"{img_path.stem}_ctd.jpg"), vis_ctd)

    print(f"\nDone! Results saved to: {output_dir}")
    print(f"\nLegend:")
    print(f"  - AnimeText_yolo boxes: GREEN")
    print(f"  - CTD blocks: BLUE")
    print(f"  - CTD mask: RED overlay")
    print(f"  - CTD lines: CYAN overlay")


if __name__ == "__main__":
    main()
