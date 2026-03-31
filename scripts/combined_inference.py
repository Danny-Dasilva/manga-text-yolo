#!/usr/bin/env python3
"""
Combined inference using all 3 trained model phases.

Loads weights from:
- Phase 1 (Segmentation): seg_net
- Phase 2 (Detection): dbnet
- Phase 3 (Block): block_det

Produces 3 outputs matching original comic-text-detector format:
- blk: Block/bubble detections
- seg: Segmentation mask
- det: Text line detection (DBNet shrink/threshold maps)

Usage:
    python scripts/combined_inference.py --image test.jpg --output outputs/
    python scripts/combined_inference.py --image-dir data/merged_val/images --limit 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import torch.nn as nn


@dataclass
class CombinedResult:
    """Container for combined model outputs."""
    blocks: List[Dict[str, Any]]  # Block detections
    mask: np.ndarray              # Segmentation mask [H, W]
    lines_map: np.ndarray         # DBNet output [2, H, W]
    original_size: Tuple[int, int]


def normalize_state_dict(state_dict: dict) -> dict:
    """Remove _orig_mod. prefix from torch.compile wrapped models."""
    return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}


def load_combined_model(
    seg_checkpoint: str,
    det_checkpoint: str,
    block_checkpoint: str,
    device: str = 'cuda'
) -> nn.Module:
    """
    Load model with weights combined from all 3 training phases.

    Creates a TextDetectorInference with:
    - Fresh UnetHead loaded with Phase 1 weights (full, not modified)
    - Fresh DBHead loaded with Phase 2 weights
    - Fresh BlockDetector loaded with Phase 3 weights

    Args:
        seg_checkpoint: Path to Phase 1 (segmentation) checkpoint
        det_checkpoint: Path to Phase 2 (detection/DBNet) checkpoint
        block_checkpoint: Path to Phase 3 (block detection) checkpoint
        device: Device to load model on

    Returns:
        TextDetectorInference model ready for inference
    """
    from src.models.backbone import create_backbone
    from src.models.heads import UnetHead, DBHead, BlockDetector, TextDetectorInference

    # Load checkpoints
    print(f"Loading Phase 1 (Seg): {seg_checkpoint}")
    seg_ckpt = torch.load(seg_checkpoint, map_location='cpu', weights_only=False)
    seg_state = normalize_state_dict(seg_ckpt.get('model_state_dict', seg_ckpt))

    print(f"Loading Phase 2 (Det): {det_checkpoint}")
    det_ckpt = torch.load(det_checkpoint, map_location='cpu', weights_only=False)
    det_state = normalize_state_dict(det_ckpt.get('model_state_dict', det_ckpt))

    print(f"Loading Phase 3 (Block): {block_checkpoint}")
    block_ckpt = torch.load(block_checkpoint, map_location='cpu', weights_only=False)
    block_state = normalize_state_dict(block_ckpt.get('model_state_dict', block_ckpt))

    # Get backbone name from config
    config = block_ckpt.get('config', {})
    backbone_name = config.get('backbone_name', 'yolo26s.pt')
    print(f"Backbone: {backbone_name}")

    # Create backbone
    backbone = create_backbone(
        model_name=backbone_name,
        pretrained=True,
        freeze=True,
        simple=True
    )

    # Load backbone weights from any checkpoint (they should be the same)
    backbone_state = {k.replace('backbone.', ''): v for k, v in seg_state.items() if k.startswith('backbone.')}
    if backbone_state:
        backbone.load_state_dict(backbone_state, strict=False)
        print(f"  Loaded backbone: {len(backbone_state)} keys")

    # Create fresh UnetHead with ALL layers (don't use initialize_db which deletes layers)
    seg_head = UnetHead(use_cib=True)
    seg_head_state = {k.replace('seg_net.', ''): v for k, v in seg_state.items() if k.startswith('seg_net.')}
    seg_head.load_state_dict(seg_head_state, strict=False)
    print(f"  Loaded seg_head: {len(seg_head_state)} keys")

    # Create fresh DBHead
    db_head = DBHead(64, use_cib=True)
    db_head_state = {k.replace('dbnet.', ''): v for k, v in det_state.items() if k.startswith('dbnet.')}
    db_head.load_state_dict(db_head_state, strict=False)
    print(f"  Loaded db_head: {len(db_head_state)} keys")

    # Create fresh BlockDetector
    block_det = BlockDetector(nc=1, ch=(128, 256, 512), anchors_per_scale=3)
    block_det_state = {k.replace('block_det.', ''): v for k, v in block_state.items() if k.startswith('block_det.')}
    block_det.load_state_dict(block_det_state, strict=False)
    print(f"  Loaded block_det: {len(block_det_state)} keys")

    # Combine into inference model
    model = TextDetectorInference(
        backbone=backbone,
        seg_head=seg_head,
        db_head=db_head,
        block_detector=block_det
    )

    # Set to eval mode
    model = model.to(device).eval()
    model.block_det.training_mode = False  # NMS-free inference

    return model


class CombinedDetector:
    """Combined detector that produces all 3 outputs."""

    def __init__(
        self,
        seg_checkpoint: str,
        det_checkpoint: str,
        block_checkpoint: str,
        input_size: int = 1024,
        device: str = 'cuda',
        conf_threshold: float = 0.1,
        mask_threshold: float = 0.3,
    ):
        self.input_size = input_size
        self.device = device
        self.conf_threshold = conf_threshold
        self.mask_threshold = mask_threshold

        self.model = load_combined_model(
            seg_checkpoint, det_checkpoint, block_checkpoint, device
        )

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, dict]:
        """Preprocess image with letterbox padding."""
        h, w = image.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        tensor = padded.astype(np.float32) / 255.0
        tensor = tensor[:, :, ::-1]  # BGR to RGB
        tensor = tensor.transpose(2, 0, 1)
        tensor = torch.from_numpy(tensor.copy()).unsqueeze(0).to(self.device)

        meta = {
            'original_size': (h, w),
            'scale': scale,
            'pad': (pad_h, pad_w),
            'new_size': (new_h, new_w),
        }
        return tensor, meta

    def __call__(self, image: np.ndarray) -> CombinedResult:
        """Run combined inference."""
        tensor, meta = self.preprocess(image)
        h, w = meta['original_size']
        scale = meta['scale']
        pad_h, pad_w = meta['pad']
        new_h, new_w = meta['new_size']

        with torch.no_grad():
            # TextDetectorInference handles all three outputs in one forward pass
            blocks_out, mask_out, lines_out = self.model(tensor)

        # Post-process mask
        mask = mask_out[0, 0].cpu().numpy()
        mask = mask[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_binary = (mask > self.mask_threshold).astype(np.uint8) * 255

        # Post-process lines (DBNet output: [shrink_maps, threshold_maps])
        lines = lines_out[0].cpu().numpy()
        lines = lines[:, pad_h:pad_h + new_h, pad_w:pad_w + new_w]
        lines_resized = np.zeros((2, h, w), dtype=np.float32)
        for i in range(2):
            lines_resized[i] = cv2.resize(lines[i], (w, h), interpolation=cv2.INTER_LINEAR)

        # Post-process blocks
        blocks_np = blocks_out[0].cpu().numpy()
        blocks = self._extract_blocks(blocks_np, meta)

        return CombinedResult(
            blocks=blocks,
            mask=mask_binary,
            lines_map=lines_resized,
            original_size=(h, w)
        )

    def _extract_blocks(self, preds: np.ndarray, meta: dict) -> List[Dict[str, Any]]:
        """Extract text blocks from YOLO predictions."""
        h, w = meta['original_size']
        scale = meta['scale']
        pad_h, pad_w = meta['pad']

        blocks = []
        for pred in preds:
            cx, cy, bw, bh = pred[0:4]
            obj_conf = float(pred[4])
            cls_conf = float(pred[5]) if len(pred) > 5 else 1.0
            conf = obj_conf * cls_conf

            if conf < self.conf_threshold:
                continue

            # Convert from normalized to pixels
            cx_px = cx * self.input_size
            cy_px = cy * self.input_size
            w_px = bw * self.input_size
            h_px = bh * self.input_size

            x1 = cx_px - w_px / 2 - pad_w
            y1 = cy_px - h_px / 2 - pad_h
            x2 = cx_px + w_px / 2 - pad_w
            y2 = cy_px + h_px / 2 - pad_h

            # Scale to original size
            x1, x2 = x1 / scale, x2 / scale
            y1, y2 = y1 / scale, y2 / scale

            # Clip
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if (x2 - x1) > 5 and (y2 - y1) > 5:
                blocks.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'area': (x2 - x1) * (y2 - y1)
                })

        blocks.sort(key=lambda x: x['confidence'], reverse=True)
        return blocks


def visualize_result(image: np.ndarray, result: CombinedResult, output_path: str):
    """Visualize all 3 outputs."""
    h, w = image.shape[:2]

    # Create visualization grid
    vis_blocks = image.copy()
    vis_mask = image.copy()
    vis_lines = image.copy()

    # Draw blocks (green)
    for block in result.blocks:
        x1, y1, x2, y2 = block['bbox']
        cv2.rectangle(vis_blocks, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_blocks, f"{block['confidence']:.2f}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Overlay mask (red)
    mask_color = np.zeros_like(vis_mask)
    mask_color[:, :, 2] = result.mask
    vis_mask = cv2.addWeighted(vis_mask, 0.7, mask_color, 0.3, 0)

    # Overlay lines shrink map (blue)
    shrink_map = (result.lines_map[0] * 255).astype(np.uint8)
    lines_color = np.zeros_like(vis_lines)
    lines_color[:, :, 0] = shrink_map
    vis_lines = cv2.addWeighted(vis_lines, 0.7, lines_color, 0.3, 0)

    # Stack horizontally
    combined = np.hstack([vis_blocks, vis_mask, vis_lines])

    # Add labels
    cv2.putText(combined, "Blocks", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Seg Mask", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Text Lines", (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite(output_path, combined)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Combined 3-output inference')
    parser.add_argument('--seg-checkpoint', default='runs/segmentation_yolo26/best.pt')
    parser.add_argument('--det-checkpoint', default='runs/detection_yolo26/best.pt')
    parser.add_argument('--block-checkpoint', default='runs/block_yolo26/best.pt')
    parser.add_argument('--image', help='Single image path')
    parser.add_argument('--image-dir', default='data/merged_val/images')
    parser.add_argument('--output', default='outputs/combined')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--conf-threshold', type=float, default=0.05,
                        help='Block detection confidence threshold (obj × cls)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load combined model
    print("=" * 60)
    print("Loading combined model from 3 checkpoints...")
    print("=" * 60)

    detector = CombinedDetector(
        seg_checkpoint=args.seg_checkpoint,
        det_checkpoint=args.det_checkpoint,
        block_checkpoint=args.block_checkpoint,
        device=args.device,
        conf_threshold=args.conf_threshold,
    )

    # Get images
    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_dir = Path(args.image_dir)
        image_paths = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))
        image_paths = image_paths[:args.limit]

    print(f"\nProcessing {len(image_paths)} images...")
    print("=" * 60)

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        result = detector(img)

        print(f"\n{img_path.name}:")
        print(f"  Blocks: {len(result.blocks)}")
        print(f"  Mask: {result.mask.shape}, non-zero={np.count_nonzero(result.mask)}")
        print(f"  Lines: {result.lines_map.shape}, max={result.lines_map.max():.3f}")

        vis_path = output_dir / f"combined_{img_path.stem}.jpg"
        visualize_result(img, result, str(vis_path))

    print(f"\nDone! Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
