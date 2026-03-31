"""Fast annotation generation using PyTorch GPU."""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Tuple, Dict


class FastTextDetector:
    """Fast PyTorch GPU-based text detector without mask refinement."""

    def __init__(self, model_path: str, input_size: int = 1024, device: str = 'cuda',
                 conf_threshold: float = 0.4, mask_threshold: float = 0.3):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.mask_threshold = mask_threshold
        self.device = device

        print(f"Loading ONNX model and converting to PyTorch...")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Convert ONNX to PyTorch
        from onnx2torch import convert
        self.model = convert(model_path).to(device).eval()
        print(f"Model loaded on {device}")

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Dict]:
        """Preprocess image with letterbox padding."""
        h, w = image.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Convert to tensor format
        tensor = padded.astype(np.float32) / 255.0
        tensor = tensor[:, :, ::-1].copy()  # BGR to RGB
        tensor = torch.from_numpy(tensor).permute(2, 0, 1)  # HWC to CHW

        meta = {
            'original_size': (h, w),
            'scale': scale,
            'pad': (pad_h, pad_w),
            'new_size': (new_h, new_w),
        }
        return tensor, meta

    def postprocess(self, seg_out: np.ndarray, det_out: np.ndarray, meta: Dict) -> Dict:
        """Postprocess model outputs."""
        h, w = meta['original_size']
        pad_h, pad_w = meta['pad']
        new_h, new_w = meta['new_size']

        # Extract mask from seg output (shape: 1, 1, H, W)
        mask = seg_out[0, 0]  # Remove batch and channel dims -> (H, W)

        # Remove padding
        mask = mask[pad_h:pad_h + new_h, pad_w:pad_w + new_w]

        # Resize to original
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # Binarize
        binary_mask = ((mask > self.mask_threshold) * 255).astype(np.uint8)

        # Extract text blocks from det output
        det = det_out[0]  # Remove batch dim -> (2, H, W)
        shrink_map = det[0, pad_h:pad_h + new_h, pad_w:pad_w + new_w]
        shrink_map = cv2.resize(shrink_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Find contours from shrink map
        binary_det = ((shrink_map > self.mask_threshold) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_det, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        text_blocks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)

            # Calculate confidence
            mask_roi = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask_roi, [contour], -1, 255, -1)
            conf = float(shrink_map[mask_roi > 0].mean()) if (mask_roi > 0).any() else 0

            if conf < self.conf_threshold:
                continue

            text_blocks.append({
                'bbox': [x, y, x + bw, y + bh],
                'confidence': conf,
                'area': area,
            })

        return {
            'mask': binary_mask,
            'text_blocks': text_blocks,
            'original_size': (h, w),
        }

    @torch.no_grad()
    def __call__(self, images: List[np.ndarray]) -> List[Dict]:
        """Process images one at a time on GPU (model has fixed batch=1)."""
        results = []
        for img in images:
            tensor, meta = self.preprocess(img)
            # Add batch dim and move to GPU
            batch = tensor.unsqueeze(0).to(self.device)

            # Run inference
            outputs = self.model(batch)
            # outputs: (blk, seg, det)
            blk_out, seg_out, det_out = outputs

            # Move to CPU for postprocessing
            seg_np = seg_out.cpu().numpy()
            det_np = det_out.cpu().numpy()

            result = self.postprocess(seg_np, det_np, meta)
            results.append(result)

        return results


def generate_annotations(
    model_path: str,
    img_dirs: List[str],
    output_dir: str,
    batch_size: int = 8,
):
    """Generate pseudo-labels using PyTorch GPU."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = FastTextDetector(model_path=model_path, input_size=1024, device=device)

    # Collect all images
    image_paths = []
    for img_dir in img_dirs:
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            image_paths.extend(Path(img_dir).glob(ext))

    print(f"Processing {len(image_paths)} images...")

    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [cv2.imread(str(p)) for p in batch_paths]

        results = detector(batch_images)

        for path, result in zip(batch_paths, results):
            name = path.stem

            # Save mask
            cv2.imwrite(str(output_dir / f"mask-{name}.png"), result['mask'])

            # Save YOLO format labels
            h, w = result['original_size']
            yolo_lines = []
            for blk in result['text_blocks']:
                x1, y1, x2, y2 = blk['bbox']
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                yolo_lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            with open(output_dir / f"{name}.txt", 'w') as f:
                f.write('\n'.join(yolo_lines))

            # Save JSON
            with open(output_dir / f"{name}.json", 'w') as f:
                json.dump({
                    'text_blocks': result['text_blocks'],
                    'original_size': result['original_size'],
                }, f)


if __name__ == '__main__':
    IMG_DIRS = [
        "/home/danny/Documents/personal/extension/backend/training/datasets/manga-bubble/train/images",
        "/home/danny/Documents/personal/extension/backend/training/datasets/manga-text-druyb/train/images",
        "/home/danny/Documents/personal/extension/backend/training/datasets/manga-bubble-detect/train/images",
        "/home/danny/Documents/personal/extension/backend/training/datasets/manga-bubble-pqdou/train/images",
        "/home/danny/Documents/personal/extension/backend/training/datasets/manga-bubble-detection-evening/train/images",
    ]

    generate_annotations(
        model_path="data/comic-text-detector.onnx",
        img_dirs=IMG_DIRS,
        output_dir="data/generated_annotations",
        batch_size=8,
    )
