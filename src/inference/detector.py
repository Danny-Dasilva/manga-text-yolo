"""
Inference Pipeline for Comic Text Detection

Features:
- Batch inference support
- Automatic preprocessing and postprocessing
- Combined block + line detection with group_output
- ONNX and TensorRT support
- FP16/INT8 quantization
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

from src.utils.textblock import TextBlock, group_output
from torchvision.ops import nms as _torchvision_nms


@dataclass
class DetectionResult:
    """Container for detection results."""
    mask: np.ndarray  # Binary segmentation mask
    lines_map: np.ndarray  # Text line detection map
    text_blocks: List[Dict[str, Any]]  # Detected text blocks with polygons
    original_size: Tuple[int, int]  # Original image size (H, W)


class TextDetector:
    """
    High-level inference interface for comic text detection.

    Supports batch processing, automatic scaling, and multiple backends.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        input_size: int = 1024,
        device: str = 'cuda',
        half: bool = False,
        backend: str = 'pytorch',  # 'pytorch', 'onnx', 'tensorrt'
        conf_threshold: float = 0.4,
        mask_threshold: float = 0.3,
    ):
        """
        Initialize text detector.

        Args:
            model_path: Path to model checkpoint
            input_size: Model input size (square)
            device: Device for inference
            half: Use FP16 inference
            backend: Inference backend
            conf_threshold: Confidence threshold for detections
            mask_threshold: Threshold for binary mask
        """
        self.input_size = input_size
        self.device = device
        self.half = half
        self.backend = backend
        self.conf_threshold = conf_threshold
        self.mask_threshold = mask_threshold

        # Load model based on backend
        if backend == 'pytorch':
            self.model = self._load_pytorch(model_path)
        elif backend == 'onnx':
            self.model = self._load_onnx(model_path)
        elif backend == 'tensorrt':
            self.model = self._load_tensorrt(model_path)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _load_pytorch(self, model_path: Union[str, Path]) -> nn.Module:
        """Load PyTorch model with block detection support."""
        from src.models.detector import create_text_detector
        from src.models.heads import TEXTDET_BLOCK

        # Load checkpoint to get config
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})

        # IMPORTANT: Use pretrained_backbone=True because EMA shadow doesn't include
        # frozen backbone weights (only trainable parameters are tracked by EMA).
        # The pretrained backbone provides the frozen feature extractor weights.
        model = create_text_detector(
            backbone_name=config.get('backbone_name', 'yolo11s.pt'),
            pretrained_backbone=True,  # Load pretrained backbone (EMA doesn't include frozen weights)
            freeze_backbone=True,
            use_cib=True,
            use_bifpn=config.get('use_bifpn', False),
            use_rank_guided=config.get('use_rank_guided', False),
            device=self.device
        )

        # Initialize block detector if checkpoint was trained with it
        training_phase = config.get('training_phase', 1)
        if training_phase >= 3 and hasattr(model, 'initialize_block_detector'):
            model.initialize_block_detector()

        # Load weights - prefer EMA for inference (better smoothed weights)
        # Note: EMA only contains trainable parameters, backbone weights come from pretrained
        # Strip _orig_mod. prefix from keys (added by torch.compile during training)
        def strip_compile_prefix(state_dict):
            """Remove _orig_mod. prefix from torch.compile wrapped model keys."""
            return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        if 'ema_shadow' in checkpoint and checkpoint['ema_shadow']:
            ema_weights = strip_compile_prefix(checkpoint['ema_shadow'])
            model.load_state_dict(ema_weights, strict=False)
        elif 'model_state_dict' in checkpoint:
            model_weights = strip_compile_prefix(checkpoint['model_state_dict'])
            model.load_state_dict(model_weights, strict=False)

        # Move entire model to correct device (important for block_det which is initialized after creation)
        model = model.to(self.device)

        # Set to block detection mode for inference
        model.forward_mode = TEXTDET_BLOCK
        if hasattr(model, 'block_det') and model.block_det is not None:
            model.block_det.training_mode = False  # Use o2o head (NMS-free)

        model.eval()
        if self.half:
            model.half()

        self._has_block_detector = hasattr(model, 'block_det') and model.block_det is not None
        return model

    def _load_onnx(self, model_path: Union[str, Path]) -> Any:
        """Load ONNX model using ONNX Runtime with robust provider selection."""
        import onnxruntime as ort

        # Dynamic provider selection (matching backend pattern)
        available = ort.get_available_providers()
        if 'cuda' in self.device:
            preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            preferred = ["CPUExecutionProvider"]
        providers = [p for p in preferred if p in available]
        providers = providers if providers else available

        try:
            session = ort.InferenceSession(str(model_path), providers=providers)
            actual_provider = session.get_providers()[0]
            print(f"ONNX model loaded with provider: {actual_provider}")
            return session
        except Exception as exc:
            if "CUDAExecutionProvider" in providers:
                print(f"CUDA init failed ({exc}). Falling back to CPU.")
                return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            raise

    def _load_tensorrt(self, model_path: Union[str, Path]) -> Any:
        """
        Load TensorRT engine with automatic caching.

        Uses the TensorRTEngine class from tensorrt_backend.py which handles:
        - Engine caching based on model hash, GPU, and TensorRT version
        - Automatic fallback to ONNX Runtime if TensorRT is unavailable
        - Dynamic input shape support
        """
        from src.inference.tensorrt_backend import TensorRTEngine, TensorRTEngineConfig

        # Configure engine for the model
        config = TensorRTEngineConfig(
            fp16=self.half,
            int8=False,  # INT8 requires calibration
            min_batch=1,
            opt_batch=1,
            max_batch=8,
            min_resolution=512,
            opt_resolution=self.input_size,
            max_resolution=2048,
        )

        # Create engine with caching
        engine = TensorRTEngine(
            onnx_path=str(model_path),
            cache_dir='.trt_cache',
            config=config,
            fallback_to_onnx=True,  # Fall back to ONNX if TensorRT fails
        )

        print(f"TensorRT engine loaded (using_fallback={engine.using_fallback})")
        return engine

    def __call__(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        batch_size: int = 4,
    ) -> Union[DetectionResult, List[DetectionResult]]:
        """
        Run detection on images.

        Args:
            images: Single image (H, W, 3) BGR or list of images
            batch_size: Batch size for processing

        Returns:
            DetectionResult or list of DetectionResults
        """
        # Handle single image
        single_image = isinstance(images, np.ndarray) and images.ndim == 3
        if single_image:
            images = [images]

        # Process in batches
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)

        return results[0] if single_image else results

    def _process_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """Process a batch of images."""
        # Preprocess
        batch_tensors, metas = self._preprocess_batch(images)

        # Inference
        blocks_out = None
        if self.backend == 'pytorch':
            with torch.no_grad():
                if self.half:
                    batch_tensors = batch_tensors.half()

                # Check if model has block detector (new architecture)
                if getattr(self, '_has_block_detector', False):
                    # Block detection mode - returns YOLO predictions
                    blocks_out = self.model(batch_tensors)
                    blocks_out = blocks_out.cpu().numpy()
                    # No mask/lines in block-only mode
                    mask_out = np.zeros((len(images), 1, self.input_size, self.input_size), dtype=np.float32)
                    lines_out = np.zeros((len(images), 2, self.input_size, self.input_size), dtype=np.float32)
                else:
                    # Legacy mode - mask and lines
                    mask_out, lines_out = self.model(batch_tensors)
                    mask_out = mask_out.cpu().numpy()
                    lines_out = lines_out.cpu().numpy()
        elif self.backend == 'tensorrt':
            # TensorRTEngine accepts torch tensors directly
            if 'cuda' in self.device:
                batch_tensors = batch_tensors.to(self.device)
            if self.half:
                batch_tensors = batch_tensors.half()
            outputs = self.model(batch_tensors)
            # Outputs may be tensors or numpy arrays depending on engine state
            if isinstance(outputs, (list, tuple)):
                if len(outputs) >= 2:
                    mask_out = outputs[0].cpu().numpy() if hasattr(outputs[0], 'cpu') else outputs[0]
                    lines_out = outputs[1].cpu().numpy() if hasattr(outputs[1], 'cpu') else outputs[1]
                else:
                    # Single output, assume it contains both mask and lines
                    mask_out = outputs[0].cpu().numpy() if hasattr(outputs[0], 'cpu') else outputs[0]
                    lines_out = mask_out  # Fallback
            else:
                mask_out = outputs.cpu().numpy() if hasattr(outputs, 'cpu') else outputs
                lines_out = mask_out
        else:  # ONNX
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: batch_tensors.numpy()})
            mask_out, lines_out = outputs[0], outputs[1]

        # Postprocess
        results = []
        for i, (img, meta) in enumerate(zip(images, metas)):
            result = self._postprocess(
                mask_out[i],
                lines_out[i],
                meta,
                blocks_out[i] if blocks_out is not None else None,
            )
            results.append(result)

        return results

    def _preprocess_batch(
        self,
        images: List[np.ndarray]
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Preprocess batch of images.

        Returns:
            Tuple of (batch_tensor, metadata_list)
        """
        batch = []
        metas = []

        for img in images:
            tensor, meta = self._preprocess_single(img)
            batch.append(tensor)
            metas.append(meta)

        batch_tensor = torch.stack(batch)
        if self.backend == 'pytorch' and 'cuda' in self.device:
            batch_tensor = batch_tensor.to(self.device)

        return batch_tensor, metas

    def _preprocess_single(
        self,
        image: np.ndarray
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess single image.

        Uses direct resize for block detector mode (matches training augmentation),
        or letterbox padding for legacy seg/det mode.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            Tuple of (tensor, metadata)
        """
        h, w = image.shape[:2]
        target_size = self.input_size

        if getattr(self, '_has_block_detector', False):
            # Block detector: direct resize to target_size x target_size
            # (matches albumentations Resize + ToTensorV2 used in training)
            # Training converts BGR->RGB at dataset.py:273 before augmentation
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

            tensor = resized.astype(np.float32) / 255.0
            tensor = tensor.transpose(2, 0, 1)  # HWC to CHW (RGB)
            tensor = torch.from_numpy(tensor.copy())

            meta = {
                'original_size': (h, w),
                'scale_x': target_size / w,
                'scale_y': target_size / h,
                'scale': min(target_size / h, target_size / w),  # kept for compat
                'pad': (0, 0),
                'new_size': (target_size, target_size),
                'direct_resize': True,
            }
        else:
            # Legacy seg/det mode: letterbox with center padding
            scale = min(target_size / h, target_size / w)
            new_h, new_w = int(h * scale), int(w * scale)

            pad_h = (target_size - new_h) // 2
            pad_w = (target_size - new_w) // 2

            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
            padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

            tensor = padded.astype(np.float32) / 255.0
            tensor = tensor[:, :, ::-1]  # BGR to RGB
            tensor = tensor.transpose(2, 0, 1)  # HWC to CHW
            tensor = torch.from_numpy(tensor.copy())

            meta = {
                'original_size': (h, w),
                'scale': scale,
                'pad': (pad_h, pad_w),
                'new_size': (new_h, new_w),
                'direct_resize': False,
            }

        return tensor, meta

    def _postprocess(
        self,
        mask: np.ndarray,
        lines: np.ndarray,
        meta: Dict,
        blocks: Optional[np.ndarray] = None,
    ) -> DetectionResult:
        """
        Postprocess model outputs.

        Args:
            mask: Segmentation mask (1, H, W)
            lines: Text line map (2, H, W) - shrink and threshold
            meta: Preprocessing metadata
            blocks: Optional YOLO block detections [N, 6] with normalized xywh + conf + cls

        Returns:
            DetectionResult
        """
        h, w = meta['original_size']
        scale = meta['scale']
        pad_h, pad_w = meta['pad']
        new_h, new_w = meta['new_size']

        # Handle block detections if available (new architecture)
        if blocks is not None:
            text_blocks = self._extract_blocks_from_yolo(blocks, meta)
            # Return with empty mask/lines for block-only mode
            return DetectionResult(
                mask=np.zeros((h, w), dtype=np.uint8),
                lines_map=np.zeros((2, h, w), dtype=np.float32),
                text_blocks=text_blocks,
                original_size=(h, w),
            )

        # Legacy path: extract from line map
        # Remove padding
        mask = mask[0, pad_h:pad_h + new_h, pad_w:pad_w + new_w]
        lines = lines[:, pad_h:pad_h + new_h, pad_w:pad_w + new_w]

        # Resize to original size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        lines_resized = np.zeros((2, h, w), dtype=np.float32)
        for i in range(2):
            lines_resized[i] = cv2.resize(lines[i], (w, h), interpolation=cv2.INTER_LINEAR)

        # Binarize mask
        binary_mask = (mask > self.mask_threshold).astype(np.uint8) * 255

        # Extract text blocks from line map
        text_blocks = self._extract_text_blocks(lines_resized, h, w)

        return DetectionResult(
            mask=binary_mask,
            lines_map=lines_resized,
            text_blocks=text_blocks,
            original_size=(h, w),
        )

    def _extract_text_blocks(
        self,
        lines_map: np.ndarray,
        h: int,
        w: int
    ) -> List[Dict[str, Any]]:
        """Extract text blocks from line detection map."""
        shrink_map = lines_map[0]
        threshold_map = lines_map[1]

        # Binarize using DB approach
        binary = (shrink_map > self.mask_threshold).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        text_blocks = []
        for contour in contours:
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < 100:
                continue

            # Get bounding polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)

            # Get bounding box
            x, y, bw, bh = cv2.boundingRect(contour)

            # Calculate average confidence
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            conf = float(shrink_map[mask > 0].mean())

            if conf < self.conf_threshold:
                continue

            text_blocks.append({
                'bbox': [x, y, x + bw, y + bh],
                'polygon': polygon.reshape(-1, 2).tolist(),
                'confidence': conf,
                'area': area,
            })

        # Sort by area (largest first)
        text_blocks.sort(key=lambda x: x['area'], reverse=True)

        return text_blocks

    def _extract_blocks_from_yolo(
        self,
        preds: np.ndarray,
        meta: Dict
    ) -> List[Dict[str, Any]]:
        """
        Extract text blocks from YOLO BlockDetector output.

        Args:
            preds: YOLO predictions [N, 6] with normalized xywh + conf + cls
            meta: Preprocessing metadata with scale and padding info

        Returns:
            List of text block dicts with bbox, polygon, confidence
        """
        h, w = meta['original_size']
        direct_resize = meta.get('direct_resize', False)

        text_blocks = []

        for pred in preds:
            # Extract prediction components
            cx, cy, bw, bh = pred[0:4]  # Normalized center x, y, width, height
            obj_conf = float(pred[4])
            # For merged single-class head (5 outputs) or legacy 6-output: use obj alone
            # For multi-class (>6 outputs): use obj * max(cls)
            if pred.shape[-1] <= 6:
                conf = obj_conf
            else:
                cls_conf = float(pred[5:].max())
                conf = obj_conf * cls_conf

            # Filter by confidence
            if conf < self.conf_threshold:
                continue

            if direct_resize:
                # Direct resize: normalized coords map directly to original image
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
            else:
                # Letterbox: convert through input_size -> remove padding -> scale
                input_size = self.input_size
                scale = meta['scale']
                pad_h, pad_w = meta['pad']

                cx_px = cx * input_size
                cy_px = cy * input_size
                w_px = bw * input_size
                h_px = bh * input_size

                x1 = int((cx_px - w_px / 2 - pad_w) / scale)
                y1 = int((cy_px - h_px / 2 - pad_h) / scale)
                x2 = int((cx_px + w_px / 2 - pad_w) / scale)
                y2 = int((cy_px + h_px / 2 - pad_h) / scale)

            # Clip to image bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            # Skip invalid/tiny boxes
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w < 10 or box_h < 10:
                continue
            area = box_w * box_h
            if area < 200:
                continue
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            text_blocks.append({
                'bbox': [x1, y1, x2, y2],
                'polygon': polygon,
                'confidence': conf,
                'area': area,
            })

        # Apply NMS to remove overlapping detections
        # Critical for unconverged o2o head where adjacent anchors fire
        if text_blocks:
            boxes_t = torch.tensor(
                [b['bbox'] for b in text_blocks], dtype=torch.float32
            )
            scores_t = torch.tensor(
                [b['confidence'] for b in text_blocks], dtype=torch.float32
            )
            keep = _torchvision_nms(boxes_t, scores_t, iou_threshold=0.4)
            text_blocks = [text_blocks[i] for i in keep.tolist()]

        # Sort by confidence (highest first) and cap max detections
        text_blocks.sort(key=lambda x: x['confidence'], reverse=True)
        text_blocks = text_blocks[:100]

        return text_blocks

    def _extract_line_polygons(
        self,
        lines_map: np.ndarray,
        min_area: int = 100
    ) -> List[np.ndarray]:
        """
        Extract line polygons from DBNet shrink map.

        Args:
            lines_map: DBNet output (2, H, W) with shrink and threshold maps
            min_area: Minimum contour area to include

        Returns:
            List of line polygons, each as numpy array of shape (4, 2)
        """
        shrink_map = lines_map[0]
        binary = (shrink_map > self.mask_threshold).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        lines = []
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            # Fit minimum area rectangle -> 4 corner points
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            lines.append(box.astype(np.int32))

        return lines


class CombinedTextDetector:
    """
    Combined detector using separate models for blocks (YOLO) and lines (DBNet).

    This detector runs two models:
    1. YOLOv10s for text block detection (speech bubbles)
    2. TextDetector for segmentation and line detection (DBNet)

    The outputs are combined using group_output() to assign lines to blocks.
    """

    def __init__(
        self,
        dbnet_model_path: Union[str, Path],
        yolo_model_path: Union[str, Path],
        input_size: int = 1024,
        device: str = 'cuda',
        conf_threshold: float = 0.4,
        mask_threshold: float = 0.3,
        use_adaptive_threshold: bool = True,
        fixed_threshold: float = 0.3,
    ):
        """
        Initialize combined detector.

        Args:
            dbnet_model_path: Path to DBNet/segmentation checkpoint
            yolo_model_path: Path to YOLOv10s block detector
            input_size: Model input size
            device: Inference device
            conf_threshold: Confidence threshold for detections
            mask_threshold: Threshold for binary mask
            use_adaptive_threshold: If True, use shrink > threshold_map (learned).
                                    If False, use shrink > fixed_threshold.
            fixed_threshold: Fixed threshold value when use_adaptive_threshold=False
        """
        self.device = device
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.mask_threshold = mask_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.fixed_threshold = fixed_threshold

        # Load DBNet model for segmentation and lines (NOT in block mode)
        self._load_dbnet(dbnet_model_path, input_size, device)

        # Load YOLOv10s for block detection
        self._load_yolo(yolo_model_path)

    def _load_dbnet(self, model_path: Union[str, Path], input_size: int, device: str):
        """
        Load DBNet model for line detection.

        IMPORTANT: Phase 2 checkpoints have a PARTIAL seg_net architecture:
        - seg_net: only has down_conv1, upconv0, upconv2 (upconv3-6 were DELETED)
        - dbnet: has upconv3, upconv4, binarize, thresh, conv

        For inference:
        1. Run seg_net in TEXTDET_DET mode -> returns (f128, f64, u32)
        2. Run dbnet with those features -> returns line detection maps
        3. Use shrink map as approximation for mask (no separate mask head)
        """
        from src.models.detector import create_text_detector
        from src.models.heads import DBHead, TEXTDET_DET

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})

        # IMPORTANT: Use model_state_dict first (has BatchNorm running stats)
        # Then optionally overlay EMA weights for trained parameters
        model_sd = checkpoint.get('model_state_dict', {})
        ema_sd = checkpoint.get('ema_shadow', {})

        # Clean keys (remove _orig_mod. prefix from torch.compile)
        cleaned_state = {}
        for k, v in model_sd.items():
            cleaned_state[k.replace('_orig_mod.', '')] = v

        # Overlay EMA weights (better for inference) if available
        if ema_sd:
            for k, v in ema_sd.items():
                cleaned_state[k.replace('_orig_mod.', '')] = v

        # Create model - we'll manually set up the partial architecture
        model = create_text_detector(
            backbone_name=config.get('backbone_name', 'yolo11s.pt'),
            pretrained_backbone=True,
            freeze_backbone=True,
            use_cib=True,
            use_bifpn=config.get('use_bifpn', False),
            use_rank_guided=config.get('use_rank_guided', False),
            device=device
        )

        # Create DBNet with matching architecture
        model.dbnet = DBHead(64, use_cib=True).to(device)

        # Delete the upconv layers from seg_net that don't exist in checkpoint
        # This matches the training architecture where initialize_db() deleted them
        if hasattr(model.seg_net, 'upconv3'):
            del model.seg_net.upconv3
        if hasattr(model.seg_net, 'upconv4'):
            del model.seg_net.upconv4
        if hasattr(model.seg_net, 'upconv5'):
            del model.seg_net.upconv5
        if hasattr(model.seg_net, 'upconv6'):
            del model.seg_net.upconv6
        if hasattr(model.seg_net, 'large_kernel'):
            del model.seg_net.large_kernel

        # Load weights - should now match architecture
        missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
        print(f"Loaded weights: {len(cleaned_state) - len(missing)} matched, {len(missing)} missing (backbone frozen)")

        model = model.to(device)
        model.eval()

        # Store components for inference
        self.backbone = model.backbone
        self.seg_net = model.seg_net
        self.dbnet = model.dbnet

        print(f"Loaded DBNet model from {model_path}")

    def _load_yolo(self, model_path: Union[str, Path]):
        """Load YOLOv10s model via ultralytics."""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(str(model_path))
            self._has_yolo = True
            print(f"Loaded YOLOv10s block detector from {model_path}")
        except ImportError:
            print("Warning: ultralytics not installed, block detection disabled")
            self.yolo_model = None
            self._has_yolo = False
        except Exception as e:
            print(f"Warning: Failed to load YOLO model: {e}")
            self.yolo_model = None
            self._has_yolo = False

    def __call__(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
    ) -> Union[DetectionResult, List[DetectionResult]]:
        """
        Run combined detection on images.

        Args:
            images: Single image (H, W, 3) BGR or list of images

        Returns:
            DetectionResult with combined blocks and lines
        """
        single_image = isinstance(images, np.ndarray) and images.ndim == 3
        if single_image:
            images = [images]

        results = []
        for img in images:
            result = self._process_single(img)
            results.append(result)

        return results[0] if single_image else results

    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Dict]:
        """Preprocess image for DBNet model.

        NOTE: The model expects BGR input in CHW format, NOT RGB.
        This matches the original comic-text-detector preprocessing.
        """
        h, w = image.shape[:2]
        target_size = self.input_size

        # Calculate scale and padding
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad with gray (114 is standard YOLO padding)
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Convert to tensor - KEEP BGR order (model trained on BGR)
        tensor = padded.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)  # HWC to CHW (BGR stays BGR)
        tensor = torch.from_numpy(np.ascontiguousarray(tensor)).unsqueeze(0)

        meta = {
            'original_size': (h, w),
            'scale': scale,
            'pad': (pad_h, pad_w),
            'new_size': (new_h, new_w),
        }

        return tensor.to(self.device), meta

    def _extract_line_polygons(
        self,
        lines_map: np.ndarray,
        min_area: int = 10,
        unclip_ratio: float = 1.5,
    ) -> List[np.ndarray]:
        """
        Extract line polygons from DBNet using adaptive thresholding.

        DBNet produces shrunk text regions that need to be expanded (unclipped)
        back to their original size using the Vatti clipping algorithm.

        Args:
            lines_map: DBNet output (2, H, W) with shrink and threshold maps
            min_area: Minimum contour area to keep (after unclipping)
            unclip_ratio: Expansion ratio (higher = larger boxes)

        Returns:
            List of line polygons as (4, 2) numpy arrays
        """
        shrink_map = lines_map[0]
        threshold_map = lines_map[1]

        # Choose thresholding method
        if self.use_adaptive_threshold:
            # Adaptive: shrink > threshold_map (learned per-pixel)
            binary = (shrink_map > threshold_map).astype(np.uint8)
        else:
            # Fixed: shrink > fixed_threshold (more robust for some models)
            binary = (shrink_map > self.fixed_threshold).astype(np.uint8)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        lines = []
        for contour in contours:
            # Skip tiny noise
            if len(contour) < 4:
                continue

            # Unclip (expand) the polygon
            expanded = self._unclip_polygon(contour, unclip_ratio)
            if expanded is None:
                continue

            # Get minimum area rectangle
            rect = cv2.minAreaRect(expanded)
            box = cv2.boxPoints(rect)

            # Filter by area after expansion
            area = cv2.contourArea(box)
            if area < min_area:
                continue

            lines.append(box.astype(np.int32))

        return lines

    def _unclip_polygon(
        self,
        contour: np.ndarray,
        unclip_ratio: float = 1.5
    ) -> Optional[np.ndarray]:
        """
        Expand a polygon using the Vatti clipping algorithm.

        The offset distance is calculated as: area * unclip_ratio / perimeter

        Args:
            contour: Input contour
            unclip_ratio: Expansion ratio

        Returns:
            Expanded contour or None if expansion fails
        """
        try:
            import pyclipper
        except ImportError:
            # Fallback: simple bounding rect expansion
            x, y, w, h = cv2.boundingRect(contour)
            expand = int(max(w, h) * (unclip_ratio - 1) / 2)
            expanded_rect = np.array([
                [x - expand, y - expand],
                [x + w + expand, y - expand],
                [x + w + expand, y + h + expand],
                [x - expand, y + h + expand],
            ], dtype=np.int32)
            return expanded_rect

        # Calculate expansion distance
        poly = contour.reshape(-1, 2)
        area = cv2.contourArea(poly)
        perimeter = cv2.arcLength(poly, True)

        if perimeter == 0:
            return None

        # Offset distance based on area/perimeter ratio
        distance = area * unclip_ratio / perimeter

        # Use pyclipper to expand
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(poly.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

        expanded = offset.Execute(distance)
        if not expanded:
            return None

        # Return the largest expanded polygon
        expanded_poly = max(expanded, key=lambda x: cv2.contourArea(np.array(x)))
        return np.array(expanded_poly, dtype=np.int32)

    def _process_single(
        self,
        image: np.ndarray,
        extract_colors: bool = True,
        right_to_left: bool = True,
    ) -> DetectionResult:
        """
        Process a single image with both detectors.

        Args:
            image: BGR image
            extract_colors: Whether to extract fg/bg colors from blocks
            right_to_left: Reading order (True for manga, False for western comics)

        Returns:
            DetectionResult with combined blocks and lines
        """
        h, w = image.shape[:2]

        # 1. Run DBNet model for lines
        # The Phase 2 architecture: backbone -> seg_net (TEXTDET_DET) -> dbnet
        from src.models.heads import TEXTDET_DET

        tensor, meta = self._preprocess(image)
        with torch.no_grad():
            # Get backbone features
            features = self.backbone(tensor)

            # Get intermediate features for dbnet (seg_net has no mask head in Phase 2)
            # Returns (f128, f64, u32) tuple
            seg_features = self.seg_net(*features, forward_mode=TEXTDET_DET)

            # Get DBNet line detection output (shrink_map, threshold_map)
            lines_out = self.dbnet(*seg_features)
            lines_out = lines_out.cpu().numpy()[0]

        # Remove padding and resize back to original size
        pad_h, pad_w = meta['pad']
        new_h, new_w = meta['new_size']

        lines_resized = np.zeros((2, h, w), dtype=np.float32)
        for i in range(2):
            channel = lines_out[i, pad_h:pad_h + new_h, pad_w:pad_w + new_w]
            lines_resized[i] = cv2.resize(channel, (w, h), interpolation=cv2.INTER_LINEAR)

        # Use DBNet's adaptive thresholding: shrink_map > threshold_map
        # This is more accurate than a fixed threshold since the model learns
        # per-pixel thresholds during training
        shrink_map = lines_resized[0]
        threshold_map = lines_resized[1]
        binary_mask = (shrink_map > threshold_map).astype(np.uint8) * 255

        # 2. Run YOLO for blocks
        yolo_blocks = []
        if self._has_yolo and self.yolo_model is not None:
            yolo_results = self.yolo_model(image, verbose=False)
            for r in yolo_results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        if conf >= self.conf_threshold:
                            yolo_blocks.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': conf,
                            })

        # 3. Extract line polygons from DBNet output
        line_polygons = self._extract_line_polygons(lines_resized)

        # 4. Combine with group_output (full pipeline)
        text_blocks = group_output(
            blocks=yolo_blocks,
            lines=line_polygons,
            im_w=w,
            im_h=h,
            mask=binary_mask,
            image=image if extract_colors else None,
            bbox_score_thresh=0.4,
            mask_score_thresh=0.1,
            sort_by_reading_order=True,
            right_to_left=right_to_left,
            extract_block_colors=extract_colors,
        )

        # Convert TextBlock objects to dicts for DetectionResult
        text_block_dicts = [blk.to_dict() for blk in text_blocks]

        return DetectionResult(
            mask=binary_mask,
            lines_map=lines_resized,
            text_blocks=text_block_dicts,
            original_size=(h, w),
        )


class BatchTextDetector:
    """
    Optimized batch detector for processing many images.

    Uses async preprocessing and batched GPU inference.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        input_size: int = 1024,
        device: str = 'cuda',
        half: bool = True,
        max_batch_size: int = 8,
        num_workers: int = 4,
    ):
        self.detector = TextDetector(
            model_path=model_path,
            input_size=input_size,
            device=device,
            half=half,
        )
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        save_masks: bool = True,
        save_json: bool = True,
    ) -> List[DetectionResult]:
        """
        Process all images in a directory.

        Args:
            input_dir: Directory with input images
            output_dir: Directory for output files
            save_masks: Save binary masks
            save_json: Save detection JSON

        Returns:
            List of DetectionResults
        """
        import json
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            image_paths.extend(input_dir.glob(ext))

        # Load images in parallel
        def load_image(path):
            return cv2.imread(str(path)), path.stem

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            loaded = list(executor.map(load_image, image_paths))

        images = [img for img, _ in loaded]
        names = [name for _, name in loaded]

        # Process in batches
        results = []
        for i in tqdm(range(0, len(images), self.max_batch_size), desc="Processing"):
            batch = images[i:i + self.max_batch_size]
            batch_names = names[i:i + self.max_batch_size]

            batch_results = self.detector(batch, batch_size=len(batch))

            for result, name in zip(batch_results, batch_names):
                results.append(result)

                if save_masks:
                    mask_path = output_dir / f"{name}_mask.png"
                    cv2.imwrite(str(mask_path), result.mask)

                if save_json:
                    json_path = output_dir / f"{name}.json"
                    with open(json_path, 'w') as f:
                        json.dump({
                            'text_blocks': result.text_blocks,
                            'original_size': result.original_size,
                        }, f, indent=2)

        return results
