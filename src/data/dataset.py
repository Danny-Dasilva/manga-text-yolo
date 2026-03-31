"""
Unified Dataset for Comic Text Detection

Supports both segmentation mask training and text line detection (DB) training
with modern albumentations-based augmentation pipeline.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ComicTextDataset(Dataset):
    """
    Unified dataset for comic text detection.

    Supports:
        - Segmentation mask training (image + binary mask pairs)
        - Text line detection training (image + polygon annotations)
        - Block detection training (image + block bounding boxes for YOLO-style detection)

    Uses albumentations for efficient, GPU-friendly augmentation.
    """

    def __init__(
        self,
        img_dirs: List[str | Path],
        annotation_dirs: List[str | Path],
        mode: str = 'segmentation',  # 'segmentation', 'detection', or 'block'
        img_size: int = 1024,
        transforms: Optional[A.Compose] = None,
        augment: bool = True,
        cache: bool = False,
        shrink_ratio: float = 0.4,
    ):
        """
        Initialize dataset.

        Args:
            img_dirs: List of directories containing images
            annotation_dirs: List of directories containing annotations
                - For segmentation: mask images
                - For detection: text files with polygon coordinates
            mode: 'segmentation' or 'detection'
            img_size: Target image size (square)
            transforms: Custom albumentations transforms (overrides default)
            augment: Whether to apply augmentation
            cache: Whether to cache loaded images in memory
            shrink_ratio: Shrink ratio for DBNet maps (detection mode only)
        """
        self.img_dirs = [Path(d) for d in img_dirs]
        self.annotation_dirs = [Path(d) for d in annotation_dirs]
        self.mode = mode
        self.img_size = img_size
        self.augment = augment
        self.cache = cache
        self.shrink_ratio = shrink_ratio

        # Build file list
        self.samples = self._build_samples()

        # Initialize cache
        self._cache: Dict[int, Any] = {} if cache else None

        # Setup transforms
        self.transforms = transforms if transforms is not None else self._default_transforms()

    def _build_samples(self) -> List[Tuple[Path, Path]]:
        """Build list of (image_path, annotation_path) tuples."""
        samples = []

        for img_dir, ann_dir in zip(self.img_dirs, self.annotation_dirs):
            img_dir = Path(img_dir)
            ann_dir = Path(ann_dir)

            # Find all images
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                for img_path in img_dir.glob(ext):
                    # Find corresponding annotation
                    if self.mode == 'segmentation':
                        # Look for mask with same name
                        ann_path = ann_dir / img_path.name
                        if not ann_path.exists():
                            ann_path = ann_dir / f"{img_path.stem}.png"
                    elif self.mode == 'unified':
                        # Unified mode: look for mask (we'll find other annotations at load time)
                        ann_path = ann_dir / f"{img_path.stem}.png"
                        if not ann_path.exists():
                            ann_path = ann_dir / img_path.name
                    elif self.mode in ('detection', 'block'):
                        # Look for JSON annotation (with bbox data for both detection and block modes)
                        ann_path = ann_dir / f"{img_path.stem}.json"

                    if ann_path.exists():
                        samples.append((img_path, ann_path))

        return samples

    def _default_transforms(self) -> A.Compose:
        """Create default augmentation pipeline."""
        if self.mode == 'block':
            # Block mode with bbox-aware transforms
            transforms = []
            if self.augment:
                transforms.extend([
                    # Geometric augmentations (applied to both image and bboxes)
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_CONSTANT),
                    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), p=0.3),
                    # Color augmentations
                    A.OneOf([
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=1.0
                        ),
                        A.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.2,
                            hue=0.1,
                            p=1.0
                        ),
                    ], p=0.5),
                    A.OneOf([
                        A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    ], p=0.2),
                    A.InvertImg(p=0.1),
                ])

            # NOTE: ImageNet normalization is applied in the ONNX export wrapper (ExportableModel).
            # Do NOT apply it here during training, or inference will have double normalization!
            # ToTensorV2() outputs [0, 1] range, which is what the ONNX export expects.
            transforms.extend([
                A.Resize(self.img_size, self.img_size),
                ToTensorV2(),
            ])

            return A.Compose(
                transforms,
                bbox_params=A.BboxParams(
                    format='pascal_voc',  # [x1, y1, x2, y2]
                    label_fields=['class_labels'],
                    min_visibility=0.3
                )
            )

        if self.mode == 'detection':
            transforms = []
            if self.augment:
                transforms.extend([
                    A.OneOf([
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=1.0
                        ),
                        A.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.2,
                            hue=0.1,
                            p=1.0
                        ),
                    ], p=0.5),
                    A.OneOf([
                        A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    ], p=0.2),
                    A.InvertImg(p=0.1),
                ])

            # NOTE: ImageNet normalization is applied in the ONNX export wrapper (ExportableModel).
            # Do NOT apply it here during training, or inference will have double normalization!
            # ToTensorV2() outputs [0, 1] range, which is what the ONNX export expects.
            transforms.extend([
                A.Resize(self.img_size, self.img_size),
                ToTensorV2(),
            ])

            return A.Compose(transforms)

        if self.augment:
            return A.Compose([
                # Geometric transforms
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
                A.RandomResizedCrop(
                    size=(self.img_size, self.img_size),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.3
                ),

                # Color transforms
                A.OneOf([
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0
                    ),
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1,
                        p=1.0
                    ),
                ], p=0.5),

                A.OneOf([
                    A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                ], p=0.2),

                # Invert colors (common for manga)
                A.InvertImg(p=0.1),

                # NOTE: ImageNet normalization is applied in the ONNX export wrapper (ExportableModel).
                # Do NOT apply it here during training, or inference will have double normalization!
                # ToTensorV2() outputs [0, 1] range, which is what the ONNX export expects.
                A.Resize(self.img_size, self.img_size),
                ToTensorV2(),
            ], additional_targets={'mask': 'mask'})
        else:
            # NOTE: ImageNet normalization is applied in the ONNX export wrapper (ExportableModel).
            # Do NOT apply it here during training, or inference will have double normalization!
            # ToTensorV2() outputs [0, 1] range, which is what the ONNX export expects.
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                ToTensorV2(),
            ], additional_targets={'mask': 'mask'})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - 'image': (3, H, W) tensor
                - 'mask': (1, H, W) tensor (segmentation mode)
                - 'shrink_map', 'threshold_map': (1, H, W) tensors (detection mode)
        """
        # Check cache
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        img_path, ann_path = self.samples[idx]

        # Load image (BGR -> RGB)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == 'segmentation':
            result = self._load_segmentation(image, ann_path)
        elif self.mode == 'detection':
            result = self._load_detection(image, ann_path)
        elif self.mode == 'unified':
            result = self._load_unified(image, ann_path)
        else:  # block mode
            result = self._load_block(image, ann_path)

        # Cache if enabled
        if self._cache is not None:
            self._cache[idx] = result

        return result

    def _load_segmentation(
        self,
        image: np.ndarray,
        mask_path: Path
    ) -> Dict[str, torch.Tensor]:
        """Load segmentation sample with mask."""
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Binarize mask
        mask = (mask > 30).astype(np.float32)

        # Apply transforms
        transformed = self.transforms(image=image, mask=mask)

        image_tensor = transformed['image']
        mask_tensor = transformed['mask']

        # Convert image to float and scale to [0, 1] range
        # ToTensorV2 outputs uint8, we need float for training
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.float() / 255.0

        # Add channel dimension to mask if needed
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        return {
            'image': image_tensor,
            'mask': mask_tensor.float(),
        }

    def _load_detection(
        self,
        image: np.ndarray,
        ann_path: Path
    ) -> Dict[str, torch.Tensor]:
        """Load detection sample with polygon annotations."""
        # Load polygon annotations
        polygons = self._load_polygons(ann_path)

        # Apply image transforms only
        transformed = self.transforms(image=image)
        image_tensor = transformed['image']

        # Convert image to float and scale to [0, 1] range
        # ToTensorV2 outputs uint8, we need float for training
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.float() / 255.0

        # Generate DBNet target maps
        h, w = self.img_size, self.img_size
        shrink_map = np.zeros((h, w), dtype=np.float32)
        threshold_map = np.zeros((h, w), dtype=np.float32)

        # Scale polygons to new size
        orig_h, orig_w = image.shape[:2]
        scale_x = w / orig_w
        scale_y = h / orig_h

        for polygon in polygons:
            # Scale polygon
            scaled_poly = polygon.copy()
            scaled_poly[:, 0] *= scale_x
            scaled_poly[:, 1] *= scale_y
            scaled_poly = scaled_poly.astype(np.int32)

            # Generate shrink map (filled polygon)
            shrunk_poly = self._shrink_polygon(scaled_poly, self.shrink_ratio)
            if shrunk_poly is not None:
                cv2.fillPoly(shrink_map, [shrunk_poly], 1.0)

            # Generate threshold map (border region)
            self._generate_border_map(scaled_poly, threshold_map)

        return {
            'image': image_tensor,
            'shrink_map': torch.from_numpy(shrink_map).unsqueeze(0),
            'threshold_map': torch.from_numpy(threshold_map).unsqueeze(0),
        }

    def _load_unified(
        self,
        image: np.ndarray,
        mask_path: Path
    ) -> Dict[str, torch.Tensor]:
        """
        Load unified sample with mask, detection maps, and block boxes.

        For unified training, we load:
        - mask: from .png file (segmentation)
        - shrink_map/threshold_map: from line-*.txt file (detection)
        - blocks: from block_annotations/*.txt file (block detection)
        """
        import json
        orig_h, orig_w = image.shape[:2]

        # 1. Load segmentation mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((orig_h, orig_w), dtype=np.float32)
        else:
            mask = (mask > 30).astype(np.float32)

        # 2. Load line annotations for detection maps
        ann_dir = mask_path.parent
        line_ann_path = ann_dir / f"line-{mask_path.stem}.txt"
        if not line_ann_path.exists():
            # Try without 'line-' prefix
            line_ann_path = ann_dir / f"{mask_path.stem}.txt"

        polygons = []
        if line_ann_path.exists():
            polygons = self._load_polygons(line_ann_path)

        # 3. Load block annotations
        # Try to find block_annotations directory (sibling to annotations)
        block_ann_dir = ann_dir.parent / 'block_annotations'
        block_ann_path = block_ann_dir / f"{mask_path.stem}.txt"

        bboxes = []
        class_labels = []
        if block_ann_path.exists():
            try:
                with open(block_ann_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x_center = float(parts[1]) * orig_w
                            y_center = float(parts[2]) * orig_h
                            width = float(parts[3]) * orig_w
                            height = float(parts[4]) * orig_h
                            x1 = x_center - width / 2
                            y1 = y_center - height / 2
                            x2 = x_center + width / 2
                            y2 = y_center + height / 2
                            # Clamp to image bounds
                            x1 = max(0, min(orig_w, x1))
                            y1 = max(0, min(orig_h, y1))
                            x2 = max(0, min(orig_w, x2))
                            y2 = max(0, min(orig_h, y2))
                            if x2 > x1 and y2 > y1:
                                bboxes.append([x1, y1, x2, y2])
                                class_labels.append(cls_id)
            except Exception:
                pass

        # Apply transforms (use segmentation-style transforms)
        transformed = self.transforms(image=image, mask=mask)
        image_tensor = transformed['image']
        mask_tensor = transformed['mask']

        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.float() / 255.0

        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        # Generate DBNet target maps
        h, w = self.img_size, self.img_size
        shrink_map = np.zeros((h, w), dtype=np.float32)
        threshold_map = np.zeros((h, w), dtype=np.float32)

        scale_x = w / orig_w
        scale_y = h / orig_h

        for polygon in polygons:
            scaled_poly = polygon.copy()
            scaled_poly[:, 0] *= scale_x
            scaled_poly[:, 1] *= scale_y
            scaled_poly = scaled_poly.astype(np.int32)

            shrunk_poly = self._shrink_polygon(scaled_poly, self.shrink_ratio)
            if shrunk_poly is not None:
                cv2.fillPoly(shrink_map, [shrunk_poly], 1.0)

            self._generate_border_map(scaled_poly, threshold_map)

        # Scale block bboxes
        blocks = []
        for bbox, cls_id in zip(bboxes, class_labels):
            x1, y1, x2, y2 = bbox
            # Scale to target size
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y
            # Convert to normalized YOLO format
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))
            blocks.append([cls_id, x_center, y_center, width, height])

        if blocks:
            blocks_tensor = torch.tensor(blocks, dtype=torch.float32)
        else:
            blocks_tensor = torch.zeros((0, 5), dtype=torch.float32)

        return {
            'image': image_tensor,
            'mask': mask_tensor.float(),
            'shrink_map': torch.from_numpy(shrink_map).unsqueeze(0),
            'threshold_map': torch.from_numpy(threshold_map).unsqueeze(0),
            'blocks': blocks_tensor,
            'num_blocks': torch.tensor(len(blocks), dtype=torch.long),
        }

    def _load_block(
        self,
        image: np.ndarray,
        ann_path: Path
    ) -> Dict[str, torch.Tensor]:
        """
        Load block detection sample with bounding box annotations.

        Loads block bounding boxes from JSON annotations for YOLO-style detection training.
        Returns normalized bounding boxes in [x_center, y_center, width, height] format.

        Bboxes are passed as raw pixel coordinates to albumentations transforms,
        which handles geometric augmentation and normalization via bbox_params.
        """
        import json

        orig_h, orig_w = image.shape[:2]

        # Load block bounding boxes from JSON as raw pixel coordinates
        bboxes = []  # pascal_voc format: [x1, y1, x2, y2]
        class_labels = []

        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)

            # Load from 'blocks' key if available (preferred for block detection)
            # Otherwise fall back to 'text_blocks' for compatibility
            block_list = data.get('blocks', data.get('text_blocks', []))

            for block in block_list:
                bbox = block.get('bbox', block.get('xyxy', []))
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]

                    # Clamp to image bounds (raw pixel coords)
                    x1 = max(0.0, min(float(orig_w), float(x1)))
                    y1 = max(0.0, min(float(orig_h), float(y1)))
                    x2 = max(0.0, min(float(orig_w), float(x2)))
                    y2 = max(0.0, min(float(orig_h), float(y2)))

                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue

                    bboxes.append([x1, y1, x2, y2])
                    class_labels.append(block.get('class_id', 0))
        except (json.JSONDecodeError, KeyError):
            pass  # Return empty blocks if JSON parsing fails

        # Apply transforms with bboxes (albumentations handles geometric transforms)
        transformed = self.transforms(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        image_tensor = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_labels = transformed['class_labels']

        # Convert image to float and scale to [0, 1] range
        # ToTensorV2 outputs uint8, we need float for training
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.float() / 255.0

        # Convert transformed bboxes to normalized YOLO format [cls, x_center, y_center, w, h]
        # After resize, image is now img_size x img_size
        blocks = []
        for bbox, cls_id in zip(transformed_bboxes, transformed_labels):
            x1, y1, x2, y2 = bbox

            # Normalize to [0, 1] based on target image size (after resize)
            x_center = ((x1 + x2) / 2) / self.img_size
            y_center = ((y1 + y2) / 2) / self.img_size
            width = (x2 - x1) / self.img_size
            height = (y2 - y1) / self.img_size

            # Clamp to valid range
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            blocks.append([cls_id, x_center, y_center, width, height])

        # Convert to tensor
        if blocks:
            blocks_tensor = torch.tensor(blocks, dtype=torch.float32)
        else:
            # Empty tensor with correct shape [0, 5] for (cls, x, y, w, h)
            blocks_tensor = torch.zeros((0, 5), dtype=torch.float32)

        return {
            'image': image_tensor,
            'blocks': blocks_tensor,  # [N, 5] where each row is [cls, x_center, y_center, w, h]
            'num_blocks': torch.tensor(len(blocks), dtype=torch.long),
        }

    def _load_block_annotations(self, ann_path: Path) -> List[Dict]:
        """
        Load block bounding box annotations from a JSON file.

        Supports JSON format with 'blocks' or 'text_blocks' keys containing
        bbox annotations in [x1, y1, x2, y2] format.

        Args:
            ann_path: Path to the JSON annotation file

        Returns:
            List of dictionaries with 'bbox' and optional 'class_id' keys
        """
        import json

        blocks = []
        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)

            # Load from 'blocks' key if available (preferred for block detection)
            # Otherwise fall back to 'text_blocks' for compatibility
            block_list = data.get('blocks', data.get('text_blocks', []))

            for block in block_list:
                bbox = block.get('bbox', block.get('xyxy', []))
                if len(bbox) >= 4:
                    blocks.append({
                        'bbox': bbox[:4],
                        'class_id': block.get('class_id', 0),
                    })
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            pass  # Return empty list if JSON parsing fails

        return blocks

    def _load_polygons(self, ann_path: Path) -> List[np.ndarray]:
        """Load polygon annotations from JSON file."""
        import json
        
        polygons = []
        
        try:
            with open(ann_path, 'r') as f:
                data = json.load(f)
            
            text_blocks = data.get('text_blocks', [])
            
            for block in text_blocks:
                bbox = block.get('bbox', [])
                if len(bbox) >= 4:
                    # bbox is [x1, y1, x2, y2] - convert to 4-point polygon
                    x1, y1, x2, y2 = bbox[:4]
                    polygon = np.array([
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ], dtype=np.float32)
                    polygons.append(polygon)
        except (json.JSONDecodeError, KeyError) as e:
            # If JSON parsing fails, try legacy text format
            with open(ann_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.replace(',', ' ').split()
                    coords = [float(x) for x in parts[:8]]
                    
                    if len(coords) >= 8:
                        polygon = np.array(coords[:8]).reshape(4, 2)
                        polygons.append(polygon)
        
        return polygons

    def _shrink_polygon(
        self,
        polygon: np.ndarray,
        ratio: float
    ) -> Optional[np.ndarray]:
        """Shrink polygon by given ratio using pyclipper."""
        try:
            import pyclipper

            # Calculate shrink distance
            area = cv2.contourArea(polygon)
            perimeter = cv2.arcLength(polygon, True)

            if perimeter == 0:
                return None

            distance = area * (1 - ratio ** 2) / perimeter

            # Use pyclipper to offset polygon
            subject = [polygon.tolist()]
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(subject[0], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrunk = pco.Execute(-distance)

            if not shrunk:
                return None

            return np.array(shrunk[0], dtype=np.int32)

        except ImportError:
            # Fallback: simple center shrink
            center = polygon.mean(axis=0)
            shrunk = center + ratio * (polygon - center)
            return shrunk.astype(np.int32)

    def _generate_border_map(
        self,
        polygon: np.ndarray,
        threshold_map: np.ndarray,
        thresh_min: float = 0.3,
        thresh_max: float = 0.7
    ):
        """
        Generate threshold map based on distance from polygon border.

        Creates a smooth gradient from thresh_max (at polygon edge) to thresh_min
        (at expanded border boundary), using the same expand distance as shrink.

        The DBNet paper uses this adaptive threshold to handle varying text sizes
        and stroke widths. The threshold_map values should be:
        - High (0.7) near the text edges
        - Low (0.3) at the outer border
        - Smooth gradient in between
        """
        h, w = threshold_map.shape

        # Calculate expand distance (same formula as shrink, but expand outward)
        area = cv2.contourArea(polygon)
        perimeter = cv2.arcLength(polygon, True)

        if perimeter == 0:
            return

        # Distance is proportional to area/perimeter (like shrink)
        distance = area * (1 - self.shrink_ratio ** 2) / perimeter

        # Expand polygon outward
        try:
            import pyclipper
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(polygon.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = pco.Execute(distance)

            if not expanded:
                return

            expanded_poly = np.array(expanded[0], dtype=np.int32)
        except ImportError:
            # Fallback: use bounding rect expansion
            x, y, bw, bh = cv2.boundingRect(polygon)
            expand = int(distance)
            expanded_poly = np.array([
                [x - expand, y - expand],
                [x + bw + expand, y - expand],
                [x + bw + expand, y + bh + expand],
                [x - expand, y + bh + expand],
            ], dtype=np.int32)

        # Create masks for original polygon and expanded polygon
        original_mask = np.zeros((h, w), dtype=np.uint8)
        expanded_mask = np.zeros((h, w), dtype=np.uint8)

        cv2.fillPoly(original_mask, [polygon], 255)
        cv2.fillPoly(expanded_mask, [expanded_poly], 255)

        # Border region = expanded - original
        border_region = cv2.bitwise_and(expanded_mask, cv2.bitwise_not(original_mask))

        if border_region.sum() == 0:
            return

        # Compute distance transform from original polygon edge
        # Distance from edge: 0 at edge, increases outward
        dist_from_edge = cv2.distanceTransform(
            cv2.bitwise_not(original_mask),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE
        )

        # Normalize distance in border region to [0, 1]
        # 0 = at original edge (thresh_max), 1 = at expanded boundary (thresh_min)
        if distance > 0:
            normalized_dist = np.clip(dist_from_edge / distance, 0, 1)
        else:
            normalized_dist = np.zeros_like(dist_from_edge)

        # Compute threshold values: high near edge, low at boundary
        # threshold = thresh_max - normalized_dist * (thresh_max - thresh_min)
        # At edge (normalized_dist=0): threshold = thresh_max (0.7)
        # At boundary (normalized_dist=1): threshold = thresh_min (0.3)
        threshold_values = thresh_max - normalized_dist * (thresh_max - thresh_min)

        # Apply only in border region
        border_mask = border_region > 0
        threshold_map[border_mask] = np.maximum(
            threshold_map[border_mask],
            threshold_values[border_mask]
        )


def create_dataloader(
    img_dirs: List[str | Path],
    annotation_dirs: List[str | Path],
    mode: str = 'segmentation',
    img_size: int = 1024,
    batch_size: int = 4,
    augment: bool = True,
    cache: bool = False,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for training or validation.

    Args:
        img_dirs: List of image directories
        annotation_dirs: List of annotation directories
        mode: 'segmentation', 'detection', or 'block'
        img_size: Target image size
        batch_size: Batch size
        augment: Whether to apply augmentation
        cache: Whether to cache data in memory
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    dataset = ComicTextDataset(
        img_dirs=img_dirs,
        annotation_dirs=annotation_dirs,
        mode=mode,
        img_size=img_size,
        augment=augment,
        cache=cache,
    )

    # Select appropriate collate function based on mode
    if mode == 'block':
        collate_fn = block_collate_fn
    elif mode == 'detection':
        collate_fn = detection_collate_fn
    elif mode == 'segmentation':
        collate_fn = segmentation_collate_fn
    elif mode == 'unified':
        collate_fn = unified_collate_fn
    else:
        collate_fn = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if shuffle else False,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=4 if num_workers > 0 else None,  # Prefetch batches
    )


# Collate functions for variable-length annotations
def detection_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for detection mode with variable polygon counts."""
    images = torch.stack([item['image'] for item in batch])
    shrink_maps = torch.stack([item['shrink_map'] for item in batch])
    threshold_maps = torch.stack([item['threshold_map'] for item in batch])

    return {
        'image': images,
        'shrink_map': shrink_maps,
        'threshold_map': threshold_maps,
    }


def segmentation_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for segmentation mode."""
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])

    return {
        'image': images,
        'mask': masks,
    }


def block_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for block detection mode with variable block counts.
    
    Handles variable-length block annotations by padding to max count in batch.
    Each block is [cls, x_center, y_center, width, height] normalized to [0, 1].
    """
    images = torch.stack([item['image'] for item in batch])
    num_blocks = torch.stack([item['num_blocks'] for item in batch])
    
    # Find max number of blocks in this batch
    max_blocks = max(item['blocks'].shape[0] for item in batch)
    
    if max_blocks == 0:
        # No blocks in any sample - return empty tensor
        blocks = torch.zeros((len(batch), 0, 5), dtype=torch.float32)
    else:
        # Pad blocks to max_blocks for batching
        padded_blocks = []
        for item in batch:
            blocks_item = item['blocks']
            n = blocks_item.shape[0]
            if n < max_blocks:
                # Pad with zeros (will be masked out by num_blocks)
                padding = torch.zeros((max_blocks - n, 5), dtype=torch.float32)
                blocks_item = torch.cat([blocks_item, padding], dim=0)
            padded_blocks.append(blocks_item)
        blocks = torch.stack(padded_blocks)
    
    return {
        'image': images,
        'blocks': blocks,  # [B, max_blocks, 5]
        'num_blocks': num_blocks,  # [B] - actual count per sample (for masking)
    }


def unified_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for unified training mode.

    Combines all annotations: mask, shrink/threshold maps, and blocks.
    """
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    shrink_maps = torch.stack([item['shrink_map'] for item in batch])
    threshold_maps = torch.stack([item['threshold_map'] for item in batch])
    num_blocks = torch.stack([item['num_blocks'] for item in batch])

    # Find max number of blocks in this batch
    max_blocks = max(item['blocks'].shape[0] for item in batch)

    if max_blocks == 0:
        blocks = torch.zeros((len(batch), 0, 5), dtype=torch.float32)
    else:
        padded_blocks = []
        for item in batch:
            blocks_item = item['blocks']
            n = blocks_item.shape[0]
            if n < max_blocks:
                padding = torch.zeros((max_blocks - n, 5), dtype=torch.float32)
                blocks_item = torch.cat([blocks_item, padding], dim=0)
            padded_blocks.append(blocks_item)
        blocks = torch.stack(padded_blocks)

    return {
        'image': images,
        'mask': masks,
        'shrink_map': shrink_maps,
        'threshold_map': threshold_maps,
        'blocks': blocks,
        'num_blocks': num_blocks,
    }
