"""
TextBlock class and post-processing functions for comic text detection.

This module implements the complete line-to-block grouping pipeline from the original
comic-text-detector, including:
- Line assignment to blocks based on spatial overlap
- Block splitting for vertical text with large gaps
- Scattered line merging
- Reading order sorting
- Language detection
- Color extraction

Reference: https://github.com/dmMaze/comic-text-detector
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import cv2
import math


# ============================================================================
# TextBlock Data Structure
# ============================================================================

@dataclass
class TextBlock:
    """
    Container for a text block with its constituent text lines.

    A text block represents a speech bubble or text region detected by YOLO,
    containing one or more text lines detected by DBNet.

    Attributes:
        xyxy: Bounding box coordinates [x1, y1, x2, y2]
        lines: List of line polygons, each as numpy array of shape (4, 2)
        language: Detected language ('ja', 'eng', 'unknown')
        vertical: Whether the text is vertically oriented
        font_size: Estimated font size in pixels
        angle: Text rotation angle in degrees
        vec: Primary direction vector [dx, dy]
        norm: Magnitude of direction vector
        distance: Distance array for line ordering
        confidence: Detection confidence score
        fg_colors: Foreground RGB colors (r, g, b)
        bg_colors: Background RGB colors (r, g, b)
        text: OCR results for each line
        translation: Translated text
        merged: Whether this block was created by merging
    """
    xyxy: List[int]
    lines: List[np.ndarray] = field(default_factory=list)
    language: str = 'unknown'
    vertical: bool = False
    font_size: float = -1
    angle: float = 0
    vec: List[float] = field(default_factory=lambda: [0, 0])
    norm: float = -1
    distance: List[float] = field(default_factory=list)
    confidence: float = 1.0
    fg_colors: Tuple[int, int, int] = (0, 0, 0)
    bg_colors: Tuple[int, int, int] = (255, 255, 255)
    text: List[str] = field(default_factory=list)
    translation: str = ""
    merged: bool = False
    _weight: float = -1  # For sorting

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'xyxy': self.xyxy,
            'bbox': self.xyxy,  # Alias for compatibility
            'lines': [line.tolist() if isinstance(line, np.ndarray) else line
                      for line in self.lines],
            'language': self.language,
            'vertical': self.vertical,
            'font_size': self.font_size,
            'angle': self.angle,
            'vec': self.vec,
            'confidence': self.confidence,
            'fg_colors': self.fg_colors,
            'bg_colors': self.bg_colors,
            'text': self.text,
            'translation': self.translation,
            'polygon': [[self.xyxy[0], self.xyxy[1]],
                        [self.xyxy[2], self.xyxy[1]],
                        [self.xyxy[2], self.xyxy[3]],
                        [self.xyxy[0], self.xyxy[3]]],
            'area': (self.xyxy[2] - self.xyxy[0]) * (self.xyxy[3] - self.xyxy[1]),
        }

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        return (
            (self.xyxy[0] + self.xyxy[2]) / 2,
            (self.xyxy[1] + self.xyxy[3]) / 2
        )

    @property
    def width(self) -> int:
        """Get width of bounding box."""
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> int:
        """Get height of bounding box."""
        return self.xyxy[3] - self.xyxy[1]


# ============================================================================
# Geometric Utility Functions
# ============================================================================

def intersection_area(bbox_a: List[float], bbox_b: List[float]) -> float:
    """
    Compute intersection area of two bounding boxes.

    Args:
        bbox_a: First bounding box [x1, y1, x2, y2]
        bbox_b: Second bounding box [x1, y1, x2, y2]

    Returns:
        Intersection area in pixels, or 0 if no overlap
    """
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def get_mini_boxes(contour: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Get minimum area bounding box for a contour.

    Args:
        contour: Contour points

    Returns:
        Tuple of (box_points, min_side_length)
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.array(sorted(box, key=lambda x: x[0]))

    # Sort by x, then separate left/right pairs
    left = box[:2]
    right = box[2:]
    left = sorted(left, key=lambda x: x[1])
    right = sorted(right, key=lambda x: x[1])

    # Order: top-left, top-right, bottom-right, bottom-left
    ordered = np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)

    w = np.linalg.norm(ordered[0] - ordered[1])
    h = np.linalg.norm(ordered[1] - ordered[2])

    return ordered, min(w, h)


def polygon_to_bbox(polygon: np.ndarray) -> List[int]:
    """Convert polygon to bounding box [x1, y1, x2, y2]."""
    x1, y1 = polygon.min(axis=0)
    x2, y2 = polygon.max(axis=0)
    return [int(x1), int(y1), int(x2), int(y2)]


def rotate_points(points: np.ndarray, angle: float, center: Tuple[float, float]) -> np.ndarray:
    """Rotate points around a center by given angle in degrees."""
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    cx, cy = center

    rotated = np.zeros_like(points)
    for i, (x, y) in enumerate(points):
        dx, dy = x - cx, y - cy
        rotated[i, 0] = cx + dx * cos_a - dy * sin_a
        rotated[i, 1] = cy + dx * sin_a + dy * cos_a

    return rotated


# ============================================================================
# Text Block Analysis Functions
# ============================================================================

def examine_textblock(
    block: TextBlock,
    im_w: int,
    im_h: int,
    sort: bool = True
) -> None:
    """
    Analyze and update text block properties in-place.

    Computes:
    - Orientation (vertical/horizontal)
    - Primary direction vector and angle
    - Font size from line heights
    - Distance array for line ordering
    - Adjusts bbox to fit lines

    Args:
        block: TextBlock to analyze
        im_w: Image width
        im_h: Image height
        sort: Whether to sort lines by distance
    """
    if not block.lines:
        return

    lines = block.lines

    # Compute center points and dimensions of each line
    centers = []
    widths = []
    heights = []

    for line in lines:
        if len(line) < 3:
            continue
        cx = line[:, 0].mean()
        cy = line[:, 1].mean()
        centers.append([cx, cy])

        w = line[:, 0].max() - line[:, 0].min()
        h = line[:, 1].max() - line[:, 1].min()
        widths.append(w)
        heights.append(h)

    if not centers:
        return

    centers = np.array(centers)
    avg_width = np.mean(widths)
    avg_height = np.mean(heights)

    # Determine orientation based on line aspect ratios
    # Vertical text has tall, narrow lines
    block.vertical = avg_height > avg_width * 1.5

    # Compute primary direction vector using PCA
    if len(centers) >= 2:
        # Center the data
        mean_center = centers.mean(axis=0)
        centered = centers - mean_center

        # Compute covariance and eigenvectors
        cov = np.cov(centered.T)
        if cov.ndim == 0:
            cov = np.array([[cov, 0], [0, cov]])

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Primary direction is eigenvector with largest eigenvalue
        primary_idx = np.argmax(eigenvalues)
        vec = eigenvectors[:, primary_idx]

        # Ensure consistent direction (pointing down/right)
        if block.vertical:
            if vec[1] < 0:
                vec = -vec
        else:
            if vec[0] < 0:
                vec = -vec

        block.vec = vec.tolist()
        block.norm = float(np.linalg.norm(vec))
        block.angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
    else:
        block.vec = [0, 1] if block.vertical else [1, 0]
        block.norm = 1.0
        block.angle = 90.0 if block.vertical else 0.0

    # Compute font size (use the smaller dimension as font size)
    if block.vertical:
        block.font_size = float(avg_width) if avg_width > 0 else -1
    else:
        block.font_size = float(avg_height) if avg_height > 0 else -1

    # Compute distance from origin for each line (for ordering)
    # Project each center onto the primary direction vector
    vec = np.array(block.vec)
    if block.norm > 0:
        vec = vec / block.norm

    origin = centers.mean(axis=0)
    distances = []
    for center in centers:
        dist = np.dot(center - origin, vec)
        distances.append(float(dist))

    block.distance = distances

    # Sort lines by distance if requested
    if sort and len(lines) > 1:
        sorted_indices = np.argsort(distances)
        block.lines = [lines[i] for i in sorted_indices]
        block.distance = [distances[i] for i in sorted_indices]

    # Adjust bbox to fit all lines
    all_points = np.vstack(lines)
    x1, y1 = all_points.min(axis=0)
    x2, y2 = all_points.max(axis=0)

    # Clip to image bounds
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(im_w, int(x2))
    y2 = min(im_h, int(y2))

    block.xyxy = [x1, y1, x2, y2]


def split_textblock(
    block: TextBlock,
    im_w: int,
    im_h: int,
    gap_threshold_ratio: float = 2.0
) -> List[TextBlock]:
    """
    Split a text block if there are large gaps between lines.

    For vertical Japanese text, speech bubbles may contain multiple
    columns that should be treated as separate blocks.

    Args:
        block: TextBlock to potentially split
        im_w: Image width
        im_h: Image height
        gap_threshold_ratio: Split if gap > font_size * this ratio

    Returns:
        List of TextBlocks (may be single block if no split needed)
    """
    if len(block.lines) < 2 or block.font_size <= 0:
        return [block]

    # Only split vertical text blocks (Japanese manga columns)
    if not block.vertical:
        return [block]

    # Compute gaps between consecutive lines
    gaps = []
    for i in range(len(block.distance) - 1):
        gap = abs(block.distance[i + 1] - block.distance[i])
        gaps.append(gap)

    # Find split points where gap exceeds threshold
    threshold = block.font_size * gap_threshold_ratio
    split_indices = [i + 1 for i, gap in enumerate(gaps) if gap > threshold]

    if not split_indices:
        return [block]

    # Create new blocks for each segment
    result = []
    prev_idx = 0

    for split_idx in split_indices + [len(block.lines)]:
        segment_lines = block.lines[prev_idx:split_idx]
        if segment_lines:
            # Create new block with segment
            all_points = np.vstack(segment_lines)
            x1, y1 = all_points.min(axis=0)
            x2, y2 = all_points.max(axis=0)

            new_block = TextBlock(
                xyxy=[int(x1), int(y1), int(x2), int(y2)],
                lines=segment_lines,
                language=block.language,
                vertical=block.vertical,
                confidence=block.confidence,
            )
            examine_textblock(new_block, im_w, im_h, sort=False)
            result.append(new_block)

        prev_idx = split_idx

    return result if result else [block]


def can_merge_textlines(
    block_a: TextBlock,
    block_b: TextBlock,
    font_size_tolerance: float = 1.3,
    angle_tolerance: float = 30.0,
    distance_tolerance: float = 2.0
) -> bool:
    """
    Check if two text blocks/lines can be merged.

    Args:
        block_a: First text block
        block_b: Second text block
        font_size_tolerance: Max ratio between font sizes
        angle_tolerance: Max angle difference in degrees
        distance_tolerance: Max distance as multiple of font size

    Returns:
        True if blocks can be merged
    """
    # Both must have valid font sizes
    if block_a.font_size <= 0 or block_b.font_size <= 0:
        return False

    # Check font size similarity
    size_ratio = max(block_a.font_size, block_b.font_size) / min(block_a.font_size, block_b.font_size)
    if size_ratio > font_size_tolerance:
        return False

    # Check orientation match
    if block_a.vertical != block_b.vertical:
        return False

    # Check angle similarity (using cosine of angle difference)
    vec_a = np.array(block_a.vec)
    vec_b = np.array(block_b.vec)
    if block_a.norm > 0 and block_b.norm > 0:
        cos_angle = np.dot(vec_a, vec_b) / (block_a.norm * block_b.norm)
        if cos_angle < np.cos(np.radians(angle_tolerance)):
            return False

    # Check distance between blocks
    center_a = np.array(block_a.center)
    center_b = np.array(block_b.center)
    distance = np.linalg.norm(center_a - center_b)
    avg_font_size = (block_a.font_size + block_b.font_size) / 2

    if distance > avg_font_size * distance_tolerance:
        return False

    return True


def merge_textlines(
    scattered: List[TextBlock],
    im_w: int,
    im_h: int
) -> List[TextBlock]:
    """
    Merge scattered text lines into coherent blocks.

    Args:
        scattered: List of single-line TextBlocks
        im_w: Image width
        im_h: Image height

    Returns:
        List of merged TextBlocks
    """
    if len(scattered) <= 1:
        return scattered

    # Examine each block first
    for block in scattered:
        examine_textblock(block, im_w, im_h, sort=False)

    # Greedy merging
    merged = []
    used = [False] * len(scattered)

    for i, block_a in enumerate(scattered):
        if used[i]:
            continue

        # Find all blocks that can merge with this one
        group = [block_a]
        used[i] = True

        for j, block_b in enumerate(scattered):
            if used[j]:
                continue

            # Check if any block in group can merge with block_b
            can_merge = any(can_merge_textlines(b, block_b) for b in group)
            if can_merge:
                group.append(block_b)
                used[j] = True

        # Create merged block
        if len(group) == 1:
            merged.append(group[0])
        else:
            all_lines = []
            for b in group:
                all_lines.extend(b.lines)

            all_points = np.vstack(all_lines)
            x1, y1 = all_points.min(axis=0)
            x2, y2 = all_points.max(axis=0)

            new_block = TextBlock(
                xyxy=[int(x1), int(y1), int(x2), int(y2)],
                lines=all_lines,
                vertical=group[0].vertical,
                confidence=np.mean([b.confidence for b in group]),
                merged=True,
            )
            examine_textblock(new_block, im_w, im_h, sort=True)
            merged.append(new_block)

    return merged


# ============================================================================
# Language Detection
# ============================================================================

def detect_language(block: TextBlock, mask: Optional[np.ndarray] = None) -> str:
    """
    Detect the language of text in a block.

    Uses heuristics based on:
    - Text orientation (vertical suggests Japanese)
    - Character density
    - Block aspect ratio

    Args:
        block: TextBlock to analyze
        mask: Optional segmentation mask

    Returns:
        Language code: 'ja', 'eng', or 'unknown'
    """
    # Vertical text is likely Japanese
    if block.vertical:
        return 'ja'

    # Check aspect ratio - very wide blocks are likely English
    if block.width > 0 and block.height > 0:
        aspect = block.width / block.height
        if aspect > 5:
            return 'eng'

    # Check line count and density
    if len(block.lines) > 0:
        avg_line_width = np.mean([
            line[:, 0].max() - line[:, 0].min()
            for line in block.lines
        ])
        # Japanese tends to have more square-ish characters
        if block.font_size > 0 and avg_line_width / block.font_size < 3:
            return 'ja'

    return 'unknown'


# ============================================================================
# Color Extraction
# ============================================================================

def extract_colors(
    block: TextBlock,
    image: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Extract foreground and background colors from a text block.

    Args:
        block: TextBlock to analyze
        image: BGR image
        mask: Optional binary mask (255 = text)

    Returns:
        Tuple of (foreground_rgb, background_rgb)
    """
    x1, y1, x2, y2 = block.xyxy

    # Clip to image bounds
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0), (255, 255, 255)

    # Extract region
    region = image[y1:y2, x1:x2]

    if mask is not None:
        mask_region = mask[y1:y2, x1:x2]
        text_pixels = region[mask_region > 127]
        bg_pixels = region[mask_region <= 127]
    else:
        # Use simple thresholding
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_pixels = region[binary < 127]
        bg_pixels = region[binary >= 127]

    # Compute median colors
    if len(text_pixels) > 0:
        fg_bgr = np.median(text_pixels, axis=0).astype(int)
        fg_rgb = (int(fg_bgr[2]), int(fg_bgr[1]), int(fg_bgr[0]))
    else:
        fg_rgb = (0, 0, 0)

    if len(bg_pixels) > 0:
        bg_bgr = np.median(bg_pixels, axis=0).astype(int)
        bg_rgb = (int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0]))
    else:
        bg_rgb = (255, 255, 255)

    return fg_rgb, bg_rgb


# ============================================================================
# Reading Order Sorting
# ============================================================================

def sort_textblocks(
    blocks: List[TextBlock],
    im_w: int,
    im_h: int,
    right_to_left: bool = True
) -> List[TextBlock]:
    """
    Sort text blocks by reading order.

    For manga (right_to_left=True):
    - Right column before left column
    - Top to bottom within columns

    For comics (right_to_left=False):
    - Left column before right column
    - Top to bottom within columns

    Args:
        blocks: List of TextBlocks
        im_w: Image width
        im_h: Image height
        right_to_left: True for manga reading order

    Returns:
        Sorted list of TextBlocks
    """
    if len(blocks) <= 1:
        return blocks

    # Compute sorting weight for each block
    # Weight = row_weight * 1000 + column_weight
    # This creates a grid-based ordering

    # Estimate row height (use median block height)
    heights = [b.height for b in blocks]
    row_height = np.median(heights) if heights else im_h / 10

    for block in blocks:
        cx, cy = block.center

        # Row index (top to bottom)
        row = int(cy / row_height)

        # Column position (normalized 0-1)
        col = cx / im_w if im_w > 0 else 0

        # For right-to-left, invert column
        if right_to_left:
            col = 1 - col

        # Combined weight: row is primary, column is secondary
        block._weight = row * 1000 + col * 100

    # Sort by weight
    return sorted(blocks, key=lambda b: b._weight)


# ============================================================================
# Expand Bounding Box (for English text)
# ============================================================================

def expand_bbox(
    block: TextBlock,
    im_w: int,
    im_h: int,
    expand_ratio: float = 0.1
) -> None:
    """
    Slightly expand bounding box for English text.

    English text often has descenders and ascenders that extend
    beyond the detected region.

    Args:
        block: TextBlock to expand (modified in-place)
        im_w: Image width
        im_h: Image height
        expand_ratio: Expansion ratio (0.1 = 10%)
    """
    if block.language != 'eng':
        return

    x1, y1, x2, y2 = block.xyxy
    w, h = x2 - x1, y2 - y1

    # Expand horizontally and vertically
    dx = int(w * expand_ratio / 2)
    dy = int(h * expand_ratio / 2)

    block.xyxy = [
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(im_w, x2 + dx),
        min(im_h, y2 + dy),
    ]


# ============================================================================
# Main Group Output Function
# ============================================================================

def group_output(
    blocks: List[Dict[str, Any]],
    lines: List[np.ndarray],
    im_w: int,
    im_h: int,
    mask: Optional[np.ndarray] = None,
    image: Optional[np.ndarray] = None,
    bbox_score_thresh: float = 0.4,
    mask_score_thresh: float = 0.1,
    sort_by_reading_order: bool = True,
    right_to_left: bool = True,
    extract_block_colors: bool = False,
) -> List[TextBlock]:
    """
    Assign detected text lines to text blocks based on spatial overlap.

    This function implements the complete post-processing pipeline:
    1. Assign lines to blocks based on overlap
    2. Handle scattered lines (unassigned)
    3. Examine each block (compute orientation, font size, etc.)
    4. Split blocks with large gaps (for vertical text)
    5. Merge scattered lines
    6. Detect language
    7. Sort by reading order
    8. Extract colors (optional)

    Args:
        blocks: Block detections from YOLO
                [{'bbox': [x1,y1,x2,y2], 'confidence': float}, ...]
        lines: Line polygons from DBNet as numpy arrays
               Each array has shape (N, 2) representing polygon points
        im_w: Image width in pixels
        im_h: Image height in pixels
        mask: Optional segmentation mask (binary, 255 = text)
        image: Optional BGR image for color extraction
        bbox_score_thresh: Minimum overlap ratio to assign line to block (0.4 = 40%)
        mask_score_thresh: Minimum mask coverage for scattered lines (0.1 = 10%)
        sort_by_reading_order: Whether to sort blocks by reading order
        right_to_left: Reading direction (True for manga)
        extract_block_colors: Whether to extract fg/bg colors

    Returns:
        List of TextBlock objects with full analysis
    """
    # Step 0: Handle edge cases
    if not blocks and not lines:
        return []

    if not blocks:
        # No blocks - each line becomes its own block
        result = []
        for line in lines:
            if len(line) < 3:
                continue
            bbox = polygon_to_bbox(line)
            block = TextBlock(xyxy=bbox, lines=[line], confidence=1.0)
            examine_textblock(block, im_w, im_h)
            result.append(block)
        if sort_by_reading_order:
            result = sort_textblocks(result, im_w, im_h, right_to_left)
        return result

    # Step 1: Create TextBlock objects from YOLO detections
    text_blocks = [
        TextBlock(
            xyxy=[int(x) for x in blk['bbox']],
            confidence=blk.get('confidence', 1.0)
        )
        for blk in blocks
    ]

    # Track scattered lines by orientation
    scattered_lines = {'hor': [], 'ver': []}

    # Step 2: Assign each line to best-matching block
    for line in lines:
        if len(line) < 3:
            continue

        # Get line bounding box
        bx1, bx2 = float(line[:, 0].min()), float(line[:, 0].max())
        by1, by2 = float(line[:, 1].min()), float(line[:, 1].max())
        line_bbox = [bx1, by1, bx2, by2]
        line_area = (by2 - by1) * (bx2 - bx1)

        if line_area <= 0:
            continue

        # Find best matching block
        best_score, best_idx = -1.0, -1
        for idx, blk in enumerate(text_blocks):
            score = intersection_area(blk.xyxy, line_bbox) / line_area
            if score > best_score:
                best_score = score
                best_idx = idx

        # Assign to block or mark as scattered
        if best_score > bbox_score_thresh and best_idx >= 0:
            text_blocks[best_idx].lines.append(line)
        else:
            # Check mask coverage if available
            if mask is not None:
                y1_idx = max(0, min(int(by1), mask.shape[0] - 1))
                y2_idx = max(0, min(int(by2), mask.shape[0]))
                x1_idx = max(0, min(int(bx1), mask.shape[1] - 1))
                x2_idx = max(0, min(int(bx2), mask.shape[1]))

                if y2_idx > y1_idx and x2_idx > x1_idx:
                    mask_region = mask[y1_idx:y2_idx, x1_idx:x2_idx]
                    mask_score = mask_region.mean() / 255.0 if mask_region.size > 0 else 0
                    if mask_score < mask_score_thresh:
                        continue

            # Determine orientation
            is_vertical = (by2 - by1) > (bx2 - bx1)
            key = 'ver' if is_vertical else 'hor'

            scattered_lines[key].append(TextBlock(
                xyxy=[int(bx1), int(by1), int(bx2), int(by2)],
                lines=[line],
                vertical=is_vertical,
                confidence=1.0
            ))

    # Step 3: Examine each block
    for block in text_blocks:
        if block.lines:
            examine_textblock(block, im_w, im_h, sort=True)

    # Step 4: Split blocks with large gaps
    split_blocks = []
    for block in text_blocks:
        if block.lines:
            split_result = split_textblock(block, im_w, im_h)
            split_blocks.extend(split_result)

    # Step 5: Merge scattered lines
    merged_hor = merge_textlines(scattered_lines['hor'], im_w, im_h)
    merged_ver = merge_textlines(scattered_lines['ver'], im_w, im_h)

    # Step 6: Combine all blocks
    final_blocks = split_blocks + merged_hor + merged_ver

    # Add blocks without lines but with high confidence
    for blk in text_blocks:
        if not blk.lines and blk.confidence > 0.7 and blk not in final_blocks:
            final_blocks.append(blk)

    # Step 7: Detect language and expand English blocks
    for block in final_blocks:
        block.language = detect_language(block, mask)
        expand_bbox(block, im_w, im_h)

    # Step 8: Extract colors if requested
    if extract_block_colors and image is not None:
        for block in final_blocks:
            fg, bg = extract_colors(block, image, mask)
            block.fg_colors = fg
            block.bg_colors = bg

    # Step 9: Sort by reading order
    if sort_by_reading_order:
        final_blocks = sort_textblocks(final_blocks, im_w, im_h, right_to_left)

    return final_blocks
