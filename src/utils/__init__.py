"""Utility functions for comic text detection."""

from .textblock import (
    TextBlock,
    group_output,
    intersection_area,
    examine_textblock,
    split_textblock,
    merge_textlines,
    can_merge_textlines,
    sort_textblocks,
    detect_language,
    extract_colors,
    expand_bbox,
    polygon_to_bbox,
    get_mini_boxes,
    rotate_points,
)

__all__ = [
    'TextBlock',
    'group_output',
    'intersection_area',
    'examine_textblock',
    'split_textblock',
    'merge_textlines',
    'can_merge_textlines',
    'sort_textblocks',
    'detect_language',
    'extract_colors',
    'expand_bbox',
    'polygon_to_bbox',
    'get_mini_boxes',
    'rotate_points',
]
