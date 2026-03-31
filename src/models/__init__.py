"""Models package for comic text detection."""

from .backbone import (
    YOLOv11Backbone,
    YOLOv11BackboneSimple,
    create_backbone,
)
from .heads import (
    UnetHead,
    DBHead,
    TextDetector,
    TextDetectorInference,
    init_weights,
    TEXTDET_MASK,
    TEXTDET_DET,
    TEXTDET_INFERENCE,
    TEXTDET_UNIFIED,
)
from .detector import (
    TextDetBase,
    TextDetBaseDNN,
    create_text_detector,
    load_text_detector,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    # Backbone
    'YOLOv11Backbone',
    'YOLOv11BackboneSimple',
    'create_backbone',
    # Heads
    'UnetHead',
    'DBHead',
    'TextDetector',
    'TextDetectorInference',
    'init_weights',
    # Constants
    'TEXTDET_MASK',
    'TEXTDET_DET',
    'TEXTDET_INFERENCE',
    'TEXTDET_UNIFIED',
    # Full detector
    'TextDetBase',
    'TextDetBaseDNN',
    'create_text_detector',
    'load_text_detector',
    'save_checkpoint',
    'load_checkpoint',
]
