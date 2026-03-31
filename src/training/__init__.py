"""Training package for comic text detection."""

from .trainer import (
    Trainer,
    TrainingConfig,
    EMA,
    EarlyStopping,
    GradientCheckpointedModel,
    apply_gradient_checkpointing,
)
from .losses import (
    BinaryDiceLoss,
    FocalLoss,
    BalancedBCELoss,
    MaskL1Loss,
    DBLoss,
    SegmentationLoss,
    UnifiedLoss,
    create_loss,
)

__all__ = [
    'Trainer',
    'TrainingConfig',
    'EMA',
    'EarlyStopping',
    'GradientCheckpointedModel',
    'apply_gradient_checkpointing',
    'BinaryDiceLoss',
    'FocalLoss',
    'BalancedBCELoss',
    'MaskL1Loss',
    'DBLoss',
    'SegmentationLoss',
    'UnifiedLoss',
    'create_loss',
]
