"""
Full Comic Text Detector Model

Combines YOLOv11 backbone with UnetHead and DBHead for complete
text detection and segmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

from .backbone import create_backbone
from .heads import UnetHead, DBHead, TextDetector, TextDetectorInference, init_weights
from .heads import TEXTDET_MASK, TEXTDET_DET, TEXTDET_INFERENCE


def _normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip torch.compile() wrapper prefixes from state dict keys."""
    if any(key.startswith('_orig_mod.') for key in state_dict):
        return {key.replace('_orig_mod.', '', 1): value for key, value in state_dict.items()}
    return state_dict


def _extract_sub_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Extract and de-prefix a submodule state dict."""
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


def _resolve_backbone_name(checkpoint: Dict[str, Any], default: str = 'yolo26s.pt') -> str:
    """Resolve backbone name from checkpoint metadata if present."""
    for source in (checkpoint.get('config'), checkpoint.get('hyp')):
        if isinstance(source, dict):
            name = source.get('backbone_name') or source.get('backbone')
            if name:
                return name
    return default


def get_base_det_models(
    model_path: str | Path,
    device: str = 'cpu',
    half: bool = False
) -> Tuple[nn.Module, UnetHead, DBHead]:
    """
    Load pretrained detector models from checkpoint.

    Args:
        model_path: Path to combined model checkpoint
        device: Device to load models on
        half: Use FP16 precision

    Returns:
        Tuple of (backbone, segmentation_head, db_head)
    """
    checkpoint = torch.load(model_path, map_location=device)

    # Create backbone and heads
    backbone_name = _resolve_backbone_name(checkpoint)
    backbone = create_backbone(model_name=backbone_name, pretrained=False, freeze=True)
    text_seg = UnetHead()
    text_det = DBHead(64)

    if 'weights' in checkpoint:
        weights = checkpoint['weights']
        if 'backbone' in weights:
            backbone.load_state_dict(weights['backbone'])
        if 'seg_net' in weights:
            text_seg.load_state_dict(weights['seg_net'])
        elif 'text_seg' in weights:
            text_seg.load_state_dict(weights['text_seg'])
        if 'dbnet' in weights:
            text_det.load_state_dict(weights['dbnet'])
    elif 'model_state_dict' in checkpoint:
        state_dict = _normalize_state_dict(checkpoint['model_state_dict'])
        backbone_state = _extract_sub_state_dict(state_dict, 'backbone.')
        seg_state = _extract_sub_state_dict(state_dict, 'seg_net.')
        db_state = _extract_sub_state_dict(state_dict, 'dbnet.')
        if backbone_state:
            backbone.load_state_dict(backbone_state)
        if seg_state:
            text_seg.load_state_dict(seg_state)
        if db_state:
            text_det.load_state_dict(db_state)
    else:
        if 'backbone' in checkpoint:
            backbone.load_state_dict(checkpoint['backbone'])
        if 'text_seg' in checkpoint:
            text_seg.load_state_dict(checkpoint['text_seg'])
        if 'text_det' in checkpoint:
            text_det.load_state_dict(checkpoint['text_det'])

    backbone = backbone.to(device).eval()
    text_seg = text_seg.to(device).eval()
    text_det = text_det.to(device).eval()

    if half:
        backbone = backbone.half()
        text_seg = text_seg.half()
        text_det = text_det.half()

    return backbone, text_seg, text_det


class TextDetBase(nn.Module):
    """
    Base text detector for inference with PyTorch backend.

    Combines backbone, segmentation, and text line detection
    into a single forward pass.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = 'cpu',
        half: bool = False,
        fuse: bool = False
    ):
        super().__init__()

        self.backbone, self.text_seg, self.text_det = get_base_det_models(
            model_path, device, half
        )

        if fuse:
            self.fuse()

    def fuse(self):
        """Fuse Conv2d and BatchNorm2d layers for faster inference."""
        def _fuse_conv_bn(m: nn.Module):
            for name, child in m.named_children():
                if isinstance(child, nn.Sequential):
                    _fuse_conv_bn(child)
                # Fuse conv+bn pairs
                if hasattr(child, 'conv') and hasattr(child, 'bn'):
                    child.conv = self._fuse_conv_and_bn(child.conv, child.bn)
                    delattr(child, 'bn')
                    if hasattr(child, 'forward_fuse'):
                        child.forward = child.forward_fuse

        _fuse_conv_bn(self.text_seg)
        _fuse_conv_bn(self.text_det)

    @staticmethod
    def _fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        """Fuse Conv2d and BatchNorm2d into a single Conv2d."""
        fusedconv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True
        ).requires_grad_(False).to(conv.weight.device)

        # Fuse weights
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # Fuse bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for complete detection.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Tuple of:
                - mask: Segmentation mask (B, 1, H, W)
                - features: Detection features
                - lines: Text line maps
        """
        # Extract features
        features = self.backbone(x)

        # Segmentation with intermediate features
        mask, seg_features = self.text_seg(*features, forward_mode=TEXTDET_INFERENCE)

        # Text line detection
        lines = self.text_det(*seg_features, step_eval=False)

        return mask, lines


class TextDetBaseDNN:
    """
    ONNX-based text detector using OpenCV DNN backend.

    Faster inference without PyTorch dependency.
    """

    def __init__(self, input_size: int, model_path: str | Path):
        """
        Initialize ONNX detector.

        Args:
            input_size: Input image size (assumes square input)
            model_path: Path to ONNX model file
        """
        self.input_size = input_size
        self.model = cv2.dnn.readNetFromONNX(str(model_path))
        self.uoln = self.model.getUnconnectedOutLayersNames()

    def __call__(
        self,
        im_in: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on input image.

        Args:
            im_in: Input image (H, W, 3) BGR format

        Returns:
            Tuple of (blocks, mask, lines_map)
        """
        blob = cv2.dnn.blobFromImage(
            im_in,
            scalefactor=1 / 255.0,
            size=(self.input_size, self.input_size)
        )
        self.model.setInput(blob)
        outputs = self.model.forward(self.uoln)

        # Parse outputs based on expected structure
        if len(outputs) == 3:
            blks, mask, lines_map = outputs
        else:
            blks, mask, lines_map = None, outputs[0], outputs[1]

        return blks, mask, lines_map


def create_text_detector(
    backbone_name: str = 'yolo26s.pt',
    pretrained_backbone: bool = True,
    freeze_backbone: bool = True,
    use_cib: bool = True,
    use_bifpn: bool = False,
    use_rank_guided: bool = False,
    device: str = 'cpu'
) -> TextDetector:
    """
    Factory function to create a text detector for training.

    Args:
        backbone_name: YOLO model variant. Supported:
            - YOLO26: 'yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt', etc. (recommended)
            - YOLOv10: 'yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt', etc.
        pretrained_backbone: Load pretrained backbone weights
        freeze_backbone: Freeze backbone during training
        use_cib: Use CIB blocks (YOLOv10 style) for efficiency
        use_bifpn: Enable EfficientBiFPN for bidirectional feature fusion
        use_rank_guided: Enable RankGuidedDecoder for compute-efficient decoding
        device: Device to create model on

    Returns:
        TextDetector instance ready for training
    """
    backbone = create_backbone(
        model_name=backbone_name,
        pretrained=pretrained_backbone,
        freeze=freeze_backbone,
        simple=True
    )

    detector = TextDetector(
        backbone=backbone,
        use_cib=use_cib,
        freeze_backbone=freeze_backbone,
        use_bifpn=use_bifpn,
        use_rank_guided=use_rank_guided
    )

    return detector.to(device)


def load_text_detector(
    checkpoint_path: str | Path,
    device: str = 'cpu',
    half: bool = False
) -> TextDetBase:
    """
    Load a trained text detector from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        half: Use FP16 precision

    Returns:
        TextDetBase instance ready for inference
    """
    detector = TextDetBase(checkpoint_path, device=device, half=half)
    return detector


def save_checkpoint(
    detector: TextDetector,
    save_path: str | Path,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    best_f1: float = 0.0,
    best_val_loss: float = float('inf'),
    run_id: Optional[str] = None,
    hyp: Optional[Dict] = None
):
    """
    Save training checkpoint.

    Args:
        detector: TextDetector model
        save_path: Path to save checkpoint
        epoch: Current training epoch
        optimizer: Optimizer state
        scheduler: LR scheduler state
        best_f1: Best F1 score achieved
        best_val_loss: Best validation loss
        run_id: WandB run ID for resumption
        hyp: Training hyperparameters
    """
    checkpoint = {
        'epoch': epoch,
        'best_f1': best_f1,
        'best_val_loss': best_val_loss,
        'run_id': run_id,
        'hyp': hyp,
        'weights': {
            'backbone': detector.backbone.state_dict(),
            'seg_net': detector.seg_net.state_dict(),
        }
    }

    if detector.dbnet is not None:
        checkpoint['weights']['dbnet'] = detector.dbnet.state_dict()

    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()

    torch.save(checkpoint, save_path)


def load_checkpoint(
    detector: TextDetector,
    checkpoint_path: str | Path,
    device: str = 'cpu',
    load_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        detector: TextDetector model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        load_optimizer: Whether to load optimizer state
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into

    Returns:
        Dictionary with checkpoint metadata (epoch, best_f1, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    if 'weights' in checkpoint:
        if 'backbone' in checkpoint['weights']:
            detector.backbone.load_state_dict(checkpoint['weights']['backbone'])
        detector.seg_net.load_state_dict(checkpoint['weights']['seg_net'])
        if 'dbnet' in checkpoint['weights'] and detector.dbnet is not None:
            detector.dbnet.load_state_dict(checkpoint['weights']['dbnet'])
    else:
        # Legacy format
        detector.seg_net.load_state_dict(checkpoint.get('text_seg', {}))

    # Load optimizer and scheduler
    if load_optimizer and optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if load_optimizer and scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'best_f1': checkpoint.get('best_f1', 0.0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'run_id': checkpoint.get('run_id'),
        'hyp': checkpoint.get('hyp'),
    }


# Create package __init__.py
__all__ = [
    'TextDetector',
    'TextDetectorInference',
    'TextDetBase',
    'TextDetBaseDNN',
    'UnetHead',
    'DBHead',
    'create_text_detector',
    'load_text_detector',
    'save_checkpoint',
    'load_checkpoint',
    'create_backbone',
    'init_weights',
    'TEXTDET_MASK',
    'TEXTDET_DET',
    'TEXTDET_INFERENCE',
]
