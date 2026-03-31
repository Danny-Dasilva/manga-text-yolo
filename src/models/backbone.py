"""
YOLO Backbone for Comic Text Detection (supports YOLOv10 and YOLO26)

This module provides a backbone wrapper that extracts multi-scale features
from YOLO models for use with UnetHead and DBHead detection heads.

Supported architectures:
- YOLO26 (recommended): NMS-free, DFL-free, 43% faster CPU inference
- YOLOv10: NMS-free one-to-one assignment

YOLO26 advantages:
- NMS-free end-to-end detection
- DFL-free (simpler head, smaller model)
- C3k2 + C2PSA architecture
- Better CPU/edge performance

YOLOv10 advantages:
- Spatial-channel decoupled downsampling (SCDown)
- Compact Inverted Blocks (CIB) for efficiency
- Partial Self-Attention (PSA) for improved accuracy
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from pathlib import Path
import re
from ultralytics import YOLO


# Channel dimensions for YOLOv10 variants at backbone output layers
# These are the actual output channels from the backbone after C2f/PSA blocks
# Layer mapping: P2=Layer2 (stride 4), P3=Layer4 (stride 8), P4=Layer6 (stride 16), P5=Layer10 (stride 32)
CHANNEL_CONFIGS_V10 = {
    'yolov10n': {'p2': 64,   'p3': 128,  'p4': 256,  'p5': 512},
    'yolov10s': {'p2': 128,  'p3': 256,  'p4': 512,  'p5': 1024},
    'yolov10m': {'p2': 192,  'p3': 384,  'p4': 576,  'p5': 1152},
    'yolov10l': {'p2': 256,  'p3': 512,  'p4': 512,  'p5': 1024},
    'yolov10x': {'p2': 320,  'p3': 640,  'p4': 640,  'p5': 1280},
}

# Channel dimensions for YOLO26 variants
# YOLO26 uses C3k2 + C2PSA architecture with different channel scaling
# Layer mapping: P2=Layer2 (stride 4), P3=Layer4 (stride 8), P4=Layer6 (stride 16), P5=Layer10 (stride 32)
CHANNEL_CONFIGS_V26 = {
    'yolo26n': {'p2': 64,   'p3': 128,  'p4': 128,  'p5': 256},
    'yolo26s': {'p2': 128,  'p3': 256,  'p4': 256,  'p5': 512},
    'yolo26m': {'p2': 256,  'p3': 512,  'p4': 512,  'p5': 512},
    'yolo26l': {'p2': 256,  'p3': 512,  'p4': 512,  'p5': 512},
    'yolo26x': {'p2': 384,  'p3': 768,  'p4': 768,  'p5': 768},
}


class YOLOv10Backbone(nn.Module):
    """
    YOLOv10 backbone for feature extraction.

    Extracts features at multiple scales (P2, P3, P4, P5) from YOLOv10
    and projects them to match the expected channel dimensions for
    the segmentation and detection heads.

    Feature scales (for 1024x1024 input):
        - P2: stride 4  -> 256x256 (early features)
        - P3: stride 8  -> 128x128
        - P4: stride 16 -> 64x64
        - P5: stride 32 -> 32x32

    We also add an additional downsampled feature at stride 64 (16x16)
    to match the 5-scale output of the original architecture.
    """

    # Channel dimensions for YOLO variants (merged v10 + v26)
    CHANNEL_CONFIGS = {**CHANNEL_CONFIGS_V10, **CHANNEL_CONFIGS_V26}

    # Expected output channels to match original architecture
    OUTPUT_CHANNELS = {
        'f256': 64,   # stride 4 (1/4 resolution)
        'f128': 128,  # stride 8 (P3 level)
        'f64': 256,   # stride 16 (P4 level)
        'f32': 512,   # stride 32 (P5 level)
        'f16': 512,   # stride 64 (extra downsampled)
    }

    def __init__(
        self,
        model_name: str = 'yolov10s.pt',
        pretrained: bool = True,
        freeze_backbone: bool = True,
        freeze_projections: bool = False
    ):
        """
        Initialize YOLOv10 backbone.

        Args:
            model_name: Name of the YOLO model to use (e.g., 'yolov10s.pt', 'yolov10m.pt')
            pretrained: Whether to load pretrained weights
            freeze_backbone: Whether to freeze backbone weights during training
            freeze_projections: Whether to freeze projection layers (prevents divergence
                               during multi-phase training)
        """
        super().__init__()

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.freeze_projections_flag = freeze_projections

        # Determine model variant for channel configuration
        variant = self._get_variant(model_name)
        self.in_channels = self.CHANNEL_CONFIGS.get(variant, self.CHANNEL_CONFIGS['yolov10s'])

        # Load YOLOv10 model (avoid registering the wrapper as a submodule)
        if pretrained:
            yolo = YOLO(model_name)
        else:
            # Load architecture without pretrained weights
            yolo = YOLO(model_name.replace('.pt', '.yaml'))

        # Get the underlying PyTorch model
        self.backbone = yolo.model.model

        # Create early feature extraction (stride 4)
        # Using first few layers of the backbone
        self.early_layers = nn.ModuleList()

        # Projection layers to match expected output channels
        self.proj_p2 = nn.Sequential(
            nn.Conv2d(self.in_channels['p2'], self.OUTPUT_CHANNELS['f256'], 1, bias=False),
            nn.BatchNorm2d(self.OUTPUT_CHANNELS['f256']),
            nn.SiLU(inplace=True)
        )

        self.proj_p3 = nn.Sequential(
            nn.Conv2d(self.in_channels['p3'], self.OUTPUT_CHANNELS['f128'], 1, bias=False),
            nn.BatchNorm2d(self.OUTPUT_CHANNELS['f128']),
            nn.SiLU(inplace=True)
        )

        self.proj_p4 = nn.Sequential(
            nn.Conv2d(self.in_channels['p4'], self.OUTPUT_CHANNELS['f64'], 1, bias=False),
            nn.BatchNorm2d(self.OUTPUT_CHANNELS['f64']),
            nn.SiLU(inplace=True)
        )

        self.proj_p5 = nn.Sequential(
            nn.Conv2d(self.in_channels['p5'], self.OUTPUT_CHANNELS['f32'], 1, bias=False),
            nn.BatchNorm2d(self.OUTPUT_CHANNELS['f32']),
            nn.SiLU(inplace=True)
        )

        # Extra downsampling for stride 64 features
        self.downsample = nn.Sequential(
            nn.Conv2d(self.in_channels['p5'], self.OUTPUT_CHANNELS['f16'], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.OUTPUT_CHANNELS['f16']),
            nn.SiLU(inplace=True)
        )

        if freeze_backbone:
            self._freeze_backbone()

        if freeze_projections:
            self._freeze_projections()

    def _get_variant(self, model_name: str) -> str:
        """Extract model variant from model name."""
        stem = Path(model_name).stem.lower()
        # Try YOLO26 first
        match = re.search(r"yolo26([nslmx])", stem)
        if match:
            return f"yolo26{match.group(1)}"
        # Try YOLOv10
        match = re.search(r"yolov10([nslmx])", stem)
        if match:
            return f"yolov10{match.group(1)}"
        return 'yolov10s'  # default

    def _freeze_backbone(self):
        """Freeze backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _freeze_projections(self):
        """Freeze projection layers to prevent divergence during multi-phase training."""
        for name, proj in [
            ('proj_p2', self.proj_p2),
            ('proj_p3', self.proj_p3),
            ('proj_p4', self.proj_p4),
            ('proj_p5', self.proj_p5),
            ('downsample', self.downsample),
        ]:
            for param in proj.parameters():
                param.requires_grad = False
            print(f"  Frozen: {name}")

    def unfreeze_backbone(self):
        """Unfreeze backbone weights for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False

    def unfreeze_projections(self):
        """Unfreeze projection layers."""
        for proj in [self.proj_p2, self.proj_p3, self.proj_p4, self.proj_p5, self.downsample]:
            for param in proj.parameters():
                param.requires_grad = True
        self.freeze_projections_flag = False

    def _extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Extract multi-scale features from YOLOv10 backbone.

        Returns features at 5 scales matching the original output:
            - f256: stride 4 (256x256 for 1024 input)
            - f128: stride 8 (128x128 for 1024 input)
            - f64: stride 16 (64x64 for 1024 input)
            - f32: stride 32 (32x32 for 1024 input)
            - f16: stride 64 (16x16 for 1024 input)
        """
        features = {}

        # Run through backbone layers and collect features
        y = []
        for i, m in enumerate(self.backbone):
            if hasattr(m, 'f'):
                if m.f != -1:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)

            # Collect features at specific layers for YOLOv10 architecture
            # YOLOv10 layer structure:
            #   Layer 0: Conv (stride 2)
            #   Layer 1: Conv (stride 4)
            #   Layer 2: C2f (stride 4) -> P2
            #   Layer 3: Conv (stride 8)
            #   Layer 4: C2f (stride 8) -> P3
            #   Layer 5: SCDown (stride 16)
            #   Layer 6: C2f (stride 16) -> P4
            #   Layer 7: SCDown (stride 32)
            #   Layer 8: C2fCIB (stride 32)
            #   Layer 9: SPPF (stride 32)
            #   Layer 10: PSA (stride 32) -> P5
            if i == 2:  # P2: After C2f at stride 4 (H/4, W/4)
                features['p2'] = x
            elif i == 4:  # P3: After C2f at stride 8 (H/8, W/8)
                features['p3'] = x
            elif i == 6:  # P4: After C2f at stride 16 (H/16, W/16)
                features['p4'] = x
            elif i == 10:  # P5: After PSA at stride 32 (H/32, W/32)
                features['p5'] = x

        return features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass extracting multi-scale features.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Tuple of 5 feature tensors:
                - f256: (B, 64, H/4, W/4)
                - f128: (B, 128, H/8, W/8)
                - f64: (B, 256, H/16, W/16)
                - f32: (B, 512, H/32, W/32)
                - f16: (B, 512, H/64, W/64)
        """
        if self.freeze_backbone:
            with torch.no_grad():
                features = self._extract_features(x)
        else:
            features = self._extract_features(x)

        # Project features to expected channel dimensions
        f256 = self.proj_p2(features['p2'])
        f128 = self.proj_p3(features['p3'])
        f64 = self.proj_p4(features['p4'])
        f32 = self.proj_p5(features['p5'])
        f16 = self.downsample(features['p5'])

        return f256, f128, f64, f32, f16


class YOLOv10BackboneSimple(nn.Module):
    """
    Simplified YOLOv10 backbone using ultralytics' built-in feature extraction.

    This version uses the ultralytics API more directly and is recommended
    for simpler integration with the existing heads.
    """

    def __init__(
        self,
        model_name: str = 'yolov10s.pt',
        pretrained: bool = True,
        freeze_backbone: bool = True,
        freeze_projections: bool = False
    ):
        """
        Initialize YOLOv10 backbone.

        Args:
            model_name: Name of the YOLO model to use (e.g., 'yolov10s.pt', 'yolov10m.pt')
            pretrained: Whether to load pretrained weights
            freeze_backbone: Whether to freeze backbone weights during training
            freeze_projections: Whether to freeze projection layers (prevents divergence
                               during multi-phase training)
        """
        super().__init__()

        self.freeze_backbone = freeze_backbone
        self.freeze_projections_flag = freeze_projections

        # Load YOLOv10 model (avoid registering the wrapper as a submodule)
        yolo = YOLO(model_name if pretrained else model_name.replace('.pt', '.yaml'))
        self.model = yolo.model

        # Get actual channel dimensions by running a test forward pass
        with torch.no_grad():
            test_input = torch.zeros(1, 3, 640, 640)
            self._detect_channels(test_input)

        # Projection layers (created after channel detection)
        self._create_projections()

        # Extra downsampling
        self.downsample = nn.Sequential(
            nn.Conv2d(self.p5_channels, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True)
        )

        if freeze_backbone:
            self._freeze_backbone()

        if freeze_projections:
            self._freeze_projections()

    def _detect_channels(self, x: torch.Tensor):
        """Detect channel dimensions from a test forward pass."""
        y = []
        for i, m in enumerate(self.model.model):
            if hasattr(m, 'f') and m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)

            # YOLOv10 layer indices for feature extraction
            # P2=Layer2 (stride 4), P3=Layer4 (stride 8), P4=Layer6 (stride 16), P5=Layer10 (stride 32)
            if i == 2:
                self.p2_channels = x.shape[1]
            elif i == 4:
                self.p3_channels = x.shape[1]
            elif i == 6:
                self.p4_channels = x.shape[1]
            elif i == 10:
                self.p5_channels = x.shape[1]

    def _create_projections(self):
        """Create projection layers based on detected channels."""
        self.proj_p2 = nn.Sequential(
            nn.Conv2d(self.p2_channels, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )

        self.proj_p3 = nn.Sequential(
            nn.Conv2d(self.p3_channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )

        self.proj_p4 = nn.Sequential(
            nn.Conv2d(self.p4_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )

        self.proj_p5 = nn.Sequential(
            nn.Conv2d(self.p5_channels, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True)
        )

    def _freeze_backbone(self):
        """Freeze YOLO backbone weights."""
        for param in self.model.parameters():
            param.requires_grad = False

    def _freeze_projections(self):
        """Freeze projection layers to prevent divergence during multi-phase training."""
        for name, proj in [
            ('proj_p2', self.proj_p2),
            ('proj_p3', self.proj_p3),
            ('proj_p4', self.proj_p4),
            ('proj_p5', self.proj_p5),
            ('downsample', self.downsample),
        ]:
            for param in proj.parameters():
                param.requires_grad = False
            print(f"  Frozen: {name}")

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.freeze_backbone = False

    def unfreeze_projections(self):
        """Unfreeze projection layers."""
        for proj in [self.proj_p2, self.proj_p3, self.proj_p4, self.proj_p5, self.downsample]:
            for param in proj.parameters():
                param.requires_grad = True
        self.freeze_projections_flag = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Extract features at 5 scales.

        Returns:
            Tuple[f256, f128, f64, f32, f16] matching original architecture expectations
        """
        features = {}
        y = []

        # Forward through backbone
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            for i, m in enumerate(self.model.model):
                if hasattr(m, 'f') and m.f != -1:
                    x_in = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                else:
                    x_in = x
                x = m(x_in)
                y.append(x)

                # YOLOv10 layer indices for feature extraction
                # P2=Layer2 (stride 4), P3=Layer4 (stride 8), P4=Layer6 (stride 16), P5=Layer10 (stride 32)
                if i == 2:
                    features['p2'] = x
                elif i == 4:
                    features['p3'] = x
                elif i == 6:
                    features['p4'] = x
                elif i == 10:
                    features['p5'] = x

        # Project to expected dimensions
        f256 = self.proj_p2(features['p2'])
        f128 = self.proj_p3(features['p3'])
        f64 = self.proj_p4(features['p4'])
        f32 = self.proj_p5(features['p5'])
        f16 = self.downsample(features['p5'])

        return f256, f128, f64, f32, f16


def create_backbone(
    model_name: str = 'yolo26s.pt',
    pretrained: bool = True,
    freeze: bool = True,
    freeze_projections: bool = False,
    simple: bool = True
) -> nn.Module:
    """
    Factory function to create YOLO backbone (supports YOLOv10 and YOLO26).

    Args:
        model_name: YOLO model variant. Supported:
            - YOLO26: 'yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt', 'yolo26l.pt', 'yolo26x.pt'
            - YOLOv10: 'yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt', 'yolov10l.pt', 'yolov10x.pt'
        pretrained: Load pretrained weights
        freeze: Freeze backbone during training
        freeze_projections: Freeze projection layers (use during detection phase
                           to prevent divergence from segmentation training)
        simple: Use simplified backbone with auto channel detection (recommended)

    Returns:
        YOLO backbone module with 5-scale feature output
    """
    if simple:
        return YOLOv10BackboneSimple(model_name, pretrained, freeze, freeze_projections)
    else:
        return YOLOv10Backbone(model_name, pretrained, freeze, freeze_projections)


# Backward compatibility aliases
YOLOv11Backbone = YOLOv10Backbone
YOLOv11BackboneSimple = YOLOv10BackboneSimple
YOLO26Backbone = YOLOv10Backbone
YOLO26BackboneSimple = YOLOv10BackboneSimple


if __name__ == '__main__':
    import sys

    # Test with specified model or default to yolo26s
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'yolo26s.pt'
    print(f"Testing backbone with {model_name}")
    backbone = create_backbone(model_name, pretrained=True, freeze=True)

    # Test forward pass
    x = torch.randn(1, 3, 1024, 1024)
    features = backbone(x)

    print("Feature shapes:")
    for i, f in enumerate(features):
        print(f"  f{i}: {f.shape}")
