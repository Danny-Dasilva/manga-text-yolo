"""
Detection Heads for Comic Text Detection

UnetHead: Text region segmentation
DBHead: Text line detection using Differentiable Binarization
BlockDetector: YOLO-style block detection with one-to-one assignment (NMS-free)
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Literal


# Forward mode constants
TEXTDET_MASK = 0       # Segmentation mask output
TEXTDET_DET = 1        # Detection (intermediate features)
TEXTDET_INFERENCE = 2  # Full inference (mask + features)
TEXTDET_BLOCK = 3      # Block detection training
TEXTDET_UNIFIED = 4    # Unified training (all heads jointly)


class DWConv(nn.Module):
    """Depthwise convolution."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=min(c1, c2), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CIB(nn.Module):
    """Compact Inverted Block - more efficient than standard bottleneck.

    Uses inverted residual: expand -> depthwise -> compress
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1)            # Pointwise expand
        self.cv2 = DWConv(c_, c_, 3)          # Depthwise 3x3
        self.cv3 = Conv(c_, c2, 1, act=False) # Pointwise compress
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class C2fCIB(nn.Module):
    """C2f with CIB blocks instead of standard Bottleneck."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SCDown(nn.Module):
    """Spatial-Channel Decoupled Downsampling.

    Separates spatial reduction from channel adjustment:
    1. Depthwise conv for spatial reduction (stride 2)
    2. Pointwise conv for channel adjustment
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2):
        super().__init__()
        self.cv1 = DWConv(c1, c1, k, s)  # Spatial reduction
        self.cv2 = Conv(c1, c2, 1)       # Channel adjustment

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    CSP Bottleneck with 2 convolutions - YOLOv8/v11 style.

    More efficient than C3 blocks used in YOLOv5.
    """
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Conv(nn.Module):
    """Standard convolution with batch normalization and activation."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None, g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p is not None else k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without batch norm (for fused inference)."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple = ((3, 3), (3, 3)), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0][0], 1)
        self.cv2 = Conv(c_, c2, k[1][0], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class PSA(nn.Module):
    """Partial Self-Attention - efficient attention on channel subset.

    Only applies attention to a portion of channels (default 50%),
    leaving the rest as skip connections. More efficient than full attention.
    """
    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4):
        super().__init__()
        self.c = c
        self.c_attn = int(c * attn_ratio)
        self.c_skip = c - self.c_attn

        self.num_heads = num_heads
        self.head_dim = self.c_attn // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection for attention portion only
        self.qkv = nn.Conv2d(self.c_attn, self.c_attn * 3, 1, bias=False)
        self.proj = nn.Conv2d(self.c_attn, self.c_attn, 1, bias=False)
        self.norm = nn.BatchNorm2d(self.c_attn)

    def forward(self, x):
        B, C, H, W = x.shape

        # Split channels into attention and skip portions
        x_attn, x_skip = x.split([self.c_attn, self.c_skip], dim=1)

        # Multi-head self-attention on attention portion
        qkv = self.qkv(x_attn).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)  # Each: [B, heads, head_dim, H*W]

        # Attention: Q @ K^T / sqrt(d)
        attn = (q.transpose(-2, -1) @ k) * self.scale  # [B, heads, H*W, H*W]
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        x_attn = (v @ attn.transpose(-2, -1))  # [B, heads, head_dim, H*W]
        x_attn = x_attn.reshape(B, self.c_attn, H, W)
        x_attn = self.norm(self.proj(x_attn))

        # Concatenate attention output with skip portion
        return torch.cat([x_attn, x_skip], dim=1)


class LargeKernelConv(nn.Module):
    """7x7 depthwise separable convolution for larger receptive field.

    Useful for text detection where larger context improves accuracy.
    More parameter-efficient than standard 7x7 conv.
    """
    def __init__(self, c1: int, c2: int, k: int = 7):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, padding=k // 2, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class EfficientBiFPN(nn.Module):
    """Lightweight bidirectional feature pyramid with learnable fusion weights."""

    def __init__(self, channels: List[int]):
        super().__init__()
        self.num_levels = len(channels)
        self.eps = 1e-4

        # Learnable fusion weights (softmax-normalized)
        self.weights_td = nn.ParameterList([
            nn.Parameter(torch.ones(2)) for _ in range(self.num_levels - 1)
        ])
        self.weights_bu = nn.ParameterList([
            nn.Parameter(torch.ones(2)) for _ in range(self.num_levels - 1)
        ])

        # Lateral connections (1x1 convs)
        self.lateral_convs = nn.ModuleList([
            Conv(c, c, 1) for c in channels
        ])

        # Fusion convs
        self.td_convs = nn.ModuleList([
            Conv(channels[i], channels[i], 3) for i in range(self.num_levels - 1)
        ])
        self.bu_convs = nn.ModuleList([
            Conv(channels[i+1], channels[i+1], 3) for i in range(self.num_levels - 1)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Apply lateral convolutions
        features = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down path with weighted fusion
        for i in range(self.num_levels - 1, 0, -1):
            w = F.softmax(self.weights_td[i-1], dim=0)
            td_feature = F.interpolate(features[i], size=features[i-1].shape[2:], mode='nearest')
            features[i-1] = self.td_convs[i-1](w[0] * features[i-1] + w[1] * td_feature)

        # Bottom-up path with weighted fusion
        for i in range(self.num_levels - 1):
            w = F.softmax(self.weights_bu[i], dim=0)
            bu_feature = F.max_pool2d(features[i], 2)
            if bu_feature.shape[2:] != features[i+1].shape[2:]:
                bu_feature = F.interpolate(bu_feature, size=features[i+1].shape[2:], mode='nearest')
            features[i+1] = self.bu_convs[i](w[0] * features[i+1] + w[1] * bu_feature)

        return features


class RankGuidedDecoder(nn.Module):
    """Decoder with compute allocated by layer importance (YOLOv10 design)."""

    # Rank-guided block counts: more blocks at P5 (semantic), fewer at P2 (resolution)
    BLOCK_COUNTS = {'p5': 3, 'p4': 2, 'p3': 1, 'p2': 0}

    def __init__(self, use_cib: bool = True, use_psa: bool = True):
        super().__init__()

        BlockClass = C2fCIB if use_cib else C2f

        # P5: 3 blocks + PSA (high semantic, low resolution)
        self.p5_blocks = nn.Sequential(*[BlockClass(512, 512, n=1) for _ in range(3)])
        self.p5_psa = PSA(512) if use_psa else nn.Identity()

        # P4: 2 blocks
        self.p4_blocks = nn.Sequential(*[BlockClass(256, 256, n=1) for _ in range(2)])

        # P3: 1 block
        self.p3_blocks = BlockClass(128, 128, n=1)

        # P2: Simple conv only (highest resolution - expensive)
        self.p2_conv = Conv(64, 64, 3)

    def forward(self, f256, f128, f64, f32):
        # Process each level with rank-guided compute
        f32 = self.p5_psa(self.p5_blocks(f32))
        f64 = self.p4_blocks(f64)
        f128 = self.p3_blocks(f128)
        f256 = self.p2_conv(f256)

        return f256, f128, f64, f32


class DoubleConvUp(nn.Module):
    """Double convolution with upsampling using C2f blocks."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, use_cib: bool = True):
        super().__init__()
        # Note: use_cib=True uses C2fCIB (CIB blocks with depthwise convs)
        # This matches the checkpoint architecture
        if use_cib:
            self.conv = nn.Sequential(
                C2fCIB(in_ch + mid_ch, mid_ch, n=1),
                nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                Conv(in_ch + mid_ch, mid_ch, 3, 1),
                Conv(mid_ch, mid_ch, 3, 1),
                nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DoubleConv(nn.Module):
    """Double convolution with optional downsampling using C2f blocks."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_cib: bool = True):
        super().__init__()
        # Use SCDown for downsampling (spatial-channel decoupled) instead of AvgPool2d
        self.down = SCDown(in_ch, in_ch, k=3, s=2) if stride > 1 else None
        # Note: use_cib=True uses C2fCIB (CIB blocks with depthwise convs)
        # This matches the checkpoint architecture
        if use_cib:
            self.conv = C2fCIB(in_ch, out_ch, n=1)
        else:
            self.conv = nn.Sequential(
                Conv(in_ch, out_ch, 3, 1),
                Conv(out_ch, out_ch, 3, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down is not None:
            x = self.down(x)
        return self.conv(x)


class UnetHead(nn.Module):
    """
    UNet-style segmentation head for text region detection.

    Takes multi-scale features from backbone and produces binary segmentation mask.

    Input features (for 1024x1024 input):
        - f256: (B, 64, 256, 256)  - stride 4
        - f128: (B, 128, 128, 128) - stride 8
        - f64: (B, 256, 64, 64)    - stride 16
        - f32: (B, 512, 32, 32)    - stride 32
        - f16: (B, 512, 16, 16)    - stride 64

    Output:
        - mask: (B, 1, H, W) binary segmentation mask
    """

    def __init__(self, use_cib: bool = True):
        super().__init__()

        # Process deepest features (keep spatial size to align with f32 after upsampling)
        self.down_conv1 = DoubleConv(512, 512, stride=1, use_cib=use_cib)

        # Upsampling path with C2fCIB blocks (more efficient than C2f)
        self.upconv0 = DoubleConvUp(0, 512, 256, use_cib=use_cib)    # 16 -> 32
        self.upconv2 = DoubleConvUp(256, 512, 256, use_cib=use_cib)  # + f32 -> 64
        self.upconv3 = DoubleConvUp(0, 512, 256, use_cib=use_cib)    # + f64 -> 128
        self.upconv4 = DoubleConvUp(128, 256, 128, use_cib=use_cib)  # + f128 -> 256
        self.upconv5 = DoubleConvUp(64, 128, 64, use_cib=use_cib)    # + f256 -> 512

        # Final mask prediction with LargeKernelConv for larger receptive field
        # This improves text detection accuracy by capturing more context
        self.large_kernel = LargeKernelConv(64, 64, k=7)
        self.upconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(
        self,
        f256: torch.Tensor,
        f128: torch.Tensor,
        f64: torch.Tensor,
        f32: torch.Tensor,
        f16: torch.Tensor,
        forward_mode: int = TEXTDET_MASK
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """
        Forward pass through UNet head.

        Args:
            f256, f128, f64, f32, f16: Multi-scale features from backbone
            forward_mode: Output mode
                - TEXTDET_MASK: Return only segmentation mask
                - TEXTDET_DET: Return intermediate features for DBHead
                - TEXTDET_INFERENCE: Return mask and features

        Returns:
            Depending on forward_mode:
                - TEXTDET_MASK: mask tensor
                - TEXTDET_DET: (f128, f64, u64) for DBHead
                - TEXTDET_INFERENCE: (mask, [f128, f64, u64])
        """
        # Downsample deepest features
        d8 = self.down_conv1(f16)  # 512@8

        # Upsample path
        u16 = self.upconv0(d8)  # 256@16
        u32 = self.upconv2(torch.cat([f32, u16], dim=1))  # 256@32

        if forward_mode == TEXTDET_DET:
            return f128, f64, u32

        u64 = self.upconv3(torch.cat([f64, u32], dim=1))    # 256@64
        u128 = self.upconv4(torch.cat([f128, u64], dim=1))  # 128@128
        u256 = self.upconv5(torch.cat([f256, u128], dim=1)) # 64@256
        # Apply large kernel conv for better context before final prediction
        u256 = self.large_kernel(u256)                       # 64@256 with 7x7 receptive field
        mask = self.upconv6(u256)                            # 1@512

        if forward_mode == TEXTDET_MASK:
            return mask
        else:  # TEXTDET_INFERENCE
            return mask, [f128, f64, u32]

    def init_weight(self, init_func):
        """Initialize weights using provided function."""
        self.apply(init_func)


class DBHead(nn.Module):
    """
    Differentiable Binarization head for text line detection.

    Produces shrink maps, threshold maps, and binary maps for
    precise text line polygon detection.

    Reference: Real-time Scene Text Detection with Differentiable Binarization
    """

    def __init__(
        self,
        in_channels: int = 64,
        k: int = 50,
        shrink_with_sigmoid: bool = True,
        use_cib: bool = True
    ):
        super().__init__()
        self.k = k
        self.shrink_with_sigmoid = shrink_with_sigmoid

        # Upsampling from UNet features using C2fCIB blocks
        self.upconv3 = DoubleConvUp(0, 512, 256, use_cib=use_cib)   # 64 -> 128
        self.upconv4 = DoubleConvUp(128, 256, 128, use_cib=use_cib) # + f128 -> 256

        # Feature processing
        self.conv = nn.Sequential(
            nn.Conv2d(128, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )

        # Binarization head
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2)
        )

        # Initialize final layer bias to shift sigmoid output higher
        nn.init.constant_(self.binarize[-1].bias, 2.0)  # sigmoid(2) ≈ 0.88

        # Threshold head
        self.thresh = self._init_thresh(in_channels)

    def _init_thresh(self, inner_channels: int) -> nn.Sequential:
        """Initialize threshold prediction head."""
        return nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels // 4),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, 2),
            nn.BatchNorm2d(inner_channels // 4),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, 1, 2, 2),
            nn.Sigmoid()
        )

    def forward(
        self,
        f128: torch.Tensor,
        f64: torch.Tensor,
        u64: torch.Tensor,
        step_eval: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through DB head.

        Args:
            f128: Features at stride 8 (128x128)
            f64: Features at stride 16 (64x64)
            u64: Upsampled features from UNet at stride 16
            step_eval: Use step function for evaluation

        Returns:
            During training: (shrink_maps, threshold_maps, binary_maps[, logits])
            During inference: (shrink_maps, threshold_maps) or binary_maps if step_eval
        """
        # Upsample
        u128 = self.upconv3(torch.cat([f64, u64], dim=1))    # 256@128
        x = self.upconv4(torch.cat([f128, u128], dim=1))     # 128@256
        x = self.conv(x)

        # Predict threshold and shrink maps
        threshold_maps = self.thresh(x)
        x = self.binarize(x)
        shrink_maps = torch.sigmoid(x)

        if self.training:
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            if self.shrink_with_sigmoid:
                return torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
            else:
                return torch.cat((shrink_maps, threshold_maps, binary_maps, x), dim=1)
        else:
            if step_eval:
                return self.step_function(shrink_maps, threshold_maps)
            else:
                return torch.cat((shrink_maps, threshold_maps), dim=1)

    def step_function(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Differentiable step function for binarization."""
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def init_weight(self, init_func):
        """Initialize weights using provided function."""
        self.apply(init_func)


class BlockDetector(nn.Module):
    """
    YOLO-style block detection head with one-to-one assignment (NMS-free).

    Uses YOLOv10's dual assignment during training:
    - One-to-many head: Rich supervision signal (topk=10)
    - One-to-one head: NMS-free inference (topk=1)

    Includes proper grid/stride decoding for bounding box predictions.
    """

    def __init__(
        self,
        nc: int = 1,  # number of classes (1 for text block)
        ch: Tuple[int, ...] = (128, 256, 512),  # input channels from P3, P4, P5
        anchors_per_scale: int = 3,
        strides: Tuple[int, ...] = (8, 16, 32),  # P3=8, P4=16, P5=32
        img_size: int = 1024,
    ):
        super().__init__()
        self.nc = nc
        # For single-class: merge obj+cls into single confidence (5 outputs: x,y,w,h,conf)
        # For multi-class: keep separate obj+cls (5+nc outputs)
        self.no = 5 if nc == 1 else nc + 5
        self.nl = len(ch)  # number of detection layers
        self.na = anchors_per_scale
        self.strides = strides
        self.img_size = img_size

        # Detection convolutions for each scale
        self.detect_convs = nn.ModuleList([
            nn.Sequential(
                Conv(c, c, 3),
                nn.Conv2d(c, self.no * self.na, 1)
            ) for c in ch
        ])

        # One-to-one head for NMS-free inference
        self.o2o_convs = nn.ModuleList([
            nn.Sequential(
                Conv(c, c, 3),
                nn.Conv2d(c, self.no, 1)  # Single prediction per location
            ) for c in ch
        ])

        self.training_mode = True  # Use both heads during training

        # Initialize objectness bias to prior probability of 0.01
        # Without this, sigmoid(0)=0.5 across all anchors makes the model
        # learn to suppress everything due to extreme fg/bg imbalance
        self._init_obj_bias()

        # Grid cache for efficiency
        self._grid_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def _init_obj_bias(self):
        """Initialize objectness/confidence bias to log(prior/(1-prior)) with prior=0.01.

        Standard YOLO practice: start with low objectness so the model
        doesn't need to first unlearn 0.5 confidence across all anchors.
        """
        import math
        prior = 0.01
        bias_value = math.log(prior / (1.0 - prior))  # -4.595
        for convs in [self.detect_convs, self.o2o_convs]:
            for conv_seq in convs:
                final_conv = conv_seq[-1]
                if final_conv.bias is not None:
                    with torch.no_grad():
                        na = self.na if convs is self.detect_convs else 1
                        for a in range(na):
                            # Confidence is at index 4 (per anchor: x,y,w,h,conf[,cls...])
                            conf_idx = a * self.no + 4
                            if conf_idx < final_conv.bias.data.shape[0]:
                                final_conv.bias.data[conf_idx] = bias_value

    def _make_grid(self, h: int, w: int, stride: int, device: torch.device) -> torch.Tensor:
        """
        Create grid of (x, y) coordinates for a feature map.

        Returns:
            Tensor of shape [1, h*w, 2] with (grid_x, grid_y) for each cell
        """
        cache_key = (h, w, stride)
        if cache_key in self._grid_cache:
            cached = self._grid_cache[cache_key]
            if cached.device == device:
                return cached

        # Create grid coordinates
        yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        grid = torch.stack([xv, yv], dim=-1).float()  # [h, w, 2]
        grid = grid.reshape(1, h * w, 2).to(device)  # [1, h*w, 2]

        self._grid_cache[cache_key] = grid
        return grid

    def _decode_predictions(
        self,
        pred: torch.Tensor,
        grid: torch.Tensor,
        stride: int,
        na: int = 1
    ) -> torch.Tensor:
        """
        Decode raw predictions to normalized bounding boxes.

        Args:
            pred: Raw predictions [B, na*h*w, no] or [B, h*w, no]
            grid: Grid coordinates [1, h*w, 2]
            stride: Stride for this feature level
            na: Number of anchors (1 for o2o, self.na for o2m)

        Returns:
            Decoded predictions [B, N, no] with normalized xywh boxes
        """
        bs = pred.shape[0]
        n_cells = grid.shape[1]

        # Expand grid for multiple anchors if needed
        if na > 1:
            grid = grid.repeat(1, na, 1)  # [1, na*h*w, 2]

        # Split predictions
        xy = pred[..., :2]  # [B, N, 2]
        wh = pred[..., 2:4]  # [B, N, 2]
        conf_cls = pred[..., 4:]  # [B, N, 1+nc]

        # Decode xy: sigmoid then add grid offset, scale by stride, normalize
        # Formula: (sigmoid(t) * 2 - 0.5 + grid) * stride / img_size
        xy_decoded = (xy.sigmoid() * 2 - 0.5 + grid) * stride / self.img_size

        # Decode wh: sigmoid then square and scale
        # Formula: (sigmoid(t) * 4)^2 * stride / img_size (anchor-free style)
        # This allows boxes up to 512px at stride=32 (was 128px with * 2)
        wh_decoded = (wh.sigmoid() * 4) ** 2 * stride / self.img_size

        # Clamp to valid range [0, 1]
        xy_decoded = xy_decoded.clamp(0, 1)
        wh_decoded = wh_decoded.clamp(0, 1)

        # Concatenate decoded boxes with confidence/class (keep as logits for training)
        return torch.cat([xy_decoded, wh_decoded, conf_cls], dim=-1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of [P3, P4, P5] feature maps

        Returns:
            During training: (o2m_preds, o2o_preds) for dual assignment loss
                - Both have decoded xywh in [0,1] range, logits for conf/cls
            During inference: o2o_preds only (NMS-free)
                - Decoded xywh in [0,1], sigmoid applied to conf/cls
        """
        if self.training and self.training_mode:
            # Dual assignment: both one-to-many and one-to-one
            o2m_outputs = []
            o2o_outputs = []

            for i, (feat, o2m_conv, o2o_conv) in enumerate(
                zip(features, self.detect_convs, self.o2o_convs)
            ):
                bs, _, h, w = feat.shape
                stride = self.strides[i]
                grid = self._make_grid(h, w, stride, feat.device)

                # One-to-many predictions
                o2m = o2m_conv(feat)
                o2m = o2m.view(bs, self.na, self.no, h, w).permute(0, 1, 3, 4, 2)
                o2m = o2m.reshape(bs, -1, self.no)
                o2m = self._decode_predictions(o2m, grid, stride, self.na)
                o2m_outputs.append(o2m)

                # One-to-one predictions
                o2o = o2o_conv(feat)
                o2o = o2o.view(bs, 1, self.no, h, w).permute(0, 1, 3, 4, 2)
                o2o = o2o.reshape(bs, -1, self.no)
                o2o = self._decode_predictions(o2o, grid, stride, 1)
                o2o_outputs.append(o2o)

            return torch.cat(o2m_outputs, 1), torch.cat(o2o_outputs, 1)
        else:
            # Inference: one-to-one only (NMS-free)
            outputs = []
            for i, (feat, o2o_conv) in enumerate(zip(features, self.o2o_convs)):
                bs, _, h, w = feat.shape
                stride = self.strides[i]
                grid = self._make_grid(h, w, stride, feat.device)

                out = o2o_conv(feat)
                out = out.view(bs, 1, self.no, h, w).permute(0, 1, 3, 4, 2)
                out = out.reshape(bs, -1, self.no)
                out = self._decode_predictions(out, grid, stride, 1)
                outputs.append(out)

            preds = torch.cat(outputs, 1)
            # Apply sigmoid to confidence and class scores for inference
            preds[..., 4:] = preds[..., 4:].sigmoid()
            return preds

    def init_weight(self, init_func):
        """Initialize weights using provided function."""
        self.apply(init_func)


class TextDetector(nn.Module):
    """
    Complete text detector combining backbone and heads.

    Supports four modes:
        - Mask training: Train segmentation head
        - DB training: Train text line detection head
        - Block training: Train block detection head
        - Inference: Full pipeline

    Optional advanced modules:
        - use_bifpn: Enable EfficientBiFPN for bidirectional feature fusion
        - use_rank_guided: Enable RankGuidedDecoder for compute-efficient decoding
    """

    def __init__(
        self,
        backbone: nn.Module,
        use_cib: bool = True,
        freeze_backbone: bool = True,
        use_bifpn: bool = False,
        use_rank_guided: bool = False
    ):
        super().__init__()

        self.backbone = backbone
        self.freeze_backbone = freeze_backbone

        # Optional EfficientBiFPN for enhanced feature fusion
        # Channels: [64, 128, 256, 512] corresponding to P2, P3, P4, P5
        self.bifpn = EfficientBiFPN([64, 128, 256, 512]) if use_bifpn else None

        # Optional RankGuidedDecoder for compute-efficient feature refinement
        # Allocates more compute to semantic-rich (P5) and less to high-resolution (P2)
        self.rank_decoder = RankGuidedDecoder(use_cib=use_cib, use_psa=True) if use_rank_guided else None

        self.seg_net = UnetHead(use_cib=use_cib)
        self.dbnet: Optional[DBHead] = None
        self.block_det: Optional[BlockDetector] = None
        self.forward_mode = TEXTDET_MASK

    def train_mask(self):
        """Set mode for training segmentation head."""
        self.forward_mode = TEXTDET_MASK
        if self.freeze_backbone:
            self.backbone.eval()
        else:
            self.backbone.train()
        self.seg_net.train()

    def initialize_db(self, unet_weights: Optional[str] = None):
        """
        Initialize DB head, optionally loading UNet weights.

        Args:
            unet_weights: Path to UNet checkpoint for initializing shared layers
        """
        self.dbnet = DBHead(64, use_cib=True)

        if unet_weights is not None:
            checkpoint = torch.load(unet_weights, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'weights' in checkpoint:
                state_dict = checkpoint['weights']
            else:
                state_dict = checkpoint
            
            # Filter and clean seg_net keys
            # Keys may be like: _orig_mod.seg_net.down_conv1... or seg_net.down_conv1...
            seg_net_state_dict = {}
            for k, v in state_dict.items():
                # Strip _orig_mod. prefix (added by torch.compile)
                clean_key = k.replace('_orig_mod.', '')
                # Only keep seg_net keys and strip the seg_net. prefix
                if clean_key.startswith('seg_net.'):
                    new_key = clean_key.replace('seg_net.', '', 1)
                    seg_net_state_dict[new_key] = v
            
            self.seg_net.load_state_dict(seg_net_state_dict)

        # Share upsampling layers with UNet
        self.dbnet.upconv3 = copy.deepcopy(self.seg_net.upconv3)
        self.dbnet.upconv4 = copy.deepcopy(self.seg_net.upconv4)

        # Keep UNet layers for checkpoint compatibility
        # (Weight sharing via copy.deepcopy is sufficient - no need to delete)
        # del self.seg_net.upconv3
        # del self.seg_net.upconv4
        # del self.seg_net.upconv5
        # del self.seg_net.upconv6

    def initialize_block_detector(
        self,
        nc: int = 1,
        ch: Tuple[int, ...] = (128, 256, 512),
        anchors_per_scale: int = 3,
        checkpoint: Optional[str] = None
    ):
        """
        Initialize block detection head.

        Args:
            nc: Number of classes (1 for text block)
            ch: Input channels from P3, P4, P5 features
            anchors_per_scale: Number of anchors per scale
            checkpoint: Optional path to checkpoint for loading weights
        """
        self.block_det = BlockDetector(nc=nc, ch=ch, anchors_per_scale=anchors_per_scale)

        if checkpoint is not None:
            ckpt = torch.load(checkpoint, map_location='cpu')

            # Handle different checkpoint formats
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            elif 'weights' in ckpt:
                state_dict = ckpt['weights']
            else:
                state_dict = ckpt

            # Filter for block_det keys
            block_det_state_dict = {}
            for k, v in state_dict.items():
                clean_key = k.replace('_orig_mod.', '')
                if clean_key.startswith('block_det.'):
                    new_key = clean_key.replace('block_det.', '', 1)
                    block_det_state_dict[new_key] = v

            if block_det_state_dict:
                self.block_det.load_state_dict(block_det_state_dict)

    def train_db(self):
        """Set mode for training DB head."""
        self.forward_mode = TEXTDET_DET
        if self.freeze_backbone:
            self.backbone.eval()
        else:
            self.backbone.train()
        self.seg_net.eval()
        if self.dbnet is not None:
            self.dbnet.train()

    def train_block(self):
        """Set model to block detection training mode."""
        self.forward_mode = TEXTDET_BLOCK
        if self.freeze_backbone:
            self.backbone.eval()
        else:
            self.backbone.train()

        # Frozen layers should be in eval mode to not update BatchNorm stats
        self.seg_net.eval()
        if self.dbnet is not None:
            self.dbnet.eval()
        if self.block_det is not None:
            self.block_det.train()

    def train_unified(self, freeze_backbone: bool = True):
        """Set model to unified training mode (all heads jointly).

        In unified mode, the model returns a dictionary with all outputs:
        - 'mask': Segmentation mask from seg_net
        - 'lines': Text line detection from dbnet
        - 'blocks': Block detection from block_det

        Args:
            freeze_backbone: Whether to freeze backbone (Phase 1) or not (Phase 2)
        """
        self.forward_mode = TEXTDET_UNIFIED
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            self.backbone.train()
            for param in self.backbone.parameters():
                param.requires_grad = True

        # All heads are trainable in unified mode
        self.seg_net.train()
        if self.dbnet is not None:
            self.dbnet.train()
        if self.block_det is not None:
            self.block_det.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through detector.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Depending on mode:
                - TEXTDET_MASK: Segmentation mask
                - TEXTDET_DET: DB head output
                - TEXTDET_BLOCK: Block detection output (o2m_preds, o2o_preds) during training
                - TEXTDET_UNIFIED: Dict with 'mask', 'lines', 'blocks' outputs
        """
        # Extract features from backbone
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            features = self.backbone(x)

        # features = (f256, f128, f64, f32, f16)
        f256, f128, f64, f32, f16 = features

        # Optional: Apply EfficientBiFPN for bidirectional feature fusion
        if self.bifpn is not None:
            # BiFPN operates on P2-P5 (f256, f128, f64, f32)
            fused = self.bifpn([f256, f128, f64, f32])
            f256, f128, f64, f32 = fused

        # Optional: Apply RankGuidedDecoder for compute-efficient refinement
        if self.rank_decoder is not None:
            # RankGuidedDecoder refines P2-P5 with rank-guided compute allocation
            f256, f128, f64, f32 = self.rank_decoder(f256, f128, f64, f32)

        # Rebuild features tuple
        features = (f256, f128, f64, f32, f16)

        if self.forward_mode == TEXTDET_MASK:
            return self.seg_net(*features, forward_mode=self.forward_mode)
        elif self.forward_mode == TEXTDET_DET:
            # Note: If freezing UNet is desired, use --freeze-seg flag in training
            with torch.no_grad():
                seg_features = self.seg_net(*features, forward_mode=self.forward_mode)
            return self.dbnet(*seg_features)
        elif self.forward_mode == TEXTDET_BLOCK:
            # Block detection uses P3, P4, P5 features
            # P3 = f128 (stride 8), P4 = f64 (stride 16), P5 = f32 (stride 32)
            block_features = [f128, f64, f32]
            return self.block_det(block_features)
        elif self.forward_mode == TEXTDET_UNIFIED:
            # Unified mode: compute all outputs in one forward pass
            outputs = {}

            # Segmentation mask
            mask, seg_features = self.seg_net(*features, forward_mode=TEXTDET_INFERENCE)
            outputs['mask'] = mask

            # Text line detection (DBNet)
            if self.dbnet is not None:
                lines = self.dbnet(*seg_features)
                outputs['lines'] = lines

            # Block detection
            if self.block_det is not None:
                block_features = [f128, f64, f32]
                blocks = self.block_det(block_features)
                outputs['blocks'] = blocks

            return outputs
        else:
            raise ValueError(f"Unknown forward mode: {self.forward_mode}")


class TextDetectorInference(nn.Module):
    """
    Inference-only text detector with all heads.

    Produces text block detections, segmentation mask, and text line maps.

    Optional advanced modules:
        - bifpn: EfficientBiFPN for bidirectional feature fusion
        - rank_decoder: RankGuidedDecoder for compute-efficient decoding
    """

    def __init__(
        self,
        backbone: nn.Module,
        seg_head: UnetHead,
        db_head: DBHead,
        block_detector: Optional[BlockDetector] = None,
        bifpn: Optional[EfficientBiFPN] = None,
        rank_decoder: Optional[RankGuidedDecoder] = None
    ):
        super().__init__()

        self.backbone = backbone
        self.seg_head = seg_head
        self.db_head = db_head
        self.block_det = block_detector
        self.bifpn = bifpn
        self.rank_decoder = rank_decoder

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Full inference forward pass.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Tuple of:
                - blocks: Text block detections (if block_detector provided)
                - mask: Segmentation mask
                - lines: Text line detection maps
        """
        # Extract features
        features = self.backbone(x)

        # features = (f256, f128, f64, f32, f16)
        f256, f128, f64, f32, f16 = features

        # Optional: Apply EfficientBiFPN for bidirectional feature fusion
        if self.bifpn is not None:
            fused = self.bifpn([f256, f128, f64, f32])
            f256, f128, f64, f32 = fused

        # Optional: Apply RankGuidedDecoder for compute-efficient refinement
        if self.rank_decoder is not None:
            f256, f128, f64, f32 = self.rank_decoder(f256, f128, f64, f32)

        # Rebuild features tuple
        features = (f256, f128, f64, f32, f16)

        # Run through heads
        mask, seg_features = self.seg_head(*features, forward_mode=TEXTDET_INFERENCE)
        lines = self.db_head(*seg_features, step_eval=False)

        # Block detection uses P3, P4, P5 features
        blocks = None
        if self.block_det is not None:
            block_features = [f128, f64, f32]
            blocks = self.block_det(block_features)

        return blocks, mask, lines


def init_weights(m: nn.Module):
    """Initialize module weights using Kaiming initialization."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
