# Technical Validation Details

## Output Shape Analysis

### Original Model (comic-text-detector.onnx)

```
ONNX Model: data/comic-text-detector.onnx
Input: (1, 3, 1024, 1024) - RGB image

Outputs:
  1. blk:  (1, 64512, 7) - Block/text bubble detections
  2. seg:  (1, 1, 1024, 1024) - Segmentation mask
  3. det:  (1, 2, 1024, 1024) - DBNet text line detection
```

**Verified via:**
```python
import onnx
model = onnx.load('data/comic-text-detector.onnx')
for output in model.graph.output:
    dims = [d.dim_value for d in output.type.tensor_type.shape.dim]
    print(f"{output.name}: {dims}")
```

---

## Our Model Architecture

### Component 1: UnetHead (Segmentation)

**File:** `src/models/heads.py` (lines 336-424)

**Input Features:**
- f256: (B, 64, 256, 256) - stride 4
- f128: (B, 128, 128, 128) - stride 8
- f64: (B, 256, 64, 64) - stride 16
- f32: (B, 512, 32, 32) - stride 32
- f16: (B, 512, 16, 16) - stride 64

**Architecture:**
```
f16 → down_conv1 → 512@16
↓
upconv0 (×2) → 256@32
↓ + f32
upconv2 (×2) → 256@64
↓ + f64
upconv3 (×2) → 256@128
↓ + f128
upconv4 (×2) → 128@256
↓ + f256
upconv5 (×2) → 64@512
↓
large_kernel_conv (7×7) → 64@512
↓
upconv6 (ConvTranspose2d, stride=2) + Sigmoid → 1@1024
```

**Output:** (B, 1, 1024, 1024) ✅
- Binary segmentation mask
- Values in [0, 1] via Sigmoid
- Matches original exactly

**Code validation:**
```python
# Line 369-372
self.upconv6 = nn.Sequential(
    nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Sigmoid()
)
```

---

### Component 2: DBHead (Text Line Detection)

**File:** `src/models/heads.py` (lines 426-534)

**Input Features (from UnetHead):**
- f128: (B, 128, 128, 128) - stride 8
- f64: (B, 256, 64, 64) - stride 16
- u64: (B, 256, 64, 64) - upsampled from UNet

**Architecture:**
```
[f64, u64] → cat → (512,64,64)
↓
upconv3 (×2) → 256@128
↓ + f128
upconv4 (×2) → 128@256
↓
conv (1×1) → 64@256
↓
┌─ binarize head (3×3 conv → ConvTranspose2d ×2 → 1@1024)
│  → shrink_maps with Sigmoid
│
└─ thresh head (3×3 conv → ConvTranspose2d ×2 → 1@1024)
   → threshold_maps with Sigmoid

[shrink_maps, threshold_maps] → cat(dim=1) → (2, 1024, 1024)
```

**Output:** (B, 2, 1024, 1024) ✅
- Channel 0: Probability/shrink map [0, 1]
- Channel 1: Threshold map [0, 1]
- Matches original exactly

**Code validation:**
```python
# Lines 511-525 (inference mode)
threshold_maps = self.thresh(x)
x = self.binarize(x)
shrink_maps = torch.sigmoid(x)

# Inference returns 2-channel output
return torch.cat((shrink_maps, threshold_maps), dim=1)  # (B, 2, 1024, 1024)
```

**Training vs Inference:**
```python
if self.training:
    # Training: returns (B, 3, 1024, 1024) or (B, 4, 1024, 1024)
    binary_maps = self.step_function(shrink_maps, threshold_maps)
    return torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
else:
    # Inference: returns (B, 2, 1024, 1024)
    return torch.cat((shrink_maps, threshold_maps), dim=1)
```

Export uses inference mode, so output is correct (2 channels).

---

### Component 3: BlockDetector (Text Blocks)

**File:** `src/models/heads.py` (lines 536-712)

**Input Features:**
- P3: features[1] - (B, 128, 128, 128) - stride 8
- P4: features[2] - (B, 256, 64, 64) - stride 16
- P5: features[3] - (B, 512, 32, 32) - stride 32

**Architecture:**
```
For each scale (P3, P4, P5):
  1. Apply detection conv (Conv 3×3 → Conv 1×1)
     Output: (B, H, W, na * no)  where na=3, no=6

  2. Reshape: (B, na, no, H, W) → permute → (B, H*W*na, no)

  3. Decode predictions:
     - xy_decoded = (sigmoid(xy) * 2 - 0.5 + grid) * stride / img_size
     - wh_decoded = (sigmoid(wh) * 4)^2 * stride / img_size
     - conf_cls unchanged (logits during training, sigmoid during inference)
```

**Concatenation across scales:**
```
o2o_outputs = [P3_preds, P4_preds, P5_preds]
concatenated = torch.cat(o2o_outputs, dim=1)
# Resulting shape: (B, (128×128×3 + 64×64×3 + 32×32×3), 6)
#                = (B, 64512, 6) ❌ WRONG - should be (B, 64512, 7)
```

**Current Output:** (B, 64512, 6) - one dense concatenation ❌
- Position 0-1: normalized x, y
- Position 2-3: normalized w, h
- Position 4: confidence score
- Position 5: class score

**Expected Output:** (B, 64512, 7) ❌
- Position 0-3: as above
- Position 4-6: Unknown (needs reverse engineering)

**Code:**
```python
# Lines 652-707
def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
    if self.training and self.training_mode:
        # Training: returns tuple (o2m_preds, o2o_preds)
        return torch.cat(o2m_outputs, 1), torch.cat(o2o_outputs, 1)
    else:
        # Inference: returns single tensor
        preds = torch.cat(outputs, 1)  # Shape: (B, 64512, 6)
        preds[..., 4:] = preds[..., 4:].sigmoid()
        return preds
```

---

## Export Wrapper Analysis

**File:** `scripts/export.py` (lines 95-122)

**ExportableModelV10:**
```python
def forward(self, x):
    # Normalize input
    x = (x - self.mean) / self.std

    # Backbone features
    features = self.backbone(x)

    # UnetHead: seg + features
    mask, seg_features = self.seg_head(*features, forward_mode=2)

    # DBHead: lines
    lines = self.db_head(*seg_features, step_eval=False)

    # BlockDetector: blocks (P3, P4, P5)
    if self.block_detector is not None:
        blocks = self.block_detector([features[1], features[2], features[3]])
    else:
        blocks = torch.zeros(x.shape[0], 0, 6, device=x.device)

    return blocks, mask, lines
```

**Output Shapes:**
- blocks: (B, 64512, 6) ❌
- mask: (B, 1, 1024, 1024) ✅
- lines: (B, 2, 1024, 1024) ✅

---

## Inference Path

**File:** `src/models/detector.py` (lines 199-240)

TextDetBaseDNN (ONNX inference):
```python
def __call__(self, im_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        Tuple of (blocks, mask, lines_map)
    """
    # Currently returns:
    # - blocks: (1, N, 6) - WILL FAIL with original (1, 64512, 7)
    # - mask: (1, 1, 1024, 1024) ✓
    # - lines_map: (1, 2, 1024, 1024) ✓
```

---

## Feature Decoding Process

**BlockDetector decoding (lines 605-650):**

```python
def _decode_predictions(self, pred, grid, stride, na=1):
    """
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

    # Expand grid for multiple anchors
    if na > 1:
        grid = grid.repeat(1, na, 1)  # [1, na*h*w, 2]

    # Split predictions
    xy = pred[..., :2]      # [B, N, 2]
    wh = pred[..., 2:4]     # [B, N, 2]
    conf_cls = pred[..., 4:]  # [B, N, 1+nc]

    # Decode xy
    # Formula: (sigmoid(t) * 2 - 0.5 + grid) * stride / img_size
    xy_decoded = (xy.sigmoid() * 2 - 0.5 + grid) * stride / self.img_size

    # Decode wh
    # Formula: (sigmoid(t) * 4)^2 * stride / img_size
    wh_decoded = (wh.sigmoid() * 4) ** 2 * stride / self.img_size

    # Clamp to [0, 1]
    xy_decoded = xy_decoded.clamp(0, 1)
    wh_decoded = wh_decoded.clamp(0, 1)

    # Return concatenated
    return torch.cat([xy_decoded, wh_decoded, conf_cls], dim=-1)
```

**Output: (B, N, 6)** where N = sum of grid cells across scales

---

## Grid Calculation

**For 1024×1024 input:**

| Scale | Stride | Height | Width | Cells | Anchors | Total |
|-------|--------|--------|-------|-------|---------|-------|
| P3 | 8 | 128 | 128 | 16,384 | 3 | 49,152 |
| P4 | 16 | 64 | 64 | 4,096 | 3 | 12,288 |
| P5 | 32 | 32 | 32 | 1,024 | 3 | 3,072 |
| **Total** | - | - | - | **21,504** | - | **64,512** |

**Verification:**
```python
# P3
128 * 128 * 3 = 49,152 ✓

# P4
64 * 64 * 3 = 12,288 ✓

# P5
32 * 32 * 3 = 3,072 ✓

# Total
49,152 + 12,288 + 3,072 = 64,512 ✓
```

---

## Training Mode Differences

**During training, BlockDetector returns tuple:**
```python
if self.training and self.training_mode:
    return torch.cat(o2m_outputs, 1), torch.cat(o2o_outputs, 1)
    # Returns: (o2m_preds, o2o_preds)
    # o2m_preds: (B, 64512, 6) - one-to-many for supervision
    # o2o_preds: (B, 64512, 6) - one-to-one for inference
```

This dual-head approach (from YOLOv10) enables:
- **o2m (one-to-many):** Multiple predictions per object for rich supervision
- **o2o (one-to-one):** Single prediction per location for NMS-free inference

**During inference, only o2o is returned:**
```python
else:
    preds = torch.cat(outputs, 1)  # (B, 64512, 6)
    preds[..., 4:] = preds[..., 4:].sigmoid()
    return preds
```

**Export uses inference mode (step_eval=False)**, so training tuple is not used.

---

## Loss Functions

**Where the 6-value output is used:**

In training, loss functions expect 6 values per detection:
```python
# From loss functions (assumed in training script)
loss = compute_block_loss(preds, targets)
# preds shape: (B, 64512, 6)
# targets: ground truth boxes [x, y, w, h, conf, cls]
```

The 7th value in the original is likely:
1. **Anchor index** - which of 3 anchors at each cell
2. **Refinement score** - additional confidence metric
3. **Auxiliary loss** - from auxiliary head in original architecture
4. **Geometric property** - aspect ratio, area, or other metric

---

## Modification Strategy

### Step 1: Identify 7th Value

**Option A: Run original model and inspect**
```python
import onnxruntime as rt
import numpy as np

sess = rt.InferenceSession('data/comic-text-detector.onnx')
x = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
outputs = sess.run(None, {'images': x})

blk_output = outputs[0]  # (1, 64512, 7)
print(f"Shape: {blk_output.shape}")
print(f"Channel 0 range: [{blk_output[0,:,0].min():.3f}, {blk_output[0,:,0].max():.3f}]")
print(f"Channel 1 range: [{blk_output[0,:,1].min():.3f}, {blk_output[0,:,1].max():.3f}]")
# ... inspect all 7 channels
print(f"Channel 6 range: [{blk_output[0,:,6].min():.3f}, {blk_output[0,:,6].max():.3f}]")
```

### Step 2: Implement 7th Value

**If anchor index:**
```python
# In BlockDetector._decode_predictions()
anchor_indices = torch.arange(na).view(1, na, 1).expand(bs, -1, n_cells).reshape(bs, -1, 1)
return torch.cat([xy_decoded, wh_decoded, conf_cls, anchor_indices], dim=-1)
```

**If refinement from o2m:**
```python
# In export wrapper
o2m_pred, o2o_pred = self.block_detector(features)
# Extract confidence refinement from o2m
refinement = o2m_pred[..., 4:5].mean(dim=0, keepdim=True)
return torch.cat([o2o_pred, refinement], dim=-1)
```

---

## Validation Checklist

- [ ] Confirm 7th value identity
- [ ] Implement 7th value computation
- [ ] Verify (B, 64512, 7) output shape
- [ ] Test ONNX export
- [ ] Compare inference results with original model
- [ ] Verify downstream code compatibility
- [ ] Update loss functions if needed

