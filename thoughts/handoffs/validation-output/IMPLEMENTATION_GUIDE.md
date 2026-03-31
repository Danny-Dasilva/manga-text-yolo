# Implementation Guide: Fix Block Detection Output

## Overview

The block detection head outputs 6 values per detection but needs to output 7. This guide provides three implementation options with code examples.

---

## Option A: Add Anchor Index (Recommended)

The 7th value is likely the anchor index (0, 1, or 2 out of 3 anchors per grid cell).

### Implementation

**File:** `src/models/heads.py`
**Location:** BlockDetector._decode_predictions() method (line 605)

**Current code:**
```python
def _decode_predictions(
    self,
    pred: torch.Tensor,
    grid: torch.Tensor,
    stride: int,
    na: int = 1
) -> torch.Tensor:
    """
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

    # Decode xy and wh...
    xy_decoded = (xy.sigmoid() * 2 - 0.5 + grid) * stride / self.img_size
    wh_decoded = (wh.sigmoid() * 4) ** 2 * stride / self.img_size

    xy_decoded = xy_decoded.clamp(0, 1)
    wh_decoded = wh_decoded.clamp(0, 1)

    # Current: returns 6 values
    return torch.cat([xy_decoded, wh_decoded, conf_cls], dim=-1)
```

**Updated code:**
```python
def _decode_predictions(
    self,
    pred: torch.Tensor,
    grid: torch.Tensor,
    stride: int,
    na: int = 1
) -> torch.Tensor:
    """
    Returns:
        Decoded predictions [B, N, 7] with normalized xywh boxes + anchor index
    """
    bs = pred.shape[0]
    n_cells = grid.shape[1]

    # Create anchor indices before expanding grid
    if na > 1:
        # Create anchor indices: [0, 0, ..., 1, 1, ..., 2, 2, ...]
        anchor_idx = torch.arange(na, device=grid.device).view(1, na, 1)
        anchor_idx = anchor_idx.expand(bs, -1, n_cells).reshape(bs, -1, 1).float()

        # Expand grid for multiple anchors
        grid = grid.repeat(1, na, 1)  # [1, na*h*w, 2]
    else:
        # For o2o predictions, anchor index is 0
        anchor_idx = torch.zeros(bs, n_cells, 1, device=grid.device)

    # Split predictions
    xy = pred[..., :2]  # [B, N, 2]
    wh = pred[..., 2:4]  # [B, N, 2]
    conf_cls = pred[..., 4:]  # [B, N, 1+nc]

    # Decode xy and wh
    xy_decoded = (xy.sigmoid() * 2 - 0.5 + grid) * stride / self.img_size
    wh_decoded = (wh.sigmoid() * 4) ** 2 * stride / self.img_size

    xy_decoded = xy_decoded.clamp(0, 1)
    wh_decoded = wh_decoded.clamp(0, 1)

    # Return 7 values: x, y, w, h, conf, cls, anchor_idx
    return torch.cat([xy_decoded, wh_decoded, conf_cls, anchor_idx], dim=-1)
```

**Validation:**
```python
# After changes, verify output shape
model = BlockDetector(nc=1, ch=(128, 256, 512), anchors_per_scale=3)
features = [torch.randn(1, 128, 128, 128),
            torch.randn(1, 256, 64, 64),
            torch.randn(1, 512, 32, 32)]

preds = model(features)
print(preds.shape)  # Should be (1, 64512, 7) ✓
```

---

## Option B: Extract Refinement from Dual-Head

Use the difference between o2m (one-to-many) and o2o (one-to-one) predictions as refinement.

### Implementation

**File:** `src/models/heads.py`
**Location:** BlockDetector.forward() method (line 652)

**Current code (inference):**
```python
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
    return preds  # Current: (B, 64512, 6)
```

**Updated code (with refinement):**
```python
else:
    # Inference: one-to-one only (NMS-free)
    o2o_outputs = []
    o2m_outputs = []

    for i, (feat, o2o_conv, o2m_conv) in enumerate(zip(features, self.o2o_convs, self.detect_convs)):
        bs, _, h, w = feat.shape
        stride = self.strides[i]
        grid = self._make_grid(h, w, stride, feat.device)

        # One-to-one prediction
        o2o = o2o_conv(feat)
        o2o = o2o.view(bs, 1, self.no, h, w).permute(0, 1, 3, 4, 2)
        o2o = o2o.reshape(bs, -1, self.no)
        o2o = self._decode_predictions(o2o, grid, stride, 1)

        # One-to-many prediction for refinement
        o2m = o2m_conv(feat)
        o2m = o2m.view(bs, self.na, self.no, h, w).permute(0, 1, 3, 4, 2)
        o2m = o2m.reshape(bs, -1, self.no)

        # Extract refinement: average confidence across anchors minus o2o confidence
        o2m_conf = o2m[..., 4:5].view(bs, h*w, self.na).mean(dim=2, keepdim=True)
        o2o_conf = o2o[..., 4:5]
        refinement = (o2m_conf - o2o_conf).clamp(-1, 1)

        # Append refinement as 7th value
        o2o = torch.cat([o2o, refinement], dim=-1)
        o2o_outputs.append(o2o)

    preds = torch.cat(o2o_outputs, 1)  # (B, 64512, 7)
    # Apply sigmoid to confidence and class scores for inference
    preds[..., 4:6] = preds[..., 4:6].sigmoid()
    return preds
```

---

## Option C: Reverse-Engineer Original Model

Analyze the original model to understand the 7th value.

### Investigation Script

```python
import onnx
import onnxruntime as rt
import numpy as np

def analyze_original_model():
    """Analyze original model's 7th channel."""

    # Load model
    model_path = 'data/comic-text-detector.onnx'
    session = rt.InferenceSession(model_path)

    # Create test input
    x = np.random.randn(1, 3, 1024, 1024).astype(np.float32)

    # Run inference multiple times
    print("Analyzing 7th channel across multiple inputs...")
    channels = [[] for _ in range(7)]

    for trial in range(10):
        x = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
        outputs = session.run(None, {'images': x})
        blk = outputs[0]  # (1, 64512, 7)

        for ch in range(7):
            channels[ch].append(blk[0, :, ch])

    # Analyze each channel
    for ch in range(7):
        all_values = np.concatenate(channels[ch], axis=0)

        print(f"\nChannel {ch}:")
        print(f"  Min: {all_values.min():.6f}")
        print(f"  Max: {all_values.max():.6f}")
        print(f"  Mean: {all_values.mean():.6f}")
        print(f"  Std: {all_values.std():.6f}")
        print(f"  Unique values (first 10): {np.unique(all_values)[:10]}")

        # Hypothesis testing
        if all_values.max() <= 1.0 and all_values.min() >= 0.0:
            if all_values.max() <= 3.5 and all([v == int(v) for v in np.unique(all_values)[:5]]):
                print(f"  → Hypothesis: Discrete value (anchor index? range: {all_values.min():.0f}-{all_values.max():.0f})")
            else:
                print(f"  → Hypothesis: Normalized probability or metric")
        elif all_values.max() <= 1024 and all_values.min() >= 0:
            print(f"  → Hypothesis: Pixel coordinate or size")
        else:
            print(f"  → Hypothesis: Raw logit or confidence score")

if __name__ == "__main__":
    analyze_original_model()
```

### Analysis Procedure

1. **Run script** to get characteristics of each channel
2. **Visualize** anchor distribution if channel 6 looks like anchor index (values 0-2)
3. **Compare** with known detection formats
4. **Implement** based on findings

---

## Testing All Options

### Test Framework

Create `tests/test_block_detector_output.py`:

```python
import torch
import numpy as np
from src.models.heads import BlockDetector

def test_block_detector_shape():
    """Test that BlockDetector outputs correct shape."""

    # Create detector
    detector = BlockDetector(nc=1, ch=(128, 256, 512), anchors_per_scale=3)
    detector.eval()

    # Create dummy features
    features = [
        torch.randn(2, 128, 128, 128),  # P3
        torch.randn(2, 256, 64, 64),    # P4
        torch.randn(2, 512, 32, 32)     # P5
    ]

    # Forward pass
    with torch.no_grad():
        output = detector(features)

    # Check shape
    assert output.shape == (2, 64512, 7), f"Expected (2, 64512, 7), got {output.shape}"

    # Check value ranges
    assert output[..., 0].min() >= 0 and output[..., 0].max() <= 1, "X coordinates out of range"
    assert output[..., 1].min() >= 0 and output[..., 1].max() <= 1, "Y coordinates out of range"
    assert output[..., 2].min() >= 0 and output[..., 2].max() <= 1, "Width out of range"
    assert output[..., 3].min() >= 0 and output[..., 3].max() <= 1, "Height out of range"
    assert output[..., 4].min() >= 0 and output[..., 4].max() <= 1, "Confidence out of range"

    print("✓ Block detector output shape is correct: (2, 64512, 7)")
    print("✓ All values within expected ranges")

def test_onnx_export():
    """Test ONNX export shape."""
    import onnx
    from scripts.export import ExportableModelV10

    # Load checkpoint and export
    from src.models.detector import load_text_detector
    detector = load_text_detector('runs/block_v2/best.pt')

    # Export to ONNX
    exporter = ExportableModelV10(
        detector.backbone,
        detector.seg_head,
        detector.db_head,
        detector.block_det
    )

    exporter.eval()
    dummy_input = torch.randn(1, 3, 1024, 1024)

    torch.onnx.export(
        exporter,
        dummy_input,
        'models/detector_test.onnx',
        input_names=['images'],
        output_names=['blk', 'seg', 'det'],
        opset_version=18
    )

    # Check ONNX shapes
    model = onnx.load('models/detector_test.onnx')
    for output in model.graph.output:
        dims = [d.dim_value for d in output.type.tensor_type.shape.dim]
        print(f"{output.name}: {dims}")

        if output.name == 'blk':
            assert dims == [1, 64512, 7], f"Expected blk shape [1, 64512, 7], got {dims}"

if __name__ == "__main__":
    test_block_detector_shape()
    print()
    test_onnx_export()
```

### Run Tests

```bash
python tests/test_block_detector_output.py
```

Expected output:
```
✓ Block detector output shape is correct: (2, 64512, 7)
✓ All values within expected ranges
blk: [1, 64512, 7]
seg: [1, 1, 1024, 1024]
det: [1, 2, 1024, 1024]
```

---

## Comparison Testing

Create `tests/compare_with_original.py`:

```python
import onnxruntime as rt
import numpy as np
import torch
from src.models.detector import load_text_detector

def compare_outputs():
    """Compare our model with original model."""

    # Load both models
    original_session = rt.InferenceSession('data/comic-text-detector.onnx')
    our_detector = load_text_detector('runs/block_v2/best.pt')
    our_detector.eval()

    # Create test image
    np.random.seed(42)
    test_image = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
    test_image = np.clip(test_image * 0.1 + 0.5, 0, 1)  # Normalize to [0, 1]

    # Run original model
    print("Running original model...")
    orig_outputs = original_session.run(None, {'images': test_image})
    orig_blk, orig_seg, orig_det = orig_outputs

    print(f"Original outputs:")
    print(f"  blk: {orig_blk.shape}")
    print(f"  seg: {orig_seg.shape}")
    print(f"  det: {orig_det.shape}")

    # Run our model
    print("\nRunning our model...")
    with torch.no_grad():
        test_tensor = torch.from_numpy(test_image)
        our_blocks, our_seg, our_lines = our_detector(test_tensor)

    our_blocks = our_blocks.numpy()
    our_seg = our_seg.numpy()
    our_lines = our_lines.numpy()

    print(f"Our outputs:")
    print(f"  blocks: {our_blocks.shape}")
    print(f"  seg: {our_seg.shape}")
    print(f"  lines: {our_lines.shape}")

    # Compare shapes
    print("\nShape comparison:")
    print(f"  blk match: {orig_blk.shape == our_blocks.shape} (expected True)")
    print(f"  seg match: {orig_seg.shape == our_seg.shape} (expected True)")
    print(f"  det match: {orig_det.shape == our_lines.shape} (expected True)")

    # Analyze value distributions
    print("\nValue range analysis:")
    for ch in range(7):
        orig_vals = orig_blk[0, :, ch]
        our_vals = our_blocks[0, :, ch]
        print(f"  Channel {ch}:")
        print(f"    Original: [{orig_vals.min():.3f}, {orig_vals.max():.3f}]")
        print(f"    Ours: [{our_vals.min():.3f}, {our_vals.max():.3f}]")

if __name__ == "__main__":
    compare_outputs()
```

---

## Implementation Checklist

- [ ] Choose implementation option (A, B, or C)
- [ ] Implement changes to `BlockDetector._decode_predictions()`
- [ ] Update `BlockDetector.forward()` if using Option B
- [ ] Run unit test: `python tests/test_block_detector_output.py`
- [ ] Check output shape: `(B, 64512, 7)`
- [ ] Run comparison test: `python tests/compare_with_original.py`
- [ ] Export ONNX model
- [ ] Verify ONNX shapes match original
- [ ] Run `scripts/visualize_comparison.py` with both models
- [ ] Update training scripts if needed
- [ ] Update documentation
- [ ] Commit changes

---

## Quick Decision Guide

| Scenario | Recommended Option |
|----------|-------------------|
| **Want quick fix** | Option A (anchor index) |
| **Want to match original exactly** | Option C (reverse engineer) |
| **Want best validation metrics** | Option B (refinement score) |
| **Uncertain** | Start with Option A, then Option C |

---

## Rollback Plan

If something breaks:

```bash
# Revert changes
git checkout src/models/heads.py

# Rebuild with old code
python scripts/export.py --checkpoint runs/block_v2/best.pt --format onnx

# Re-run tests
python tests/test_block_detector_output.py
```

