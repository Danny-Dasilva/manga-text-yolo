# Comic Text Detector - Architecture Validation Report

**Validation Date:** 2026-01-16
**Status:** NEEDS REVIEW ⚠️

## Quick Summary

Your training architecture produces **2 out of 3 outputs** that exactly match the original comic-text-detector model:

| Output | Match Status | Details |
|--------|-------------|---------|
| **seg** (segmentation) | ✅ EXACT MATCH | Shape: (1, 1, 1024, 1024) |
| **det** (text lines) | ✅ EXACT MATCH | Shape: (1, 2, 1024, 1024) |
| **blk** (blocks) | ❌ MISMATCH | Shape: (1, N, 6) vs (1, 64512, 7) |

## The Problem: Block Detection Output

### What We Have
```
BlockDetector outputs: (1, N_detections, 6)
├─ Position 0-1: x, y (normalized 0-1)
├─ Position 2-3: w, h (normalized 0-1)
├─ Position 4: confidence score
└─ Position 5: class score
```

### What We Need
```
Original model outputs: (1, 64512, 7)
├─ Dense grid format (all grid cells, not just detections)
├─ Position 0-3: x, y, w, h
├─ Position 4-6: Unknown (3 mystery values)
└─ Layout: All P3 cells → All P4 cells → All P5 cells
```

### The Numbers Match (Good!)
```
P3 (stride 8):   128 × 128 × 3 anchors = 49,152 ✓
P4 (stride 16):  64 × 64 × 3 anchors = 12,288 ✓
P5 (stride 32):  32 × 32 × 3 anchors = 3,072 ✓
─────────────────────────────────────
Total:                              64,512 ✓ (matches original)
```

## What Matches Perfectly

### 1. Segmentation Mask (seg)
- **Our output:** UnetHead → (1, 1, 1024, 1024) with Sigmoid
- **Original:** (1, 1, 1024, 1024)
- **Verdict:** Perfect match ✅

Code location: `src/models/heads.py` line 369-372

### 2. Text Line Detection (det)
- **Our output:** DBHead → (1, 2, 1024, 1024) concatenating [shrink_map, threshold_map]
- **Original:** (1, 2, 1024, 1024)
- **Verdict:** Perfect match ✅

Code location: `src/models/heads.py` line 521-525

## Critical Issue: Block Detection (blk)

### Mismatch Details

| Aspect | Original | Ours | Issue |
|--------|----------|------|-------|
| Shape | (1, 64512, 7) | (1, N, 6) | Different dimensions |
| Format | Dense grid | Detection list | Different representation |
| 7th value | Unknown* | Missing | Unknown parameter missing |
| Training output | Same format | (o2m, o2o) tuple | Different during training |

*The 7th value could be: anchor index, refinement score, auxiliary confidence, scale factor, or geometric property

### Code Location
`src/models/heads.py`, BlockDetector class, lines 536-712

## Recommended Fixes

### Fix Option 1: Dense Grid Conversion (RECOMMENDED)
Convert BlockDetector to output dense grid format:

```python
# Current (wrong):
return blocks  # Shape: (B, N, 6)

# Needed (correct):
# 1. Create 64512-element grid
# 2. Place each detection in its grid cell
# 3. Add 7th value for each cell
# 4. Sort by P3→P4→P5 order
return dense_grid  # Shape: (B, 64512, 7)
```

**Pro:** True compatibility with original
**Con:** May lose variable-length detection advantage
**Effort:** Moderate - need to rewrite BlockDetector.forward()

### Fix Option 2: Analyze Original Model
Use ONNX tools to reverse-engineer what the 7th value is:

```bash
# Use netron.app to visualize the model
# Or use onnx-simplifier to understand the graph
```

**Pro:** Exact compatibility guaranteed
**Con:** Time to analyze
**Effort:** Low - just investigation

### Fix Option 3: Post-Processing Conversion
Keep current output, add conversion in export wrapper:

```python
# In scripts/export.py ExportableModelV10.forward()
blocks = self.block_detector(...)  # (B, N, 6)
blocks_dense = convert_to_dense_grid(blocks)  # (B, 64512, 7)
return blocks_dense, mask, lines
```

**Pro:** Doesn't break training pipeline
**Con:** Extra post-processing complexity
**Effort:** Low - wrapper approach

## Action Items

1. **Immediate:** Decide which fix option to implement
2. **Quick test:** Run inference with current export to confirm mismatch
3. **Implement:** Convert block detection output to dense grid (Option 1 or 3)
4. **Validate:** Compare exported model against original using `visualize_comparison.py`
5. **Test:** Ensure downstream inference code handles 7-value format

## Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `src/models/heads.py` | BlockDetector.forward() - add dense grid logic | HIGH |
| `scripts/export.py` | ExportableModelV10 - handle 7th value | HIGH |
| `src/inference/compiled_model.py` | Verify post-processing compatibility | MEDIUM |

## Files That Are Correct ✅

- `src/models/heads.py` - UnetHead (seg output) ✓
- `src/models/heads.py` - DBHead (det output) ✓
- `scripts/export.py` - seg/det export handling ✓

## Verification Commands

```bash
# Check current export shape
python -c "
import onnx
m = onnx.load('models/detector.onnx')
for o in m.graph.output:
    print(f'{o.name}: {[d.dim_value for d in o.type.tensor_type.shape.dim]}')
"

# Compare with original
python -c "
import onnx
m = onnx.load('data/comic-text-detector.onnx')
for o in m.graph.output:
    print(f'{o.name}: {[d.dim_value for d in o.type.tensor_type.shape.dim]}')
"
```

Expected original output:
```
blk: [1, 64512, 7]
seg: [1, 1, 1024, 1024]
det: [1, 2, 1024, 1024]
```

Expected our current output (after fix):
```
blk: [1, 64512, 7]  <- WILL BE FIXED
seg: [1, 1, 1024, 1024]  ✓
det: [1, 2, 1024, 1024]  ✓
```

## Confidence Levels

| Finding | Confidence | Evidence |
|---------|-----------|----------|
| seg output matches | **HIGH** | Direct shape comparison, code inspection |
| det output matches | **HIGH** | Direct shape comparison, code inspection |
| blk mismatch exists | **HIGH** | BlockDetector clearly outputs (B, N, 6) not (B, 64512, 7) |
| 7th value purpose unknown | **MEDIUM** | Requires ONNX graph analysis |

---

For detailed analysis, see: `.claude/cache/agents/validate-agent/latest-output.md`
