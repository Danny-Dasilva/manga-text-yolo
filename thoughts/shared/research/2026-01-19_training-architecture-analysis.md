# Comic Text Detector Training Architecture Analysis

**Date**: 2026-01-19
**Status**: Complete
**Agents Used**: 10 parallel research agents

---

## Executive Summary

Comprehensive analysis of the 3-phase training pipeline identified **multiple critical issues** causing:
1. **DBNet lines output nearly invisible** (max 0.6 instead of ~1.0)
2. **Block detector low confidence** (~0.05-0.1 instead of 0.4+)
3. **113 vs 22 boxes** (granularity mismatch - expected behavior)

**Primary Recommendation**: Switch from 3-phase sequential training to **hybrid unified training**.

---

## Issues Identified

### 1. DBNet Weak Output (Lines Don't Appear)

**Root Causes**:

| Issue | Location | Problem |
|-------|----------|---------|
| No bias initialization | `src/models/heads.py:459-467` | Final ConvTranspose2d bias=0, sigmoid(0)=0.5 |
| Loss weight inverted | `src/training/losses.py:230` | beta=2.0 (threshold) > alpha=1.0 (shrink) |

**Fix Required**:
```python
# heads.py - Add after binarize Sequential definition
nn.init.constant_(self.binarize[-1].bias, 2.0)  # sigmoid(2) ≈ 0.88

# losses.py - Change beta from 2.0 to 1.0
beta: float = 1.0  # Was 2.0
```

### 2. Block Detector Low Confidence

**Root Causes**:

| Issue | Location | Problem |
|-------|----------|---------|
| Wrong confidence calc | `src/inference/detector.py:494` | Uses only `pred[4]` not `obj × cls` |
| FocalLoss alpha too low | `src/training/losses.py:667` | alpha=0.25 gives 3x weight to background |
| Box loss dominates | `src/training/losses.py:647-649` | box=5.0, obj=1.5, cls=1.0 |

**Fix Required**:
```python
# detector.py:494 - Fix confidence calculation
obj_conf = float(pred[4])
cls_conf = float(pred[5]) if pred.shape[-1] > 5 else 1.0
conf = obj_conf * cls_conf  # Was: conf = float(pred[4])

# losses.py:667 - Increase alpha
self.obj_loss = FocalLoss(alpha=0.5, gamma=2.0)  # Was alpha=0.25

# losses.py:647-649 - Rebalance weights
box_weight: float = 5.0
obj_weight: float = 2.5  # Was 1.5
cls_weight: float = 1.5  # Was 1.0
```

### 3. Granularity Mismatch (113 vs 22 boxes)

**Finding**: This is **expected behavior**, not a bug.
- GT annotations are **block-level** (speech bubbles)
- Model detects **line-level** (individual text lines)
- Ratio: 113/22 = 5.1x (each block has ~5 lines)

**Decision Needed**: Accept fine-grained output OR re-annotate dataset with line-level GT.

### 4. Training Pipeline Issues

| Issue | Severity | Location | Problem |
|-------|----------|----------|---------|
| Weight deletion | 🔴 HIGH | `heads.py:801-805` | UNet layers deleted during Phase 2 init |
| Missing gradient guard | 🟡 MEDIUM | `heads.py:919-922` | No `torch.no_grad()` for frozen UNet |
| Wrong BatchNorm state | 🟡 MEDIUM | `heads.py:858-875` | Frozen layers in train() instead of eval() |
| LR scheduler reset | 🟡 MEDIUM | `trainer.py:1076-1100` | Optimizer state lost across phases |

**Fix Required**:
```python
# heads.py:919-922 - Add gradient guard
elif self.forward_mode == TEXTDET_DET:
    with torch.no_grad():  # ADD THIS
        seg_features = self.seg_net(*features, forward_mode=self.forward_mode)
    return self.dbnet(*seg_features)

# heads.py:858-875 - Use eval() for frozen layers
def train_block(self):
    self.seg_net.eval()  # Was train()
    if self.dbnet is not None:
        self.dbnet.eval()  # Was train()
```

### 5. Original vs Our Architecture

| Aspect | Original CTD | Our Implementation |
|--------|-------------|-------------------|
| Training phases | 2 + separate block | 3 integrated |
| Block detector | Official YOLOv5 separate | Integrated Phase 3 |
| Weight sharing | Copy then delete | Copy then delete (broken) |
| Feature sharing | Separate | Poor (phased isolation) |

### 6. YOLO26 Features Not Used

| Feature | Purpose | Status |
|---------|---------|--------|
| ProgLoss | Dynamic loss weighting | Not implemented |
| STAL | Small-Target-Aware Assignment | Not implemented |
| MuSGD | Hybrid optimizer | Using AdamW instead |
| Native NMS-free | End-to-end inference | Using dual-assignment |

---

## Recommended Training Strategy

### Current (Problematic)
```
Phase 1 (100 epochs): Seg only, backbone frozen
    ↓ (weight deletion breaks continuity)
Phase 2 (160 epochs): DBNet only, seg frozen
    ↓ (seg_net randomly reinitialized!)
Phase 3 (50 epochs): Block only, seg+det frozen
```

### Recommended: Hybrid Unified Training
```
Phase 1 (50 epochs): ALL heads jointly, backbone frozen
    - Equal loss weights initially
    - Feature sharing between heads

Phase 2 (100 epochs): ALL heads + backbone unfrozen
    - Lower backbone LR (1e-5 vs 1e-4)
    - Uncertainty weighting for loss balancing
    - ProgLoss-style dynamic reweighting
```

**Benefits**:
- Excellent feature sharing between heads
- No weight deletion/corruption
- DBNet benefits from joint optimization
- Handles loss scale mismatch automatically

---

## Implementation Checklist

### Immediate Fixes (No Retraining)
- [ ] Fix block confidence: `conf = obj * cls`
- [ ] Add DBNet bias init: `nn.init.constant_(bias, 2.0)`

### Retraining Required
- [ ] Fix DBNet loss beta: 2.0 → 1.0
- [ ] Fix FocalLoss alpha: 0.25 → 0.5
- [ ] Rebalance loss weights: obj 1.5→2.5, cls 1.0→1.5
- [ ] Add `torch.no_grad()` for frozen forward
- [ ] Fix BatchNorm state (eval for frozen)
- [ ] Remove weight deletion in initialize_db()
- [ ] Implement hybrid unified training

### Optional Enhancements
- [ ] Implement ProgLoss (dynamic loss weighting)
- [ ] Implement STAL (small target assignment)
- [ ] Switch to MuSGD optimizer
- [ ] Adjust lrf from 0.000001 to 0.01

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/models/heads.py` | Bias init, gradient guard, BatchNorm state, remove deletion |
| `src/training/losses.py` | Beta, alpha, loss weights |
| `src/inference/detector.py` | Confidence calculation |
| `train.py` | Hybrid unified training mode |
| `src/training/trainer.py` | LR handling, phase transitions |

---

## Research Sources

1. Original CTD: https://github.com/dmMaze/comic-text-detector
2. YOLO26 docs: https://docs.ultralytics.com/models/yolo26/
3. DBNet paper: https://arxiv.org/pdf/1911.08947
4. Multi-task learning: Kendall et al. CVPR 2018 (uncertainty weighting)
5. PCGrad: NeurIPS 2020 (gradient conflict mitigation)

---

## Agent Findings Summary

| Agent | Focus | Key Finding |
|-------|-------|-------------|
| 1 | Original CTD | 2-phase + separate block detector |
| 2 | Local architecture | Sound design, poor implementation |
| 3 | Multi-head training | Recommend hybrid unified |
| 4 | Checkpoint structure | Phase 2 loses 109 seg_net keys |
| 5 | GT vs predictions | 5.1x granularity mismatch (expected) |
| 6 | DBNet weak output | No bias init + wrong loss weights |
| 7 | Block confidence | Wrong calculation + FocalLoss alpha |
| 8 | YOLO26 practices | Missing ProgLoss, STAL, MuSGD |
| 9 | 3-phase training | Weight deletion, BatchNorm, gradients |
| 10 | Loss/hyperparams | DBNet beta inverted, box loss dominates |
