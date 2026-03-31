# Handoff: Block Detection v5 — Full Architecture Fix + Training

## Context

We're training a manga text detection model (YOLO26s backbone + 3 heads: segmentation, DBNet, block detection). Phase 3 (block detection) was fundamentally broken across multiple dimensions. This session diagnosed all root causes via 10 parallel investigation agents and implemented two rounds of fixes.

## What Was Done

### Round 1 Investigation: 10 Parallel Agents

Spawned 10 sleuth agents to investigate every layer of the training pipeline. Found **8 bugs** and confirmed data pipeline, anchor config, and annotation format were correct.

### Round 1 Fixes (8 bugs)

| # | Bug | Fix | File |
|---|-----|-----|------|
| 1 | `model.train()` overrides frozen module `eval()` — BatchNorm stats drift | Call phase-specific `train_block()` | trainer.py:1011-1028 |
| 2 | Zero gradient when no fg matches (`pred_obj.sum() * 0.0`) | Return real BCE loss × 0.1 | losses.py:764-766 |
| 3 | Missing objectness bias init (0 instead of -4.6) | Init bias to `log(0.01/0.99)` | heads.py:593-614 |
| 4 | IoU^6 (`beta=6.0`) cold-start death spiral | Reduced to beta=2.0 (later restored, see round 2) | losses.py:434 |
| 5 | `obj * cls` kills single-class confidence | Use `obj` alone for nc=1 | detector.py:514-520 |
| 6 | BGR/RGB mismatch (train=RGB, inference=BGR) | Added `cvtColor(BGR2RGB)` | detector.py:332-338 |
| 7 | FocalLoss alpha=0.5 for 1:2149 fg/bg ratio | Increased to 0.75 | losses.py:667 |
| 8 | Unified mode block loss silently skipped | Added `'blocks'` to condition | losses.py:1035 |

### Round 1 Training (v4_fixed, 76 epochs)

- Max confidence improved from 0.066 → 0.350
- TP improved from 12 → 230 (on 200 val images)
- But: confidence hit ceiling at ~0.35, FP growing 7x faster than TP
- Root cause: FocalLoss gamma=2.0 suppresses foreground gradients, obj*cls architecture creates mathematical ceiling at ~0.35

### Round 2 Investigation: 5 Research Agents

Spawned 5 agents researching confidence improvement strategies. Key findings:

1. **VariFocal Loss** — asymmetric: full gradient on positives, focal only on negatives. Used by modern YOLO variants.
2. **Merge obj+cls** — for nc=1, having separate obj and cls is redundant. YOLOv8+ dropped obj head entirely.
3. **TaskAlignedAssigner beta=6.0** is now safe (zero-gradient bug fixed) and creates positive feedback loop.
4. **block_lr too low** — was 3e-4, should be 1e-3 for fresh head training.
5. **NMS needed** — o2o "NMS-free" design assumes converged model. Unconverged model needs NMS as safety net.

### Round 2 Fixes

| # | Change | File |
|---|--------|------|
| 1 | Replace FocalLoss with **VariFocalLoss** + soft IoU targets for objectness | losses.py (new class + _compute_loss) |
| 2 | **Merge obj+cls** into single head: `self.no = 5` for nc=1 | heads.py:561 |
| 3 | Remove cls_loss for nc=1 (obj is the only confidence signal) | losses.py:_compute_loss |
| 4 | **TaskAlignedAssigner**: alpha=1.0, beta=6.0 | losses.py:433-434 |
| 5 | **topk**: 10→13 for o2m | losses.py:661 |
| 6 | **block_lr**: 3e-4→1e-3 | trainer.py:93 |
| 7 | **min_lr**: 1e-6→1e-4 | trainer.py:101 |
| 8 | **EMA decay**: 0.999→0.9999 | trainer.py:109 |
| 9 | **Add NMS** (IoU=0.4) to inference | detector.py:570-582 |
| 10 | **Max 100 detections** per image | detector.py:585 |
| 11 | **Min area filter** (200 sq px) | detector.py:557-558 |

### Post-Implementation Audit (5 Parallel Agents)

Spawned 5 critic/sleuth agents to audit all changes. Found and fixed 3 additional issues:

| # | Audit Finding | Fix Applied |
|---|--------------|-------------|
| 1 | BlockDetectionLoss silently computed cls_loss on empty [B,N,0] tensor for no=5 | Added `pred_cls = None` guard, skip cls_loss for merged head |
| 2 | VFL obj_loss used `.mean()` (anchor-count-dependent) | Changed to `.sum() / max(num_fg, 1)` |
| 3 | `assigned_scores.squeeze(-1)` fails for num_classes>1 | Changed to `.max(dim=-1).values` |
| 4 | torchvision NMS import inside hot path | Moved to module-level import |
| 5 | Zero-fg fallback had unjustified `* 0.1` multiplier | Removed — VFL focal weighting handles this |
| 6 | cosine_warmup_scheduler used `config.lr` instead of `block_lr` for phase 3 | Use phase-aware `base_lr` |
| 7 | OneCycleLR ignored `block_lr` | Use `effective_lr` for phase 3 |
| 8 | `_train_epoch` unwrap order inconsistent with rest of codebase | Fixed to `.model` first, then `._orig_mod` |
| 9 | v2 checkpoint (phase 3) would crash with strict=True due to no=6→5 mismatch | Use phase 2 `detection_v6/best.pt` instead |
| 10 | Gradient clip 1.0 too tight for fresh head + VFL | Raised to 10.0 |

### Verification

All changes verified with end-to-end test:
- BlockDetector nc=1 outputs [B, 21504, 5] (was [B, 21504, 6])
- VariFocalLoss computes gradients correctly
- Both DualAssignmentLoss and BlockDetectionLoss pass forward+backward with nc=1
- Zero-fg case produces real gradients (not zero)
- Bias init confirmed at -4.595

## What Needs To Be Done

### 1. Run Fresh Phase 3 Training (v5)

```bash
cd /home/danny/Documents/personal/extension/training/comic-text-detector

.venv/bin/python scripts/train.py \
    --no-unified --training-phase 3 --mode block \
    --checkpoint runs/detection_v6/best.pt \
    --freeze-seg --freeze-det --freeze-projections \
    --img-dirs data/merged_train/images \
    --block-ann-dirs data/merged_train/block_annotations_ensemble \
    --val-img-dirs data/merged_val/images \
    --epochs 100 --batch-size 16 --lr 1e-3 --optimizer adamw \
    --backbone yolo26s.pt --save-dir runs/block_ensemble_v5 --no-wandb \
    --accumulation-steps 1
```

**Important notes:**
- Uses **phase 2** checkpoint `detection_v6/best.pt` (NOT v2 which is phase 3 and would crash due to no=6→no=5 shape mismatch with strict=True)
- Cross-phase resume (phase 2→3) triggers strict=False, block_det initializes fresh
- Block head trains from scratch (new architecture: 5 outputs instead of 6)
- LR is 1e-3 (3x higher than round 1), min_lr is 1e-4
- Gradient clip raised to 10.0 (was 1.0, too tight for fresh head)
- `--accumulation-steps 1` to avoid scheduler bug (see learnings)

### 2. Evaluate at Epochs 25, 50, 75, 100

```bash
.venv/bin/python scripts/evaluate_vs_gt.py \
    --checkpoint runs/block_ensemble_v5/best.pt \
    --block-ann-dir data/merged_val/block_annotations_ensemble \
    --limit 200 --conf-threshold 0.1 \
    --output-dir outputs/eval_v5
```

Also check confidence distribution:
```bash
.venv/bin/python -c "
import cv2, torch, sys
sys.path.insert(0, '.')
from src.inference.detector import TextDetector
det = TextDetector('runs/block_ensemble_v5/best.pt', input_size=1024, device='cuda', conf_threshold=0.01, backend='pytorch')
img = cv2.imread('data/merged_val/images/animetext_1000011.jpg')
result = det(img)
confs = [b['confidence'] for b in result.text_blocks]
print(f'Max conf: {max(confs):.4f}, boxes: {len(confs)}')
for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
    print(f'  conf >= {t}: {sum(1 for c in confs if c >= t)}')
"
```

### 3. Expected Targets

| Metric | v4 (round 1) | v5 Target | Teacher (YOLO12-X) |
|--------|-------------|-----------|---------------------|
| Max confidence | 0.35 | >0.7 | 0.96 |
| F1 (conf>0.3) | 0.000 | >0.3 | 0.831 |
| Recall | 0.169 | >0.5 | 0.724 |
| Precision | 0.001 | >0.3 | 0.975 |

### 4. If Performance Is Still Low

- Check if VariFocalLoss positive targets (IoU values) are too low → may need to use `max(iou, 0.5)` floor
- Try unfreezing P5 backbone layer (--no-freeze-projections or partial unfreeze)
- Consider reducing o2m anchors_per_scale from 3→1 (less memory, no expressivity loss)
- Try longer training (200 epochs) — with min_lr=1e-4, late epochs are still productive

## Key Learnings

1. **model.train() overrides eval()** — Always use phase-specific train methods, never blanket model.train()
2. **Zero-gradient traps** — Any `num_fg == 0` fallback MUST produce real gradients, not `* 0.0`
3. **obj*cls is wrong for single-class** — Merge into single confidence for nc=1
4. **FocalLoss suppresses foreground** — VariFocalLoss is strictly better (asymmetric treatment)
5. **Bias init matters** — Without prior=0.01 init, model wastes epochs unlearning 0.5 objectness
6. **NMS is always needed** — Even "NMS-free" designs need it during early training/unconverged
7. **--checkpoint vs --resume** — `--checkpoint` resets optimizer/epoch, `--resume` continues
8. **BGR/RGB matters** — Always verify color space matches between train and inference
9. **block_lr override** — trainer.py:466 overrides --lr with config.block_lr for phase 3
10. **EMA decay** — 0.999 too aggressive, smooths away confidence gains. Use 0.9999

## Key Files Modified

| File | Changes |
|------|---------|
| `src/training/losses.py` | Added VariFocalLoss class, soft IoU targets, removed cls_loss for nc=1, TAL alpha=1.0/beta=6.0, topk=13 |
| `src/models/heads.py` | Merged head (no=5 for nc=1), bias init with bounds check |
| `src/inference/detector.py` | BGR→RGB fix, NMS (IoU=0.4), max 100 detections, min area 200px, merged head confidence |
| `src/training/trainer.py` | Phase-specific train mode, block_lr=1e-3, min_lr=1e-4, ema_decay=0.9999 |

## Data & Weights Setup (Not in Git)

Large files are NOT in git. To set up on a new machine:

### Required for Training
```bash
# 1. Backbone weights (download or copy from NAS)
# yolo26s.pt goes in repo root
cp /path/to/nas/manga-text-yolo/yolo26s.pt .

# 2. Phase 2 checkpoint (base for phase 3 training)
mkdir -p runs/detection_v6
cp /path/to/nas/manga-text-yolo/runs/detection_v6/best.pt runs/detection_v6/

# 3. Training data (49K train + 10K val images + annotations)
mkdir -p data/merged_train data/merged_val
# Images
cp -r /path/to/nas/manga-text-yolo/data/merged_train/images data/merged_train/
cp -r /path/to/nas/manga-text-yolo/data/merged_val/images data/merged_val/
# Ensemble block annotations
cp -r /path/to/nas/manga-text-yolo/data/merged_train/block_annotations_ensemble data/merged_train/
cp -r /path/to/nas/manga-text-yolo/data/merged_val/block_annotations_ensemble data/merged_val/
# GT annotations (for evaluation)
cp -r /path/to/nas/manga-text-yolo/data/merged_val/annotations data/merged_val/
```

### Optional
```bash
# Teacher model (for comparison only)
mkdir -p models/animetext-yolo/yolo12x_animetext
cp /path/to/nas/manga-text-yolo/models/animetext-yolo/yolo12x_animetext/model.onnx models/animetext-yolo/yolo12x_animetext/

# Previous training runs (for reference)
cp -r /path/to/nas/manga-text-yolo/runs/block_ensemble_v2 runs/
cp -r /path/to/nas/manga-text-yolo/runs/block_ensemble_v4_fixed runs/
```

## Key Paths

| Path | Description | In Git? |
|------|-------------|---------|
| `src/models/` | Model architecture (backbone, heads, detector) | Yes |
| `src/data/` | Dataset loaders | Yes |
| `src/training/` | Trainer, losses (with all v5 fixes) | Yes |
| `src/inference/` | Inference pipeline (with NMS, BGR fix) | Yes |
| `scripts/train.py` | Training script | Yes |
| `yolo26s.pt` | Backbone weights | No — NAS |
| `runs/detection_v6/best.pt` | Phase 2 checkpoint (base for v5) | No — NAS |
| `data/merged_train/` | 49,213 training images + annotations | No — NAS |
| `data/merged_val/` | 9,859 validation images + annotations | No — NAS |
| `models/animetext-yolo/` | Teacher model (YOLO12-X) | No — NAS |

## Training History Summary

| Run | Epochs | Best Loss | Max Conf | F1 | Issue |
|-----|--------|-----------|----------|-----|-------|
| block_ensemble (v1) | 50 | NaN | — | — | Bug: 'blocks' key not supported |
| block_ensemble_v2 | 50 | 3.56 | 0.58 | 0.194 | Bug: model.train() override, zero gradient |
| block_ensemble_v3 | 60+43 | 3.62 | 0.25 | 0.004 | Same bugs, used --checkpoint instead of --resume |
| block_ensemble_v4_fixed | 76 | 3.71 | 0.35 | 0.006 | Round 1 fixes. obj*cls ceiling at 0.35 |
| block_ensemble_v5 | TBD | TBD | TBD | TBD | Round 2: VFL + merged head + full fixes |
