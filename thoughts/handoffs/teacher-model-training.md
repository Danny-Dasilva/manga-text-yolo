# Handoff: Teacher Model Training — Phase 3 Block Detection

## Context

We're training a manga text detection model (YOLO26s backbone + 3 heads: segmentation, DBNet, block detection). This session focused on improving Phase 3 (block detection) training labels using better teacher models.

## What Was Done

### 1. Teacher Model Research & Evaluation
- Downloaded **YOLO12-X** (225MB) from `deepghs/AnimeText_yolo` — trained on 735K images
- Evaluated 3 models against AnimeText ground truth (500 images):
  - YOLO12-X: F1=0.831, Precision=0.975, Recall=0.724
  - YOLO12-S: F1=0.826
  - CTD (old): F1=0.616 — 35% worse than YOLO12-X

### 2. Ensemble Annotation Pipeline
- Created `scripts/generate_ensemble_annotations.py`
- Strategy: YOLO12-X as primary teacher, Grounding DINO (zero-shot) as validation signal
- 63% of YOLO12-X boxes independently confirmed by Grounding DINO
- **Re-annotated all 59,072 images** (49,213 train + 9,859 val)
- Output: `data/merged_{train,val}/block_annotations_ensemble/`
- Quality: F1=0.831, Avg IoU=0.921 vs old CTD annotations F1=0.646, IoU=0.811

### 3. Bug Fix: BlockDetectionLoss
- **Critical bug found**: `src/training/losses.py` `BlockDetectionLoss._compute_single_head_loss()` only accepted `targets['boxes']` but the dataset provides `targets['blocks']`
- First 50-epoch run (`runs/block_ensemble/`) learned nothing — loss decreased but from NaN handler, not real box regression
- **Fixed at line ~887**: Added `'blocks'` key support matching `DualAssignmentLoss` pattern
- Second run (`runs/block_ensemble_v2/`) shows real learning: loss 5.22→3.56, model produces meaningful boxes

### 4. Inference Preprocessing Fix
- **Bug**: `src/inference/detector.py` `_preprocess_single()` used letterbox center-padding + BGR→RGB
- **Training uses**: albumentations `Resize(1024, 1024)` direct stretch + keeps BGR (via `ToTensorV2`)
- **Fixed**: Block detector mode now uses direct resize, no color conversion (matches training)
- Also fixed `_extract_blocks_from_yolo()` to handle `direct_resize` coordinate mapping

### 5. Current Model State (v2)
- Checkpoint: `runs/block_ensemble_v2/best.pt` (50 epochs, loss=3.56)
- Model produces meaningful predictions (14/22 GT boxes matched at low threshold)
- Max confidence is 0.58 — needs more training to raise confidence scores
- Loss was still decreasing at epoch 50 (not converged)

## What Needs To Be Done

### Resume Phase 3 Training (100 more epochs)

```bash
cd /home/danny/Documents/personal/extension/training/comic-text-detector

.venv/bin/python scripts/train.py \
    --no-unified --training-phase 3 --mode block \
    --checkpoint runs/block_ensemble_v2/best.pt \
    --freeze-seg --freeze-det --freeze-projections \
    --img-dirs data/merged_train/images \
    --block-ann-dirs data/merged_train/block_annotations_ensemble \
    --val-img-dirs data/merged_val/images \
    --epochs 100 --batch-size 16 --lr 5e-4 --optimizer adamw \
    --backbone yolo26s.pt --save-dir runs/block_ensemble_v3 --no-wandb
```

### After Training Completes
1. Evaluate with `scripts/evaluate_teacher.py` against GT
2. Test inference with `scripts/evaluate_vs_gt.py --checkpoint runs/block_ensemble_v3/best.pt`
3. If confidence is still low, consider:
   - Using `obj` score alone instead of `obj * cls` in `_extract_blocks_from_yolo`
   - Training with higher LR or unfreezing some backbone layers
   - More epochs (loss may need 200+ epochs to converge for block detection)

## Key Files Modified

| File | Change |
|------|--------|
| `src/training/losses.py:~887` | Added `'blocks'` key support in `_compute_single_head_loss` |
| `src/inference/detector.py:310-370` | Block detector uses direct resize, keeps BGR |
| `src/inference/detector.py:468-554` | `_extract_blocks_from_yolo` handles `direct_resize` flag |
| `scripts/generate_ensemble_annotations.py` | NEW — ensemble annotation pipeline |
| `scripts/evaluate_teacher.py` | NEW — teacher model evaluation vs GT |
| `scripts/train_phase3_ensemble.sh` | NEW — Phase 3 training launcher |

## Key Paths

| Path | Description |
|------|-------------|
| `models/animetext-yolo/yolo12x_animetext/model.onnx` | YOLO12-X teacher (225MB) |
| `data/merged_train/block_annotations_ensemble/` | 49,213 ensemble annotations |
| `data/merged_val/block_annotations_ensemble/` | 9,859 ensemble annotations |
| `data/merged_val/annotations/` | AnimeText ground truth |
| `runs/block_ensemble_v2/best.pt` | Current best model (50 epochs) |
| `runs/block_ensemble_v2/training_history.json` | Training loss history |
