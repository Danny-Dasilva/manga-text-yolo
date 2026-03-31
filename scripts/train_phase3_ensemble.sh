#!/bin/bash
# Phase 3: Block detection training with ensemble annotations
#
# Prerequisites:
#   - Phase 2 checkpoint: runs/detection_yolo26/best.pt
#   - Ensemble annotations: data/merged_{train,val}/block_annotations_ensemble/
#
# Run: bash scripts/train_phase3_ensemble.sh

set -e

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f "runs/detection_yolo26/best.pt" ]; then
    echo "ERROR: Phase 2 checkpoint not found at runs/detection_yolo26/best.pt"
    exit 1
fi

TRAIN_ANN_COUNT=$(ls data/merged_train/block_annotations_ensemble/*.json 2>/dev/null | wc -l)
VAL_ANN_COUNT=$(ls data/merged_val/block_annotations_ensemble/*.json 2>/dev/null | wc -l)

echo "Training annotations: $TRAIN_ANN_COUNT"
echo "Validation annotations: $VAL_ANN_COUNT"

if [ "$TRAIN_ANN_COUNT" -lt 1000 ]; then
    echo "ERROR: Too few training annotations ($TRAIN_ANN_COUNT). Wait for ensemble annotation to complete."
    exit 1
fi

echo ""
echo "=========================================="
echo "Phase 3: Block Detection (Ensemble Labels)"
echo "=========================================="
echo ""

.venv/bin/python scripts/train.py \
    --no-unified \
    --training-phase 3 \
    --mode block \
    --checkpoint runs/detection_yolo26/best.pt \
    --freeze-seg \
    --freeze-det \
    --freeze-projections \
    --img-dirs data/merged_train/images \
    --block-ann-dirs data/merged_train/block_annotations_ensemble \
    --val-img-dirs data/merged_val/images \
    --epochs 50 \
    --batch-size 16 \
    --accumulation-steps 1 \
    --lr 3e-4 \
    --optimizer adamw \
    --backbone yolo26s.pt \
    --save-dir runs/block_ensemble \
    --no-wandb \
    "$@"
