# Comic Text Detector - Project Context

## Overview

Training pipeline for a comic/manga text detection model. Uses a YOLO backbone (YOLO26 or YOLOv10) with three specialized heads for different detection tasks.

## Architecture

- **Backbone**: YOLO26 (default) or YOLOv10 (configurable size: N/S/M/L/X)
  - YOLO26: NMS-free, DFL-free, 43% faster CPU inference (recommended)
  - YOLOv10: NMS-free with C2f + PSA architecture
- **Heads**:
  - UNet head for segmentation
  - DBNet head for text line detection
  - BlockDetector for YOLO-style block detection

## 3-Phase Training Pipeline

| Phase | Epochs | Task | Loss |
|-------|--------|------|------|
| 1 | 100 | Segmentation | BCE + Dice |
| 2 | 160 | Detection (DBNet) | BCE + L1 |
| 3 | 50 | Block Detection | YOLO loss |

## Dataset

> **Full documentation**: [`data/DATASET_README.md`](data/DATASET_README.md)

### Current: AnimeText (59,072 images)

| Split | Count | Source |
|-------|-------|--------|
| Training | 49,213 | `data/merged_train/` or `data/animetext/` |
| Validation | 9,859 | `data/merged_val/` or `data/animetext_val/` |

**Source**: [deepghs/AnimeText](https://huggingface.co/datasets/deepghs/AnimeText) (735K total available)

### Annotation Formats (4 types per image)

| Type | Pattern | Format | Training Phase |
|------|---------|--------|----------------|
| YOLO TXT | `{name}.txt` | `cls xcen ycen w h` (normalized) | Block Detection |
| JSON | `{name}.json` | `{"text_blocks": [{"bbox": [x1,y1,x2,y2]}]}` | Detection/Block |
| Line TXT | `line-{name}.txt` | `x1 y1 x2 y1 x2 y2 x1 y2` (polygon) | DBNet |
| Mask PNG | `mask-{name}.png` | 8-bit grayscale, 255=text | Segmentation |

### Data Preparation Scripts

```bash
# Download training data
python scripts/prepare_animetext.py --num-samples 50000 --split train --output-dir data/animetext

# Download validation data
python scripts/prepare_animetext.py --num-samples 10000 --split validation --output-dir data/animetext_val

# Merge into unified directories
python scripts/merge_datasets.py \
    --sources data/animetext \
    --output data/merged_train \
    --val-sources data/animetext_val \
    --val-output data/merged_val
```

## Key Files

| Path | Purpose |
|------|---------|
| `train.py` | Main training script |
| `src/model.py` | Model architecture |
| `src/data/dataset.py` | Unified dataset loader |
| `src/data/seg_dataset.py` | Segmentation dataset |
| `src/data/db_dataset.py` | DBNet detection dataset |
| `scripts/prepare_animetext.py` | Download & convert AnimeText |
| `scripts/merge_datasets.py` | Merge multiple data sources |
| `scripts/visualize_comparison.py` | Compare models ([docs](docs/visualize_comparison.md)) |
| `data/default.yaml` | Training config |

## Training Commands

```bash
# Full training (3 phases)
python train.py --config data/default.yaml

# Single phase
python train.py --config data/default.yaml --phase seg
python train.py --config data/default.yaml --phase detection
python train.py --config data/default.yaml --phase block
```

## Model Comparison

> **Full documentation**: [`docs/visualize_comparison.md`](docs/visualize_comparison.md)

Compare detection results between HuggingFace models:

```bash
# Compare AnimeText_yolo vs mayocream/comic-text-detector
python scripts/visualize_comparison.py --image test.jpg --output outputs/comparison
```

Models auto-download from HuggingFace if missing:
- [deepghs/AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo) - YOLO12 (640x640)
- [mayocream/comic-text-detector-onnx](https://huggingface.co/mayocream/comic-text-detector-onnx) - 3-output CTD (1024x1024)

## Backbone Selection

| Model | GPU Latency | CPU Latency | Size | Recommendation |
|-------|-------------|-------------|------|----------------|
| yolo26s | ~2-3ms | 87ms | ~20MB | Default, best balance |
| yolo26n | ~1-2ms | ~50ms | ~10MB | Edge deployment |
| yolov10s | ~4ms | ~120ms | ~17MB | Legacy support |

## Hardware Notes

- Developed on RTX 5090 (32GB VRAM)
- Recommended batch size: 16 for YOLO26-S, 4-8 for larger models
- Training time: ~24-36 hours for full pipeline

## License

- Code: MIT
- AnimeText Dataset: CC-BY-NC-SA-4.0 (non-commercial)
