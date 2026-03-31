# AnimeText Dataset for Comic Text Detector

## Overview

This dataset was prepared from [deepghs/AnimeText](https://huggingface.co/datasets/deepghs/AnimeText) on HuggingFace for training the comic-text-detector model.

| Split | Images | Annotations per Image |
|-------|--------|-----------------------|
| Training | 49,213 | 4 types |
| Validation | 9,859 | 4 types |
| **Total** | **59,072** | - |

## Directory Structure

```
data/
├── animetext/              # Raw training data (source)
│   ├── images/             # 49,213 JPG images
│   └── annotations/        # All annotation types
├── animetext_val/          # Raw validation data (source)
│   ├── images/             # 9,859 JPG images
│   └── annotations/
├── merged_train/           # Merged training set (symlinks)
│   ├── images/
│   └── annotations/
└── merged_val/             # Merged validation set (symlinks)
    ├── images/
    └── annotations/
```

## Annotation Formats

All 4 annotation types required for the 3-phase training pipeline are present:

### 1. YOLO Block Detection (`animetext_*.txt`)

```
1 0.802620 0.300000 0.098230 0.044220
1 0.648130 0.788920 0.090800 0.033540
```

- **Format**: `class_id x_center y_center width height`
- **Coordinates**: Normalized to [0, 1]
- **Class ID**: 1 = text block
- **Used by**: Phase 3 (Block Detection)

### 2. JSON Unified Detection (`animetext_*.json`)

```json
{
  "text_blocks": [
    {"bbox": [568, 269, 643, 312], "confidence": 1.0, "area": 3225.0}
  ],
  "original_size": [755, 970]
}
```

- **Format**: JSON with `text_blocks` array
- **Coordinates**: Pixel values `[x1, y1, x2, y2]`
- **Used by**: Phase 2 (Detection) and Phase 3 (Block Detection)

### 3. Line Polygon Detection (`line-animetext_*.txt`)

```
568 269 643 269 643 312 568 312
455 748 523 748 523 781 455 781
```

- **Format**: 8 space-separated integers per line
- **Structure**: 4-point polygon `x1 y1 x2 y1 x2 y2 x1 y2`
- **Coordinates**: Pixel values
- **Used by**: Phase 2 (DBNet Detection)

### 4. Segmentation Masks (`mask-animetext_*.png`)

- **Format**: 8-bit grayscale PNG
- **Values**: 0 = background, 255 = text region
- **Size**: Matches source image dimensions
- **Used by**: Phase 1 (Segmentation)

## Training Pipeline Compatibility

| Training Phase | Config Mode | Annotation Used |
|----------------|-------------|-----------------|
| Phase 1: Segmentation | `mode: seg` | `mask-*.png` |
| Phase 2: Detection | `mode: detection` | `*.json` → polygons |
| Phase 3: Block Detection | `mode: block` | `*.json` |

## How to Use

### For Training

Update your training config to point to the merged directories:

```yaml
# In your training config
train_dir: data/merged_train
val_dir: data/merged_val
```

Or use the raw directories directly:

```yaml
train_dir: data/animetext
val_dir: data/animetext_val
```

### To Regenerate or Expand

Download more samples from AnimeText:

```bash
# Download training data (50,000 samples)
python scripts/prepare_animetext.py \
    --num-samples 50000 \
    --split train \
    --output-dir data/animetext

# Download validation data (10,000 samples)
python scripts/prepare_animetext.py \
    --num-samples 10000 \
    --split validation \
    --output-dir data/animetext_val

# Merge datasets
python scripts/merge_datasets.py \
    --sources data/animetext \
    --output data/merged_train \
    --val-sources data/animetext_val \
    --val-output data/merged_val
```

### Available Splits

The AnimeText dataset has 3 splits:
- `train`: ~514,000 images
- `valid`: ~147,000 images
- `test`: ~74,000 images

## Source Dataset Info

- **Name**: AnimeText
- **Source**: [deepghs/AnimeText](https://huggingface.co/datasets/deepghs/AnimeText)
- **Paper**: arXiv 2510.07951
- **Total Size**: 735,000 images, 4.2 million annotations
- **License**: CC-BY-NC-SA-4.0 (non-commercial)

## Preparation Scripts

| Script | Purpose |
|--------|---------|
| `scripts/prepare_animetext.py` | Download and convert AnimeText to all 4 formats |
| `scripts/merge_datasets.py` | Merge multiple data sources via symlinks |

## Notes

- Images are saved as JPEG (quality 95)
- Symlink structure avoids data duplication
- Some images with no text objects are skipped during download
- Confidence is set to 1.0 for all annotations (ground truth)
