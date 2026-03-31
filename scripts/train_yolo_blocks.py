#!/usr/bin/env python3
"""
Train YOLOv10s for text block detection.

This trains a separate YOLO model specifically for block detection,
following the approach used by the original comic-text-detector.

Usage:
    python scripts/train_yolo_blocks.py [--resume PATH]
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv10s for block detection")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov10s.pt",
        help="Model to use (default: yolov10s.pt)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/yolo_blocks/blocks.yaml",
        help="Path to dataset YAML"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Image size"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/yolo_blocks",
        help="Project directory"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolov10s_blocks",
        help="Experiment name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device(s)"
    )
    args = parser.parse_args()

    # Load model
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"Loading pretrained {args.model}")
        model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
        # Optimization
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,  # Final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        # Other
        patience=50,  # Early stopping patience
        save=True,
        save_period=10,
        cache=False,  # Don't cache - too many images
        workers=8,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=42,
        val=True,
        plots=True,
    )

    print(f"\nTraining complete!")
    print(f"Best model saved to: {args.project}/{args.name}/weights/best.pt")
    print(f"Last model saved to: {args.project}/{args.name}/weights/last.pt")

    return results


if __name__ == "__main__":
    main()
