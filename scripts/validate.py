#!/usr/bin/env python3
"""Validation script for trained models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from tqdm import tqdm

from src.models import create_text_detector
from src.data import create_dataloader
from src.training import create_loss


def validate(model, val_loader, criterion, device):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch['image'].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != 'image'}

            outputs = model(images)
            loss = criterion(outputs, targets)

            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description='Validate trained model')
    parser.add_argument('--mode', type=str, required=True, choices=['segmentation', 'detection'],
                        help='Validation mode')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--val-img-dirs', type=str, nargs='+', required=True,
                        help='Validation image directories')
    parser.add_argument('--val-ann-dirs', type=str, nargs='+', required=True,
                        help='Validation annotation directories')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Checkpoint: {args.checkpoint}")

    # Create model
    print("\nLoading model...")
    model = create_text_detector(device=device)

    # Initialize DB head for detection mode
    if args.mode == 'detection':
        model.initialize_db()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Get state dict from checkpoint
    if 'ema_state_dict' in checkpoint:
        state_dict = checkpoint['ema_state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove '_orig_mod.' prefix from torch.compile() saved models
    # Also handle dbnet -> seg_net renaming for detection checkpoints
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove _orig_mod. prefix
        if k.startswith('_orig_mod.'):
            k = k[10:]
        # Map dbnet -> seg_net for compatibility
        if k.startswith('dbnet.'):
            k = 'seg_net.' + k[6:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

    model = model.to(device)

    # Set model mode
    if args.mode == 'segmentation':
        model.train_mask()
        ann_dirs = args.val_ann_dirs
    else:
        model.train_db()
        ann_dirs = args.val_ann_dirs

    # Create validation dataloader
    print("Creating validation dataloader...")
    val_loader = create_dataloader(
        img_dirs=args.val_img_dirs,
        annotation_dirs=ann_dirs,
        mode=args.mode,
        img_size=args.img_size,
        batch_size=args.batch_size,
        augment=False,
        shuffle=False,
        num_workers=4,
    )
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Create loss
    criterion = create_loss(mode=args.mode)

    # Run validation
    print("\nRunning validation...")
    val_loss = validate(model, val_loader, criterion, device)

    print(f"\n{'='*50}")
    print(f"Validation Results ({args.mode})")
    print(f"{'='*50}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
