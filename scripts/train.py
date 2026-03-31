#!/usr/bin/env python3
"""
Unified Training Script for Comic Text Detection

Usage:
    # Train segmentation (mask) model - Phase 1
    python scripts/train.py --mode segmentation --img-dirs data/images --mask-dirs data/masks

    # Train detection (DB) model - Phase 2
    python scripts/train.py --mode detection --img-dirs data/images --ann-dirs data/annotations

    # Train with gradient checkpointing (memory-efficient)
    python scripts/train.py --mode segmentation --gradient-checkpointing --img-dirs data/images --mask-dirs data/masks

    # 3-Phase training workflow:
    # Phase 1: Segmentation
    python scripts/train.py --training-phase 1 --mode segmentation --img-dirs data/images --mask-dirs data/masks

    # Phase 2: Detection (with frozen segmentation)
    python scripts/train.py --training-phase 2 --mode detection --unet-weights runs/seg/best.pt --freeze-seg

    # Phase 3: Block detection (with frozen seg + det)
    python scripts/train.py --training-phase 3 --mode block --checkpoint runs/det/best.pt --freeze-seg --freeze-det

    # Resume training
    python scripts/train.py --mode segmentation --resume runs/train/last.pt

    # Custom configuration
    python scripts/train.py --mode segmentation --epochs 150 --batch-size 8 --backbone yolo11m.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to path so `import src...` works
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import create_text_detector
from src.data import create_dataloader
from src.training import Trainer, TrainingConfig, create_loss, apply_gradient_checkpointing


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Comic Text Detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
3-Phase Training Workflow:
  Phase 1: Train segmentation head (mask output)
  Phase 2: Train detection head (DB text lines) with frozen segmentation
  Phase 3: Train block detector with frozen seg + det heads

Examples:
  # Phase 1: Segmentation with gradient checkpointing
  python train.py --training-phase 1 --mode segmentation --gradient-checkpointing \\
      --img-dirs data/images --mask-dirs data/masks

  # Phase 2: Detection (load seg weights, freeze seg)
  python train.py --training-phase 2 --mode detection --freeze-seg \\
      --unet-weights runs/phase1/best.pt --img-dirs data/images --ann-dirs data/annotations

  # Phase 3: Block detection (freeze everything except block detector)
  python train.py --training-phase 3 --mode block --freeze-seg --freeze-det \\
      --checkpoint runs/phase2/best.pt --img-dirs data/images --block-ann-dirs data/blocks
        """
    )

    # Mode
    parser.add_argument('--mode', type=str, default='segmentation',
                        choices=['segmentation', 'detection', 'block'],
                        help='Training mode: segmentation (mask), detection (DB), or block (block detection)')

    # Data
    parser.add_argument('--img-dirs', type=str, nargs='+', required=True,
                        help='Directories containing training images')
    parser.add_argument('--mask-dirs', type=str, nargs='+', default=None,
                        help='Directories containing mask images (segmentation mode)')
    parser.add_argument('--ann-dirs', type=str, nargs='+', default=None,
                        help='Directories containing annotations (detection mode)')
    parser.add_argument('--block-ann-dirs', type=str, nargs='+', default=None,
                        help='Directories containing block annotations (block detection mode)')
    parser.add_argument('--val-img-dirs', type=str, nargs='+', default=None,
                        help='Validation image directories')
    parser.add_argument('--val-mask-dirs', type=str, nargs='+', default=None,
                        help='Validation mask directories')
    parser.add_argument('--val-ann-dirs', type=str, nargs='+', default=None,
                        help='Validation annotation directories')

    # Model
    parser.add_argument('--backbone', type=str, default='yolo26s.pt',
                        help='Backbone model (yolo26s.pt, yolo26m.pt, yolov10s.pt, etc.)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='Do not use pretrained backbone')
    parser.add_argument('--freeze-backbone', action='store_true', default=True,
                        help='Freeze backbone weights (default: True)')
    parser.add_argument('--unfreeze-backbone', dest='freeze_backbone', action='store_false',
                        help='Train backbone weights')
    parser.add_argument('--unet-weights', type=str, default=None,
                        help='Path to UNet/segmentation weights for detection mode initialization')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to full model checkpoint (for block detection mode)')

    # Unified Training Mode (default - joint training of all heads)
    parser.add_argument('--unified', action='store_true', default=True,
                        help='Enable unified training mode: train all heads jointly (default)')
    parser.add_argument('--no-unified', dest='unified', action='store_false',
                        help='Disable unified training mode (use legacy 3-phase training)')
    parser.add_argument('--unified-phase1-epochs', type=int, default=50,
                        help='Epochs for unified phase 1 (frozen backbone, default: 50)')
    parser.add_argument('--unified-phase2-epochs', type=int, default=100,
                        help='Epochs for unified phase 2 (unfrozen backbone, default: 100)')
    parser.add_argument('--backbone-lr', type=float, default=1e-5,
                        help='Backbone learning rate for unified phase 2 (default: 1e-5)')
    parser.add_argument('--head-lr', type=float, default=1e-4,
                        help='Head learning rate for unified phase 2 (default: 1e-4)')
    parser.add_argument('--unified-seg-weight', type=float, default=1.0,
                        help='Segmentation loss weight in unified mode (default: 1.0)')
    parser.add_argument('--unified-det-weight', type=float, default=1.0,
                        help='Detection loss weight in unified mode (default: 1.0)')
    parser.add_argument('--unified-block-weight', type=float, default=1.0,
                        help='Block detection loss weight in unified mode (default: 1.0)')

    # 3-Phase Training (legacy - use --unified instead)
    parser.add_argument('--training-phase', type=int, default=1, choices=[1, 2, 3],
                        help='Training phase: 1=segmentation, 2=detection, 3=block detection')
    parser.add_argument('--phase-epochs', type=int, nargs=3, default=None,
                        metavar=('SEG', 'DET', 'BLOCK'),
                        help='Epochs per phase for auto mode [default: 100 100 50]')
    parser.add_argument('--freeze-seg', action='store_true', default=False,
                        help='Freeze segmentation head (use in Phase 2+)')
    parser.add_argument('--freeze-det', action='store_true', default=False,
                        help='Freeze detection (DB) head (use in Phase 3)')
    parser.add_argument('--freeze-projections', action='store_true', default=False,
                        help='Freeze projection layers (prevents divergence during multi-phase training)')
    parser.add_argument('--freeze-heads', action='store_true', default=False,
                        help='Freeze all heads except the one being trained (use in Phase 3)')
    parser.add_argument('--dual-assignment', action='store_true', default=False,
                        help='Use dual assignment training for block detection (YOLOv10 style)')
    parser.add_argument('--nms-free', action='store_true', default=False,
                        help='Train for NMS-free inference (one-to-one assignment)')

    # Advanced feature fusion
    parser.add_argument('--use-bifpn', action='store_true', default=False,
                        help='Enable EfficientBiFPN for bidirectional feature fusion (recommended for block detection)')
    parser.add_argument('--use-rank-guided', action='store_true', default=False,
                        help='Enable RankGuidedDecoder for compute-efficient decoding')

    # Gradient Checkpointing (memory-efficient training)
    parser.add_argument('--gradient-checkpointing', action='store_true', default=False,
                        help='Enable gradient checkpointing for memory-efficient training (~50%% memory reduction)')
    parser.add_argument('--no-checkpoint-backbone', dest='checkpoint_backbone', action='store_false', default=True,
                        help='Disable checkpointing for backbone (only checkpoint decoder)')
    parser.add_argument('--no-checkpoint-decoder', dest='checkpoint_decoder', action='store_false', default=True,
                        help='Disable checkpointing for decoder (only checkpoint backbone)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size per GPU')
    parser.add_argument('--accumulation-steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--img-size', type=int, default=1024,
                        help='Training image size')

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--block-lr', type=float, default=3e-4,
                        help='Learning rate for Phase 3 block detection (lower for fine-tuning)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')

    # Features
    parser.add_argument('--no-compile', dest='compile', action='store_false', default=True,
                        help='Disable torch.compile()')
    parser.add_argument('--no-amp', dest='amp', action='store_false', default=True,
                        help='Disable automatic mixed precision')
    parser.add_argument('--no-ema', dest='ema', action='store_false', default=True,
                        help='Disable EMA')
    parser.add_argument('--no-early-stopping', dest='early_stopping', action='store_false', default=True,
                        help='Disable early stopping')

    # Misc
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--cache', action='store_true', default=False,
                        help='Cache data in memory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    parser.add_argument('--save-dir', '--output', type=str, default='runs/train',
                        dest='save_dir',
                        help='Save directory (--output is an alias)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # WandB
    parser.add_argument('--no-wandb', dest='wandb', action='store_false', default=True,
                        help='Disable WandB logging')
    parser.add_argument('--project', type=str, default='comic-text-detector',
                        help='WandB project name')
    parser.add_argument('--run-name', type=str, default=None,
                        help='WandB run name')

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate arguments based on mode (skip for unified mode)
    if not args.unified:
        if args.mode == 'segmentation' and args.mask_dirs is None:
            raise ValueError("--mask-dirs required for segmentation mode")
        if args.mode == 'detection' and args.ann_dirs is None:
            raise ValueError("--ann-dirs required for detection mode")
        if args.mode == 'block' and args.block_ann_dirs is None:
            raise ValueError("--block-ann-dirs required for block detection mode")
    else:
        # Unified mode: need either ann_dirs or mask_dirs
        if args.ann_dirs is None and args.mask_dirs is None:
            raise ValueError("--ann-dirs or --mask-dirs required for unified mode")

    # Infer training phase from mode if not explicitly set
    if args.training_phase == 1 and args.mode == 'detection':
        args.training_phase = 2
    elif args.training_phase == 1 and args.mode == 'block':
        args.training_phase = 3

    # Ensure mode matches training phase (auto-correct if mismatched)
    if args.training_phase == 3 and args.mode != 'block':
        print(f"Warning: --training-phase 3 requires --mode block, auto-correcting from '{args.mode}'")
        args.mode = 'block'
    elif args.training_phase == 2 and args.mode == 'segmentation':
        print(f"Warning: --training-phase 2 requires --mode detection, auto-correcting")
        args.mode = 'detection'
    elif args.training_phase == 1 and args.mode != 'segmentation':
        print(f"Warning: --training-phase 1 requires --mode segmentation, auto-correcting from '{args.mode}'")
        args.mode = 'segmentation'

    # Create configuration with all new options
    config = TrainingConfig(
        backbone_name=args.backbone,
        pretrained_backbone=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        use_compile=args.compile,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        # Unified training options
        unified_mode=args.unified,
        unified_phase1_epochs=args.unified_phase1_epochs,
        unified_phase2_epochs=args.unified_phase2_epochs,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        unified_seg_weight=args.unified_seg_weight,
        unified_det_weight=args.unified_det_weight,
        unified_block_weight=args.unified_block_weight,
        # 3-Phase training options (legacy)
        training_phase=args.training_phase,
        phase_epochs=args.phase_epochs,
        freeze_seg=args.freeze_seg,
        freeze_det=args.freeze_det,
        freeze_projections=args.freeze_projections,
        freeze_heads=args.freeze_heads,
        # Block detection options (Phase 3)
        dual_assignment=args.dual_assignment,
        nms_free=args.nms_free,
        # Gradient checkpointing options
        gradient_checkpointing=args.gradient_checkpointing,
        checkpoint_backbone=args.checkpoint_backbone,
        checkpoint_decoder=args.checkpoint_decoder,
        # Optimizer
        optimizer=args.optimizer,
        lr=args.lr,
        block_lr=args.block_lr,
        weight_decay=args.weight_decay,
        use_amp=args.amp,
        use_ema=args.ema,
        early_stopping=args.early_stopping,
        img_size=args.img_size,
        num_workers=args.workers,
        cache_data=args.cache,
        save_dir=args.save_dir,
        resume=args.resume,
        use_wandb=args.wandb,
        project_name=args.project,
        run_name=args.run_name,
    )

    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"\n{'='*60}")
    print(f"Comic Text Detector Training")
    print(f"{'='*60}")
    if args.unified:
        print(f"Mode: UNIFIED (all heads jointly)")
        print(f"  Phase 1 epochs: {args.unified_phase1_epochs} (backbone frozen)")
        print(f"  Phase 2 epochs: {args.unified_phase2_epochs} (backbone unfrozen)")
        print(f"  Backbone LR: {args.backbone_lr}, Head LR: {args.head_lr}")
        print(f"  Loss weights: seg={args.unified_seg_weight}, det={args.unified_det_weight}, block={args.unified_block_weight}")
    else:
        print(f"Mode: {args.mode}")
        print(f"Training Phase: {args.training_phase}")
    print(f"Backbone: {args.backbone}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {args.accumulation_steps} = {args.batch_size * args.accumulation_steps}")
    print(f"Gradient Checkpointing: {args.gradient_checkpointing}")
    if args.freeze_seg:
        print(f"Freeze Segmentation: Yes")
    if args.freeze_det:
        print(f"Freeze Detection: Yes")
    if args.freeze_projections:
        print(f"Freeze Projections: Yes")
    if args.freeze_heads:
        print(f"Freeze Heads: Yes")
    if args.dual_assignment:
        print(f"Dual Assignment: Yes (YOLOv10 style)")
    if args.nms_free:
        print(f"NMS-Free: Yes (one-to-one assignment)")
    print(f"{'='*60}\n")

    # Create model
    # BiFPN disabled by default (has channel mismatch bug that needs fixing)
    use_bifpn = args.use_bifpn
    print("Creating model...")
    if use_bifpn:
        print("BiFPN: Enabled (bidirectional feature fusion)")
    if args.use_rank_guided:
        print("RankGuidedDecoder: Enabled")
    model = create_text_detector(
        backbone_name=args.backbone,
        pretrained_backbone=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        use_bifpn=use_bifpn,
        use_rank_guided=args.use_rank_guided,
        device=device
    )

    # Set up model based on mode and phase
    if args.unified:
        # Unified mode: train all heads jointly
        # Initialize all heads
        model.initialize_db(args.unet_weights)
        model.initialize_block_detector()
        # Set unified training mode (starts with backbone frozen)
        model.train_unified(freeze_backbone=True)
        # For unified mode, use annotation directories (masks + line annotations)
        # The dataset will find block_annotations as a sibling directory
        ann_dirs = args.ann_dirs if args.ann_dirs else args.mask_dirs
        val_ann_dirs = args.val_ann_dirs if args.val_ann_dirs else args.val_mask_dirs
        print("Unified mode: Initialized all heads (seg_net, dbnet, block_det)")

    elif args.mode == 'segmentation':
        model.train_mask()
        ann_dirs = args.mask_dirs
        val_ann_dirs = args.val_mask_dirs

    elif args.mode == 'detection':
        # Initialize DB head (optionally load UNet weights)
        model.initialize_db(args.unet_weights)
        model.train_db()
        ann_dirs = args.ann_dirs
        val_ann_dirs = args.val_ann_dirs

    elif args.mode == 'block':
        # Initialize block detector
        # Load full checkpoint if provided
        if args.checkpoint:
            print(f"Loading checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'weights' in checkpoint:
                # Handle legacy checkpoint format
                if 'backbone' in checkpoint['weights']:
                    model.backbone.load_state_dict(checkpoint['weights']['backbone'])
                if 'seg_net' in checkpoint['weights']:
                    model.seg_net.load_state_dict(checkpoint['weights']['seg_net'])
                if 'dbnet' in checkpoint['weights'] and model.dbnet is not None:
                    model.dbnet.load_state_dict(checkpoint['weights']['dbnet'])

        # Initialize block detection head
        model.initialize_block_detector()
        model.train_block()
        ann_dirs = args.block_ann_dirs
        val_ann_dirs = args.val_ann_dirs

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Apply gradient checkpointing if enabled
    model = apply_gradient_checkpointing(model, config)

    # Create data loaders
    print("Creating data loaders...")
    dataloader_mode = 'unified' if args.unified else args.mode
    train_loader = create_dataloader(
        img_dirs=args.img_dirs,
        annotation_dirs=ann_dirs,
        mode=dataloader_mode,
        img_size=args.img_size,
        batch_size=args.batch_size,
        augment=True,
        cache=args.cache,
        num_workers=args.workers,
        shuffle=True,
    )

    val_loader = None
    if args.val_img_dirs and val_ann_dirs:
        val_loader = create_dataloader(
            img_dirs=args.val_img_dirs,
            annotation_dirs=val_ann_dirs,
            mode=dataloader_mode,
            img_size=args.img_size,
            batch_size=args.batch_size,
            augment=False,
            cache=False,
            num_workers=args.workers,
            shuffle=False,
        )

    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")

    # Create loss function
    if args.unified:
        # Unified mode uses combined loss for all heads
        criterion = create_loss(
            mode='unified',
            seg_weight=args.unified_seg_weight,
            det_weight=args.unified_det_weight,
            block_weight=args.unified_block_weight,
        )
    else:
        # Block mode uses dual assignment loss (model returns o2m/o2o tuple during training)
        loss_mode = 'dual_assignment' if args.mode == 'block' else args.mode
        criterion = create_loss(mode=loss_mode)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config,
        device=device,
    )

    # Start training
    trainer.train()

    print("\nDone!")


if __name__ == '__main__':
    main()
