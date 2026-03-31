"""
Modern Training Pipeline for Comic Text Detection

Features:
- torch.compile() for 2x speedup
- Automatic Mixed Precision (AMP)
- Gradient checkpointing for memory-efficient training
- 3-phase training support (segmentation, detection, block detection)
- Gradient clipping
- Exponential Moving Average (EMA)
- Early stopping
- Cosine annealing with warm restarts
- WandB integration
"""

from __future__ import annotations

import os
import copy
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults."""

    # Model
    backbone_name: str = 'yolo11s.pt'
    pretrained_backbone: bool = True
    freeze_backbone: bool = True
    use_compile: bool = True  # torch.compile()

    # Training
    epochs: int = 100
    batch_size: int = 4
    accumulation_steps: int = 4  # Effective batch = batch_size * accumulation_steps

    # Unified Training Mode (default - replaces 3-phase sequential)
    unified_mode: bool = True  # Train all heads jointly (recommended)
    unified_phase1_epochs: int = 50  # Phase 1: all heads, backbone frozen
    unified_phase2_epochs: int = 100  # Phase 2: all heads + backbone unfrozen
    backbone_lr: float = 1e-5  # Lower LR for backbone in phase 2
    head_lr: float = 1e-4  # Higher LR for heads in phase 2
    unified_seg_weight: float = 1.0  # Segmentation loss weight
    unified_det_weight: float = 1.0  # DBNet loss weight
    unified_block_weight: float = 1.0  # Block detection loss weight

    # 3-Phase Training Support (legacy - use unified_mode=True instead)
    training_phase: int = 1  # 1: segmentation, 2: detection, 3: block detection
    phase_epochs: Optional[List[int]] = None  # Epochs per phase for auto mode [100, 100, 50]
    freeze_seg: bool = False  # Freeze segmentation head
    freeze_det: bool = False  # Freeze detection (DB) head
    freeze_projections: bool = False  # Freeze projection layers (prevents divergence)
    freeze_heads: bool = False  # Freeze all heads except the one being trained

    # Block detection options (Phase 3)
    dual_assignment: bool = False  # Use dual assignment training (YOLOv10 style)
    nms_free: bool = False  # Train for NMS-free inference (one-to-one assignment)

    # Phase-specific loss weights
    seg_loss_weight: float = 1.0
    det_loss_weight: float = 1.0
    block_loss_weight: float = 1.0

    # Gradient Checkpointing (memory-efficient training)
    gradient_checkpointing: bool = False
    checkpoint_backbone: bool = True  # Apply checkpointing to backbone
    checkpoint_decoder: bool = True   # Apply checkpointing to decoder heads

    # Optimizer
    optimizer: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    lr: float = 1e-3
    block_lr: float = 1e-3  # Phase 3 block detection LR (higher for fresh head training)
    weight_decay: float = 0.01
    momentum: float = 0.937
    betas: Tuple[float, float] = (0.937, 0.999)

    # Scheduler
    scheduler: str = 'cosine_warmup'  # 'cosine', 'cosine_warmup', 'onecycle'
    warmup_epochs: int = 3
    min_lr: float = 1e-4  # Higher floor keeps last 20 epochs productive

    # Regularization
    gradient_clip: float = 10.0  # Generous for fresh heads; VFL + CIoU produce large grads early
    label_smoothing: float = 0.0

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999  # Conservative EMA preserves confidence gains

    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4

    # Mixed precision
    use_amp: bool = True

    # Data
    img_size: int = 1024
    num_workers: int = 4
    cache_data: bool = False

    # Logging
    log_interval: int = 10
    eval_interval: int = 1
    save_interval: int = 5

    # Paths
    save_dir: str = 'runs/train'
    resume: Optional[str] = None

    # WandB
    use_wandb: bool = True
    project_name: str = 'comic-text-detector'
    run_name: Optional[str] = None

    # TensorBoard
    use_tensorboard: bool = True


class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Apply shadow weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


class GradientCheckpointedModel(nn.Module):
    """Wrapper for memory-efficient training with gradient checkpointing.

    Applies torch.utils.checkpoint to backbone and decoder stages to reduce
    memory usage by ~50-60%, trading off some compute for memory.

    Usage:
        model = TextDetector(backbone, ...)
        wrapped_model = GradientCheckpointedModel(model, checkpoint_backbone=True)
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_backbone: bool = True,
        checkpoint_decoder: bool = True
    ):
        super().__init__()
        self.model = model
        self.checkpoint_backbone = checkpoint_backbone
        self.checkpoint_decoder = checkpoint_decoder

    def _backbone_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Checkpointed backbone forward pass."""
        return self.model.backbone(x)

    def _seg_forward(self, *features) -> torch.Tensor:
        """Checkpointed segmentation head forward pass."""
        return self.model.seg_net(*features, forward_mode=self.model.forward_mode)

    def _db_forward(self, *seg_features) -> torch.Tensor:
        """Checkpointed DB head forward pass."""
        return self.model.dbnet(*seg_features)

    def _block_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Checkpointed block detection forward pass."""
        return self.model.block_det(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gradient checkpointing.

        Checkpointing is only applied during training. During eval,
        the standard forward pass is used.
        """
        # Extract features with optional checkpointing
        if self.checkpoint_backbone and self.training:
            # Use checkpoint for backbone
            features = checkpoint(
                self._backbone_forward,
                x,
                use_reentrant=False
            )
        else:
            # Standard forward or frozen backbone
            if hasattr(self.model, 'freeze_backbone') and self.model.freeze_backbone:
                with torch.no_grad():
                    features = self.model.backbone(x)
            else:
                features = self.model.backbone(x)

        # Get forward mode from wrapped model
        forward_mode = getattr(self.model, 'forward_mode', 0)

        # Apply appropriate head with optional checkpointing
        if forward_mode == 0:  # TEXTDET_MASK - segmentation
            if self.checkpoint_decoder and self.training:
                return checkpoint(
                    self._seg_forward,
                    *features,
                    use_reentrant=False
                )
            else:
                return self.model.seg_net(*features, forward_mode=forward_mode)

        elif forward_mode == 1:  # TEXTDET_DET - detection
            with torch.no_grad():
                seg_features = self.model.seg_net(*features, forward_mode=forward_mode)

            if self.checkpoint_decoder and self.training and self.model.dbnet is not None:
                return checkpoint(
                    self._db_forward,
                    *seg_features,
                    use_reentrant=False
                )
            else:
                return self.model.dbnet(*seg_features)

        elif forward_mode == 3:  # TEXTDET_BLOCK - block detection
            # Block detection uses P3, P4, P5 features
            block_features = [features[1], features[2], features[3]]  # f128, f64, f32

            if self.checkpoint_decoder and self.training and self.model.block_det is not None:
                return checkpoint(
                    self._block_forward,
                    block_features,
                    use_reentrant=False
                )
            else:
                return self.model.block_det(block_features)

        else:
            raise ValueError(f"Unknown forward mode: {forward_mode}")

    def train_mask(self):
        """Set mode for training segmentation head."""
        self.model.train_mask()

    def train_db(self):
        """Set mode for training DB head."""
        self.model.train_db()

    def train_block(self):
        """Set mode for training block detection head."""
        self.model.train_block()

    def initialize_db(self, unet_weights: Optional[str] = None):
        """Initialize DB head."""
        self.model.initialize_db(unet_weights)

    def initialize_block_detector(self, *args, **kwargs):
        """Initialize block detector."""
        self.model.initialize_block_detector(*args, **kwargs)

    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def apply_gradient_checkpointing(model: nn.Module, config: TrainingConfig) -> nn.Module:
    """Apply gradient checkpointing wrapper if enabled in config.

    Args:
        model: The model to wrap
        config: Training configuration

    Returns:
        Wrapped model if checkpointing enabled, otherwise original model
    """
    if config.gradient_checkpointing:
        print("Enabling gradient checkpointing for memory-efficient training...")
        print(f"  Checkpoint backbone: {config.checkpoint_backbone}")
        print(f"  Checkpoint decoder: {config.checkpoint_decoder}")
        return GradientCheckpointedModel(
            model,
            checkpoint_backbone=config.checkpoint_backbone,
            checkpoint_decoder=config.checkpoint_decoder
        )
    return model


class Trainer:
    """Modern training pipeline for text detection."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        criterion: nn.Module,
        config: TrainingConfig,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.config = config
        self.device = device

        # Setup save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Compile model if enabled (PyTorch 2.0+)
        if config.use_compile and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)

        # Mixed precision
        self.scaler = GradScaler('cuda') if config.use_amp else None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode='min'
        ) if config.early_stopping else None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.best_f1 = 0.0

        # 3-Phase training state
        self.current_phase = config.training_phase
        self.phase_epochs = config.phase_epochs or [100, 100, 50]  # Default epochs per phase

        # Initialize phase-specific settings BEFORE creating optimizer
        # This ensures the model is in the correct mode for differential LR
        self._setup_training_phase(self.current_phase)

        # Setup optimizer AFTER phase setup (for correct requires_grad state)
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # EMA (after optimizer setup)
        self.ema = EMA(self.model, config.ema_decay) if config.use_ema else None

        # History for plotting
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'epoch': [],
            'phase': []
        }

        # TensorBoard
        self.tb_writer = None
        if config.use_tensorboard:
            tb_log_dir = self.save_dir / 'tensorboard'
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
            print(f"TensorBoard logging to: {tb_log_dir}")

        # WandB
        self.wandb_run = None
        if config.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()

        # Resume from checkpoint
        if config.resume:
            self._load_checkpoint(config.resume)

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config.

        For unified mode phase 2, uses differential learning rates:
        - Backbone: config.backbone_lr (default 1e-5)
        - Heads: config.head_lr (default 1e-4)
        """
        # Check if unified mode with phase 2 (backbone unfrozen)
        if self.config.unified_mode and not self._is_backbone_frozen():
            return self._create_differential_optimizer()

        params = [p for p in self.model.parameters() if p.requires_grad]

        # Use block_lr for Phase 3 block detection (lower LR for fine-tuning)
        lr = self.config.block_lr if self.config.training_phase == 3 else self.config.lr

        if self.config.optimizer == 'adam':
            return optim.Adam(
                params,
                lr=lr,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(
                params,
                lr=lr,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return optim.SGD(
                params,
                lr=lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_differential_optimizer(self) -> optim.Optimizer:
        """Create optimizer with differential learning rates for unified phase 2."""
        model = self.model

        # Handle wrapped models
        if hasattr(model, 'model'):
            model = model.model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod

        # Collect backbone and head parameters
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': self.config.backbone_lr},
            {'params': head_params, 'lr': self.config.head_lr},
        ]

        print(f"  Differential LR: backbone={self.config.backbone_lr}, heads={self.config.head_lr}")
        print(f"  Backbone params: {len(backbone_params)}, Head params: {len(head_params)}")

        if self.config.optimizer == 'adam':
            return optim.Adam(
                param_groups,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(
                param_groups,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return optim.SGD(
                param_groups,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _is_backbone_frozen(self) -> bool:
        """Check if the backbone is currently frozen."""
        model = self.model

        # Handle wrapped models
        if hasattr(model, 'model'):
            model = model.model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod

        if hasattr(model, 'backbone'):
            # Check if any backbone parameter has requires_grad=True
            for param in model.backbone.parameters():
                if param.requires_grad:
                    return False
            return True
        return True  # Default to frozen if no backbone found

    def _check_phase_transition(self):
        """Check if it's time to transition from unified phase 1 to phase 2."""
        if not self.config.unified_mode:
            return

        # Phase 1 ends after unified_phase1_epochs
        if self.epoch >= self.config.unified_phase1_epochs and self._is_backbone_frozen():
            print(f"\n{'='*60}")
            print(f"UNIFIED MODE: Transitioning to Phase 2 at epoch {self.epoch}")
            print(f"  - Unfreezing backbone with differential LR")
            print(f"{'='*60}")

            model = self.model
            if hasattr(model, 'model'):
                model = model.model
            if hasattr(model, '_orig_mod'):
                model = model._orig_mod

            # Unfreeze backbone
            if hasattr(model, 'train_unified'):
                model.train_unified(freeze_backbone=False)

            # Recreate optimizer with differential LR
            self.optimizer = self._create_optimizer()

            # Recreate scheduler for remaining epochs
            remaining_epochs = self.config.unified_phase2_epochs
            total_steps = len(self.train_loader) * remaining_epochs
            warmup_steps = len(self.train_loader) * min(3, remaining_epochs // 10)

            if self.config.scheduler == 'cosine_warmup':
                self.scheduler = self._cosine_warmup_scheduler(total_steps, warmup_steps)

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = len(self.train_loader) * self.config.warmup_epochs

        if self.config.scheduler == 'cosine':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.epochs // 3,
                T_mult=2,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler == 'cosine_warmup':
            return self._cosine_warmup_scheduler(total_steps, warmup_steps)
        elif self.config.scheduler == 'onecycle':
            effective_lr = self.config.block_lr if self.config.training_phase == 3 else self.config.lr
            return OneCycleLR(
                self.optimizer,
                max_lr=effective_lr,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
        else:
            return None

    def _cosine_warmup_scheduler(self, total_steps: int, warmup_steps: int):
        """Create cosine scheduler with linear warmup."""
        # Use the actual base LR passed to the optimizer (block_lr for phase 3)
        base_lr = self.config.block_lr if self.config.training_phase == 3 else self.config.lr
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return self.config.min_lr / base_lr + (
                    1 - self.config.min_lr / base_lr
                ) * 0.5 * (1 + math.cos(math.pi * progress))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _init_wandb(self):
        """Initialize WandB logging."""
        run_name = self.config.run_name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.wandb_run = wandb.init(
            project=self.config.project_name,
            name=run_name,
            config=self.config.__dict__,
            resume='allow'
        )

    def _setup_training_phase(self, phase: int):
        """Setup model for a specific training phase with proper layer freezing.

        For unified_mode=True (default):
            Phase 1: ALL heads jointly, backbone FROZEN
            Phase 2: ALL heads + backbone UNFROZEN (differential LR)

        For unified_mode=False (legacy 3-phase):
            Phase 1 (Segmentation):
                - Backbone: FROZEN (always)
                - Projections: TRAINABLE (learn feature adaptation)
                - UnetHead (seg_net): TRAINABLE
                - DBHead: not initialized / frozen if exists
                - BlockHead: not initialized / frozen if exists

            Phase 2 (Detection):
                - Backbone: FROZEN
                - Projections: FROZEN (prevents divergence!)
                - UnetHead: FROZEN (from Phase 1)
                - DBHead: TRAINABLE
                - BlockHead: not initialized / frozen if exists

            Phase 3 (Block Detection):
                - Backbone: FROZEN
                - Projections: FROZEN
                - UnetHead: FROZEN
                - DBHead: FROZEN
                - BlockHead: TRAINABLE (dual assignment)

        Args:
            phase: Training phase (1, 2, or 3 for legacy; 1 or 2 for unified)
        """
        self.current_phase = phase
        model = self.model

        # Handle wrapped models (GradientCheckpointedModel, torch.compile)
        if hasattr(model, 'model'):
            model = model.model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod

        # Handle unified mode
        if self.config.unified_mode:
            print(f"\n{'='*60}")
            print(f"UNIFIED MODE: Setting up Phase {phase}")
            print(f"{'='*60}")

            # Initialize all heads if not already done
            if hasattr(model, 'dbnet') and model.dbnet is None:
                print("  - Initializing DBNet head...")
                if hasattr(model, 'initialize_db'):
                    model.initialize_db()

            if hasattr(model, 'block_det') and model.block_det is None:
                print("  - Initializing Block detection head...")
                if hasattr(model, 'initialize_block_detector'):
                    model.initialize_block_detector()

            # Set unified forward mode
            freeze_backbone = (phase == 1)
            if hasattr(model, 'train_unified'):
                model.train_unified(freeze_backbone=freeze_backbone)

            if phase == 1:
                print("  Phase 1: ALL heads jointly, backbone FROZEN")
                print(f"    - Epochs: {self.config.unified_phase1_epochs}")
            else:
                print("  Phase 2: ALL heads + backbone UNFROZEN")
                print(f"    - Epochs: {self.config.unified_phase2_epochs}")
                print(f"    - Backbone LR: {self.config.backbone_lr}")
                print(f"    - Head LR: {self.config.head_lr}")

            return  # Exit early for unified mode

        print(f"\n{'='*60}")
        print(f"Setting up Training Phase {phase}")
        print(f"{'='*60}")

        # Helper to get backbone (handles nested model structures)
        def get_backbone(m):
            if hasattr(m, 'backbone'):
                return m.backbone
            return None

        backbone = get_backbone(model)

        if phase == 1:
            # Phase 1: Segmentation - train projections + UnetHead
            print("Phase 1: Training SEGMENTATION (projections + UnetHead)")
            print("  - Backbone: FROZEN")
            print("  - Projections: TRAINABLE")
            print("  - UnetHead (seg_net): TRAINABLE")
            print("  - DBHead: NOT INITIALIZED / FROZEN")
            print("  - BlockHead: NOT INITIALIZED / FROZEN")

            # Set forward mode to segmentation
            if hasattr(model, 'train_mask'):
                model.train_mask()

            # Ensure projections are trainable in Phase 1
            if backbone is not None and hasattr(backbone, 'unfreeze_projections'):
                backbone.unfreeze_projections()
                print("  - Projections: UNFROZEN (trainable)")

            # Freeze detection heads if they exist
            if hasattr(model, 'dbnet') and model.dbnet is not None:
                for param in model.dbnet.parameters():
                    param.requires_grad = False
                print("  - DBNet: FROZEN")

            if hasattr(model, 'block_det') and model.block_det is not None:
                for param in model.block_det.parameters():
                    param.requires_grad = False
                print("  - BlockHead: FROZEN")

        elif phase == 2:
            # Phase 2: Detection - freeze projections + UnetHead, train DBHead
            print("Phase 2: Training DETECTION (DBHead only)")
            print("  - Backbone: FROZEN")
            print("  - Projections: FROZEN (prevents divergence!)")
            print("  - UnetHead (seg_net): FROZEN (from Phase 1)")
            print("  - DBHead: TRAINABLE")
            print("  - BlockHead: NOT INITIALIZED / FROZEN")

            # Set forward mode to detection
            if hasattr(model, 'train_db'):
                model.train_db()

            # CRITICAL: Freeze projections to prevent divergence
            if backbone is not None and hasattr(backbone, '_freeze_projections'):
                backbone._freeze_projections()
                print("  - Projections: FROZEN (prevents feature divergence)")

            # Freeze segmentation head (UnetHead)
            if hasattr(model, 'seg_net'):
                for param in model.seg_net.parameters():
                    param.requires_grad = False
                print("  - UnetHead (seg_net): FROZEN")

            # Freeze block detector if exists
            if hasattr(model, 'block_det') and model.block_det is not None:
                for param in model.block_det.parameters():
                    param.requires_grad = False
                print("  - BlockHead: FROZEN")

        elif phase == 3:
            # Phase 3: Block detection - freeze everything except BlockHead
            print("Phase 3: Training BLOCK DETECTION (BlockHead only)")
            print("  - Backbone: FROZEN")
            print("  - Projections: FROZEN")
            print("  - UnetHead (seg_net): FROZEN")
            print("  - DBHead: FROZEN")
            print("  - BlockHead: TRAINABLE (dual assignment)")

            # Set forward mode to block detection
            if hasattr(model, 'train_block'):
                model.train_block()

            # Freeze projections
            if backbone is not None and hasattr(backbone, '_freeze_projections'):
                backbone._freeze_projections()
                print("  - Projections: FROZEN")

            # Freeze segmentation head (UnetHead)
            if hasattr(model, 'seg_net'):
                for param in model.seg_net.parameters():
                    param.requires_grad = False
                print("  - UnetHead (seg_net): FROZEN")

            # Freeze detection head (DBHead)
            if hasattr(model, 'dbnet') and model.dbnet is not None:
                for param in model.dbnet.parameters():
                    param.requires_grad = False
                print("  - DBHead: FROZEN")

            # Ensure BlockHead is trainable
            if hasattr(model, 'block_det') and model.block_det is not None:
                for param in model.block_det.parameters():
                    param.requires_grad = True
                print("  - BlockHead: TRAINABLE")

        else:
            raise ValueError(f"Invalid training phase: {phase}. Must be 1, 2, or 3.")

        # Apply additional freeze settings from config (overrides)
        if self.config.freeze_seg and hasattr(model, 'seg_net'):
            for param in model.seg_net.parameters():
                param.requires_grad = False
            print("  - Config override: UnetHead (seg_net) FROZEN")

        if self.config.freeze_det and hasattr(model, 'dbnet') and model.dbnet is not None:
            for param in model.dbnet.parameters():
                param.requires_grad = False
            print("  - Config override: DBHead FROZEN")

        # Count and display trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_params = total_params - trainable_params
        print(f"\nParameter Summary:")
        print(f"  Total:     {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"  Frozen:    {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        print(f"{'='*60}\n")

    def switch_phase(self, new_phase: int):
        """Switch to a new training phase.

        This resets the optimizer and scheduler for the new phase,
        preserving the model weights.

        Args:
            new_phase: New phase to switch to (1, 2, or 3)
        """
        print(f"\nSwitching from Phase {self.current_phase} to Phase {new_phase}")

        # Setup new phase
        self._setup_training_phase(new_phase)

        # Recreate optimizer with new trainable parameters
        self.optimizer = self._create_optimizer()

        # Recreate scheduler
        self.scheduler = self._create_scheduler()

        # Reset early stopping for new phase
        if self.early_stopping is not None:
            self.early_stopping.counter = 0
            self.early_stopping.best_score = None
            self.early_stopping.should_stop = False

        # Reset EMA for new phase
        if self.ema is not None:
            self.ema = EMA(self.model, self.config.ema_decay)

        # Reset best loss for new phase
        self.best_loss = float('inf')

    def get_phase_loss_weight(self) -> float:
        """Get the loss weight for the current training phase."""
        if self.current_phase == 1:
            return self.config.seg_loss_weight
        elif self.current_phase == 2:
            return self.config.det_loss_weight
        elif self.current_phase == 3:
            return self.config.block_loss_weight
        return 1.0

    def train(self):
        """Main training loop."""
        # For unified mode, calculate total epochs
        if self.config.unified_mode:
            total_epochs = self.config.unified_phase1_epochs + self.config.unified_phase2_epochs
            print(f"\nStarting UNIFIED training for {total_epochs} epochs...")
            print(f"  Phase 1: {self.config.unified_phase1_epochs} epochs (backbone frozen)")
            print(f"  Phase 2: {self.config.unified_phase2_epochs} epochs (backbone unfrozen)")
            # Override config epochs for unified mode
            self.config.epochs = total_epochs
        else:
            print(f"\nStarting training for {self.config.epochs} epochs...")
            print(f"Training Phase: {self.current_phase}")

        print(f"Effective batch size: {self.config.batch_size * self.config.accumulation_steps}")
        print(f"Device: {self.device}")
        print(f"AMP: {self.config.use_amp}, EMA: {self.config.use_ema}")
        print(f"Gradient Checkpointing: {self.config.gradient_checkpointing}")

        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch

            # Check for unified mode phase transition
            self._check_phase_transition()

            # Training epoch
            train_metrics = self._train_epoch()

            # Validation
            val_metrics = {}
            if self.val_loader is not None and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self._validate()

            # Log metrics
            self._log_metrics(train_metrics, val_metrics)

            # Save checkpoints
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint('last.pt')

            val_loss = val_metrics.get('val_loss', train_metrics['train_loss'])
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_checkpoint('best.pt')
                print(f"New best model saved! Loss: {val_loss:.4f}")

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break

        print("\nTraining complete!")
        self._save_checkpoint('final.pt')
        
        # Save training plots
        self._save_training_plots()
        
        # Close TensorBoard writer
        if self.tb_writer is not None:
            self.tb_writer.close()

        if self.wandb_run is not None:
            wandb.finish()

    def _save_training_plots(self):
        """Save training history plots as PNG files."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            epochs = self.history['epoch']
            
            # Loss plot
            ax1 = axes[0]
            ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            if self.history['val_loss']:
                ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('Training & Validation Loss', fontsize=14)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(left=1)
            
            # Learning rate plot
            ax2 = axes[1]
            ax2.plot(epochs, self.history['lr'], 'g-', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Learning Rate', fontsize=12)
            ax2.set_title('Learning Rate Schedule', fontsize=14)
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(left=1)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.save_dir / 'training_curves.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Training plots saved to: {plot_path}")
            
            # Also save history as JSON for later analysis
            import json
            history_path = self.save_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            print(f"Training history saved to: {history_path}")
            
        except ImportError:
            print("matplotlib not available, skipping plot generation")

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        # Restore phase-specific train/eval mode instead of blanket model.train()
        # which overrides frozen module eval() and causes BatchNorm stats drift
        model = self.model
        if hasattr(model, 'model'):
            model = model.model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod

        if self.config.training_phase == 3 and hasattr(model, 'train_block'):
            model.train_block()
        elif self.config.training_phase == 2 and hasattr(model, 'train_db'):
            model.train_db()
        elif self.config.training_phase == 1 and hasattr(model, 'train_mask'):
            model.train_mask()
        else:
            self.model.train()

        total_loss = 0.0
        num_batches = len(self.train_loader)
        nan_count = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch.items() if k != 'image'}

            # Forward pass with AMP
            with autocast('cuda', enabled=self.config.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss = loss / self.config.accumulation_steps

            # Check for NaN loss and skip batch if detected
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count <= 5:  # Only warn first 5 times
                    print(f"\nWarning: NaN/Inf loss detected at batch {batch_idx}, skipping...")
                self.optimizer.zero_grad()
                continue

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Update EMA
                if self.ema is not None:
                    self.ema.update()

                # Update scheduler (per step)
                if self.scheduler is not None and self.config.scheduler in ['cosine_warmup', 'onecycle']:
                    self.scheduler.step()

                self.global_step += 1

            total_loss += loss.item() * self.config.accumulation_steps

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item() * self.config.accumulation_steps:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

        # Update scheduler (per epoch)
        if self.scheduler is not None and self.config.scheduler == 'cosine':
            self.scheduler.step()

        # Report NaN count if any
        if nan_count > 0:
            print(f"\nWarning: {nan_count} batches had NaN/Inf loss and were skipped this epoch")

        valid_batches = num_batches - nan_count
        avg_loss = total_loss / max(valid_batches, 1)
        return {'train_loss': avg_loss}

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Validate the model."""
        # Use EMA weights for validation
        if self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()

        total_loss = 0.0
        num_batches = len(self.val_loader)
        valid_batches = 0
        sample_logged = False

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
            images = batch['image'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch.items() if k != 'image'}

            with autocast('cuda', enabled=self.config.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # Skip NaN/Inf losses
            loss_val = loss.item()
            if not (math.isnan(loss_val) or math.isinf(loss_val)):
                total_loss += loss_val
                valid_batches += 1
            
            # Log sample predictions to TensorBoard (first batch only)
            if not sample_logged and self.tb_writer is not None and batch_idx == 0:
                self._log_sample_predictions(images, outputs, targets)
                sample_logged = True

        # Restore original weights
        if self.ema is not None:
            self.ema.restore()

        avg_loss = total_loss / max(valid_batches, 1)
        return {'val_loss': avg_loss}

    def _log_sample_predictions(self, images: torch.Tensor, outputs, targets: Dict):
        """Log sample predictions to TensorBoard for visualization."""
        try:
            # Get mask predictions (handle both tuple and tensor outputs)
            if isinstance(outputs, tuple):
                pred_mask = outputs[0]  # First element is usually the mask
            else:
                pred_mask = outputs
            
            # Take first few samples (max 4)
            n_samples = min(4, images.size(0))

            # Images are already in [0, 1] range from ToTensorV2 (no ImageNet normalization in training).
            # ImageNet normalization is only applied in the ONNX export wrapper to avoid double-normalization.
            images_vis = torch.clamp(images[:n_samples], 0, 1)
            
            # Get ground truth mask
            gt_mask = targets.get('mask', None)
            
            if gt_mask is not None:
                # Create comparison grid: [image, gt_mask, pred_mask]
                pred_mask_rgb = pred_mask[:n_samples].repeat(1, 3, 1, 1)  # Convert to 3 channels
                gt_mask_rgb = gt_mask[:n_samples].repeat(1, 3, 1, 1)
                
                # Stack horizontally: original | ground truth | prediction
                comparison = torch.cat([images_vis, gt_mask_rgb, pred_mask_rgb], dim=3)
                
                # Add to TensorBoard
                from torchvision.utils import make_grid
                grid = make_grid(comparison, nrow=1, normalize=False)
                self.tb_writer.add_image('Validation/Predictions', grid, self.epoch + 1)
            else:
                # Just log predictions
                pred_mask_rgb = pred_mask[:n_samples].repeat(1, 3, 1, 1)
                comparison = torch.cat([images_vis, pred_mask_rgb], dim=3)
                
                from torchvision.utils import make_grid
                grid = make_grid(comparison, nrow=1, normalize=False)
                self.tb_writer.add_image('Validation/Predictions', grid, self.epoch + 1)
                
        except Exception as e:
            # Don't fail training if visualization fails
            print(f"Warning: Failed to log sample predictions: {e}")

    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to console, TensorBoard, and WandB."""
        lr = self.optimizer.param_groups[0]['lr']

        # Console
        log_str = f"Epoch {self.epoch + 1}/{self.config.epochs}"
        log_str += f" [Phase {self.current_phase}]"
        log_str += f" | Train Loss: {train_metrics['train_loss']:.4f}"
        if val_metrics:
            log_str += f" | Val Loss: {val_metrics['val_loss']:.4f}"
        log_str += f" | LR: {lr:.2e}"
        print(log_str)

        # Update history
        self.history['train_loss'].append(train_metrics['train_loss'])
        self.history['lr'].append(lr)
        self.history['epoch'].append(self.epoch + 1)
        self.history['phase'].append(self.current_phase)
        if val_metrics:
            self.history['val_loss'].append(val_metrics['val_loss'])

        # TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('Loss/train', train_metrics['train_loss'], self.epoch + 1)
            self.tb_writer.add_scalar('Learning_Rate', lr, self.epoch + 1)
            self.tb_writer.add_scalar('Training/phase', self.current_phase, self.epoch + 1)
            if val_metrics:
                self.tb_writer.add_scalar('Loss/val', val_metrics['val_loss'], self.epoch + 1)
            self.tb_writer.flush()

        # WandB
        if self.wandb_run is not None:
            metrics = {
                'epoch': self.epoch + 1,
                'phase': self.current_phase,
                'lr': lr,
                **train_metrics,
                **val_metrics
            }
            wandb.log(metrics, step=self.global_step)

    def _save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch + 1,
            'global_step': self.global_step,
            'training_phase': self.current_phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config.__dict__,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow

        torch.save(checkpoint, self.save_dir / filename)

    def _load_checkpoint(self, path: str):
        """Load training checkpoint."""
        print(f"Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Check if this is a cross-phase resume (different architectures)
        saved_phase = checkpoint.get('training_phase', self.current_phase)
        cross_phase_resume = saved_phase != self.current_phase
        
        if cross_phase_resume:
            print(f"Cross-phase resume: Phase {saved_phase} -> Phase {self.current_phase}")
            # Load with strict=False to allow missing/unexpected keys
            # This happens when going from segmentation (seg_net) to detection (dbnet)
            missing, unexpected = self.model.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )
            if missing:
                print(f"  Missing keys (expected for new head): {len(missing)} keys")
            if unexpected:
                print(f"  Unexpected keys (from previous phase): {len(unexpected)} keys")
            
            # Don't restore optimizer/scheduler state for cross-phase (fresh start for new head)
            print("  Using fresh optimizer state for new training phase")
            # Reset epoch counter for new phase
            self.epoch = 0
            self.global_step = 0
            # Keep best_loss high to ensure new checkpoints are saved
            self.best_loss = float('inf')
        else:
            # Same phase resume - load everything
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_loss = checkpoint['best_loss']
            
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if 'scaler_state_dict' in checkpoint and self.scaler is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            if 'ema_shadow' in checkpoint and self.ema is not None:
                self.ema.shadow = checkpoint['ema_shadow']

        print(f"Resumed from epoch {self.epoch}, phase {self.current_phase}")
