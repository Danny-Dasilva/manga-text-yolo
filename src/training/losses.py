"""
Loss Functions for Comic Text Detection

Includes:
- BinaryDiceLoss: For segmentation mask training
- DBLoss: Composite loss for text line detection
- FocalLoss: For handling class imbalance
- CIoULoss: Complete IoU Loss for bounding box regression
- DualAssignmentLoss: YOLOv10-style dual label assignment
- BlockDetectionLoss: Loss for block detection head
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union


class BinaryDiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1.0, p: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            pred: Predictions (B, 1, H, W) after sigmoid
            target: Targets (B, 1, H, W) binary

        Returns:
            Dice loss
        """
        # Flatten
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        # Compute intersection and union
        intersection = (pred * target).sum(dim=1)
        union = pred.pow(self.p).sum(dim=1) + target.pow(self.p).sum(dim=1)

        # Dice coefficient
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss.

        Args:
            pred: Predictions (B, 1, H, W) logits (before sigmoid)
            target: Targets (B, 1, H, W) binary

        Returns:
            Focal loss
        """
        # BCE with logits
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Probability
        p = torch.sigmoid(pred)
        p_t = p * target + (1 - p) * (1 - target)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss = alpha_t * focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class VariFocalLoss(nn.Module):
    """VariFocal Loss — asymmetric treatment of positives and negatives.

    Positives get full gradient (weighted by IoU target).
    Negatives get down-weighted by predicted score (focal suppression).
    Use with soft IoU targets for objectness instead of hard binary 0/1.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Asymmetric weighting: positives get full weight, negatives get focal
        is_positive = (target > 0.0).float()
        focal_weight = (
            target * is_positive +  # pos: weight = IoU target value
            self.alpha * (pred_sigmoid - target).abs().pow(self.gamma) * (1.0 - is_positive)  # neg: focal
        )
        loss = bce * focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class BalancedBCELoss(nn.Module):
    """
    Balanced BCE loss with Online Hard Example Mining (OHEM).

    Mines hard negative examples to balance positive/negative ratio.
    """

    def __init__(self, negative_ratio: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute balanced BCE loss.

        Args:
            pred: Predictions (B, 1, H, W) after sigmoid
            target: Targets (B, 1, H, W)
            mask: Optional mask for valid regions

        Returns:
            Balanced BCE loss
        """
        positive = target.byte()
        negative = (1 - target).byte()

        if mask is not None:
            positive = positive & mask.byte()
            negative = negative & mask.byte()

        positive_count = int(positive.sum())
        negative_count = min(int(negative.sum()), int(positive_count * self.negative_ratio))

        # BCE loss - disable autocast to avoid AMP issues
        with torch.amp.autocast('cuda', enabled=False):
            pred_fp32 = pred.float()
            target_fp32 = target.float()
            pred_clamped = torch.clamp(pred_fp32, 1e-7, 1 - 1e-7)
            loss = F.binary_cross_entropy(pred_clamped, target_fp32, reduction='none')

        # Positive loss
        positive_loss = (loss * positive.float()).sum()

        # Hard negative mining
        negative_loss = loss * negative.float()
        negative_loss = negative_loss.view(-1)

        if negative_count > 0:
            # Get top-k hard negatives
            hard_negative_loss, _ = negative_loss.topk(negative_count)
            negative_loss = hard_negative_loss.sum()
        else:
            negative_loss = torch.tensor(0.0, device=pred.device)

        # Balance
        total_count = positive_count + negative_count
        if total_count > 0:
            return (positive_loss + negative_loss) / total_count
        else:
            return torch.tensor(0.0, device=pred.device)


class MaskL1Loss(nn.Module):
    """L1 loss with mask."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked L1 loss.

        Args:
            pred: Predictions (B, 1, H, W)
            target: Targets (B, 1, H, W)
            mask: Mask (B, 1, H, W)

        Returns:
            Masked L1 loss
        """
        loss = torch.abs(pred - target)
        loss = (loss * mask).sum() / (mask.sum() + self.eps)
        return loss


class DBLoss(nn.Module):
    """
    Composite loss for DBNet text detection.

    Combines:
    - Shrink map loss (Dice + BCE)
    - Threshold map loss (L1)
    - Binary map loss (Dice + BCE)
    - Distribution regularization (prevents shrink map collapse)
    """

    def __init__(
        self,
        alpha: float = 1.0,  # Shrink map weight
        beta: float = 1.0,  # Threshold map weight - balanced with shrink
        use_focal: bool = False,
        ohem_ratio: float = 3.0,
        eps: float = 1e-6,
        use_distribution_reg: bool = True,  # Regularize shrink map distribution
        distribution_reg_weight: float = 0.5,  # Weight for distribution regularization
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.use_distribution_reg = use_distribution_reg
        self.distribution_reg_weight = distribution_reg_weight

        # Loss components
        self.dice_loss = BinaryDiceLoss()

        if use_focal:
            self.bce_loss = FocalLoss()
        else:
            self.bce_loss = BalancedBCELoss(negative_ratio=ohem_ratio)

        self.l1_loss = MaskL1Loss()

    def forward(
        self,
        pred: torch.Tensor,
        target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute DBNet loss.

        Args:
            pred: Model output (B, 3 or 4, H, W)
                - Channel 0: Shrink map (probability)
                - Channel 1: Threshold map
                - Channel 2: Binary map
                - Channel 3 (optional): Shrink logits for BCE

            target: Dictionary with:
                - 'shrink_map': (B, 1, H, W)
                - 'threshold_map': (B, 1, H, W)

        Returns:
            Total loss
        """
        # Parse predictions
        shrink_pred = pred[:, 0:1]
        threshold_pred = pred[:, 1:2]
        binary_pred = pred[:, 2:3] if pred.size(1) > 2 else shrink_pred

        # Parse targets
        shrink_target = target['shrink_map']
        threshold_target = target['threshold_map']

        # Shrink map loss
        shrink_dice = self.dice_loss(shrink_pred, shrink_target)
        shrink_bce = self.bce_loss(shrink_pred, shrink_target)
        shrink_loss = shrink_dice + shrink_bce

        # Threshold map loss (only in border region)
        border_mask = (threshold_target > 0.3).float()
        threshold_loss = self.l1_loss(threshold_pred, threshold_target, border_mask)

        # Binary map loss
        binary_dice = self.dice_loss(binary_pred, shrink_target)
        binary_bce = self.bce_loss(binary_pred, shrink_target)
        binary_loss = binary_dice + binary_bce

        # Distribution regularization: prevent shrink map collapse
        # Penalizes if predicted shrink mean deviates too far from target mean
        distribution_loss = 0.0
        if self.use_distribution_reg:
            pred_mean = shrink_pred.mean()
            target_mean = shrink_target.mean()
            # L1 penalty on mean difference
            distribution_loss = torch.abs(pred_mean - target_mean)

        # Total loss
        total_loss = (
            self.alpha * shrink_loss +
            self.beta * threshold_loss +
            binary_loss +
            self.distribution_reg_weight * distribution_loss
        )

        return total_loss


class SegmentationLoss(nn.Module):
    """
    Loss for segmentation mask training.

    Combines Dice loss and BCE loss.
    Computes BCE in full precision for AMP compatibility.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        use_focal: bool = False
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.dice_loss = BinaryDiceLoss()
        self.use_focal = use_focal

        if use_focal:
            self.bce_loss = FocalLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute segmentation loss.

        Args:
            pred: Predictions (B, 1, H, W) after sigmoid
            target: Dictionary with 'mask' key (B, 1, H, W)

        Returns:
            Combined loss
        """
        mask_target = target['mask']

        dice = self.dice_loss(pred, mask_target)
        
        if self.use_focal:
            bce = self.bce_loss(pred, mask_target)
        else:
            # Compute BCE in full precision with autocast disabled
            # This is required because F.binary_cross_entropy is unsafe with autocast
            with torch.amp.autocast('cuda', enabled=False):
                pred_fp32 = pred.float()
                target_fp32 = mask_target.float()
                # Clamp to avoid log(0) issues
                pred_clamped = torch.clamp(pred_fp32, 1e-7, 1 - 1e-7)
                bce = F.binary_cross_entropy(pred_clamped, target_fp32)

        return self.dice_weight * dice + self.bce_weight * bce


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        box1: [N, 4] boxes in xyxy format
        box2: [M, 4] boxes in xyxy format

    Returns:
        [N, M] IoU matrix
    """
    # Get areas - clamp to prevent negative values from invalid boxes
    area1 = torch.clamp((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), min=eps)  # [N]
    area2 = torch.clamp((box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1]), min=eps)  # [M]

    # Intersection
    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])  # [N, M]
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / (union_area + eps)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from xywh to xyxy format."""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x - w/2, y - h/2, x + w/2, y + h/2], dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from xyxy to xywh format."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2)/2, (y1 + y2)/2, x2 - x1, y2 - y1], dim=-1)


class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned Assigner for YOLO-style object detection.

    Uses task alignment metric: t = s^alpha * u^beta
    where s = classification score, u = IoU

    For one-to-many: topk=10 provides rich supervision
    For one-to-one: topk=1 enables NMS-free inference
    """

    def __init__(
        self,
        topk: int = 10,
        num_classes: int = 1,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9
    ):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pred_scores: torch.Tensor,  # [B, N, num_classes] - predicted class scores (after sigmoid)
        pred_boxes: torch.Tensor,   # [B, N, 4] - predicted boxes in xyxy format
        gt_labels: torch.Tensor,    # [B, M] - ground truth class labels
        gt_boxes: torch.Tensor,     # [B, M, 4] - ground truth boxes in xyxy format
        gt_mask: torch.Tensor,      # [B, M] - valid ground truth mask (1 = valid, 0 = padding)
    ) -> tuple:
        """
        Assign predictions to ground truth using task-aligned metric.

        Returns:
            assigned_gt_idx: [B, N] - index of assigned GT for each prediction (-1 if unassigned)
            assigned_labels: [B, N] - assigned class labels (0 for background)
            assigned_boxes: [B, N, 4] - assigned GT boxes
            assigned_scores: [B, N, num_classes] - soft labels for training
            fg_mask: [B, N] - foreground mask (True for assigned predictions)
        """
        batch_size, num_preds, _ = pred_boxes.shape
        _, num_gts = gt_mask.shape
        device = pred_boxes.device

        # Initialize outputs
        assigned_gt_idx = torch.full((batch_size, num_preds), -1, dtype=torch.long, device=device)
        assigned_labels = torch.zeros((batch_size, num_preds), dtype=torch.long, device=device)
        assigned_boxes = torch.zeros((batch_size, num_preds, 4), dtype=pred_boxes.dtype, device=device)
        assigned_scores = torch.zeros((batch_size, num_preds, self.num_classes), dtype=pred_scores.dtype, device=device)
        fg_mask = torch.zeros((batch_size, num_preds), dtype=torch.bool, device=device)

        for b in range(batch_size):
            # Get valid GTs for this batch
            valid_mask = gt_mask[b] > 0
            n_valid_gt = valid_mask.sum().item()

            if n_valid_gt == 0:
                continue

            valid_gt_boxes = gt_boxes[b, valid_mask]  # [n_valid, 4]
            valid_gt_labels = gt_labels[b, valid_mask]  # [n_valid]

            # Compute IoU between predictions and GTs
            iou = box_iou(pred_boxes[b], valid_gt_boxes)  # [N, n_valid]

            # Get predicted scores for the GT classes
            # For single class, just use the score directly
            if self.num_classes == 1:
                pred_scores_for_gt = pred_scores[b, :, 0:1].expand(-1, n_valid_gt)  # [N, n_valid]
            else:
                # Multi-class: get score for each GT's class
                pred_scores_for_gt = pred_scores[b].gather(
                    1, valid_gt_labels[None, :].expand(num_preds, -1)
                )  # [N, n_valid]

            # Task alignment metric: t = s^alpha * u^beta
            # Clamp values to avoid numerical issues with pow()
            pred_scores_clamped = pred_scores_for_gt.clamp(self.eps, 1.0)
            iou_clamped = iou.clamp(0, 1.0)
            alignment_metric = (pred_scores_clamped.pow(self.alpha) * iou_clamped.pow(self.beta))  # [N, n_valid]
            # Replace any NaN values with 0
            alignment_metric = torch.nan_to_num(alignment_metric, nan=0.0, posinf=0.0, neginf=0.0)

            # Select top-k predictions for each GT
            topk_metric, topk_idx = alignment_metric.topk(
                min(self.topk, num_preds), dim=0, largest=True
            )  # [topk, n_valid]

            # Create candidate mask: which (pred, gt) pairs are candidates
            candidate_mask = torch.zeros((num_preds, n_valid_gt), dtype=torch.bool, device=device)
            for gt_i in range(n_valid_gt):
                candidate_mask[topk_idx[:, gt_i], gt_i] = True

            # Filter candidates by IoU > 0
            candidate_mask = candidate_mask & (iou > self.eps)

            # Compute alignment scores for candidates
            alignment_scores = alignment_metric * candidate_mask.float()  # [N, n_valid]

            # For each prediction, select the GT with highest alignment score
            max_align_scores, max_gt_idx = alignment_scores.max(dim=1)  # [N], [N]

            # Predictions with max_align_score > 0 are foreground
            fg = max_align_scores > self.eps

            # One-to-one assignment: each GT can only be assigned to one prediction
            if self.topk == 1:
                # For each GT, find the best prediction
                for gt_i in range(n_valid_gt):
                    gt_candidates = (max_gt_idx == gt_i) & fg
                    if gt_candidates.sum() > 0:
                        # Keep only the best prediction for this GT
                        candidate_scores = alignment_metric[:, gt_i] * gt_candidates.float()
                        best_pred = candidate_scores.argmax()
                        # Zero out other assignments to this GT
                        other_preds = gt_candidates.clone()
                        other_preds[best_pred] = False
                        fg[other_preds] = False
                        max_gt_idx[other_preds] = 0

            # Store assignments
            fg_mask[b] = fg
            assigned_gt_idx[b, fg] = max_gt_idx[fg]
            assigned_labels[b, fg] = valid_gt_labels[max_gt_idx[fg]]
            # Cast to same dtype as assigned_boxes (may be half from AMP)
            assigned_boxes[b, fg] = valid_gt_boxes[max_gt_idx[fg]].to(assigned_boxes.dtype)

            # Compute soft labels based on IoU (cast to match assigned_scores dtype for AMP)
            fg_iou = iou[fg, max_gt_idx[fg]].to(assigned_scores.dtype)  # IoU for assigned pairs
            if self.num_classes == 1:
                assigned_scores[b, fg, 0] = fg_iou
            else:
                # Set score for assigned class
                assigned_scores[b, fg, assigned_labels[b, fg]] = fg_iou

        return assigned_gt_idx, assigned_labels, assigned_boxes, assigned_scores, fg_mask


class CIoULoss(nn.Module):
    """Complete IoU Loss for bounding box regression.

    CIoU = IoU - (distance^2 / c^2) - alpha * v
    where v measures aspect ratio consistency.
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted boxes [N, 4] in xyxy format
            target: Target boxes [N, 4] in xyxy format
        """
        # Intersection
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union - clamp areas to prevent negative values from invalid boxes
        pred_area = torch.clamp((pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]), min=self.eps)
        target_area = torch.clamp((target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1]), min=self.eps)
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + self.eps)
        
        # Enclosing box
        enclose_x1 = torch.min(pred[:, 0], target[:, 0])
        enclose_y1 = torch.min(pred[:, 1], target[:, 1])
        enclose_x2 = torch.max(pred[:, 2], target[:, 2])
        enclose_y2 = torch.max(pred[:, 3], target[:, 3])
        
        # Diagonal distance squared
        c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + self.eps
        
        # Center distance squared
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2
        rho2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # Aspect ratio - clamp dimensions to prevent division by zero/negative values
        pred_w = torch.clamp(pred[:, 2] - pred[:, 0], min=self.eps)
        pred_h = torch.clamp(pred[:, 3] - pred[:, 1], min=self.eps)
        target_w = torch.clamp(target[:, 2] - target[:, 0], min=self.eps)
        target_h = torch.clamp(target[:, 3] - target[:, 1], min=self.eps)
        
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2
        )
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)
        
        # CIoU
        ciou = iou - rho2 / c2 - alpha * v
        
        # Clamp final loss to prevent NaN propagation and extreme values
        loss = 1 - ciou
        return torch.clamp(loss, min=0.0, max=10.0)


class DualAssignmentLoss(nn.Module):
    """YOLOv10-style dual label assignment loss.

    Uses two assignment strategies during training:
    - One-to-many (o2m): Rich supervision signal, topk=10
    - One-to-one (o2o): NMS-free inference head, topk=1

    Only the o2o head is used at inference time.

    NOTE: o2m_weight changed from 0.25 to 1.0 to give equal weight to
    both heads. The original 0.25 underweighted the rich supervision
    signal from the o2m head, contributing to overfitting.
    """
    def __init__(
        self,
        o2m_weight: float = 1.0,  # Was 0.25 - increased for balanced training
        o2o_weight: float = 1.0,
        box_weight: float = 5.0,  # Was 7.5 - reduced to prevent box loss dominating
        cls_weight: float = 1.5,  # Increased from 1.0 - more gradient to classification
        obj_weight: float = 2.5,  # Increased from 1.5 - more gradient to confidence
        num_classes: int = 1,
    ):
        super().__init__()
        self.o2m_weight = o2m_weight
        self.o2o_weight = o2o_weight
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.num_classes = num_classes

        # Task-aligned assigners for one-to-many and one-to-one
        self.o2m_assigner = TaskAlignedAssigner(topk=13, num_classes=num_classes)
        self.o2o_assigner = TaskAlignedAssigner(topk=1, num_classes=num_classes)

        self.box_loss = CIoULoss()
        # VariFocal Loss: asymmetric — full gradient on positives, focal on negatives
        # Uses soft IoU targets instead of hard binary 0/1
        self.obj_loss = VariFocalLoss(alpha=0.75, gamma=2.0, reduction='none')

    def forward(
        self,
        predictions: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            predictions: Either a tuple (pred_o2m, pred_o2o) or single tensor
                - pred_o2m: One-to-many predictions [B, N, 5+num_classes] (x, y, w, h, obj, cls...)
                - pred_o2o: One-to-one predictions [B, N, 5+num_classes]
            targets: Dict with:
                - 'boxes': [B, M, 4] GT boxes in xywh format (normalized 0-1)
                - 'labels': [B, M] GT class labels
                - 'mask': [B, M] valid GT mask (1 = valid, 0 = padding)
        """
        # Handle tuple input from BlockDetector during training
        if isinstance(predictions, tuple):
            pred_o2m, pred_o2o = predictions
        else:
            # Single tensor - use same predictions for both (inference mode)
            pred_o2m = pred_o2o = predictions
        
        loss_o2m = self._compute_loss(pred_o2m, targets, self.o2m_assigner)
        loss_o2o = self._compute_loss(pred_o2o, targets, self.o2o_assigner)

        return self.o2m_weight * loss_o2m + self.o2o_weight * loss_o2o

    def _compute_loss(
        self,
        pred: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        assigner: TaskAlignedAssigner
    ) -> torch.Tensor:
        """
        Compute detection loss with task-aligned assignment.

        Args:
            pred: [B, N, 5+num_classes] predictions (x, y, w, h, obj, cls...)
            targets: Dict with 'boxes', 'labels', 'mask'
            assigner: TaskAlignedAssigner instance

        Returns:
            Combined box + objectness + classification loss
        """
        device = pred.device
        batch_size, num_preds, _ = pred.shape

        # Parse predictions
        pred_boxes_xywh = pred[..., :4]  # [B, N, 4] xywh
        pred_obj = pred[..., 4]  # [B, N] objectness logits
        pred_cls = pred[..., 5:] if pred.shape[-1] > 5 else None  # class logits (None for merged head)

        # Convert predictions to xyxy for IoU computation
        pred_boxes_xyxy = xywh_to_xyxy(pred_boxes_xywh)  # [B, N, 4]

        # Get target data - handle both formats:
        # Format 1: 'boxes' [B, M, 4] and 'labels' [B, M] (standard format)
        # Format 2: 'blocks' [B, M, 5] with [cls, x, y, w, h] (from block dataset)
        if 'boxes' in targets:
            gt_boxes_xywh = targets['boxes']  # [B, M, 4] xywh
            gt_labels = targets['labels']  # [B, M]
            gt_mask = targets.get('mask', (gt_boxes_xywh.sum(dim=-1) > 0).float())  # [B, M]
        elif 'blocks' in targets:
            # Extract from blocks tensor [B, M, 5] = [cls, x, y, w, h]
            blocks = targets['blocks']
            gt_labels = blocks[..., 0].long()  # [B, M] class labels
            gt_boxes_xywh = blocks[..., 1:5]  # [B, M, 4] xywh
            # Create mask from num_blocks
            if 'num_blocks' in targets:
                num_blocks = targets['num_blocks']  # [B]
                max_blocks = blocks.shape[1]
                # Create mask: True for indices < num_blocks
                indices = torch.arange(max_blocks, device=device).unsqueeze(0)  # [1, M]
                gt_mask = (indices < num_blocks.unsqueeze(1)).float()  # [B, M]
            else:
                gt_mask = (gt_boxes_xywh.sum(dim=-1) > 0).float()
        else:
            raise KeyError("targets must contain either 'boxes' or 'blocks'")

        # Convert GT boxes to xyxy
        gt_boxes_xyxy = xywh_to_xyxy(gt_boxes_xywh)  # [B, M, 4]

        # Get predicted scores for assignment (use sigmoid of objectness)
        pred_scores = torch.sigmoid(pred_obj).unsqueeze(-1)  # [B, N, 1]
        if self.num_classes > 1 and pred_cls is not None:
            pred_scores = pred_scores * torch.sigmoid(pred_cls)  # [B, N, num_classes]

        # Run task-aligned assignment
        _, assigned_labels, assigned_boxes_xyxy, assigned_scores, fg_mask = assigner(
            pred_scores, pred_boxes_xyxy, gt_labels, gt_boxes_xyxy, gt_mask
        )

        # Count foreground samples
        num_fg = fg_mask.sum().item()

        if num_fg == 0:
            # No foreground samples - push all objectness toward background
            # VFL focal weighting already suppresses well-classified negatives
            return self.obj_loss(pred_obj, torch.zeros_like(pred_obj)).mean()

        # Convert assigned boxes back to xywh for loss computation
        assigned_boxes_xywh = xyxy_to_xywh(assigned_boxes_xyxy)

        # === Box Loss (CIoU) ===
        # Only compute for foreground predictions
        fg_pred_boxes = pred_boxes_xyxy[fg_mask]  # [num_fg, 4]
        fg_gt_boxes = assigned_boxes_xyxy[fg_mask]  # [num_fg, 4]
        box_loss = self.box_loss(fg_pred_boxes, fg_gt_boxes).mean()

        # === Objectness Loss (VariFocal) ===
        # Soft IoU targets: foreground anchors get IoU value, background gets 0
        obj_targets = torch.zeros_like(pred_obj)
        # Use IoU from assignment as soft target (richer signal than binary 0/1)
        fg_ious = assigned_scores[fg_mask].max(dim=-1).values  # [num_fg] — robust for any num_classes
        obj_targets[fg_mask] = fg_ious.detach()
        # Normalize by num_fg for stable gradients regardless of anchor count
        obj_loss = self.obj_loss(pred_obj, obj_targets).sum() / max(num_fg, 1)

        # === Classification Loss ===
        if self.num_classes > 1 and pred_cls is not None:
            # Multi-class: compute cls loss on foreground
            fg_cls_pred = pred_cls[fg_mask]
            fg_cls_target = assigned_scores[fg_mask]
            cls_loss = F.binary_cross_entropy_with_logits(
                fg_cls_pred, fg_cls_target, reduction='none'
            ).mean()
            total_loss = (
                self.box_weight * box_loss +
                self.obj_weight * obj_loss +
                self.cls_weight * cls_loss
            )
        else:
            # Single-class: obj is the only confidence signal (cls merged into obj)
            total_loss = (
                self.box_weight * box_loss +
                self.obj_weight * obj_loss
            )

        return total_loss


class BlockDetectionLoss(nn.Module):
    """Loss for block detection head.

    Uses task-aligned assignment for matching predictions to ground truth.
    Computes CIoU box loss, BCE objectness loss, and BCE classification loss.
    """

    def __init__(
        self,
        box_weight: float = 5.0,  # Was 7.5 - reduced to prevent box loss dominating
        obj_weight: float = 1.5,  # Was 1.0 - increased for better foreground/background separation
        cls_weight: float = 1.0,  # Was 0.5 - increased for better classification
        num_classes: int = 1,
        topk: int = 10,
    ):
        super().__init__()
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.num_classes = num_classes

        # Task-aligned assigner for matching
        self.assigner = TaskAlignedAssigner(topk=topk, num_classes=num_classes)

        self.box_loss = CIoULoss()
        # VariFocal Loss: asymmetric — full gradient on positives, focal on negatives
        self.obj_loss = VariFocalLoss(alpha=0.75, gamma=2.0, reduction='none')
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        predictions: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute block detection loss.

        Args:
            predictions: Either single tensor [B, N, 5+num_classes] or tuple (o2m_preds, o2o_preds)
                        for dual assignment training (YOLOv10 style)
            targets: Dict with:
                - 'boxes': [B, M, 4] GT boxes in xywh format (normalized 0-1)
                - 'labels': [B, M] GT class labels
                - 'mask': [B, M] valid GT mask (optional, inferred from boxes if missing)

        Returns:
            Combined weighted loss (box + obj + cls)
        """
        # Handle dual assignment (YOLOv10 style)
        if isinstance(predictions, tuple):
            o2m_preds, o2o_preds = predictions
            # Compute loss for both heads and combine
            o2m_loss = self._compute_single_head_loss(o2m_preds, targets)
            o2o_loss = self._compute_single_head_loss(o2o_preds, targets)
            # One-to-many loss is main supervision, one-to-one is auxiliary
            return o2m_loss + 0.5 * o2o_loss
        else:
            return self._compute_single_head_loss(predictions, targets)

    def _compute_single_head_loss(
        self,
        predictions: torch.Tensor,
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss for a single prediction head."""
        device = predictions.device
        dtype = predictions.dtype
        batch_size, num_preds, _ = predictions.shape

        # Check for NaN/Inf in predictions
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            # Return a small valid loss to keep training going
            return torch.tensor(0.1, device=device, dtype=dtype, requires_grad=True)

        # Parse predictions - clamp to avoid extreme values
        pred_boxes_xywh = predictions[..., :4].clamp(-10, 10)  # [B, N, 4]
        pred_obj = predictions[..., 4].clamp(-20, 20)  # [B, N]
        pred_cls = predictions[..., 5:].clamp(-20, 20) if predictions.shape[-1] > 5 else None

        # Convert to xyxy for IoU computation
        pred_boxes_xyxy = xywh_to_xyxy(pred_boxes_xywh)  # [B, N, 4]

        # Get targets - support both 'boxes' [B, M, 4] and 'blocks' [B, M, 5] formats
        if 'boxes' in targets:
            gt_boxes_xywh = targets['boxes']  # [B, M, 4] xywh
            gt_labels = targets.get('labels', torch.zeros(gt_boxes_xywh.shape[:2], dtype=torch.long, device=device))
            gt_mask = targets.get('mask', (gt_boxes_xywh.sum(dim=-1) > 0).float())
        elif 'blocks' in targets:
            blocks = targets['blocks']  # [B, M, 5] = [cls, x, y, w, h]
            gt_labels = blocks[..., 0].long()
            gt_boxes_xywh = blocks[..., 1:5]
            if 'num_blocks' in targets:
                num_blocks = targets['num_blocks']
                max_blocks = blocks.shape[1]
                indices = torch.arange(max_blocks, device=device).unsqueeze(0)
                gt_mask = (indices < num_blocks.unsqueeze(1)).float()
            else:
                gt_mask = (gt_boxes_xywh.sum(dim=-1) > 0).float()
        else:
            raise KeyError("targets must contain either 'boxes' or 'blocks'")

        # Convert GT to xyxy
        gt_boxes_xyxy = xywh_to_xyxy(gt_boxes_xywh)  # [B, M, 4]

        # Compute predicted scores for assignment - clamp sigmoid output
        pred_scores = torch.sigmoid(pred_obj).clamp(1e-6, 1 - 1e-6).unsqueeze(-1)  # [B, N, 1]
        if self.num_classes > 1 and pred_cls is not None:
            pred_scores = pred_scores * torch.sigmoid(pred_cls).clamp(1e-6, 1 - 1e-6)  # [B, N, num_classes]

        # Run task-aligned assignment
        _, assigned_labels, assigned_boxes_xyxy, assigned_scores, fg_mask = self.assigner(
            pred_scores, pred_boxes_xyxy, gt_labels, gt_boxes_xyxy, gt_mask
        )

        # Count foreground
        num_fg = fg_mask.sum().item()

        if num_fg == 0:
            # No matches - push all objectness toward background
            return self.obj_loss(pred_obj, torch.zeros_like(pred_obj)).mean()

        # === Box Loss (CIoU) ===
        fg_pred_boxes = pred_boxes_xyxy[fg_mask]  # [num_fg, 4]
        fg_gt_boxes = assigned_boxes_xyxy[fg_mask]  # [num_fg, 4]
        box_loss = self.box_loss(fg_pred_boxes, fg_gt_boxes)
        # Handle potential NaN in box loss
        if torch.isnan(box_loss).any():
            box_loss = torch.zeros_like(box_loss)
        box_loss = box_loss.mean()

        # === Objectness Loss (VariFocal) ===
        # Soft IoU targets: foreground gets IoU value, background gets 0
        obj_targets = torch.zeros_like(pred_obj)
        fg_ious = assigned_scores[fg_mask].max(dim=-1).values
        obj_targets[fg_mask] = fg_ious.detach()
        obj_loss = self.obj_loss(pred_obj, obj_targets).sum() / max(num_fg, 1)

        # === Classification Loss ===
        if pred_cls is not None:
            fg_cls_pred = pred_cls[fg_mask]
            fg_cls_target = assigned_scores[fg_mask].clamp(0, 1)
            cls_loss = self.cls_loss(fg_cls_pred, fg_cls_target).mean()
            total_loss = (
                self.box_weight * box_loss +
                self.obj_weight * obj_loss +
                self.cls_weight * cls_loss
            )
        else:
            # Single-class merged head: obj is the only confidence signal
            total_loss = (
                self.box_weight * box_loss +
                self.obj_weight * obj_loss
            )

        # Final NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.1, device=device, dtype=dtype, requires_grad=True)

        return total_loss.clamp(0, 100)


class UnifiedLoss(nn.Module):
    """
    Combined loss for unified training mode.

    Computes all three losses (segmentation, detection, block) in one pass
    and returns their weighted sum for joint training.

    Args:
        seg_weight: Weight for segmentation loss (default: 1.0)
        det_weight: Weight for detection loss (default: 1.0)
        block_weight: Weight for block detection loss (default: 1.0)
        return_components: If True, return dict with individual losses (default: False)
    """

    def __init__(
        self,
        seg_weight: float = 1.0,
        det_weight: float = 1.0,
        block_weight: float = 1.0,
        return_components: bool = False,
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.det_weight = det_weight
        self.block_weight = block_weight
        self.return_components = return_components

        # Individual loss functions
        self.seg_loss = SegmentationLoss()
        self.det_loss = DBLoss()
        self.block_loss = DualAssignmentLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute unified loss.

        Args:
            outputs: Dict with 'mask', 'lines', 'blocks' from unified forward
            targets: Dict with 'mask', 'shrink_map', 'threshold_map', 'bboxes', 'labels'

        Returns:
            Total loss (scalar) or dict with individual losses if return_components=True
        """
        device = next(iter(outputs.values())).device if isinstance(outputs, dict) else outputs.device
        dtype = torch.float32

        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        loss_dict = {}

        # Segmentation loss
        if 'mask' in outputs and 'mask' in targets:
            seg_loss_val = self.seg_loss(outputs['mask'], {'mask': targets['mask']})
            loss_dict['seg_loss'] = seg_loss_val
            total_loss = total_loss + self.seg_weight * seg_loss_val

        # Detection (DBNet) loss
        if 'lines' in outputs and 'shrink_map' in targets and 'threshold_map' in targets:
            det_targets = {
                'shrink_map': targets['shrink_map'],
                'threshold_map': targets['threshold_map'],
            }
            det_loss_val = self.det_loss(outputs['lines'], det_targets)
            loss_dict['det_loss'] = det_loss_val
            total_loss = total_loss + self.det_weight * det_loss_val

        # Block detection loss
        if 'blocks' in outputs and ('bboxes' in targets or 'boxes' in targets or 'blocks' in targets):
            # Support 'blocks', 'bboxes', and 'boxes' keys for flexibility
            if 'blocks' in targets:
                # blocks format: [B, M, 5] as [cls, x, y, w, h] normalized
                block_targets = {
                    'blocks': targets['blocks'],
                    'num_blocks': targets.get('num_blocks', None),
                }
            else:
                boxes_key = 'boxes' if 'boxes' in targets else 'bboxes'
                block_targets = {
                    'boxes': targets[boxes_key],
                    'labels': targets.get('labels', torch.zeros(targets[boxes_key].shape[0], targets[boxes_key].shape[1], device=device).long()),
                }
            block_loss_val = self.block_loss(outputs['blocks'], block_targets)
            loss_dict['block_loss'] = block_loss_val
            total_loss = total_loss + self.block_weight * block_loss_val

        if self.return_components:
            loss_dict['total'] = total_loss
            return loss_dict

        return total_loss


def create_loss(mode: str = 'segmentation', **kwargs) -> nn.Module:
    """
    Factory function to create appropriate loss.

    Args:
        mode: 'segmentation', 'detection', 'block', 'dual_assignment', or 'unified'
        **kwargs: Additional arguments for loss

    Returns:
        Loss module
    """
    if mode == 'segmentation':
        return SegmentationLoss(**kwargs)
    elif mode == 'detection':
        return DBLoss(**kwargs)
    elif mode == 'block':
        return BlockDetectionLoss(**kwargs)
    elif mode == 'dual_assignment':
        return DualAssignmentLoss(**kwargs)
    elif mode == 'unified':
        return UnifiedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
