#!/usr/bin/env python3
"""
Evaluate teacher models (YOLO12-X, YOLO12-S, CTD) against AnimeText ground truth.

Compares:
- YOLO12-X (deepghs/AnimeText_yolo) as teacher candidate
- YOLO12-S (deepghs/AnimeText_yolo) as faster alternative
- CTD (mayocream/comic-text-detector) as baseline reference
- Ground truth from AnimeText annotations

Outputs:
- Per-model precision/recall/F1/mAP at multiple IoU thresholds
- Cross-model agreement analysis
- Accuracy gap audit (what each model misses)
- Visual samples of disagreements

Usage:
    python scripts/evaluate_teacher.py --limit 500
    python scripts/evaluate_teacher.py --limit 500 --visualize --vis-limit 20
    python scripts/evaluate_teacher.py --models yolo12x,ctd --limit 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float

    @property
    def area(self) -> float:
        return max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1])


@dataclass
class ModelResult:
    name: str
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0


@dataclass
class ImageMetrics:
    image_name: str
    gt_count: int
    results: Dict[str, ModelResult] = field(default_factory=dict)
    # Per-model metrics at IoU=0.5
    per_model: Dict[str, Dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IoU and matching
# ---------------------------------------------------------------------------

def compute_iou(box1: List[int], box2: List[int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def match_detections(
    predictions: List[Detection],
    gt_boxes: List[Detection],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int, List[float]]:
    """Match predictions to GT. Returns (TP, FP, FN, matched_ious)."""
    if not gt_boxes:
        return 0, len(predictions), 0, []
    if not predictions:
        return 0, 0, len(gt_boxes), []

    gt_matched = [False] * len(gt_boxes)
    matched_ious = []
    tp = fp = 0

    # Sort predictions by confidence (highest first)
    sorted_preds = sorted(predictions, key=lambda d: d.confidence, reverse=True)

    for pred in sorted_preds:
        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            if gt_matched[i]:
                continue
            iou = compute_iou(pred.bbox, gt.bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_threshold and best_idx >= 0:
            tp += 1
            gt_matched[best_idx] = True
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)
    return tp, fp, fn, matched_ious


def compute_ap(
    predictions: List[Detection],
    gt_boxes: List[Detection],
    iou_threshold: float = 0.5,
) -> float:
    """Compute Average Precision at a given IoU threshold."""
    if not gt_boxes:
        return 1.0 if not predictions else 0.0
    if not predictions:
        return 0.0

    sorted_preds = sorted(predictions, key=lambda d: d.confidence, reverse=True)
    gt_matched = [False] * len(gt_boxes)
    tp_list = []
    fp_list = []

    for pred in sorted_preds:
        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            if gt_matched[i]:
                continue
            iou = compute_iou(pred.bbox, gt.bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_threshold and best_idx >= 0:
            tp_list.append(1)
            fp_list.append(0)
            gt_matched[best_idx] = True
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        prec_at_recall = precisions[recalls >= t]
        ap += max(prec_at_recall) / 11.0 if len(prec_at_recall) > 0 else 0.0
    return ap


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_yolo12_onnx(model_path: str, device: str = "cuda"):
    """Load YOLO12 ONNX model (works for both S and X variants)."""
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "cuda" in device
        else ["CPUExecutionProvider"]
    )
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]

    session = ort.InferenceSession(str(model_path), providers=providers)
    print(f"  Loaded {model_path}")
    print(f"    Provider: {session.get_providers()[0]}")
    return session


def load_ctd_onnx(model_path: str, device: str = "cuda"):
    """Load CTD ONNX model."""
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "cuda" in device
        else ["CPUExecutionProvider"]
    )
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]

    session = ort.InferenceSession(str(model_path), providers=providers)
    print(f"  Loaded {model_path}")
    print(f"    Provider: {session.get_providers()[0]}")
    return session


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_yolo(image: np.ndarray, input_size: int = 640) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Letterbox resize for YOLO12."""
    h, w = image.shape[:2]
    scale = min(input_size / h, input_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    tensor = padded.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[None]  # [1, 3, H, W]
    return tensor, scale, (h, w)


def preprocess_ctd(image: np.ndarray, input_size: int = 1024) -> Tuple[np.ndarray, dict]:
    """Direct resize for CTD model (no letterbox — matches generate_block_annotations.py)."""
    h, w = image.shape[:2]

    # CTD uses direct resize to 1024x1024 (not letterbox/center-pad)
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    tensor = img_rgb.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[None]

    meta = {"scale_x": w / input_size, "scale_y": h / input_size, "original_size": (h, w)}
    return tensor, meta


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

def postprocess_yolo12(
    output: np.ndarray,
    orig_size: Tuple[int, int],
    scale: float,
    conf_threshold: float = 0.272,
    nms_threshold: float = 0.45,
) -> List[Detection]:
    """Parse YOLO12 output [1, 5, 8400] -> detections."""
    h, w = orig_size

    if output.ndim == 3:
        output = output[0]
    if output.shape[0] == 5:
        output = output.T  # -> [8400, 5]

    cx, cy, bw, bh, conf = output[:, 0], output[:, 1], output[:, 2], output[:, 3], output[:, 4]

    mask = conf >= conf_threshold
    if not np.any(mask):
        return []

    cx, cy, bw, bh, conf = cx[mask], cy[mask], bw[mask], bh[mask], conf[mask]

    x1 = np.clip((cx - bw / 2) / scale, 0, w).astype(int)
    y1 = np.clip((cy - bh / 2) / scale, 0, h).astype(int)
    x2 = np.clip((cx + bw / 2) / scale, 0, w).astype(int)
    y2 = np.clip((cy + bh / 2) / scale, 0, h).astype(int)

    valid = (x2 > x1) & (y2 > y1)
    x1, y1, x2, y2, conf = x1[valid], y1[valid], x2[valid], y2[valid], conf[valid]

    if len(conf) == 0:
        return []

    # NMS
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(float)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), conf_threshold, nms_threshold)
    if len(indices) == 0:
        return []
    indices = indices.flatten()

    return [
        Detection(bbox=[int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])], confidence=float(conf[i]))
        for i in indices
    ]


def postprocess_ctd(
    outputs: list,
    meta: dict,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.5,
) -> List[Detection]:
    """Parse CTD block output [1, 64512, 7] -> detections.

    Matches generate_block_annotations.py: direct resize, conf=objectness only.
    """
    h, w = meta["original_size"]
    scale_x = meta["scale_x"]
    scale_y = meta["scale_y"]

    blk = outputs[0]  # [1, 64512, 7]
    if blk is None or blk.size == 0:
        return []

    blk = blk.squeeze(0)  # [64512, 7]
    cx, cy, bw, bh = blk[:, 0], blk[:, 1], blk[:, 2], blk[:, 3]
    # CTD uses objectness directly as confidence (not obj*cls)
    conf = blk[:, 4]

    mask = conf >= conf_threshold
    if not np.any(mask):
        return []

    cx, cy, bw, bh, conf = cx[mask], cy[mask], bw[mask], bh[mask], conf[mask]

    # Scale from 1024 space back to original (direct resize, not letterbox)
    x1 = np.clip((cx - bw / 2) * scale_x, 0, w).astype(int)
    y1 = np.clip((cy - bh / 2) * scale_y, 0, h).astype(int)
    x2 = np.clip((cx + bw / 2) * scale_x, 0, w).astype(int)
    y2 = np.clip((cy + bh / 2) * scale_y, 0, h).astype(int)

    valid = (x2 - x1 > 5) & (y2 - y1 > 5)
    x1, y1, x2, y2, conf = x1[valid], y1[valid], x2[valid], y2[valid], conf[valid]

    if len(conf) == 0:
        return []

    # NMS
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(float)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), conf_threshold, nms_threshold)
    if len(indices) == 0:
        return []
    indices = indices.flatten()

    return [
        Detection(bbox=[int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])], confidence=float(conf[i]))
        for i in indices
    ]


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

def load_gt_annotations(ann_path: Path) -> List[Detection]:
    """Load ground truth from AnimeText annotations/*.json."""
    if not ann_path.exists():
        return []
    with open(ann_path) as f:
        data = json.load(f)

    detections = []
    for block in data.get("text_blocks", []):
        bbox = block["bbox"]  # [x1, y1, x2, y2]
        if (bbox[2] - bbox[0]) > 2 and (bbox[3] - bbox[1]) > 2:
            detections.append(Detection(bbox=bbox, confidence=block.get("confidence", 1.0)))
    return detections


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

COLORS = {
    "gt": (128, 128, 128),      # Gray
    "yolo12x": (0, 255, 0),     # Green
    "yolo12s": (255, 165, 0),   # Orange (BGR)
    "ctd": (255, 0, 0),         # Blue
}


def draw_detections(
    image: np.ndarray,
    detections: List[Detection],
    color: Tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> np.ndarray:
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        text = f"{label}:{det.confidence:.2f}"
        cv2.putText(vis, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return vis


def create_comparison_grid(
    image: np.ndarray,
    gt: List[Detection],
    model_results: Dict[str, ModelResult],
    image_name: str,
) -> np.ndarray:
    """Create a grid comparing GT vs all models."""
    panels = []

    # Panel 1: GT
    vis_gt = draw_detections(image, gt, COLORS["gt"], "GT")
    cv2.putText(vis_gt, f"Ground Truth ({len(gt)} boxes)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    panels.append(vis_gt)

    # Panel per model
    for name, result in model_results.items():
        color = COLORS.get(name, (0, 255, 255))
        vis = draw_detections(image, result.detections, color, name[:4])
        # Also draw GT in thin gray
        for det in gt:
            cv2.rectangle(vis, tuple(det.bbox[:2]), tuple(det.bbox[2:]), COLORS["gt"], 1)
        cv2.putText(vis, f"{name} ({len(result.detections)} boxes, {result.inference_ms:.1f}ms)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panels.append(vis)

    # Arrange in grid
    n = len(panels)
    if n <= 2:
        return np.hstack(panels)
    elif n <= 4:
        # 2x2 grid
        h, w = panels[0].shape[:2]
        while len(panels) < 4:
            panels.append(np.zeros_like(panels[0]))
        top = np.hstack(panels[:2])
        bottom = np.hstack(panels[2:4])
        return np.vstack([top, bottom])
    else:
        return np.hstack(panels)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate teacher models against GT")
    parser.add_argument("--models", default="yolo12x,yolo12s,ctd",
                        help="Comma-separated models to evaluate (yolo12x,yolo12s,ctd)")
    parser.add_argument("--image-dir", default="data/merged_val/images")
    parser.add_argument("--ann-dir", default="data/merged_val/annotations",
                        help="AnimeText GT annotations directory")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)

    # Model paths
    parser.add_argument("--yolo12x-path", default="models/animetext-yolo/yolo12x_animetext/model.onnx")
    parser.add_argument("--yolo12s-path", default="models/animetext-yolo/yolo12s_animetext/model.onnx")
    parser.add_argument("--ctd-path", default="models/comic-text-detector.onnx")

    # Thresholds
    parser.add_argument("--yolo12x-conf", type=float, default=0.425,
                        help="YOLO12-X confidence threshold (from threshold.json)")
    parser.add_argument("--yolo12s-conf", type=float, default=0.272,
                        help="YOLO12-S confidence threshold (from threshold.json)")
    parser.add_argument("--ctd-conf", type=float, default=0.3)

    # Visualization
    parser.add_argument("--visualize", action="store_true", help="Save visual comparisons")
    parser.add_argument("--vis-limit", type=int, default=20, help="Max images to visualize")
    parser.add_argument("--output-dir", default="outputs/teacher_eval")

    # IoU thresholds for mAP
    parser.add_argument("--iou-thresholds", default="0.5,0.75",
                        help="IoU thresholds for evaluation")

    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    iou_thresholds = [float(t) for t in args.iou_thresholds.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("TEACHER MODEL EVALUATION")
    print("=" * 70)

    sessions = {}
    model_configs = {
        "yolo12x": {"path": args.yolo12x_path, "conf": args.yolo12x_conf, "input_size": 640, "type": "yolo"},
        "yolo12s": {"path": args.yolo12s_path, "conf": args.yolo12s_conf, "input_size": 640, "type": "yolo"},
        "ctd":     {"path": args.ctd_path,     "conf": args.ctd_conf,     "input_size": 1024, "type": "ctd"},
    }

    print("\nLoading models...")
    for name in model_names:
        cfg = model_configs[name]
        path = Path(cfg["path"])
        if not path.exists():
            # Try relative to script parent
            path = Path(__file__).parent.parent / cfg["path"]
        if not path.exists():
            print(f"  SKIP {name}: model not found at {cfg['path']}")
            continue
        if cfg["type"] == "yolo":
            sessions[name] = load_yolo12_onnx(str(path), args.device)
        else:
            sessions[name] = load_ctd_onnx(str(path), args.device)

    if not sessions:
        print("ERROR: No models loaded!")
        sys.exit(1)

    active_models = [n for n in model_names if n in sessions]
    print(f"\nActive models: {active_models}")

    # -----------------------------------------------------------------------
    # Collect images
    # -----------------------------------------------------------------------
    image_dir = Path(args.image_dir)
    ann_dir = Path(args.ann_dir)

    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    # Only keep images that have GT annotations
    image_paths = [p for p in image_paths if (ann_dir / f"{p.stem}.json").exists()]
    image_paths = image_paths[:args.limit]

    print(f"Evaluating on {len(image_paths)} images with GT annotations")
    print(f"IoU thresholds: {iou_thresholds}")

    # -----------------------------------------------------------------------
    # Warmup
    # -----------------------------------------------------------------------
    print(f"\nWarming up ({args.warmup} iterations)...")
    warmup_img = cv2.imread(str(image_paths[0]))
    for _ in range(args.warmup):
        for name in active_models:
            cfg = model_configs[name]
            if cfg["type"] == "yolo":
                tensor, scale, orig = preprocess_yolo(warmup_img, cfg["input_size"])
                sessions[name].run(None, {sessions[name].get_inputs()[0].name: tensor})
            else:
                tensor, meta = preprocess_ctd(warmup_img, cfg["input_size"])
                sessions[name].run(None, {sessions[name].get_inputs()[0].name: tensor})

    # -----------------------------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------------------------
    print("\nRunning evaluation...")

    # Accumulators per model, per IoU threshold
    # {model_name: {iou_thresh: {tp, fp, fn, ious, aps}}}
    metrics = {
        name: {
            iou: {"tp": 0, "fp": 0, "fn": 0, "ious": [], "aps": []}
            for iou in iou_thresholds
        }
        for name in active_models
    }
    timing = {name: [] for name in active_models}
    det_counts = {name: [] for name in active_models}

    # Cross-model agreement tracking
    agreement = {}
    for i, m1 in enumerate(active_models):
        for m2 in active_models[i + 1:]:
            agreement[f"{m1}_vs_{m2}"] = {"matched": 0, "total_m1": 0, "total_m2": 0}

    vis_count = 0

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Load GT
        gt = load_gt_annotations(ann_dir / f"{img_path.stem}.json")

        # Run each model
        model_results: Dict[str, ModelResult] = {}

        for name in active_models:
            cfg = model_configs[name]
            session = sessions[name]
            input_name = session.get_inputs()[0].name

            t0 = time.perf_counter()
            if cfg["type"] == "yolo":
                tensor, scale, orig_size = preprocess_yolo(img, cfg["input_size"])
                outputs = session.run(None, {input_name: tensor})
                dets = postprocess_yolo12(outputs[0], orig_size, scale, cfg["conf"])
            else:
                tensor, meta = preprocess_ctd(img, cfg["input_size"])
                outputs = session.run(None, {input_name: tensor})
                dets = postprocess_ctd(outputs, meta, cfg["conf"])
            elapsed = (time.perf_counter() - t0) * 1000

            model_results[name] = ModelResult(name=name, detections=dets, inference_ms=elapsed)
            timing[name].append(elapsed)
            det_counts[name].append(len(dets))

            # Compute metrics at each IoU threshold
            for iou_thresh in iou_thresholds:
                tp, fp, fn, ious = match_detections(dets, gt, iou_thresh)
                m = metrics[name][iou_thresh]
                m["tp"] += tp
                m["fp"] += fp
                m["fn"] += fn
                m["ious"].extend(ious)
                m["aps"].append(compute_ap(dets, gt, iou_thresh))

        # Cross-model agreement
        for i, m1 in enumerate(active_models):
            for m2 in active_models[i + 1:]:
                key = f"{m1}_vs_{m2}"
                d1 = model_results[m1].detections
                d2 = model_results[m2].detections
                agreement[key]["total_m1"] += len(d1)
                agreement[key]["total_m2"] += len(d2)
                # Count how many d1 boxes match a d2 box at IoU>=0.5
                matched_set = set()
                for det1 in d1:
                    for j, det2 in enumerate(d2):
                        if j in matched_set:
                            continue
                        if compute_iou(det1.bbox, det2.bbox) >= 0.5:
                            agreement[key]["matched"] += 1
                            matched_set.add(j)
                            break

        # Visualize disagreements
        if args.visualize and vis_count < args.vis_limit:
            vis = create_comparison_grid(img, gt, model_results, img_path.stem)
            vis_path = output_dir / f"compare_{img_path.stem}.jpg"
            cv2.imwrite(str(vis_path), vis)
            vis_count += 1

        # Progress
        if (idx + 1) % 50 == 0 or idx == len(image_paths) - 1:
            print(f"  [{idx + 1}/{len(image_paths)}] processed")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    results_json = {"models": {}, "agreement": {}, "config": {
        "num_images": len(image_paths),
        "iou_thresholds": iou_thresholds,
    }}

    for iou_thresh in iou_thresholds:
        print(f"\n--- IoU Threshold: {iou_thresh} ---")
        print(f"{'Model':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'mAP':>10} "
              f"{'Avg IoU':>10} {'Avg ms':>10} {'Det/img':>10}")
        print("-" * 82)

        for name in active_models:
            m = metrics[name][iou_thresh]
            tp, fp, fn = m["tp"], m["fp"], m["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            mAP = np.mean(m["aps"]) if m["aps"] else 0
            avg_iou = np.mean(m["ious"]) if m["ious"] else 0
            avg_ms = np.mean(timing[name])
            avg_det = np.mean(det_counts[name])

            print(f"{name:<12} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {mAP:>10.3f} "
                  f"{avg_iou:>10.3f} {avg_ms:>10.1f} {avg_det:>10.1f}")

            if name not in results_json["models"]:
                results_json["models"][name] = {"timing_ms": avg_ms, "avg_det_per_img": avg_det}
            results_json["models"][name][f"iou_{iou_thresh}"] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "mAP": round(mAP, 4),
                "avg_iou": round(avg_iou, 4),
                "tp": tp, "fp": fp, "fn": fn,
            }

    # Cross-model agreement
    print(f"\n--- Cross-Model Agreement (IoU>=0.5) ---")
    for key, vals in agreement.items():
        m1_name, m2_name = key.split("_vs_")
        total = vals["total_m1"]
        matched = vals["matched"]
        pct = 100 * matched / total if total > 0 else 0
        print(f"  {key}: {matched}/{total} ({pct:.1f}%) of {m1_name} boxes match {m2_name}")
        results_json["agreement"][key] = {
            "matched": matched, "total_m1": vals["total_m1"],
            "total_m2": vals["total_m2"], "pct": round(pct, 2),
        }

    # Speed comparison
    print(f"\n--- Speed Comparison ---")
    for name in active_models:
        t = timing[name]
        print(f"  {name}: mean={np.mean(t):.1f}ms | p50={np.percentile(t, 50):.1f}ms | "
              f"p95={np.percentile(t, 95):.1f}ms | p99={np.percentile(t, 99):.1f}ms")

    # Accuracy gap audit
    print(f"\n--- Accuracy Gap Audit ---")
    if "yolo12x" in active_models:
        for other in active_models:
            if other == "yolo12x":
                continue
            m_x = metrics["yolo12x"][0.5]
            m_o = metrics[other][0.5]
            fn_diff = m_o["fn"] - m_x["fn"]
            fp_diff = m_o["fp"] - m_x["fp"]
            print(f"  yolo12x vs {other}:")
            print(f"    YOLO12-X catches {fn_diff} more GT boxes (fewer FN)")
            print(f"    YOLO12-X has {fp_diff} {'more' if fp_diff > 0 else 'fewer'} false positives")

    # Save results
    results_path = output_dir / "teacher_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    if args.visualize:
        print(f"Visualizations saved to: {output_dir}/ ({vis_count} images)")


if __name__ == "__main__":
    main()
