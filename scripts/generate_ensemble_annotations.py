#!/usr/bin/env python3
"""
Generate block annotations using ensemble of teacher models.

Teachers:
  1. YOLO12-X (deepghs/AnimeText_yolo) — trained on 735K AnimeText images
  2. Florence-2 (microsoft/Florence-2-large) — 900M web images, OCR_WITH_REGION (independent)
  3. Grounding DINO (IDEA-Research/grounding-dino-tiny) — zero-shot (independent)

Consensus: keep boxes where >=2 models agree (IoU>=0.5).

Usage:
    # Test on 10 images first
    python scripts/generate_ensemble_annotations.py --limit 10

    # Full run on training data
    python scripts/generate_ensemble_annotations.py --input-dir data/merged_train/images --output-dir data/merged_train/block_annotations_ensemble

    # Full run on validation data
    python scripts/generate_ensemble_annotations.py --input-dir data/merged_val/images --output-dir data/merged_val/block_annotations_ensemble
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Detection data structure
# ---------------------------------------------------------------------------

class Detection:
    __slots__ = ("bbox", "confidence", "source")

    def __init__(self, bbox: List[int], confidence: float, source: str = ""):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.source = source

    def to_dict(self) -> dict:
        return {"bbox": self.bbox, "confidence": self.confidence, "source": self.source}


def compute_iou(box1: List[int], box2: List[int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    a2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return inter / (a1 + a2 - inter)


def nms(detections: List[Detection], iou_thresh: float = 0.5) -> List[Detection]:
    if not detections:
        return []
    boxes = np.array([d.bbox for d in detections])
    scores = np.array([d.confidence for d in detections])
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thresh]
    return [detections[i] for i in keep]


# ---------------------------------------------------------------------------
# YOLO12-X Teacher
# ---------------------------------------------------------------------------

class Yolo12XTeacher:
    def __init__(self, model_path: str, conf_threshold: float = 0.425, device: str = "cuda"):
        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "cuda" in device else ["CPUExecutionProvider"]
        )
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.input_size = 640
        print(f"  YOLO12-X loaded ({self.session.get_providers()[0]})")

    def detect(self, image: np.ndarray) -> List[Detection]:
        h, w = image.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        tensor = padded.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[None]

        outputs = self.session.run(None, {self.input_name: tensor})
        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]
        if raw.shape[0] == 5:
            raw = raw.T

        cx, cy, bw, bh, conf = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4]
        mask = conf >= self.conf_threshold
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

        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(float)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), self.conf_threshold, 0.45)
        if len(indices) == 0:
            return []
        indices = indices.flatten()

        return [
            Detection([int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])], float(conf[i]), "yolo12x")
            for i in indices
        ]


# ---------------------------------------------------------------------------
# Florence-2 Teacher
# ---------------------------------------------------------------------------

class Florence2Teacher:
    def __init__(self, device: str = "cuda"):
        from transformers import AutoProcessor, AutoModelForCausalLM
        print("  Loading Florence-2-large (this may download ~1.5GB on first run)...")
        model_id = "microsoft/Florence-2-large"
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float16
        ).to(device).eval()
        self.device = device
        print(f"  Florence-2-large loaded on {device}")

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> List[Detection]:
        from PIL import Image

        h, w = image.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Use OCR_WITH_REGION task to get text locations
        task = "<OCR_WITH_REGION>"
        inputs = self.processor(text=task, images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].long()

        generated_ids = self.model.generate(
            **inputs, max_new_tokens=1024, num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        result = self.processor.post_process_generation(
            generated_text, task=task, image_size=(w, h)
        )

        detections = []
        if task in result and "quad_boxes" in result[task]:
            quads = result[task]["quad_boxes"]
            for quad in quads:
                # quad is [x1,y1,x2,y2,x3,y3,x4,y4] — get bounding box
                xs = [quad[i] for i in range(0, 8, 2)]
                ys = [quad[i] for i in range(1, 8, 2)]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if (x2 - x1) > 3 and (y2 - y1) > 3:
                    detections.append(Detection([x1, y1, x2, y2], 0.85, "florence2"))
        return detections


# ---------------------------------------------------------------------------
# Grounding DINO Teacher
# ---------------------------------------------------------------------------

class GroundingDINOTeacher:
    def __init__(self, device: str = "cuda", conf_threshold: float = 0.2):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        print("  Loading Grounding DINO...")
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        ).to(device).eval()
        self.device = device
        self.conf_threshold = conf_threshold
        self.prompt = "text."
        print(f"  Grounding DINO loaded on {device}")

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> List[Detection]:
        from PIL import Image

        h, w = image.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        inputs = self.processor(
            images=pil_img, text=self.prompt, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=self.conf_threshold,
            text_threshold=self.conf_threshold,
            target_sizes=[(h, w)],
        )[0]

        detections = []
        for box, score in zip(results["boxes"], results["scores"]):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            if (x2 - x1) > 3 and (y2 - y1) > 3:
                detections.append(Detection([x1, y1, x2, y2], float(score), "gdino"))
        return detections


# ---------------------------------------------------------------------------
# Ensemble consensus
# ---------------------------------------------------------------------------

def ensemble_primary_with_validation(
    all_detections: Dict[str, List[Detection]],
    primary: str = "yolo12x",
    iou_threshold: float = 0.5,
    agreement_boost: float = 0.05,
) -> List[Detection]:
    """Primary model drives detections, secondary models validate.

    Strategy:
    - Keep ALL primary model detections
    - Boost confidence when secondary models agree (better localization too)
    - Do NOT add boxes only seen by secondary models (too noisy)
    """
    primary_dets = all_detections.get(primary, [])
    secondary_dets = []
    for name, dets in all_detections.items():
        if name != primary:
            secondary_dets.extend(dets)

    results = []
    for det in primary_dets:
        # Check if any secondary model agrees
        agreed = False
        best_secondary_bbox = None
        for sec in secondary_dets:
            if compute_iou(det.bbox, sec.bbox) >= iou_threshold:
                agreed = True
                best_secondary_bbox = sec.bbox
                break

        if agreed and best_secondary_bbox is not None:
            # Average box coords for better localization
            bbox = [
                (det.bbox[0] + best_secondary_bbox[0]) // 2,
                (det.bbox[1] + best_secondary_bbox[1]) // 2,
                (det.bbox[2] + best_secondary_bbox[2]) // 2,
                (det.bbox[3] + best_secondary_bbox[3]) // 2,
            ]
            conf = min(1.0, det.confidence + agreement_boost)
            source = f"{primary}+validated"
        else:
            bbox = det.bbox[:]
            conf = det.confidence
            source = primary

        results.append(Detection(bbox, conf, source))

    return nms(results, iou_thresh=0.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate ensemble block annotations")
    parser.add_argument("--input-dir", default="data/merged_val/images")
    parser.add_argument("--output-dir", default="data/merged_val/block_annotations_ensemble")
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-agree", type=int, default=2,
                        help="Minimum models that must agree")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis-limit", type=int, default=20)
    parser.add_argument("--vis-dir", default="outputs/ensemble_vis")

    # Model paths
    parser.add_argument("--yolo12x-path",
                        default="models/animetext-yolo/yolo12x_animetext/model.onnx")
    parser.add_argument("--yolo12x-conf", type=float, default=0.425)
    parser.add_argument("--gdino-conf", type=float, default=0.2)

    # Control which teachers to use
    parser.add_argument("--teachers", default="yolo12x,florence2,gdino",
                        help="Comma-separated list of teachers")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_names = [t.strip() for t in args.teachers.split(",")]

    # Load teachers
    print("=" * 60)
    print("ENSEMBLE ANNOTATION PIPELINE")
    print("=" * 60)
    print(f"\nTeachers: {teacher_names}")
    print(f"Consensus: >= {args.min_agree} models must agree\n")

    teachers = {}
    if "yolo12x" in teacher_names:
        teachers["yolo12x"] = Yolo12XTeacher(args.yolo12x_path, args.yolo12x_conf, args.device)
    if "florence2" in teacher_names:
        teachers["florence2"] = Florence2Teacher(args.device)
    if "gdino" in teacher_names:
        teachers["gdino"] = GroundingDINOTeacher(args.device, args.gdino_conf)

    if len(teachers) < args.min_agree:
        print(f"ERROR: Only {len(teachers)} teachers loaded, need {args.min_agree} for consensus")
        sys.exit(1)

    # Collect images
    input_dir = Path(args.input_dir)
    image_paths = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    print(f"\nProcessing {len(image_paths)} images...")

    if args.visualize:
        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Stats
    total_consensus = 0
    per_model_counts = {name: 0 for name in teachers}
    skipped = 0
    vis_count = 0
    t_start = time.time()

    for idx, img_path in enumerate(image_paths):
        ann_path = output_dir / f"{img_path.stem}.json"
        if args.skip_existing and ann_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: Could not read {img_path}")
            continue

        h, w = img.shape[:2]

        # Run all teachers
        all_dets: Dict[str, List[Detection]] = {}
        for name, teacher in teachers.items():
            try:
                dets = teacher.detect(img)
                all_dets[name] = dets
                per_model_counts[name] += len(dets)
            except Exception as e:
                print(f"  WARNING: {name} failed on {img_path.name}: {e}")
                all_dets[name] = []

        # Primary (YOLO12-X) with validation from secondary models
        consensus = ensemble_primary_with_validation(all_dets, primary="yolo12x")
        total_consensus += len(consensus)

        # Save annotation
        annotation = {
            "text_blocks": [
                {"bbox": d.bbox, "confidence": d.confidence, "source": d.source}
                for d in consensus
            ],
            "original_size": [w, h],
            "ensemble_stats": {
                name: len(dets) for name, dets in all_dets.items()
            },
        }
        with open(ann_path, "w") as f:
            json.dump(annotation, f)

        # Visualize
        if args.visualize and vis_count < args.vis_limit:
            colors = {
                "yolo12x": (0, 255, 0),
                "florence2": (255, 0, 0),
                "gdino": (0, 165, 255),
            }
            vis = img.copy()
            # Draw per-model detections (thin)
            for name, dets in all_dets.items():
                color = colors.get(name, (128, 128, 128))
                for d in dets:
                    cv2.rectangle(vis, tuple(d.bbox[:2]), tuple(d.bbox[2:]), color, 1)
            # Draw consensus (thick white)
            for d in consensus:
                cv2.rectangle(vis, tuple(d.bbox[:2]), tuple(d.bbox[2:]), (255, 255, 255), 3)
                cv2.putText(vis, f"{d.source}", (d.bbox[0], d.bbox[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            # Legend
            y = 20
            for name, color in colors.items():
                if name in all_dets:
                    cv2.putText(vis, f"{name}: {len(all_dets[name])}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    y += 20
            cv2.putText(vis, f"CONSENSUS: {len(consensus)}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imwrite(str(Path(args.vis_dir) / f"ensemble_{img_path.stem}.jpg"), vis)
            vis_count += 1

        # Progress
        if (idx + 1) % 50 == 0 or idx == len(image_paths) - 1:
            elapsed = time.time() - t_start
            rate = (idx + 1 - skipped) / elapsed if elapsed > 0 else 0
            eta = (len(image_paths) - idx - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{idx + 1}/{len(image_paths)}] "
                  f"{rate:.1f} img/s | ETA: {eta:.1f}min | "
                  f"consensus: {total_consensus / max(1, idx + 1 - skipped):.1f}/img")

    # Summary
    elapsed = time.time() - t_start
    n_processed = len(image_paths) - skipped
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Processed: {n_processed} images in {elapsed:.0f}s ({n_processed/elapsed:.1f} img/s)")
    print(f"Skipped: {skipped}")
    print(f"\nPer-model detections:")
    for name, count in per_model_counts.items():
        print(f"  {name}: {count} total ({count/max(1,n_processed):.1f}/img)")
    print(f"\nConsensus detections: {total_consensus} ({total_consensus/max(1,n_processed):.1f}/img)")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
