#!/usr/bin/env python3
"""
Model Export Script

Export trained models to various formats:
- ONNX (CPU/GPU optimized)
- TensorRT (NVIDIA GPU)
- OpenVINO (Intel/AMD)
- CoreML (Apple)

Usage:
    # Export to ONNX
    python scripts/export.py --checkpoint runs/train/best.pt --format onnx --output models/detector.onnx

    # Export to ONNX with FP16
    python scripts/export.py --checkpoint runs/train/best.pt --format onnx --half --output models/detector_fp16.onnx

    # Export with dynamic batch size
    python scripts/export.py --checkpoint runs/train/best.pt --format onnx --dynamic --output models/detector_dynamic.onnx

    # Export YOLOv10 model with 3 outputs (NMS-free)
    python scripts/export.py --checkpoint runs/block_v2/best.pt --seg-checkpoint runs/segmentation_v2/best.pt --format onnx --nms-free --output models/ctd_v10.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

try:
    import onnx
    from onnxsim import simplify
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to various formats")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (detection or segmentation)")
    parser.add_argument("--seg-checkpoint", type=str, default=None,
                        help="Path to segmentation checkpoint (required for detection models to get full UnetHead)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file path")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "tensorrt", "openvino", "coreml"],
                        help="Export format")
    parser.add_argument("--input-size", type=int, default=1024,
                        help="Model input size")
    parser.add_argument("--opset", type=int, default=18,
                        help="ONNX opset version (default: 18)")
    parser.add_argument("--half", action="store_true",
                        help="Export with FP16 precision")
    parser.add_argument("--dynamic", action="store_true",
                        help="Enable dynamic batch size")
    parser.add_argument("--simplify", action="store_true", default=True,
                        help="Simplify ONNX model")
    parser.add_argument("--no-simplify", dest="simplify", action="store_false",
                        help="Do not simplify ONNX model")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for export")
    parser.add_argument("--nms-free", action="store_true",
                        help="Export YOLOv10 model with 3 outputs (NMS-free block detection)")

    return parser.parse_args()


class ExportableModel(nn.Module):
    """Wrapper to make model ONNX-exportable with baked-in normalization."""

    def __init__(self, backbone, seg_head, db_head):
        super().__init__()
        self.backbone = backbone
        self.seg_head = seg_head
        self.db_head = db_head
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        features = self.backbone(x)
        mask, seg_features = self.seg_head(*features, forward_mode=2)
        lines = self.db_head(*seg_features, step_eval=False)
        return mask, lines


class ExportableModelV10(nn.Module):
    """Wrapper for YOLOv10-based model export with 3 outputs (NMS-free).

    Outputs:
        - blocks: Block detection predictions [batch, num_detections, 6] (x, y, w, h, conf, class)
        - mask: Segmentation mask [batch, 1, H, W]
        - lines: Text line probability and threshold maps [batch, 2, H, W]
    """

    def __init__(self, backbone, seg_head, db_head, block_detector=None):
        super().__init__()
        self.backbone = backbone
        self.seg_head = seg_head
        self.db_head = db_head
        self.block_detector = block_detector
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        features = self.backbone(x)
        mask, seg_features = self.seg_head(*features, forward_mode=2)
        lines = self.db_head(*seg_features, step_eval=False)
        if self.block_detector is not None:
            blocks = self.block_detector([features[1], features[2], features[3]])
        else:
            blocks = torch.zeros(x.shape[0], 0, 6, device=x.device)
        return blocks, mask, lines


def _normalize_state_dict(state_dict: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _extract_sub_state_dict(state_dict: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _is_detection_checkpoint(state_dict: dict) -> bool:
    has_dbnet = any("dbnet" in k for k in state_dict.keys())
    has_upconv3 = any("seg_net.upconv3" in k for k in state_dict.keys())
    return has_dbnet and not has_upconv3


def load_models_for_export(checkpoint_path: str, seg_checkpoint_path: str = None, device: str = "cpu"):
    from src.models.backbone import create_backbone
    from src.models.heads import UnetHead, DBHead

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = _normalize_state_dict(checkpoint["model_state_dict"])
    elif "weights" in checkpoint:
        weights = checkpoint["weights"]
        state_dict = {}
        for module_name, module_dict in weights.items():
            for k, v in module_dict.items():
                state_dict[f"{module_name}.{k}"] = v
    else:
        state_dict = _normalize_state_dict(checkpoint)

    is_detection = _is_detection_checkpoint(state_dict)

    if is_detection:
        print("  Detected detection checkpoint (missing seg_net upconv3-6)")
        if seg_checkpoint_path is None:
            raise ValueError("Detection checkpoint requires --seg-checkpoint")
        seg_checkpoint = torch.load(seg_checkpoint_path, map_location=device)
        if "model_state_dict" in seg_checkpoint:
            seg_state_dict = _normalize_state_dict(seg_checkpoint["model_state_dict"])
        else:
            seg_state_dict = _normalize_state_dict(seg_checkpoint)
        backbone_state = _extract_sub_state_dict(state_dict, "backbone.")
        seg_net_state = _extract_sub_state_dict(seg_state_dict, "seg_net.")
        db_state = _extract_sub_state_dict(state_dict, "dbnet.")
    else:
        print("  Detected segmentation checkpoint (full model)")
        seg_net_state = _extract_sub_state_dict(state_dict, "seg_net.")
        backbone_state = _extract_sub_state_dict(state_dict, "backbone.")
        db_state = _extract_sub_state_dict(state_dict, "dbnet.") if any("dbnet" in k for k in state_dict) else None

    backbone = create_backbone(model_name="yolo11s.pt", pretrained=False, freeze=True)
    seg_head = UnetHead()
    if backbone_state:
        backbone.load_state_dict(backbone_state)
    if seg_net_state:
        seg_head.load_state_dict(seg_net_state)
    db_head = DBHead(64)
    if db_state:
        db_head.load_state_dict(db_state)

    backbone = backbone.to(device).eval()
    seg_head = seg_head.to(device).eval()
    db_head = db_head.to(device).eval()

    return backbone, seg_head, db_head


def load_models_for_export_v10(checkpoint_path: str, seg_checkpoint_path: str = None, device: str = "cpu"):
    from src.models.backbone import create_backbone
    from src.models.heads import UnetHead, DBHead

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get backbone name from checkpoint config (YOLO26 vs YOLO11 etc)
    config = checkpoint.get("config", {})
    backbone_name = config.get("backbone_name", "yolo11s.pt")
    print(f"  Backbone from config: {backbone_name}")

    if "model_state_dict" in checkpoint:
        state_dict = _normalize_state_dict(checkpoint["model_state_dict"])
    elif "weights" in checkpoint:
        weights = checkpoint["weights"]
        state_dict = {}
        for module_name, module_dict in weights.items():
            for k, v in module_dict.items():
                state_dict[f"{module_name}.{k}"] = v
    else:
        state_dict = _normalize_state_dict(checkpoint)

    # Check for block detector with either naming convention (block_detector or block_det)
    has_block_detector = any("block_detector" in k or "block_det" in k for k in state_dict.keys())
    is_detection = _is_detection_checkpoint(state_dict)

    if is_detection or has_block_detector:
        print("  Detected detection/block checkpoint")
        if seg_checkpoint_path is None:
            raise ValueError("Detection/block checkpoint requires --seg-checkpoint")
        seg_checkpoint = torch.load(seg_checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in seg_checkpoint:
            seg_state_dict = _normalize_state_dict(seg_checkpoint["model_state_dict"])
        else:
            seg_state_dict = _normalize_state_dict(seg_checkpoint)
        backbone_state = _extract_sub_state_dict(state_dict, "backbone.")
        seg_net_state = _extract_sub_state_dict(seg_state_dict, "seg_net.")
        db_state = _extract_sub_state_dict(state_dict, "dbnet.")
        # Try both naming conventions for block detector
        block_state = _extract_sub_state_dict(state_dict, "block_detector.")
        if not block_state:
            block_state = _extract_sub_state_dict(state_dict, "block_det.")
    else:
        print("  Detected segmentation checkpoint (full model)")
        seg_net_state = _extract_sub_state_dict(state_dict, "seg_net.")
        backbone_state = _extract_sub_state_dict(state_dict, "backbone.")
        db_state = _extract_sub_state_dict(state_dict, "dbnet.") if any("dbnet" in k for k in state_dict) else None
        # Also check for block_det in full model checkpoints
        block_state = _extract_sub_state_dict(state_dict, "block_det.") if any("block_det" in k for k in state_dict) else None

    backbone = create_backbone(model_name=backbone_name, pretrained=False, freeze=True)
    seg_head = UnetHead()
    if backbone_state:
        backbone.load_state_dict(backbone_state)
    if seg_net_state:
        seg_head.load_state_dict(seg_net_state)
    db_head = DBHead(64)
    if db_state:
        db_head.load_state_dict(db_state)

    block_detector = None
    if block_state:
        try:
            from src.models.heads import BlockDetector
            # Channel config: P3=128, P4=256, P5=512 (matches YOLO26 backbone)
            block_detector = BlockDetector(nc=1, ch=(128, 256, 512))
            block_detector.load_state_dict(block_state)
            block_detector.training_mode = False  # Use o2o head (NMS-free)
            print("  Loaded BlockDetector head (NMS-free mode)")
        except Exception as e:
            print(f"  Warning: Failed to load BlockDetector: {e}")

    backbone = backbone.to(device).eval()
    seg_head = seg_head.to(device).eval()
    db_head = db_head.to(device).eval()
    if block_detector is not None:
        block_detector = block_detector.to(device).eval()

    return backbone, seg_head, db_head, block_detector


def validate_no_nms(onnx_path: str, raise_on_nms: bool = True) -> bool:
    """Validate that no NMS operations exist in the exported ONNX graph.

    Args:
        onnx_path: Path to the ONNX model file
        raise_on_nms: If True, raises ValueError when NMS nodes are found

    Returns:
        True if no NMS nodes found, False otherwise

    Raises:
        ValueError: If NMS nodes are found and raise_on_nms is True
    """
    if not ONNX_AVAILABLE:
        print("  Warning: ONNX not available, skipping NMS validation")
        return True
    model = onnx.load(onnx_path)
    nms_ops = ["NonMaxSuppression", "NMSRotated", "BatchedNMS", "NMSPlugin"]
    found_nms_nodes = []
    for node in model.graph.node:
        if node.op_type in nms_ops:
            found_nms_nodes.append((node.op_type, node.name))

    if found_nms_nodes:
        for op_type, name in found_nms_nodes:
            print(f"  ERROR: Found {op_type} node: {name}")
        if raise_on_nms:
            raise ValueError(f"Found {len(found_nms_nodes)} NMS node(s) in exported ONNX graph. "
                           "YOLOv10 models should be NMS-free. Check model architecture.")
        return False


def export_onnx(model, output_path, input_size=1024, opset=18, half=False, dynamic=False, simplify_model=True, device="cpu"):
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX not available")
    model = model.to(device).eval()
    if half and device != "cpu":
        model = model.half()
    dtype = torch.float16 if half and device != "cpu" else torch.float32
    dummy = torch.randn(1, 3, input_size, input_size, dtype=dtype, device=device)
    dynamic_axes = {"images": {0: "batch"}, "mask": {0: "batch"}, "lines": {0: "batch"}} if dynamic else None
    torch.onnx.export(model, dummy, output_path, opset_version=opset, input_names=["images"],
                      output_names=["mask", "lines"], dynamic_axes=dynamic_axes, do_constant_folding=True)
    print(f"  Exported to: {output_path}")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    if simplify_model:
        try:
            onnx_model, check = simplify(onnx_model, dynamic_input_shape=dynamic,
                                         input_shapes={"images": [1, 3, input_size, input_size]} if not dynamic else None)
            if check:
                onnx.save(onnx_model, output_path)
        except Exception as e:
            print(f"  Warning: Simplification failed: {e}")
    return output_path


def export_onnx_v10(model, output_path, input_size=1024, opset=18, half=False, dynamic=False, simplify_model=True, device="cpu"):
    """Export YOLOv10 model to ONNX with 3 outputs (NMS-free).

    Outputs:
        - blocks: [batch, num_detections, 6] - Block detection predictions
        - mask: [batch, 1, H, W] - Segmentation mask
        - lines: [batch, 2, H, W] - Text line probability and threshold maps

    Args:
        model: The ExportableModelV10 instance
        output_path: Path to save the ONNX model
        input_size: Input image size (default: 1024)
        opset: ONNX opset version (default: 18)
        half: Use FP16 precision (default: False)
        dynamic: Enable dynamic axes (default: False)
        simplify_model: Run onnxsim simplification (default: True)
        device: Device for export (default: "cpu")

    Returns:
        Path to the exported ONNX model

    Raises:
        ValueError: If NMS nodes are found in the exported graph
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX not available")
    model = model.to(device).eval()
    if half and device != "cpu":
        model = model.half()
    dtype = torch.float16 if half and device != "cpu" else torch.float32
    dummy = torch.randn(1, 3, input_size, input_size, dtype=dtype, device=device)

    # Dynamic axes for all 3 outputs
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "blocks": {0: "batch", 1: "num_detections"},
            "mask": {0: "batch"},
            "lines": {0: "batch"},
        }

    print(f"Exporting YOLOv10 model to ONNX (opset {opset})...")
    print(f"  Outputs: blocks, mask, lines (3 outputs)")
    print(f"  Dynamic axes: {dynamic}")

    # Export with 3 output names: blocks, mask, lines
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["images"],
        output_names=["blocks", "mask", "lines"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    print(f"  Exported to: {output_path}")

    # Validate ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Validate no NMS operations - raises ValueError if NMS found
    validate_no_nms(output_path, raise_on_nms=True)

    if simplify_model:
        try:
            onnx_model, check = simplify(onnx_model, dynamic_input_shape=dynamic,
                                         input_shapes={"images": [1, 3, input_size, input_size]} if not dynamic else None)
            if check:
                onnx.save(onnx_model, output_path)
                print("  Simplified ONNX model")
        except Exception as e:
            print(f"  Warning: Simplification failed: {e}")

    return output_path


def export_tensorrt(onnx_path, output_path, input_size=1024, half=True, workspace_size=4):
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError("TensorRT not available")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("Failed to parse ONNX model")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30))
    if half:
        config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    with open(output_path, "wb") as f:
        f.write(engine)
    return output_path


def export_openvino(onnx_path, output_dir, input_size=1024, half=False):
    try:
        from openvino.tools import mo
        from openvino.runtime import serialize
    except ImportError:
        raise ImportError("OpenVINO not available")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = mo.convert_model(onnx_path, input_shape=[1, 3, input_size, input_size], compress_to_fp16=half)
    output_path = output_dir / "model.xml"
    serialize(model, str(output_path))
    return str(output_path)


def main():
    args = parse_args()
    print(f"Loading checkpoint: {args.checkpoint}")
    if args.nms_free:
        backbone, seg_head, db_head, block_detector = load_models_for_export_v10(
            args.checkpoint, args.seg_checkpoint, "cpu")
        model = ExportableModelV10(backbone, seg_head, db_head, block_detector)
    else:
        backbone, seg_head, db_head = load_models_for_export(args.checkpoint, args.seg_checkpoint, "cpu")
        model = ExportableModel(backbone, seg_head, db_head)
    model.eval()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "onnx":
        if args.nms_free:
            export_onnx_v10(model, str(output_path), args.input_size, args.opset, args.half, args.dynamic, args.simplify, args.device)
        else:
            export_onnx(model, str(output_path), args.input_size, args.opset, args.half, args.dynamic, args.simplify, args.device)
    elif args.format == "tensorrt":
        onnx_path = str(output_path.with_suffix(".onnx"))
        if args.nms_free:
            export_onnx_v10(model, onnx_path, args.input_size, args.opset, False, False, args.simplify, "cpu")
        else:
            export_onnx(model, onnx_path, args.input_size, args.opset, False, False, args.simplify, "cpu")
        export_tensorrt(onnx_path, str(output_path), args.input_size, args.half)
    elif args.format == "openvino":
        onnx_path = str(output_path.with_suffix(".onnx"))
        if args.nms_free:
            export_onnx_v10(model, onnx_path, args.input_size, args.opset, False, False, args.simplify, "cpu")
        else:
            export_onnx(model, onnx_path, args.input_size, args.opset, False, False, args.simplify, "cpu")
        export_openvino(onnx_path, str(output_path.parent), args.input_size, args.half)
    elif args.format == "coreml":
        raise NotImplementedError("CoreML export not yet implemented")
    print("Export complete!")


if __name__ == "__main__":
    main()
