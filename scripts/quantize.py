#!/usr/bin/env python3
"""
Model Quantization Script

Supports:
- FP16: 1.5-2x speedup, <0.1% accuracy loss
- INT8 Dynamic: 2-3x speedup, 1-2% accuracy loss
- INT8 Static: 3-4x speedup, 0.5-1% accuracy loss (requires calibration)

Usage:
    python scripts/quantize.py --input models/ctd_v10.onnx --output models/ctd_v10_fp16.onnx --precision fp16
    python scripts/quantize.py --input models/ctd_v10.onnx --output models/ctd_v10_int8.onnx --precision int8-dynamic
    python scripts/quantize.py --input models/ctd_v10.onnx --output models/ctd_v10_int8.onnx --precision int8-static --calibration-dir ./images
"""

import argparse
from pathlib import Path
import numpy as np


def quantize_fp16(input_model: str, output_model: str):
    """Convert model to FP16 precision."""
    import onnx
    from onnxconverter_common import float16

    print(f"Converting to FP16: {input_model}")

    model = onnx.load(input_model)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, output_model)

    orig_size = Path(input_model).stat().st_size / (1024 * 1024)
    new_size = Path(output_model).stat().st_size / (1024 * 1024)
    print(f"Size: {orig_size:.1f} MB -> {new_size:.1f} MB ({100*(orig_size-new_size)/orig_size:.1f}% reduction)")
    print(f"Saved to: {output_model}")


def quantize_int8_dynamic(input_model: str, output_model: str):
    """Dynamic INT8 quantization (no calibration needed)."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"Quantizing to INT8 (dynamic): {input_model}")

    quantize_dynamic(
        input_model,
        output_model,
        weight_type=QuantType.QInt8,
    )

    orig_size = Path(input_model).stat().st_size / (1024 * 1024)
    new_size = Path(output_model).stat().st_size / (1024 * 1024)
    print(f"Size: {orig_size:.1f} MB -> {new_size:.1f} MB ({100*(orig_size-new_size)/orig_size:.1f}% reduction)")
    print(f"Saved to: {output_model}")


def quantize_int8_static(input_model: str, output_model: str, calibration_dir: str, num_samples: int = 100):
    """Static INT8 quantization with calibration data."""
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
    import cv2

    class ImageCalibrationReader(CalibrationDataReader):
        def __init__(self, calibration_dir: str, input_size: int = 1024, num_samples: int = 100):
            self.input_size = input_size
            self.image_paths = []

            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                self.image_paths.extend(Path(calibration_dir).glob(ext))

            self.image_paths = self.image_paths[:num_samples]
            self.index = 0
            print(f"Found {len(self.image_paths)} calibration images")

        def get_next(self):
            if self.index >= len(self.image_paths):
                return None

            img_path = self.image_paths[self.index]
            self.index += 1

            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_size, self.input_size))
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)[np.newaxis, ...]

            return {"images": img}

    print(f"Quantizing to INT8 (static): {input_model}")
    print(f"Using calibration data from: {calibration_dir}")

    calibration_reader = ImageCalibrationReader(calibration_dir, num_samples=num_samples)

    quantize_static(
        input_model,
        output_model,
        calibration_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
    )

    orig_size = Path(input_model).stat().st_size / (1024 * 1024)
    new_size = Path(output_model).stat().st_size / (1024 * 1024)
    print(f"Size: {orig_size:.1f} MB -> {new_size:.1f} MB ({100*(orig_size-new_size)/orig_size:.1f}% reduction)")
    print(f"Saved to: {output_model}")


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model")
    parser.add_argument("--input", "-i", required=True, help="Input ONNX model")
    parser.add_argument("--output", "-o", required=True, help="Output ONNX model")
    parser.add_argument("--precision", "-p", choices=["fp16", "int8-dynamic", "int8-static"],
                        default="fp16", help="Target precision")
    parser.add_argument("--calibration-dir", type=str, default=None,
                        help="Directory with calibration images (for int8-static)")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of calibration samples")

    args = parser.parse_args()

    if args.precision == "fp16":
        quantize_fp16(args.input, args.output)
    elif args.precision == "int8-dynamic":
        quantize_int8_dynamic(args.input, args.output)
    elif args.precision == "int8-static":
        if args.calibration_dir is None:
            raise ValueError("--calibration-dir required for int8-static quantization")
        quantize_int8_static(args.input, args.output, args.calibration_dir, args.num_samples)


if __name__ == "__main__":
    main()
