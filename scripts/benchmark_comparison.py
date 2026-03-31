#!/usr/bin/env python3
"""
Benchmark Comparison Script: comic-text-detector vs AnimeText_yolo

Compares two text detection models:
1. comic-text-detector (mayocream/comic-text-detector) - YOLOv5 + DBNet + UNet hybrid
2. AnimeText_yolo (deepghs/AnimeText_yolo) - YOLO12 based

Usage:
    python scripts/benchmark_comparison.py --warmup 50 --iterations 200

    # With specific image
    python scripts/benchmark_comparison.py --image test.jpg --visualize
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check available libraries
TORCH_AVAILABLE = False
ONNX_AVAILABLE = False
CUDA_AVAILABLE = False
ULTRALYTICS_AVAILABLE = False
SAFETENSORS_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    pass

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    pass

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class LatencyStats:
    """Container for latency statistics."""
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_latencies(cls, latencies: List[float]) -> "LatencyStats":
        arr = np.array(latencies)
        return cls(
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
        )


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    backend: str
    input_size: Tuple[int, int]
    warmup_iterations: int
    benchmark_iterations: int

    latency: LatencyStats
    throughput_fps: float
    nms_time_ms: float  # Time spent on NMS (0 for NMS-free models)

    peak_memory_mb: float
    model_size_mb: float

    num_detections: int  # Average detections per image

    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class AnimeTextYoloDetector:
    """Wrapper for AnimeText_yolo YOLO12 model."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        input_size: int = 640,
        conf_threshold: float = 0.272,  # From threshold.json
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.model = None
        self.use_onnx = False

    def load(self):
        """Load the model."""
        if self.model_path.suffix == ".onnx":
            self._load_onnx()
        elif self.model_path.suffix == ".pt":
            self._load_pytorch()
        else:
            raise ValueError(f"Unsupported model format: {self.model_path.suffix}")

    def _load_onnx(self):
        """Load ONNX model."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available")

        providers = []
        if self.device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options,
            providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.use_onnx = True
        print(f"  Loaded ONNX model with provider: {self.session.get_providers()[0]}")

    def _load_pytorch(self):
        """Load PyTorch model using Ultralytics."""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not available. Install with: pip install ultralytics")

        self.model = YOLO(str(self.model_path))
        self.use_onnx = False
        print(f"  Loaded YOLO12 PyTorch model")

    def create_dummy_input(self, batch_size: int = 1) -> np.ndarray:
        """Create dummy input for benchmarking."""
        return np.random.rand(batch_size, 3, self.input_size, self.input_size).astype(np.float32)

    def infer(self, inputs: np.ndarray) -> Tuple[Any, float]:
        """
        Run inference.

        Returns:
            Tuple of (outputs, nms_time_ms)

        Note: YOLO12 uses NMS, so we measure it separately.
        """
        if self.use_onnx:
            return self._infer_onnx(inputs)
        else:
            return self._infer_pytorch(inputs)

    def _infer_onnx(self, inputs: np.ndarray) -> Tuple[Any, float]:
        """ONNX inference with NMS timing."""
        # ONNX model typically has NMS built-in
        outputs = self.session.run(None, {self.input_name: inputs})
        # NMS is part of the model, so we can't measure it separately
        return outputs, 0.0

    def _infer_pytorch(self, inputs: np.ndarray) -> Tuple[Any, float]:
        """PyTorch inference with NMS timing."""
        # Convert to PIL or tensor
        import torch

        # Ultralytics expects [H, W, C] images or file paths
        # For benchmarking with raw arrays, we need to handle this
        batch_size = inputs.shape[0]

        # Create synthetic image tensor
        tensor = torch.from_numpy(inputs).to(self.device)

        # Run inference (NMS is handled internally by YOLO)
        # The predict method includes NMS
        results = self.model.predict(
            source=tensor,
            conf=self.conf_threshold,
            verbose=False,
        )

        # NMS is included in predict, can't measure separately
        return results, 0.0

    def cleanup(self):
        """Cleanup resources."""
        self.model = None
        self.session = None
        gc.collect()
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()


class ComicTextDetector:
    """Wrapper for comic-text-detector (mayocream) hybrid model."""

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        input_size: int = 1024,
        conf_threshold: float = 0.4,
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.yolo = None
        self.dbnet = None
        self.unet = None

    def load(self):
        """Load all three model components."""
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors not available. Install with: pip install safetensors")
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        print("  Loading comic-text-detector components...")

        # Load safetensor files
        yolo_path = self.model_dir / "yolo-v5.safetensors"
        dbnet_path = self.model_dir / "dbnet.safetensors"
        unet_path = self.model_dir / "unet.safetensors"

        self.yolo_weights = load_file(str(yolo_path))
        self.dbnet_weights = load_file(str(dbnet_path))
        self.unet_weights = load_file(str(unet_path))

        print(f"    YOLOv5 weights: {len(self.yolo_weights)} tensors")
        print(f"    DBNet weights: {len(self.dbnet_weights)} tensors")
        print(f"    UNet weights: {len(self.unet_weights)} tensors")

        # For benchmarking, we'll use ONNX if available
        # Check if there's a combined ONNX model
        onnx_path = self.model_dir.parent / "detector.onnx"
        if onnx_path.exists() and ONNX_AVAILABLE:
            print(f"  Loading combined ONNX model: {onnx_path}")
            self._load_onnx(str(onnx_path))
        else:
            # Use the locally trained model instead
            ctd_onnx = self.model_dir.parent / "ctd.onnx"
            if ctd_onnx.exists():
                print(f"  Loading local CTD ONNX model: {ctd_onnx}")
                self._load_onnx(str(ctd_onnx))
            else:
                print("  Warning: No ONNX model found, using weights only (limited functionality)")

    def _load_onnx(self, onnx_path: str):
        """Load ONNX model for inference."""
        providers = []
        if self.device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.use_onnx = True

        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        if isinstance(input_shape[2], int):
            self.input_size = input_shape[2]

        print(f"    ONNX loaded with provider: {self.session.get_providers()[0]}")
        print(f"    Input shape: {input_shape}")
        print(f"    Outputs: {[o.name for o in self.session.get_outputs()]}")

    def create_dummy_input(self, batch_size: int = 1) -> np.ndarray:
        """Create dummy input for benchmarking."""
        return np.random.rand(batch_size, 3, self.input_size, self.input_size).astype(np.float32)

    def infer(self, inputs: np.ndarray) -> Tuple[Any, float]:
        """
        Run inference.

        Returns:
            Tuple of (outputs, nms_time_ms)
        """
        if hasattr(self, 'use_onnx') and self.use_onnx:
            # ONNX inference
            start_nms = time.perf_counter()
            outputs = self.session.run(None, {self.input_name: inputs})
            # NMS is built into the model, so this is 0
            nms_time = 0.0
            return outputs, nms_time
        else:
            # No ONNX model available
            raise RuntimeError("No ONNX model loaded for comic-text-detector")

    def cleanup(self):
        """Cleanup resources."""
        self.yolo_weights = None
        self.dbnet_weights = None
        self.unet_weights = None
        self.session = None
        gc.collect()
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()


def run_benchmark(
    detector,
    model_name: str,
    warmup_iterations: int = 50,
    benchmark_iterations: int = 200,
    batch_size: int = 1,
) -> BenchmarkResult:
    """Run benchmark on a detector."""

    device = detector.device if hasattr(detector, 'device') else "cuda"
    input_size = detector.input_size if hasattr(detector, 'input_size') else 1024

    # Reset memory stats
    if CUDA_AVAILABLE:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Create dummy input
    dummy_input = detector.create_dummy_input(batch_size)

    # Warmup
    print(f"  Warming up ({warmup_iterations} iterations)...")
    for _ in range(warmup_iterations):
        _ = detector.infer(dummy_input)

    if CUDA_AVAILABLE:
        torch.cuda.synchronize()

    # Benchmark
    print(f"  Benchmarking ({benchmark_iterations} iterations)...")
    latencies = []
    nms_times = []
    total_detections = 0

    for i in range(benchmark_iterations):
        if CUDA_AVAILABLE:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            outputs, nms_time = detector.infer(dummy_input)
            end.record()

            torch.cuda.synchronize()
            latency_ms = start.elapsed_time(end)
        else:
            start = time.perf_counter()
            outputs, nms_time = detector.infer(dummy_input)
            latency_ms = (time.perf_counter() - start) * 1000

        latencies.append(latency_ms)
        nms_times.append(nms_time)

        # Count detections (model-specific)
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            # ONNX outputs
            first_out = outputs[0]
            if hasattr(first_out, 'shape'):
                if len(first_out.shape) >= 2:
                    total_detections += first_out.shape[0] if first_out.shape[0] < 1000 else 0

    # Calculate stats
    latency_stats = LatencyStats.from_latencies(latencies)
    throughput_fps = (batch_size * 1000) / latency_stats.mean_ms if latency_stats.mean_ms > 0 else 0

    # Memory stats
    peak_memory_mb = 0.0
    if CUDA_AVAILABLE:
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Model size
    model_size_mb = 0.0
    if hasattr(detector, 'model_path'):
        model_path = Path(detector.model_path)
        if model_path.exists():
            if model_path.is_file():
                model_size_mb = model_path.stat().st_size / (1024 * 1024)
            elif model_path.is_dir():
                model_size_mb = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / (1024 * 1024)
    elif hasattr(detector, 'model_dir'):
        model_dir = Path(detector.model_dir)
        if model_dir.exists():
            model_size_mb = sum(f.stat().st_size for f in model_dir.rglob("*.safetensors")) / (1024 * 1024)

    return BenchmarkResult(
        model_name=model_name,
        backend="ONNX" if hasattr(detector, 'use_onnx') and detector.use_onnx else "PyTorch",
        input_size=(input_size, input_size),
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
        latency=latency_stats,
        throughput_fps=throughput_fps,
        nms_time_ms=np.mean(nms_times) if nms_times else 0.0,
        peak_memory_mb=peak_memory_mb,
        model_size_mb=model_size_mb,
        num_detections=total_detections // benchmark_iterations if benchmark_iterations > 0 else 0,
    )


def print_comparison_table(results: List[BenchmarkResult]):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("BENCHMARK COMPARISON RESULTS")
    print("=" * 100)

    # Header
    print(f"\n{'Model':<30} {'Backend':<10} {'Size':<12} {'Mean (ms)':<12} {'P95 (ms)':<12} {'FPS':<10} {'Memory (MB)':<12}")
    print("-" * 100)

    for r in results:
        size_str = f"{r.input_size[0]}x{r.input_size[1]}"
        print(f"{r.model_name:<30} {r.backend:<10} {size_str:<12} {r.latency.mean_ms:<12.2f} {r.latency.p95_ms:<12.2f} {r.throughput_fps:<10.1f} {r.peak_memory_mb:<12.0f}")

    print("-" * 100)

    # Detailed comparison
    if len(results) >= 2:
        baseline = results[0]
        print(f"\nDetailed Comparison (baseline: {baseline.model_name}):")
        print("-" * 60)

        for r in results[1:]:
            speedup = baseline.latency.mean_ms / r.latency.mean_ms if r.latency.mean_ms > 0 else 0
            latency_diff = r.latency.mean_ms - baseline.latency.mean_ms

            print(f"\n{r.model_name} vs {baseline.model_name}:")
            print(f"  Latency difference: {latency_diff:+.2f} ms ({speedup:.2f}x speedup)")
            print(f"  FPS difference: {r.throughput_fps - baseline.throughput_fps:+.1f} FPS")

            if r.nms_time_ms > 0 or baseline.nms_time_ms > 0:
                print(f"  NMS time: {r.model_name}={r.nms_time_ms:.2f}ms, {baseline.model_name}={baseline.nms_time_ms:.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark comparison: comic-text-detector vs AnimeText_yolo")

    parser.add_argument("--ctd-model", type=str,
                        default="models/comic-text-detector-hf",
                        help="Path to comic-text-detector model directory")
    parser.add_argument("--animetext-model", type=str,
                        default="models/animetext-yolo/yolo12s_animetext/model.onnx",
                        help="Path to AnimeText_yolo model")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Benchmark iterations")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--image", type=str, default=None,
                        help="Optional: test image for visualization")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualization of results")

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not CUDA_AVAILABLE:
        print("Warning: CUDA not available, using CPU")
        args.device = "cpu"

    print("=" * 70)
    print("Comic Text Detector Benchmark Comparison")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Warmup: {args.warmup} iterations")
    print(f"Benchmark: {args.iterations} iterations")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)

    results = []

    # Benchmark AnimeText_yolo (YOLO12)
    print("\n[1/2] Loading AnimeText_yolo (YOLO12)...")
    try:
        animetext_path = Path(args.animetext_model)
        if not animetext_path.exists():
            # Try relative to script
            animetext_path = Path(__file__).parent.parent / args.animetext_model

        if animetext_path.exists():
            animetext_detector = AnimeTextYoloDetector(
                model_path=str(animetext_path),
                device=args.device,
                input_size=640,  # YOLO12 default
            )
            animetext_detector.load()

            animetext_result = run_benchmark(
                animetext_detector,
                model_name="AnimeText_yolo (YOLO12s)",
                warmup_iterations=args.warmup,
                benchmark_iterations=args.iterations,
                batch_size=args.batch_size,
            )
            results.append(animetext_result)
            animetext_detector.cleanup()
        else:
            print(f"  Error: Model not found at {animetext_path}")
    except Exception as e:
        print(f"  Error loading AnimeText_yolo: {e}")
        import traceback
        traceback.print_exc()

    # Benchmark comic-text-detector
    print("\n[2/2] Loading comic-text-detector (YOLOv5+DBNet+UNet)...")
    try:
        ctd_path = Path(args.ctd_model)
        if not ctd_path.exists():
            ctd_path = Path(__file__).parent.parent / args.ctd_model

        if ctd_path.exists():
            ctd_detector = ComicTextDetector(
                model_dir=str(ctd_path),
                device=args.device,
                input_size=1024,  # CTD default
            )
            ctd_detector.load()

            ctd_result = run_benchmark(
                ctd_detector,
                model_name="comic-text-detector (Hybrid)",
                warmup_iterations=args.warmup,
                benchmark_iterations=args.iterations,
                batch_size=args.batch_size,
            )
            results.append(ctd_result)
            ctd_detector.cleanup()
        else:
            print(f"  Error: Model not found at {ctd_path}")
    except Exception as e:
        print(f"  Error loading comic-text-detector: {e}")
        import traceback
        traceback.print_exc()

    # Print comparison
    if results:
        print_comparison_table(results)
    else:
        print("\nNo benchmark results collected!")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
Key Findings:
- AnimeText_yolo uses YOLO12 with NMS (not NMS-free)
- comic-text-detector uses YOLOv5 hybrid with NMS
- Both models include NMS in their pipeline

For lowest latency, consider:
- YOLOv10-based model (true NMS-free)
- TensorRT optimization
- FP16 quantization
""")


if __name__ == "__main__":
    main()
