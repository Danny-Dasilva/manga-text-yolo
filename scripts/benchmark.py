#!/usr/bin/env python3
"""
Benchmark Script for YOLOv10 Comic Text Detector

Comprehensive benchmarking across multiple backends:
- PyTorch (native)
- ONNX Runtime (CPU/GPU)
- TensorRT
- torch.compile

Metrics measured:
- Latency: mean, p50, p95, p99
- Throughput: images/sec
- Memory: peak GPU memory usage
- First inference: warmup cost

Usage:
    # Benchmark ONNX model with TensorRT
    python scripts/benchmark.py --model models/ctd_v10.onnx --backend tensorrt --batch-sizes 1,2,4,8 --warmup 10 --iterations 100

    # Benchmark PyTorch model with torch.compile
    python scripts/benchmark.py --model runs/block_v2/best.pt --backend torch-compile --batch-sizes 1,2,4 --warmup 5

    # Full benchmark with profiling
    python scripts/benchmark.py --model models/ctd_v10.onnx --backend onnx --profile --output-dir benchmark_results

    # Compare backends
    python scripts/benchmark.py --model models/ctd_v10.onnx --compare-backends --batch-sizes 1,4,8
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Optional imports with availability checks
TORCH_AVAILABLE = False
ONNX_AVAILABLE = False
TENSORRT_AVAILABLE = False
CUDA_AVAILABLE = False

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
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
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
        """Compute statistics from list of latencies (in ms)."""
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
class MemoryStats:
    """Container for memory statistics."""
    peak_allocated_mb: float
    peak_reserved_mb: float
    current_allocated_mb: float

    @classmethod
    def capture(cls) -> "MemoryStats":
        """Capture current CUDA memory statistics."""
        if not CUDA_AVAILABLE:
            return cls(0.0, 0.0, 0.0)

        return cls(
            peak_allocated_mb=torch.cuda.max_memory_allocated() / (1024 * 1024),
            peak_reserved_mb=torch.cuda.max_memory_reserved() / (1024 * 1024),
            current_allocated_mb=torch.cuda.memory_allocated() / (1024 * 1024),
        )

    @classmethod
    def reset(cls):
        """Reset CUDA memory statistics."""
        if CUDA_AVAILABLE:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()


@dataclass
class OutputStats:
    """Container for model output statistics."""
    num_blocks: int = 0
    avg_block_confidence: float = 0.0
    mask_coverage: float = 0.0
    mask_max_value: float = 0.0
    lines_coverage: float = 0.0

    @classmethod
    def from_outputs(cls, outputs: List, conf_thresh: float = 0.25) -> "OutputStats":
        """Compute statistics from V10 model outputs (blocks, mask, lines).

        Args:
            outputs: List of model outputs. Can be:
                - 2 outputs (legacy): [mask, lines]
                - 3 outputs (V10): [blocks, mask, lines]
            conf_thresh: Confidence threshold for block detection

        Returns:
            OutputStats with computed metrics
        """
        num_outputs = len(outputs)

        if num_outputs == 2:
            # Legacy 2-output model: [mask, lines]
            mask_output = outputs[0]
            lines_output = outputs[1]
            blocks_output = None
        elif num_outputs == 3:
            # V10 3-output model: [blocks, mask, lines]
            blocks_output = outputs[0]
            mask_output = outputs[1]
            lines_output = outputs[2]
        else:
            return cls()

        # Convert to numpy if needed
        def to_numpy(x):
            if x is None:
                return None
            if hasattr(x, 'cpu'):
                return x.cpu().numpy()
            return np.array(x)

        blocks_np = to_numpy(blocks_output)
        mask_np = to_numpy(mask_output)
        lines_np = to_numpy(lines_output)

        stats = cls()

        # Block detection metrics (V10 only)
        if blocks_np is not None:
            # blocks shape: [batch, num_predictions, 6] (x, y, w, h, conf, class)
            if blocks_np.ndim == 3:
                # Get confidence scores (index 4)
                confidences = blocks_np[0, :, 4] if blocks_np.shape[0] > 0 else np.array([])
                # Filter by confidence threshold
                high_conf_mask = confidences > conf_thresh
                stats.num_blocks = int(np.sum(high_conf_mask))
                if stats.num_blocks > 0:
                    stats.avg_block_confidence = float(np.mean(confidences[high_conf_mask]))
            elif blocks_np.ndim == 2:
                # Flat format: [num_detections, 6]
                confidences = blocks_np[:, 4] if blocks_np.shape[0] > 0 else np.array([])
                high_conf_mask = confidences > conf_thresh
                stats.num_blocks = int(np.sum(high_conf_mask))
                if stats.num_blocks > 0:
                    stats.avg_block_confidence = float(np.mean(confidences[high_conf_mask]))

        # Mask metrics
        if mask_np is not None:
            stats.mask_max_value = float(np.max(mask_np))
            # Coverage: percentage of pixels above 0.5 threshold
            stats.mask_coverage = float(np.mean(mask_np > 0.5) * 100)

        # Lines metrics
        if lines_np is not None:
            # Lines output typically has 2 channels (probability, threshold)
            if lines_np.ndim >= 3:
                prob_channel = lines_np[0] if lines_np.ndim == 3 else lines_np[0, 0]
                stats.lines_coverage = float(np.mean(prob_channel > 0.5) * 100)

        return stats


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    backend: str
    batch_size: int
    input_size: int
    warmup_iterations: int
    benchmark_iterations: int
    device: str

    # Timing metrics
    latency: LatencyStats = None
    first_inference_ms: float = 0.0
    warmup_time_ms: float = 0.0
    throughput_fps: float = 0.0

    # V10-specific: NMS-free timing
    nms_time_ms: float = 0.0  # Always 0 for V10 (NMS-free)

    # Memory metrics
    memory: MemoryStats = None

    # Output metrics (V10 3-output model)
    output_stats: OutputStats = None

    # Model info
    model_path: str = ""
    model_size_mb: float = 0.0
    num_outputs: int = 2  # 2 for legacy, 3 for V10

    # Profiler data (optional)
    profiler_trace_path: str = ""

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        return result


class BaseBackend:
    """Base class for inference backends."""

    def __init__(self, model_path: str, device: str = "cuda", input_size: int = 1024):
        self.model_path = model_path
        self.device = device
        self.input_size = input_size
        self.model = None

    def load(self):
        """Load the model."""
        raise NotImplementedError

    def infer(self, inputs: np.ndarray) -> Any:
        """Run inference on inputs."""
        raise NotImplementedError

    def create_dummy_input(self, batch_size: int) -> np.ndarray:
        """Create dummy input for benchmarking."""
        return np.random.rand(batch_size, 3, self.input_size, self.input_size).astype(np.float32)

    def cleanup(self):
        """Cleanup resources."""
        self.model = None
        gc.collect()
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()


class PyTorchBackend(BaseBackend):
    """PyTorch native inference backend."""

    def __init__(self, model_path: str, device: str = "cuda", input_size: int = 1024, half: bool = False):
        super().__init__(model_path, device, input_size)
        self.half = half

    def load(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        from src.models.backbone import create_backbone
        from src.models.heads import UnetHead, DBHead

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "weights" in checkpoint:
            weights = checkpoint["weights"]
            state_dict = {}
            for module_name, module_dict in weights.items():
                for k, v in module_dict.items():
                    state_dict[f"{module_name}.{k}"] = v
        else:
            state_dict = checkpoint

        # Normalize state dict keys
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # Create model components
        backbone = create_backbone(model_name="yolo11s.pt", pretrained=False, freeze=True)
        seg_head = UnetHead()
        db_head = DBHead(64)

        # Load weights
        backbone_state = {k[9:]: v for k, v in state_dict.items() if k.startswith("backbone.")}
        seg_state = {k[8:]: v for k, v in state_dict.items() if k.startswith("seg_net.")}
        db_state = {k[6:]: v for k, v in state_dict.items() if k.startswith("dbnet.")}

        if backbone_state:
            backbone.load_state_dict(backbone_state)
        if seg_state:
            seg_head.load_state_dict(seg_state)
        if db_state:
            db_head.load_state_dict(db_state)

        # Combine into single model
        class CombinedModel(nn.Module):
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

        self.model = CombinedModel(backbone, seg_head, db_head)
        self.model = self.model.to(self.device).eval()

        if self.half and self.device != "cpu":
            self.model = self.model.half()

    def infer(self, inputs: np.ndarray) -> Any:
        tensor = torch.from_numpy(inputs).to(self.device)
        if self.half and self.device != "cpu":
            tensor = tensor.half()

        with torch.no_grad():
            outputs = self.model(tensor)

        return outputs

    def create_dummy_input(self, batch_size: int) -> np.ndarray:
        dtype = np.float16 if self.half else np.float32
        return np.random.rand(batch_size, 3, self.input_size, self.input_size).astype(dtype)


class TorchCompileBackend(PyTorchBackend):
    """PyTorch with torch.compile optimization."""

    def __init__(self, model_path: str, device: str = "cuda", input_size: int = 1024,
                 half: bool = False, mode: str = "reduce-overhead"):
        super().__init__(model_path, device, input_size, half)
        self.compile_mode = mode

    def load(self):
        super().load()

        if not hasattr(torch, 'compile'):
            raise RuntimeError("torch.compile requires PyTorch 2.0+")

        print(f"  Compiling model with mode='{self.compile_mode}'...")
        self.model = torch.compile(self.model, mode=self.compile_mode, backend="inductor")

        # Warmup compilation
        dummy = torch.randn(1, 3, self.input_size, self.input_size, device=self.device)
        if self.half and self.device != "cpu":
            dummy = dummy.half()

        for _ in range(3):
            with torch.no_grad():
                _ = self.model(dummy)

        print("  Compilation complete")


class ONNXBackend(BaseBackend):
    """ONNX Runtime inference backend."""

    def __init__(self, model_path: str, device: str = "cuda", input_size: int = 1024):
        super().__init__(model_path, device, input_size)
        self.session = None
        self.input_name = None

    def load(self):
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available")

        # Select providers based on device
        available = ort.get_available_providers()
        if "cuda" in self.device:
            preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            preferred = ["CPUExecutionProvider"]

        providers = [p for p in preferred if p in available]
        if not providers:
            providers = available

        # Session options for optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options,
                providers=providers
            )
            actual_provider = self.session.get_providers()[0]
            print(f"  ONNX loaded with provider: {actual_provider}")
        except Exception as e:
            if "CUDAExecutionProvider" in providers:
                print(f"  CUDA init failed: {e}. Falling back to CPU.")
                self.session = ort.InferenceSession(
                    self.model_path,
                    sess_options,
                    providers=["CPUExecutionProvider"]
                )
            else:
                raise

        self.input_name = self.session.get_inputs()[0].name
        self.model = self.session

    def infer(self, inputs: np.ndarray) -> Any:
        return self.session.run(None, {self.input_name: inputs})


class TensorRTBackend(BaseBackend):
    """TensorRT inference backend."""

    def __init__(self, model_path: str, device: str = "cuda", input_size: int = 1024,
                 fp16: bool = True, workspace_gb: int = 4):
        super().__init__(model_path, device, input_size)
        self.fp16 = fp16
        self.workspace_gb = workspace_gb
        self.engine = None
        self.context = None
        self.bindings = None
        self.stream = None

    def load(self):
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available")
        if not CUDA_AVAILABLE:
            raise RuntimeError("TensorRT requires CUDA")

        import pycuda.driver as cuda
        import pycuda.autoinit

        model_path = Path(self.model_path)

        # Check if it's an ONNX file - need to build engine
        if model_path.suffix == ".onnx":
            engine_path = model_path.with_suffix(".engine")

            if not engine_path.exists():
                print(f"  Building TensorRT engine from ONNX...")
                self._build_engine(str(model_path), str(engine_path))
            else:
                print(f"  Loading cached TensorRT engine: {engine_path}")

            model_path = engine_path

        # Load engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(model_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate buffers
        self.bindings = []
        self.inputs = []
        self.outputs = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)

            # Replace -1 with actual batch size (1 for now)
            shape = tuple(1 if s == -1 else s for s in shape)
            size = int(np.prod(shape))

            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({"host": host_mem, "device": device_mem, "shape": shape, "name": name})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem, "shape": shape, "name": name})

        self.model = self.engine
        print(f"  TensorRT engine loaded")

    def _build_engine(self, onnx_path: str, engine_path: str):
        """Build TensorRT engine from ONNX model."""
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  Parse error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_gb << 30)

        if self.fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        print(f"  Building engine (this may take a few minutes)...")
        engine = builder.build_serialized_network(network, config)

        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        with open(engine_path, "wb") as f:
            f.write(engine)

        print(f"  Engine saved to: {engine_path}")

    def infer(self, inputs: np.ndarray) -> Any:
        import pycuda.driver as cuda

        # Update input shape for batch size
        batch_size = inputs.shape[0]
        input_shape = (batch_size, 3, self.input_size, self.input_size)

        # Copy input to device
        np.copyto(self.inputs[0]["host"].reshape(input_shape), inputs)
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)

        # Set input shape
        self.context.set_input_shape(self.inputs[0]["name"], input_shape)

        # Set tensor addresses
        for inp in self.inputs:
            self.context.set_tensor_address(inp["name"], int(inp["device"]))
        for out in self.outputs:
            self.context.set_tensor_address(out["name"], int(out["device"]))

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs back
        outputs = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
            outputs.append(out["host"].reshape(out["shape"]))

        self.stream.synchronize()
        return outputs

    def cleanup(self):
        if self.context:
            del self.context
        if self.engine:
            del self.engine
        super().cleanup()


def get_backend(backend_name: str, model_path: str, device: str, input_size: int, **kwargs) -> BaseBackend:
    """Factory function to create backend instance."""
    backends = {
        "pytorch": PyTorchBackend,
        "torch-compile": TorchCompileBackend,
        "onnx": ONNXBackend,
        "tensorrt": TensorRTBackend,
    }

    if backend_name not in backends:
        raise ValueError(f"Unknown backend: {backend_name}. Available: {list(backends.keys())}")

    return backends[backend_name](model_path, device, input_size, **kwargs)


def run_benchmark(
    backend: BaseBackend,
    batch_size: int,
    warmup_iterations: int = 10,
    benchmark_iterations: int = 100,
    use_profiler: bool = False,
    profiler_output_dir: Optional[str] = None,
) -> BenchmarkResult:
    """Run benchmark on a backend with specified parameters."""

    device = backend.device
    input_size = backend.input_size

    # Reset memory stats
    MemoryStats.reset()

    # Create dummy input
    dummy_input = backend.create_dummy_input(batch_size)

    # Measure first inference time
    if CUDA_AVAILABLE and "cuda" in device:
        torch.cuda.synchronize()

    first_start = time.perf_counter()
    _ = backend.infer(dummy_input)

    if CUDA_AVAILABLE and "cuda" in device:
        torch.cuda.synchronize()

    first_inference_ms = (time.perf_counter() - first_start) * 1000

    # Warmup
    warmup_start = time.perf_counter()
    for _ in range(warmup_iterations):
        _ = backend.infer(dummy_input)

    if CUDA_AVAILABLE and "cuda" in device:
        torch.cuda.synchronize()

    warmup_time_ms = (time.perf_counter() - warmup_start) * 1000

    # Benchmark
    latencies = []

    # Optional profiling
    profiler_trace_path = ""
    profiler_context = None

    if use_profiler and TORCH_AVAILABLE and CUDA_AVAILABLE:
        profiler_output_dir = profiler_output_dir or "./profiler_traces"
        Path(profiler_output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_name = f"benchmark_{backend.__class__.__name__}_bs{batch_size}_{timestamp}"
        profiler_trace_path = str(Path(profiler_output_dir) / trace_name)

        profiler_context = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=3, active=5, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_trace_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler_context.start()

    # Run benchmark iterations and capture last output for stats
    last_output = None
    for i in range(benchmark_iterations):
        if CUDA_AVAILABLE and "cuda" in device:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            output = backend.infer(dummy_input)
            end.record()

            torch.cuda.synchronize()
            latency_ms = start.elapsed_time(end)
        else:
            start = time.perf_counter()
            output = backend.infer(dummy_input)
            latency_ms = (time.perf_counter() - start) * 1000

        latencies.append(latency_ms)
        last_output = output

        if profiler_context:
            profiler_context.step()

    if profiler_context:
        profiler_context.stop()

    # Capture memory stats
    memory_stats = MemoryStats.capture()

    # Calculate throughput
    mean_latency_ms = np.mean(latencies)
    throughput_fps = (batch_size * 1000) / mean_latency_ms if mean_latency_ms > 0 else 0

    # Get model size
    model_size_mb = 0.0
    model_path = Path(backend.model_path)
    if model_path.exists():
        model_size_mb = model_path.stat().st_size / (1024 * 1024)

    # Compute output statistics (supports both 2 and 3 output models)
    output_stats = None
    num_outputs = 0
    if last_output is not None:
        # Handle tuple output (PyTorch) vs list output (ONNX)
        if isinstance(last_output, (tuple, list)):
            num_outputs = len(last_output)
            output_stats = OutputStats.from_outputs(last_output)
        else:
            # Single output - wrap in list
            num_outputs = 1
            output_stats = OutputStats.from_outputs([last_output])

    return BenchmarkResult(
        backend=backend.__class__.__name__,
        batch_size=batch_size,
        input_size=input_size,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
        device=device,
        latency=LatencyStats.from_latencies(latencies),
        first_inference_ms=first_inference_ms,
        warmup_time_ms=warmup_time_ms,
        throughput_fps=throughput_fps,
        nms_time_ms=0.0,  # V10 is NMS-free
        memory=memory_stats,
        output_stats=output_stats,
        model_path=str(backend.model_path),
        model_size_mb=model_size_mb,
        num_outputs=num_outputs,
        profiler_trace_path=profiler_trace_path,
    )


def format_results_table(results: List[BenchmarkResult], baseline: Optional[BenchmarkResult] = None) -> str:
    """Format benchmark results as markdown table."""
    lines = []

    # Check if any results have block detection (V10 3-output model)
    has_blocks = any(r.num_outputs == 3 for r in results)

    # Header
    if has_blocks:
        lines.append("| Backend | Batch | Resolution | Mean (ms) | P50 (ms) | P95 (ms) | FPS | Memory (MB) | Blocks | Speedup |")
        lines.append("|---------|-------|------------|-----------|----------|----------|-----|-------------|--------|---------|")
    else:
        lines.append("| Backend | Batch | Resolution | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | FPS | Memory (MB) | Speedup |")
        lines.append("|---------|-------|------------|-----------|----------|----------|----------|-----|-------------|---------|")

    baseline_latency = baseline.latency.mean_ms if baseline else None

    for r in results:
        speedup = ""
        if baseline_latency and r.latency.mean_ms > 0:
            speedup_val = baseline_latency / r.latency.mean_ms
            speedup = f"{speedup_val:.2f}x"

        memory_mb = r.memory.peak_allocated_mb if r.memory else 0

        if has_blocks:
            blocks_str = str(r.output_stats.num_blocks) if r.output_stats else "N/A"
            lines.append(
                f"| {r.backend} | {r.batch_size} | {r.input_size}x{r.input_size} | "
                f"{r.latency.mean_ms:.2f} | {r.latency.p50_ms:.2f} | {r.latency.p95_ms:.2f} | "
                f"{r.throughput_fps:.1f} | {memory_mb:.0f} | {blocks_str} | {speedup} |"
            )
        else:
            lines.append(
                f"| {r.backend} | {r.batch_size} | {r.input_size}x{r.input_size} | "
                f"{r.latency.mean_ms:.2f} | {r.latency.p50_ms:.2f} | {r.latency.p95_ms:.2f} | {r.latency.p99_ms:.2f} | "
                f"{r.throughput_fps:.1f} | {memory_mb:.0f} | {speedup} |"
            )

    return "\n".join(lines)


def format_summary(results: List[BenchmarkResult]) -> str:
    """Format a summary of benchmark results."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("BENCHMARK SUMMARY")
    lines.append("=" * 70)

    for r in results:
        lines.append(f"\n{r.backend} (batch={r.batch_size}, {r.input_size}x{r.input_size}):")
        lines.append(f"  Model outputs: {r.num_outputs} {'(V10 3-output)' if r.num_outputs == 3 else '(legacy 2-output)'}")
        lines.append(f"  Latency:     {r.latency.mean_ms:.2f} +/- {r.latency.std_ms:.2f} ms")
        lines.append(f"  P50/P95/P99: {r.latency.p50_ms:.2f} / {r.latency.p95_ms:.2f} / {r.latency.p99_ms:.2f} ms")
        lines.append(f"  Throughput:  {r.throughput_fps:.1f} images/sec")
        lines.append(f"  NMS time:    {r.nms_time_ms:.2f} ms {'(NMS-free)' if r.nms_time_ms == 0 else ''}")
        lines.append(f"  First infer: {r.first_inference_ms:.2f} ms")
        lines.append(f"  Warmup time: {r.warmup_time_ms:.2f} ms ({r.warmup_iterations} iters)")

        if r.memory and r.memory.peak_allocated_mb > 0:
            lines.append(f"  Peak memory: {r.memory.peak_allocated_mb:.0f} MB allocated, {r.memory.peak_reserved_mb:.0f} MB reserved")

        # V10 output statistics (block detection metrics)
        if r.output_stats:
            lines.append(f"  --- Output Statistics ---")
            if r.num_outputs == 3:
                lines.append(f"  Blocks detected:  {r.output_stats.num_blocks} (conf > 0.25)")
                if r.output_stats.avg_block_confidence > 0:
                    lines.append(f"  Avg block conf:   {r.output_stats.avg_block_confidence:.3f}")
            lines.append(f"  Mask max value:   {r.output_stats.mask_max_value:.3f}")
            lines.append(f"  Mask coverage:    {r.output_stats.mask_coverage:.2f}% (>0.5 threshold)")
            lines.append(f"  Lines coverage:   {r.output_stats.lines_coverage:.2f}% (>0.5 threshold)")

        if r.profiler_trace_path:
            lines.append(f"  Profiler:    {r.profiler_trace_path}")

    return "\n".join(lines)


def save_results(results: List[BenchmarkResult], output_dir: str, prefix: str = "benchmark"):
    """Save benchmark results to JSON and markdown files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_path / f"{prefix}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    # Save markdown
    md_path = output_path / f"{prefix}_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(f"# Benchmark Results\n\n")
        f.write(f"**Date:** {timestamp}\n\n")
        f.write("## Results Table\n\n")
        f.write(format_results_table(results))
        f.write("\n\n## Detailed Summary\n")
        f.write(format_summary(results))
    print(f"Markdown saved to: {md_path}")

    return json_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLOv10 Comic Text Detector across multiple backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark ONNX model
  python scripts/benchmark.py --model models/ctd_v10.onnx --backend onnx --batch-sizes 1,2,4,8

  # Benchmark with TensorRT
  python scripts/benchmark.py --model models/ctd_v10.onnx --backend tensorrt --warmup 10 --iterations 100

  # Compare all backends
  python scripts/benchmark.py --model models/ctd_v10.onnx --compare-backends --output-dir results

  # PyTorch with torch.compile
  python scripts/benchmark.py --model runs/block_v2/best.pt --backend torch-compile

  # Full profiling
  python scripts/benchmark.py --model models/ctd_v10.onnx --backend onnx --profile --output-dir profiling
        """
    )

    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to model file (.onnx, .pt, or .engine)")
    parser.add_argument("--backend", "-b", type=str, default="onnx",
                        choices=["pytorch", "torch-compile", "onnx", "tensorrt"],
                        help="Inference backend (default: onnx)")
    parser.add_argument("--batch-sizes", type=str, default="1",
                        help="Comma-separated list of batch sizes to test (default: 1)")
    parser.add_argument("--input-sizes", type=str, default="1024",
                        help="Comma-separated list of input resolutions (default: 1024)")
    parser.add_argument("--warmup", "-w", type=int, default=10,
                        help="Number of warmup iterations (default: 10)")
    parser.add_argument("--iterations", "-i", type=int, default=100,
                        help="Number of benchmark iterations (default: 100)")
    parser.add_argument("--device", "-d", type=str, default="cuda",
                        help="Device for inference (default: cuda)")
    parser.add_argument("--half", action="store_true",
                        help="Use FP16 precision for PyTorch backend")
    parser.add_argument("--profile", action="store_true",
                        help="Enable torch.profiler for detailed analysis")
    parser.add_argument("--output-dir", "-o", type=str, default="benchmark_results",
                        help="Directory to save results (default: benchmark_results)")
    parser.add_argument("--compare-backends", action="store_true",
                        help="Compare all available backends")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Path to baseline model for speedup comparison")
    parser.add_argument("--json-only", action="store_true",
                        help="Only output JSON results (no console output)")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (default: reduce-overhead)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse batch sizes and input sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    input_sizes = [int(x) for x in args.input_sizes.split(",")]

    # Check device availability
    if args.device == "cuda" and not CUDA_AVAILABLE:
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"

    if not args.json_only:
        print("=" * 70)
        print("YOLOv10 Comic Text Detector Benchmark")
        print("=" * 70)
        print(f"Model:      {args.model}")
        print(f"Backend:    {args.backend}")
        print(f"Device:     {args.device}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Input sizes: {input_sizes}")
        print(f"Warmup:     {args.warmup} iterations")
        print(f"Benchmark:  {args.iterations} iterations")
        print(f"Profiler:   {'enabled' if args.profile else 'disabled'}")
        print("=" * 70)

    all_results = []

    # Determine which backends to test
    if args.compare_backends:
        backends_to_test = []

        # Check availability
        model_path = Path(args.model)
        if model_path.suffix == ".pt":
            if TORCH_AVAILABLE:
                backends_to_test.append("pytorch")
                if hasattr(torch, 'compile'):
                    backends_to_test.append("torch-compile")
        else:  # ONNX or engine file
            if ONNX_AVAILABLE:
                backends_to_test.append("onnx")
            if TENSORRT_AVAILABLE and CUDA_AVAILABLE:
                backends_to_test.append("tensorrt")

        if not backends_to_test:
            print("Error: No backends available for the given model type")
            sys.exit(1)

        print(f"Comparing backends: {backends_to_test}\n")
    else:
        backends_to_test = [args.backend]

    # Run benchmarks
    for backend_name in backends_to_test:
        for input_size in input_sizes:
            for batch_size in batch_sizes:
                if not args.json_only:
                    print(f"\nBenchmarking {backend_name} (batch={batch_size}, size={input_size}x{input_size})...")

                try:
                    # Backend-specific kwargs
                    kwargs = {}
                    if backend_name in ["pytorch", "torch-compile"]:
                        kwargs["half"] = args.half
                    if backend_name == "torch-compile":
                        kwargs["mode"] = args.compile_mode

                    backend = get_backend(
                        backend_name,
                        args.model,
                        args.device,
                        input_size,
                        **kwargs
                    )

                    backend.load()

                    result = run_benchmark(
                        backend,
                        batch_size=batch_size,
                        warmup_iterations=args.warmup,
                        benchmark_iterations=args.iterations,
                        use_profiler=args.profile,
                        profiler_output_dir=args.output_dir if args.profile else None,
                    )

                    all_results.append(result)

                    if not args.json_only:
                        print(f"  Mean latency: {result.latency.mean_ms:.2f} ms")
                        print(f"  Throughput:   {result.throughput_fps:.1f} FPS")

                    backend.cleanup()

                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()

    if not all_results:
        print("No benchmark results collected")
        sys.exit(1)

    # Print summary
    if not args.json_only:
        print(format_summary(all_results))
        print("\n" + "=" * 70)
        print("RESULTS TABLE")
        print("=" * 70 + "\n")
        print(format_results_table(all_results))

    # Save results
    json_path, md_path = save_results(all_results, args.output_dir)

    # If baseline provided, calculate speedups
    if args.baseline:
        print(f"\nBaseline comparison with: {args.baseline}")
        # Load baseline and compare (implementation details depend on baseline format)


if __name__ == "__main__":
    main()
