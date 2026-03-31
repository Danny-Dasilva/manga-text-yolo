#!/usr/bin/env python3
"""
TensorRT Engine Builder Script

Build optimized TensorRT engines from ONNX models with:
- FP16/FP32/INT8 precision support
- Dynamic/fixed batch sizes
- INT8 calibration with calibration dataset
- Engine caching to disk for reuse
- Timing/profiling output
- Support for multiple input resolutions (common manga page sizes)

Usage:
    # Basic FP16 engine build
    python scripts/build_tensorrt.py --onnx model.onnx --output model.engine --precision fp16

    # INT8 with calibration
    python scripts/build_tensorrt.py --onnx model.onnx --output model.engine --precision int8 --calibration-dir ./images

    # Dynamic batch size
    python scripts/build_tensorrt.py --onnx model.onnx --output model.engine --precision fp16 --dynamic-batch --min-batch 1 --opt-batch 4 --max-batch 16

    # Multiple resolutions (manga page sizes)
    python scripts/build_tensorrt.py --onnx model.onnx --output model.engine --precision fp16 --resolutions 768 1024 1280

    # With profiling
    python scripts/build_tensorrt.py --onnx model.onnx --output model.engine --precision fp16 --profile
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Common manga/comic page resolutions
COMMON_RESOLUTIONS = {
    'small': 512,
    'medium': 768,
    'standard': 1024,
    'large': 1280,
    'xlarge': 1536,
    'full': 2048,
}


def _check_tensorrt():
    """Check TensorRT availability and return module if available."""
    try:
        import tensorrt as trt
        return trt, trt.__version__
    except ImportError:
        return None, None


def _check_pycuda():
    """Check PyCUDA availability for INT8 calibration."""
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        return cuda, True
    except ImportError:
        return None, False


# Lazy check - actual import happens when needed
TRT_MODULE, TRT_VERSION = _check_tensorrt()
TRT_AVAILABLE = TRT_MODULE is not None
CUDA_MODULE, PYCUDA_AVAILABLE = _check_pycuda()


def _create_trt_logger(verbose: bool = False):
    """Create TensorRT logger with severity filtering."""
    if not TRT_AVAILABLE:
        raise ImportError("TensorRT not available")

    trt = TRT_MODULE

    class TRTLogger(trt.ILogger):
        """Custom TensorRT logger with severity filtering."""

        def __init__(self, verbose: bool = False):
            super().__init__()
            self.verbose = verbose

        def log(self, severity, msg):
            if self.verbose:
                if severity == trt.Logger.ERROR:
                    print(f"[TRT ERROR] {msg}")
                elif severity == trt.Logger.WARNING:
                    print(f"[TRT WARNING] {msg}")
                elif severity == trt.Logger.INFO:
                    print(f"[TRT INFO] {msg}")
            else:
                # Only show errors and warnings in non-verbose mode
                if severity <= trt.Logger.WARNING:
                    print(f"[TRT] {msg}")

    return TRTLogger(verbose)


def _create_int8_calibrator(
    calibration_dir: str,
    cache_file: str,
    input_size: int = 1024,
    batch_size: int = 1,
    num_samples: int = 100,
):
    """Create INT8 calibrator using calibration images.

    Returns an instance of a dynamically created calibrator class.
    This is necessary because we need TensorRT to be available at class definition time.
    """
    if not TRT_AVAILABLE:
        raise ImportError("TensorRT not available")
    if not PYCUDA_AVAILABLE:
        raise ImportError("PyCUDA required for INT8 calibration: pip install pycuda")

    trt = TRT_MODULE
    cuda = CUDA_MODULE

    class INT8Calibrator(trt.IInt8EntropyCalibrator2):
        """INT8 calibrator using calibration images."""

        def __init__(
            self,
            calibration_dir: str,
            cache_file: str,
            input_size: int = 1024,
            batch_size: int = 1,
            num_samples: int = 100,
        ):
            super().__init__()
            self.cache_file = cache_file
            self.input_size = input_size
            self.batch_size = batch_size
            self.num_samples = num_samples

            # Collect calibration images
            self.image_paths = []
            cal_path = Path(calibration_dir)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                self.image_paths.extend(cal_path.glob(ext))
                self.image_paths.extend(cal_path.glob(ext.upper()))

            self.image_paths = self.image_paths[:num_samples]
            self.current_index = 0

            if len(self.image_paths) == 0:
                raise ValueError(f"No calibration images found in {calibration_dir}")

            print(f"INT8 Calibrator initialized with {len(self.image_paths)} images")

            # Allocate device memory for batch
            self.device_input = cuda.mem_alloc(
                batch_size * 3 * input_size * input_size * np.float32().itemsize
            )
            self.batch_data = np.zeros(
                (batch_size, 3, input_size, input_size), dtype=np.float32
            )

        def get_batch_size(self) -> int:
            return self.batch_size

        def get_batch(self, names: List[str]) -> Optional[List[int]]:
            """Load and preprocess a batch of calibration images."""
            if self.current_index >= len(self.image_paths):
                return None

            try:
                import cv2
            except ImportError:
                raise ImportError("OpenCV required for INT8 calibration: pip install opencv-python")

            batch_images = []
            for i in range(self.batch_size):
                if self.current_index + i >= len(self.image_paths):
                    # Repeat last image to fill batch
                    img_path = self.image_paths[-1]
                else:
                    img_path = self.image_paths[self.current_index + i]

                # Load and preprocess
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Warning: Failed to load {img_path}, using zeros")
                    img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.input_size, self.input_size))

                # Normalize to [0, 1]
                img = img.astype(np.float32) / 255.0
                # CHW format
                img = img.transpose(2, 0, 1)
                batch_images.append(img)

            self.current_index += self.batch_size

            # Stack and copy to device
            self.batch_data = np.stack(batch_images, axis=0).astype(np.float32)
            cuda.memcpy_htod(self.device_input, self.batch_data.ravel())

            return [int(self.device_input)]

        def read_calibration_cache(self) -> Optional[bytes]:
            """Read cached calibration data if available."""
            if os.path.exists(self.cache_file):
                print(f"Reading calibration cache from {self.cache_file}")
                with open(self.cache_file, 'rb') as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache: bytes):
            """Write calibration data to cache file."""
            print(f"Writing calibration cache to {self.cache_file}")
            with open(self.cache_file, 'wb') as f:
                f.write(cache)

    return INT8Calibrator(
        calibration_dir=calibration_dir,
        cache_file=cache_file,
        input_size=input_size,
        batch_size=batch_size,
        num_samples=num_samples,
    )


class TensorRTBuilder:
    """TensorRT engine builder with caching and profiling support."""

    def __init__(
        self,
        onnx_path: str,
        output_path: str,
        precision: str = 'fp16',
        workspace_gb: float = 4.0,
        batch_size: int = 1,
        dynamic_batch: bool = False,
        min_batch: int = 1,
        opt_batch: int = 1,
        max_batch: int = 16,
        input_size: int = 1024,
        dynamic_shapes: bool = False,
        min_resolution: int = 512,
        opt_resolution: int = 1024,
        max_resolution: int = 2048,
        calibration_dir: Optional[str] = None,
        calibration_cache: Optional[str] = None,
        num_calibration_samples: int = 100,
        cache_dir: Optional[str] = None,
        verbose: bool = False,
        profile: bool = False,
    ):
        if not TRT_AVAILABLE:
            raise ImportError(
                "TensorRT not available. Install with: pip install tensorrt"
            )

        self.onnx_path = Path(onnx_path)
        self.output_path = Path(output_path)
        self.precision = precision.lower()
        self.workspace_gb = workspace_gb
        self.batch_size = batch_size
        self.dynamic_batch = dynamic_batch
        self.min_batch = min_batch
        self.opt_batch = opt_batch
        self.max_batch = max_batch
        self.input_size = input_size
        self.dynamic_shapes = dynamic_shapes
        self.min_resolution = min_resolution
        self.opt_resolution = opt_resolution
        self.max_resolution = max_resolution
        self.calibration_dir = calibration_dir
        self.calibration_cache = calibration_cache
        self.num_calibration_samples = num_calibration_samples
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.verbose = verbose
        self.profile = profile

        # Validate precision
        valid_precisions = ['fp32', 'fp16', 'int8']
        if self.precision not in valid_precisions:
            raise ValueError(f"Invalid precision: {precision}. Must be one of {valid_precisions}")

        # INT8 requires calibration data
        if self.precision == 'int8':
            if not PYCUDA_AVAILABLE:
                raise ImportError(
                    "PyCUDA required for INT8 calibration: pip install pycuda"
                )
            if not calibration_dir:
                raise ValueError("INT8 precision requires --calibration-dir")

        # Initialize TensorRT logger
        self.logger = _create_trt_logger(verbose)

    def _compute_cache_key(self) -> str:
        """Compute unique cache key for this engine configuration."""
        # Read ONNX file hash
        with open(self.onnx_path, 'rb') as f:
            model_hash = hashlib.md5(f.read()).hexdigest()[:8]

        # Get GPU info
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).replace(' ', '_')
            else:
                gpu_name = "unknown_gpu"
        except ImportError:
            gpu_name = "unknown_gpu"

        # Build config string
        config = f"{model_hash}_{gpu_name}_{TRT_VERSION}_{self.precision}"
        if self.dynamic_batch:
            config += f"_batch{self.min_batch}-{self.opt_batch}-{self.max_batch}"
        else:
            config += f"_batch{self.batch_size}"
        if self.dynamic_shapes:
            config += f"_res{self.min_resolution}-{self.opt_resolution}-{self.max_resolution}"
        else:
            config += f"_res{self.input_size}"

        return config

    def _get_cached_engine_path(self) -> Optional[Path]:
        """Get path to cached engine if caching is enabled."""
        if not self.cache_dir:
            return None

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._compute_cache_key()
        return self.cache_dir / f"{cache_key}.engine"

    def _load_cached_engine(self) -> Optional[bytes]:
        """Load engine from cache if available."""
        cache_path = self._get_cached_engine_path()
        if cache_path and cache_path.exists():
            print(f"Loading cached TensorRT engine from: {cache_path}")
            with open(cache_path, 'rb') as f:
                return f.read()
        return None

    def _save_engine_to_cache(self, engine_bytes: bytes):
        """Save engine to cache."""
        cache_path = self._get_cached_engine_path()
        if cache_path:
            print(f"Caching TensorRT engine to: {cache_path}")
            with open(cache_path, 'wb') as f:
                f.write(engine_bytes)

    def build(self) -> bytes:
        """Build TensorRT engine from ONNX model."""
        print(f"\n{'='*60}")
        print("TensorRT Engine Builder")
        print(f"{'='*60}")
        print(f"ONNX Model: {self.onnx_path}")
        print(f"Output: {self.output_path}")
        print(f"Precision: {self.precision.upper()}")
        print(f"Workspace: {self.workspace_gb} GB")

        if self.dynamic_batch:
            print(f"Batch Size: dynamic (min={self.min_batch}, opt={self.opt_batch}, max={self.max_batch})")
        else:
            print(f"Batch Size: {self.batch_size} (fixed)")

        if self.dynamic_shapes:
            print(f"Resolution: dynamic (min={self.min_resolution}, opt={self.opt_resolution}, max={self.max_resolution})")
        else:
            print(f"Resolution: {self.input_size}x{self.input_size} (fixed)")

        print(f"TensorRT Version: {TRT_VERSION}")
        print(f"{'='*60}\n")

        # Check cache first
        cached_engine = self._load_cached_engine()
        if cached_engine:
            # Save to output path as well
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, 'wb') as f:
                f.write(cached_engine)
            print(f"Engine saved to: {self.output_path}")
            return cached_engine

        # Build engine
        start_time = time.time()
        engine_bytes = self._build_engine()
        build_time = time.time() - start_time

        print(f"\nEngine build completed in {build_time:.1f} seconds")

        # Save to output path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'wb') as f:
            f.write(engine_bytes)
        print(f"Engine saved to: {self.output_path}")

        # Save to cache
        self._save_engine_to_cache(engine_bytes)

        # Print engine info
        self._print_engine_info(engine_bytes)

        # Run profiling if requested
        if self.profile:
            self._run_profiling(engine_bytes)

        return engine_bytes

    def _build_engine(self) -> bytes:
        """Internal method to build TensorRT engine."""
        trt = TRT_MODULE

        # Create builder and network
        builder = trt.Builder(self.logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)

        # Parse ONNX model
        print("Parsing ONNX model...")
        with open(self.onnx_path, 'rb') as f:
            onnx_data = f.read()

        if not parser.parse(onnx_data):
            errors = []
            for i in range(parser.num_errors):
                errors.append(parser.get_error(i))
            raise RuntimeError(f"ONNX parsing failed:\n" + "\n".join(str(e) for e in errors))

        print(f"  Network inputs: {network.num_inputs}")
        print(f"  Network outputs: {network.num_outputs}")
        print(f"  Network layers: {network.num_layers}")

        # Print input/output info
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            print(f"  Input {i}: {inp.name} - {inp.shape} ({inp.dtype})")
        for i in range(network.num_outputs):
            out = network.get_output(i)
            print(f"  Output {i}: {out.name} - {out.shape} ({out.dtype})")

        # Create builder config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(self.workspace_gb * (1 << 30))
        )

        # Set precision flags
        if self.precision == 'fp16':
            if builder.platform_has_fast_fp16:
                print("Enabling FP16 precision")
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                print("Warning: FP16 not supported on this platform, using FP32")
        elif self.precision == 'int8':
            if builder.platform_has_fast_int8:
                print("Enabling INT8 precision")
                config.set_flag(trt.BuilderFlag.INT8)
                # Also enable FP16 as fallback for layers that don't support INT8
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)

                # Set up calibrator
                calibration_cache_file = self.calibration_cache or str(
                    self.output_path.with_suffix('.calibration_cache')
                )
                calibrator = _create_int8_calibrator(
                    calibration_dir=self.calibration_dir,
                    cache_file=calibration_cache_file,
                    input_size=self.opt_resolution if self.dynamic_shapes else self.input_size,
                    batch_size=self.opt_batch if self.dynamic_batch else self.batch_size,
                    num_samples=self.num_calibration_samples,
                )
                config.int8_calibrator = calibrator
            else:
                print("Warning: INT8 not supported on this platform, using FP16")
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)

        # Configure dynamic shapes if enabled
        if self.dynamic_batch or self.dynamic_shapes:
            print("Configuring dynamic shapes...")
            profile = builder.create_optimization_profile()

            input_tensor = network.get_input(0)
            input_name = input_tensor.name

            if self.dynamic_batch and self.dynamic_shapes:
                # Both batch and resolution are dynamic
                min_shape = (self.min_batch, 3, self.min_resolution, self.min_resolution)
                opt_shape = (self.opt_batch, 3, self.opt_resolution, self.opt_resolution)
                max_shape = (self.max_batch, 3, self.max_resolution, self.max_resolution)
            elif self.dynamic_batch:
                # Only batch is dynamic
                min_shape = (self.min_batch, 3, self.input_size, self.input_size)
                opt_shape = (self.opt_batch, 3, self.input_size, self.input_size)
                max_shape = (self.max_batch, 3, self.input_size, self.input_size)
            else:
                # Only resolution is dynamic
                min_shape = (self.batch_size, 3, self.min_resolution, self.min_resolution)
                opt_shape = (self.batch_size, 3, self.opt_resolution, self.opt_resolution)
                max_shape = (self.batch_size, 3, self.max_resolution, self.max_resolution)

            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            print(f"  Min shape: {min_shape}")
            print(f"  Opt shape: {opt_shape}")
            print(f"  Max shape: {max_shape}")

            config.add_optimization_profile(profile)

        # Build engine
        print("\nBuilding TensorRT engine (this may take several minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        return serialized_engine

    def _print_engine_info(self, engine_bytes):
        """Print information about the built engine."""
        print(f"\n{'='*60}")
        print("Engine Information")
        print(f"{'='*60}")

        # Engine size - handle both bytes and IHostMemory
        # IHostMemory has .nbytes attribute in TensorRT 10.x
        if hasattr(engine_bytes, 'nbytes'):
            size_mb = engine_bytes.nbytes / (1024 * 1024)
        elif hasattr(engine_bytes, '__len__'):
            size_mb = len(engine_bytes) / (1024 * 1024)
        else:
            # Fall back to file size
            size_mb = self.output_path.stat().st_size / (1024 * 1024)
        print(f"Engine size: {size_mb:.2f} MB")

        # ONNX model size for comparison
        onnx_size_mb = self.onnx_path.stat().st_size / (1024 * 1024)
        print(f"ONNX size: {onnx_size_mb:.2f} MB")
        print(f"Size ratio: {size_mb / onnx_size_mb:.2f}x")

        # Deserialize engine to get more info
        trt = TRT_MODULE
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)

        if engine:
            print(f"\nEngine bindings:")
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                shape = engine.get_tensor_shape(name)
                dtype = engine.get_tensor_dtype(name)
                mode = "INPUT" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "OUTPUT"
                print(f"  {mode}: {name} - {list(shape)} ({dtype})")

    def _run_profiling(self, engine_bytes: bytes):
        """Run inference profiling on the built engine."""
        print(f"\n{'='*60}")
        print("Inference Profiling")
        print(f"{'='*60}")

        try:
            import torch
        except ImportError:
            print("PyTorch required for profiling. Skipping.")
            return

        if not torch.cuda.is_available():
            print("CUDA not available. Skipping profiling.")
            return

        trt = TRT_MODULE

        # Deserialize engine
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        context = engine.create_execution_context()

        # Determine input shape
        if self.dynamic_batch or self.dynamic_shapes:
            batch = self.opt_batch if self.dynamic_batch else self.batch_size
            res = self.opt_resolution if self.dynamic_shapes else self.input_size
        else:
            batch = self.batch_size
            res = self.input_size

        input_shape = (batch, 3, res, res)

        # Allocate buffers
        input_name = engine.get_tensor_name(0)
        context.set_input_shape(input_name, input_shape)

        # Allocate input/output tensors
        inputs = {}
        outputs = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = context.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            tensor = torch.zeros(tuple(shape), dtype=torch.float32, device='cuda')
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs[name] = tensor
            else:
                outputs[name] = tensor
            context.set_tensor_address(name, tensor.data_ptr())

        # Warmup
        print(f"\nWarmup (10 iterations)...")
        for _ in range(10):
            context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        # Benchmark
        num_iterations = 100
        print(f"Running {num_iterations} iterations...")

        latencies = []
        for _ in range(num_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            end.record()

            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))

        latencies = np.array(latencies)

        print(f"\nResults (input shape: {input_shape}):")
        print(f"  Mean latency: {np.mean(latencies):.2f} ms")
        print(f"  Std latency: {np.std(latencies):.2f} ms")
        print(f"  P50 latency: {np.percentile(latencies, 50):.2f} ms")
        print(f"  P95 latency: {np.percentile(latencies, 95):.2f} ms")
        print(f"  P99 latency: {np.percentile(latencies, 99):.2f} ms")
        print(f"  Min latency: {np.min(latencies):.2f} ms")
        print(f"  Max latency: {np.max(latencies):.2f} ms")
        print(f"  Throughput: {1000.0 / np.mean(latencies):.1f} FPS")

        # Memory usage
        torch.cuda.synchronize()
        mem_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        mem_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"\nGPU Memory:")
        print(f"  Allocated: {mem_allocated:.1f} MB")
        print(f"  Reserved: {mem_reserved:.1f} MB")


def build_for_multiple_resolutions(
    onnx_path: str,
    output_dir: str,
    resolutions: List[int],
    precision: str = 'fp16',
    **kwargs
) -> Dict[int, Path]:
    """Build separate engines for multiple input resolutions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engines = {}
    for res in resolutions:
        print(f"\n{'='*60}")
        print(f"Building engine for resolution: {res}x{res}")
        print(f"{'='*60}")

        output_path = output_dir / f"model_{res}x{res}_{precision}.engine"

        builder = TensorRTBuilder(
            onnx_path=onnx_path,
            output_path=str(output_path),
            precision=precision,
            input_size=res,
            **kwargs
        )
        builder.build()
        engines[res] = output_path

    return engines


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build TensorRT engine from ONNX model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic FP16 engine
  python scripts/build_tensorrt.py --onnx model.onnx --output model.engine --precision fp16

  # INT8 with calibration
  python scripts/build_tensorrt.py --onnx model.onnx --output model.engine --precision int8 --calibration-dir ./calibration_images

  # Dynamic batch size
  python scripts/build_tensorrt.py --onnx model.onnx --output model.engine --precision fp16 --dynamic-batch --min-batch 1 --opt-batch 4 --max-batch 16

  # Multiple resolutions for manga pages
  python scripts/build_tensorrt.py --onnx model.onnx --output-dir ./engines --precision fp16 --resolutions 768 1024 1280

  # With engine caching and profiling
  python scripts/build_tensorrt.py --onnx model.onnx --output model.engine --precision fp16 --cache-dir .trt_cache --profile
        """
    )

    # Required arguments
    parser.add_argument('--onnx', '-i', type=str, required=True,
                        help='Path to input ONNX model')

    # Output arguments (mutually exclusive)
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument('--output', '-o', type=str,
                              help='Path to output TensorRT engine file')
    output_group.add_argument('--output-dir', type=str,
                              help='Directory for output engines (when using --resolutions)')

    # Precision
    parser.add_argument('--precision', '-p', type=str, default='fp16',
                        choices=['fp32', 'fp16', 'int8'],
                        help='Precision mode (default: fp16)')

    # Batch size options
    parser.add_argument('--batch', '-b', type=int, default=1,
                        help='Batch size for fixed batch mode (default: 1)')
    parser.add_argument('--dynamic-batch', action='store_true',
                        help='Enable dynamic batch size')
    parser.add_argument('--min-batch', type=int, default=1,
                        help='Minimum batch size for dynamic batching (default: 1)')
    parser.add_argument('--opt-batch', type=int, default=1,
                        help='Optimal batch size for dynamic batching (default: 1)')
    parser.add_argument('--max-batch', type=int, default=16,
                        help='Maximum batch size for dynamic batching (default: 16)')

    # Resolution options
    parser.add_argument('--input-size', type=int, default=1024,
                        help='Input resolution for fixed resolution mode (default: 1024)')
    parser.add_argument('--resolutions', type=int, nargs='+',
                        help='Build engines for multiple resolutions (e.g., --resolutions 768 1024 1280)')
    parser.add_argument('--dynamic-shapes', action='store_true',
                        help='Enable dynamic input resolution')
    parser.add_argument('--min-resolution', type=int, default=512,
                        help='Minimum resolution for dynamic shapes (default: 512)')
    parser.add_argument('--opt-resolution', type=int, default=1024,
                        help='Optimal resolution for dynamic shapes (default: 1024)')
    parser.add_argument('--max-resolution', type=int, default=2048,
                        help='Maximum resolution for dynamic shapes (default: 2048)')

    # Workspace
    parser.add_argument('--workspace', '-w', type=float, default=4.0,
                        help='Workspace size in GB (default: 4.0)')

    # INT8 calibration
    parser.add_argument('--calibration-dir', type=str,
                        help='Directory with calibration images for INT8')
    parser.add_argument('--calibration-cache', type=str,
                        help='Path to calibration cache file')
    parser.add_argument('--num-calibration-samples', type=int, default=100,
                        help='Number of calibration samples (default: 100)')

    # Engine caching
    parser.add_argument('--cache-dir', type=str,
                        help='Directory to cache built engines')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable engine caching')

    # Profiling and output
    parser.add_argument('--profile', action='store_true',
                        help='Run inference profiling after build')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    return parser.parse_args()


def main():
    args = parse_args()

    # Check TensorRT availability
    if not TRT_AVAILABLE:
        print("ERROR: TensorRT not available")
        print("Install with: pip install tensorrt")
        sys.exit(1)

    # Check ONNX file exists
    if not Path(args.onnx).exists():
        print(f"ERROR: ONNX file not found: {args.onnx}")
        sys.exit(1)

    # Handle multiple resolutions
    if args.resolutions:
        if not args.output_dir:
            print("ERROR: --output-dir required when using --resolutions")
            sys.exit(1)

        build_for_multiple_resolutions(
            onnx_path=args.onnx,
            output_dir=args.output_dir,
            resolutions=args.resolutions,
            precision=args.precision,
            workspace_gb=args.workspace,
            batch_size=args.batch,
            dynamic_batch=args.dynamic_batch,
            min_batch=args.min_batch,
            opt_batch=args.opt_batch,
            max_batch=args.max_batch,
            calibration_dir=args.calibration_dir,
            calibration_cache=args.calibration_cache,
            num_calibration_samples=args.num_calibration_samples,
            cache_dir=None if args.no_cache else args.cache_dir,
            verbose=args.verbose,
            profile=args.profile,
        )
    else:
        # Single engine build
        if not args.output:
            print("ERROR: --output required for single engine build")
            sys.exit(1)

        builder = TensorRTBuilder(
            onnx_path=args.onnx,
            output_path=args.output,
            precision=args.precision,
            workspace_gb=args.workspace,
            batch_size=args.batch,
            dynamic_batch=args.dynamic_batch,
            min_batch=args.min_batch,
            opt_batch=args.opt_batch,
            max_batch=args.max_batch,
            input_size=args.input_size,
            dynamic_shapes=args.dynamic_shapes,
            min_resolution=args.min_resolution,
            opt_resolution=args.opt_resolution,
            max_resolution=args.max_resolution,
            calibration_dir=args.calibration_dir,
            calibration_cache=args.calibration_cache,
            num_calibration_samples=args.num_calibration_samples,
            cache_dir=None if args.no_cache else args.cache_dir,
            verbose=args.verbose,
            profile=args.profile,
        )
        builder.build()

    print("\nDone!")


if __name__ == '__main__':
    main()
