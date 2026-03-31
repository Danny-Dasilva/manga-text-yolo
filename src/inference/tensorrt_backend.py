"""
TensorRT Inference Backend with Persistent Engine Caching

This module provides a high-performance TensorRT inference backend that:
- Automatically caches compiled engines to disk based on model hash, GPU, and TRT version
- Falls back to ONNX Runtime when TensorRT is unavailable
- Supports dynamic input shapes with optimization profiles
- Provides memory-efficient execution with pre-allocated buffers

Usage:
    # Load from ONNX (will build or load cached engine)
    engine = TensorRTEngine('model.onnx', cache_dir='.trt_cache')

    # Run inference
    outputs = engine(input_tensor)

    # Or use context manager for automatic cleanup
    with TensorRTEngine('model.onnx') as engine:
        outputs = engine(input_tensor)
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def _check_tensorrt() -> Tuple[Any, Optional[str]]:
    """Check TensorRT availability and return module if available."""
    try:
        import tensorrt as trt
        return trt, trt.__version__
    except ImportError:
        return None, None


def _check_torch_cuda() -> Tuple[bool, Optional[str]]:
    """Check PyTorch CUDA availability and return GPU name."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
        return False, None
    except ImportError:
        return False, None


# Lazy initialization
TRT_MODULE, TRT_VERSION = _check_tensorrt()
TRT_AVAILABLE = TRT_MODULE is not None
CUDA_AVAILABLE, GPU_NAME = _check_torch_cuda()


@dataclass
class TensorRTEngineConfig:
    """Configuration for TensorRT engine building."""
    workspace_gb: float = 4.0
    fp16: bool = True
    int8: bool = False
    strict_types: bool = False
    dla_core: int = -1  # -1 = don't use DLA
    allow_gpu_fallback: bool = True

    # Dynamic shape optimization profiles
    min_batch: int = 1
    opt_batch: int = 1
    max_batch: int = 8
    min_resolution: int = 512
    opt_resolution: int = 1024
    max_resolution: int = 2048


class TensorRTEngine:
    """
    TensorRT inference engine with persistent caching.

    Automatically handles:
    - Engine building from ONNX with optimal settings
    - Disk caching based on model hash + GPU + TRT version
    - Input/output buffer allocation and management
    - Graceful fallback to ONNX Runtime

    Args:
        onnx_path: Path to ONNX model file
        cache_dir: Directory to store cached engines (default: '.trt_cache')
        config: Engine configuration (optional)
        force_rebuild: Force rebuild even if cached engine exists
        fallback_to_onnx: Fall back to ONNX Runtime if TensorRT fails

    Example:
        >>> engine = TensorRTEngine('model.onnx', cache_dir='.trt_cache')
        >>> output = engine(torch.randn(1, 3, 1024, 1024, device='cuda'))
    """

    def __init__(
        self,
        onnx_path: Union[str, Path],
        cache_dir: Union[str, Path] = '.trt_cache',
        config: Optional[TensorRTEngineConfig] = None,
        force_rebuild: bool = False,
        fallback_to_onnx: bool = True,
    ):
        self.onnx_path = Path(onnx_path)
        self.cache_dir = Path(cache_dir)
        self.config = config or TensorRTEngineConfig()
        self.force_rebuild = force_rebuild
        self.fallback_to_onnx = fallback_to_onnx

        # Runtime state
        self.engine = None
        self.context = None
        self.stream = None
        self.buffers: Dict[str, Any] = {}
        self.input_names: List[str] = []
        self.output_names: List[str] = []

        # Fallback
        self.onnx_session = None
        self.using_fallback = False

        # Validate ONNX path
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        # Generate cache key and load/build engine
        self.cache_key = self._compute_cache_key()
        self.engine_path = self.cache_dir / f"{self.cache_key}.engine"

        self._initialize()

    def _compute_cache_key(self) -> str:
        """
        Compute unique cache key based on model content, GPU, and TRT version.

        The cache key ensures that engines are rebuilt when:
        - Model content changes (MD5 hash)
        - GPU hardware changes (different optimization)
        - TensorRT version changes (different kernels)
        - Config changes (precision, batch size, etc.)
        """
        # Model hash
        with open(self.onnx_path, 'rb') as f:
            model_hash = hashlib.md5(f.read()).hexdigest()[:8]

        # GPU identifier
        gpu_id = GPU_NAME.replace(' ', '_').replace('/', '-') if GPU_NAME else 'unknown_gpu'

        # TensorRT version
        trt_ver = TRT_VERSION.replace('.', '_') if TRT_VERSION else 'no_trt'

        # Config hash
        config_str = (
            f"fp16={self.config.fp16}_"
            f"int8={self.config.int8}_"
            f"batch={self.config.min_batch}-{self.config.opt_batch}-{self.config.max_batch}_"
            f"res={self.config.min_resolution}-{self.config.opt_resolution}-{self.config.max_resolution}"
        )
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]

        return f"{model_hash}_{gpu_id}_{trt_ver}_{config_hash}"

    def _initialize(self):
        """Initialize engine from cache or build new one."""
        if not TRT_AVAILABLE:
            logger.warning("TensorRT not available, falling back to ONNX Runtime")
            self._init_onnx_fallback()
            return

        if not CUDA_AVAILABLE:
            logger.warning("CUDA not available, falling back to ONNX Runtime")
            self._init_onnx_fallback()
            return

        try:
            # Try to load cached engine
            if not self.force_rebuild and self.engine_path.exists():
                logger.info(f"Loading cached TensorRT engine: {self.engine_path}")
                self._load_engine()
            else:
                # Build new engine
                logger.info(f"Building TensorRT engine (this may take a minute)...")
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._build_engine()
                self._save_engine()

            # Setup execution context
            self._setup_context()

        except Exception as e:
            logger.error(f"TensorRT initialization failed: {e}")
            if self.fallback_to_onnx:
                logger.warning("Falling back to ONNX Runtime")
                self._init_onnx_fallback()
            else:
                raise

    def _init_onnx_fallback(self):
        """Initialize ONNX Runtime as fallback."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Neither TensorRT nor ONNX Runtime available")

        self.using_fallback = True

        # Try CUDA provider first
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available]

        self.onnx_session = ort.InferenceSession(
            str(self.onnx_path),
            providers=providers
        )

        # Store input/output names
        self.input_names = [inp.name for inp in self.onnx_session.get_inputs()]
        self.output_names = [out.name for out in self.onnx_session.get_outputs()]

        logger.info(f"ONNX Runtime initialized with provider: {self.onnx_session.get_providers()[0]}")

    def _build_engine(self):
        """Build TensorRT engine from ONNX model."""
        trt = TRT_MODULE

        # Create logger with warning level
        trt_logger = trt.Logger(trt.Logger.WARNING)

        # Create builder and network
        builder = trt.Builder(trt_logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, trt_logger)

        # Parse ONNX model
        with open(self.onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                errors = [parser.get_error(i) for i in range(parser.num_errors)]
                raise RuntimeError(f"ONNX parsing failed:\n" + "\n".join(str(e) for e in errors))

        logger.info(f"Parsed ONNX model: {network.num_inputs} inputs, {network.num_outputs} outputs, {network.num_layers} layers")

        # Create builder config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(self.config.workspace_gb * (1 << 30))
        )

        # Set precision flags
        if self.config.fp16 and builder.platform_has_fast_fp16:
            logger.info("Enabling FP16 precision")
            config.set_flag(trt.BuilderFlag.FP16)

        if self.config.int8 and builder.platform_has_fast_int8:
            logger.info("Enabling INT8 precision")
            config.set_flag(trt.BuilderFlag.INT8)
            # Note: INT8 requires calibration, which should be set up separately

        if self.config.strict_types:
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # Setup dynamic shape optimization profile
        input_tensor = network.get_input(0)
        input_name = input_tensor.name

        profile = builder.create_optimization_profile()

        min_shape = (
            self.config.min_batch, 3,
            self.config.min_resolution, self.config.min_resolution
        )
        opt_shape = (
            self.config.opt_batch, 3,
            self.config.opt_resolution, self.config.opt_resolution
        )
        max_shape = (
            self.config.max_batch, 3,
            self.config.max_resolution, self.config.max_resolution
        )

        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        logger.info(f"Dynamic shapes: min={min_shape}, opt={opt_shape}, max={max_shape}")

        # Build engine
        self.serialized_engine = builder.build_serialized_network(network, config)

        if self.serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Deserialize
        runtime = trt.Runtime(trt_logger)
        self.engine = runtime.deserialize_cuda_engine(self.serialized_engine)

        logger.info(f"TensorRT engine built successfully ({len(self.serialized_engine) / (1024*1024):.1f} MB)")

    def _load_engine(self):
        """Load cached TensorRT engine from disk."""
        trt = TRT_MODULE

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(self.engine_path, 'rb') as f:
            self.serialized_engine = f.read()

        self.engine = runtime.deserialize_cuda_engine(self.serialized_engine)

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine from {self.engine_path}")

        logger.info(f"Loaded cached engine ({len(self.serialized_engine) / (1024*1024):.1f} MB)")

    def _save_engine(self):
        """Save serialized engine to cache."""
        with open(self.engine_path, 'wb') as f:
            f.write(self.serialized_engine)

        logger.info(f"Saved TensorRT engine to: {self.engine_path}")

        # Also save metadata
        metadata_path = self.engine_path.with_suffix('.json')
        import json
        metadata = {
            'onnx_path': str(self.onnx_path),
            'cache_key': self.cache_key,
            'gpu': GPU_NAME,
            'trt_version': TRT_VERSION,
            'config': {
                'fp16': self.config.fp16,
                'int8': self.config.int8,
                'opt_batch': self.config.opt_batch,
                'opt_resolution': self.config.opt_resolution,
            }
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _setup_context(self):
        """Setup execution context and allocate buffers."""
        import torch

        trt = TRT_MODULE

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Create CUDA stream
        self.stream = torch.cuda.Stream()

        # Get input/output names and shapes
        self.input_names = []
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        logger.info(f"Engine inputs: {self.input_names}")
        logger.info(f"Engine outputs: {self.output_names}")

    def _allocate_buffers(self, input_shape: Tuple[int, ...]):
        """Allocate input/output buffers for given input shape."""
        import torch

        trt = TRT_MODULE

        # Set input shape
        for name in self.input_names:
            self.context.set_input_shape(name, input_shape)

        # Allocate buffers
        for name in self.input_names + self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            # Convert numpy dtype to torch dtype
            if dtype == np.float32:
                torch_dtype = torch.float32
            elif dtype == np.float16:
                torch_dtype = torch.float16
            elif dtype == np.int32:
                torch_dtype = torch.int32
            elif dtype == np.int64:
                torch_dtype = torch.int64
            else:
                torch_dtype = torch.float32

            self.buffers[name] = torch.zeros(tuple(shape), dtype=torch_dtype, device='cuda')
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())

    def __call__(
        self,
        x: Any,  # torch.Tensor or numpy array
        **kwargs
    ) -> Union[Any, Dict[str, Any]]:
        """
        Run inference on input tensor.

        Args:
            x: Input tensor (torch.Tensor or numpy array)
            **kwargs: Additional inputs for multi-input models

        Returns:
            Output tensor(s) - single tensor if one output, dict if multiple
        """
        if self.using_fallback:
            return self._run_onnx(x, **kwargs)
        else:
            return self._run_tensorrt(x, **kwargs)

    def _run_tensorrt(self, x: Any, **kwargs) -> Union[Any, Dict[str, Any]]:
        """Run TensorRT inference."""
        import torch

        # Convert to torch tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # Ensure CUDA tensor
        if not x.is_cuda:
            x = x.cuda()

        # Ensure contiguous
        x = x.contiguous()

        # Check if we need to reallocate buffers
        input_shape = tuple(x.shape)
        if (self.input_names[0] not in self.buffers or
            tuple(self.buffers[self.input_names[0]].shape) != input_shape):
            self._allocate_buffers(input_shape)

        # Copy input
        self.buffers[self.input_names[0]].copy_(x)

        # Handle additional inputs
        for name, tensor in kwargs.items():
            if name in self.buffers:
                if isinstance(tensor, np.ndarray):
                    tensor = torch.from_numpy(tensor)
                if not tensor.is_cuda:
                    tensor = tensor.cuda()
                self.buffers[name].copy_(tensor)

        # Execute
        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)

        self.stream.synchronize()

        # Return outputs
        if len(self.output_names) == 1:
            return self.buffers[self.output_names[0]].clone()
        else:
            return {name: self.buffers[name].clone() for name in self.output_names}

    def _run_onnx(self, x: Any, **kwargs) -> Union[Any, Dict[str, Any]]:
        """Run ONNX Runtime inference (fallback)."""
        import torch

        # Convert to numpy if needed
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x

        # Prepare inputs
        inputs = {self.input_names[0]: x_np}
        for name, tensor in kwargs.items():
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.cpu().numpy()
            inputs[name] = tensor

        # Run inference
        outputs = self.onnx_session.run(self.output_names, inputs)

        # Convert to torch tensors
        outputs = [torch.from_numpy(out).cuda() for out in outputs]

        if len(outputs) == 1:
            return outputs[0]
        else:
            return dict(zip(self.output_names, outputs))

    def warmup(self, input_shape: Tuple[int, ...] = (1, 3, 1024, 1024), iterations: int = 3):
        """
        Warmup the engine with dummy inference runs.

        Args:
            input_shape: Shape of dummy input
            iterations: Number of warmup iterations
        """
        import torch

        logger.info(f"Warming up engine with {iterations} iterations...")

        dummy = torch.randn(input_shape, device='cuda')
        for _ in range(iterations):
            _ = self(dummy)

        torch.cuda.synchronize()
        logger.info("Warmup complete")

    def benchmark(
        self,
        input_shape: Tuple[int, ...] = (1, 3, 1024, 1024),
        iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark inference latency.

        Args:
            input_shape: Shape of benchmark input
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            Dictionary with latency statistics
        """
        import torch

        dummy = torch.randn(input_shape, device='cuda')

        # Warmup
        for _ in range(warmup_iterations):
            _ = self(dummy)
        torch.cuda.synchronize()

        # Benchmark
        latencies = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = self(dummy)
            end.record()

            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))

        return {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'throughput_fps': 1000.0 / np.mean(latencies),
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False

    def cleanup(self):
        """Release TensorRT resources."""
        self.buffers.clear()
        self.context = None
        self.engine = None
        self.stream = None

        if self.onnx_session is not None:
            self.onnx_session = None

        logger.info("TensorRT engine resources released")

    @property
    def is_tensorrt(self) -> bool:
        """Check if using TensorRT (vs fallback)."""
        return not self.using_fallback

    @property
    def device(self) -> str:
        """Get device string."""
        return 'cuda' if (TRT_AVAILABLE and CUDA_AVAILABLE) or \
               (self.onnx_session and 'CUDA' in self.onnx_session.get_providers()[0]) \
               else 'cpu'


def load_tensorrt_engine(
    onnx_path: Union[str, Path],
    cache_dir: str = '.trt_cache',
    **kwargs
) -> TensorRTEngine:
    """
    Convenience function to load or build a TensorRT engine.

    Args:
        onnx_path: Path to ONNX model
        cache_dir: Cache directory for engines
        **kwargs: Additional arguments for TensorRTEngine

    Returns:
        Initialized TensorRTEngine
    """
    return TensorRTEngine(onnx_path, cache_dir=cache_dir, **kwargs)


__all__ = [
    'TensorRTEngine',
    'TensorRTEngineConfig',
    'load_tensorrt_engine',
    'TRT_AVAILABLE',
    'CUDA_AVAILABLE',
]
