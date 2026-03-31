"""
Compiled Model Inference with torch.compile and OpenVINO Support

This module provides model compilation utilities for optimized inference:
- torch.compile with inductor backend for PyTorch 2.0+ optimization
- OpenVINO conversion for Intel hardware acceleration
- Warmup logic for compilation and kernel fusion

Key benefits:
- torch.compile: 1.5-3x speedup through kernel fusion and optimization
- OpenVINO: Optimized inference on Intel CPUs, iGPUs, and discrete GPUs

Usage:
    # PyTorch compilation
    compiled_model = compile_pytorch(model, mode='reduce-overhead')
    output = compiled_model(input_tensor)

    # OpenVINO conversion
    ov_model = convert_to_openvino('model.onnx', output_dir='./openvino_model')
    output = run_openvino_inference(ov_model, input_array)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def _check_torch_compile() -> Tuple[bool, Optional[str]]:
    """Check if torch.compile is available (PyTorch 2.0+)."""
    try:
        import torch
        if hasattr(torch, 'compile'):
            return True, torch.__version__
        return False, f"torch.compile requires PyTorch 2.0+, found {torch.__version__}"
    except ImportError:
        return False, "PyTorch not installed"


def _check_openvino() -> Tuple[bool, Optional[str]]:
    """Check if OpenVINO is available."""
    try:
        from openvino.runtime import Core
        import openvino
        return True, openvino.__version__
    except ImportError:
        return False, "OpenVINO not installed. Install with: pip install openvino"


TORCH_COMPILE_AVAILABLE, TORCH_VERSION = _check_torch_compile()
OPENVINO_AVAILABLE, OPENVINO_VERSION = _check_openvino()


@dataclass
class CompileConfig:
    """Configuration for torch.compile."""
    mode: str = 'reduce-overhead'  # 'default', 'reduce-overhead', 'max-autotune'
    backend: str = 'inductor'
    fullgraph: bool = False
    dynamic: bool = False
    disable: bool = False


@dataclass
class OpenVINOConfig:
    """Configuration for OpenVINO conversion."""
    compress_to_fp16: bool = True
    performance_hint: str = 'LATENCY'  # 'LATENCY', 'THROUGHPUT', 'CUMULATIVE_THROUGHPUT'
    num_streams: str = 'AUTO'
    device: str = 'CPU'  # 'CPU', 'GPU', 'AUTO'
    cache_dir: Optional[str] = None


def compile_pytorch(
    model: Any,  # nn.Module
    mode: str = 'reduce-overhead',
    backend: str = 'inductor',
    warmup_shape: Tuple[int, ...] = (1, 3, 1024, 1024),
    warmup_iterations: int = 3,
    device: str = 'cuda',
    config: Optional[CompileConfig] = None,
) -> Any:
    """
    Compile PyTorch model with torch.compile for kernel fusion and optimization.

    torch.compile provides significant speedups through:
    - Kernel fusion: Combining multiple operations into single kernels
    - Memory optimization: Reducing memory bandwidth usage
    - Graph-level optimization: Analyzing and optimizing entire computation graphs

    Args:
        model: PyTorch nn.Module to compile
        mode: Compilation mode:
            - 'default': Good balance of compile time and speedup
            - 'reduce-overhead': Minimize CPU overhead, best for inference
            - 'max-autotune': Maximum speedup, longer compile time
        backend: Compiler backend ('inductor' recommended for most cases)
        warmup_shape: Input shape for warmup iterations
        warmup_iterations: Number of warmup runs to trigger compilation
        device: Device for warmup ('cuda' or 'cpu')
        config: Optional detailed configuration

    Returns:
        Compiled model callable

    Example:
        >>> model = load_model().cuda().eval()
        >>> compiled = compile_pytorch(model, mode='reduce-overhead')
        >>> output = compiled(input_tensor)

    Note:
        The first inference after compilation may be slow as the compiler
        generates optimized kernels. Use warmup_iterations > 0 to ensure
        compilation completes before timing.
    """
    if not TORCH_COMPILE_AVAILABLE:
        logger.warning(f"torch.compile not available: {TORCH_VERSION}. Returning original model.")
        return model

    import torch

    # Merge config if provided
    if config is not None:
        mode = config.mode
        backend = config.backend

        if config.disable:
            logger.info("torch.compile disabled by configuration")
            return model

    logger.info(f"Compiling model with torch.compile (mode={mode}, backend={backend})")

    # Ensure model is in eval mode for inference
    if hasattr(model, 'eval'):
        model.eval()

    # Compile the model
    try:
        compile_kwargs = {
            'mode': mode,
            'backend': backend,
        }

        if config is not None:
            compile_kwargs['fullgraph'] = config.fullgraph
            compile_kwargs['dynamic'] = config.dynamic

        compiled = torch.compile(model, **compile_kwargs)

    except Exception as e:
        logger.error(f"torch.compile failed: {e}. Returning original model.")
        return model

    # Warmup to trigger compilation
    if warmup_iterations > 0 and warmup_shape is not None:
        logger.info(f"Running {warmup_iterations} warmup iterations to trigger compilation...")

        try:
            dummy = torch.randn(warmup_shape, device=device)

            with torch.no_grad():
                for i in range(warmup_iterations):
                    _ = compiled(dummy)
                    if device == 'cuda':
                        torch.cuda.synchronize()

            logger.info("Warmup complete, model compiled")

        except Exception as e:
            logger.warning(f"Warmup failed: {e}. Model may compile on first real inference.")

    return compiled


def convert_to_openvino(
    onnx_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[OpenVINOConfig] = None,
    save_model: bool = True,
) -> Any:
    """
    Convert ONNX model to OpenVINO IR format for Intel hardware optimization.

    OpenVINO provides optimized inference for:
    - Intel CPUs (with AVX-512, VNNI acceleration)
    - Intel integrated GPUs
    - Intel discrete GPUs (Arc series)
    - Intel VPUs and FPGAs

    Args:
        onnx_path: Path to ONNX model file
        output_dir: Directory to save OpenVINO IR files (optional)
        config: OpenVINO configuration options
        save_model: Whether to save the converted model to disk

    Returns:
        Compiled OpenVINO model ready for inference

    Example:
        >>> ov_model = convert_to_openvino('model.onnx', output_dir='./ov_model')
        >>> # Run inference
        >>> result = ov_model(input_array)

    Note:
        OpenVINO IR consists of two files: .xml (model structure) and .bin (weights).
        These are saved to output_dir if specified.
    """
    if not OPENVINO_AVAILABLE:
        raise ImportError(f"OpenVINO not available: {OPENVINO_VERSION}")

    from openvino.runtime import Core
    from openvino import convert_model, save_model as ov_save_model

    config = config or OpenVINOConfig()
    onnx_path = Path(onnx_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    logger.info(f"Converting ONNX model to OpenVINO IR: {onnx_path}")

    # Convert ONNX to OpenVINO IR
    ov_model = convert_model(str(onnx_path), compress_to_fp16=config.compress_to_fp16)

    logger.info(f"Model converted (FP16 compression: {config.compress_to_fp16})")

    # Save model if requested
    if save_model and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"{onnx_path.stem}.xml"
        ov_save_model(ov_model, str(model_path))
        logger.info(f"Saved OpenVINO IR to: {model_path}")

    # Compile model for inference
    core = Core()

    # Set cache directory for compiled models
    if config.cache_dir:
        core.set_property({'CACHE_DIR': config.cache_dir})

    # Configure device properties
    device_config = {
        'PERFORMANCE_HINT': config.performance_hint,
    }

    if config.num_streams != 'AUTO':
        device_config['NUM_STREAMS'] = config.num_streams

    compiled_model = core.compile_model(ov_model, config.device, device_config)

    logger.info(f"OpenVINO model compiled for device: {config.device}")

    return compiled_model


def load_openvino_model(
    model_path: Union[str, Path],
    config: Optional[OpenVINOConfig] = None,
) -> Any:
    """
    Load a pre-converted OpenVINO IR model.

    Args:
        model_path: Path to .xml file of OpenVINO IR model
        config: OpenVINO configuration options

    Returns:
        Compiled OpenVINO model ready for inference
    """
    if not OPENVINO_AVAILABLE:
        raise ImportError(f"OpenVINO not available: {OPENVINO_VERSION}")

    from openvino.runtime import Core

    config = config or OpenVINOConfig()
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"OpenVINO model not found: {model_path}")

    logger.info(f"Loading OpenVINO model: {model_path}")

    core = Core()

    # Set cache directory
    if config.cache_dir:
        core.set_property({'CACHE_DIR': config.cache_dir})

    # Read and compile model
    model = core.read_model(str(model_path))

    device_config = {
        'PERFORMANCE_HINT': config.performance_hint,
    }

    if config.num_streams != 'AUTO':
        device_config['NUM_STREAMS'] = config.num_streams

    compiled_model = core.compile_model(model, config.device, device_config)

    logger.info(f"OpenVINO model loaded and compiled for: {config.device}")

    return compiled_model


def run_openvino_inference(
    compiled_model: Any,
    input_data: np.ndarray,
    output_names: Optional[List[str]] = None,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Run inference on compiled OpenVINO model.

    Args:
        compiled_model: Compiled OpenVINO model
        input_data: Input numpy array
        output_names: Optional list of output names to return

    Returns:
        Single numpy array if one output, dict of arrays if multiple
    """
    # Create infer request
    infer_request = compiled_model.create_infer_request()

    # Run inference
    infer_request.infer({0: input_data})

    # Get outputs
    outputs = {}
    for i, output in enumerate(compiled_model.outputs):
        name = output.get_any_name()
        outputs[name] = infer_request.get_output_tensor(i).data.copy()

    # Filter outputs if names specified
    if output_names is not None:
        outputs = {k: v for k, v in outputs.items() if k in output_names}

    # Return single array if only one output
    if len(outputs) == 1:
        return list(outputs.values())[0]

    return outputs


class CompiledModelWrapper:
    """
    Unified wrapper for compiled models (torch.compile or OpenVINO).

    Provides a consistent interface regardless of the underlying compilation
    method, making it easy to switch between backends.

    Args:
        model: Original PyTorch model or path to ONNX model
        backend: Compilation backend ('torch', 'openvino', or 'auto')
        device: Target device
        **kwargs: Additional arguments for compilation

    Example:
        >>> wrapper = CompiledModelWrapper(model, backend='torch')
        >>> output = wrapper(input_tensor)
    """

    def __init__(
        self,
        model: Any,
        backend: str = 'auto',
        device: str = 'cuda',
        **kwargs,
    ):
        self.backend = backend
        self.device = device
        self.compiled_model = None
        self._is_openvino = False

        # Auto-select backend
        if backend == 'auto':
            if device in ('cuda', 'gpu') and TORCH_COMPILE_AVAILABLE:
                backend = 'torch'
            elif OPENVINO_AVAILABLE:
                backend = 'openvino'
            else:
                backend = 'none'
                logger.warning("No compilation backend available")

        # Compile based on backend
        if backend == 'torch':
            self.compiled_model = compile_pytorch(model, device=device, **kwargs)
        elif backend == 'openvino':
            if isinstance(model, (str, Path)):
                self.compiled_model = convert_to_openvino(model, **kwargs)
            else:
                raise ValueError("OpenVINO backend requires ONNX model path")
            self._is_openvino = True
        else:
            self.compiled_model = model

        self.backend = backend

    def __call__(self, x: Any) -> Any:
        """Run inference on compiled model."""
        if self._is_openvino:
            # OpenVINO expects numpy input
            if hasattr(x, 'cpu'):
                x = x.cpu().numpy()
            return run_openvino_inference(self.compiled_model, x)
        else:
            return self.compiled_model(x)

    @property
    def is_compiled(self) -> bool:
        """Check if model is compiled."""
        return self.backend in ('torch', 'openvino')


def benchmark_compilation(
    model: Any,
    input_shape: Tuple[int, ...] = (1, 3, 1024, 1024),
    iterations: int = 100,
    warmup: int = 10,
    device: str = 'cuda',
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark model with and without torch.compile.

    Args:
        model: PyTorch model to benchmark
        input_shape: Input shape for benchmarking
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations
        device: Device for benchmarking

    Returns:
        Dictionary with 'baseline' and 'compiled' benchmark results
    """
    import torch

    results = {}

    # Ensure model is in eval mode
    if hasattr(model, 'eval'):
        model.eval()

    if device == 'cuda':
        model = model.cuda()

    dummy = torch.randn(input_shape, device=device)

    # Benchmark baseline
    logger.info("Benchmarking baseline inference...")

    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy)
    if device == 'cuda':
        torch.cuda.synchronize()

    latencies = []
    for _ in range(iterations):
        if device == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            import time
            start_time = time.perf_counter()

        with torch.no_grad():
            _ = model(dummy)

        if device == 'cuda':
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))
        else:
            latencies.append((time.perf_counter() - start_time) * 1000)

    results['baseline'] = {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'throughput_fps': 1000.0 / np.mean(latencies),
    }

    # Benchmark compiled
    if TORCH_COMPILE_AVAILABLE:
        logger.info("Benchmarking compiled inference...")

        compiled = compile_pytorch(
            model,
            mode='reduce-overhead',
            warmup_shape=input_shape,
            warmup_iterations=warmup,
            device=device,
        )

        latencies = []
        for _ in range(iterations):
            if device == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            else:
                start_time = time.perf_counter()

            with torch.no_grad():
                _ = compiled(dummy)

            if device == 'cuda':
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
            else:
                latencies.append((time.perf_counter() - start_time) * 1000)

        results['compiled'] = {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'throughput_fps': 1000.0 / np.mean(latencies),
        }

        speedup = results['baseline']['mean_ms'] / results['compiled']['mean_ms']
        results['speedup'] = speedup
        logger.info(f"torch.compile speedup: {speedup:.2f}x")
    else:
        logger.warning("torch.compile not available for benchmarking")
        results['compiled'] = None
        results['speedup'] = 1.0

    return results


__all__ = [
    # PyTorch compilation
    'compile_pytorch',
    'CompileConfig',
    'TORCH_COMPILE_AVAILABLE',
    # OpenVINO
    'convert_to_openvino',
    'load_openvino_model',
    'run_openvino_inference',
    'OpenVINOConfig',
    'OPENVINO_AVAILABLE',
    # Wrapper
    'CompiledModelWrapper',
    # Benchmarking
    'benchmark_compilation',
]
