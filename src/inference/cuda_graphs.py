"""
CUDA Graphs for Low-Latency Inference

CUDA Graphs capture a sequence of GPU operations and replay them with minimal
CPU overhead. This is particularly effective for:
- Small batch sizes where CPU launch overhead dominates
- Repeated inference with fixed input shapes
- Real-time applications requiring consistent latency

Key benefits:
- 10-50% latency reduction by eliminating CPU kernel launch overhead
- More consistent latency (lower jitter)
- Reduced CPU utilization

Limitations:
- Input shape must be fixed after graph capture
- Cannot use dynamic control flow during captured region
- Memory addresses are fixed at capture time

Usage:
    # Basic usage with PyTorch model
    graph_inference = CUDAGraphInference(model, input_shape=(1, 3, 1024, 1024))
    output = graph_inference(input_tensor)

    # With warmup and multiple output support
    graph_inference = CUDAGraphInference(
        model,
        input_shape=(1, 3, 1024, 1024),
        warmup_iterations=5,
        capture_stream=True
    )
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def _check_cuda_available() -> Tuple[bool, Optional[str]]:
    """Check if CUDA is available and supports graph capture."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"

        # Check CUDA capability (graphs require CC 7.0+)
        major, minor = torch.cuda.get_device_capability()
        if major < 7:
            return False, f"CUDA Graphs require compute capability 7.0+, found {major}.{minor}"

        return True, None
    except ImportError:
        return False, "PyTorch not installed"


CUDA_GRAPHS_AVAILABLE, CUDA_GRAPHS_ERROR = _check_cuda_available()


@dataclass
class CUDAGraphConfig:
    """Configuration for CUDA Graph capture."""
    warmup_iterations: int = 3
    capture_stream: bool = True
    pool_id: Optional[int] = None  # Memory pool ID for graph capture
    debug_mode: bool = False


class CUDAGraphInference:
    """
    CUDA Graphs wrapper for eliminating CPU launch overhead.

    This class captures a model's forward pass into a CUDA graph and replays it
    for subsequent inferences. This eliminates the CPU overhead of launching
    individual kernels, resulting in lower and more consistent latency.

    Args:
        model: PyTorch model or callable
        input_shape: Shape of input tensor (must be fixed)
        config: Configuration options
        device: CUDA device to use (default: 'cuda')

    Example:
        >>> model = load_model().cuda().eval()
        >>> graph_inf = CUDAGraphInference(model, input_shape=(1, 3, 1024, 1024))
        >>> output = graph_inf(torch.randn(1, 3, 1024, 1024, device='cuda'))

    Note:
        The input shape is fixed after graph capture. If you need different
        input shapes, create multiple CUDAGraphInference instances or use
        CUDAGraphPool for common shapes.
    """

    def __init__(
        self,
        model: Callable,
        input_shape: Tuple[int, ...],
        config: Optional[CUDAGraphConfig] = None,
        device: str = 'cuda',
    ):
        if not CUDA_GRAPHS_AVAILABLE:
            raise RuntimeError(f"CUDA Graphs not available: {CUDA_GRAPHS_ERROR}")

        import torch

        self.model = model
        self.input_shape = input_shape
        self.config = config or CUDAGraphConfig()
        self.device = device

        # Graph state
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_input: Optional[torch.Tensor] = None
        self.static_output: Optional[Any] = None  # Can be tensor, tuple, or dict
        self.stream: Optional[torch.cuda.Stream] = None

        # Track output structure
        self._output_is_dict = False
        self._output_is_tuple = False
        self._output_keys: Optional[List[str]] = None

        # Capture the graph
        self._capture_graph()

    def _capture_graph(self):
        """Capture model forward pass into CUDA graph."""
        import torch

        logger.info(f"Capturing CUDA graph for input shape: {self.input_shape}")

        # Ensure model is in eval mode
        if hasattr(self.model, 'eval'):
            self.model.eval()

        # Create static input buffer
        self.static_input = torch.zeros(
            self.input_shape,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False
        )

        # Create dedicated stream for graph operations
        if self.config.capture_stream:
            self.stream = torch.cuda.Stream()

        # Warmup runs (outside graph capture)
        # This ensures all lazy initialization is complete
        logger.info(f"Running {self.config.warmup_iterations} warmup iterations...")

        with torch.no_grad():
            for i in range(self.config.warmup_iterations):
                warmup_output = self.model(self.static_input)
                if self.config.debug_mode:
                    logger.debug(f"Warmup {i+1}: output type = {type(warmup_output)}")

        # Detect output structure from warmup
        self._analyze_output_structure(warmup_output)

        # Synchronize before capture
        torch.cuda.synchronize()

        # Capture graph
        self.graph = torch.cuda.CUDAGraph()

        # Use memory pool if specified
        capture_kwargs = {}
        if self.config.pool_id is not None:
            capture_kwargs['pool'] = self.config.pool_id

        with torch.no_grad():
            with torch.cuda.graph(self.graph, **capture_kwargs):
                self.static_output = self.model(self.static_input)

        # Synchronize after capture
        torch.cuda.synchronize()

        logger.info("CUDA graph captured successfully")

        # Print memory stats
        if self.config.debug_mode:
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            logger.debug(f"GPU memory: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")

    def _analyze_output_structure(self, output: Any):
        """Analyze model output structure for proper cloning."""
        import torch

        if isinstance(output, dict):
            self._output_is_dict = True
            self._output_keys = list(output.keys())
        elif isinstance(output, (tuple, list)):
            self._output_is_tuple = True
        # Single tensor is the default case

    def __call__(self, x: Any) -> Any:
        """
        Run inference using captured CUDA graph.

        Args:
            x: Input tensor (must match captured input_shape)

        Returns:
            Model output (cloned from static buffer)

        Raises:
            ValueError: If input shape doesn't match captured shape
        """
        import torch

        # Validate input shape
        if isinstance(x, torch.Tensor):
            if x.shape != self.input_shape:
                raise ValueError(
                    f"Input shape {tuple(x.shape)} doesn't match captured shape {self.input_shape}. "
                    f"CUDA Graphs require fixed input shapes."
                )
        elif isinstance(x, np.ndarray):
            if x.shape != self.input_shape:
                raise ValueError(
                    f"Input shape {x.shape} doesn't match captured shape {self.input_shape}. "
                    f"CUDA Graphs require fixed input shapes."
                )
            x = torch.from_numpy(x)

        # Ensure input is on correct device and contiguous
        if not x.is_cuda:
            x = x.to(self.device)
        x = x.contiguous()

        # Copy input to static buffer
        self.static_input.copy_(x)

        # Replay graph
        self.graph.replay()

        # Clone and return output (static_output buffer is reused)
        return self._clone_output()

    def _clone_output(self) -> Any:
        """Clone static output buffer."""
        import torch

        if self._output_is_dict:
            return {k: v.clone() for k, v in self.static_output.items()}
        elif self._output_is_tuple:
            return tuple(t.clone() if isinstance(t, torch.Tensor) else t
                        for t in self.static_output)
        else:
            return self.static_output.clone()

    def warmup(self, iterations: int = 3):
        """
        Run warmup inferences after graph capture.

        Useful for ensuring consistent timing in benchmarks.
        """
        import torch

        dummy = torch.randn(self.input_shape, device=self.device)
        for _ in range(iterations):
            _ = self(dummy)
        torch.cuda.synchronize()

    def benchmark(
        self,
        iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark graph replay latency.

        Args:
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            Dictionary with latency statistics
        """
        import torch

        # Warmup
        self.warmup(warmup_iterations)

        # Benchmark
        latencies = []
        dummy = torch.randn(self.input_shape, device=self.device)

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

    def reset(self):
        """Reset and recapture the graph."""
        import torch

        self.graph = None
        self.static_output = None
        torch.cuda.synchronize()

        self._capture_graph()

    @property
    def is_captured(self) -> bool:
        """Check if graph is captured."""
        return self.graph is not None


class CUDAGraphPool:
    """
    Pool of CUDA graphs for multiple input shapes.

    Useful when you have a known set of common input shapes and want to
    benefit from CUDA Graphs for all of them.

    Args:
        model: PyTorch model or callable
        shapes: List of input shapes to pre-capture
        config: Configuration options

    Example:
        >>> pool = CUDAGraphPool(model, shapes=[
        ...     (1, 3, 512, 512),
        ...     (1, 3, 1024, 1024),
        ...     (1, 3, 2048, 2048),
        ... ])
        >>> output = pool(input_tensor)  # Automatically selects correct graph
    """

    def __init__(
        self,
        model: Callable,
        shapes: List[Tuple[int, ...]],
        config: Optional[CUDAGraphConfig] = None,
    ):
        if not CUDA_GRAPHS_AVAILABLE:
            raise RuntimeError(f"CUDA Graphs not available: {CUDA_GRAPHS_ERROR}")

        self.model = model
        self.config = config or CUDAGraphConfig()
        self.graphs: Dict[Tuple[int, ...], CUDAGraphInference] = {}

        # Pre-capture graphs for all shapes
        logger.info(f"Pre-capturing CUDA graphs for {len(shapes)} shapes...")

        for shape in shapes:
            self.graphs[shape] = CUDAGraphInference(
                model=model,
                input_shape=shape,
                config=self.config,
            )

        logger.info(f"CUDAGraphPool ready with {len(self.graphs)} cached graphs")

    def __call__(self, x: Any) -> Any:
        """
        Run inference, using cached graph if input shape matches.

        Args:
            x: Input tensor

        Returns:
            Model output

        Raises:
            KeyError: If input shape not in pool (and fallback is disabled)
        """
        import torch

        if isinstance(x, np.ndarray):
            shape = x.shape
        else:
            shape = tuple(x.shape)

        if shape in self.graphs:
            return self.graphs[shape](x)
        else:
            # Fallback to regular inference
            logger.warning(
                f"Input shape {shape} not in graph pool. "
                f"Available shapes: {list(self.graphs.keys())}. "
                f"Falling back to regular inference."
            )
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).cuda()
            elif not x.is_cuda:
                x = x.cuda()

            with torch.no_grad():
                return self.model(x)

    def add_shape(self, shape: Tuple[int, ...]):
        """Add and capture a new shape to the pool."""
        if shape not in self.graphs:
            self.graphs[shape] = CUDAGraphInference(
                model=self.model,
                input_shape=shape,
                config=self.config,
            )
            logger.info(f"Added shape {shape} to graph pool")

    def has_shape(self, shape: Tuple[int, ...]) -> bool:
        """Check if shape is in pool."""
        return shape in self.graphs

    @property
    def available_shapes(self) -> List[Tuple[int, ...]]:
        """List all available shapes."""
        return list(self.graphs.keys())


class CUDAGraphWrapper:
    """
    Automatic CUDA graph wrapper with lazy capture and fallback.

    This wrapper automatically captures CUDA graphs on first use and
    provides transparent fallback when graphs cannot be used.

    Args:
        model: PyTorch model or callable
        enable_graphs: Whether to use CUDA graphs (default: auto-detect)
        config: Configuration options

    Example:
        >>> wrapper = CUDAGraphWrapper(model)
        >>> output = wrapper(input_tensor)  # First call captures graph
        >>> output = wrapper(input_tensor)  # Subsequent calls replay graph
    """

    def __init__(
        self,
        model: Callable,
        enable_graphs: Optional[bool] = None,
        config: Optional[CUDAGraphConfig] = None,
    ):
        self.model = model
        self.config = config or CUDAGraphConfig()

        # Auto-detect if not specified
        if enable_graphs is None:
            enable_graphs = CUDA_GRAPHS_AVAILABLE

        self.enable_graphs = enable_graphs
        self.graphs: Dict[Tuple[int, ...], CUDAGraphInference] = {}

        if not self.enable_graphs and CUDA_GRAPHS_AVAILABLE:
            logger.info("CUDA Graphs disabled by configuration")
        elif not CUDA_GRAPHS_AVAILABLE:
            logger.warning(f"CUDA Graphs unavailable: {CUDA_GRAPHS_ERROR}")

    def __call__(self, x: Any) -> Any:
        """
        Run inference, capturing graph on first use for each shape.

        Args:
            x: Input tensor

        Returns:
            Model output
        """
        import torch

        if isinstance(x, np.ndarray):
            shape = x.shape
            x = torch.from_numpy(x)
        else:
            shape = tuple(x.shape)

        if not x.is_cuda:
            x = x.cuda()

        # Use graphs if enabled
        if self.enable_graphs:
            if shape not in self.graphs:
                logger.info(f"Capturing CUDA graph for new shape: {shape}")
                try:
                    self.graphs[shape] = CUDAGraphInference(
                        model=self.model,
                        input_shape=shape,
                        config=self.config,
                    )
                except Exception as e:
                    logger.warning(f"Failed to capture CUDA graph: {e}. Using regular inference.")
                    with torch.no_grad():
                        return self.model(x)

            return self.graphs[shape](x)

        # Fallback to regular inference
        with torch.no_grad():
            return self.model(x)

    def clear_graphs(self):
        """Clear all cached graphs."""
        self.graphs.clear()
        logger.info("Cleared all cached CUDA graphs")


def benchmark_cuda_graphs(
    model: Callable,
    input_shape: Tuple[int, ...] = (1, 3, 1024, 1024),
    iterations: int = 100,
    warmup: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark model with and without CUDA graphs.

    Args:
        model: PyTorch model to benchmark
        input_shape: Input shape for benchmarking
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations

    Returns:
        Dictionary with 'regular' and 'cuda_graphs' benchmark results
    """
    import torch

    results = {}

    # Ensure model is in eval mode
    if hasattr(model, 'eval'):
        model.eval()

    # Benchmark regular inference
    logger.info("Benchmarking regular inference...")
    dummy = torch.randn(input_shape, device='cuda')

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy)
    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            _ = model(dummy)
        end.record()

        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))

    results['regular'] = {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'throughput_fps': 1000.0 / np.mean(latencies),
    }

    # Benchmark with CUDA graphs
    if CUDA_GRAPHS_AVAILABLE:
        logger.info("Benchmarking CUDA graphs inference...")
        graph_inf = CUDAGraphInference(model, input_shape)
        results['cuda_graphs'] = graph_inf.benchmark(iterations, warmup)

        # Calculate speedup
        speedup = results['regular']['mean_ms'] / results['cuda_graphs']['mean_ms']
        results['speedup'] = speedup
        logger.info(f"CUDA Graphs speedup: {speedup:.2f}x")
    else:
        logger.warning("CUDA Graphs not available for benchmarking")
        results['cuda_graphs'] = None
        results['speedup'] = 1.0

    return results


__all__ = [
    'CUDAGraphInference',
    'CUDAGraphPool',
    'CUDAGraphWrapper',
    'CUDAGraphConfig',
    'benchmark_cuda_graphs',
    'CUDA_GRAPHS_AVAILABLE',
]
