"""Inference package for comic text detection."""

from .detector import TextDetector, BatchTextDetector, DetectionResult
from .async_pipeline import AsyncInferencePipeline, StreamingInferencePipeline, create_pipeline
from .output_cache import PerceptualHashCache, ContentHashCache, create_cache

# TensorRT backend (with fallback support)
from .tensorrt_backend import (
    TensorRTEngine,
    TensorRTEngineConfig,
    load_tensorrt_engine,
    TRT_AVAILABLE,
)

# CUDA Graphs for low-latency inference
from .cuda_graphs import (
    CUDAGraphInference,
    CUDAGraphPool,
    CUDAGraphWrapper,
    CUDAGraphConfig,
    benchmark_cuda_graphs,
    CUDA_GRAPHS_AVAILABLE,
)

# Compiled model inference (torch.compile and OpenVINO)
from .compiled_model import (
    compile_pytorch,
    CompileConfig,
    convert_to_openvino,
    load_openvino_model,
    run_openvino_inference,
    OpenVINOConfig,
    CompiledModelWrapper,
    benchmark_compilation,
    TORCH_COMPILE_AVAILABLE,
    OPENVINO_AVAILABLE,
)

__all__ = [
    # Detector classes
    'TextDetector',
    'BatchTextDetector',
    'DetectionResult',
    # Async pipeline
    'AsyncInferencePipeline',
    'StreamingInferencePipeline',
    'create_pipeline',
    # Output caching
    'PerceptualHashCache',
    'ContentHashCache',
    'create_cache',
    # TensorRT backend
    'TensorRTEngine',
    'TensorRTEngineConfig',
    'load_tensorrt_engine',
    'TRT_AVAILABLE',
    # CUDA Graphs
    'CUDAGraphInference',
    'CUDAGraphPool',
    'CUDAGraphWrapper',
    'CUDAGraphConfig',
    'benchmark_cuda_graphs',
    'CUDA_GRAPHS_AVAILABLE',
    # Compiled models (torch.compile / OpenVINO)
    'compile_pytorch',
    'CompileConfig',
    'convert_to_openvino',
    'load_openvino_model',
    'run_openvino_inference',
    'OpenVINOConfig',
    'CompiledModelWrapper',
    'benchmark_compilation',
    'TORCH_COMPILE_AVAILABLE',
    'OPENVINO_AVAILABLE',
]
