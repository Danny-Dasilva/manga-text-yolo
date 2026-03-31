"""
Async Inference Pipeline for Comic Text Detection

Features:
- 3-stage async pipeline: I/O -> Preprocess -> Inference
- Thread-based parallelism for CPU-bound I/O and preprocessing
- Queue-based communication between stages
- Graceful shutdown with sentinel values
- Configurable worker counts and queue sizes
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union, Iterator, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineItem:
    """Container for data flowing through the pipeline."""
    path: str
    image: Optional[np.ndarray] = None
    tensor: Optional[torch.Tensor] = None
    meta: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None


# Sentinel value for shutdown
_SHUTDOWN = object()


class AsyncInferencePipeline:
    """
    3-stage async pipeline: I/O -> Preprocess -> Inference.

    This pipeline overlaps I/O, preprocessing, and GPU inference to maximize
    throughput when processing many images.

    Architecture:
    ```
    [I/O Workers] --io_queue--> [Preproc Workers] --preproc_queue--> [Inference Worker]
         |                            |                                     |
    ThreadPool(4)              ThreadPool(2)                          Main Thread
    ```

    Example:
        ```python
        pipeline = AsyncInferencePipeline(model, io_workers=4, preproc_workers=2)

        image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        for path, result in pipeline.process_batch(image_paths):
            print(f"{path}: {len(result['text_blocks'])} blocks detected")

        pipeline.shutdown()
        ```
    """

    def __init__(
        self,
        model: Any,
        input_size: int = 1024,
        io_workers: int = 4,
        preproc_workers: int = 2,
        prefetch_size: int = 16,
        device: str = 'cuda',
    ):
        """
        Initialize the async pipeline.

        Args:
            model: Inference model (PyTorch nn.Module or ONNX session)
            input_size: Model input size (square)
            io_workers: Number of I/O worker threads
            preproc_workers: Number of preprocessing worker threads
            prefetch_size: Maximum queue sizes (controls memory usage)
            device: Device for inference ('cuda' or 'cpu')
        """
        self.model = model
        self.input_size = input_size
        self.device = device
        self.prefetch_size = prefetch_size

        # Queues for inter-stage communication
        self.io_queue: Queue[Union[PipelineItem, object]] = Queue(maxsize=prefetch_size)
        self.preproc_queue: Queue[Union[PipelineItem, object]] = Queue(maxsize=prefetch_size)
        self.result_queue: Queue[Union[PipelineItem, object]] = Queue()

        # Thread pools for I/O and preprocessing
        self.io_executor = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix='io')
        self.preproc_executor = ThreadPoolExecutor(max_workers=preproc_workers, thread_name_prefix='preproc')

        # Worker control
        self._shutdown_event = threading.Event()
        self._preproc_threads: List[threading.Thread] = []
        self._inference_thread: Optional[threading.Thread] = None

        # Start worker threads
        self._start_workers(preproc_workers)

        logger.info(f"AsyncInferencePipeline initialized: io={io_workers}, preproc={preproc_workers}, prefetch={prefetch_size}")

    def _start_workers(self, preproc_workers: int) -> None:
        """Start preprocessing and inference worker threads."""
        # Start preprocessing workers
        for i in range(preproc_workers):
            thread = threading.Thread(
                target=self._preproc_worker,
                name=f'preproc-{i}',
                daemon=True,
            )
            thread.start()
            self._preproc_threads.append(thread)

        # Start inference worker
        self._inference_thread = threading.Thread(
            target=self._inference_worker,
            name='inference',
            daemon=True,
        )
        self._inference_thread.start()

    def _io_worker(self, image_paths: List[str]) -> None:
        """
        I/O worker: Read images from disk and put them in the I/O queue.

        This runs in the thread pool (multiple concurrent readers).
        """
        for path in image_paths:
            if self._shutdown_event.is_set():
                break

            item = PipelineItem(path=path)
            try:
                # Read image with OpenCV
                img = cv2.imread(path)
                if img is None:
                    item.error = ValueError(f"Failed to read image: {path}")
                    logger.warning(f"Failed to read image: {path}")
                else:
                    item.image = img
            except Exception as e:
                item.error = e
                logger.error(f"I/O error for {path}: {e}")

            self.io_queue.put(item)

        # Signal end of this batch
        self.io_queue.put(_SHUTDOWN)

    def _preproc_worker(self) -> None:
        """
        Preprocessing worker: Prepare images for model inference.

        This runs in dedicated threads (CPU-intensive but parallelizable).
        """
        while not self._shutdown_event.is_set():
            try:
                item = self.io_queue.get(timeout=0.1)
            except Empty:
                continue

            if item is _SHUTDOWN:
                # Pass shutdown signal to next stage
                self.preproc_queue.put(_SHUTDOWN)
                break

            if item.error is not None:
                # Pass errors through
                self.preproc_queue.put(item)
                continue

            try:
                # Preprocess image
                tensor, meta = self._preprocess(item.image)
                item.tensor = tensor
                item.meta = meta
            except Exception as e:
                item.error = e
                logger.error(f"Preprocessing error for {item.path}: {e}")

            self.preproc_queue.put(item)

    def _inference_worker(self) -> None:
        """
        Inference worker: Run model inference on preprocessed tensors.

        This runs in a single thread to serialize GPU access.
        """
        batch: List[PipelineItem] = []
        shutdown_count = 0
        expected_shutdowns = len(self._preproc_threads)

        while not self._shutdown_event.is_set():
            try:
                item = self.preproc_queue.get(timeout=0.1)
            except Empty:
                # Process any pending batch
                if batch:
                    self._process_batch(batch)
                    batch = []
                continue

            if item is _SHUTDOWN:
                shutdown_count += 1
                if shutdown_count >= expected_shutdowns:
                    # All preproc workers done, process final batch
                    if batch:
                        self._process_batch(batch)
                    self.result_queue.put(_SHUTDOWN)
                    break
                continue

            if item.error is not None:
                # Pass errors through
                self.result_queue.put(item)
                continue

            batch.append(item)

            # Process batch when full
            if len(batch) >= self.prefetch_size // 2:
                self._process_batch(batch)
                batch = []

    def _process_batch(self, batch: List[PipelineItem]) -> None:
        """Process a batch of items through the model."""
        if not batch:
            return

        try:
            # Stack tensors
            tensors = torch.stack([item.tensor for item in batch])

            # Run inference
            if hasattr(self.model, 'forward'):
                # PyTorch model
                with torch.no_grad():
                    if 'cuda' in self.device:
                        tensors = tensors.to(self.device)
                    outputs = self.model(tensors)
                    if isinstance(outputs, tuple):
                        # Handle multiple outputs (mask, lines, blocks)
                        outputs = [o.cpu().numpy() if hasattr(o, 'cpu') else o for o in outputs]
                    else:
                        outputs = [outputs.cpu().numpy()]
            else:
                # ONNX model
                input_name = self.model.get_inputs()[0].name
                outputs = self.model.run(None, {input_name: tensors.numpy()})

            # Assign results
            for i, item in enumerate(batch):
                item.result = {
                    'outputs': [o[i] if o.ndim > 0 else o for o in outputs],
                    'meta': item.meta,
                }
                self.result_queue.put(item)

        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            for item in batch:
                item.error = e
                self.result_queue.put(item)

    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Preprocess single image with letterbox padding.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            Tuple of (tensor, metadata)
        """
        h, w = image.shape[:2]
        target_size = self.input_size

        # Calculate scale and padding
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Convert to tensor
        tensor = padded.astype(np.float32) / 255.0
        tensor = tensor[:, :, ::-1]  # BGR to RGB
        tensor = tensor.transpose(2, 0, 1)  # HWC to CHW
        tensor = torch.from_numpy(tensor.copy())

        meta = {
            'original_size': (h, w),
            'scale': scale,
            'pad': (pad_h, pad_w),
            'new_size': (new_h, new_w),
        }

        return tensor, meta

    def process_batch(
        self,
        image_paths: List[str],
        timeout: float = 30.0,
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        Process a batch of images through the pipeline.

        Args:
            image_paths: List of image file paths
            timeout: Timeout for waiting on results (seconds)

        Yields:
            Tuples of (path, result_dict) or (path, error_dict)
        """
        if not image_paths:
            return

        # Start I/O worker
        self.io_executor.submit(self._io_worker, image_paths)

        # Collect results
        received = 0
        expected = len(image_paths)

        while received < expected:
            try:
                item = self.result_queue.get(timeout=timeout)
            except Empty:
                logger.warning(f"Timeout waiting for results ({received}/{expected} received)")
                break

            if item is _SHUTDOWN:
                break

            received += 1

            if item.error is not None:
                yield item.path, {'error': str(item.error)}
            else:
                yield item.path, item.result

    def process_single(self, image_path: str, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Process a single image through the pipeline.

        Args:
            image_path: Path to image file
            timeout: Timeout for waiting on result

        Returns:
            Result dictionary with outputs and metadata
        """
        results = list(self.process_batch([image_path], timeout=timeout))
        if results:
            return results[0][1]
        return {'error': 'Timeout waiting for result'}

    def process_images_direct(
        self,
        images: List[np.ndarray],
        timeout: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        Process images directly (without I/O stage).

        Args:
            images: List of BGR images as numpy arrays
            timeout: Timeout for processing

        Returns:
            List of result dictionaries
        """
        results = []

        # Directly feed to preproc queue
        for i, img in enumerate(images):
            item = PipelineItem(path=f"image_{i}", image=img)
            try:
                tensor, meta = self._preprocess(img)
                item.tensor = tensor
                item.meta = meta
            except Exception as e:
                item.error = e
            self.preproc_queue.put(item)

        # Signal end
        self.preproc_queue.put(_SHUTDOWN)

        # Collect results
        while len(results) < len(images):
            try:
                item = self.result_queue.get(timeout=timeout)
            except Empty:
                break

            if item is _SHUTDOWN:
                break

            if item.error is not None:
                results.append({'error': str(item.error)})
            else:
                results.append(item.result)

        return results

    def shutdown(self, wait: bool = True, timeout: float = 5.0) -> None:
        """
        Gracefully shutdown the pipeline.

        Args:
            wait: Whether to wait for threads to finish
            timeout: Timeout for waiting on each thread
        """
        logger.info("Shutting down async pipeline...")

        # Signal shutdown
        self._shutdown_event.set()

        # Send shutdown signals to queues
        for _ in range(len(self._preproc_threads)):
            try:
                self.io_queue.put_nowait(_SHUTDOWN)
            except Exception:
                pass

        if wait:
            # Wait for threads
            for thread in self._preproc_threads:
                thread.join(timeout=timeout)

            if self._inference_thread is not None:
                self._inference_thread.join(timeout=timeout)

        # Shutdown executors
        self.io_executor.shutdown(wait=wait)
        self.preproc_executor.shutdown(wait=wait)

        logger.info("Async pipeline shutdown complete")

    def __enter__(self) -> 'AsyncInferencePipeline':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.shutdown()

    def __del__(self) -> None:
        """Destructor - ensure cleanup."""
        if hasattr(self, '_shutdown_event') and not self._shutdown_event.is_set():
            self.shutdown(wait=False)


class StreamingInferencePipeline(AsyncInferencePipeline):
    """
    Streaming variant of the async pipeline for real-time processing.

    Processes images as they arrive without waiting for batch completion.
    """

    def __init__(
        self,
        model: Any,
        input_size: int = 1024,
        device: str = 'cuda',
        max_latency_ms: float = 50.0,
    ):
        """
        Initialize streaming pipeline.

        Args:
            model: Inference model
            input_size: Model input size
            device: Device for inference
            max_latency_ms: Maximum latency before forcing batch processing
        """
        super().__init__(
            model=model,
            input_size=input_size,
            io_workers=2,  # Lower for streaming
            preproc_workers=1,  # Lower for streaming
            prefetch_size=4,  # Lower for latency
            device=device,
        )
        self.max_latency_ms = max_latency_ms

    def submit(self, image: np.ndarray, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Submit an image for processing with async callback.

        Args:
            image: BGR image as numpy array
            callback: Function to call with result
        """
        # TODO: Implement callback-based streaming
        raise NotImplementedError("Streaming callbacks not yet implemented")


def create_pipeline(
    model: Any,
    input_size: int = 1024,
    device: str = 'cuda',
    mode: str = 'batch',
    **kwargs,
) -> AsyncInferencePipeline:
    """
    Factory function to create an inference pipeline.

    Args:
        model: Inference model
        input_size: Model input size
        device: Device for inference
        mode: Pipeline mode ('batch' or 'streaming')
        **kwargs: Additional arguments for the pipeline

    Returns:
        Configured inference pipeline
    """
    if mode == 'batch':
        return AsyncInferencePipeline(
            model=model,
            input_size=input_size,
            device=device,
            **kwargs,
        )
    elif mode == 'streaming':
        return StreamingInferencePipeline(
            model=model,
            input_size=input_size,
            device=device,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
