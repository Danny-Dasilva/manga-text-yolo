#!/usr/bin/env python3
"""
Generate annotated comparison images for all model variants.

Usage:
    python scripts/generate_comparison.py --images data/val/images/*.jpg --output-dir results/images
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


def load_and_preprocess(image_path: str, input_size: int = 1024):
    """Load and preprocess image for model inference."""
    img = cv2.imread(str(image_path))
    original_shape = img.shape[:2]

    # Resize to input size
    img_resized = cv2.resize(img, (input_size, input_size))

    # Convert to RGB and normalize
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    return img, img_resized, img_tensor, original_shape


def draw_blocks(image: np.ndarray, blocks: np.ndarray, conf_thresh: float = 0.25) -> tuple:
    """Draw bounding boxes for detected blocks."""
    img = image.copy()
    h, w = img.shape[:2]

    block_count = 0
    for box in blocks:
        x, y, bw, bh, conf, cls = box
        if conf > conf_thresh:
            # Convert from normalized coords
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{conf:.2f}', (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            block_count += 1

    return img, block_count


def draw_mask(image: np.ndarray, mask: np.ndarray, threshold: float = 0.5,
              color: tuple = (0, 0, 255), alpha: float = 0.5) -> tuple:
    """Overlay segmentation mask on image."""
    img = image.copy()
    h, w = img.shape[:2]

    # Resize mask to image size
    mask_resized = cv2.resize(mask, (w, h))
    binary_mask = (mask_resized > threshold).astype(np.uint8)

    # Create colored overlay
    overlay = img.copy()
    overlay[binary_mask > 0] = color

    # Blend
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    mask_coverage = (binary_mask > 0).sum() / binary_mask.size * 100
    return img, mask_coverage


def run_onnx_inference(model_path: str, img_tensor: np.ndarray) -> dict:
    """Run inference with ONNX model."""
    sess = ort.InferenceSession(str(model_path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    outputs = sess.run(None, {'images': img_tensor})

    result = {}
    output_names = [o.name for o in sess.get_outputs()]
    
    for name, out in zip(output_names, outputs):
        # Normalize output names for compatibility
        if name == 'blk':
            # Reference model: [x, y, w, h, conf, cls1, cls2] in pixel coords
            # Convert to normalized format: [x, y, w, h, conf, cls]
            blk = out.copy()
            # Normalize coordinates from pixel (0-1024) to (0-1)
            blk[..., 0] /= 1024.0  # x
            blk[..., 1] /= 1024.0  # y
            blk[..., 2] /= 1024.0  # w
            blk[..., 3] /= 1024.0  # h
            # Keep conf as-is (col 4), use max of class probs as class
            result['blocks'] = blk[..., :6]  # Take first 6 cols
        elif name == 'seg':
            result['mask'] = out
        elif name == 'det':
            result['lines'] = out
        else:
            result[name] = out

    return result


def annotate_image(image: np.ndarray, outputs: dict, model_name: str) -> np.ndarray:
    """Annotate image with model outputs."""
    img = image.copy()
    h, w = img.shape[:2]

    stats = []

    # Draw blocks if present
    if 'blocks' in outputs:
        blocks = outputs['blocks'][0]  # Remove batch dim
        img, block_count = draw_blocks(img, blocks, conf_thresh=0.1)
        stats.append(f'Blocks: {block_count}')

    # Draw mask if present
    if 'mask' in outputs:
        mask = outputs['mask'][0, 0]  # Remove batch and channel dims
        img, mask_cov = draw_mask(img, mask, threshold=0.5, color=(0, 0, 255), alpha=0.3)
        stats.append(f'Mask: {mask_cov:.1f}%')

    # Add background for text
    cv2.rectangle(img, (0, 0), (w, 80), (0, 0, 0), -1)

    # Add model name (larger, white)
    cv2.putText(img, model_name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Add stats (smaller, yellow)
    stats_text = ' | '.join(stats)
    cv2.putText(img, stats_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    return img



def run_tensorrt_inference(engine_path: str, img_tensor: np.ndarray) -> dict:
    """Run inference with TensorRT engine."""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    # Load engine
    with open(engine_path, 'rb') as f:
        engine_data = f.read()

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    stream = cuda.Stream()

    # Allocate input buffer
    img_contiguous = np.ascontiguousarray(img_tensor)
    d_input = cuda.mem_alloc(img_contiguous.nbytes)

    # Allocate output buffers
    output_buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            host_buf = np.empty(shape, dtype=dtype)
            output_buffers[name] = {
                'host': np.ascontiguousarray(host_buf),
                'device': cuda.mem_alloc(host_buf.nbytes)
            }

    # Set tensor addresses
    context.set_tensor_address('images', int(d_input))
    for name, bufs in output_buffers.items():
        context.set_tensor_address(name, int(bufs['device']))

    # Run inference
    cuda.memcpy_htod_async(d_input, img_contiguous, stream)
    context.execute_async_v3(stream.handle)

    # Copy outputs back
    for name, bufs in output_buffers.items():
        cuda.memcpy_dtoh_async(bufs['host'], bufs['device'], stream)
    stream.synchronize()

    # Return as dict
    result = {}
    for name, bufs in output_buffers.items():
        result[name] = bufs['host']

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--output-dir', default='results/images')
    parser.add_argument('--num-images', type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # All 7 models to compare with clear labels
    models = [
        ('Reference (HuggingFace)', 'data/comic-text-detector.onnx', 'onnx', '8.43ms'),
        ('V3 FP32 (ONNX)', 'models/ctd_v3.onnx', 'onnx', '8.88ms'),
        ('V3 Optimized (ONNX)', 'models/ctd_v3_optimized.onnx', 'onnx', '~8.9ms'),
        ('V3 FP16 (ONNX)', 'models/ctd_v3_fp16.onnx', 'onnx', '6.72ms'),
        ('V3 INT8 (ONNX)', 'models/ctd_v3_int8.onnx', 'onnx', '15.27ms'),
        ('V3 FP16 (TensorRT)', 'models/ctd_v3_fp16.engine', 'tensorrt', '1.79ms'),
        ('V3 INT8 (TensorRT)', 'models/ctd_v3_int8.engine', 'tensorrt', '2.90ms'),
    ]

    # Process images
    image_paths = args.images[:args.num_images]

    for img_path in image_paths:
        img_path = Path(img_path)
        print(f'Processing {img_path.name}...')

        orig_img, img_resized, img_tensor, orig_shape = load_and_preprocess(str(img_path))

        annotated_images = []

        for model_name, model_path, backend, latency in models:
            if not Path(model_path).exists():
                print(f'  Skipping {model_name} (not found)')
                continue

            try:
                if backend == 'onnx':
                    outputs = run_onnx_inference(model_path, img_tensor)
                else:
                    outputs = run_tensorrt_inference(model_path, img_tensor)
                
                # Create label with model name and latency
                label = f'{model_name} [{latency}]'
                annotated = annotate_image(img_resized.copy(), outputs, label)
                annotated_images.append((model_name, annotated))
                print(f'  {model_name}: OK')
            except Exception as e:
                print(f'  {model_name}: Error - {e}')

        if annotated_images:
            # Create grid - 4 columns for 7 images (2 rows)
            n = len(annotated_images)
            cols = 4
            rows = (n + cols - 1) // cols

            h, w = annotated_images[0][1].shape[:2]
            grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

            for i, (name, img) in enumerate(annotated_images):
                r, c = i // cols, i % cols
                grid[r*h:(r+1)*h, c*w:(c+1)*w] = img

            # Save
            output_path = output_dir / f'{img_path.stem}_comparison.jpg'
            cv2.imwrite(str(output_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f'  Saved: {output_path}')


if __name__ == '__main__':
    main()
