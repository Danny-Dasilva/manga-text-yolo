"""
Output Caching with Perceptual Hashing for Comic Text Detection

Features:
- Perceptual hash-based image similarity detection
- LRU memory cache for fast lookups
- Optional disk cache persistence
- Configurable Hamming distance threshold
- Thread-safe operations
"""

from __future__ import annotations

import numpy as np
import json
import hashlib
import pickle
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Conditional import for imagehash (optional dependency)
try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    logger.warning("imagehash not available. Install with: pip install imagehash")


@dataclass
class CacheEntry:
    """Container for cached inference results."""
    phash: str
    result: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)


class LRUCache:
    """Thread-safe LRU cache implementation."""

    def __init__(self, maxsize: int = 1000):
        """
        Initialize LRU cache.

        Args:
            maxsize: Maximum number of entries
        """
        self.maxsize = maxsize
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry and move to end (most recently used)."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                self._cache.move_to_end(key)
                return entry
            return None

    def put(self, key: str, entry: CacheEntry) -> None:
        """Add entry, evicting oldest if necessary."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = entry

    def remove(self, key: str) -> bool:
        """Remove entry by key."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        """Get number of entries."""
        with self._lock:
            return len(self._cache)

    def keys(self) -> List[str]:
        """Get all keys."""
        with self._lock:
            return list(self._cache.keys())


class PerceptualHashCache:
    """
    Cache inference results using perceptual hashing.

    Perceptual hashing allows finding similar images even when they have
    minor differences (compression artifacts, slight crops, etc.).

    Example:
        ```python
        cache = PerceptualHashCache(memory_size=1000, disk_cache_dir='.cache')

        def compute_inference(image):
            return model(preprocess(image))

        # First call computes and caches
        result1 = cache.get_or_compute(image1, compute_inference)

        # Similar image hits cache
        result2 = cache.get_or_compute(similar_image, compute_inference)

        # Get cache stats
        print(cache.get_stats())
        ```
    """

    def __init__(
        self,
        memory_size: int = 1000,
        disk_cache_dir: Optional[str] = None,
        hamming_threshold: int = 5,
        hash_size: int = 8,
        use_average_hash: bool = False,
    ):
        """
        Initialize perceptual hash cache.

        Args:
            memory_size: Maximum entries in memory cache
            disk_cache_dir: Optional directory for persistent disk cache
            hamming_threshold: Maximum Hamming distance for similarity match
            hash_size: Size of perceptual hash (default 8 gives 64-bit hash)
            use_average_hash: Use average hash instead of perceptual hash (faster but less robust)
        """
        if not IMAGEHASH_AVAILABLE:
            raise ImportError("imagehash required. Install with: pip install imagehash Pillow")

        self.memory_cache = LRUCache(maxsize=memory_size)
        self.hamming_threshold = hamming_threshold
        self.hash_size = hash_size
        self.use_average_hash = use_average_hash

        # Disk cache setup
        self.disk_cache_dir: Optional[Path] = None
        if disk_cache_dir:
            self.disk_cache_dir = Path(disk_cache_dir)
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_index()
        else:
            self._disk_index: Dict[str, Path] = {}

        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'similar_hits': 0,
            'disk_hits': 0,
            'computations': 0,
        }
        self._stats_lock = threading.Lock()

        logger.info(f"PerceptualHashCache initialized: memory={memory_size}, threshold={hamming_threshold}")

    def _compute_phash(self, image: np.ndarray) -> str:
        """
        Compute perceptual hash for an image.

        Args:
            image: Image as numpy array (BGR or RGB, any shape)

        Returns:
            Hex string representation of perceptual hash
        """
        # Convert numpy array to PIL Image
        if image.ndim == 2:
            # Grayscale
            pil_img = Image.fromarray(image.astype(np.uint8))
        elif image.shape[2] == 4:
            # RGBA
            pil_img = Image.fromarray(image.astype(np.uint8), 'RGBA')
        else:
            # BGR to RGB
            rgb = image[:, :, ::-1] if image.shape[2] == 3 else image
            pil_img = Image.fromarray(rgb.astype(np.uint8))

        # Compute hash
        if self.use_average_hash:
            phash = imagehash.average_hash(pil_img, hash_size=self.hash_size)
        else:
            phash = imagehash.phash(pil_img, hash_size=self.hash_size)

        return str(phash)

    def _compute_dhash(self, image: np.ndarray) -> str:
        """
        Compute difference hash (faster alternative).

        Args:
            image: Image as numpy array

        Returns:
            Hex string representation of difference hash
        """
        if image.ndim == 2:
            pil_img = Image.fromarray(image.astype(np.uint8))
        else:
            rgb = image[:, :, ::-1] if image.shape[2] == 3 else image
            pil_img = Image.fromarray(rgb.astype(np.uint8))

        return str(imagehash.dhash(pil_img, hash_size=self.hash_size))

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        Compute Hamming distance between two hashes.

        Args:
            hash1: First hash as hex string
            hash2: Second hash as hex string

        Returns:
            Hamming distance (number of different bits)
        """
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2  # imagehash uses __sub__ for Hamming distance

    def _find_similar(self, phash: str) -> Optional[CacheEntry]:
        """
        Find similar image in cache within Hamming threshold.

        Args:
            phash: Perceptual hash to search for

        Returns:
            CacheEntry if similar image found, None otherwise
        """
        # First check exact match
        exact = self.memory_cache.get(phash)
        if exact is not None:
            return exact

        # Search for similar hashes
        for cached_hash in self.memory_cache.keys():
            distance = self._hamming_distance(phash, cached_hash)
            if distance <= self.hamming_threshold:
                entry = self.memory_cache.get(cached_hash)
                if entry is not None:
                    return entry

        return None

    def _load_disk_index(self) -> None:
        """Load disk cache index."""
        self._disk_index = {}
        if self.disk_cache_dir is None:
            return

        index_path = self.disk_cache_dir / 'index.json'
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    self._disk_index = {k: self.disk_cache_dir / v for k, v in json.load(f).items()}
                logger.info(f"Loaded disk cache index with {len(self._disk_index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load disk cache index: {e}")
                self._disk_index = {}

    def _save_disk_index(self) -> None:
        """Save disk cache index."""
        if self.disk_cache_dir is None:
            return

        index_path = self.disk_cache_dir / 'index.json'
        try:
            relative_index = {k: v.name for k, v in self._disk_index.items()}
            with open(index_path, 'w') as f:
                json.dump(relative_index, f)
        except Exception as e:
            logger.warning(f"Failed to save disk cache index: {e}")

    def _load_from_disk(self, phash: str) -> Optional[Dict[str, Any]]:
        """Load result from disk cache."""
        if phash not in self._disk_index:
            return None

        cache_path = self._disk_index[phash]
        if not cache_path.exists():
            del self._disk_index[phash]
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            return None

    def _save_to_disk(self, phash: str, result: Dict[str, Any]) -> None:
        """Save result to disk cache."""
        if self.disk_cache_dir is None:
            return

        try:
            # Use hash as filename
            cache_path = self.disk_cache_dir / f"{phash}.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)

            self._disk_index[phash] = cache_path
            self._save_disk_index()
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")

    def get(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Get cached result for image.

        Args:
            image: Image as numpy array

        Returns:
            Cached result if found, None otherwise
        """
        phash = self._compute_phash(image)

        # Check memory cache (exact and similar)
        entry = self._find_similar(phash)
        if entry is not None:
            with self._stats_lock:
                if entry.phash == phash:
                    self._stats['hits'] += 1
                else:
                    self._stats['similar_hits'] += 1
            return entry.result

        # Check disk cache
        if self.disk_cache_dir:
            disk_result = self._load_from_disk(phash)
            if disk_result is not None:
                # Promote to memory cache
                entry = CacheEntry(phash=phash, result=disk_result)
                self.memory_cache.put(phash, entry)
                with self._stats_lock:
                    self._stats['disk_hits'] += 1
                return disk_result

        with self._stats_lock:
            self._stats['misses'] += 1
        return None

    def put(self, image: np.ndarray, result: Dict[str, Any]) -> str:
        """
        Store result in cache.

        Args:
            image: Image as numpy array
            result: Inference result to cache

        Returns:
            Perceptual hash of the image
        """
        phash = self._compute_phash(image)

        # Store in memory cache
        entry = CacheEntry(phash=phash, result=result)
        self.memory_cache.put(phash, entry)

        # Store in disk cache
        if self.disk_cache_dir:
            self._save_to_disk(phash, result)

        return phash

    def get_or_compute(
        self,
        image: np.ndarray,
        compute_fn: Callable[[np.ndarray], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Get cached result or compute and cache.

        Args:
            image: Image as numpy array
            compute_fn: Function to compute result if not cached

        Returns:
            Cached or computed result
        """
        # Try to get from cache
        cached = self.get(image)
        if cached is not None:
            return cached

        # Compute result
        with self._stats_lock:
            self._stats['computations'] += 1

        result = compute_fn(image)

        # Cache result
        self.put(image, result)

        return result

    def invalidate(self, image: np.ndarray) -> bool:
        """
        Invalidate cache entry for image.

        Args:
            image: Image to invalidate

        Returns:
            True if entry was found and removed
        """
        phash = self._compute_phash(image)

        # Remove from memory
        removed = self.memory_cache.remove(phash)

        # Remove from disk
        if phash in self._disk_index:
            try:
                self._disk_index[phash].unlink()
                del self._disk_index[phash]
                self._save_disk_index()
            except Exception as e:
                logger.warning(f"Failed to remove disk cache entry: {e}")

        return removed

    def clear(self) -> None:
        """Clear all cache entries."""
        self.memory_cache.clear()

        if self.disk_cache_dir:
            # Clear disk cache
            for path in self._disk_index.values():
                try:
                    path.unlink()
                except Exception:
                    pass
            self._disk_index = {}
            self._save_disk_index()

        # Reset stats
        with self._stats_lock:
            for key in self._stats:
                self._stats[key] = 0

        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._stats_lock:
            stats = self._stats.copy()

        total_requests = stats['hits'] + stats['similar_hits'] + stats['disk_hits'] + stats['misses']
        hit_rate = (stats['hits'] + stats['similar_hits'] + stats['disk_hits']) / max(total_requests, 1)

        return {
            **stats,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'memory_entries': len(self.memory_cache),
            'disk_entries': len(self._disk_index),
        }

    def __len__(self) -> int:
        """Get total number of cached entries."""
        return len(self.memory_cache) + len(self._disk_index)


class ContentHashCache:
    """
    Simple content-based hash cache (no perceptual similarity).

    Faster than perceptual hashing but only matches exact images.
    """

    def __init__(
        self,
        memory_size: int = 1000,
        disk_cache_dir: Optional[str] = None,
    ):
        """
        Initialize content hash cache.

        Args:
            memory_size: Maximum entries in memory cache
            disk_cache_dir: Optional directory for disk cache
        """
        self.memory_cache = LRUCache(maxsize=memory_size)
        self.disk_cache_dir = Path(disk_cache_dir) if disk_cache_dir else None

        if self.disk_cache_dir:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_hash(self, image: np.ndarray) -> str:
        """Compute content hash using MD5."""
        return hashlib.md5(image.tobytes()).hexdigest()

    def get_or_compute(
        self,
        image: np.ndarray,
        compute_fn: Callable[[np.ndarray], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get cached result or compute and cache."""
        content_hash = self._compute_hash(image)

        # Check memory cache
        entry = self.memory_cache.get(content_hash)
        if entry is not None:
            return entry.result

        # Compute result
        result = compute_fn(image)

        # Cache result
        entry = CacheEntry(phash=content_hash, result=result)
        self.memory_cache.put(content_hash, entry)

        return result


def create_cache(
    cache_type: str = 'perceptual',
    memory_size: int = 1000,
    disk_cache_dir: Optional[str] = None,
    **kwargs,
) -> PerceptualHashCache | ContentHashCache:
    """
    Factory function to create a cache instance.

    Args:
        cache_type: Type of cache ('perceptual' or 'content')
        memory_size: Maximum entries in memory cache
        disk_cache_dir: Optional directory for disk cache
        **kwargs: Additional arguments for cache

    Returns:
        Cache instance
    """
    if cache_type == 'perceptual':
        return PerceptualHashCache(
            memory_size=memory_size,
            disk_cache_dir=disk_cache_dir,
            **kwargs,
        )
    elif cache_type == 'content':
        return ContentHashCache(
            memory_size=memory_size,
            disk_cache_dir=disk_cache_dir,
        )
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")
