"""
Performance optimization and caching utilities for spike-snn-event-vision-kit.

Provides caching, memory optimization, GPU acceleration, and performance
profiling tools for high-throughput neuromorphic vision processing.
"""

import time
import threading
import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple, List, Union
from collections import OrderedDict, defaultdict
from functools import wraps, lru_cache
import pickle
import hashlib
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import gc
import weakref
import os
from queue import Queue

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .monitoring import get_metrics_collector


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    total_requests: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
        
    @property
    def memory_usage_mb(self) -> float:
        """Get memory usage in MB."""
        return self.memory_usage_bytes / 1024 / 1024


class CacheInterface(ABC):
    """Abstract interface for cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Any:
        """Get item from cache."""
        pass
        
    @abstractmethod
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put item in cache."""
        pass
        
    @abstractmethod
    def delete(self, key: str):
        """Delete item from cache."""
        pass
        
    @abstractmethod
    def clear(self):
        """Clear all cache entries."""
        pass
        
    @abstractmethod
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


class LRUCache(CacheInterface):
    """Least Recently Used cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = OrderedDict()
        self._timestamps = {}
        self._ttls = {}
        self._stats = CacheStats()
        self._lock = threading.RLock()
        
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._ttls or self._ttls[key] is None:
            return False
        return time.time() > self._timestamps[key] + self._ttls[key]
        
    def _evict_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key in list(self._cache.keys()):
            if self._is_expired(key):
                expired_keys.append(key)
                
        for key in expired_keys:
            self._remove_key(key)
            self._stats.evictions += 1
            
    def _remove_key(self, key: str):
        """Remove key from all internal structures."""
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]
        if key in self._ttls:
            del self._ttls[key]
            
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            if hasattr(obj, 'nbytes'):  # numpy arrays
                return obj.nbytes
            elif TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
                return obj.element_size() * obj.nelement()
            else:
                return len(pickle.dumps(obj))
        except Exception:
            return 1024  # Default estimate
            
    def get(self, key: str) -> Any:
        """Get item from cache."""
        with self._lock:
            self._stats.total_requests += 1
            
            # Clean expired entries periodically
            if self._stats.total_requests % 100 == 0:
                self._evict_expired()
                
            if key not in self._cache:
                self._stats.misses += 1
                return None
                
            if self._is_expired(key):
                self._remove_key(key)
                self._stats.misses += 1
                self._stats.evictions += 1
                return None
                
            # Move to end (most recently used)
            value = self._cache.pop(key)
            self._cache[key] = value
            self._stats.hits += 1
            
            return value
            
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put item in cache."""
        with self._lock:
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl
                
            # Remove existing entry if present
            if key in self._cache:
                self._remove_key(key)
                
            # Evict oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                self._remove_key(oldest_key)
                self._stats.evictions += 1
                
            # Add new entry
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._ttls[key] = ttl
            
            # Update memory usage estimate
            self._stats.memory_usage_bytes += self._estimate_size(value)
            
    def delete(self, key: str):
        """Delete item from cache."""
        with self._lock:
            if key in self._cache:
                self._stats.memory_usage_bytes -= self._estimate_size(self._cache[key])
                self._remove_key(key)
                
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._ttls.clear()
            self._stats.memory_usage_bytes = 0
            
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                memory_usage_bytes=self._stats.memory_usage_bytes,
                total_requests=self._stats.total_requests
            )


class ModelCache:
    """Specialized cache for neural network models and computed features."""
    
    def __init__(self, max_models: int = 5, max_features_mb: int = 500):
        self.max_models = max_models
        self.max_features_mb = max_features_mb
        self.model_cache = LRUCache(max_size=max_models)
        self.feature_cache = LRUCache(max_size=1000, default_ttl=300.0)  # 5 min TTL
        self.logger = logging.getLogger(__name__)
        
    def get_model(self, model_key: str):
        """Get cached model."""
        return self.model_cache.get(model_key)
        
    def cache_model(self, model_key: str, model: Any):
        """Cache a model."""
        self.model_cache.put(model_key, model)
        self.logger.info(f"Cached model: {model_key}")
        
    def get_features(self, input_hash: str) -> Optional[Any]:
        """Get cached feature computation."""
        return self.feature_cache.get(input_hash)
        
    def cache_features(self, input_hash: str, features: Any):
        """Cache computed features."""
        # Check memory limits
        feature_size_mb = self._estimate_size_mb(features)
        
        if feature_size_mb > self.max_features_mb:
            self.logger.warning(f"Feature size {feature_size_mb}MB exceeds cache limit")
            return
            
        # Evict old features if needed
        while (self.feature_cache.stats().memory_usage_mb + feature_size_mb > 
               self.max_features_mb):
            # Force eviction by adding dummy entry
            oldest_key = next(iter(self.feature_cache._cache), None)
            if oldest_key:
                self.feature_cache.delete(oldest_key)
            else:
                break
                
        self.feature_cache.put(input_hash, features)
        
    def _estimate_size_mb(self, obj: Any) -> float:
        """Estimate object size in MB."""
        try:
            if hasattr(obj, 'nbytes'):
                return obj.nbytes / 1024 / 1024
            elif TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
                return (obj.element_size() * obj.nelement()) / 1024 / 1024
            else:
                return len(pickle.dumps(obj)) / 1024 / 1024
        except Exception:
            return 1.0  # 1MB default estimate
            
    def clear_all(self):
        """Clear all caches."""
        self.model_cache.clear()
        self.feature_cache.clear()
        self.logger.info("Cleared all caches")
        
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all caches."""
        return {
            'models': self.model_cache.stats(),
            'features': self.feature_cache.stats()
        }


class MemoryOptimizer:
    """Memory usage optimization utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._weak_refs = []
        
    def optimize_numpy_arrays(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        """Optimize numpy array memory usage."""
        optimized = []
        
        for arr in arrays:
            # Convert to most compact dtype if possible
            if arr.dtype == np.float64:
                # Check if values can fit in float32
                if np.allclose(arr, arr.astype(np.float32)):
                    arr = arr.astype(np.float32)
                    
            elif arr.dtype == np.int64:
                # Check if values can fit in smaller int types
                if arr.min() >= -128 and arr.max() <= 127:
                    arr = arr.astype(np.int8)
                elif arr.min() >= -32768 and arr.max() <= 32767:
                    arr = arr.astype(np.int16)
                elif arr.min() >= -2147483648 and arr.max() <= 2147483647:
                    arr = arr.astype(np.int32)
                    
            # Ensure arrays are C-contiguous for better cache performance
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
                
            optimized.append(arr)
            
        return optimized
        
    def chunk_large_operations(
        self, 
        data: np.ndarray, 
        operation: Callable,
        chunk_size: int = 10000
    ) -> np.ndarray:
        """Process large arrays in chunks to reduce memory usage."""
        if len(data) <= chunk_size:
            return operation(data)
            
        results = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            result = operation(chunk)
            results.append(result)
            
        return np.concatenate(results, axis=0)
        
    def clear_gpu_cache(self):
        """Clear GPU memory cache if available."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.debug("Cleared GPU cache")
            
    def force_garbage_collection(self):
        """Force garbage collection."""
        collected = gc.collect()
        self.logger.debug(f"Garbage collection freed {collected} objects")
        
    def track_memory_usage(self, obj: Any) -> int:
        """Track memory usage of an object."""
        if hasattr(obj, 'nbytes'):
            return obj.nbytes
        elif TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        else:
            try:
                return len(pickle.dumps(obj))
            except Exception:
                return 0
                
    def create_memory_mapped_array(
        self, 
        shape: Tuple[int, ...], 
        dtype: np.dtype,
        filename: Optional[str] = None
    ) -> np.ndarray:
        """Create memory-mapped array for large datasets."""
        if filename is None:
            # Create temporary file
            import tempfile
            fd, filename = tempfile.mkstemp(suffix='.dat')
            os.close(fd)
            
        return np.memmap(filename, dtype=dtype, mode='w+', shape=shape)


class GPUAccelerator:
    """GPU acceleration utilities."""
    
    def __init__(self):
        self.device = None
        self.logger = logging.getLogger(__name__)
        self._initialize_gpu()
        
    def _initialize_gpu(self):
        """Initialize GPU if available."""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, GPU acceleration disabled")
            return
            
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            # Optimize GPU settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            self.logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("GPU not available, using CPU")
            
    def to_gpu(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Move data to GPU."""
        if not TORCH_AVAILABLE or self.device.type == 'cpu':
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data)
            return data
            
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        else:
            tensor = data
            
        return tensor.to(self.device, non_blocking=True)
        
    def to_cpu(self, data: torch.Tensor) -> np.ndarray:
        """Move data from GPU to CPU."""
        if not TORCH_AVAILABLE:
            return data
            
        return data.cpu().numpy()
        
    def optimize_model_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model for inference."""
        if not TORCH_AVAILABLE:
            return model
            
        # Set to evaluation mode
        model.eval()
        
        # Move to GPU
        if self.device.type == 'cuda':
            model = model.to(self.device)
            
        # Optimize with TorchScript if possible
        try:
            # Create dummy input for tracing
            dummy_input = torch.randn(1, 2, 128, 128, 10).to(self.device)
            model = torch.jit.trace(model, dummy_input)
            self.logger.info("Model optimized with TorchScript")
        except Exception as e:
            self.logger.warning(f"TorchScript optimization failed: {e}")
            
        return model
        
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get GPU memory information."""
        if not TORCH_AVAILABLE or self.device.type == 'cpu':
            return {'available': 0, 'total': 0, 'used': 0}
            
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        return {
            'allocated_gb': allocated,
            'cached_gb': cached,
            'total_gb': total,
            'free_gb': total - cached
        }


# Global instances
_global_model_cache = None
_global_memory_optimizer = None
_global_gpu_accelerator = None


def get_model_cache() -> ModelCache:
    """Get global model cache instance."""
    global _global_model_cache
    if _global_model_cache is None:
        _global_model_cache = ModelCache()
    return _global_model_cache


def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    global _global_memory_optimizer
    if _global_memory_optimizer is None:
        _global_memory_optimizer = MemoryOptimizer()
    return _global_memory_optimizer


def get_gpu_accelerator() -> GPUAccelerator:
    """Get global GPU accelerator instance."""
    global _global_gpu_accelerator
    if _global_gpu_accelerator is None:
        _global_gpu_accelerator = GPUAccelerator()
    return _global_gpu_accelerator