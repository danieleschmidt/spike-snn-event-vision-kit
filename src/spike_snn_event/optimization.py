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
        self.logger.info(f\"Cached model: {model_key}\")\n        \n    def get_features(self, input_hash: str) -> Optional[Any]:\n        \"\"\"Get cached feature computation.\"\"\"\n        return self.feature_cache.get(input_hash)\n        \n    def cache_features(self, input_hash: str, features: Any):\n        \"\"\"Cache computed features.\"\"\"\n        # Check memory limits\n        feature_size_mb = self._estimate_size_mb(features)\n        \n        if feature_size_mb > self.max_features_mb:\n            self.logger.warning(f\"Feature size {feature_size_mb}MB exceeds cache limit\")\n            return\n            \n        # Evict old features if needed\n        while (self.feature_cache.stats().memory_usage_mb + feature_size_mb > \n               self.max_features_mb):\n            # Force eviction by adding dummy entry\n            oldest_key = next(iter(self.feature_cache._cache), None)\n            if oldest_key:\n                self.feature_cache.delete(oldest_key)\n            else:\n                break\n                \n        self.feature_cache.put(input_hash, features)\n        \n    def _estimate_size_mb(self, obj: Any) -> float:\n        \"\"\"Estimate object size in MB.\"\"\"\n        try:\n            if hasattr(obj, 'nbytes'):\n                return obj.nbytes / 1024 / 1024\n            elif TORCH_AVAILABLE and isinstance(obj, torch.Tensor):\n                return (obj.element_size() * obj.nelement()) / 1024 / 1024\n            else:\n                return len(pickle.dumps(obj)) / 1024 / 1024\n        except Exception:\n            return 1.0  # 1MB default estimate\n            \n    def clear_all(self):\n        \"\"\"Clear all caches.\"\"\"\n        self.model_cache.clear()\n        self.feature_cache.clear()\n        self.logger.info(\"Cleared all caches\")\n        \n    def get_stats(self) -> Dict[str, CacheStats]:\n        \"\"\"Get statistics for all caches.\"\"\"\n        return {\n            'models': self.model_cache.stats(),\n            'features': self.feature_cache.stats()\n        }\n\n\nclass MemoryOptimizer:\n    \"\"\"Memory usage optimization utilities.\"\"\"\n    \n    def __init__(self):\n        self.logger = logging.getLogger(__name__)\n        self._weak_refs = []\n        \n    def optimize_numpy_arrays(self, arrays: List[np.ndarray]) -> List[np.ndarray]:\n        \"\"\"Optimize numpy array memory usage.\"\"\"\n        optimized = []\n        \n        for arr in arrays:\n            # Convert to most compact dtype if possible\n            if arr.dtype == np.float64:\n                # Check if values can fit in float32\n                if np.allclose(arr, arr.astype(np.float32)):\n                    arr = arr.astype(np.float32)\n                    \n            elif arr.dtype == np.int64:\n                # Check if values can fit in smaller int types\n                if arr.min() >= -128 and arr.max() <= 127:\n                    arr = arr.astype(np.int8)\n                elif arr.min() >= -32768 and arr.max() <= 32767:\n                    arr = arr.astype(np.int16)\n                elif arr.min() >= -2147483648 and arr.max() <= 2147483647:\n                    arr = arr.astype(np.int32)\n                    \n            # Ensure arrays are C-contiguous for better cache performance\n            if not arr.flags['C_CONTIGUOUS']:\n                arr = np.ascontiguousarray(arr)\n                \n            optimized.append(arr)\n            \n        return optimized\n        \n    def chunk_large_operations(\n        self, \n        data: np.ndarray, \n        operation: Callable,\n        chunk_size: int = 10000\n    ) -> np.ndarray:\n        \"\"\"Process large arrays in chunks to reduce memory usage.\"\"\"\n        if len(data) <= chunk_size:\n            return operation(data)\n            \n        results = []\n        for i in range(0, len(data), chunk_size):\n            chunk = data[i:i + chunk_size]\n            result = operation(chunk)\n            results.append(result)\n            \n        return np.concatenate(results, axis=0)\n        \n    def clear_gpu_cache(self):\n        \"\"\"Clear GPU memory cache if available.\"\"\"\n        if TORCH_AVAILABLE and torch.cuda.is_available():\n            torch.cuda.empty_cache()\n            self.logger.debug(\"Cleared GPU cache\")\n            \n    def force_garbage_collection(self):\n        \"\"\"Force garbage collection.\"\"\"\n        collected = gc.collect()\n        self.logger.debug(f\"Garbage collection freed {collected} objects\")\n        \n    def track_memory_usage(self, obj: Any) -> int:\n        \"\"\"Track memory usage of an object.\"\"\"\n        if hasattr(obj, 'nbytes'):\n            return obj.nbytes\n        elif TORCH_AVAILABLE and isinstance(obj, torch.Tensor):\n            return obj.element_size() * obj.nelement()\n        else:\n            try:\n                return len(pickle.dumps(obj))\n            except Exception:\n                return 0\n                \n    def create_memory_mapped_array(\n        self, \n        shape: Tuple[int, ...], \n        dtype: np.dtype,\n        filename: Optional[str] = None\n    ) -> np.ndarray:\n        \"\"\"Create memory-mapped array for large datasets.\"\"\"\n        if filename is None:\n            # Create temporary file\n            import tempfile\n            fd, filename = tempfile.mkstemp(suffix='.dat')\n            os.close(fd)\n            \n        return np.memmap(filename, dtype=dtype, mode='w+', shape=shape)\n\n\nclass GPUAccelerator:\n    \"\"\"GPU acceleration utilities.\"\"\"\n    \n    def __init__(self):\n        self.device = None\n        self.logger = logging.getLogger(__name__)\n        self._initialize_gpu()\n        \n    def _initialize_gpu(self):\n        \"\"\"Initialize GPU if available.\"\"\"\n        if not TORCH_AVAILABLE:\n            self.logger.warning(\"PyTorch not available, GPU acceleration disabled\")\n            return\n            \n        if torch.cuda.is_available():\n            self.device = torch.device('cuda')\n            # Optimize GPU settings\n            torch.backends.cudnn.benchmark = True\n            torch.backends.cudnn.deterministic = False\n            self.logger.info(f\"GPU acceleration enabled: {torch.cuda.get_device_name()}\")\n        else:\n            self.device = torch.device('cpu')\n            self.logger.info(\"GPU not available, using CPU\")\n            \n    def to_gpu(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:\n        \"\"\"Move data to GPU.\"\"\"\n        if not TORCH_AVAILABLE or self.device.type == 'cpu':\n            if isinstance(data, np.ndarray):\n                return torch.from_numpy(data)\n            return data\n            \n        if isinstance(data, np.ndarray):\n            tensor = torch.from_numpy(data)\n        else:\n            tensor = data\n            \n        return tensor.to(self.device, non_blocking=True)\n        \n    def to_cpu(self, data: torch.Tensor) -> np.ndarray:\n        \"\"\"Move data from GPU to CPU.\"\"\"\n        if not TORCH_AVAILABLE:\n            return data\n            \n        return data.cpu().numpy()\n        \n    def optimize_model_for_inference(self, model: nn.Module) -> nn.Module:\n        \"\"\"Optimize model for inference.\"\"\"\n        if not TORCH_AVAILABLE:\n            return model\n            \n        # Set to evaluation mode\n        model.eval()\n        \n        # Move to GPU\n        if self.device.type == 'cuda':\n            model = model.to(self.device)\n            \n        # Optimize with TorchScript if possible\n        try:\n            # Create dummy input for tracing\n            dummy_input = torch.randn(1, 2, 128, 128, 10).to(self.device)\n            model = torch.jit.trace(model, dummy_input)\n            self.logger.info(\"Model optimized with TorchScript\")\n        except Exception as e:\n            self.logger.warning(f\"TorchScript optimization failed: {e}\")\n            \n        return model\n        \n    def batch_process_on_gpu(\n        self, \n        data_list: List[np.ndarray],\n        process_func: Callable,\n        batch_size: int = 32\n    ) -> List[np.ndarray]:\n        \"\"\"Process multiple arrays on GPU in batches.\"\"\"\n        if not TORCH_AVAILABLE or self.device.type == 'cpu':\n            return [process_func(data) for data in data_list]\n            \n        results = []\n        \n        for i in range(0, len(data_list), batch_size):\n            batch = data_list[i:i + batch_size]\n            \n            # Convert to tensors and move to GPU\n            gpu_batch = [self.to_gpu(data) for data in batch]\n            \n            # Process batch\n            with torch.no_grad():\n                batch_results = [process_func(data) for data in gpu_batch]\n                \n            # Convert back to numpy and move to CPU\n            cpu_results = [self.to_cpu(result) for result in batch_results]\n            results.extend(cpu_results)\n            \n        return results\n        \n    def get_gpu_memory_info(self) -> Dict[str, float]:\n        \"\"\"Get GPU memory information.\"\"\"\n        if not TORCH_AVAILABLE or self.device.type == 'cpu':\n            return {'available': 0, 'total': 0, 'used': 0}\n            \n        allocated = torch.cuda.memory_allocated() / 1024**3  # GB\n        cached = torch.cuda.memory_reserved() / 1024**3  # GB\n        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB\n        \n        return {\n            'allocated_gb': allocated,\n            'cached_gb': cached,\n            'total_gb': total,\n            'free_gb': total - cached\n        }\n\n\nclass BatchProcessor:\n    \"\"\"Efficient batch processing for high-throughput scenarios.\"\"\"\n    \n    def __init__(\n        self, \n        batch_size: int = 32,\n        max_queue_size: int = 1000,\n        num_workers: int = 4\n    ):\n        self.batch_size = batch_size\n        self.max_queue_size = max_queue_size\n        self.num_workers = num_workers\n        \n        # Processing queues\n        self.input_queue = Queue(maxsize=max_queue_size)\n        self.output_queue = Queue(maxsize=max_queue_size)\n        \n        # Worker management\n        self.workers = []\n        self.is_processing = False\n        self.logger = logging.getLogger(__name__)\n        \n    def start_processing(self, process_func: Callable):\n        \"\"\"Start batch processing with worker threads.\"\"\"\n        if self.is_processing:\n            return\n            \n        self.is_processing = True\n        \n        # Create worker threads\n        for i in range(self.num_workers):\n            worker = threading.Thread(\n                target=self._worker_loop,\n                args=(process_func,),\n                daemon=True\n            )\n            worker.start()\n            self.workers.append(worker)\n            \n        self.logger.info(f\"Started {self.num_workers} processing workers\")\n        \n    def stop_processing(self):\n        \"\"\"Stop batch processing.\"\"\"\n        self.is_processing = False\n        \n        # Wait for workers to finish\n        for worker in self.workers:\n            worker.join(timeout=5.0)\n            \n        self.workers.clear()\n        self.logger.info(\"Stopped processing workers\")\n        \n    def submit(self, data: Any, callback: Optional[Callable] = None) -> bool:\n        \"\"\"Submit data for processing.\"\"\"\n        try:\n            self.input_queue.put((data, callback), timeout=1.0)\n            return True\n        except:\n            return False\n            \n    def get_result(self, timeout: float = 1.0) -> Optional[Tuple[Any, Optional[Callable]]]:\n        \"\"\"Get processed result.\"\"\"\n        try:\n            return self.output_queue.get(timeout=timeout)\n        except:\n            return None\n            \n    def _worker_loop(self, process_func: Callable):\n        \"\"\"Main worker loop for batch processing.\"\"\"\n        batch = []\n        \n        while self.is_processing:\n            try:\n                # Collect batch\n                while len(batch) < self.batch_size and self.is_processing:\n                    try:\n                        item = self.input_queue.get(timeout=0.1)\n                        batch.append(item)\n                    except:\n                        break\n                        \n                if not batch:\n                    continue\n                    \n                # Process batch\n                data_batch = [item[0] for item in batch]\n                callbacks = [item[1] for item in batch]\n                \n                try:\n                    results = process_func(data_batch)\n                    \n                    # Ensure results is a list\n                    if not isinstance(results, list):\n                        results = [results]\n                        \n                    # Submit results\n                    for result, callback in zip(results, callbacks):\n                        self.output_queue.put((result, callback))\n                        \n                except Exception as e:\n                    self.logger.error(f\"Batch processing error: {e}\")\n                    # Submit error results\n                    for callback in callbacks:\n                        self.output_queue.put((None, callback))\n                        \n                batch.clear()\n                \n            except Exception as e:\n                self.logger.error(f\"Worker loop error: {e}\")\n                batch.clear()\n\n\ndef cached_computation(\n    cache_key_func: Optional[Callable] = None,\n    ttl: Optional[float] = None,\n    cache_instance: Optional[CacheInterface] = None\n):\n    \"\"\"Decorator for caching expensive computations.\"\"\"\n    \n    if cache_instance is None:\n        cache_instance = LRUCache(max_size=1000, default_ttl=ttl)\n        \n    def decorator(func: Callable) -> Callable:\n        @wraps(func)\n        def wrapper(*args, **kwargs):\n            # Generate cache key\n            if cache_key_func:\n                cache_key = cache_key_func(*args, **kwargs)\n            else:\n                # Default key generation\n                key_data = (func.__name__, args, tuple(sorted(kwargs.items())))\n                cache_key = hashlib.md5(str(key_data).encode()).hexdigest()\n                \n            # Try to get from cache\n            result = cache_instance.get(cache_key)\n            if result is not None:\n                # Record cache hit\n                metrics = get_metrics_collector()\n                metrics.record_events_processed(1)  # Use as cache hit counter\n                return result\n                \n            # Compute result\n            start_time = time.time()\n            result = func(*args, **kwargs)\n            computation_time = time.time() - start_time\n            \n            # Cache result\n            cache_instance.put(cache_key, result, ttl)\n            \n            # Record metrics\n            metrics = get_metrics_collector()\n            metrics.record_inference_latency(computation_time * 1000)  # ms\n            \n            return result\n            \n        # Add cache management methods\n        wrapper._cache = cache_instance\n        wrapper.clear_cache = cache_instance.clear\n        wrapper.cache_stats = cache_instance.stats\n        \n        return wrapper\n    return decorator\n\n\n@dataclass\nclass PerformanceProfile:\n    \"\"\"Performance profiling results.\"\"\"\n    function_name: str\n    total_time: float\n    call_count: int\n    avg_time_per_call: float\n    memory_peak_mb: float\n    gpu_memory_peak_mb: float = 0.0\n    \n\nclass PerformanceProfiler:\n    \"\"\"Performance profiling and analysis tools.\"\"\"\n    \n    def __init__(self):\n        self.profiles: Dict[str, List[float]] = defaultdict(list)\n        self.memory_usage: Dict[str, List[float]] = defaultdict(list)\n        self.gpu_accelerator = GPUAccelerator() if TORCH_AVAILABLE else None\n        self.logger = logging.getLogger(__name__)\n        \n    def profile_function(self, func_name: Optional[str] = None):\n        \"\"\"Decorator for profiling function performance.\"\"\"\n        def decorator(func: Callable) -> Callable:\n            name = func_name or f\"{func.__module__}.{func.__qualname__}\"\n            \n            @wraps(func)\n            def wrapper(*args, **kwargs):\n                # Memory before\n                mem_before = self._get_memory_usage()\n                gpu_mem_before = self._get_gpu_memory_usage()\n                \n                # Timing\n                start_time = time.perf_counter()\n                result = func(*args, **kwargs)\n                end_time = time.perf_counter()\n                \n                # Memory after\n                mem_after = self._get_memory_usage()\n                gpu_mem_after = self._get_gpu_memory_usage()\n                \n                # Record measurements\n                execution_time = end_time - start_time\n                memory_delta = mem_after - mem_before\n                gpu_memory_delta = gpu_mem_after - gpu_mem_before\n                \n                self.profiles[name].append(execution_time)\n                self.memory_usage[name].append(memory_delta)\n                \n                return result\n                \n            return wrapper\n        return decorator\n        \n    def _get_memory_usage(self) -> float:\n        \"\"\"Get current memory usage in MB.\"\"\"\n        try:\n            import psutil\n            process = psutil.Process()\n            return process.memory_info().rss / 1024 / 1024\n        except ImportError:\n            return 0.0\n            \n    def _get_gpu_memory_usage(self) -> float:\n        \"\"\"Get current GPU memory usage in MB.\"\"\"\n        if self.gpu_accelerator and TORCH_AVAILABLE:\n            info = self.gpu_accelerator.get_gpu_memory_info()\n            return info.get('allocated_gb', 0) * 1024\n        return 0.0\n        \n    def get_profile_report(self) -> Dict[str, PerformanceProfile]:\n        \"\"\"Generate performance profile report.\"\"\"\n        report = {}\n        \n        for func_name, times in self.profiles.items():\n            if not times:\n                continue\n                \n            memory_deltas = self.memory_usage.get(func_name, [0])\n            \n            profile = PerformanceProfile(\n                function_name=func_name,\n                total_time=sum(times),\n                call_count=len(times),\n                avg_time_per_call=np.mean(times),\n                memory_peak_mb=max(memory_deltas) if memory_deltas else 0.0\n            )\n            \n            report[func_name] = profile\n            \n        return report\n        \n    def print_report(self):\n        \"\"\"Print performance profile report.\"\"\"\n        report = self.get_profile_report()\n        \n        print(\"\\n\" + \"=\"*80)\n        print(\"PERFORMANCE PROFILE REPORT\")\n        print(\"=\"*80)\n        \n        for func_name, profile in sorted(report.items(), \n                                       key=lambda x: x[1].total_time, \n                                       reverse=True):\n            print(f\"\\nFunction: {func_name}\")\n            print(f\"  Total Time:     {profile.total_time:.4f}s\")\n            print(f\"  Call Count:     {profile.call_count}\")\n            print(f\"  Avg Time/Call:  {profile.avg_time_per_call:.4f}s\")\n            print(f\"  Peak Memory:    {profile.memory_peak_mb:.1f}MB\")\n            \n        print(\"=\"*80)\n        \n    def clear_profiles(self):\n        \"\"\"Clear all profiling data.\"\"\"\n        self.profiles.clear()\n        self.memory_usage.clear()\n        self.logger.info(\"Cleared performance profiles\")\n\n\n# Global instances\n_global_model_cache = None\n_global_memory_optimizer = None\n_global_gpu_accelerator = None\n_global_profiler = None\n\n\ndef get_model_cache() -> ModelCache:\n    \"\"\"Get global model cache instance.\"\"\"\n    global _global_model_cache\n    if _global_model_cache is None:\n        _global_model_cache = ModelCache()\n    return _global_model_cache\n\n\ndef get_memory_optimizer() -> MemoryOptimizer:\n    \"\"\"Get global memory optimizer instance.\"\"\"\n    global _global_memory_optimizer\n    if _global_memory_optimizer is None:\n        _global_memory_optimizer = MemoryOptimizer()\n    return _global_memory_optimizer\n\n\ndef get_gpu_accelerator() -> GPUAccelerator:\n    \"\"\"Get global GPU accelerator instance.\"\"\"\n    global _global_gpu_accelerator\n    if _global_gpu_accelerator is None:\n        _global_gpu_accelerator = GPUAccelerator()\n    return _global_gpu_accelerator\n\n\ndef get_performance_profiler() -> PerformanceProfiler:\n    \"\"\"Get global performance profiler instance.\"\"\"\n    global _global_profiler\n    if _global_profiler is None:\n        _global_profiler = PerformanceProfiler()\n    return _global_profiler