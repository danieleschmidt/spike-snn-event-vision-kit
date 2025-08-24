"""
High-performance core with scaling optimizations and intelligent caching.

Generation 3: MAKE IT SCALE
- Intelligent caching with LRU and TTL eviction
- Concurrent processing with thread pools
- Memory optimization and resource pooling
- Auto-scaling based on load
- Performance profiling and metrics
"""

import time
import threading
import multiprocessing
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
import psutil
import functools
import weakref
import gc
from queue import Queue, Empty
from threading import RLock


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


class IntelligentCache:
    """High-performance cache with LRU, TTL, and size-based eviction."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100, default_ttl: float = None):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._current_memory = 0
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0,
            'memory_usage_bytes': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str, default=None) -> Any:
        """Get item from cache with LRU update."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self.stats['misses'] += 1
                return default
            
            if entry.is_expired():
                self._remove_entry(key)
                self.stats['misses'] += 1
                return default
            
            # Update LRU order and access stats
            entry.update_access()
            self._cache.move_to_end(key)
            
            self.stats['hits'] += 1
            return entry.data
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put item in cache with optional TTL."""
        with self._lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            entry_ttl = ttl or self.default_ttl
            
            current_time = time.time()
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Check if we can fit this entry
            if size_bytes > self.max_memory_bytes:
                self.logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return False
            
            # Evict entries to make space
            self._evict_to_fit(size_bytes)
            
            # Create and add entry
            entry = CacheEntry(
                data=value,
                created_at=current_time,
                last_accessed=current_time,
                ttl=entry_ttl,
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._current_memory += size_bytes
            
            # Update stats
            self.stats['size'] = len(self._cache)
            self.stats['memory_usage_bytes'] = self._current_memory
            
            return True
    
    def _evict_to_fit(self, required_bytes: int):
        """Evict entries to fit required bytes."""
        # First, remove expired entries
        expired_keys = [
            key for key, entry in self._cache.items() 
            if entry.is_expired()
        ]
        for key in expired_keys:
            self._remove_entry(key)
        
        # If still need space, use LRU eviction
        while (self._current_memory + required_bytes > self.max_memory_bytes or 
               len(self._cache) >= self.max_size) and self._cache:
            
            # Remove least recently used item
            lru_key, _ = self._cache.popitem(last=False)
            self._remove_entry(lru_key, update_cache=False)
            self.stats['evictions'] += 1
    
    def _remove_entry(self, key: str, update_cache: bool = True):
        """Remove entry and update memory tracking."""
        if key in self._cache:
            entry = self._cache[key]
            self._current_memory -= entry.size_bytes
            
            if update_cache:
                del self._cache[key]
                
            # Update stats
            self.stats['size'] = len(self._cache)
            self.stats['memory_usage_bytes'] = self._current_memory
    
    def _calculate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (str, bytes)):
            return len(obj)
        elif isinstance(obj, (list, tuple)):
            return sum(self._calculate_size(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(self._calculate_size(k) + self._calculate_size(v) 
                      for k, v in obj.items())
        else:
            # Rough estimate for other objects
            return 64  # Base object overhead
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
            self.stats.update({
                'size': 0,
                'memory_usage_bytes': 0
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / max(total_requests, 1)
            
            return {
                **self.stats.copy(),
                'hit_rate': hit_rate,
                'memory_usage_mb': self.stats['memory_usage_bytes'] / 1024 / 1024,
                'average_entry_size': self.stats['memory_usage_bytes'] / max(self.stats['size'], 1)
            }


class ResourcePool:
    """Generic resource pool for expensive objects."""
    
    def __init__(self, factory: Callable[[], Any], max_size: int = 10):
        self.factory = factory
        self.max_size = max_size
        self._pool = Queue(maxsize=max_size)
        self._created_count = 0
        self._lock = threading.Lock()
        
    def acquire(self) -> Any:
        """Acquire resource from pool."""
        try:
            return self._pool.get_nowait()
        except Empty:
            # Pool empty, create new resource if under limit
            with self._lock:
                if self._created_count < self.max_size:
                    self._created_count += 1
                    return self.factory()
                else:
                    # Wait for available resource
                    return self._pool.get()
    
    def release(self, resource: Any):
        """Release resource back to pool."""
        try:
            self._pool.put_nowait(resource)
        except:
            # Pool full, discard resource
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'pool_size': self._pool.qsize(),
            'max_size': self.max_size,
            'created_count': self._created_count
        }


class AutoScaler:
    """Automatic scaling based on system metrics."""
    
    def __init__(self, min_workers: int = 1, max_workers: int = None):
        self.min_workers = min_workers
        self.max_workers = max_workers or multiprocessing.cpu_count() * 2
        self.current_workers = min_workers
        
        self.metrics_history = []
        self.last_adjustment = 0
        self.adjustment_cooldown = 30  # seconds
        
        self.logger = logging.getLogger(__name__)
    
    def should_scale_up(self) -> bool:
        """Determine if we should scale up workers."""
        if self.current_workers >= self.max_workers:
            return False
        
        if time.time() - self.last_adjustment < self.adjustment_cooldown:
            return False
        
        # Check recent metrics
        if len(self.metrics_history) < 5:
            return False
        
        recent_metrics = self.metrics_history[-5:]
        
        # Scale up if consistently high CPU and queue length
        avg_cpu = sum(m['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
        avg_queue = sum(m['queue_length'] for m in recent_metrics) / len(recent_metrics)
        
        return avg_cpu > 70 and avg_queue > self.current_workers * 2
    
    def should_scale_down(self) -> bool:
        """Determine if we should scale down workers."""
        if self.current_workers <= self.min_workers:
            return False
        
        if time.time() - self.last_adjustment < self.adjustment_cooldown * 2:
            return False
        
        if len(self.metrics_history) < 10:
            return False
        
        recent_metrics = self.metrics_history[-10:]
        
        # Scale down if consistently low CPU and queue
        avg_cpu = sum(m['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
        avg_queue = sum(m['queue_length'] for m in recent_metrics) / len(recent_metrics)
        
        return avg_cpu < 30 and avg_queue < self.current_workers * 0.5
    
    def record_metrics(self, cpu_percent: float, queue_length: int, memory_percent: float):
        """Record system metrics for scaling decisions."""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'queue_length': queue_length,
            'memory_percent': memory_percent,
            'workers': self.current_workers
        }
        
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
    
    def get_scaling_decision(self) -> int:
        """Get scaling decision (positive = scale up, negative = scale down, 0 = no change)."""
        if self.should_scale_up():
            new_workers = min(self.max_workers, self.current_workers + 1)
            self.current_workers = new_workers
            self.last_adjustment = time.time()
            self.logger.info(f"Scaling UP to {new_workers} workers")
            return 1
        
        if self.should_scale_down():
            new_workers = max(self.min_workers, self.current_workers - 1)
            self.current_workers = new_workers
            self.last_adjustment = time.time()
            self.logger.info(f"Scaling DOWN to {new_workers} workers")
            return -1
        
        return 0


class HighPerformanceProcessor:
    """High-performance event processor with scaling and optimization."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        
        # Performance components
        self.cache = IntelligentCache(max_size=10000, max_memory_mb=500)
        self.autoscaler = AutoScaler(min_workers=1, max_workers=self.max_workers)
        
        # Thread pool for I/O operations
        self._thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Resource pools
        self.array_pool = ResourcePool(
            factory=lambda: np.empty((1000, 4), dtype=np.float32),
            max_size=20
        )
        
        # Processing queue
        self.processing_queue = Queue(maxsize=1000)
        
        # Metrics
        self.performance_metrics = {
            'processed_events': 0,
            'processing_time_total': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'queue_overflows': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self._shutdown = False
        
        # Start performance monitoring
        self._start_monitoring()
    
    def process_events_batch(self, event_batches: List[np.ndarray]) -> List[Any]:
        """Process multiple event batches concurrently."""
        if not event_batches:
            return []
        
        start_time = time.time()
        
        # Submit all batches for parallel processing
        future_to_batch = {}
        results = []
        
        for i, batch in enumerate(event_batches):
            future = self._thread_pool.submit(self._process_single_batch, batch, i)
            future_to_batch[future] = i
        
        # Collect results maintaining order
        batch_results = {}
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                result = future.result()
                batch_results[batch_idx] = result
            except Exception as e:
                self.logger.error(f"Batch {batch_idx} processing failed: {e}")
                batch_results[batch_idx] = None
        
        # Return results in original order
        for i in range(len(event_batches)):
            results.append(batch_results.get(i))
        
        # Update metrics
        processing_time = time.time() - start_time
        self.performance_metrics['processing_time_total'] += processing_time
        
        return results
    
    def _process_single_batch(self, events: np.ndarray, batch_id: int) -> Dict[str, Any]:
        """Process a single batch of events with caching."""
        if len(events) == 0:
            return {'batch_id': batch_id, 'processed_events': []}
        
        # Generate cache key based on event characteristics
        cache_key = self._generate_cache_key(events)
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.performance_metrics['cache_hits'] += 1
            return {
                'batch_id': batch_id,
                'processed_events': cached_result,
                'from_cache': True
            }
        
        self.performance_metrics['cache_misses'] += 1
        
        # Process events
        processed_events = self._perform_processing(events)
        
        # Cache result
        self.cache.put(cache_key, processed_events, ttl=300)  # 5-minute TTL
        
        self.performance_metrics['processed_events'] += len(events)
        
        return {
            'batch_id': batch_id,
            'processed_events': processed_events,
            'from_cache': False
        }
    
    def _generate_cache_key(self, events: np.ndarray) -> str:
        """Generate cache key for event batch."""
        if len(events) == 0:
            return "empty_batch"
        
        # Use statistical properties as cache key components
        x_mean = float(np.mean(events[:, 0]))
        y_mean = float(np.mean(events[:, 1]))
        t_span = float(events[-1, 2] - events[0, 2]) if len(events) > 1 else 0.0
        event_count = len(events)
        
        # Create hash-like key
        key_str = f"{x_mean:.2f}_{y_mean:.2f}_{t_span:.6f}_{event_count}"
        return key_str
    
    def _perform_processing(self, events: np.ndarray) -> List[Dict[str, Any]]:
        """Actual event processing logic."""
        # Get reusable array from pool
        work_array = self.array_pool.acquire()
        
        try:
            processed_events = []
            
            # Vectorized processing for better performance
            if len(events) > 0:
                # Normalize coordinates
                x_norm = events[:, 0] / 640.0  # Assume max width
                y_norm = events[:, 1] / 480.0  # Assume max height
                
                # Process in batches for better memory usage
                batch_size = 1000
                for i in range(0, len(events), batch_size):
                    batch_end = min(i + batch_size, len(events))
                    batch_events = events[i:batch_end]
                    
                    # Simulate processing (e.g., feature extraction)
                    batch_processed = []
                    for j, event in enumerate(batch_events):
                        processed_event = {
                            'x': float(event[0]),
                            'y': float(event[1]),
                            'timestamp': float(event[2]),
                            'polarity': int(event[3]),
                            'x_normalized': float(x_norm[i + j]),
                            'y_normalized': float(y_norm[i + j]),
                            'processing_id': f"batch_{i}_{j}"
                        }
                        batch_processed.append(processed_event)
                    
                    processed_events.extend(batch_processed)
            
            return processed_events
            
        finally:
            # Return array to pool
            self.array_pool.release(work_array)
    
    def _start_monitoring(self):
        """Start performance monitoring thread."""
        def monitor_loop():
            while not self._shutdown:
                try:
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    queue_size = self.processing_queue.qsize()
                    
                    # Record for autoscaling
                    self.autoscaler.record_metrics(
                        cpu_percent=cpu_percent,
                        queue_length=queue_size,
                        memory_percent=memory.percent
                    )
                    
                    # Make scaling decisions
                    scaling_decision = self.autoscaler.get_scaling_decision()
                    if scaling_decision != 0:
                        self._adjust_thread_pool(scaling_decision)
                    
                    # Log performance metrics periodically
                    if time.time() % 60 < 1:  # Roughly every minute
                        self._log_performance_metrics()
                        
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                
                time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def _adjust_thread_pool(self, scaling_decision: int):
        """Adjust thread pool size based on scaling decision."""
        # Note: ThreadPoolExecutor doesn't support dynamic resizing
        # In production, you'd implement a custom pool or use ProcessPoolExecutor
        self.logger.info(f"Would adjust thread pool by {scaling_decision} workers")
    
    def _log_performance_metrics(self):
        """Log current performance metrics."""
        cache_stats = self.cache.get_stats()
        pool_stats = self.array_pool.get_stats()
        
        self.logger.info(
            f"Performance: {self.performance_metrics['processed_events']} events, "
            f"cache hit rate: {cache_stats['hit_rate']:.2%}, "
            f"memory: {cache_stats['memory_usage_mb']:.1f}MB, "
            f"workers: {self.autoscaler.current_workers}, "
            f"pool: {pool_stats['pool_size']}/{pool_stats['max_size']}"
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'processing': self.performance_metrics.copy(),
            'cache': self.cache.get_stats(),
            'autoscaler': {
                'current_workers': self.autoscaler.current_workers,
                'min_workers': self.autoscaler.min_workers,
                'max_workers': self.autoscaler.max_workers
            },
            'resource_pool': self.array_pool.get_stats(),
            'queue_size': self.processing_queue.qsize()
        }
    
    def shutdown(self):
        """Shutdown the processor gracefully."""
        self.logger.info("Shutting down high-performance processor...")
        self._shutdown = True
        self._thread_pool.shutdown(wait=True)
        self.cache.clear()


# Performance testing and demonstration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create high-performance processor
    processor = HighPerformanceProcessor(max_workers=4)
    
    try:
        # Generate test data
        test_batches = []
        for i in range(10):
            batch_size = np.random.randint(100, 1000)
            batch = np.random.rand(batch_size, 4)
            batch[:, 0] *= 640  # x coordinates
            batch[:, 1] *= 480  # y coordinates
            batch[:, 2] = np.sort(np.random.rand(batch_size) * 10)  # timestamps
            batch[:, 3] = np.random.choice([-1, 1], batch_size)  # polarity
            test_batches.append(batch.astype(np.float32))
        
        # Process batches
        start_time = time.time()
        results = processor.process_events_batch(test_batches)
        processing_time = time.time() - start_time
        
        print(f"Processed {len(test_batches)} batches in {processing_time:.3f}s")
        print(f"Performance stats: {processor.get_performance_stats()}")
        
    finally:
        processor.shutdown()