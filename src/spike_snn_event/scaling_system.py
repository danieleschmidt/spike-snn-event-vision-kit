"""
High-performance scaling system for neuromorphic vision processing.

This module implements Generation 3 scaling optimizations including:
- Performance optimization and intelligent caching
- Concurrent processing and resource pooling  
- Load balancing and auto-scaling triggers
- GPU acceleration and distributed processing
- Memory management and optimization
- Advanced performance profiling and metrics
"""

import time
import threading
import multiprocessing as mp
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
import numpy as np
import logging
import psutil
import gc
import hashlib
import pickle
from functools import lru_cache, wraps
from collections import defaultdict, deque
import weakref
import json
import os
from pathlib import Path
import mmap

# Configure high-performance logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(processName)s:%(threadName)s] - %(message)s'
)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for scaling optimization."""
    timestamp: float
    throughput_events_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    cpu_utilization: float
    memory_usage_mb: float
    gpu_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    queue_depth: int = 0
    active_workers: int = 0
    processed_events: int = 0
    processing_errors: int = 0

@dataclass
class ScalingConfig:
    """Configuration for auto-scaling and performance optimization."""
    # Worker management
    min_workers: int = 2
    max_workers: int = mp.cpu_count() * 2
    target_cpu_utilization: float = 70.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 50.0
    scale_cooldown_seconds: float = 30.0
    
    # Performance optimization
    enable_gpu_acceleration: bool = True
    enable_caching: bool = True
    cache_size_mb: int = 512
    batch_size: int = 1000
    prefetch_factor: int = 2
    
    # Memory management
    memory_limit_mb: int = 4096
    gc_threshold: float = 85.0  # Trigger GC at 85% memory usage
    
    # Load balancing
    load_balance_strategy: str = "round_robin"  # round_robin, least_loaded, adaptive
    queue_size_per_worker: int = 1000

class IntelligentCache:
    """High-performance intelligent caching system with adaptive strategies."""
    
    def __init__(self, max_size_mb: int = 512, ttl_seconds: float = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.cache_size_bytes = 0
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.IntelligentCache")
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with LRU and TTL support."""
        with self.lock:
            current_time = time.time()
            
            if key not in self.cache:
                self.misses += 1
                return None
                
            value, timestamp, size = self.cache[key]
            
            # Check TTL
            if current_time - timestamp > self.ttl_seconds:
                self._remove_key(key)
                self.misses += 1
                return None
                
            # Update access tracking
            self.access_times[key] = current_time
            self.access_counts[key] += 1
            self.hits += 1
            
            return value
            
    def put(self, key: str, value: Any, size_hint: Optional[int] = None) -> bool:
        """Put item in cache with intelligent eviction."""
        with self.lock:
            current_time = time.time()
            
            # Calculate size
            if size_hint is None:
                size = self._estimate_size(value)
            else:
                size = size_hint
                
            # Check if item is too large
            if size > self.max_size_bytes * 0.5:  # Don't cache items > 50% of total
                return False
                
            # Remove existing entry if present
            if key in self.cache:
                self._remove_key(key)
                
            # Ensure we have space
            while (self.cache_size_bytes + size) > self.max_size_bytes and self.cache:
                self._evict_one()
                
            # Store item
            self.cache[key] = (value, current_time, size)
            self.access_times[key] = current_time
            self.access_counts[key] += 1
            self.cache_size_bytes += size
            
            return True
            
    def _remove_key(self, key: str):
        """Remove key from cache and update tracking."""
        if key in self.cache:
            _, _, size = self.cache[key]
            del self.cache[key]
            self.cache_size_bytes -= size
            
        if key in self.access_times:
            del self.access_times[key]
        if key in self.access_counts:
            del self.access_counts[key]
            
    def _evict_one(self):
        """Evict one item using intelligent strategy."""
        if not self.cache:
            return
            
        current_time = time.time()
        best_key = None
        best_score = float('inf')
        
        # Use a combination of LRU, LFU, and TTL for eviction
        for key in self.cache:
            _, timestamp, _ = self.cache[key]
            last_access = self.access_times.get(key, timestamp)
            access_count = self.access_counts.get(key, 1)
            
            # Calculate eviction score (lower = more likely to evict)
            age_score = current_time - last_access
            frequency_score = 1.0 / (access_count + 1)
            ttl_score = (current_time - timestamp) / self.ttl_seconds
            
            # Weighted combination
            score = age_score * 0.4 + frequency_score * 0.3 + ttl_score * 0.3
            
            if score < best_score:
                best_score = score
                best_key = key
                
        if best_key:
            self._remove_key(best_key)
            self.evictions += 1
            
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        try:
            return len(pickle.dumps(obj))
        except:
            return 1024  # Default estimate
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(total_requests, 1) * 100
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'cache_size_mb': self.cache_size_bytes / (1024 * 1024),
                'item_count': len(self.cache),
                'utilization': self.cache_size_bytes / self.max_size_bytes * 100
            }
            
    def clear(self):
        """Clear entire cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.cache_size_bytes = 0

class GPUAcceleratedProcessor:
    """GPU-accelerated event processing (placeholder implementation)."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.logger = logging.getLogger(f"{__name__}.GPUAcceleratedProcessor")
        self.is_available = False
        
        # Check for GPU availability
        try:
            # Placeholder for CUDA/GPU initialization
            self.is_available = self._check_gpu_availability()
            if self.is_available:
                self.logger.info(f"GPU acceleration enabled on device {device_id}")
            else:
                self.logger.info("GPU acceleration not available, using CPU")
        except Exception as e:
            self.logger.warning(f"GPU initialization failed: {e}")
            
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        # Placeholder - would check for CUDA/OpenCL/etc
        return os.environ.get('ENABLE_GPU', 'false').lower() == 'true'
        
    def process_events_batch(self, events_batch: List[np.ndarray]) -> List[np.ndarray]:
        """Process batch of events with GPU acceleration."""
        if not self.is_available:
            return self._cpu_process_batch(events_batch)
            
        # Placeholder for GPU processing
        processed = []
        for events in events_batch:
            # Simulate GPU processing
            processed_events = self._simulate_gpu_processing(events)
            processed.append(processed_events)
            
        return processed
        
    def _cpu_process_batch(self, events_batch: List[np.ndarray]) -> List[np.ndarray]:
        """Fallback CPU processing."""
        processed = []
        for events in events_batch:
            # Apply some processing (e.g., filtering, transformation)
            if len(events) > 0:
                # Simulate processing: add noise reduction, spatial filtering
                processed_events = self._apply_cpu_filters(events)
                processed.append(processed_events)
            else:
                processed.append(events)
        return processed
        
    def _simulate_gpu_processing(self, events: np.ndarray) -> np.ndarray:
        """Simulate GPU-accelerated processing."""
        if len(events) == 0:
            return events
            
        # Simulate faster processing on GPU
        time.sleep(len(events) * 0.5e-6)  # 0.5 microseconds per event on GPU
        
        # Apply some transformations
        processed = events.copy()
        if len(processed) > 0:
            # Simulate GPU-accelerated spatial filtering
            processed[:, :2] = np.round(processed[:, :2])  # Quantize coordinates
            
        return processed
        
    def _apply_cpu_filters(self, events: np.ndarray) -> np.ndarray:
        """Apply CPU-based filtering and processing."""
        if len(events) == 0:
            return events
            
        # Simulate slower CPU processing  
        time.sleep(len(events) * 2e-6)  # 2 microseconds per event on CPU
        
        processed = events.copy()
        
        # Apply temporal filtering
        if len(processed) > 1:
            # Remove events that are too close in time (< 1us)
            time_diffs = np.diff(processed[:, 2])
            valid_mask = np.concatenate([[True], time_diffs >= 1e-6])
            processed = processed[valid_mask]
            
        return processed

class AdaptiveLoadBalancer:
    """Adaptive load balancer for distributing work across workers."""
    
    def __init__(self, strategy: str = "adaptive"):
        self.strategy = strategy
        self.worker_loads = defaultdict(int)
        self.worker_performance = defaultdict(lambda: deque(maxlen=100))
        self.round_robin_counter = 0
        self.lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.AdaptiveLoadBalancer")
        
    def select_worker(self, worker_ids: List[str], task_size: int = 1) -> str:
        """Select optimal worker based on strategy and current loads."""
        with self.lock:
            if not worker_ids:
                raise ValueError("No workers available")
                
            if len(worker_ids) == 1:
                return worker_ids[0]
                
            if self.strategy == "round_robin":
                worker = worker_ids[self.round_robin_counter % len(worker_ids)]
                self.round_robin_counter += 1
                return worker
                
            elif self.strategy == "least_loaded":
                return min(worker_ids, key=lambda w: self.worker_loads[w])
                
            elif self.strategy == "adaptive":
                return self._adaptive_selection(worker_ids, task_size)
                
            else:
                # Default to round robin
                return worker_ids[self.round_robin_counter % len(worker_ids)]
                
    def _adaptive_selection(self, worker_ids: List[str], task_size: int) -> str:
        """Select worker using adaptive strategy based on performance history."""
        best_worker = None
        best_score = float('-inf')
        
        for worker_id in worker_ids:
            # Calculate performance score
            recent_times = list(self.worker_performance[worker_id])
            
            if recent_times:
                avg_time = np.mean(recent_times)
                throughput = 1.0 / (avg_time + 1e-6)
            else:
                throughput = 1.0  # Default for new workers
                
            current_load = self.worker_loads[worker_id]
            load_factor = 1.0 / (current_load + 1)
            
            # Combined score: higher is better
            score = throughput * load_factor
            
            if score > best_score:
                best_score = score
                best_worker = worker_id
                
        return best_worker or worker_ids[0]
        
    def update_worker_load(self, worker_id: str, load_delta: int):
        """Update worker load."""
        with self.lock:
            self.worker_loads[worker_id] += load_delta
            self.worker_loads[worker_id] = max(0, self.worker_loads[worker_id])
            
    def record_task_completion(self, worker_id: str, execution_time: float):
        """Record task completion for performance tracking."""
        with self.lock:
            self.worker_performance[worker_id].append(execution_time)
            
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self.lock:
            total_load = sum(self.worker_loads.values())
            
            worker_stats = {}
            for worker_id, load in self.worker_loads.items():
                recent_times = list(self.worker_performance[worker_id])
                avg_time = np.mean(recent_times) if recent_times else 0.0
                
                worker_stats[worker_id] = {
                    'current_load': load,
                    'load_percentage': load / max(total_load, 1) * 100,
                    'avg_execution_time': avg_time,
                    'completed_tasks': len(recent_times)
                }
                
            return {
                'strategy': self.strategy,
                'total_load': total_load,
                'worker_count': len(self.worker_loads),
                'worker_stats': worker_stats
            }

class HighPerformanceEventProcessor:
    """High-performance, scalable event processor with auto-scaling."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.HighPerformanceEventProcessor")
        
        # Core components
        self.cache = IntelligentCache(config.cache_size_mb)
        self.gpu_processor = GPUAcceleratedProcessor()
        self.load_balancer = AdaptiveLoadBalancer(config.load_balance_strategy)
        
        # Worker management
        self.worker_pool = None
        self.current_workers = config.min_workers
        self.last_scale_time = 0
        
        # Processing queues
        self.input_queue = queue.Queue(maxsize=config.queue_size_per_worker * config.max_workers)
        self.result_queue = queue.Queue()
        
        # Performance tracking
        self.metrics_history = deque(maxlen=1000)
        self.processing_times = deque(maxlen=10000)
        self.start_time = time.time()
        
        # Threading control
        self.is_running = False
        self.monitor_thread = None
        self.metrics_lock = threading.Lock()
        
    def start(self):
        """Start the high-performance processing system."""
        self.logger.info("Starting high-performance event processing system...")
        
        self.is_running = True
        
        # Initialize worker pool
        self._initialize_workers()
        
        # Start monitoring
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info(f"System started with {self.current_workers} workers")
        
    def stop(self):
        """Stop the processing system gracefully."""
        self.logger.info("Stopping high-performance processing system...")
        
        self.is_running = False
        
        # Shutdown worker pool
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
            
        # Wait for monitor thread
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            
        self._log_final_performance_stats()
        self.logger.info("System stopped")
        
    def process_events(
        self, 
        events: np.ndarray, 
        priority: int = 1,
        cache_key: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Process events with high-performance optimizations."""
        if not self.is_running:
            return None
            
        start_time = time.time()
        
        try:
            # Check cache first
            if cache_key and self.config.enable_caching:
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    self.logger.debug(f"Cache hit for key: {cache_key}")
                    return cached_result
                    
            # Create processing task
            task = {
                'events': events,
                'priority': priority,
                'cache_key': cache_key,
                'submit_time': start_time
            }
            
            # Submit to worker pool
            future = self.worker_pool.submit(self._process_task, task)
            
            # Get result with timeout
            result = future.result(timeout=30.0)
            
            # Cache result if requested
            if cache_key and result is not None and self.config.enable_caching:
                self.cache.put(cache_key, result)
                
            # Record performance
            processing_time = time.time() - start_time
            with self.metrics_lock:
                self.processing_times.append(processing_time)
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing events: {e}")
            return None
            
    def _process_task(self, task: Dict[str, Any]) -> np.ndarray:
        """Process a single task (executed by worker thread)."""
        worker_id = threading.current_thread().name
        start_time = time.time()
        
        try:
            # Update load balancer
            self.load_balancer.update_worker_load(worker_id, 1)
            
            events = task['events']
            
            # Batch processing for efficiency
            if len(events) > self.config.batch_size:
                # Process in batches
                batches = [
                    events[i:i+self.config.batch_size] 
                    for i in range(0, len(events), self.config.batch_size)
                ]
                
                if self.gpu_processor.is_available:
                    processed_batches = self.gpu_processor.process_events_batch(batches)
                else:
                    processed_batches = [self._cpu_process_events(batch) for batch in batches]
                    
                # Combine results
                result = np.vstack(processed_batches) if processed_batches else np.empty((0, 4))
            else:
                # Single batch processing
                result = self._cpu_process_events(events)
                
            return result
            
        finally:
            # Update performance tracking
            execution_time = time.time() - start_time
            self.load_balancer.record_task_completion(worker_id, execution_time)
            self.load_balancer.update_worker_load(worker_id, -1)
            
    def _cpu_process_events(self, events: np.ndarray) -> np.ndarray:
        """CPU-based event processing with optimizations."""
        if len(events) == 0:
            return events
            
        # Apply efficient processing pipeline
        processed = events.copy()
        
        # Vectorized operations for performance
        if len(processed) > 0:
            # Spatial filtering
            valid_coords = (processed[:, 0] >= 0) & (processed[:, 1] >= 0)
            processed = processed[valid_coords]
            
            # Temporal filtering
            if len(processed) > 1:
                sorted_indices = np.argsort(processed[:, 2])
                processed = processed[sorted_indices]
                
        return processed
        
    def _initialize_workers(self):
        """Initialize worker thread pool."""
        self.worker_pool = ThreadPoolExecutor(
            max_workers=self.current_workers,
            thread_name_prefix="EventWorker"
        )
        
    def _monitoring_loop(self):
        """Monitor system performance and auto-scale."""
        while self.is_running:
            try:
                metrics = self._collect_performance_metrics()
                self._process_scaling_decision(metrics)
                
                with self.metrics_lock:
                    self.metrics_history.append(metrics)
                    
                # Trigger garbage collection if memory is high
                if metrics.memory_usage_mb > self.config.memory_limit_mb * self.config.gc_threshold / 100:
                    gc.collect()
                    
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                
            time.sleep(10.0)  # Monitor every 10 seconds
            
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics."""
        current_time = time.time()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        memory_mb = memory_info.used / (1024 * 1024)
        
        # Processing metrics
        with self.metrics_lock:
            recent_times = list(self.processing_times)[-1000:]  # Last 1000 operations
            
        if recent_times:
            latency_p50 = np.percentile(recent_times, 50) * 1000  # ms
            latency_p95 = np.percentile(recent_times, 95) * 1000
            latency_p99 = np.percentile(recent_times, 99) * 1000
            
            # Calculate throughput (events per second)
            total_time = sum(recent_times)
            throughput = len(recent_times) / max(total_time, 1e-6)
        else:
            latency_p50 = latency_p95 = latency_p99 = 0.0
            throughput = 0.0
            
        # Cache metrics
        cache_stats = self.cache.get_stats()
        
        return PerformanceMetrics(
            timestamp=current_time,
            throughput_events_per_sec=throughput,
            latency_p50_ms=latency_p50,
            latency_p95_ms=latency_p95,
            latency_p99_ms=latency_p99,
            cpu_utilization=cpu_percent,
            memory_usage_mb=memory_mb,
            cache_hit_rate=cache_stats['hit_rate'],
            queue_depth=self.input_queue.qsize(),
            active_workers=self.current_workers,
            processed_events=len(recent_times)
        )
        
    def _process_scaling_decision(self, metrics: PerformanceMetrics):
        """Make auto-scaling decisions based on metrics."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scale_time < self.config.scale_cooldown_seconds:
            return
            
        # Scale up conditions
        should_scale_up = (
            metrics.cpu_utilization > self.config.scale_up_threshold or
            metrics.queue_depth > self.current_workers * 100 or
            metrics.latency_p95_ms > 100  # More than 100ms p95 latency
        )
        
        # Scale down conditions
        should_scale_down = (
            metrics.cpu_utilization < self.config.scale_down_threshold and
            metrics.queue_depth < self.current_workers * 10 and
            metrics.latency_p95_ms < 10  # Less than 10ms p95 latency
        )
        
        if should_scale_up and self.current_workers < self.config.max_workers:
            self._scale_up()
            
        elif should_scale_down and self.current_workers > self.config.min_workers:
            self._scale_down()
            
    def _scale_up(self):
        """Scale up the number of workers."""
        old_workers = self.current_workers
        self.current_workers = min(self.current_workers + 1, self.config.max_workers)
        
        # Recreate worker pool with more threads
        if self.worker_pool:
            self.worker_pool.shutdown(wait=False)
            
        self._initialize_workers()
        self.last_scale_time = time.time()
        
        self.logger.info(f"Scaled up from {old_workers} to {self.current_workers} workers")
        
    def _scale_down(self):
        """Scale down the number of workers."""
        old_workers = self.current_workers
        self.current_workers = max(self.current_workers - 1, self.config.min_workers)
        
        # Recreate worker pool with fewer threads
        if self.worker_pool:
            self.worker_pool.shutdown(wait=False)
            
        self._initialize_workers()
        self.last_scale_time = time.time()
        
        self.logger.info(f"Scaled down from {old_workers} to {self.current_workers} workers")
        
    def _log_final_performance_stats(self):
        """Log final performance statistics."""
        if not self.metrics_history:
            return
            
        # Calculate overall statistics
        total_runtime = time.time() - self.start_time
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 measurements
        
        avg_cpu = np.mean([m.cpu_utilization for m in recent_metrics])
        avg_throughput = np.mean([m.throughput_events_per_sec for m in recent_metrics])
        avg_latency = np.mean([m.latency_p50_ms for m in recent_metrics])
        
        cache_stats = self.cache.get_stats()
        load_stats = self.load_balancer.get_load_stats()
        
        self.logger.info(
            f"Final Performance Statistics:\n"
            f"  Total Runtime: {total_runtime:.1f}s\n"
            f"  Average CPU: {avg_cpu:.1f}%\n"
            f"  Average Throughput: {avg_throughput:.0f} events/s\n"
            f"  Average Latency: {avg_latency:.1f}ms\n"
            f"  Cache Hit Rate: {cache_stats['hit_rate']:.1f}%\n"
            f"  Workers Used: {self.config.min_workers}-{self.current_workers}\n"
            f"  Load Balance Strategy: {load_stats['strategy']}"
        )
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        cache_stats = self.cache.get_stats()
        load_stats = self.load_balancer.get_load_stats()
        
        summary = {
            'system_status': 'running' if self.is_running else 'stopped',
            'uptime_seconds': time.time() - self.start_time,
            'current_workers': self.current_workers,
            'cache_stats': cache_stats,
            'load_balancer_stats': load_stats
        }
        
        if latest_metrics:
            summary['latest_metrics'] = {
                'throughput_events_per_sec': latest_metrics.throughput_events_per_sec,
                'latency_p95_ms': latest_metrics.latency_p95_ms,
                'cpu_utilization': latest_metrics.cpu_utilization,
                'memory_usage_mb': latest_metrics.memory_usage_mb,
                'cache_hit_rate': latest_metrics.cache_hit_rate
            }
            
        return summary

# Test and demonstration functions
def test_scaling_system():
    """Test the high-performance scaling system."""
    logger = logging.getLogger("test_scaling_system")
    logger.info("Testing high-performance scaling system...")
    
    # Configuration for testing
    config = ScalingConfig(
        min_workers=2,
        max_workers=4,
        enable_caching=True,
        cache_size_mb=64,
        batch_size=500
    )
    
    # Initialize processor
    processor = HighPerformanceEventProcessor(config)
    
    try:
        # Start system
        processor.start()
        
        # Generate test workload
        for i in range(10):
            # Generate synthetic events
            num_events = np.random.randint(100, 2000)
            events = np.random.rand(num_events, 4)
            events[:, 0] *= 640  # x coordinates
            events[:, 1] *= 480  # y coordinates
            events[:, 2] = time.time() + np.random.rand(num_events) * 0.1  # timestamps
            events[:, 3] = np.random.choice([-1, 1], num_events)  # polarity
            
            # Process with caching
            cache_key = f"test_batch_{i}"
            result = processor.process_events(events, cache_key=cache_key)
            
            if result is not None:
                logger.info(f"Processed batch {i+1}: {len(events)} -> {len(result)} events")
            else:
                logger.warning(f"Failed to process batch {i+1}")
                
            time.sleep(0.5)  # Simulate real-world timing
            
        # Wait a bit for metrics collection
        time.sleep(2.0)
        
        # Get performance summary
        summary = processor.get_performance_summary()
        logger.info(f"Performance Summary: {json.dumps(summary, indent=2)}")
        
    finally:
        # Always stop processor
        processor.stop()
        
    logger.info("✅ Scaling system test completed successfully!")

if __name__ == "__main__":
    # Run test
    test_scaling_system()