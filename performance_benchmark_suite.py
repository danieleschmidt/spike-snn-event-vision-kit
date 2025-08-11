"""
Comprehensive Performance Benchmark Suite for Neuromorphic Vision System.

This suite provides extensive benchmarking and validation for the high-performance
neuromorphic vision processing system, testing millions of events per second
with sub-millisecond latency validation and comprehensive scaling tests.
"""

import time
import asyncio
import threading
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import statistics
from contextlib import contextmanager
import psutil
import gc

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.spike_snn_event.gpu_distributed_processor import (
    get_distributed_gpu_processor,
    ProcessingTask,
    NeuromorphicEvent as GPUEvent
)
from src.spike_snn_event.async_event_processor import (
    get_async_event_pipeline,
    NeuromorphicEvent,
    EventPriority,
    process_events_stream
)
from src.spike_snn_event.intelligent_cache_system import (
    get_intelligent_cache,
    cached_operation
)
from src.spike_snn_event.intelligent_autoscaler import get_intelligent_autoscaler
from src.spike_snn_event.advanced_telemetry import (
    get_telemetry_system,
    profile_performance
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    # Test duration and scale
    duration_seconds: float = 60.0
    warmup_seconds: float = 10.0
    cooldown_seconds: float = 5.0
    
    # Event generation
    target_event_rate_eps: int = 1_000_000  # 1M events per second
    event_resolution: Tuple[int, int] = (640, 480)
    event_polarity_ratio: float = 0.5
    
    # Processing parameters  
    batch_sizes: List[int] = field(default_factory=lambda: [1, 10, 50, 100, 500])
    worker_counts: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    
    # Performance targets
    max_latency_ms: float = 1.0  # Sub-millisecond target
    min_throughput_eps: int = 100_000  # 100K events per second minimum
    min_cache_hit_rate: float = 0.8
    max_cpu_utilization: float = 90.0
    max_memory_utilization: float = 85.0
    
    # Test types
    test_gpu_processing: bool = True
    test_async_pipeline: bool = True
    test_cache_system: bool = True
    test_auto_scaling: bool = True
    test_telemetry: bool = True
    test_stress_conditions: bool = True


@dataclass
class BenchmarkResult:
    """Results from a benchmark test."""
    test_name: str
    config: BenchmarkConfig
    
    # Performance metrics
    average_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    throughput_eps: float = 0.0
    peak_throughput_eps: float = 0.0
    
    # Resource utilization
    average_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    average_memory_percent: float = 0.0
    peak_memory_percent: float = 0.0
    average_gpu_utilization: float = 0.0
    
    # Cache performance
    cache_hit_rate: float = 0.0
    cache_miss_count: int = 0
    
    # Scaling metrics
    scale_up_time_ms: float = 0.0
    scale_down_time_ms: float = 0.0
    scaling_accuracy: float = 0.0
    
    # Error rates
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    
    # Success flags
    meets_latency_target: bool = False
    meets_throughput_target: bool = False
    meets_resource_targets: bool = False
    overall_success: bool = False
    
    # Additional data
    raw_measurements: Dict[str, List[float]] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EventGenerator:
    """High-performance event generator for benchmarks."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.width, self.height = config.event_resolution
        self.logger = logging.getLogger(__name__)
        
    def generate_synthetic_events(self, count: int, start_time: float = None) -> List[NeuromorphicEvent]:
        """Generate synthetic neuromorphic events."""
        if start_time is None:
            start_time = time.time()
            
        events = []
        time_step = 1.0 / self.config.target_event_rate_eps
        
        for i in range(count):
            # Generate realistic spatial distribution (slightly clustered)
            if i % 100 == 0:  # New cluster every 100 events
                cluster_x = np.random.randint(50, self.width - 50)
                cluster_y = np.random.randint(50, self.height - 50)
            else:
                # Add some randomness around cluster
                cluster_x = getattr(self, 'cluster_x', self.width // 2)
                cluster_y = getattr(self, 'cluster_y', self.height // 2)
                
            self.cluster_x = cluster_x
            self.cluster_y = cluster_y
            
            # Generate event around cluster with some spread
            x = int(np.clip(np.random.normal(cluster_x, 20), 0, self.width - 1))
            y = int(np.clip(np.random.normal(cluster_y, 20), 0, self.height - 1))
            
            # Generate timestamp
            timestamp = start_time + (i * time_step)
            
            # Generate polarity
            polarity = 1 if np.random.random() < self.config.event_polarity_ratio else -1
            
            # Create event
            event = NeuromorphicEvent(
                x=x,
                y=y,
                timestamp=timestamp,
                polarity=polarity,
                event_id=i,
                priority=EventPriority.MEDIUM
            )
            
            events.append(event)
            
        return events
        
    async def generate_event_stream(self, duration_seconds: float):
        """Generate continuous event stream for async testing."""
        start_time = time.time()
        event_id = 0
        target_interval = 1.0 / self.config.target_event_rate_eps
        
        while time.time() - start_time < duration_seconds:
            # Generate batch of events for efficiency
            batch_size = min(1000, int(self.config.target_event_rate_eps * 0.01))  # 10ms worth
            
            batch_start_time = time.time()
            for i in range(batch_size):
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)
                timestamp = batch_start_time + (i * target_interval)
                polarity = 1 if np.random.random() < 0.5 else -1
                
                event = NeuromorphicEvent(
                    x=x, y=y, timestamp=timestamp, polarity=polarity,
                    event_id=event_id, priority=EventPriority.MEDIUM
                )
                
                yield event
                event_id += 1
                
            # Control rate
            batch_duration = time.time() - batch_start_time
            expected_duration = batch_size * target_interval
            if batch_duration < expected_duration:
                await asyncio.sleep(expected_duration - batch_duration)


class PerformanceMonitor:
    """Real-time performance monitoring during benchmarks."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.is_monitoring = False
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            
    def _monitoring_loop(self):
        """Performance monitoring loop."""
        while self.is_monitoring:
            try:
                timestamp = time.time()
                
                # System metrics
                self.metrics['cpu_percent'].append(psutil.cpu_percent(interval=None))
                memory = psutil.virtual_memory()
                self.metrics['memory_percent'].append(memory.percent)
                
                # GPU metrics if available
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                    self.metrics['gpu_memory_gb'].append(gpu_memory_allocated)
                    
                self.metrics['timestamp'].append(timestamp)
                
                time.sleep(0.1)  # 100ms intervals
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        stats = {}
        
        for metric_name, values in self.metrics.items():
            if metric_name == 'timestamp':
                continue
                
            if values:
                stats[metric_name] = {
                    'average': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'p50': statistics.median(values),
                    'p95': np.percentile(values, 95) if len(values) >= 20 else max(values),
                    'sample_count': len(values)
                }
                
        return stats


class GPUProcessingBenchmark:
    """Benchmark for GPU distributed processing system."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.event_generator = EventGenerator(config)
        self.logger = logging.getLogger(__name__)
        
    @profile_performance("gpu_processing_benchmark")
    def run_benchmark(self) -> BenchmarkResult:
        """Run GPU processing benchmark."""
        self.logger.info("Starting GPU processing benchmark")
        
        result = BenchmarkResult(
            test_name="GPU_Distributed_Processing",
            config=self.config
        )
        
        # Get GPU processor
        gpu_processor = get_distributed_gpu_processor()
        gpu_processor.start_processing()
        
        try:
            # Test different batch sizes and worker counts
            best_throughput = 0.0
            latency_measurements = []
            
            for batch_size in self.config.batch_sizes:
                for worker_count in [min(w, mp.cpu_count()) for w in self.config.worker_counts]:
                    
                    self.logger.info(f"Testing batch_size={batch_size}, workers={worker_count}")
                    
                    # Generate test events
                    test_events = self.event_generator.generate_synthetic_events(
                        count=batch_size * 100  # 100 batches
                    )
                    
                    # Create processing tasks
                    tasks = []
                    for i in range(0, len(test_events), batch_size):
                        batch = test_events[i:i+batch_size]
                        task_data = np.random.random((batch_size, 784))  # Dummy input data
                        
                        task = ProcessingTask(
                            task_id=f"batch_{i//batch_size}",
                            data=task_data,
                            model_name="benchmark_model",
                            batch_size=batch_size,
                            priority=1
                        )
                        tasks.append(task)
                        
                    # Warmup
                    warmup_tasks = tasks[:10]
                    warmup_results = gpu_processor.optimize_batch_processing(warmup_tasks)
                    
                    # Actual benchmark
                    start_time = time.time()
                    results = gpu_processor.optimize_batch_processing(tasks)
                    end_time = time.time()
                    
                    # Calculate metrics
                    total_time = end_time - start_time
                    total_events = len(test_events)
                    throughput = total_events / total_time
                    
                    if throughput > best_throughput:
                        best_throughput = throughput
                        
                    # Collect latency measurements
                    for res in results:
                        if res and not res.error:
                            latency_measurements.append(res.processing_time / 1_000_000)  # Convert to ms
                            
            # Calculate final metrics
            result.throughput_eps = best_throughput
            result.peak_throughput_eps = best_throughput
            
            if latency_measurements:
                result.average_latency_ms = statistics.mean(latency_measurements)
                result.p50_latency_ms = statistics.median(latency_measurements)
                result.p95_latency_ms = np.percentile(latency_measurements, 95)
                result.p99_latency_ms = np.percentile(latency_measurements, 99)
                result.max_latency_ms = max(latency_measurements)
                
            # Check targets
            result.meets_latency_target = result.p95_latency_ms <= self.config.max_latency_ms
            result.meets_throughput_target = result.throughput_eps >= self.config.min_throughput_eps
            
            # Get GPU stats
            gpu_stats = gpu_processor.get_processing_stats()
            result.average_gpu_utilization = gpu_stats['resource_stats'].get('average_utilization', 0.0)
            
            result.raw_measurements['latency_ms'] = latency_measurements
            
        except Exception as e:
            self.logger.error(f"GPU benchmark failed: {e}")
            result.error_rate = 1.0
            
        finally:
            gpu_processor.stop_processing()
            
        result.overall_success = (
            result.meets_latency_target and 
            result.meets_throughput_target and 
            result.error_rate < 0.01
        )
        
        self.logger.info(f"GPU benchmark completed: {result.overall_success}")
        return result


class AsyncPipelineBenchmark:
    """Benchmark for async event processing pipeline."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.event_generator = EventGenerator(config)
        self.logger = logging.getLogger(__name__)
        
    @profile_performance("async_pipeline_benchmark")
    async def run_benchmark(self) -> BenchmarkResult:
        """Run async pipeline benchmark."""
        self.logger.info("Starting async pipeline benchmark")
        
        result = BenchmarkResult(
            test_name="Async_Event_Pipeline",
            config=self.config
        )
        
        # Get pipeline
        pipeline = get_async_event_pipeline()
        await pipeline.start_pipeline()
        
        try:
            # Performance monitoring
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            # Warmup
            self.logger.info("Warming up pipeline...")
            warmup_events = self.event_generator.generate_synthetic_events(10000)
            submitted = pipeline.submit_events_batch(warmup_events)
            await asyncio.sleep(self.config.warmup_seconds)
            
            # Clear results queue
            while pipeline.get_result(timeout=0.001) is not None:
                pass
                
            # Main benchmark test
            self.logger.info("Starting main benchmark test...")
            
            # Generate event stream
            event_stream = self.event_generator.generate_event_stream(self.config.duration_seconds)
            
            events_submitted = 0
            results_collected = 0
            latency_measurements = []
            start_time = time.time()
            
            # Process event stream
            async for event in event_stream:
                if pipeline.submit_event(event):
                    events_submitted += 1
                    
                # Collect results
                while True:
                    result_obj = pipeline.get_result(timeout=0.001)
                    if result_obj is None:
                        break
                        
                    results_collected += 1
                    if not result_obj.error:
                        # Calculate latency (time from event creation to result)
                        processing_latency = (result_obj.completed_at - event.timestamp) * 1000  # ms
                        latency_measurements.append(processing_latency)
                        
            # Wait for remaining results
            await asyncio.sleep(2.0)
            while True:
                result_obj = pipeline.get_result(timeout=0.1)
                if result_obj is None:
                    break
                results_collected += 1
                
            end_time = time.time()
            total_time = end_time - start_time
            
            # Stop monitoring
            monitor.stop_monitoring()
            monitor_stats = monitor.get_statistics()
            
            # Calculate metrics
            result.throughput_eps = events_submitted / total_time
            
            if latency_measurements:
                result.average_latency_ms = statistics.mean(latency_measurements)
                result.p50_latency_ms = statistics.median(latency_measurements)
                result.p95_latency_ms = np.percentile(latency_measurements, 95)
                result.p99_latency_ms = np.percentile(latency_measurements, 99)
                result.max_latency_ms = max(latency_measurements)
                
            # Resource utilization
            if 'cpu_percent' in monitor_stats:
                result.average_cpu_percent = monitor_stats['cpu_percent']['average']
                result.peak_cpu_percent = monitor_stats['cpu_percent']['max']
                
            if 'memory_percent' in monitor_stats:
                result.average_memory_percent = monitor_stats['memory_percent']['average']
                result.peak_memory_percent = monitor_stats['memory_percent']['max']
                
            # Pipeline stats
            pipeline_stats = pipeline.get_comprehensive_stats()
            result.peak_throughput_eps = pipeline_stats['pipeline']['peak_throughput_eps']
            
            # Error rate
            total_events = events_submitted
            failed_events = total_events - results_collected
            result.error_rate = failed_events / max(total_events, 1)
            
            # Check targets
            result.meets_latency_target = result.p95_latency_ms <= self.config.max_latency_ms
            result.meets_throughput_target = result.throughput_eps >= self.config.min_throughput_eps
            result.meets_resource_targets = (
                result.peak_cpu_percent <= self.config.max_cpu_utilization and
                result.peak_memory_percent <= self.config.max_memory_utilization
            )
            
            result.raw_measurements['latency_ms'] = latency_measurements
            
        except Exception as e:
            self.logger.error(f"Async pipeline benchmark failed: {e}")
            result.error_rate = 1.0
            
        finally:
            await pipeline.stop_pipeline()
            
        result.overall_success = (
            result.meets_latency_target and
            result.meets_throughput_target and
            result.meets_resource_targets and
            result.error_rate < 0.01
        )
        
        self.logger.info(f"Async pipeline benchmark completed: {result.overall_success}")
        return result


class CacheSystemBenchmark:
    """Benchmark for intelligent cache system."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    @profile_performance("cache_system_benchmark")
    def run_benchmark(self) -> BenchmarkResult:
        """Run cache system benchmark."""
        self.logger.info("Starting cache system benchmark")
        
        result = BenchmarkResult(
            test_name="Intelligent_Cache_System", 
            config=self.config
        )
        
        cache = get_intelligent_cache()
        
        try:
            # Test data
            test_data_size = 10000
            test_data = {}
            
            # Generate test data with different sizes and access patterns
            for i in range(test_data_size):
                if i < test_data_size * 0.1:  # 10% large objects
                    data = np.random.random((1000, 1000))  # Large arrays
                elif i < test_data_size * 0.3:  # 20% medium objects  
                    data = np.random.random((100, 100))
                else:  # 70% small objects
                    data = np.random.random(100)
                    
                test_data[f"key_{i}"] = data
                
            # Warmup - populate cache
            self.logger.info("Populating cache...")
            for key, data in test_data.items():
                cache.put(key, data)
                
            # Test access patterns
            access_times = []
            hit_count = 0
            miss_count = 0
            
            # Pattern 1: Sequential access (tests LRU behavior)
            self.logger.info("Testing sequential access...")
            start_time = time.time()
            
            for i in range(test_data_size):
                key = f"key_{i}"
                access_start = time.time_ns()
                value, hit = cache.get(key)
                access_time = (time.time_ns() - access_start) / 1_000_000  # ms
                
                access_times.append(access_time)
                if hit:
                    hit_count += 1
                else:
                    miss_count += 1
                    
            sequential_time = time.time() - start_time
            
            # Pattern 2: Random access (tests cache efficiency)  
            self.logger.info("Testing random access...")
            random_keys = np.random.choice(list(test_data.keys()), size=5000, replace=True)
            
            start_time = time.time()
            for key in random_keys:
                access_start = time.time_ns()
                value, hit = cache.get(key)
                access_time = (time.time_ns() - access_start) / 1_000_000  # ms
                
                access_times.append(access_time)
                if hit:
                    hit_count += 1
                else:
                    miss_count += 1
                    
            random_time = time.time() - start_time
            
            # Pattern 3: Hotspot access (80/20 rule)
            self.logger.info("Testing hotspot access...")
            hotspot_keys = [f"key_{i}" for i in range(int(test_data_size * 0.2))]  # Top 20%
            hotspot_accesses = np.random.choice(hotspot_keys, size=10000, replace=True)
            
            start_time = time.time()
            for key in hotspot_accesses:
                access_start = time.time_ns()
                value, hit = cache.get(key)
                access_time = (time.time_ns() - access_start) / 1_000_000  # ms
                
                access_times.append(access_time)
                if hit:
                    hit_count += 1
                else:
                    miss_count += 1
                    
            hotspot_time = time.time() - start_time
            
            # Calculate metrics
            total_accesses = hit_count + miss_count
            result.cache_hit_rate = hit_count / total_accesses if total_accesses > 0 else 0.0
            result.cache_miss_count = miss_count
            
            if access_times:
                result.average_latency_ms = statistics.mean(access_times)
                result.p50_latency_ms = statistics.median(access_times)
                result.p95_latency_ms = np.percentile(access_times, 95)
                result.p99_latency_ms = np.percentile(access_times, 99)
                result.max_latency_ms = max(access_times)
                
            # Throughput calculation
            total_time = sequential_time + random_time + hotspot_time
            result.throughput_eps = total_accesses / total_time
            
            # Cache stats
            cache_stats = cache.get_comprehensive_stats()
            
            result.raw_measurements['access_times_ms'] = access_times
            result.raw_measurements['cache_stats'] = cache_stats
            
            # Check targets
            result.meets_latency_target = result.p95_latency_ms <= 0.1  # 100μs for cache
            result.meets_throughput_target = result.cache_hit_rate >= self.config.min_cache_hit_rate
            
        except Exception as e:
            self.logger.error(f"Cache benchmark failed: {e}")
            result.error_rate = 1.0
            
        result.overall_success = (
            result.meets_latency_target and
            result.meets_throughput_target and
            result.error_rate < 0.01
        )
        
        self.logger.info(f"Cache benchmark completed: {result.overall_success}")
        return result


class AutoScalingBenchmark:
    """Benchmark for intelligent auto-scaling system."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.event_generator = EventGenerator(config)
        self.logger = logging.getLogger(__name__)
        
    @profile_performance("auto_scaling_benchmark")
    async def run_benchmark(self) -> BenchmarkResult:
        """Run auto-scaling benchmark."""
        self.logger.info("Starting auto-scaling benchmark")
        
        result = BenchmarkResult(
            test_name="Intelligent_Auto_Scaling",
            config=self.config
        )
        
        autoscaler = get_intelligent_autoscaler()
        
        try:
            # Start autoscaler
            autoscaler.start_intelligent_scaling()
            
            # Performance monitoring
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            # Test different load patterns
            scaling_times = []
            accuracy_scores = []
            
            # Phase 1: Low load
            self.logger.info("Phase 1: Low load")
            low_load_events = self.event_generator.generate_synthetic_events(1000)
            initial_workers = autoscaler.current_workers
            
            await asyncio.sleep(30)  # Let system stabilize
            
            # Phase 2: Sudden load increase (test scale-up)
            self.logger.info("Phase 2: Load spike (testing scale-up)")
            scale_up_start = time.time()
            
            # Generate high load
            high_load_events = self.event_generator.generate_synthetic_events(50000)
            
            # Wait for scale-up decision
            scale_up_detected = False
            scale_up_time = 0.0
            
            for _ in range(120):  # Wait up to 2 minutes
                await asyncio.sleep(1)
                current_workers = autoscaler.current_workers
                
                if current_workers > initial_workers and not scale_up_detected:
                    scale_up_time = time.time() - scale_up_start
                    scale_up_detected = True
                    self.logger.info(f"Scale-up detected in {scale_up_time:.1f}s: {initial_workers} -> {current_workers}")
                    break
                    
            scaling_times.append(scale_up_time)
            
            # Phase 3: Load decrease (test scale-down)
            self.logger.info("Phase 3: Load decrease (testing scale-down)")
            scale_down_start = time.time()
            
            # Reduce load dramatically
            await asyncio.sleep(60)  # Wait for scale-down
            
            scale_down_detected = False
            scale_down_time = 0.0
            peak_workers = autoscaler.current_workers
            
            for _ in range(300):  # Wait up to 5 minutes (scale-down is usually slower)
                await asyncio.sleep(1)
                current_workers = autoscaler.current_workers
                
                if current_workers < peak_workers and not scale_down_detected:
                    scale_down_time = time.time() - scale_down_start
                    scale_down_detected = True
                    self.logger.info(f"Scale-down detected in {scale_down_time:.1f}s: {peak_workers} -> {current_workers}")
                    break
                    
            scaling_times.append(scale_down_time)
            
            # Stop monitoring
            monitor.stop_monitoring()
            monitor_stats = monitor.get_statistics()
            
            # Get autoscaler stats
            scaler_stats = autoscaler.get_intelligent_scaling_stats()
            
            # Calculate metrics
            if scaling_times:
                result.scale_up_time_ms = scaling_times[0] * 1000
                if len(scaling_times) > 1:
                    result.scale_down_time_ms = scaling_times[1] * 1000
                    
            result.scaling_accuracy = scaler_stats['intelligent_scaling'].get('average_decision_accuracy', 0.0)
            
            # Resource utilization
            if 'cpu_percent' in monitor_stats:
                result.average_cpu_percent = monitor_stats['cpu_percent']['average']
                result.peak_cpu_percent = monitor_stats['cpu_percent']['max']
                
            if 'memory_percent' in monitor_stats:
                result.average_memory_percent = monitor_stats['memory_percent']['average']
                result.peak_memory_percent = monitor_stats['memory_percent']['max']
                
            # Check targets
            result.meets_latency_target = result.scale_up_time_ms <= 30000  # 30s max scale-up time
            result.meets_throughput_target = result.scaling_accuracy >= 0.7  # 70% accuracy
            result.meets_resource_targets = (
                result.peak_cpu_percent <= self.config.max_cpu_utilization and
                result.peak_memory_percent <= self.config.max_memory_utilization
            )
            
            result.raw_measurements['scaling_times_s'] = scaling_times
            result.raw_measurements['scaler_stats'] = scaler_stats
            
        except Exception as e:
            self.logger.error(f"Auto-scaling benchmark failed: {e}")
            result.error_rate = 1.0
            
        finally:
            autoscaler.stop_intelligent_scaling()
            
        result.overall_success = (
            result.meets_latency_target and
            result.meets_throughput_target and
            result.meets_resource_targets and
            result.error_rate < 0.01
        )
        
        self.logger.info(f"Auto-scaling benchmark completed: {result.overall_success}")
        return result


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    async def run_all_benchmarks(self) -> Dict[str, BenchmarkResult]:
        """Run all benchmark tests."""
        self.logger.info("Starting comprehensive performance benchmark suite")
        self.logger.info(f"Target: {self.config.target_event_rate_eps:,} events/sec, "
                        f"<{self.config.max_latency_ms}ms latency")
        
        results = {}
        
        # Start telemetry system
        telemetry = get_telemetry_system()
        telemetry.start_telemetry()
        
        try:
            # 1. GPU Processing Benchmark
            if self.config.test_gpu_processing:
                self.logger.info("\n" + "="*60)
                self.logger.info("RUNNING GPU PROCESSING BENCHMARK")
                self.logger.info("="*60)
                
                gpu_benchmark = GPUProcessingBenchmark(self.config)
                results['gpu_processing'] = gpu_benchmark.run_benchmark()
                
                # Cooldown
                await asyncio.sleep(self.config.cooldown_seconds)
                gc.collect()
                
            # 2. Async Pipeline Benchmark  
            if self.config.test_async_pipeline:
                self.logger.info("\n" + "="*60)
                self.logger.info("RUNNING ASYNC PIPELINE BENCHMARK")
                self.logger.info("="*60)
                
                pipeline_benchmark = AsyncPipelineBenchmark(self.config)
                results['async_pipeline'] = await pipeline_benchmark.run_benchmark()
                
                # Cooldown
                await asyncio.sleep(self.config.cooldown_seconds)
                gc.collect()
                
            # 3. Cache System Benchmark
            if self.config.test_cache_system:
                self.logger.info("\n" + "="*60)
                self.logger.info("RUNNING CACHE SYSTEM BENCHMARK")
                self.logger.info("="*60)
                
                cache_benchmark = CacheSystemBenchmark(self.config)
                results['cache_system'] = cache_benchmark.run_benchmark()
                
                # Cooldown
                await asyncio.sleep(self.config.cooldown_seconds)
                gc.collect()
                
            # 4. Auto-scaling Benchmark
            if self.config.test_auto_scaling:
                self.logger.info("\n" + "="*60)
                self.logger.info("RUNNING AUTO-SCALING BENCHMARK")
                self.logger.info("="*60)
                
                scaling_benchmark = AutoScalingBenchmark(self.config)
                results['auto_scaling'] = await scaling_benchmark.run_benchmark()
                
                # Cooldown
                await asyncio.sleep(self.config.cooldown_seconds)
                gc.collect()
                
            # Generate comprehensive report
            self._generate_benchmark_report(results)
            
        except Exception as e:
            self.logger.error(f"Benchmark suite failed: {e}")
            
        finally:
            telemetry.stop_telemetry()
            
        return results
        
    def _generate_benchmark_report(self, results: Dict[str, BenchmarkResult]):
        """Generate comprehensive benchmark report."""
        self.logger.info("\n" + "="*80)
        self.logger.info("COMPREHENSIVE BENCHMARK RESULTS")
        self.logger.info("="*80)
        
        overall_success = True
        total_tests = len(results)
        passed_tests = 0
        
        # Individual test results
        for test_name, result in results.items():
            self.logger.info(f"\n{test_name.upper().replace('_', ' ')} RESULTS:")
            self.logger.info("-" * 50)
            
            # Performance metrics
            if result.throughput_eps > 0:
                self.logger.info(f"Throughput: {result.throughput_eps:,.0f} events/sec")
                
            if result.average_latency_ms > 0:
                self.logger.info(f"Average Latency: {result.average_latency_ms:.3f} ms")
                self.logger.info(f"P95 Latency: {result.p95_latency_ms:.3f} ms")
                self.logger.info(f"P99 Latency: {result.p99_latency_ms:.3f} ms")
                self.logger.info(f"Max Latency: {result.max_latency_ms:.3f} ms")
                
            if result.cache_hit_rate > 0:
                self.logger.info(f"Cache Hit Rate: {result.cache_hit_rate:.1%}")
                
            if result.scaling_accuracy > 0:
                self.logger.info(f"Scaling Accuracy: {result.scaling_accuracy:.1%}")
                self.logger.info(f"Scale-up Time: {result.scale_up_time_ms:.0f} ms")
                self.logger.info(f"Scale-down Time: {result.scale_down_time_ms:.0f} ms")
                
            # Resource utilization
            if result.peak_cpu_percent > 0:
                self.logger.info(f"Peak CPU: {result.peak_cpu_percent:.1f}%")
                self.logger.info(f"Peak Memory: {result.peak_memory_percent:.1f}%")
                
            # Success criteria
            status = "PASS" if result.overall_success else "FAIL"
            self.logger.info(f"Overall Status: {status}")
            
            if result.overall_success:
                passed_tests += 1
            else:
                overall_success = False
                
            # Detailed criteria
            self.logger.info("Target Achievement:")
            self.logger.info(f"  ✓ Latency Target: {'PASS' if result.meets_latency_target else 'FAIL'}")
            self.logger.info(f"  ✓ Throughput Target: {'PASS' if result.meets_throughput_target else 'FAIL'}")  
            self.logger.info(f"  ✓ Resource Targets: {'PASS' if result.meets_resource_targets else 'FAIL'}")
            self.logger.info(f"  ✓ Error Rate: {result.error_rate:.1%} ({'PASS' if result.error_rate < 0.01 else 'FAIL'})")
            
        # Overall summary
        self.logger.info("\n" + "="*80)
        self.logger.info("OVERALL BENCHMARK SUMMARY")
        self.logger.info("="*80)
        
        self.logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        self.logger.info(f"Overall Success: {'YES' if overall_success else 'NO'}")
        
        if overall_success:
            self.logger.info("\n🎉 ALL BENCHMARKS PASSED!")
            self.logger.info("System meets performance targets for high-throughput")
            self.logger.info("neuromorphic vision processing with sub-millisecond latency.")
        else:
            self.logger.info("\n⚠️  SOME BENCHMARKS FAILED")
            self.logger.info("Review individual test results for optimization opportunities.")
            
        # Save detailed report
        report_path = Path("benchmark_report.json")
        detailed_report = {
            'config': {
                'target_event_rate_eps': self.config.target_event_rate_eps,
                'max_latency_ms': self.config.max_latency_ms,
                'min_throughput_eps': self.config.min_throughput_eps,
                'duration_seconds': self.config.duration_seconds
            },
            'results': {
                name: {
                    'test_name': result.test_name,
                    'overall_success': result.overall_success,
                    'throughput_eps': result.throughput_eps,
                    'average_latency_ms': result.average_latency_ms,
                    'p95_latency_ms': result.p95_latency_ms,
                    'p99_latency_ms': result.p99_latency_ms,
                    'cache_hit_rate': result.cache_hit_rate,
                    'scaling_accuracy': result.scaling_accuracy,
                    'peak_cpu_percent': result.peak_cpu_percent,
                    'peak_memory_percent': result.peak_memory_percent,
                    'error_rate': result.error_rate,
                    'meets_latency_target': result.meets_latency_target,
                    'meets_throughput_target': result.meets_throughput_target,
                    'meets_resource_targets': result.meets_resource_targets
                }
                for name, result in results.items()
            },
            'summary': {
                'overall_success': overall_success,
                'tests_passed': passed_tests,
                'total_tests': total_tests,
                'timestamp': time.time()
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
            
        self.logger.info(f"\nDetailed report saved to: {report_path}")


async def main():
    """Main benchmark execution."""
    # High-performance configuration
    config = BenchmarkConfig(
        duration_seconds=60.0,
        target_event_rate_eps=1_000_000,  # 1M events/sec target
        max_latency_ms=1.0,               # Sub-millisecond target
        min_throughput_eps=500_000,       # 500K events/sec minimum
        min_cache_hit_rate=0.85,          # 85% cache hit rate
        
        # Enable all tests
        test_gpu_processing=True,
        test_async_pipeline=True,
        test_cache_system=True,
        test_auto_scaling=True,
        test_stress_conditions=True
    )
    
    # Run comprehensive benchmark suite
    suite = PerformanceBenchmarkSuite(config)
    results = await suite.run_all_benchmarks()
    
    return results


if __name__ == "__main__":
    # Run benchmarks
    import asyncio
    results = asyncio.run(main())