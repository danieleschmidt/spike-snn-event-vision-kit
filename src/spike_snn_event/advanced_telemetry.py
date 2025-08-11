"""
Advanced Monitoring and Telemetry System with Sub-Millisecond Accuracy.

Provides cutting-edge real-time performance metrics collection, bottleneck detection,
flame graph profiling, and advanced telemetry for neuromorphic vision processing
with sub-millisecond accuracy tracking and automatic resolution capabilities.
"""

import time
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict, Counter
from abc import ABC, abstractmethod
import numpy as np
import logging
import json
import psutil
from pathlib import Path
import weakref
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import traceback
from contextlib import contextmanager
import struct
import mmap

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from .monitoring import get_metrics_collector
from .gpu_distributed_processor import get_distributed_gpu_processor
from .async_event_processor import get_async_event_pipeline
from .intelligent_cache_system import get_intelligent_cache
from .intelligent_autoscaler import get_intelligent_autoscaler


@dataclass
class PerformanceMetric:
    """High-precision performance metric."""
    name: str
    value: float
    unit: str
    timestamp_ns: int
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    
    @property
    def timestamp_ms(self) -> float:
        return self.timestamp_ns / 1_000_000
        
    @property
    def timestamp_us(self) -> float:
        return self.timestamp_ns / 1_000


@dataclass
class BottleneckInfo:
    """Information about detected performance bottleneck."""
    component: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    root_cause: str
    impact_score: float  # 0.0 - 1.0
    suggested_resolution: List[str]
    detection_confidence: float  # 0.0 - 1.0
    first_detected: float
    last_seen: float
    occurrence_count: int = 1


@dataclass
class FlameGraphNode:
    """Node in flame graph for performance profiling."""
    function_name: str
    source_file: str
    line_number: int
    total_time_ns: int
    self_time_ns: int
    call_count: int
    children: Dict[str, 'FlameGraphNode'] = field(default_factory=dict)
    parent: Optional['FlameGraphNode'] = None
    
    @property
    def total_time_ms(self) -> float:
        return self.total_time_ns / 1_000_000
        
    @property
    def self_time_ms(self) -> float:
        return self.self_time_ns / 1_000_000


class HighPrecisionTimer:
    """High-precision timer for sub-millisecond measurements."""
    
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.is_running = False
        
    def start(self):
        """Start timing."""
        self.start_time = time.time_ns()
        self.is_running = True
        
    def stop(self) -> int:
        """Stop timing and return elapsed nanoseconds."""
        if not self.is_running:
            return 0
        self.end_time = time.time_ns()
        self.is_running = False
        return self.end_time - self.start_time
        
    def elapsed_ns(self) -> int:
        """Get elapsed nanoseconds."""
        if self.is_running:
            return time.time_ns() - self.start_time
        return self.end_time - self.start_time
        
    def elapsed_us(self) -> float:
        """Get elapsed microseconds."""
        return self.elapsed_ns() / 1_000
        
    def elapsed_ms(self) -> float:
        """Get elapsed milliseconds."""
        return self.elapsed_ns() / 1_000_000
        
    @contextmanager
    def measure(self):
        """Context manager for timing operations."""
        self.start()
        try:
            yield self
        finally:
            self.stop()


class MetricsBuffer:
    """High-performance circular buffer for metrics storage."""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0
        self.lock = threading.RLock()
        
    def put(self, metric: PerformanceMetric) -> bool:
        """Add metric to buffer."""
        with self.lock:
            if self.size >= self.capacity:
                return False  # Buffer full
                
            self.buffer[self.head] = metric
            self.head = (self.head + 1) % self.capacity
            self.size += 1
            return True
            
    def get_batch(self, max_count: int = 1000) -> List[PerformanceMetric]:
        """Get batch of metrics from buffer."""
        with self.lock:
            if self.size == 0:
                return []
                
            result = []
            count = min(max_count, self.size)
            
            for _ in range(count):
                result.append(self.buffer[self.tail])
                self.buffer[self.tail] = None  # Clear reference
                self.tail = (self.tail + 1) % self.capacity
                self.size -= 1
                
            return result
            
    def peek_recent(self, count: int = 100) -> List[PerformanceMetric]:
        """Peek at recent metrics without removing them."""
        with self.lock:
            if self.size == 0:
                return []
                
            result = []
            peek_count = min(count, self.size)
            
            # Start from most recent
            pos = (self.head - 1) % self.capacity
            for _ in range(peek_count):
                if self.buffer[pos] is not None:
                    result.append(self.buffer[pos])
                pos = (pos - 1) % self.capacity
                
            return result[::-1]  # Reverse to chronological order


class FlameGraphProfiler:
    """Advanced flame graph profiler for performance analysis."""
    
    def __init__(self, sampling_interval_ns: int = 100_000):  # 100μs default
        self.sampling_interval_ns = sampling_interval_ns
        self.is_profiling = False
        self.profile_thread = None
        self.call_stack = []
        self.root_nodes = {}
        self.current_node = None
        self.profile_lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
    def start_profiling(self):
        """Start flame graph profiling."""
        if self.is_profiling:
            return
            
        self.is_profiling = True
        self.profile_thread = threading.Thread(target=self._profiling_loop, daemon=True)
        self.profile_thread.start()
        self.logger.info("Flame graph profiling started")
        
    def stop_profiling(self):
        """Stop flame graph profiling."""
        self.is_profiling = False
        if self.profile_thread:
            self.profile_thread.join(timeout=5.0)
        self.logger.info("Flame graph profiling stopped")
        
    @contextmanager
    def profile_function(self, function_name: str, source_file: str = "", line_number: int = 0):
        """Context manager for profiling function execution."""
        timer = HighPrecisionTimer()
        
        with self.profile_lock:
            # Create or get node
            node_key = f"{function_name}:{source_file}:{line_number}"
            
            if self.current_node is None:
                # Root node
                if node_key not in self.root_nodes:
                    self.root_nodes[node_key] = FlameGraphNode(
                        function_name=function_name,
                        source_file=source_file,
                        line_number=line_number,
                        total_time_ns=0,
                        self_time_ns=0,
                        call_count=0
                    )
                node = self.root_nodes[node_key]
            else:
                # Child node
                if node_key not in self.current_node.children:
                    self.current_node.children[node_key] = FlameGraphNode(
                        function_name=function_name,
                        source_file=source_file,
                        line_number=line_number,
                        total_time_ns=0,
                        self_time_ns=0,
                        call_count=0,
                        parent=self.current_node
                    )
                node = self.current_node.children[node_key]
                
            # Push to call stack
            self.call_stack.append(self.current_node)
            self.current_node = node
            node.call_count += 1
            
        timer.start()
        child_time_ns = 0
        
        try:
            yield timer
        finally:
            elapsed_ns = timer.stop()
            
            with self.profile_lock:
                # Update timing information
                node.total_time_ns += elapsed_ns
                node.self_time_ns += elapsed_ns - child_time_ns
                
                # Pop from call stack
                self.current_node = self.call_stack.pop() if self.call_stack else None
                
                # Update parent's child time
                if self.current_node:
                    child_time_ns += elapsed_ns
                    
    def _profiling_loop(self):
        """Background profiling loop for sampling."""
        while self.is_profiling:
            try:
                # Sample current call stack
                with self.profile_lock:
                    if self.call_stack:
                        # Record stack trace sample
                        pass  # Could implement statistical sampling here
                        
                time.sleep(self.sampling_interval_ns / 1_000_000_000)  # Convert to seconds
                
            except Exception as e:
                self.logger.error(f"Profiling loop error: {e}")
                
    def generate_flame_graph_data(self) -> Dict[str, Any]:
        """Generate flame graph data structure."""
        with self.profile_lock:
            flame_graph_data = {
                'nodes': [],
                'total_samples': sum(node.call_count for node in self.root_nodes.values()),
                'profiling_duration_ms': 0  # Would need to track this
            }
            
            def serialize_node(node: FlameGraphNode, depth: int = 0):
                return {
                    'name': node.function_name,
                    'source_file': node.source_file,
                    'line_number': node.line_number,
                    'total_time_ms': node.total_time_ms,
                    'self_time_ms': node.self_time_ms,
                    'call_count': node.call_count,
                    'depth': depth,
                    'children': [serialize_node(child, depth + 1) 
                               for child in node.children.values()]
                }
                
            for node in self.root_nodes.values():
                flame_graph_data['nodes'].append(serialize_node(node))
                
            return flame_graph_data
            
    def get_hotspots(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get performance hotspots."""
        with self.profile_lock:
            all_nodes = []
            
            def collect_nodes(node: FlameGraphNode):
                all_nodes.append(node)
                for child in node.children.values():
                    collect_nodes(child)
                    
            for root in self.root_nodes.values():
                collect_nodes(root)
                
            # Sort by self time (where actual work happens)
            hotspots = sorted(all_nodes, key=lambda n: n.self_time_ns, reverse=True)[:top_n]
            
            return [
                {
                    'function_name': node.function_name,
                    'source_file': node.source_file,
                    'self_time_ms': node.self_time_ms,
                    'call_count': node.call_count,
                    'avg_time_per_call_us': node.self_time_ns / 1000 / max(node.call_count, 1)
                }
                for node in hotspots
            ]


class BottleneckDetector:
    """Intelligent bottleneck detection system."""
    
    def __init__(self):
        self.detected_bottlenecks = {}  # component -> BottleneckInfo
        self.metrics_history = deque(maxlen=1000)
        self.detection_rules = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize detection rules
        self._initialize_detection_rules()
        
    def _initialize_detection_rules(self):
        """Initialize bottleneck detection rules."""
        self.detection_rules = [
            self._detect_cpu_bottleneck,
            self._detect_memory_bottleneck,
            self._detect_gpu_bottleneck,
            self._detect_io_bottleneck,
            self._detect_queue_bottleneck,
            self._detect_cache_bottleneck,
            self._detect_network_bottleneck
        ]
        
    def analyze_metrics(self, metrics: List[PerformanceMetric]):
        """Analyze metrics for bottlenecks."""
        self.metrics_history.extend(metrics)
        
        # Run detection rules
        for rule in self.detection_rules:
            try:
                bottlenecks = rule(list(self.metrics_history)[-100:])  # Last 100 metrics
                for bottleneck in bottlenecks:
                    self._update_bottleneck(bottleneck)
            except Exception as e:
                self.logger.error(f"Bottleneck detection rule failed: {e}")
                
    def _detect_cpu_bottleneck(self, metrics: List[PerformanceMetric]) -> List[BottleneckInfo]:
        """Detect CPU bottlenecks."""
        cpu_metrics = [m for m in metrics if m.name == 'cpu_utilization']
        if len(cpu_metrics) < 10:
            return []
            
        recent_cpu = [m.value for m in cpu_metrics[-10:]]
        avg_cpu = np.mean(recent_cpu)
        
        bottlenecks = []
        
        if avg_cpu > 90:
            bottlenecks.append(BottleneckInfo(
                component='cpu',
                severity='critical',
                description=f'CPU utilization critically high: {avg_cpu:.1f}%',
                root_cause='Insufficient CPU resources or inefficient algorithms',
                impact_score=min(1.0, (avg_cpu - 80) / 20),
                suggested_resolution=[
                    'Scale up CPU resources',
                    'Optimize compute-intensive algorithms',
                    'Enable CPU-specific optimizations (SIMD, vectorization)',
                    'Distribute load across more workers'
                ],
                detection_confidence=0.9,
                first_detected=time.time(),
                last_seen=time.time()
            ))
        elif avg_cpu > 80:
            bottlenecks.append(BottleneckInfo(
                component='cpu',
                severity='high',
                description=f'CPU utilization high: {avg_cpu:.1f}%',
                root_cause='High computational load',
                impact_score=min(1.0, (avg_cpu - 70) / 30),
                suggested_resolution=[
                    'Monitor for continued high usage',
                    'Consider scaling up if sustained'
                ],
                detection_confidence=0.7,
                first_detected=time.time(),
                last_seen=time.time()
            ))
            
        return bottlenecks
        
    def _detect_memory_bottleneck(self, metrics: List[PerformanceMetric]) -> List[BottleneckInfo]:
        """Detect memory bottlenecks."""
        memory_metrics = [m for m in metrics if m.name == 'memory_utilization']
        if len(memory_metrics) < 5:
            return []
            
        recent_memory = [m.value for m in memory_metrics[-5:]]
        avg_memory = np.mean(recent_memory)
        memory_trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]
        
        bottlenecks = []
        
        if avg_memory > 95:
            bottlenecks.append(BottleneckInfo(
                component='memory',
                severity='critical',
                description=f'Memory utilization critically high: {avg_memory:.1f}%',
                root_cause='Memory leak or insufficient memory allocation',
                impact_score=min(1.0, (avg_memory - 90) / 10),
                suggested_resolution=[
                    'Investigate memory leaks',
                    'Increase available memory',
                    'Optimize memory usage patterns',
                    'Enable memory compression',
                    'Implement aggressive garbage collection'
                ],
                detection_confidence=0.95,
                first_detected=time.time(),
                last_seen=time.time()
            ))
        elif avg_memory > 85 and memory_trend > 0:
            bottlenecks.append(BottleneckInfo(
                component='memory',
                severity='high',
                description=f'Memory utilization high and increasing: {avg_memory:.1f}%',
                root_cause='Growing memory usage pattern',
                impact_score=min(1.0, (avg_memory - 80) / 20),
                suggested_resolution=[
                    'Monitor memory growth pattern',
                    'Review recent changes for memory leaks',
                    'Consider preemptive scaling'
                ],
                detection_confidence=0.8,
                first_detected=time.time(),
                last_seen=time.time()
            ))
            
        return bottlenecks
        
    def _detect_gpu_bottleneck(self, metrics: List[PerformanceMetric]) -> List[BottleneckInfo]:
        """Detect GPU bottlenecks."""
        gpu_metrics = [m for m in metrics if 'gpu' in m.name]
        if len(gpu_metrics) < 5:
            return []
            
        gpu_util_metrics = [m for m in gpu_metrics if 'utilization' in m.name]
        gpu_memory_metrics = [m for m in gpu_metrics if 'memory' in m.name]
        
        bottlenecks = []
        
        if gpu_util_metrics:
            avg_gpu_util = np.mean([m.value for m in gpu_util_metrics[-5:]])
            if avg_gpu_util > 95:
                bottlenecks.append(BottleneckInfo(
                    component='gpu',
                    severity='high',
                    description=f'GPU utilization very high: {avg_gpu_util:.1f}%',
                    root_cause='GPU compute saturation',
                    impact_score=min(1.0, (avg_gpu_util - 90) / 10),
                    suggested_resolution=[
                        'Add more GPU resources',
                        'Optimize CUDA kernels',
                        'Implement model parallelism',
                        'Use mixed precision training'
                    ],
                    detection_confidence=0.85,
                    first_detected=time.time(),
                    last_seen=time.time()
                ))
                
        if gpu_memory_metrics:
            avg_gpu_memory = np.mean([m.value for m in gpu_memory_metrics[-5:]])
            if avg_gpu_memory > 90:
                bottlenecks.append(BottleneckInfo(
                    component='gpu_memory',
                    severity='critical',
                    description=f'GPU memory utilization critical: {avg_gpu_memory:.1f}%',
                    root_cause='GPU memory exhaustion',
                    impact_score=min(1.0, (avg_gpu_memory - 85) / 15),
                    suggested_resolution=[
                        'Reduce batch sizes',
                        'Implement gradient checkpointing',
                        'Use memory-mapped tensors',
                        'Clear unused GPU cache'
                    ],
                    detection_confidence=0.9,
                    first_detected=time.time(),
                    last_seen=time.time()
                ))
                
        return bottlenecks
        
    def _detect_io_bottleneck(self, metrics: List[PerformanceMetric]) -> List[BottleneckInfo]:
        """Detect I/O bottlenecks."""
        io_metrics = [m for m in metrics if 'io' in m.name or 'disk' in m.name]
        
        bottlenecks = []
        # Implementation would analyze disk I/O patterns, network I/O, etc.
        
        return bottlenecks
        
    def _detect_queue_bottleneck(self, metrics: List[PerformanceMetric]) -> List[BottleneckInfo]:
        """Detect queue bottlenecks."""
        queue_metrics = [m for m in metrics if 'queue' in m.name]
        if len(queue_metrics) < 5:
            return []
            
        queue_sizes = [m.value for m in queue_metrics[-10:]]
        avg_queue_size = np.mean(queue_sizes)
        
        bottlenecks = []
        
        if avg_queue_size > 1000:
            bottlenecks.append(BottleneckInfo(
                component='processing_queue',
                severity='high',
                description=f'Processing queue backing up: {avg_queue_size:.0f} items',
                root_cause='Processing capacity insufficient for incoming load',
                impact_score=min(1.0, avg_queue_size / 2000),
                suggested_resolution=[
                    'Scale up processing workers',
                    'Optimize processing algorithms',
                    'Implement load shedding',
                    'Add priority queuing'
                ],
                detection_confidence=0.8,
                first_detected=time.time(),
                last_seen=time.time()
            ))
            
        return bottlenecks
        
    def _detect_cache_bottleneck(self, metrics: List[PerformanceMetric]) -> List[BottleneckInfo]:
        """Detect cache bottlenecks."""
        cache_metrics = [m for m in metrics if 'cache' in m.name]
        
        bottlenecks = []
        
        hit_rate_metrics = [m for m in cache_metrics if 'hit_rate' in m.name]
        if hit_rate_metrics:
            avg_hit_rate = np.mean([m.value for m in hit_rate_metrics[-10:]])
            if avg_hit_rate < 0.6:  # Less than 60% hit rate
                bottlenecks.append(BottleneckInfo(
                    component='cache',
                    severity='medium',
                    description=f'Cache hit rate low: {avg_hit_rate:.1%}',
                    root_cause='Cache misses causing performance degradation',
                    impact_score=1.0 - avg_hit_rate,
                    suggested_resolution=[
                        'Increase cache size',
                        'Improve cache replacement policy',
                        'Optimize data access patterns',
                        'Implement predictive caching'
                    ],
                    detection_confidence=0.7,
                    first_detected=time.time(),
                    last_seen=time.time()
                ))
                
        return bottlenecks
        
    def _detect_network_bottleneck(self, metrics: List[PerformanceMetric]) -> List[BottleneckInfo]:
        """Detect network bottlenecks."""
        network_metrics = [m for m in metrics if 'network' in m.name]
        
        bottlenecks = []
        # Implementation would analyze network latency, bandwidth usage, etc.
        
        return bottlenecks
        
    def _update_bottleneck(self, new_bottleneck: BottleneckInfo):
        """Update bottleneck information."""
        key = new_bottleneck.component
        
        if key in self.detected_bottlenecks:
            existing = self.detected_bottlenecks[key]
            existing.last_seen = time.time()
            existing.occurrence_count += 1
            # Update severity if worse
            severities = ['low', 'medium', 'high', 'critical']
            if severities.index(new_bottleneck.severity) > severities.index(existing.severity):
                existing.severity = new_bottleneck.severity
                existing.description = new_bottleneck.description
        else:
            self.detected_bottlenecks[key] = new_bottleneck
            
    def get_active_bottlenecks(self, max_age_seconds: float = 300) -> List[BottleneckInfo]:
        """Get currently active bottlenecks."""
        current_time = time.time()
        active = []
        
        for bottleneck in self.detected_bottlenecks.values():
            if current_time - bottleneck.last_seen <= max_age_seconds:
                active.append(bottleneck)
                
        return sorted(active, key=lambda b: b.impact_score, reverse=True)
        
    def resolve_bottleneck(self, component: str):
        """Mark bottleneck as resolved."""
        if component in self.detected_bottlenecks:
            del self.detected_bottlenecks[component]


class AdvancedTelemetrySystem:
    """Advanced telemetry system with sub-millisecond accuracy."""
    
    def __init__(
        self,
        buffer_capacity: int = 100000,
        enable_flame_graphs: bool = True,
        enable_bottleneck_detection: bool = True,
        collection_interval_ms: float = 100  # 100ms default
    ):
        self.buffer_capacity = buffer_capacity
        self.enable_flame_graphs = enable_flame_graphs
        self.enable_bottleneck_detection = enable_bottleneck_detection
        self.collection_interval_ms = collection_interval_ms
        
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.metrics_buffer = MetricsBuffer(buffer_capacity)
        self.flame_profiler = FlameGraphProfiler() if enable_flame_graphs else None
        self.bottleneck_detector = BottleneckDetector() if enable_bottleneck_detection else None
        
        # Prometheus integration
        self.prometheus_metrics = {}
        if PROMETHEUS_AVAILABLE:
            self._initialize_prometheus_metrics()
            
        # Collection thread
        self.is_collecting = False
        self.collection_thread = None
        
        # System integration
        self.gpu_processor = get_distributed_gpu_processor()
        self.async_pipeline = get_async_event_pipeline()
        self.cache_system = get_intelligent_cache()
        self.autoscaler = get_intelligent_autoscaler()
        
        # Performance tracking
        self.collection_stats = {
            'metrics_collected': 0,
            'collection_cycles': 0,
            'avg_collection_time_us': 0.0,
            'bottlenecks_detected': 0,
            'last_collection_time': 0.0
        }
        
    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        # Core system metrics
        self.prometheus_metrics['cpu_utilization'] = Gauge(
            'neuromorphic_cpu_utilization_percent',
            'CPU utilization percentage'
        )
        
        self.prometheus_metrics['memory_utilization'] = Gauge(
            'neuromorphic_memory_utilization_percent',
            'Memory utilization percentage'
        )
        
        self.prometheus_metrics['gpu_utilization'] = Gauge(
            'neuromorphic_gpu_utilization_percent',
            'GPU utilization percentage'
        )
        
        self.prometheus_metrics['event_throughput'] = Gauge(
            'neuromorphic_event_throughput_eps',
            'Events processed per second'
        )
        
        self.prometheus_metrics['processing_latency'] = Histogram(
            'neuromorphic_processing_latency_seconds',
            'Processing latency distribution',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        self.prometheus_metrics['cache_hit_rate'] = Gauge(
            'neuromorphic_cache_hit_rate',
            'Cache hit rate'
        )
        
        # Bottleneck metrics
        self.prometheus_metrics['active_bottlenecks'] = Gauge(
            'neuromorphic_active_bottlenecks',
            'Number of active bottlenecks',
            ['component', 'severity']
        )
        
    def start_telemetry(self):
        """Start advanced telemetry collection."""
        if self.is_collecting:
            return
            
        self.is_collecting = True
        
        # Start flame profiler
        if self.flame_profiler:
            self.flame_profiler.start_profiling()
            
        # Start collection thread
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self.collection_thread.start()
        
        self.logger.info("Advanced telemetry system started")
        
    def stop_telemetry(self):
        """Stop advanced telemetry collection."""
        if not self.is_collecting:
            return
            
        self.is_collecting = False
        
        # Stop flame profiler
        if self.flame_profiler:
            self.flame_profiler.stop_profiling()
            
        # Stop collection thread
        if self.collection_thread:
            self.collection_thread.join(timeout=10.0)
            
        self.logger.info("Advanced telemetry system stopped")
        
    def _collection_loop(self):
        """Main telemetry collection loop."""
        while self.is_collecting:
            cycle_start = time.time_ns()
            
            try:
                # Collect comprehensive metrics
                metrics = self._collect_comprehensive_metrics()
                
                # Store in buffer
                for metric in metrics:
                    self.metrics_buffer.put(metric)
                    
                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE:
                    self._update_prometheus_metrics(metrics)
                    
                # Detect bottlenecks
                if self.bottleneck_detector:
                    self.bottleneck_detector.analyze_metrics(metrics)
                    
                # Update collection stats
                cycle_time_ns = time.time_ns() - cycle_start
                self.collection_stats['metrics_collected'] += len(metrics)
                self.collection_stats['collection_cycles'] += 1
                
                # Update average collection time
                cycle_time_us = cycle_time_ns / 1000
                alpha = 0.1
                if self.collection_stats['avg_collection_time_us'] == 0:
                    self.collection_stats['avg_collection_time_us'] = cycle_time_us
                else:
                    self.collection_stats['avg_collection_time_us'] = (
                        alpha * cycle_time_us + 
                        (1 - alpha) * self.collection_stats['avg_collection_time_us']
                    )
                    
                self.collection_stats['last_collection_time'] = time.time()
                
                # Sleep until next collection
                sleep_time = max(0, (self.collection_interval_ms / 1000) - (cycle_time_ns / 1e9))
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Telemetry collection error: {e}")
                time.sleep(self.collection_interval_ms / 1000)
                
    def _collect_comprehensive_metrics(self) -> List[PerformanceMetric]:
        """Collect comprehensive system metrics."""
        current_time_ns = time.time_ns()
        metrics = []
        
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            metrics.extend([
                PerformanceMetric(
                    name='cpu_utilization',
                    value=cpu_percent,
                    unit='percent',
                    timestamp_ns=current_time_ns,
                    source='system'
                ),
                PerformanceMetric(
                    name='memory_utilization',
                    value=memory.percent,
                    unit='percent',
                    timestamp_ns=current_time_ns,
                    source='system'
                ),
                PerformanceMetric(
                    name='memory_available',
                    value=memory.available / (1024**3),  # GB
                    unit='GB',
                    timestamp_ns=current_time_ns,
                    source='system'
                )
            ])
            
            # GPU metrics
            gpu_stats = self.gpu_processor.get_processing_stats()
            if gpu_stats['resource_stats']['total_devices'] > 0:
                gpu_util = gpu_stats['resource_stats']['average_utilization']
                gpu_memory_percent = (
                    gpu_stats['resource_stats']['used_memory_mb'] /
                    gpu_stats['resource_stats']['total_memory_mb'] * 100
                )
                
                metrics.extend([
                    PerformanceMetric(
                        name='gpu_utilization',
                        value=gpu_util,
                        unit='percent',
                        timestamp_ns=current_time_ns,
                        source='gpu'
                    ),
                    PerformanceMetric(
                        name='gpu_memory_utilization',
                        value=gpu_memory_percent,
                        unit='percent',
                        timestamp_ns=current_time_ns,
                        source='gpu'
                    )
                ])
                
            # Processing pipeline metrics
            pipeline_stats = self.async_pipeline.get_comprehensive_stats()
            pipeline_data = pipeline_stats['pipeline']
            
            metrics.extend([
                PerformanceMetric(
                    name='event_throughput',
                    value=pipeline_data['current_throughput_eps'],
                    unit='events/second',
                    timestamp_ns=current_time_ns,
                    source='pipeline'
                ),
                PerformanceMetric(
                    name='processing_latency',
                    value=pipeline_data['average_latency_us'],
                    unit='microseconds',
                    timestamp_ns=current_time_ns,
                    source='pipeline'
                ),
                PerformanceMetric(
                    name='queue_utilization',
                    value=pipeline_data['queue_utilization_percent'],
                    unit='percent',
                    timestamp_ns=current_time_ns,
                    source='pipeline'
                )
            ])
            
            # Cache metrics
            cache_stats = self.cache_system.get_comprehensive_stats()
            
            metrics.append(
                PerformanceMetric(
                    name='cache_hit_rate',
                    value=cache_stats['global']['overall_hit_rate'],
                    unit='ratio',
                    timestamp_ns=current_time_ns,
                    source='cache'
                )
            )
            
            # Auto-scaler metrics
            scaler_stats = self.autoscaler.get_intelligent_scaling_stats()
            
            metrics.extend([
                PerformanceMetric(
                    name='worker_count',
                    value=scaler_stats['intelligent_scaling']['current_workers'],
                    unit='count',
                    timestamp_ns=current_time_ns,
                    source='autoscaler'
                ),
                PerformanceMetric(
                    name='scaling_decision_accuracy',
                    value=scaler_stats['intelligent_scaling']['average_decision_accuracy'],
                    unit='ratio',
                    timestamp_ns=current_time_ns,
                    source='autoscaler'
                )
            ])
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            
        return metrics
        
    def _update_prometheus_metrics(self, metrics: List[PerformanceMetric]):
        """Update Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        for metric in metrics:
            try:
                if metric.name in self.prometheus_metrics:
                    prom_metric = self.prometheus_metrics[metric.name]
                    
                    if isinstance(prom_metric, Histogram):
                        prom_metric.observe(metric.value)
                    elif isinstance(prom_metric, Gauge):
                        prom_metric.set(metric.value)
                        
            except Exception as e:
                self.logger.debug(f"Error updating Prometheus metric {metric.name}: {e}")
                
    @contextmanager
    def profile_operation(self, operation_name: str, **tags):
        """Context manager for profiling operations with flame graphs."""
        timer = HighPrecisionTimer()
        
        # Flame graph profiling if enabled
        if self.flame_profiler:
            with self.flame_profiler.profile_function(
                operation_name,
                traceback.extract_stack()[-2].filename,
                traceback.extract_stack()[-2].lineno
            ) as flame_timer:
                with timer.measure():
                    yield timer
        else:
            with timer.measure():
                yield timer
                
        # Record performance metric
        metric = PerformanceMetric(
            name=f'operation_latency_{operation_name}',
            value=timer.elapsed_us(),
            unit='microseconds',
            timestamp_ns=time.time_ns(),
            source='profiler',
            tags=tags
        )
        
        self.metrics_buffer.put(metric)
        
    def record_custom_metric(
        self,
        name: str,
        value: float,
        unit: str,
        source: str = 'custom',
        **tags
    ):
        """Record custom metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp_ns=time.time_ns(),
            source=source,
            tags=tags
        )
        
        self.metrics_buffer.put(metric)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        recent_metrics = self.metrics_buffer.peek_recent(1000)
        
        # Group metrics by name
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.name].append(metric.value)
            
        # Calculate statistics
        summary = {
            'collection_stats': self.collection_stats.copy(),
            'metric_summary': {}
        }
        
        for name, values in metric_groups.items():
            if values:
                summary['metric_summary'][name] = {
                    'current': values[-1] if values else 0,
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'sample_count': len(values)
                }
                
        # Add bottleneck information
        if self.bottleneck_detector:
            active_bottlenecks = self.bottleneck_detector.get_active_bottlenecks()
            summary['bottlenecks'] = {
                'active_count': len(active_bottlenecks),
                'critical_count': sum(1 for b in active_bottlenecks if b.severity == 'critical'),
                'high_count': sum(1 for b in active_bottlenecks if b.severity == 'high'),
                'details': [
                    {
                        'component': b.component,
                        'severity': b.severity,
                        'description': b.description,
                        'impact_score': b.impact_score,
                        'suggestions': b.suggested_resolution[:3]  # Top 3 suggestions
                    }
                    for b in active_bottlenecks[:5]  # Top 5 bottlenecks
                ]
            }
            
        # Add flame graph hotspots
        if self.flame_profiler:
            hotspots = self.flame_profiler.get_hotspots(10)
            summary['hotspots'] = {
                'top_functions': hotspots[:5],
                'total_functions_profiled': len(hotspots)
            }
            
        return summary
        
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        summary = self.get_performance_summary()
        
        # Add system health assessment
        health_score = self._calculate_system_health_score(summary)
        
        # Add recommendations
        recommendations = self._generate_performance_recommendations(summary)
        
        report = {
            'timestamp': time.time(),
            'system_health_score': health_score,
            'performance_summary': summary,
            'recommendations': recommendations,
            'flame_graph_data': self.flame_profiler.generate_flame_graph_data() if self.flame_profiler else None
        }
        
        return report
        
    def _calculate_system_health_score(self, summary: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)."""
        score = 100.0
        
        # Deduct for high resource utilization
        metrics = summary.get('metric_summary', {})
        
        if 'cpu_utilization' in metrics:
            cpu_util = metrics['cpu_utilization']['current']
            if cpu_util > 90:
                score -= 20
            elif cpu_util > 80:
                score -= 10
                
        if 'memory_utilization' in metrics:
            memory_util = metrics['memory_utilization']['current']
            if memory_util > 95:
                score -= 25
            elif memory_util > 85:
                score -= 15
                
        # Deduct for active bottlenecks
        bottlenecks = summary.get('bottlenecks', {})
        critical_bottlenecks = bottlenecks.get('critical_count', 0)
        high_bottlenecks = bottlenecks.get('high_count', 0)
        
        score -= critical_bottlenecks * 15
        score -= high_bottlenecks * 10
        
        # Deduct for poor cache performance
        if 'cache_hit_rate' in metrics:
            hit_rate = metrics['cache_hit_rate']['current']
            if hit_rate < 0.6:
                score -= 15
            elif hit_rate < 0.8:
                score -= 8
                
        return max(0.0, score)
        
    def _generate_performance_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        metrics = summary.get('metric_summary', {})
        bottlenecks = summary.get('bottlenecks', {})
        
        # High-level recommendations based on metrics
        if 'cpu_utilization' in metrics and metrics['cpu_utilization']['current'] > 85:
            recommendations.append("Consider scaling up CPU resources or optimizing compute-intensive algorithms")
            
        if 'memory_utilization' in metrics and metrics['memory_utilization']['current'] > 90:
            recommendations.append("Memory usage is critical - investigate potential memory leaks and consider increasing available memory")
            
        if 'cache_hit_rate' in metrics and metrics['cache_hit_rate']['current'] < 0.7:
            recommendations.append("Cache performance is suboptimal - consider increasing cache size or optimizing access patterns")
            
        if 'processing_latency' in metrics and metrics['processing_latency']['p95'] > 50000:  # 50ms
            recommendations.append("High processing latency detected - optimize critical path algorithms")
            
        # Add bottleneck-specific recommendations
        for bottleneck in bottlenecks.get('details', [])[:3]:  # Top 3 bottlenecks
            recommendations.extend(bottleneck['suggestions'][:2])  # Top 2 suggestions per bottleneck
            
        return list(set(recommendations))  # Remove duplicates


# Global telemetry instance
_global_telemetry_system = None


def get_telemetry_system() -> AdvancedTelemetrySystem:
    """Get global telemetry system instance."""
    global _global_telemetry_system
    if _global_telemetry_system is None:
        _global_telemetry_system = AdvancedTelemetrySystem()
    return _global_telemetry_system


# Convenient decorators
def profile_performance(operation_name: str = None):
    """Decorator for automatic performance profiling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            telemetry = get_telemetry_system()
            
            with telemetry.profile_operation(op_name):
                return func(*args, **kwargs)
                
        return wrapper
    return decorator