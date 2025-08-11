"""
Auto-scaling and load balancing for spike-snn-event-vision-kit.

Provides intelligent resource management, auto-scaling triggers, and load
balancing for high-throughput neuromorphic vision processing workloads.
"""

import time
import threading
import psutil
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import deque
import numpy as np
from contextlib import contextmanager
import asyncio
from queue import Queue, PriorityQueue
import json
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .monitoring import get_metrics_collector
from .concurrency import get_concurrent_processor, ConcurrentProcessor
from .validation import ValidationError
from .intelligent_cache_system import get_intelligent_cache
from .gpu_distributed_processor import get_distributed_gpu_processor
from .async_event_processor import get_async_event_pipeline


@dataclass
class ResourceMetrics:
    """System resource metrics snapshot."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_utilization: float = 0.0
    disk_io_percent: float = 0.0
    network_io_mbps: float = 0.0
    inference_queue_size: int = 0
    average_inference_time: float = 0.0
    error_rate: float = 0.0
    event_throughput_eps: float = 0.0
    cache_hit_rate: float = 0.0
    pipeline_latency_ns: float = 0.0
    gpu_memory_fragmentation: float = 0.0
    timestamp: float = field(default_factory=time.time)
    

@dataclass
class WorkloadPrediction:
    """Workload prediction for proactive scaling."""
    predicted_cpu_percent: float = 0.0
    predicted_memory_percent: float = 0.0
    predicted_throughput_eps: float = 0.0
    predicted_latency_ms: float = 0.0
    confidence_score: float = 0.0
    prediction_horizon_seconds: float = 300.0
    prediction_timestamp: float = field(default_factory=time.time)
    

@dataclass
class ScalingDecision:
    """Intelligent scaling decision with reasoning."""
    action: str  # 'scale_up', 'scale_down', 'no_action'
    target_workers: int
    confidence: float
    reasoning: List[str]
    expected_improvement: Dict[str, float]
    risk_assessment: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingPolicy:
    """Configuration for auto-scaling behavior."""
    # Worker/instance parameters
    min_workers: int = 1
    max_workers: int = 10
    scale_step_size: int = 1
    
    # CPU thresholds
    cpu_scale_up_threshold: float = 70.0
    cpu_scale_down_threshold: float = 30.0
    
    # Memory thresholds
    memory_scale_up_threshold: float = 80.0
    memory_scale_down_threshold: float = 40.0
    
    # Queue size thresholds
    queue_scale_up_threshold: int = 100
    queue_scale_down_threshold: int = 20
    
    # Latency thresholds (ms)
    latency_scale_up_threshold: float = 500.0
    latency_scale_down_threshold: float = 100.0
    
    # Timing parameters
    scale_cooldown: float = 300.0  # 5 minutes between scaling actions
    scale_check_interval: float = 30.0  # Check every 30 seconds
    
    # Legacy compatibility
    @property
    def min_instances(self) -> int:
        return self.min_workers
    
    @property  
    def max_instances(self) -> int:
        return self.max_workers


@dataclass
class LoadBalancerConfig:
    """Configuration for load balancing."""
    algorithm: str = "round_robin"  # round_robin, least_connections, weighted_response_time
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    sticky_sessions: bool = False
    connection_timeout: float = 30.0


class ScalingTrigger(ABC):
    """Abstract base class for scaling triggers."""
    
    @abstractmethod
    def should_scale_up(self, metrics: ResourceMetrics, policy: ScalingPolicy) -> bool:
        """Check if scale-up is needed."""
        pass
        
    @abstractmethod
    def should_scale_down(self, metrics: ResourceMetrics, policy: ScalingPolicy) -> bool:
        """Check if scale-down is needed."""  
        pass


class CPUScalingTrigger(ScalingTrigger):
    """CPU-based scaling trigger."""
    
    def should_scale_up(self, metrics: ResourceMetrics, policy: ScalingPolicy) -> bool:
        return metrics.cpu_percent > policy.cpu_scale_up_threshold
        
    def should_scale_down(self, metrics: ResourceMetrics, policy: ScalingPolicy) -> bool:
        return metrics.cpu_percent < policy.cpu_scale_down_threshold


class MemoryScalingTrigger(ScalingTrigger):
    """Memory-based scaling trigger."""
    
    def should_scale_up(self, metrics: ResourceMetrics, policy: ScalingPolicy) -> bool:
        return metrics.memory_percent > policy.memory_scale_up_threshold
        
    def should_scale_down(self, metrics: ResourceMetrics, policy: ScalingPolicy) -> bool:
        return metrics.memory_percent < policy.memory_scale_down_threshold


class QueueScalingTrigger(ScalingTrigger):
    """Queue size-based scaling trigger."""
    
    def should_scale_up(self, metrics: ResourceMetrics, policy: ScalingPolicy) -> bool:
        return metrics.inference_queue_size > policy.queue_scale_up_threshold
        
    def should_scale_down(self, metrics: ResourceMetrics, policy: ScalingPolicy) -> bool:
        return metrics.inference_queue_size < policy.queue_scale_down_threshold


class LatencyScalingTrigger(ScalingTrigger):
    """Latency-based scaling trigger."""
    
    def should_scale_up(self, metrics: ResourceMetrics, policy: ScalingPolicy) -> bool:
        return metrics.average_inference_time > policy.latency_scale_up_threshold
        
    def should_scale_down(self, metrics: ResourceMetrics, policy: ScalingPolicy) -> bool:
        return metrics.average_inference_time < policy.latency_scale_down_threshold


class ResourceMonitor:
    """Monitors system resources for auto-scaling decisions."""
    
    def __init__(self, monitoring_interval: float = 10.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history: deque = deque(maxlen=100)
        self.is_monitoring = False
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)
        
        # GPU monitoring setup
        self.gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return True
        except Exception:
            pass
        return False
        
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Resource monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                time.sleep(self.monitoring_interval)
                
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_percent = 0.0  # Simplified
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_io_mbps = 0.0  # Simplified
        
        # GPU metrics
        gpu_memory_percent = 0.0
        gpu_utilization = 0.0
        
        if self.gpu_available:
            try:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    gpu_memory_percent = (
                        torch.cuda.memory_allocated() / 
                        torch.cuda.max_memory_allocated() * 100
                    )
                    # GPU utilization would need nvidia-ml-py for accurate reading
                    gpu_utilization = 0.0
            except Exception:
                pass
                
        # Processing metrics (from concurrent processor if available)  
        inference_queue_size = 0
        average_inference_time = 0.0
        error_rate = 0.0
        
        try:
            processor = get_concurrent_processor()
            stats = processor.get_stats()
            inference_queue_size = stats.get('queue_size', 0)
        except Exception:
            pass
            
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_memory_percent=gpu_memory_percent,
            gpu_utilization=gpu_utilization,
            disk_io_percent=disk_io_percent,
            network_io_mbps=network_io_mbps,
            inference_queue_size=inference_queue_size,
            average_inference_time=average_inference_time,
            error_rate=error_rate
        )
        
    def get_recent_metrics(self, window_size: int = 10) -> List[ResourceMetrics]:
        """Get recent metrics within window."""
        return list(self.metrics_history)[-window_size:]
        
    def get_average_metrics(self, window_size: int = 10) -> ResourceMetrics:
        """Get averaged metrics over window."""
        recent_metrics = self.get_recent_metrics(window_size)
        
        if not recent_metrics:
            return ResourceMetrics()
            
        # Calculate averages
        cpu_avg = np.mean([m.cpu_percent for m in recent_metrics])
        memory_avg = np.mean([m.memory_percent for m in recent_metrics])
        gpu_memory_avg = np.mean([m.gpu_memory_percent for m in recent_metrics])
        gpu_util_avg = np.mean([m.gpu_utilization for m in recent_metrics])
        queue_avg = np.mean([m.inference_queue_size for m in recent_metrics])
        latency_avg = np.mean([m.average_inference_time for m in recent_metrics])
        error_avg = np.mean([m.error_rate for m in recent_metrics])
        
        return ResourceMetrics(
            cpu_percent=cpu_avg,
            memory_percent=memory_avg,
            gpu_memory_percent=gpu_memory_avg,
            gpu_utilization=gpu_util_avg,
            inference_queue_size=int(queue_avg),
            average_inference_time=latency_avg,
            error_rate=error_avg
        )


class AutoScaler:
    """Intelligent auto-scaling system for neuromorphic processing workloads."""
    
    def __init__(
        self,
        policy: Optional[ScalingPolicy] = None,
        custom_triggers: Optional[List[ScalingTrigger]] = None
    ):
        self.policy = policy or ScalingPolicy()
        self.resource_monitor = ResourceMonitor()
        
        # Default scaling triggers
        self.triggers = custom_triggers or [
            CPUScalingTrigger(),
            MemoryScalingTrigger(),
            QueueScalingTrigger(),
            LatencyScalingTrigger()
        ]
        
        # Auto-scaling state
        self.current_workers = self.policy.min_workers
        self.last_scale_action = 0.0
        self.scaling_history = deque(maxlen=50)
        self.is_auto_scaling = False
        self.auto_scale_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Worker management
        self.worker_pool = []
        self.worker_queue = Queue()
        
        # Performance tracking
        self.scale_up_count = 0
        self.scale_down_count = 0
        
    def start_auto_scaling(self):
        """Start automatic scaling based on resource metrics."""
        if self.is_auto_scaling:
            return
            
        self.is_auto_scaling = True
        self.resource_monitor.start_monitoring()
        
        self.auto_scale_thread = threading.Thread(
            target=self._auto_scaling_loop,
            daemon=True
        )
        self.auto_scale_thread.start()
        self.logger.info(f"Auto-scaling started with {self.current_workers} workers")
        
    def stop_auto_scaling(self):
        """Stop automatic scaling."""
        self.is_auto_scaling = False
        self.resource_monitor.stop_monitoring()
        
        if self.auto_scale_thread:
            self.auto_scale_thread.join(timeout=5.0)
            
        self.logger.info("Auto-scaling stopped")
        
    def _auto_scaling_loop(self):
        """Main auto-scaling decision loop."""
        while self.is_auto_scaling:
            try:
                # Get averaged metrics over recent window
                metrics = self.resource_monitor.get_average_metrics(window_size=3)
                
                # Make scaling decision
                scale_decision = self._make_scaling_decision(metrics)
                
                if scale_decision != 0:
                    self._execute_scaling_action(scale_decision, metrics)
                    
                # Wait before next evaluation
                time.sleep(self.policy.scale_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in auto-scaling loop: {e}")
                time.sleep(self.policy.scale_check_interval)
                
    def _make_scaling_decision(self, metrics: ResourceMetrics) -> int:
        """Make scaling decision based on current metrics.
        
        Returns:
            int: -1 for scale down, 0 for no action, 1 for scale up
        """
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scale_action < self.policy.scale_cooldown:
            return 0
            
        # Count trigger votes
        scale_up_votes = 0
        scale_down_votes = 0
        
        for trigger in self.triggers:
            if trigger.should_scale_up(metrics, self.policy):
                scale_up_votes += 1
            elif trigger.should_scale_down(metrics, self.policy):
                scale_down_votes += 1
                
        # Decision logic with hysteresis
        if scale_up_votes >= 2 and self.current_workers < self.policy.max_workers:
            self.logger.info(f"Scale up decision: {scale_up_votes} triggers voted")
            return 1
        elif (scale_down_votes >= 2 and 
              self.current_workers > self.policy.min_workers and
              scale_up_votes == 0):  # Hysteresis: no conflicting scale-up signals
            self.logger.info(f"Scale down decision: {scale_down_votes} triggers voted")
            return -1
            
        return 0
        
    def _execute_scaling_action(self, scale_decision: int, metrics: ResourceMetrics):
        """Execute scaling action (scale up or down)."""
        if scale_decision == 1:
            self._scale_up(metrics)
        elif scale_decision == -1:
            self._scale_down(metrics)
            
        self.last_scale_action = time.time()
        
    def _scale_up(self, metrics: ResourceMetrics):
        """Scale up by adding workers."""
        new_workers = min(
            self.current_workers + self.policy.scale_step_size,
            self.policy.max_workers
        )
        
        workers_to_add = new_workers - self.current_workers
        
        if workers_to_add > 0:
            # Add new workers
            for _ in range(workers_to_add):
                worker = self._create_worker()
                self.worker_pool.append(worker)
                
            self.current_workers = new_workers
            self.scale_up_count += 1
            
            # Log scaling action
            scaling_event = {
                'timestamp': time.time(),
                'action': 'scale_up',
                'workers_before': self.current_workers - workers_to_add,
                'workers_after': self.current_workers,
                'reason': {
                    'cpu': metrics.cpu_percent,
                    'memory': metrics.memory_percent,
                    'queue_size': metrics.inference_queue_size,
                    'latency': metrics.average_inference_time
                }
            }
            self.scaling_history.append(scaling_event)
            
            self.logger.info(
                f"Scaled up from {self.current_workers - workers_to_add} to "
                f"{self.current_workers} workers (CPU: {metrics.cpu_percent:.1f}%, "
                f"Memory: {metrics.memory_percent:.1f}%, Queue: {metrics.inference_queue_size})"
            )
            
    def _scale_down(self, metrics: ResourceMetrics):
        """Scale down by removing workers."""
        new_workers = max(
            self.current_workers - self.policy.scale_step_size,
            self.policy.min_workers
        )
        
        workers_to_remove = self.current_workers - new_workers
        
        if workers_to_remove > 0:
            # Remove workers gracefully
            for _ in range(workers_to_remove):
                if self.worker_pool:
                    worker = self.worker_pool.pop()
                    self._shutdown_worker(worker)
                    
            self.current_workers = new_workers
            self.scale_down_count += 1
            
            # Log scaling action
            scaling_event = {
                'timestamp': time.time(),
                'action': 'scale_down',
                'workers_before': self.current_workers + workers_to_remove,
                'workers_after': self.current_workers,
                'reason': {
                    'cpu': metrics.cpu_percent,
                    'memory': metrics.memory_percent,
                    'queue_size': metrics.inference_queue_size,
                    'latency': metrics.average_inference_time
                }
            }
            self.scaling_history.append(scaling_event)
            
            self.logger.info(
                f"Scaled down from {self.current_workers + workers_to_remove} to "
                f"{self.current_workers} workers (CPU: {metrics.cpu_percent:.1f}%, "
                f"Memory: {metrics.memory_percent:.1f}%, Queue: {metrics.inference_queue_size})"
            )
            
    def _create_worker(self):
        """Create a new worker instance."""
        # Placeholder for actual worker creation
        # In real implementation, this would spawn new processing threads/processes
        worker_id = f"worker_{len(self.worker_pool) + 1}"
        worker = {
            'id': worker_id,
            'created_at': time.time(),
            'status': 'active'
        }
        self.logger.debug(f"Created worker: {worker_id}")
        return worker
        
    def _shutdown_worker(self, worker):
        """Gracefully shutdown a worker."""
        # Placeholder for actual worker shutdown
        self.logger.debug(f"Shutting down worker: {worker['id']}")
        worker['status'] = 'shutdown'
        
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        return {
            'current_workers': self.current_workers,
            'min_workers': self.policy.min_workers,
            'max_workers': self.policy.max_workers,
            'scale_up_count': self.scale_up_count,
            'scale_down_count': self.scale_down_count,
            'scaling_history': list(self.scaling_history),
            'last_scale_action': self.last_scale_action,
            'is_auto_scaling': self.is_auto_scaling
        }
        
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get current resource metrics."""
        recent_metrics = self.resource_monitor.get_recent_metrics(window_size=1)
        return recent_metrics[0] if recent_metrics else None


# Global auto-scaler instance  
_global_auto_scaler = None


def get_auto_scaler() -> AutoScaler:
    """Get global auto-scaler instance."""
    global _global_auto_scaler
    if _global_auto_scaler is None:
        _global_auto_scaler = AutoScaler()
    return _global_auto_scaler
