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
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingPolicy:
    """Configuration for auto-scaling behavior."""
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
    
    # Scaling parameters
    min_instances: int = 1
    max_instances: int = 10
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes
    
    # Stability requirements
    metrics_window_size: int = 10
    threshold_breach_ratio: float = 0.6  # 60% of metrics must breach threshold


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
        
        # Scaling state
        self.current_instances = self.policy.min_instances
        self.last_scale_up_time = 0.0
        self.last_scale_down_time = 0.0
        self.scaling_actions = []
        
        # Control
        self.is_running = False
        self.scaling_thread = None
        
        self.logger = logging.getLogger(__name__)
        
    def start(self):
        """Start auto-scaling system."""
        if self.is_running:
            return
            
        self.is_running = True
        self.resource_monitor.start_monitoring()
        
        self.scaling_thread = threading.Thread(
            target=self._scaling_loop,
            daemon=True
        )
        self.scaling_thread.start()
        
        self.logger.info("Auto-scaler started")
        
    def stop(self):
        """Stop auto-scaling system."""
        self.is_running = False
        self.resource_monitor.stop_monitoring()
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5.0)
            
        self.logger.info("Auto-scaler stopped")
        
    def _scaling_loop(self):
        """Main scaling decision loop."""
        while self.is_running:
            try:
                self._evaluate_scaling()
                time.sleep(30.0)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Scaling evaluation error: {e}")
                time.sleep(30.0)
                
    def _evaluate_scaling(self):
        """Evaluate whether scaling is needed."""
        current_time = time.time()
        
        # Get recent metrics
        recent_metrics = self.resource_monitor.get_recent_metrics(
            self.policy.metrics_window_size
        )
        
        if len(recent_metrics) < self.policy.metrics_window_size:
            return  # Not enough data
            
        # Check scale-up conditions
        if self._should_scale_up(recent_metrics, current_time):
            self._scale_up()
            
        # Check scale-down conditions  
        elif self._should_scale_down(recent_metrics, current_time):
            self._scale_down()
            
    def _should_scale_up(self, metrics_list: List[ResourceMetrics], current_time: float) -> bool:
        """Check if scale-up is needed."""
        # Check cooldown period
        if current_time - self.last_scale_up_time < self.policy.scale_up_cooldown:
            return False
            
        # Check instance limit
        if self.current_instances >= self.policy.max_instances:
            return False
            
        # Check if enough metrics breach thresholds
        breach_counts = {trigger.__class__.__name__: 0 for trigger in self.triggers}
        
        for metrics in metrics_list:
            for trigger in self.triggers:
                if trigger.should_scale_up(metrics, self.policy):
                    breach_counts[trigger.__class__.__name__] += 1
                    
        # Calculate breach ratios
        total_metrics = len(metrics_list)
        threshold_count = int(total_metrics * self.policy.threshold_breach_ratio)
        
        # At least one trigger must have enough breaches
        for trigger_name, breach_count in breach_counts.items():
            if breach_count >= threshold_count:
                self.logger.info(
                    f"Scale-up triggered by {trigger_name}: "
                    f"{breach_count}/{total_metrics} metrics breached threshold"
                )
                return True
                
        return False
        
    def _should_scale_down(self, metrics_list: List[ResourceMetrics], current_time: float) -> bool:
        """Check if scale-down is needed."""
        # Check cooldown period
        if current_time - self.last_scale_down_time < self.policy.scale_down_cooldown:
            return False
            
        # Check instance minimum
        if self.current_instances <= self.policy.min_instances:
            return False
            
        # All triggers must indicate scale-down is safe
        total_metrics = len(metrics_list)
        threshold_count = int(total_metrics * self.policy.threshold_breach_ratio)
        
        all_triggers_safe = True
        
        for trigger in self.triggers:
            safe_count = 0
            for metrics in metrics_list:
                if trigger.should_scale_down(metrics, self.policy):
                    safe_count += 1
                    
            if safe_count < threshold_count:
                all_triggers_safe = False
                break
                
        if all_triggers_safe:
            self.logger.info("Scale-down conditions met for all triggers")
            return True
            
        return False
        
    def _scale_up(self):
        """Scale up instances."""
        if self.current_instances >= self.policy.max_instances:
            return
            
        old_instances = self.current_instances
        self.current_instances = min(
            self.current_instances + 1,
            self.policy.max_instances
        )
        self.last_scale_up_time = time.time()
        
        # Record scaling action
        action = {
            'action': 'scale_up',
            'from_instances': old_instances,
            'to_instances': self.current_instances,
            'timestamp': self.last_scale_up_time
        }
        self.scaling_actions.append(action)
        
        self.logger.info(
            f"Scaled up from {old_instances} to {self.current_instances} instances"
        )
        
        # TODO: Implement actual instance creation
        self._create_instance()
        
    def _scale_down(self):
        """Scale down instances."""
        if self.current_instances <= self.policy.min_instances:
            return
            
        old_instances = self.current_instances
        self.current_instances = max(
            self.current_instances - 1,
            self.policy.min_instances
        )
        self.last_scale_down_time = time.time()
        
        # Record scaling action
        action = {
            'action': 'scale_down',
            'from_instances': old_instances,
            'to_instances': self.current_instances,
            'timestamp': self.last_scale_down_time
        }
        self.scaling_actions.append(action)
        
        self.logger.info(
            f"Scaled down from {old_instances} to {self.current_instances} instances"
        )
        
        # TODO: Implement actual instance termination
        self._remove_instance()
        
    def _create_instance(self):
        """Create new processing instance."""
        # Placeholder for actual instance creation
        # In practice, this might involve:
        # - Spawning new processes
        # - Starting new containers
        # - Allocating new GPU resources
        # - Updating load balancer configuration
        pass
        
    def _remove_instance(self):
        """Remove processing instance."""
        # Placeholder for actual instance removal
        # In practice, this might involve:
        # - Gracefully shutting down processes
        # - Stopping containers
        # - Releasing GPU resources
        # - Updating load balancer configuration
        pass
        
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        recent_metrics = self.resource_monitor.get_average_metrics(5)
        
        return {
            'current_instances': self.current_instances,
            'min_instances': self.policy.min_instances,
            'max_instances': self.policy.max_instances,
            'last_scale_up': self.last_scale_up_time,
            'last_scale_down': self.last_scale_down_time,
            'recent_actions': self.scaling_actions[-10:],  # Last 10 actions
            'current_metrics': {
                'cpu_percent': recent_metrics.cpu_percent,
                'memory_percent': recent_metrics.memory_percent,
                'queue_size': recent_metrics.inference_queue_size,
                'average_latency': recent_metrics.average_inference_time
            }
        }


class LoadBalancer:
    """Load balancer for distributing inference requests across instances."""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        self.instances = {}
        self.instance_health = {}
        self.instance_connections = {}
        self.instance_response_times = {}
        
        # Round-robin state
        self.current_instance_index = 0
        
        # Health checking
        self.health_check_thread = None
        self.is_health_checking = False
        
        self.logger = logging.getLogger(__name__)
        
    def add_instance(self, instance_id: str, endpoint: str, weight: float = 1.0):
        """Add instance to load balancer."""
        self.instances[instance_id] = {
            'endpoint': endpoint,
            'weight': weight,
            'created_at': time.time()
        }
        self.instance_health[instance_id] = True
        self.instance_connections[instance_id] = 0
        self.instance_response_times[instance_id] = deque(maxlen=100)
        
        self.logger.info(f"Added instance {instance_id} at {endpoint}")
        
    def remove_instance(self, instance_id: str):
        """Remove instance from load balancer."""
        if instance_id in self.instances:
            del self.instances[instance_id]
            del self.instance_health[instance_id]
            del self.instance_connections[instance_id]
            del self.instance_response_times[instance_id]
            
            self.logger.info(f"Removed instance {instance_id}")
            
    def get_next_instance(self, request_context: Optional[Dict] = None) -> Optional[str]:
        """Get next instance for request using configured algorithm."""
        healthy_instances = [
            instance_id for instance_id, healthy in self.instance_health.items()
            if healthy
        ]
        
        if not healthy_instances:
            return None
            
        if self.config.algorithm == "round_robin":
            return self._round_robin_selection(healthy_instances)
        elif self.config.algorithm == "least_connections":
            return self._least_connections_selection(healthy_instances)
        elif self.config.algorithm == "weighted_response_time":
            return self._weighted_response_time_selection(healthy_instances)
        else:
            return self._round_robin_selection(healthy_instances)
            
    def _round_robin_selection(self, healthy_instances: List[str]) -> str:
        """Round-robin instance selection."""
        if not healthy_instances:
            return None
            
        instance = healthy_instances[self.current_instance_index % len(healthy_instances)]
        self.current_instance_index += 1
        return instance
        
    def _least_connections_selection(self, healthy_instances: List[str]) -> str:
        """Least connections instance selection."""
        min_connections = float('inf')
        best_instance = None
        
        for instance_id in healthy_instances:
            connections = self.instance_connections.get(instance_id, 0)
            if connections < min_connections:
                min_connections = connections
                best_instance = instance_id
                
        return best_instance
        
    def _weighted_response_time_selection(self, healthy_instances: List[str]) -> str:
        """Weighted response time instance selection."""
        best_score = float('inf')
        best_instance = None
        
        for instance_id in healthy_instances:
            weight = self.instances[instance_id]['weight']
            response_times = self.instance_response_times.get(instance_id, deque([100]))
            avg_response_time = np.mean(response_times) if response_times else 100
            
            # Lower is better (response time / weight)
            score = avg_response_time / weight
            
            if score < best_score:
                best_score = score
                best_instance = instance_id
                
        return best_instance
        
    def record_request_start(self, instance_id: str):
        """Record request start for connection tracking."""
        if instance_id in self.instance_connections:
            self.instance_connections[instance_id] += 1
            
    def record_request_end(self, instance_id: str, response_time: float):
        """Record request completion."""
        if instance_id in self.instance_connections:
            self.instance_connections[instance_id] = max(
                0, self.instance_connections[instance_id] - 1
            )
            
        if instance_id in self.instance_response_times:
            self.instance_response_times[instance_id].append(response_time)
            
    def start_health_checks(self):
        """Start periodic health checks."""
        if self.is_health_checking:
            return
            
        self.is_health_checking = True
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
        
        self.logger.info("Health checking started")
        
    def stop_health_checks(self):
        """Stop health checks."""
        self.is_health_checking = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5.0)
            
        self.logger.info("Health checking stopped")
        
    def _health_check_loop(self):
        """Main health check loop."""
        while self.is_health_checking:
            try:
                self._perform_health_checks()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                time.sleep(self.config.health_check_interval)
                
    def _perform_health_checks(self):
        """Perform health checks on all instances."""
        for instance_id, instance_info in self.instances.items():
            try:
                # Placeholder health check - in practice would ping endpoint
                # For now, assume all instances are healthy
                is_healthy = self._check_instance_health(instance_info['endpoint'])
                self.instance_health[instance_id] = is_healthy
            except Exception as e:
                self.logger.warning(f"Health check failed for {instance_id}: {e}")
                self.instance_health[instance_id] = False
                
    def _check_instance_health(self, endpoint: str) -> bool:
        """Check health of specific instance."""
        # Placeholder implementation
        # In practice, would make HTTP request or ping
        return True
        
    def get_status(self) -> Dict[str, Any]:
        """Get load balancer status."""
        healthy_count = sum(1 for healthy in self.instance_health.values() if healthy)
        total_count = len(self.instances)
        
        return {
            'algorithm': self.config.algorithm,
            'total_instances': total_count,
            'healthy_instances': healthy_count,
            'instance_details': {
                instance_id: {
                    'endpoint': info['endpoint'],
                    'healthy': self.instance_health.get(instance_id, False),
                    'connections': self.instance_connections.get(instance_id, 0),
                    'avg_response_time': np.mean(self.instance_response_times.get(instance_id, [0]))
                }
                for instance_id, info in self.instances.items()
            }
        }


# Global instances
_global_auto_scaler = None
_global_load_balancer = None


def get_auto_scaler(policy: Optional[ScalingPolicy] = None) -> AutoScaler:
    """Get global auto-scaler instance."""
    global _global_auto_scaler
    if _global_auto_scaler is None:
        _global_auto_scaler = AutoScaler(policy)
    return _global_auto_scaler


def get_load_balancer(config: Optional[LoadBalancerConfig] = None) -> LoadBalancer:
    """Get global load balancer instance."""
    global _global_load_balancer
    if _global_load_balancer is None:
        _global_load_balancer = LoadBalancer(config)
    return _global_load_balancer


# Utility functions
def create_scaling_policy(**kwargs) -> ScalingPolicy:
    """Create scaling policy with custom parameters."""
    return ScalingPolicy(**kwargs)


def create_load_balancer_config(**kwargs) -> LoadBalancerConfig:
    """Create load balancer configuration."""
    return LoadBalancerConfig(**kwargs)


@contextmanager
def load_balanced_request(instance_id: str, load_balancer: LoadBalancer):
    """Context manager for load-balanced requests."""
    start_time = time.time()
    load_balancer.record_request_start(instance_id)
    
    try:
        yield
    finally:
        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # Convert to ms
        load_balancer.record_request_end(instance_id, response_time)


class ScalingOrchestrator:
    """High-level orchestrator for scaling and load balancing."""
    
    def __init__(
        self,
        scaling_policy: Optional[ScalingPolicy] = None,
        lb_config: Optional[LoadBalancerConfig] = None
    ):
        self.auto_scaler = AutoScaler(scaling_policy)
        self.load_balancer = LoadBalancer(lb_config)
        self.logger = logging.getLogger(__name__)
        
    def start(self):
        """Start orchestrator components."""
        self.auto_scaler.start()
        self.load_balancer.start_health_checks()
        self.logger.info("Scaling orchestrator started")
        
    def stop(self):  
        """Stop orchestrator components."""
        self.auto_scaler.stop()
        self.load_balancer.stop_health_checks()
        self.logger.info("Scaling orchestrator stopped")
        
    def add_instance(self, instance_id: str, endpoint: str, weight: float = 1.0):
        """Add instance to load balancer."""
        self.load_balancer.add_instance(instance_id, endpoint, weight)
        
    def remove_instance(self, instance_id: str):
        """Remove instance from load balancer."""
        self.load_balancer.remove_instance(instance_id)
        
    def get_next_instance(self, request_context: Optional[Dict] = None) -> Optional[str]:
        """Get next instance for processing."""
        return self.load_balancer.get_next_instance(request_context)
        
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status."""
        return {
            'auto_scaler': self.auto_scaler.get_scaling_status(),
            'load_balancer': self.load_balancer.get_status(),
            'timestamp': time.time()
        }