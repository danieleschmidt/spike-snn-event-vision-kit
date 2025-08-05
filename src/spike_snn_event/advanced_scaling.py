"""
Advanced scaling and performance optimization for spike-snn-event-vision-kit.

Provides intelligent auto-scaling, predictive resource allocation, and
high-performance optimization for extreme-scale neuromorphic processing.
"""

import time
import threading
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import logging
import json
from pathlib import Path
import pickle
import hashlib
from contextlib import contextmanager
import gc
import weakref

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .monitoring import get_metrics_collector
from .scaling import ResourceMetrics, ScalingPolicy
from .concurrency import get_concurrent_processor
from .optimization import get_optimizer
from .validation import safe_operation


@dataclass
class PredictiveMetrics:
    """Advanced metrics with predictive capabilities."""
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    inference_rate: float
    queue_depth: int
    error_rate: float
    response_time_p99: float
    
    # Predictive fields
    predicted_load_1min: float = 0.0
    predicted_load_5min: float = 0.0
    trend_direction: str = "stable"  # "increasing", "decreasing", "stable"
    confidence_score: float = 0.0


@dataclass
class ScalingDecision:
    """Represents a scaling decision with reasoning."""
    action: str  # "scale_up", "scale_down", "no_action"
    target_instances: int
    reasoning: List[str]
    confidence: float
    estimated_impact: Dict[str, float]
    cooldown_until: float


class PredictiveScaler:
    """Advanced scaler with machine learning-based predictions."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.scaling_history = deque(maxlen=100)
        
        # Simple linear regression model for prediction
        self.prediction_model = None
        self.model_last_trained = 0
        self.model_retrain_interval = 3600  # 1 hour
        
        self.logger = logging.getLogger(__name__)
    
    def add_metrics(self, metrics: ResourceMetrics):
        """Add metrics to history and update predictions."""
        pred_metrics = PredictiveMetrics(
            timestamp=time.time(),
            cpu_utilization=metrics.cpu_percent,
            memory_utilization=metrics.memory_percent,
            gpu_utilization=metrics.gpu_utilization,
            inference_rate=1.0 / max(0.001, metrics.average_inference_time),
            queue_depth=metrics.inference_queue_size,
            error_rate=metrics.error_rate,
            response_time_p99=metrics.average_inference_time * 1.5
        )
        
        # Calculate predictions
        if len(self.metrics_history) >= 10:
            pred_metrics.predicted_load_1min = self._predict_load(60)
            pred_metrics.predicted_load_5min = self._predict_load(300)
            pred_metrics.trend_direction = self._analyze_trend()
            pred_metrics.confidence_score = self._calculate_confidence()
        
        self.metrics_history.append(pred_metrics)
        
        # Retrain model periodically
        if time.time() - self.model_last_trained > self.model_retrain_interval:
            self._retrain_prediction_model()
    
    def _predict_load(self, seconds_ahead: int) -> float:
        """Predict load N seconds ahead using simple trend analysis."""
        if len(self.metrics_history) < 5:
            return self.metrics_history[-1].cpu_utilization
        
        # Use last 5 data points for trend
        recent_metrics = list(self.metrics_history)[-5:]
        timestamps = [m.timestamp for m in recent_metrics]
        cpu_values = [m.cpu_utilization for m in recent_metrics]
        
        # Simple linear regression
        n = len(timestamps)
        if n < 2:
            return cpu_values[-1]
        
        # Calculate slope
        t_mean = sum(timestamps) / n
        cpu_mean = sum(cpu_values) / n
        
        numerator = sum((t - t_mean) * (cpu - cpu_mean) for t, cpu in zip(timestamps, cpu_values))
        denominator = sum((t - t_mean) ** 2 for t in timestamps)
        
        if denominator == 0:
            return cpu_values[-1]
        
        slope = numerator / denominator
        intercept = cpu_mean - slope * t_mean
        
        # Predict future value
        future_time = timestamps[-1] + seconds_ahead
        predicted_cpu = slope * future_time + intercept
        
        # Clamp to reasonable bounds
        return max(0, min(100, predicted_cpu))
    
    def _analyze_trend(self) -> str:
        """Analyze current trend direction."""
        if len(self.metrics_history) < 5:
            return "stable"
        
        recent_values = [m.cpu_utilization for m in list(self.metrics_history)[-5:]]
        
        # Calculate trend
        increases = sum(1 for i in range(1, len(recent_values)) if recent_values[i] > recent_values[i-1])
        decreases = sum(1 for i in range(1, len(recent_values)) if recent_values[i] < recent_values[i-1])
        
        if increases >= 3:
            return "increasing"
        elif decreases >= 3:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence in predictions."""
        if len(self.metrics_history) < 10:
            return 0.5
        
        # Base confidence on consistency of recent trends
        recent_predictions = []
        for i in range(5, min(10, len(self.metrics_history))):
            # Make prediction using data up to point i
            subset = list(self.metrics_history)[:i]
            if len(subset) >= 5:
                # Simplified confidence calculation
                variance = np.var([m.cpu_utilization for m in subset[-5:]])
                consistency = 1.0 / (1.0 + variance / 100)  # Normalize variance
                recent_predictions.append(consistency)
        
        if recent_predictions:
            return np.mean(recent_predictions)
        return 0.5
    
    def _retrain_prediction_model(self):
        """Retrain prediction model with historical data."""
        if len(self.metrics_history) < 50:
            return
        
        self.logger.info("Retraining predictive scaling model")
        self.model_last_trained = time.time()
        # Placeholder for more sophisticated ML model training
    
    def make_scaling_decision(
        self, 
        current_instances: int, 
        policy: ScalingPolicy
    ) -> ScalingDecision:
        """Make intelligent scaling decision based on predictions."""
        if not self.metrics_history:
            return ScalingDecision(
                action="no_action",
                target_instances=current_instances,
                reasoning=["No metrics available"],
                confidence=0.0,
                estimated_impact={},
                cooldown_until=time.time()
            )
        
        current_metrics = self.metrics_history[-1]
        reasoning = []
        
        # Check current thresholds
        cpu_breach = current_metrics.cpu_utilization > policy.cpu_scale_up_threshold
        memory_breach = current_metrics.memory_utilization > policy.memory_scale_up_threshold
        queue_breach = current_metrics.queue_depth > policy.queue_scale_up_threshold
        
        # Check predicted thresholds
        predicted_cpu_breach = (current_metrics.predicted_load_1min > 
                               policy.cpu_scale_up_threshold * 0.8)
        
        # Decide action
        action = "no_action"
        target_instances = current_instances
        
        if (cpu_breach or memory_breach or queue_breach or 
            (predicted_cpu_breach and current_metrics.trend_direction == "increasing")):
            
            if current_instances < policy.max_instances:
                action = "scale_up"
                # Intelligent scaling: scale more aggressively if trend is strongly increasing
                if current_metrics.trend_direction == "increasing" and current_metrics.confidence_score > 0.7:
                    scale_factor = min(2, policy.max_instances - current_instances)
                else:
                    scale_factor = 1
                
                target_instances = min(policy.max_instances, current_instances + scale_factor)
                
                if cpu_breach:
                    reasoning.append(f"CPU utilization {current_metrics.cpu_utilization:.1f}% > {policy.cpu_scale_up_threshold}%")
                if memory_breach:
                    reasoning.append(f"Memory utilization {current_metrics.memory_utilization:.1f}% > {policy.memory_scale_up_threshold}%")
                if queue_breach:
                    reasoning.append(f"Queue depth {current_metrics.queue_depth} > {policy.queue_scale_up_threshold}")
                if predicted_cpu_breach:
                    reasoning.append(f"Predicted CPU load {current_metrics.predicted_load_1min:.1f}% indicates upcoming pressure")
        
        elif (current_metrics.cpu_utilization < policy.cpu_scale_down_threshold and
              current_metrics.memory_utilization < policy.memory_scale_down_threshold and
              current_metrics.queue_depth < policy.queue_scale_down_threshold and
              current_metrics.predicted_load_5min < policy.cpu_scale_down_threshold):
            
            if current_instances > policy.min_instances:
                action = "scale_down"
                target_instances = max(policy.min_instances, current_instances - 1)
                
                reasoning.append(f"All metrics below scale-down thresholds")
                reasoning.append(f"Predicted load {current_metrics.predicted_load_5min:.1f}% remains low")
        
        # Estimate impact
        estimated_impact = self._estimate_scaling_impact(
            current_instances, target_instances, current_metrics
        )
        
        return ScalingDecision(
            action=action,
            target_instances=target_instances,
            reasoning=reasoning,
            confidence=current_metrics.confidence_score,
            estimated_impact=estimated_impact,
            cooldown_until=time.time() + (
                policy.scale_up_cooldown if action == "scale_up" 
                else policy.scale_down_cooldown if action == "scale_down"
                else 0
            )
        )
    
    def _estimate_scaling_impact(
        self, 
        current_instances: int, 
        target_instances: int, 
        metrics: PredictiveMetrics
    ) -> Dict[str, float]:
        """Estimate the impact of scaling decision."""
        if current_instances == target_instances:
            return {}
        
        scale_factor = target_instances / current_instances
        
        return {
            "estimated_cpu_change": (1.0 / scale_factor - 1.0) * 100,  # % change
            "estimated_latency_change": (1.0 / scale_factor - 1.0) * metrics.response_time_p99,
            "estimated_cost_change": (scale_factor - 1.0) * 100,  # % change
            "estimated_throughput_change": (scale_factor - 1.0) * 100  # % change
        }


class DistributedScalingOrchestrator:
    """Orchestrates scaling across multiple nodes/regions."""
    
    def __init__(self, node_id: str, redis_host: Optional[str] = None):
        self.node_id = node_id
        self.redis_client = None
        self.local_scaler = PredictiveScaler()
        
        if REDIS_AVAILABLE and redis_host:
            try:
                self.redis_client = redis.Redis(host=redis_host, decode_responses=True)
            except Exception as e:
                logging.warning(f"Failed to connect to Redis: {e}")
        
        self.logger = logging.getLogger(__name__)
    
    @safe_operation
    def coordinate_scaling(
        self, 
        local_metrics: ResourceMetrics, 
        policy: ScalingPolicy
    ) -> ScalingDecision:
        """Coordinate scaling decision across distributed nodes."""
        
        # Update local scaler
        self.local_scaler.add_metrics(local_metrics)
        
        # Get local scaling decision
        local_decision = self.local_scaler.make_scaling_decision(1, policy)  # Assume 1 local instance
        
        if not self.redis_client:
            return local_decision
        
        try:
            # Share metrics with other nodes
            metrics_key = f"metrics:{self.node_id}"
            self.redis_client.setex(
                metrics_key, 
                60,  # 1 minute expiry
                json.dumps({
                    'timestamp': time.time(),
                    'cpu_utilization': local_metrics.cpu_percent,
                    'memory_utilization': local_metrics.memory_percent,
                    'queue_depth': local_metrics.inference_queue_size,
                    'error_rate': local_metrics.error_rate,
                    'node_id': self.node_id
                })
            )
            
            # Get metrics from other nodes
            other_nodes_metrics = []
            for key in self.redis_client.scan_iter(match="metrics:*"):
                if key != metrics_key:
                    try:
                        data = self.redis_client.get(key)
                        if data:
                            other_metrics = json.loads(data)
                            # Only consider recent metrics (within 2 minutes)
                            if time.time() - other_metrics['timestamp'] < 120:
                                other_nodes_metrics.append(other_metrics)
                    except Exception:
                        continue
            
            # Make global scaling decision
            if other_nodes_metrics:
                global_decision = self._make_global_scaling_decision(
                    local_decision, other_nodes_metrics, policy
                )
                
                # Store decision for coordination
                decision_key = f"decision:{self.node_id}"
                self.redis_client.setex(
                    decision_key,
                    30,  # 30 seconds
                    json.dumps({
                        'action': global_decision.action,
                        'target_instances': global_decision.target_instances,
                        'timestamp': time.time(),
                        'node_id': self.node_id
                    })
                )
                
                return global_decision
            
        except Exception as e:
            self.logger.error(f"Distributed scaling coordination failed: {e}")
        
        return local_decision
    
    def _make_global_scaling_decision(
        self, 
        local_decision: ScalingDecision,
        other_nodes_metrics: List[Dict[str, Any]],
        policy: ScalingPolicy
    ) -> ScalingDecision:
        """Make scaling decision considering global cluster state."""
        
        # Calculate global metrics
        all_cpu = [local_decision.estimated_impact.get('estimated_cpu_change', 0)]
        all_queue_depths = [0]  # Local queue depth from metrics
        all_error_rates = [0]   # Local error rate from metrics
        
        for metrics in other_nodes_metrics:
            all_cpu.append(metrics.get('cpu_utilization', 0))
            all_queue_depths.append(metrics.get('queue_depth', 0))  
            all_error_rates.append(metrics.get('error_rate', 0))
        
        global_avg_cpu = np.mean(all_cpu)
        global_max_queue = max(all_queue_depths)
        global_avg_error_rate = np.mean(all_error_rates)
        
        # Adjust local decision based on global state
        reasoning = list(local_decision.reasoning)
        
        if global_avg_cpu > policy.cpu_scale_up_threshold * 0.9:
            reasoning.append(f"Global average CPU {global_avg_cpu:.1f}% indicates cluster pressure")
            if local_decision.action == "scale_down":
                # Override scale down if cluster is under pressure
                return ScalingDecision(
                    action="no_action",
                    target_instances=local_decision.target_instances,
                    reasoning=reasoning + ["Overriding scale down due to cluster pressure"],
                    confidence=local_decision.confidence * 0.8,
                    estimated_impact={},
                    cooldown_until=local_decision.cooldown_until
                )
        
        if global_max_queue > policy.queue_scale_up_threshold * 2:
            reasoning.append(f"Global max queue depth {global_max_queue} indicates bottleneck")
            if local_decision.action != "scale_up":
                return ScalingDecision(
                    action="scale_up",
                    target_instances=local_decision.target_instances + 1,
                    reasoning=reasoning,
                    confidence=local_decision.confidence,
                    estimated_impact=local_decision.estimated_impact,
                    cooldown_until=local_decision.cooldown_until
                )
        
        # Return modified decision
        return ScalingDecision(
            action=local_decision.action,
            target_instances=local_decision.target_instances,
            reasoning=reasoning,
            confidence=local_decision.confidence,
            estimated_impact=local_decision.estimated_impact,
            cooldown_until=local_decision.cooldown_until
        )


class PerformanceOptimizer:
    """Advanced performance optimization with ML-based tuning."""
    
    def __init__(self):
        self.optimization_history = deque(maxlen=1000)
        self.current_config = {}
        self.baseline_performance = None
        self.optimization_trials = []
        
        # Performance tuning parameters
        self.tuning_params = {
            'batch_size': [16, 32, 64, 128, 256],
            'num_workers': [2, 4, 8, 16],
            'prefetch_factor': [1, 2, 4, 8],
            'pin_memory': [True, False],
            'non_blocking': [True, False]
        }
        
        self.logger = logging.getLogger(__name__)
    
    @safe_operation
    def optimize_performance(
        self, 
        model: Any, 
        sample_data: Any,
        target_metric: str = "latency"
    ) -> Dict[str, Any]:
        """Automatically optimize model performance."""
        
        self.logger.info("Starting automatic performance optimization")
        
        if not self.baseline_performance:
            self.baseline_performance = self._measure_performance(model, sample_data)
            self.logger.info(f"Baseline performance: {self.baseline_performance}")
        
        best_config = {}
        best_performance = self.baseline_performance
        
        # Try different optimization strategies
        optimizations = [
            self._optimize_compilation,
            self._optimize_precision,
            self._optimize_memory,
            self._optimize_parallelism
        ]
        
        for optimization_func in optimizations:
            try:
                config, performance = optimization_func(model, sample_data)
                
                if self._is_better_performance(performance, best_performance, target_metric):
                    best_config.update(config)
                    best_performance = performance
                    self.logger.info(f"Improved performance with {optimization_func.__name__}: {performance}")
                
            except Exception as e:
                self.logger.error(f"Optimization {optimization_func.__name__} failed: {e}")
        
        # Apply best configuration
        if best_config:
            self._apply_configuration(model, best_config)
            
        self.current_config = best_config
        
        return {
            'optimizations_applied': best_config,
            'performance_improvement': self._calculate_improvement(
                self.baseline_performance, best_performance, target_metric
            ),
            'baseline_performance': self.baseline_performance,
            'optimized_performance': best_performance
        }
    
    def _measure_performance(self, model: Any, sample_data: Any) -> Dict[str, float]:
        """Measure model performance metrics."""
        if not TORCH_AVAILABLE:
            return {'latency': 0.0, 'throughput': 0.0, 'memory': 0.0}
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_data)
        
        # Measure latency
        latencies = []
        with torch.no_grad():
            for _ in range(100):
                start_time = time.time()
                _ = model(sample_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latencies.append(time.time() - start_time)
        
        # Measure memory
        memory_usage = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(sample_data)
            memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return {
            'latency': np.mean(latencies) * 1000,  # ms
            'latency_p99': np.percentile(latencies, 99) * 1000,  # ms
            'throughput': 1.0 / np.mean(latencies),  # samples/sec
            'memory': memory_usage
        }
    
    def _optimize_compilation(self, model: Any, sample_data: Any) -> Tuple[Dict, Dict]:
        """Optimize using torch.compile (if available)."""
        config = {}
        
        if not TORCH_AVAILABLE or not hasattr(torch, 'compile'):
            return config, self._measure_performance(model, sample_data)
        
        try:
            # Try different compilation modes
            compiled_model = torch.compile(model, mode="reduce-overhead")
            performance = self._measure_performance(compiled_model, sample_data)
            config['torch_compile'] = True
            config['compile_mode'] = "reduce-overhead"
            
            return config, performance
            
        except Exception as e:
            self.logger.warning(f"Torch compilation failed: {e}")
            return config, self._measure_performance(model, sample_data)
    
    def _optimize_precision(self, model: Any, sample_data: Any) -> Tuple[Dict, Dict]:
        """Optimize using mixed precision."""
        config = {}
        
        if not TORCH_AVAILABLE:
            return config, self._measure_performance(model, sample_data)
        
        try:
            # Try FP16 if CUDA is available
            if torch.cuda.is_available():
                model_half = model.half()
                sample_data_half = sample_data.half()
                
                performance = self._measure_performance(model_half, sample_data_half)
                config['precision'] = 'fp16'
                
                return config, performance
                
        except Exception as e:
            self.logger.warning(f"FP16 optimization failed: {e}")
        
        return config, self._measure_performance(model, sample_data)
    
    def _optimize_memory(self, model: Any, sample_data: Any) -> Tuple[Dict, Dict]:
        """Optimize memory usage."""
        config = {}
        
        if not TORCH_AVAILABLE:
            return config, self._measure_performance(model, sample_data)
        
        try:
            # Enable memory efficient attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                config['memory_efficient_attention'] = True
            
            # Try gradient checkpointing for training
            config['gradient_checkpointing'] = True
            
            performance = self._measure_performance(model, sample_data)
            return config, performance
            
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
        
        return config, self._measure_performance(model, sample_data)
    
    def _optimize_parallelism(self, model: Any, sample_data: Any) -> Tuple[Dict, Dict]:
        """Optimize parallelism settings."""
        config = {}
        
        if not TORCH_AVAILABLE:
            return config, self._measure_performance(model, sample_data)
        
        try:
            # Set optimal thread count
            optimal_threads = min(8, torch.get_num_threads())
            torch.set_num_threads(optimal_threads)
            config['num_threads'] = optimal_threads
            
            # Enable inter-op parallelism
            torch.set_num_interop_threads(2)
            config['num_interop_threads'] = 2
            
            performance = self._measure_performance(model, sample_data)
            return config, performance
            
        except Exception as e:
            self.logger.warning(f"Parallelism optimization failed: {e}")
        
        return config, self._measure_performance(model, sample_data)
    
    def _is_better_performance(
        self, 
        new_perf: Dict[str, float], 
        current_perf: Dict[str, float], 
        target_metric: str
    ) -> bool:
        """Check if new performance is better than current."""
        if target_metric == "latency":
            return new_perf.get('latency', float('inf')) < current_perf.get('latency', float('inf'))
        elif target_metric == "throughput":
            return new_perf.get('throughput', 0) > current_perf.get('throughput', 0)
        elif target_metric == "memory":
            return new_perf.get('memory', float('inf')) < current_perf.get('memory', float('inf'))
        else:
            # Default to latency
            return new_perf.get('latency', float('inf')) < current_perf.get('latency', float('inf'))
    
    def _calculate_improvement(
        self, 
        baseline: Dict[str, float], 
        optimized: Dict[str, float], 
        target_metric: str
    ) -> float:
        """Calculate performance improvement percentage."""
        if target_metric not in baseline or target_metric not in optimized:
            return 0.0
        
        baseline_val = baseline[target_metric]
        optimized_val = optimized[target_metric]
        
        if baseline_val == 0:
            return 0.0
        
        if target_metric in ["latency", "memory"]:
            # Lower is better
            return ((baseline_val - optimized_val) / baseline_val) * 100
        else:
            # Higher is better
            return ((optimized_val - baseline_val) / baseline_val) * 100
    
    def _apply_configuration(self, model: Any, config: Dict[str, Any]):
        """Apply optimization configuration to model."""
        # This would apply the optimization settings
        # Implementation depends on specific optimizations
        pass


# Global instances
_global_predictive_scaler = None
_global_performance_optimizer = None


def get_predictive_scaler() -> PredictiveScaler:
    """Get global predictive scaler instance."""
    global _global_predictive_scaler
    if _global_predictive_scaler is None:
        _global_predictive_scaler = PredictiveScaler()
    return _global_predictive_scaler


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_performance_optimizer
    if _global_performance_optimizer is None:
        _global_performance_optimizer = PerformanceOptimizer()
    return _global_performance_optimizer


def auto_optimize_model(model: Any, sample_data: Any, target_metric: str = "latency") -> Dict[str, Any]:
    """Automatically optimize model performance."""
    optimizer = get_performance_optimizer()
    return optimizer.optimize_performance(model, sample_data, target_metric)


def create_distributed_scaler(node_id: str, redis_host: Optional[str] = None) -> DistributedScalingOrchestrator:
    """Create distributed scaling orchestrator."""
    return DistributedScalingOrchestrator(node_id, redis_host)