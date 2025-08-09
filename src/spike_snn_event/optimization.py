"""
Memory and performance optimization for spike-snn-event-vision-kit.

Provides intelligent memory management, garbage collection optimization,
and performance tuning for high-throughput neuromorphic processing.
"""

import gc
import time
import threading
import psutil
import logging
import weakref
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import numpy as np
from abc import ABC, abstractmethod

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .monitoring import get_metrics_collector


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_mb: float = 0.0
    used_mb: float = 0.0
    free_mb: float = 0.0
    percent: float = 0.0
    gpu_total_mb: float = 0.0
    gpu_used_mb: float = 0.0
    gpu_free_mb: float = 0.0
    gpu_percent: float = 0.0
    gc_collections: int = 0
    gc_freed_objects: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationConfig:
    """Configuration for memory optimization."""
    enable_aggressive_gc: bool = True
    gc_frequency_seconds: float = 30.0
    memory_threshold_percent: float = 85.0
    gpu_memory_threshold_percent: float = 90.0
    enable_memory_profiling: bool = True
    enable_object_tracking: bool = True
    max_cached_items: int = 1000
    cache_ttl_seconds: float = 300.0
    enable_tensor_cleanup: bool = True
    enable_memory_mapping: bool = False


class MemoryTracker:
    """Advanced memory usage tracking and optimization."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Memory tracking
        self.memory_history = deque(maxlen=100)
        self.large_objects = weakref.WeakSet()
        self.object_registry = defaultdict(int)
        
        # GC tracking
        self.gc_stats = {
            'collections': 0,
            'freed_objects': 0,
            'last_collection': 0.0
        }
        
        # GPU monitoring
        self.gpu_available = self._check_gpu_availability()
        
        # Optimization state
        self.is_monitoring = False
        self.monitor_thread = None
        self.optimization_enabled = True
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return True
        except Exception:
            pass
        return False
        
    def start_monitoring(self):
        """Start memory monitoring and optimization."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Memory monitoring started")
        
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Memory monitoring stopped")
        
    def _monitoring_loop(self):
        """Main memory monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect memory statistics
                stats = self._collect_memory_stats()
                self.memory_history.append(stats)
                
                # Trigger optimization if needed
                if self.optimization_enabled:
                    self._check_optimization_triggers(stats)
                    
                # Sleep until next check
                time.sleep(self.config.gc_frequency_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")
                time.sleep(self.config.gc_frequency_seconds)
                
    def _collect_memory_stats(self) -> MemoryStats:
        """Collect current memory statistics."""
        # System memory
        memory = psutil.virtual_memory()
        
        # GPU memory
        gpu_total_mb = 0.0
        gpu_used_mb = 0.0
        gpu_free_mb = 0.0
        gpu_percent = 0.0
        
        if self.gpu_available:
            try:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_stats()
                    gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                    gpu_used_mb = torch.cuda.memory_allocated() / (1024**2)
                    gpu_free_mb = gpu_total_mb - gpu_used_mb
                    gpu_percent = (gpu_used_mb / gpu_total_mb) * 100 if gpu_total_mb > 0 else 0
            except Exception:
                pass
                
        return MemoryStats(
            total_mb=memory.total / (1024**2),
            used_mb=memory.used / (1024**2),
            free_mb=memory.free / (1024**2),
            percent=memory.percent,
            gpu_total_mb=gpu_total_mb,
            gpu_used_mb=gpu_used_mb,
            gpu_free_mb=gpu_free_mb,
            gpu_percent=gpu_percent,
            gc_collections=self.gc_stats['collections'],
            gc_freed_objects=self.gc_stats['freed_objects']
        )
        
    def _check_optimization_triggers(self, stats: MemoryStats):
        """Check if memory optimization is needed."""
        triggers_fired = []
        
        # System memory threshold
        if stats.percent > self.config.memory_threshold_percent:
            triggers_fired.append('system_memory')
            
        # GPU memory threshold
        if (self.gpu_available and 
            stats.gpu_percent > self.config.gpu_memory_threshold_percent):
            triggers_fired.append('gpu_memory')
            
        # Time-based GC trigger
        current_time = time.time()
        if (current_time - self.gc_stats['last_collection'] > 
            self.config.gc_frequency_seconds):
            triggers_fired.append('time_based')
            
        if triggers_fired:
            self._trigger_optimization(triggers_fired, stats)
            
    def _trigger_optimization(self, triggers: List[str], stats: MemoryStats):
        """Trigger memory optimization based on triggers."""
        self.logger.info(f"Memory optimization triggered: {triggers}")
        
        freed_memory = 0
        freed_objects = 0
        
        try:
            # Aggressive garbage collection
            if self.config.enable_aggressive_gc:
                freed_objects += self._aggressive_gc()
                
            # GPU memory cleanup
            if 'gpu_memory' in triggers and self.gpu_available:
                freed_memory += self._cleanup_gpu_memory()
                
            # Object cleanup
            if self.config.enable_object_tracking:
                freed_objects += self._cleanup_tracked_objects()
                
            # Cache cleanup
            freed_objects += self._cleanup_caches()
            
            self.logger.info(
                f"Memory optimization completed: "
                f"freed {freed_memory:.1f}MB, {freed_objects} objects"
            )
            
        except Exception as e:
            self.logger.error(f"Error during memory optimization: {e}")
            
    def _aggressive_gc(self) -> int:
        """Perform aggressive garbage collection."""
        initial_objects = len(gc.get_objects())
        
        # Force collection of all generations
        freed_gen0 = gc.collect(0)
        freed_gen1 = gc.collect(1) 
        freed_gen2 = gc.collect(2)
        
        total_freed = freed_gen0 + freed_gen1 + freed_gen2
        
        # Update statistics
        self.gc_stats['collections'] += 3
        self.gc_stats['freed_objects'] += total_freed
        self.gc_stats['last_collection'] = time.time()
        
        final_objects = len(gc.get_objects())
        objects_freed = initial_objects - final_objects
        
        self.logger.debug(f"GC freed {total_freed} references, {objects_freed} objects")
        
        return objects_freed
        
    def _cleanup_gpu_memory(self) -> float:
        """Cleanup GPU memory."""
        if not (TORCH_AVAILABLE and torch.cuda.is_available()):
            return 0.0
            
        try:
            initial_memory = torch.cuda.memory_allocated() / (1024**2)
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Force synchronization
            torch.cuda.synchronize()
            
            final_memory = torch.cuda.memory_allocated() / (1024**2)
            freed_memory = initial_memory - final_memory
            
            self.logger.debug(f"GPU memory cleanup freed {freed_memory:.1f}MB")
            
            return freed_memory
            
        except Exception as e:
            self.logger.error(f"Error cleaning GPU memory: {e}")
            return 0.0
            
    def _cleanup_tracked_objects(self) -> int:
        """Cleanup tracked large objects."""
        initial_count = len(self.large_objects)
        
        # Objects will be automatically removed by WeakSet when they're garbage collected
        # We just trigger GC to help the process
        gc.collect()
        
        final_count = len(self.large_objects)
        freed_count = initial_count - final_count
        
        if freed_count > 0:
            self.logger.debug(f"Tracked object cleanup freed {freed_count} objects")
            
        return freed_count
        
    def _cleanup_caches(self) -> int:
        """Cleanup internal caches."""
        # This would cleanup application-specific caches
        # Placeholder implementation
        return 0
        
    def register_large_object(self, obj: Any):
        """Register a large object for tracking."""
        if self.config.enable_object_tracking:
            self.large_objects.add(obj)
            self.object_registry[type(obj).__name__] += 1
            
    def get_memory_stats(self) -> Optional[MemoryStats]:
        """Get current memory statistics."""
        if self.memory_history:
            return self.memory_history[-1]
        return None
        
    def get_memory_trend(self, window_size: int = 10) -> Dict[str, float]:
        """Get memory usage trend over window."""
        if len(self.memory_history) < 2:
            return {}
            
        recent_stats = list(self.memory_history)[-window_size:]
        
        if len(recent_stats) < 2:
            return {}
            
        # Calculate trends
        memory_values = [s.percent for s in recent_stats]
        gpu_values = [s.gpu_percent for s in recent_stats if s.gpu_percent > 0]
        
        memory_trend = (memory_values[-1] - memory_values[0]) / len(memory_values)
        gpu_trend = ((gpu_values[-1] - gpu_values[0]) / len(gpu_values) 
                    if len(gpu_values) >= 2 else 0.0)
                    
        return {
            'memory_trend_percent_per_sample': memory_trend,
            'gpu_trend_percent_per_sample': gpu_trend,
            'samples': len(recent_stats)
        }
        
    def force_optimization(self):
        """Force immediate memory optimization."""
        if not self.optimization_enabled:
            return
            
        current_stats = self._collect_memory_stats()
        self._trigger_optimization(['manual'], current_stats)
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get memory optimization statistics."""
        return {
            'gc_collections': self.gc_stats['collections'],
            'gc_freed_objects': self.gc_stats['freed_objects'],
            'last_collection': self.gc_stats['last_collection'],
            'tracked_objects': len(self.large_objects),
            'object_types': dict(self.object_registry),
            'monitoring_enabled': self.is_monitoring,
            'optimization_enabled': self.optimization_enabled
        }


@contextmanager
def memory_profiler(tracker: MemoryTracker, operation_name: str):
    """Context manager for memory profiling operations."""
    if not tracker.config.enable_memory_profiling:
        yield
        return
        
    # Get initial memory
    initial_stats = tracker._collect_memory_stats()
    start_time = time.time()
    
    try:
        yield
    finally:
        # Get final memory
        final_stats = tracker._collect_memory_stats()
        duration = time.time() - start_time
        
        memory_delta = final_stats.used_mb - initial_stats.used_mb
        gpu_delta = final_stats.gpu_used_mb - initial_stats.gpu_used_mb
        
        tracker.logger.debug(
            f"Memory profile '{operation_name}': "
            f"RAM: {memory_delta:+.1f}MB, GPU: {gpu_delta:+.1f}MB, "
            f"Duration: {duration:.3f}s"
        )


class TensorOptimizer:
    """PyTorch tensor memory optimization utilities."""
    
    def __init__(self, memory_tracker: Optional[MemoryTracker] = None):
        self.memory_tracker = memory_tracker
        self.logger = logging.getLogger(__name__)
        
    @staticmethod
    def optimize_tensor_memory(tensor: 'torch.Tensor') -> 'torch.Tensor':
        """Optimize tensor memory usage."""
        if not TORCH_AVAILABLE:
            return tensor
            
        try:
            # Use contiguous memory layout
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
                
            # Pin memory for faster GPU transfers if on CPU
            if tensor.device.type == 'cpu' and not tensor.is_pinned():
                tensor = tensor.pin_memory()
                
            return tensor
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Tensor optimization failed: {e}")
            return tensor
            
    @contextmanager
    def efficient_inference(self, model: Any):
        """Context manager for memory-efficient inference."""
        if not TORCH_AVAILABLE:
            yield model
            return
            
        try:
            # Set to evaluation mode and disable gradients
            was_training = model.training
            model.eval()
            
            with torch.no_grad():
                yield model
                
        finally:
            # Restore training mode
            if was_training:
                model.train()
                
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def cleanup_model_memory(self, model: Any):
        """Clean up model memory after inference."""
        if not TORCH_AVAILABLE:
            return
            
        try:
            # Delete intermediate activations if any
            if hasattr(model, '_intermediate_outputs'):
                del model._intermediate_outputs
                
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            # Trigger garbage collection
            gc.collect()
            
        except Exception as e:
            self.logger.warning(f"Model memory cleanup failed: {e}")


# Global optimization instances
_global_memory_tracker = None
_global_tensor_optimizer = None


def get_memory_tracker() -> MemoryTracker:
    """Get global memory tracker instance."""
    global _global_memory_tracker
    if _global_memory_tracker is None:
        _global_memory_tracker = MemoryTracker()
    return _global_memory_tracker


def get_tensor_optimizer() -> TensorOptimizer:
    """Get global tensor optimizer instance."""
    global _global_tensor_optimizer
    if _global_tensor_optimizer is None:
        _global_tensor_optimizer = TensorOptimizer(get_memory_tracker())
    return _global_tensor_optimizer


def optimize_memory_usage():
    """Convenient function to force memory optimization."""
    tracker = get_memory_tracker()
    tracker.force_optimization()


# Decorators for automatic optimization
def memory_optimized(func):
    """Decorator to automatically optimize memory before/after function execution."""
    def wrapper(*args, **kwargs):
        tracker = get_memory_tracker()
        
        with memory_profiler(tracker, func.__name__):
            # Pre-optimization
            if tracker.optimization_enabled:
                initial_stats = tracker._collect_memory_stats()
                if initial_stats.percent > 80.0:  # High memory usage
                    tracker._aggressive_gc()
                    
            # Execute function
            result = func(*args, **kwargs)
            
            # Post-optimization for GPU functions
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return result
            
    return wrapper


def gpu_memory_managed(func):
    """Decorator for GPU memory management."""
    def wrapper(*args, **kwargs):
        if not (TORCH_AVAILABLE and torch.cuda.is_available()):
            return func(*args, **kwargs)
            
        # Pre-execution cleanup
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Post-execution cleanup
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            
            # Log significant memory increases
            memory_delta = (final_memory - initial_memory) / (1024**2)  # MB
            if memory_delta > 100:  # More than 100MB increase
                logging.getLogger(__name__).debug(
                    f"Function '{func.__name__}' increased GPU memory by {memory_delta:.1f}MB"
                )
                
    return wrapper