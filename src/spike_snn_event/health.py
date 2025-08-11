"""
Comprehensive health check system for spike-snn-event-vision-kit.

Provides unified health monitoring, diagnostic tools, and system status reporting
for production environments.
"""

import time
import json
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

import torch
import numpy as np

from .monitoring import get_metrics_collector, get_health_checker
from .validation import ValidationError, HardwareError, CircuitBreakerError, DataIntegrityError
from .security import get_input_sanitizer
from .security_enhancements import get_adversarial_defense, get_memory_safety_manager


@dataclass
class ComponentHealth:
    """Health status for individual system component."""
    name: str
    status: str  # "healthy", "warning", "critical", "unknown"
    message: str
    details: Dict[str, Any]
    last_check: float
    response_time_ms: Optional[float] = None


class SystemHealthChecker:
    """Comprehensive system health checker."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = get_metrics_collector()
        self.health_checker = get_health_checker()
        self.input_sanitizer = get_input_sanitizer()
        
        # Component checkers
        self.component_checkers = {
            'pytorch': self._check_pytorch,
            'cuda': self._check_cuda,
            'memory': self._check_memory,
            'disk': self._check_disk,
            'dependencies': self._check_dependencies,
            'models': self._check_models,
            'data_pipeline': self._check_data_pipeline,
        }
        
    def check_all_components(self) -> Dict[str, ComponentHealth]:
        """Check health of all system components."""
        results = {}
        
        for component_name, checker_func in self.component_checkers.items():
            try:
                start_time = time.time()
                result = checker_func()
                response_time = (time.time() - start_time) * 1000
                
                results[component_name] = ComponentHealth(
                    name=component_name,
                    status=result['status'],
                    message=result['message'],
                    details=result.get('details', {}),
                    last_check=time.time(),
                    response_time_ms=response_time
                )
                
            except Exception as e:
                self.logger.error(f"Health check failed for {component_name}: {e}")
                results[component_name] = ComponentHealth(
                    name=component_name,
                    status="critical",
                    message=f"Health check failed: {e}",
                    details={"error": str(e)},
                    last_check=time.time()
                )
                
        return results
        
    def _check_pytorch(self) -> Dict[str, Any]:
        """Check PyTorch installation and functionality."""
        try:
            # Check PyTorch version
            version = torch.__version__
            
            # Test tensor operations
            test_tensor = torch.randn(10, 10)
            result = torch.mm(test_tensor, test_tensor.T)
            
            # Check if CUDA is available and working
            cuda_available = torch.cuda.is_available()
            cuda_devices = torch.cuda.device_count() if cuda_available else 0
            
            return {
                'status': 'healthy',
                'message': f'PyTorch {version} working correctly',
                'details': {
                    'version': version,
                    'cuda_available': cuda_available,
                    'cuda_devices': cuda_devices,
                    'tensor_ops_working': True
                }
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'PyTorch check failed: {e}',
                'details': {'error': str(e)}
            }
            
    def _check_cuda(self) -> Dict[str, Any]:
        """Check CUDA availability and functionality."""
        try:
            if not torch.cuda.is_available():
                return {
                    'status': 'warning',
                    'message': 'CUDA not available - running on CPU only',
                    'details': {'cuda_available': False}
                }
                
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            # Test CUDA operations
            test_tensor = torch.randn(100, 100, device='cuda')
            result = torch.mm(test_tensor, test_tensor.T)
            
            # Check memory
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_cached = torch.cuda.memory_reserved() / 1024**2  # MB
            total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**2
            
            return {
                'status': 'healthy',
                'message': f'CUDA working on {device_name}',
                'details': {
                    'cuda_available': True,
                    'device_count': device_count,
                    'current_device': current_device,
                    'device_name': device_name,
                    'memory_allocated_mb': memory_allocated,
                    'memory_cached_mb': memory_cached,
                    'total_memory_mb': total_memory,
                    'cuda_ops_working': True
                }
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'CUDA check failed: {e}',
                'details': {'error': str(e)}
            }
            
    def _check_memory(self) -> Dict[str, Any]:
        """Check system memory usage."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            memory_usage_percent = memory.percent
            available_gb = memory.available / 1024**3
            
            if memory_usage_percent > 90:
                status = 'critical'
                message = f'Critical memory usage: {memory_usage_percent:.1f}%'
            elif memory_usage_percent > 80:
                status = 'warning'
                message = f'High memory usage: {memory_usage_percent:.1f}%'
            else:
                status = 'healthy'
                message = f'Memory usage normal: {memory_usage_percent:.1f}%'
                
            return {
                'status': status,
                'message': message,
                'details': {
                    'total_gb': memory.total / 1024**3,
                    'available_gb': available_gb,
                    'used_percent': memory_usage_percent,
                    'swap_used_percent': swap.percent
                }
            }
            
        except ImportError:
            return {
                'status': 'unknown',
                'message': 'psutil not available for memory monitoring',
                'details': {}
            }
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Memory check failed: {e}',
                'details': {'error': str(e)}
            }
            
    def _check_disk(self) -> Dict[str, Any]:
        """Check disk space availability."""
        try:
            import shutil
            
            current_dir = Path.cwd()
            total, used, free = shutil.disk_usage(current_dir)
            
            free_gb = free / 1024**3
            usage_percent = (used / total) * 100
            
            if free_gb < 1:  # Less than 1GB free
                status = 'critical'
                message = f'Critical disk space: {free_gb:.1f}GB free'
            elif free_gb < 5:  # Less than 5GB free
                status = 'warning'
                message = f'Low disk space: {free_gb:.1f}GB free'
            else:
                status = 'healthy'
                message = f'Disk space adequate: {free_gb:.1f}GB free'
                
            return {
                'status': status,
                'message': message,
                'details': {
                    'total_gb': total / 1024**3,
                    'used_gb': used / 1024**3,
                    'free_gb': free_gb,
                    'usage_percent': usage_percent
                }
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Disk check failed: {e}',
                'details': {'error': str(e)}
            }
            
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check critical dependencies."""
        required_packages = {
            'numpy': 'numpy',
            'torch': 'torch',
            'torchvision': 'torchvision',
            'opencv': 'cv2',
            'matplotlib': 'matplotlib',
            'tqdm': 'tqdm',
            'yaml': 'yaml',
        }
        
        missing_packages = []
        package_versions = {}
        
        for name, import_name in required_packages.items():
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'unknown')
                package_versions[name] = version
            except ImportError:
                missing_packages.append(name)
                
        if missing_packages:
            return {
                'status': 'critical',
                'message': f'Missing required packages: {", ".join(missing_packages)}',
                'details': {
                    'missing_packages': missing_packages,
                    'available_packages': package_versions
                }
            }
        else:
            return {
                'status': 'healthy',
                'message': 'All required dependencies available',
                'details': {'package_versions': package_versions}
            }
            
    def _check_models(self) -> Dict[str, Any]:
        """Check model loading and basic functionality."""
        try:
            from .models import CustomSNN, SpikingYOLO
            
            # Test model creation
            test_model = CustomSNN(
                input_size=(32, 32),
                hidden_channels=[16, 32],
                output_classes=2
            )
            
            # Test forward pass
            dummy_input = torch.randn(1, 2, 32, 32, 5)
            with torch.no_grad():
                output = test_model(dummy_input)
                
            param_count = sum(p.numel() for p in test_model.parameters())
            
            return {
                'status': 'healthy',
                'message': 'Model creation and inference working',
                'details': {
                    'test_model_parameters': param_count,
                    'test_input_shape': list(dummy_input.shape),
                    'test_output_shape': list(output.shape),
                    'inference_successful': True
                }
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Model check failed: {e}',
                'details': {'error': str(e)}
            }
            
    def _check_data_pipeline(self) -> Dict[str, Any]:
        """Check data processing pipeline."""
        try:
            from .core import DVSCamera, EventPreprocessor
            
            # Test camera creation
            camera = DVSCamera(sensor_type="DVS128")
            
            # Test event generation
            test_events = camera._generate_synthetic_events(100)
            
            # Test event processing
            if len(test_events) > 0:
                filtered_events = camera._apply_noise_filter(test_events)
                
            return {
                'status': 'healthy',
                'message': 'Data pipeline working correctly',
                'details': {
                    'camera_created': True,
                    'event_generation_working': len(test_events) > 0,
                    'event_filtering_working': True,
                    'test_events_generated': len(test_events),
                    'events_after_filtering': len(filtered_events) if len(test_events) > 0 else 0
                }
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Data pipeline check failed: {e}',
                'details': {'error': str(e)}
            }
            
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary."""
        component_health = self.check_all_components()
        overall_metrics = self.health_checker.check_health()
        
        # Determine overall system status
        component_statuses = [comp.status for comp in component_health.values()]
        if 'critical' in component_statuses:
            overall_status = 'critical'
        elif 'warning' in component_statuses:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'
            
        # Count statuses
        status_counts = {
            'healthy': sum(1 for s in component_statuses if s == 'healthy'),
            'warning': sum(1 for s in component_statuses if s == 'warning'),
            'critical': sum(1 for s in component_statuses if s == 'critical'),
            'unknown': sum(1 for s in component_statuses if s == 'unknown')
        }
        
        return {
            'timestamp': time.time(),
            'overall_status': overall_status,
            'component_count': len(component_health),
            'status_counts': status_counts,
            'components': {name: asdict(comp) for name, comp in component_health.items()},
            'system_metrics': asdict(overall_metrics),
            'recommendations': self._generate_recommendations(component_health, overall_metrics)
        }
        
    def _generate_recommendations(
        self, 
        component_health: Dict[str, ComponentHealth],
        overall_metrics
    ) -> List[str]:
        """Generate system recommendations based on health status."""
        recommendations = []
        
        # Check for critical components
        critical_components = [name for name, comp in component_health.items() if comp.status == 'critical']
        if critical_components:
            recommendations.append(f"Address critical issues in: {', '.join(critical_components)}")
            
        # Memory recommendations
        memory_comp = component_health.get('memory')
        if memory_comp and 'used_percent' in memory_comp.details:
            if memory_comp.details['used_percent'] > 85:
                recommendations.append("Consider restarting services to free memory")
                
        # CUDA recommendations
        cuda_comp = component_health.get('cuda')
        if cuda_comp and cuda_comp.status == 'warning':
            recommendations.append("Enable CUDA for better performance")
            
        # Disk space recommendations
        disk_comp = component_health.get('disk')
        if disk_comp and 'free_gb' in disk_comp.details:
            if disk_comp.details['free_gb'] < 5:
                recommendations.append("Clean up disk space or add storage")
                
        # Model performance recommendations
        if hasattr(overall_metrics, 'inference_latency_ms'):
            if overall_metrics.inference_latency_ms > 100:
                recommendations.append("Consider model optimization or GPU acceleration")
                
        return recommendations
        
    def export_health_report(self, filepath: str):
        """Export comprehensive health report to file."""
        health_summary = self.get_system_summary()
        
        with open(filepath, 'w') as f:
            json.dump(health_summary, f, indent=2, default=str)
            
        self.logger.info(f"Health report exported to {filepath}")


# Global health checker instance
_global_health_checker = None


def get_system_health_checker() -> SystemHealthChecker:
    """Get global system health checker instance."""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = SystemHealthChecker()
    return _global_health_checker


def quick_health_check() -> str:
    """Perform quick health check and return status."""
    checker = get_system_health_checker()
    summary = checker.get_system_summary()
    return summary['overall_status']


def detailed_health_check() -> Dict[str, Any]:
    """Perform detailed health check and return full report."""
    checker = get_system_health_checker()
    return checker.get_system_summary()


def export_health_report(filepath: str = "health_report.json"):
    """Export health report to file."""
    checker = get_system_health_checker()
    checker.export_health_report(filepath)
    return filepath


class ResilientOperationManager:
    """Manager for resilient operations with graceful degradation and recovery."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.operation_history = {}
        self.degradation_active = False
        self.recovery_attempts = {}
        
        # Circuit breakers for different operation types
        from .validation import CircuitBreaker
        self.circuit_breakers = {
            'model_inference': CircuitBreaker(failure_threshold=3, recovery_timeout=30.0),
            'event_processing': CircuitBreaker(failure_threshold=5, recovery_timeout=10.0),
            'data_loading': CircuitBreaker(failure_threshold=3, recovery_timeout=60.0),
            'hardware_access': CircuitBreaker(failure_threshold=2, recovery_timeout=120.0)
        }
        
    def execute_with_resilience(self, operation_name: str, operation_func: callable, 
                              *args, fallback_func: callable = None, **kwargs):
        """Execute operation with resilience patterns."""
        if operation_name not in self.circuit_breakers:
            operation_name = 'default'
            
        circuit_breaker = self.circuit_breakers.get(operation_name, self.circuit_breakers['event_processing'])
        
        try:
            # Execute with circuit breaker protection
            result = circuit_breaker.call(operation_func, *args, **kwargs)
            self._record_successful_operation(operation_name)
            return result
            
        except CircuitBreakerError as e:
            self.logger.error(f"Circuit breaker open for {operation_name}: {e}")
            if fallback_func:
                self.logger.info(f"Attempting fallback for {operation_name}")
                return self._execute_fallback(operation_name, fallback_func, *args, **kwargs)
            else:
                raise
                
        except Exception as e:
            self.logger.error(f"Operation {operation_name} failed: {e}")
            self._record_failed_operation(operation_name, str(e))
            
            # Try recovery if available
            recovery_result = self._attempt_recovery(operation_name, e)
            if recovery_result['recovered']:
                self.logger.info(f"Recovery successful for {operation_name}")
                return self.execute_with_resilience(operation_name, operation_func, *args, **kwargs)
            
            # Use fallback if recovery failed
            if fallback_func:
                return self._execute_fallback(operation_name, fallback_func, *args, **kwargs)
            else:
                raise
                
    def _execute_fallback(self, operation_name: str, fallback_func: callable, *args, **kwargs):
        """Execute fallback operation."""
        try:
            self.logger.info(f"Executing fallback for {operation_name}")
            result = fallback_func(*args, **kwargs)
            self.degradation_active = True
            return result
        except Exception as e:
            self.logger.error(f"Fallback failed for {operation_name}: {e}")
            raise
            
    def _attempt_recovery(self, operation_name: str, error: Exception) -> Dict[str, Any]:
        """Attempt recovery from operation failure."""
        recovery_key = f"{operation_name}_{type(error).__name__}"
        
        if recovery_key not in self.recovery_attempts:
            self.recovery_attempts[recovery_key] = {
                'count': 0,
                'last_attempt': 0,
                'max_attempts': 3
            }
            
        recovery_info = self.recovery_attempts[recovery_key]
        current_time = time.time()
        
        # Check if we should attempt recovery
        if (recovery_info['count'] >= recovery_info['max_attempts'] or
            current_time - recovery_info['last_attempt'] < 60):  # Wait 60s between attempts
            return {'recovered': False, 'reason': 'max_attempts_exceeded'}
            
        recovery_info['count'] += 1
        recovery_info['last_attempt'] = current_time
        
        # Attempt specific recovery strategies
        recovery_result = self._execute_recovery_strategy(operation_name, error)
        
        if recovery_result['recovered']:
            # Reset recovery counter on success
            recovery_info['count'] = 0
            
        return recovery_result
        
    def _execute_recovery_strategy(self, operation_name: str, error: Exception) -> Dict[str, Any]:
        """Execute recovery strategy based on operation and error type."""
        recovery_strategies = {
            'model_inference': self._recover_model_inference,
            'event_processing': self._recover_event_processing,
            'data_loading': self._recover_data_loading,
            'hardware_access': self._recover_hardware_access
        }
        
        strategy_func = recovery_strategies.get(operation_name, self._generic_recovery)
        
        try:
            return strategy_func(error)
        except Exception as recovery_error:
            self.logger.error(f"Recovery strategy failed: {recovery_error}")
            return {'recovered': False, 'reason': str(recovery_error)}
            
    def _recover_model_inference(self, error: Exception) -> Dict[str, Any]:
        """Recovery strategy for model inference failures."""
        try:
            import torch
            if isinstance(error, torch.cuda.OutOfMemoryError):
                # GPU memory recovery
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                return {'recovered': True, 'action': 'gpu_memory_cleanup'}
                
            elif 'CUDA' in str(error):
                # CUDA error recovery
                try:
                    torch.cuda.synchronize()
                    return {'recovered': True, 'action': 'cuda_sync'}
                except:
                    return {'recovered': False, 'reason': 'cuda_sync_failed'}
        except ImportError:
            pass
            
        return {'recovered': False, 'reason': 'unknown_inference_error'}
        
    def _recover_event_processing(self, error: Exception) -> Dict[str, Any]:
        """Recovery strategy for event processing failures."""
        if isinstance(error, (MemoryError, OverflowError)):
            # Memory-related recovery
            memory_manager = get_memory_safety_manager()
            cleanup_result = memory_manager.force_cleanup()
            return {'recovered': True, 'action': 'memory_cleanup', 'details': cleanup_result}
            
        return {'recovered': False, 'reason': 'unknown_processing_error'}
        
    def _recover_data_loading(self, error: Exception) -> Dict[str, Any]:
        """Recovery strategy for data loading failures."""
        if isinstance(error, (FileNotFoundError, PermissionError)):
            # File access recovery
            return {'recovered': False, 'reason': 'file_access_error'}
            
        if isinstance(error, DataIntegrityError):
            # Data integrity recovery
            return {'recovered': False, 'reason': 'data_integrity_error'}
            
        return {'recovered': False, 'reason': 'unknown_data_error'}
        
    def _recover_hardware_access(self, error: Exception) -> Dict[str, Any]:
        """Recovery strategy for hardware access failures."""
        if isinstance(error, HardwareError):
            # Hardware reset attempt
            time.sleep(1.0)  # Brief pause
            return {'recovered': True, 'action': 'hardware_reset_wait'}
            
        return {'recovered': False, 'reason': 'hardware_failure'}
        
    def _generic_recovery(self, error: Exception) -> Dict[str, Any]:
        """Generic recovery strategy."""
        # Basic cleanup and retry
        import gc
        gc.collect()
        return {'recovered': True, 'action': 'generic_cleanup'}
        
    def _record_successful_operation(self, operation_name: str):
        """Record successful operation."""
        if operation_name not in self.operation_history:
            self.operation_history[operation_name] = {'success': 0, 'failure': 0}
        self.operation_history[operation_name]['success'] += 1
        
    def _record_failed_operation(self, operation_name: str, error_msg: str):
        """Record failed operation."""
        if operation_name not in self.operation_history:
            self.operation_history[operation_name] = {'success': 0, 'failure': 0}
        self.operation_history[operation_name]['failure'] += 1
        
    def get_resilience_stats(self) -> Dict[str, Any]:
        """Get resilience statistics."""
        circuit_states = {name: cb.get_state() for name, cb in self.circuit_breakers.items()}
        
        return {
            'degradation_active': self.degradation_active,
            'operation_history': self.operation_history,
            'circuit_breaker_states': circuit_states,
            'recovery_attempts': self.recovery_attempts,
            'total_operations': sum(h['success'] + h['failure'] for h in self.operation_history.values()),
            'success_rate': self._calculate_success_rate()
        }
        
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        total_success = sum(h['success'] for h in self.operation_history.values())
        total_operations = sum(h['success'] + h['failure'] for h in self.operation_history.values())
        return total_success / max(1, total_operations)


class ResourceMonitor:
    """Comprehensive resource monitoring with alerts."""
    
    def __init__(self, alert_thresholds: Dict[str, float] = None):
        self.logger = logging.getLogger(__name__)
        self.alert_thresholds = alert_thresholds or {
            'memory_percent': 85.0,
            'gpu_memory_percent': 90.0,
            'cpu_percent': 80.0,
            'disk_percent': 90.0,
            'inference_latency_ms': 1000.0,
            'event_processing_rate': 100.0  # events per second minimum
        }
        
        self.resource_history = []
        self.alerts_triggered = []
        self.memory_manager = get_memory_safety_manager()
        
    def monitor_resources(self) -> Dict[str, Any]:
        """Monitor all system resources."""
        current_time = time.time()
        
        # System resources
        system_resources = self._monitor_system_resources()
        
        # GPU resources
        gpu_resources = self._monitor_gpu_resources()
        
        # Application-specific metrics
        app_metrics = self._monitor_application_metrics()
        
        # Combine all metrics
        resource_snapshot = {
            'timestamp': current_time,
            'system': system_resources,
            'gpu': gpu_resources,
            'application': app_metrics
        }
        
        # Check for alerts
        alerts = self._check_alert_conditions(resource_snapshot)
        if alerts:
            self.alerts_triggered.extend(alerts)
            
        # Store in history
        self.resource_history.append(resource_snapshot)
        
        # Keep only recent history (last hour)
        cutoff_time = current_time - 3600
        self.resource_history = [r for r in self.resource_history if r['timestamp'] > cutoff_time]
        
        return resource_snapshot
        
    def _monitor_system_resources(self) -> Dict[str, Any]:
        """Monitor system-level resources."""
        try:
            import psutil
            
            # Memory
            memory = psutil.virtual_memory()
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Disk
            disk = psutil.disk_usage('/')
            
            return {
                'memory_total_gb': memory.total / (1024**3),
                'memory_used_gb': memory.used / (1024**3),
                'memory_percent': memory.percent,
                'cpu_percent': cpu_percent,
                'cpu_count': psutil.cpu_count(),
                'disk_total_gb': disk.total / (1024**3),
                'disk_used_gb': disk.used / (1024**3),
                'disk_percent': (disk.used / disk.total) * 100,
                'available': True
            }
        except ImportError:
            return {'available': False, 'error': 'psutil not available'}
        except Exception as e:
            return {'available': False, 'error': str(e)}
            
    def _monitor_gpu_resources(self) -> Dict[str, Any]:
        """Monitor GPU resources."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return {'available': False, 'reason': 'cuda_not_available'}
                
            gpu_stats = []
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_cached = torch.cuda.memory_reserved(i) / (1024**3)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                
                gpu_stats.append({
                    'device_id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_allocated_gb': memory_allocated,
                    'memory_cached_gb': memory_cached,
                    'memory_total_gb': total_memory,
                    'memory_percent': ((memory_allocated + memory_cached) / total_memory) * 100
                })
                
            return {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'devices': gpu_stats
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}
            
    def _monitor_application_metrics(self) -> Dict[str, Any]:
        """Monitor application-specific metrics."""
        try:
            # Memory safety metrics
            memory_stats = self.memory_manager.monitor_memory_usage()
            
            # Adversarial defense metrics
            defense_stats = get_adversarial_defense().get_defense_stats()
            
            return {
                'memory_safety': memory_stats,
                'adversarial_defense': defense_stats,
                'available': True
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}
            
    def _check_alert_conditions(self, resource_snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        current_time = time.time()
        
        # System memory alert
        if (resource_snapshot['system'].get('available', False) and 
            resource_snapshot['system']['memory_percent'] > self.alert_thresholds['memory_percent']):
            alerts.append({
                'timestamp': current_time,
                'type': 'memory_high',
                'severity': 'warning',
                'value': resource_snapshot['system']['memory_percent'],
                'threshold': self.alert_thresholds['memory_percent'],
                'message': f"High memory usage: {resource_snapshot['system']['memory_percent']:.1f}%"
            })
            
        # GPU memory alert
        if resource_snapshot['gpu'].get('available', False):
            for gpu in resource_snapshot['gpu']['devices']:
                if gpu['memory_percent'] > self.alert_thresholds['gpu_memory_percent']:
                    alerts.append({
                        'timestamp': current_time,
                        'type': 'gpu_memory_high',
                        'severity': 'warning',
                        'device_id': gpu['device_id'],
                        'value': gpu['memory_percent'],
                        'threshold': self.alert_thresholds['gpu_memory_percent'],
                        'message': f"High GPU memory usage on device {gpu['device_id']}: {gpu['memory_percent']:.1f}%"
                    })
                    
        # CPU alert
        if (resource_snapshot['system'].get('available', False) and 
            resource_snapshot['system']['cpu_percent'] > self.alert_thresholds['cpu_percent']):
            alerts.append({
                'timestamp': current_time,
                'type': 'cpu_high',
                'severity': 'warning',
                'value': resource_snapshot['system']['cpu_percent'],
                'threshold': self.alert_thresholds['cpu_percent'],
                'message': f"High CPU usage: {resource_snapshot['system']['cpu_percent']:.1f}%"
            })
            
        return alerts
        
    def get_resource_trends(self) -> Dict[str, Any]:
        """Get resource usage trends."""
        if len(self.resource_history) < 2:
            return {'insufficient_data': True}
            
        # Calculate trends over last 10 minutes
        recent_cutoff = time.time() - 600
        recent_data = [r for r in self.resource_history if r['timestamp'] > recent_cutoff]
        
        if len(recent_data) < 2:
            return {'insufficient_recent_data': True}
            
        # Memory trend
        memory_values = [r['system'].get('memory_percent', 0) for r in recent_data 
                        if r['system'].get('available', False)]
        
        trends = {}
        if memory_values:
            trends['memory_trend'] = self._calculate_trend(memory_values)
            
        # Add recent alerts
        recent_alerts = [a for a in self.alerts_triggered 
                        if time.time() - a['timestamp'] < 300]  # Last 5 minutes
        
        return {
            'trends': trends,
            'recent_alerts': recent_alerts,
            'alert_count': len(recent_alerts)
        }
        
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return 'insufficient_data'
            
        start_avg = np.mean(values[:len(values)//2])
        end_avg = np.mean(values[len(values)//2:])
        
        change_percent = ((end_avg - start_avg) / start_avg) * 100 if start_avg > 0 else 0
        
        if change_percent > 5:
            return 'increasing'
        elif change_percent < -5:
            return 'decreasing'
        else:
            return 'stable'


# Global instances
_global_resilient_operation_manager = None
_global_resource_monitor = None


def get_resilient_operation_manager() -> ResilientOperationManager:
    """Get global resilient operation manager instance."""
    global _global_resilient_operation_manager
    if _global_resilient_operation_manager is None:
        _global_resilient_operation_manager = ResilientOperationManager()
    return _global_resilient_operation_manager


def get_resource_monitor() -> ResourceMonitor:
    """Get global resource monitor instance."""
    global _global_resource_monitor
    if _global_resource_monitor is None:
        _global_resource_monitor = ResourceMonitor()
    return _global_resource_monitor