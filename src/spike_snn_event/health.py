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
from .validation import ValidationError, HardwareError
from .security import get_input_sanitizer


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