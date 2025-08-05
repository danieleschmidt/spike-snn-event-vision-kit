"""
Monitoring and health check utilities for spike-snn-event-vision-kit.

Provides comprehensive monitoring, logging, and health check functionality
for production deployment and system observability.
"""

import logging
import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import torch
from datetime import datetime, timedelta


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    events_processed_per_sec: float = 0.0
    inference_latency_ms: float = 0.0
    detection_accuracy: float = 0.0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0


@dataclass
class HealthStatus:
    """System health status."""
    timestamp: float
    overall_status: str  # "healthy", "warning", "critical"
    component_statuses: Dict[str, str]
    alerts: List[str]
    recommendations: List[str]


class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.component_metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.start_time = time.time()
        self._lock = threading.Lock()
        
        # Initialize counters
        self.counters = {
            'events_processed': 0,
            'detections_made': 0,
            'errors_occurred': 0,
            'inference_calls': 0
        }
        
        # Performance tracking
        self.latency_samples = deque(maxlen=100)
        self.accuracy_samples = deque(maxlen=100)
        
    def record_events_processed(self, count: int):
        """Record number of events processed."""
        with self._lock:
            self.counters['events_processed'] += count
            
    def record_detection(self, accuracy: Optional[float] = None):
        """Record a detection event."""
        with self._lock:
            self.counters['detections_made'] += 1
            if accuracy is not None:
                self.accuracy_samples.append(accuracy)
                
    def record_error(self, error_type: str = "unknown"):
        """Record an error occurrence."""
        with self._lock:
            self.counters['errors_occurred'] += 1
            # Could extend to track error types
            
    def record_inference_latency(self, latency_ms: float):
        """Record inference latency."""
        with self._lock:
            self.counters['inference_calls'] += 1
            self.latency_samples.append(latency_ms)
            
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        with self._lock:
            # Calculate rates (per second over last minute)
            time_window = min(60.0, uptime)
            events_per_sec = self.counters['events_processed'] / time_window if time_window > 0 else 0.0
            error_rate = self.counters['errors_occurred'] / max(1, self.counters['inference_calls'])
            
            # Calculate averages
            avg_latency = np.mean(self.latency_samples) if self.latency_samples else 0.0
            avg_accuracy = np.mean(self.accuracy_samples) if self.accuracy_samples else 0.0
            
        # Get system resource usage
        cpu_usage, memory_usage, gpu_memory = self._get_system_resources()
        
        metrics = SystemMetrics(
            timestamp=current_time,
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_usage,
            gpu_memory_mb=gpu_memory,
            events_processed_per_sec=events_per_sec,
            inference_latency_ms=avg_latency,
            detection_accuracy=avg_accuracy,
            error_rate=error_rate,
            uptime_seconds=uptime
        )
        
        self.metrics_history.append(metrics)
        return metrics
        
    def _get_system_resources(self) -> tuple:
        """Get system resource utilization."""
        cpu_usage = 0.0
        memory_usage = 0.0
        gpu_memory = 0.0
        
        try:
            import psutil
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.used / 1024 / 1024  # MB
        except ImportError:
            pass
            
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        except Exception:
            pass
            
        return cpu_usage, memory_usage, gpu_memory
        
    def get_metrics_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get summary of metrics over time window."""
        cutoff_time = time.time() - (time_window_minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No recent metrics available"}
            
        return {
            "time_window_minutes": time_window_minutes,
            "sample_count": len(recent_metrics),
            "avg_cpu_usage": np.mean([m.cpu_usage_percent for m in recent_metrics]),
            "avg_memory_mb": np.mean([m.memory_usage_mb for m in recent_metrics]),
            "avg_gpu_memory_mb": np.mean([m.gpu_memory_mb for m in recent_metrics]),
            "avg_events_per_sec": np.mean([m.events_processed_per_sec for m in recent_metrics]),
            "avg_latency_ms": np.mean([m.inference_latency_ms for m in recent_metrics]),
            "avg_accuracy": np.mean([m.detection_accuracy for m in recent_metrics]),
            "avg_error_rate": np.mean([m.error_rate for m in recent_metrics]),
            "max_cpu_usage": np.max([m.cpu_usage_percent for m in recent_metrics]),
            "max_memory_mb": np.max([m.memory_usage_mb for m in recent_metrics]),
            "min_accuracy": np.min([m.detection_accuracy for m in recent_metrics]) if recent_metrics[0].detection_accuracy > 0 else 0,
            "max_latency_ms": np.max([m.inference_latency_ms for m in recent_metrics])
        }


class HealthChecker:
    """Monitors system health and generates alerts."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.thresholds = {
            'cpu_usage_warning': 80.0,
            'cpu_usage_critical': 95.0,
            'memory_usage_warning': 1000.0,  # MB
            'memory_usage_critical': 2000.0,  # MB
            'latency_warning': 100.0,  # ms
            'latency_critical': 500.0,  # ms
            'error_rate_warning': 0.05,  # 5%
            'error_rate_critical': 0.20,  # 20%
            'accuracy_warning': 0.70,  # Below 70%
            'accuracy_critical': 0.50,  # Below 50%
        }
        
    def check_health(self) -> HealthStatus:
        """Perform comprehensive health check."""
        current_metrics = self.metrics_collector.get_current_metrics()
        component_statuses = {}
        alerts = []
        recommendations = []
        
        # Check CPU usage
        cpu_status = self._check_threshold(
            current_metrics.cpu_usage_percent,
            self.thresholds['cpu_usage_warning'],
            self.thresholds['cpu_usage_critical']
        )
        component_statuses['cpu'] = cpu_status
        if cpu_status == 'critical':
            alerts.append(f"Critical CPU usage: {current_metrics.cpu_usage_percent:.1f}%")
            recommendations.append("Consider reducing batch size or using GPU acceleration")
        elif cpu_status == 'warning':
            alerts.append(f"High CPU usage: {current_metrics.cpu_usage_percent:.1f}%")
            
        # Check memory usage
        memory_status = self._check_threshold(
            current_metrics.memory_usage_mb,
            self.thresholds['memory_usage_warning'],
            self.thresholds['memory_usage_critical']
        )
        component_statuses['memory'] = memory_status
        if memory_status == 'critical':
            alerts.append(f"Critical memory usage: {current_metrics.memory_usage_mb:.1f}MB")
            recommendations.append("Restart system or reduce processing load")
        elif memory_status == 'warning':
            alerts.append(f"High memory usage: {current_metrics.memory_usage_mb:.1f}MB")
            
        # Check inference latency
        latency_status = self._check_threshold(
            current_metrics.inference_latency_ms,
            self.thresholds['latency_warning'],
            self.thresholds['latency_critical']
        )
        component_statuses['latency'] = latency_status
        if latency_status == 'critical':
            alerts.append(f"Critical inference latency: {current_metrics.inference_latency_ms:.1f}ms")
            recommendations.append("Check GPU availability or reduce model complexity")
        elif latency_status == 'warning':
            alerts.append(f"High inference latency: {current_metrics.inference_latency_ms:.1f}ms")
            
        # Check error rate
        error_status = self._check_threshold(
            current_metrics.error_rate,
            self.thresholds['error_rate_warning'],
            self.thresholds['error_rate_critical']
        )
        component_statuses['errors'] = error_status
        if error_status == 'critical':
            alerts.append(f"Critical error rate: {current_metrics.error_rate:.1%}")
            recommendations.append("Check logs for recurring errors and restart if necessary")
        elif error_status == 'warning':
            alerts.append(f"High error rate: {current_metrics.error_rate:.1%}")
            
        # Check accuracy (if available)
        if current_metrics.detection_accuracy > 0:
            accuracy_status = self._check_threshold(
                current_metrics.detection_accuracy,
                self.thresholds['accuracy_critical'],
                self.thresholds['accuracy_warning'],
                reverse=True  # Lower values are worse
            )
            component_statuses['accuracy'] = accuracy_status
            if accuracy_status == 'critical':
                alerts.append(f"Low detection accuracy: {current_metrics.detection_accuracy:.1%}")
                recommendations.append("Check model weights or retrain model")
            elif accuracy_status == 'warning':
                alerts.append(f"Reduced detection accuracy: {current_metrics.detection_accuracy:.1%}")
        else:
            component_statuses['accuracy'] = 'unknown'
            
        # Determine overall status
        overall_status = self._determine_overall_status(component_statuses)
        
        return HealthStatus(
            timestamp=time.time(),
            overall_status=overall_status,
            component_statuses=component_statuses,
            alerts=alerts,
            recommendations=recommendations
        )
        
    def _check_threshold(
        self, 
        value: float, 
        warning_threshold: float, 
        critical_threshold: float,
        reverse: bool = False
    ) -> str:
        """Check value against thresholds."""
        if reverse:
            if value < critical_threshold:
                return 'critical'
            elif value < warning_threshold:
                return 'warning'
        else:
            if value > critical_threshold:
                return 'critical'
            elif value > warning_threshold:
                return 'warning'
        return 'healthy'
        
    def _determine_overall_status(self, component_statuses: Dict[str, str]) -> str:
        """Determine overall system status."""
        if any(status == 'critical' for status in component_statuses.values()):
            return 'critical'
        elif any(status == 'warning' for status in component_statuses.values()):
            return 'warning'
        else:
            return 'healthy'


class Logger:
    """Enhanced logging with structured output and monitoring integration."""
    
    def __init__(self, name: str, level: str = "INFO", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        # Metrics integration
        self.metrics_collector = None
        
    def set_metrics_collector(self, metrics_collector: MetricsCollector):
        """Associate with metrics collector for error tracking."""
        self.metrics_collector = metrics_collector
        
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(self._format_message(message, **kwargs))
        
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(self._format_message(message, **kwargs))
        
    def error(self, message: str, **kwargs):
        """Log error message and update metrics."""
        self.logger.error(self._format_message(message, **kwargs))
        if self.metrics_collector:
            self.metrics_collector.record_error(kwargs.get('error_type', 'general'))
            
    def critical(self, message: str, **kwargs):
        """Log critical message and update metrics."""
        self.logger.critical(self._format_message(message, **kwargs))
        if self.metrics_collector:
            self.metrics_collector.record_error(kwargs.get('error_type', 'critical'))
            
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(self._format_message(message, **kwargs))
        
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with additional context."""
        if kwargs:
            context = ' | '.join(f"{k}={v}" for k, v in kwargs.items())
            return f"{message} | {context}"
        return message


class MonitoringDashboard:
    """Simple monitoring dashboard for system status."""
    
    def __init__(
        self, 
        metrics_collector: MetricsCollector,
        health_checker: HealthChecker,
        update_interval: float = 30.0
    ):
        self.metrics_collector = metrics_collector
        self.health_checker = health_checker
        self.update_interval = update_interval
        self.running = False
        self._thread = None
        
    def start(self):
        """Start monitoring dashboard."""
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            
    def stop(self):
        """Stop monitoring dashboard."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                health_status = self.health_checker.check_health()
                metrics_summary = self.metrics_collector.get_metrics_summary(60)
                
                self._print_status(health_status, metrics_summary)
                
            except Exception as e:
                logging.error(f"Monitoring dashboard error: {e}")
                
            time.sleep(self.update_interval)
            
    def _print_status(self, health_status: HealthStatus, metrics_summary: Dict[str, Any]):
        """Print current system status."""
        print("\n" + "="*60)
        print(f"SYSTEM STATUS - {datetime.fromtimestamp(health_status.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Overall status
        status_symbol = {
            'healthy': '✓',
            'warning': '⚠',
            'critical': '✗'
        }.get(health_status.overall_status, '?')
        
        print(f"Overall Status: {status_symbol} {health_status.overall_status.upper()}")
        
        # Component status
        print("\nComponent Status:")
        for component, status in health_status.component_statuses.items():
            symbol = {'healthy': '✓', 'warning': '⚠', 'critical': '✗', 'unknown': '?'}.get(status, '?')
            print(f"  {component.capitalize():12} {symbol} {status}")
            
        # Key metrics
        if 'error' not in metrics_summary:
            print("\nKey Metrics (last 60 min):")
            print(f"  Events/sec:     {metrics_summary.get('avg_events_per_sec', 0):.1f}")
            print(f"  Avg Latency:    {metrics_summary.get('avg_latency_ms', 0):.1f}ms")
            print(f"  CPU Usage:      {metrics_summary.get('avg_cpu_usage', 0):.1f}%")
            print(f"  Memory Usage:   {metrics_summary.get('avg_memory_mb', 0):.1f}MB")
            print(f"  Error Rate:     {metrics_summary.get('avg_error_rate', 0):.1%}")
            if metrics_summary.get('avg_accuracy', 0) > 0:
                print(f"  Accuracy:       {metrics_summary.get('avg_accuracy', 0):.1%}")
                
        # Alerts
        if health_status.alerts:
            print("\nAlerts:")
            for alert in health_status.alerts:
                print(f"  ⚠ {alert}")
                
        # Recommendations
        if health_status.recommendations:
            print("\nRecommendations:")
            for rec in health_status.recommendations:
                print(f"  💡 {rec}")
                
        print("="*60)
        
    def export_metrics(self, filepath: str):
        """Export current metrics to file."""
        health_status = self.health_checker.check_health()
        metrics_summary = self.metrics_collector.get_metrics_summary()
        current_metrics = self.metrics_collector.get_current_metrics()
        
        export_data = {
            'timestamp': time.time(),
            'health_status': asdict(health_status),
            'metrics_summary': metrics_summary,
            'current_metrics': asdict(current_metrics)
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)


# Global monitoring instances
_global_metrics_collector = None
_global_health_checker = None
_global_dashboard = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker(get_metrics_collector())
    return _global_health_checker


def get_dashboard() -> MonitoringDashboard:
    """Get global monitoring dashboard instance."""
    global _global_dashboard
    if _global_dashboard is None:
        _global_dashboard = MonitoringDashboard(
            get_metrics_collector(),
            get_health_checker()
        )
    return _global_dashboard


def start_monitoring():
    """Start global monitoring."""
    dashboard = get_dashboard()
    dashboard.start()
    logging.info("Monitoring dashboard started")


def stop_monitoring():
    """Stop global monitoring."""
    dashboard = get_dashboard()
    dashboard.stop()
    logging.info("Monitoring dashboard stopped")


def export_system_report(filepath: str = "system_report.json"):
    """Export comprehensive system report."""
    dashboard = get_dashboard()
    dashboard.export_metrics(filepath)
    logging.info(f"System report exported to {filepath}")