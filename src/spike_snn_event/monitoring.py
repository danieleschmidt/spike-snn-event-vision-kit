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
        cpu_status = self._check_threshold(\n            current_metrics.cpu_usage_percent,\n            self.thresholds['cpu_usage_warning'],\n            self.thresholds['cpu_usage_critical']\n        )\n        component_statuses['cpu'] = cpu_status\n        if cpu_status == 'critical':\n            alerts.append(f\"Critical CPU usage: {current_metrics.cpu_usage_percent:.1f}%\")\n            recommendations.append(\"Consider reducing batch size or using GPU acceleration\")\n        elif cpu_status == 'warning':\n            alerts.append(f\"High CPU usage: {current_metrics.cpu_usage_percent:.1f}%\")\n            \n        # Check memory usage\n        memory_status = self._check_threshold(\n            current_metrics.memory_usage_mb,\n            self.thresholds['memory_usage_warning'],\n            self.thresholds['memory_usage_critical']\n        )\n        component_statuses['memory'] = memory_status\n        if memory_status == 'critical':\n            alerts.append(f\"Critical memory usage: {current_metrics.memory_usage_mb:.1f}MB\")\n            recommendations.append(\"Restart system or reduce processing load\")\n        elif memory_status == 'warning':\n            alerts.append(f\"High memory usage: {current_metrics.memory_usage_mb:.1f}MB\")\n            \n        # Check inference latency\n        latency_status = self._check_threshold(\n            current_metrics.inference_latency_ms,\n            self.thresholds['latency_warning'],\n            self.thresholds['latency_critical']\n        )\n        component_statuses['latency'] = latency_status\n        if latency_status == 'critical':\n            alerts.append(f\"Critical inference latency: {current_metrics.inference_latency_ms:.1f}ms\")\n            recommendations.append(\"Check GPU availability or reduce model complexity\")\n        elif latency_status == 'warning':\n            alerts.append(f\"High inference latency: {current_metrics.inference_latency_ms:.1f}ms\")\n            \n        # Check error rate\n        error_status = self._check_threshold(\n            current_metrics.error_rate,\n            self.thresholds['error_rate_warning'],\n            self.thresholds['error_rate_critical']\n        )\n        component_statuses['errors'] = error_status\n        if error_status == 'critical':\n            alerts.append(f\"Critical error rate: {current_metrics.error_rate:.1%}\")\n            recommendations.append(\"Check logs for recurring errors and restart if necessary\")\n        elif error_status == 'warning':\n            alerts.append(f\"High error rate: {current_metrics.error_rate:.1%}\")\n            \n        # Check accuracy (if available)\n        if current_metrics.detection_accuracy > 0:\n            accuracy_status = self._check_threshold(\n                current_metrics.detection_accuracy,\n                self.thresholds['accuracy_critical'],\n                self.thresholds['accuracy_warning'],\n                reverse=True  # Lower values are worse\n            )\n            component_statuses['accuracy'] = accuracy_status\n            if accuracy_status == 'critical':\n                alerts.append(f\"Low detection accuracy: {current_metrics.detection_accuracy:.1%}\")\n                recommendations.append(\"Check model weights or retrain model\")\n            elif accuracy_status == 'warning':\n                alerts.append(f\"Reduced detection accuracy: {current_metrics.detection_accuracy:.1%}\")\n        else:\n            component_statuses['accuracy'] = 'unknown'\n            \n        # Determine overall status\n        overall_status = self._determine_overall_status(component_statuses)\n        \n        return HealthStatus(\n            timestamp=time.time(),\n            overall_status=overall_status,\n            component_statuses=component_statuses,\n            alerts=alerts,\n            recommendations=recommendations\n        )\n        \n    def _check_threshold(\n        self, \n        value: float, \n        warning_threshold: float, \n        critical_threshold: float,\n        reverse: bool = False\n    ) -> str:\n        \"\"\"Check value against thresholds.\"\"\"\n        if reverse:\n            if value < critical_threshold:\n                return 'critical'\n            elif value < warning_threshold:\n                return 'warning'\n        else:\n            if value > critical_threshold:\n                return 'critical'\n            elif value > warning_threshold:\n                return 'warning'\n        return 'healthy'\n        \n    def _determine_overall_status(self, component_statuses: Dict[str, str]) -> str:\n        \"\"\"Determine overall system status.\"\"\"\n        if any(status == 'critical' for status in component_statuses.values()):\n            return 'critical'\n        elif any(status == 'warning' for status in component_statuses.values()):\n            return 'warning'\n        else:\n            return 'healthy'\n\n\nclass Logger:\n    \"\"\"Enhanced logging with structured output and monitoring integration.\"\"\"\n    \n    def __init__(self, name: str, level: str = \"INFO\", log_file: Optional[str] = None):\n        self.logger = logging.getLogger(name)\n        self.logger.setLevel(getattr(logging, level.upper()))\n        \n        # Create formatter\n        formatter = logging.Formatter(\n            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'\n        )\n        \n        # Console handler\n        console_handler = logging.StreamHandler()\n        console_handler.setFormatter(formatter)\n        self.logger.addHandler(console_handler)\n        \n        # File handler if specified\n        if log_file:\n            file_handler = logging.FileHandler(log_file)\n            file_handler.setFormatter(formatter)\n            self.logger.addHandler(file_handler)\n            \n        # Metrics integration\n        self.metrics_collector = None\n        \n    def set_metrics_collector(self, metrics_collector: MetricsCollector):\n        \"\"\"Associate with metrics collector for error tracking.\"\"\"\n        self.metrics_collector = metrics_collector\n        \n    def info(self, message: str, **kwargs):\n        \"\"\"Log info message.\"\"\"\n        self.logger.info(self._format_message(message, **kwargs))\n        \n    def warning(self, message: str, **kwargs):\n        \"\"\"Log warning message.\"\"\"\n        self.logger.warning(self._format_message(message, **kwargs))\n        \n    def error(self, message: str, **kwargs):\n        \"\"\"Log error message and update metrics.\"\"\"\n        self.logger.error(self._format_message(message, **kwargs))\n        if self.metrics_collector:\n            self.metrics_collector.record_error(kwargs.get('error_type', 'general'))\n            \n    def critical(self, message: str, **kwargs):\n        \"\"\"Log critical message and update metrics.\"\"\"\n        self.logger.critical(self._format_message(message, **kwargs))\n        if self.metrics_collector:\n            self.metrics_collector.record_error(kwargs.get('error_type', 'critical'))\n            \n    def debug(self, message: str, **kwargs):\n        \"\"\"Log debug message.\"\"\"\n        self.logger.debug(self._format_message(message, **kwargs))\n        \n    def _format_message(self, message: str, **kwargs) -> str:\n        \"\"\"Format message with additional context.\"\"\"\n        if kwargs:\n            context = ' | '.join(f\"{k}={v}\" for k, v in kwargs.items())\n            return f\"{message} | {context}\"\n        return message\n\n\nclass MonitoringDashboard:\n    \"\"\"Simple monitoring dashboard for system status.\"\"\"\n    \n    def __init__(\n        self, \n        metrics_collector: MetricsCollector,\n        health_checker: HealthChecker,\n        update_interval: float = 30.0\n    ):\n        self.metrics_collector = metrics_collector\n        self.health_checker = health_checker\n        self.update_interval = update_interval\n        self.running = False\n        self._thread = None\n        \n    def start(self):\n        \"\"\"Start monitoring dashboard.\"\"\"\n        if not self.running:\n            self.running = True\n            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)\n            self._thread.start()\n            \n    def stop(self):\n        \"\"\"Stop monitoring dashboard.\"\"\"\n        self.running = False\n        if self._thread:\n            self._thread.join(timeout=5.0)\n            \n    def _monitor_loop(self):\n        \"\"\"Main monitoring loop.\"\"\"\n        while self.running:\n            try:\n                health_status = self.health_checker.check_health()\n                metrics_summary = self.metrics_collector.get_metrics_summary(60)\n                \n                self._print_status(health_status, metrics_summary)\n                \n            except Exception as e:\n                logging.error(f\"Monitoring dashboard error: {e}\")\n                \n            time.sleep(self.update_interval)\n            \n    def _print_status(self, health_status: HealthStatus, metrics_summary: Dict[str, Any]):\n        \"\"\"Print current system status.\"\"\"\n        print(\"\\n\" + \"=\"*60)\n        print(f\"SYSTEM STATUS - {datetime.fromtimestamp(health_status.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\")\n        print(\"=\"*60)\n        \n        # Overall status\n        status_symbol = {\n            'healthy': '✓',\n            'warning': '⚠',\n            'critical': '✗'\n        }.get(health_status.overall_status, '?')\n        \n        print(f\"Overall Status: {status_symbol} {health_status.overall_status.upper()}\")\n        \n        # Component status\n        print(\"\\nComponent Status:\")\n        for component, status in health_status.component_statuses.items():\n            symbol = {'healthy': '✓', 'warning': '⚠', 'critical': '✗', 'unknown': '?'}.get(status, '?')\n            print(f\"  {component.capitalize():12} {symbol} {status}\")\n            \n        # Key metrics\n        if 'error' not in metrics_summary:\n            print(\"\\nKey Metrics (last 60 min):\")\n            print(f\"  Events/sec:     {metrics_summary.get('avg_events_per_sec', 0):.1f}\")\n            print(f\"  Avg Latency:    {metrics_summary.get('avg_latency_ms', 0):.1f}ms\")\n            print(f\"  CPU Usage:      {metrics_summary.get('avg_cpu_usage', 0):.1f}%\")\n            print(f\"  Memory Usage:   {metrics_summary.get('avg_memory_mb', 0):.1f}MB\")\n            print(f\"  Error Rate:     {metrics_summary.get('avg_error_rate', 0):.1%}\")\n            if metrics_summary.get('avg_accuracy', 0) > 0:\n                print(f\"  Accuracy:       {metrics_summary.get('avg_accuracy', 0):.1%}\")\n                \n        # Alerts\n        if health_status.alerts:\n            print(\"\\nAlerts:\")\n            for alert in health_status.alerts:\n                print(f\"  ⚠ {alert}\")\n                \n        # Recommendations\n        if health_status.recommendations:\n            print(\"\\nRecommendations:\")\n            for rec in health_status.recommendations:\n                print(f\"  💡 {rec}\")\n                \n        print(\"=\"*60)\n        \n    def export_metrics(self, filepath: str):\n        \"\"\"Export current metrics to file.\"\"\"\n        health_status = self.health_checker.check_health()\n        metrics_summary = self.metrics_collector.get_metrics_summary()\n        current_metrics = self.metrics_collector.get_current_metrics()\n        \n        export_data = {\n            'timestamp': time.time(),\n            'health_status': asdict(health_status),\n            'metrics_summary': metrics_summary,\n            'current_metrics': asdict(current_metrics)\n        }\n        \n        with open(filepath, 'w') as f:\n            json.dump(export_data, f, indent=2, default=str)\n\n\n# Global monitoring instances\n_global_metrics_collector = None\n_global_health_checker = None\n_global_dashboard = None\n\n\ndef get_metrics_collector() -> MetricsCollector:\n    \"\"\"Get global metrics collector instance.\"\"\"\n    global _global_metrics_collector\n    if _global_metrics_collector is None:\n        _global_metrics_collector = MetricsCollector()\n    return _global_metrics_collector\n\n\ndef get_health_checker() -> HealthChecker:\n    \"\"\"Get global health checker instance.\"\"\"\n    global _global_health_checker\n    if _global_health_checker is None:\n        _global_health_checker = HealthChecker(get_metrics_collector())\n    return _global_health_checker\n\n\ndef get_dashboard() -> MonitoringDashboard:\n    \"\"\"Get global monitoring dashboard instance.\"\"\"\n    global _global_dashboard\n    if _global_dashboard is None:\n        _global_dashboard = MonitoringDashboard(\n            get_metrics_collector(),\n            get_health_checker()\n        )\n    return _global_dashboard\n\n\ndef start_monitoring():\n    \"\"\"Start global monitoring.\"\"\"\n    dashboard = get_dashboard()\n    dashboard.start()\n    logging.info(\"Monitoring dashboard started\")\n\n\ndef stop_monitoring():\n    \"\"\"Stop global monitoring.\"\"\"\n    dashboard = get_dashboard()\n    dashboard.stop()\n    logging.info(\"Monitoring dashboard stopped\")\n\n\ndef export_system_report(filepath: str = \"system_report.json\"):\n    \"\"\"Export comprehensive system report.\"\"\"\n    dashboard = get_dashboard()\n    dashboard.export_metrics(filepath)\n    logging.info(f\"System report exported to {filepath}\")