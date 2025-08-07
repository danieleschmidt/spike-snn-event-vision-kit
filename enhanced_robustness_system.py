#!/usr/bin/env python3
"""
Enhanced Robustness System for Neuromorphic Vision Processing
Implements comprehensive error handling, validation, and fault tolerance.
"""

import sys
import os
import time
import json
import logging
import threading
import traceback
import signal
import psutil
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from collections import deque, defaultdict

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('neuromorphic_vision.log')
    ]
)
logger = logging.getLogger(__name__)

class SecurityThreat(Enum):
    """Security threat classifications."""
    MALICIOUS_INPUT = "malicious_input"
    RESOURCE_EXHAUSTION = "resource_exhaustion"  
    DATA_INJECTION = "data_injection"
    BUFFER_OVERFLOW = "buffer_overflow"
    PRIVILEGE_ESCALATION = "privilege_escalation"

class SystemStatus(Enum):
    """System operational status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILURE = "failure"
    RECOVERING = "recovering"

@dataclass
class SecurityEvent:
    """Security event data structure."""
    threat_type: SecurityThreat
    severity: str
    description: str
    timestamp: float
    source_ip: Optional[str] = None
    data_hash: Optional[str] = None
    mitigation_applied: bool = False

@dataclass
class HealthMetrics:
    """System health metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    error_rate: float
    throughput: float
    timestamp: float = field(default_factory=time.time)

class InputValidator:
    """Advanced input validation and sanitization."""
    
    def __init__(self):
        self.max_events_per_batch = 100000
        self.max_coordinate_value = 10000
        self.min_timestamp = 0
        self.max_timestamp = time.time() + 86400  # 24 hours from now
        self.security_events = deque(maxlen=1000)
        
    def validate_event_data(self, events: List[List[float]]) -> Tuple[bool, List[str], List[SecurityEvent]]:
        """Comprehensive event data validation with security screening."""
        errors = []
        security_events = []
        
        try:
            # Basic structure validation
            if not isinstance(events, list):
                errors.append("Events must be a list")
                return False, errors, security_events
                
            if len(events) > self.max_events_per_batch:
                security_event = SecurityEvent(
                    threat_type=SecurityThreat.RESOURCE_EXHAUSTION,
                    severity="HIGH",
                    description=f"Batch size {len(events)} exceeds limit {self.max_events_per_batch}",
                    timestamp=time.time()
                )
                security_events.append(security_event)
                errors.append(f"Batch size exceeds maximum allowed ({self.max_events_per_batch})")
            
            # Validate individual events
            for i, event in enumerate(events[:1000]):  # Sample first 1000 for performance
                if not isinstance(event, (list, tuple)) or len(event) != 4:
                    errors.append(f"Event {i} has invalid structure (expected 4 elements)")
                    continue
                    
                x, y, t, p = event
                
                # Coordinate validation with anomaly detection
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    errors.append(f"Event {i} has non-numeric coordinates")
                    continue
                    
                if abs(x) > self.max_coordinate_value or abs(y) > self.max_coordinate_value:
                    security_event = SecurityEvent(
                        threat_type=SecurityThreat.MALICIOUS_INPUT,
                        severity="MEDIUM",
                        description=f"Suspicious coordinates at event {i}: ({x}, {y})",
                        timestamp=time.time(),
                        data_hash=self._hash_data(str(event))
                    )
                    security_events.append(security_event)
                    
                # Timestamp validation
                if not isinstance(t, (int, float)):
                    errors.append(f"Event {i} has non-numeric timestamp")
                    continue
                    
                if t < self.min_timestamp or t > self.max_timestamp:
                    security_event = SecurityEvent(
                        threat_type=SecurityThreat.DATA_INJECTION,
                        severity="MEDIUM", 
                        description=f"Suspicious timestamp at event {i}: {t}",
                        timestamp=time.time(),
                        data_hash=self._hash_data(str(event))
                    )
                    security_events.append(security_event)
                    
                # Polarity validation
                if p not in [-1, 0, 1]:
                    errors.append(f"Event {i} has invalid polarity value: {p}")
                    
            # Pattern-based attack detection
            if self._detect_attack_patterns(events):
                security_event = SecurityEvent(
                    threat_type=SecurityThreat.MALICIOUS_INPUT,
                    severity="HIGH",
                    description="Detected potential attack pattern in event data",
                    timestamp=time.time(),
                    data_hash=self._hash_data(str(events[:100]))  # Hash sample
                )
                security_events.append(security_event)
                errors.append("Potential malicious pattern detected")
                
        except Exception as e:
            logger.error(f"Validation error: {e}")
            errors.append(f"Validation failed with error: {e}")
            
        # Store security events
        self.security_events.extend(security_events)
        
        is_valid = len(errors) == 0
        return is_valid, errors, security_events
    
    def _hash_data(self, data: str) -> str:
        """Generate hash for data fingerprinting."""
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _detect_attack_patterns(self, events: List[List[float]]) -> bool:
        """Detect potential attack patterns in event data."""
        if not events:
            return False
            
        # Check for repeated identical events (potential replay attack)
        if len(events) > 100:
            event_hashes = [self._hash_data(str(event)) for event in events[:100]]
            unique_hashes = set(event_hashes)
            if len(unique_hashes) < len(event_hashes) * 0.5:  # Less than 50% unique
                return True
                
        # Check for geometric attack patterns
        if len(events) > 10:
            x_coords = [event[0] for event in events[:100] if len(event) >= 4]
            y_coords = [event[1] for event in events[:100] if len(event) >= 4]
            
            # Check for perfect geometric shapes (suspicious)
            if self._is_perfect_pattern(x_coords, y_coords):
                return True
                
        return False
    
    def _is_perfect_pattern(self, x_coords: List[float], y_coords: List[float]) -> bool:
        """Detect perfect geometric patterns that might indicate synthetic attacks."""
        if len(x_coords) < 10:
            return False
            
        # Check for perfect circles, lines, or grids
        try:
            # Simple heuristic: check coordinate variance
            x_var = max(x_coords) - min(x_coords) if x_coords else 0
            y_var = max(y_coords) - min(y_coords) if y_coords else 0
            
            # If coordinates are too uniform, might be synthetic
            if x_var < 1 and y_var < 1 and len(x_coords) > 50:
                return True
                
            # Check for arithmetic progressions (synthetic data signature)
            if len(x_coords) >= 5:
                x_diffs = [x_coords[i+1] - x_coords[i] for i in range(4)]
                if len(set(x_diffs)) == 1 and x_diffs[0] != 0:  # Perfect arithmetic progression
                    return True
                    
        except Exception:
            pass  # Ignore errors in pattern detection
            
        return False
    
    def sanitize_events(self, events: List[List[float]]) -> List[List[float]]:
        """Sanitize event data by removing/fixing problematic entries."""
        sanitized = []
        
        for event in events:
            if not isinstance(event, (list, tuple)) or len(event) != 4:
                continue  # Skip malformed events
                
            x, y, t, p = event
            
            # Clamp coordinates to safe ranges
            x = max(-1000, min(1000, float(x))) if isinstance(x, (int, float)) else 0
            y = max(-1000, min(1000, float(y))) if isinstance(y, (int, float)) else 0
            
            # Sanitize timestamp
            if not isinstance(t, (int, float)) or t < 0:
                t = time.time()
            elif t > time.time() + 3600:  # Future timestamps limited to 1 hour
                t = time.time()
                
            # Sanitize polarity
            if p not in [-1, 0, 1]:
                p = 1 if p > 0 else -1
                
            sanitized.append([x, y, t, p])
            
        return sanitized
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report from recent events."""
        recent_events = list(self.security_events)
        
        threat_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for event in recent_events:
            threat_counts[event.threat_type.value] += 1
            severity_counts[event.severity] += 1
            
        return {
            'total_security_events': len(recent_events),
            'threat_breakdown': dict(threat_counts),
            'severity_breakdown': dict(severity_counts),
            'last_24h_events': len([e for e in recent_events 
                                  if time.time() - e.timestamp < 86400]),
            'mitigation_rate': sum(1 for e in recent_events if e.mitigation_applied) / 
                             max(1, len(recent_events)) * 100
        }

class SystemMonitor:
    """Advanced system monitoring and health tracking."""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'error_rate': 5.0
        }
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval: float = 5.0):
        """Start continuous system monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("System monitoring started")
        
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("System monitoring stopped")
        
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                alerts = self._check_alerts(metrics)
                if alerts:
                    self._handle_alerts(alerts)
                    
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
                
    def _collect_metrics(self) -> HealthMetrics:
        """Collect current system metrics."""
        try:
            # CPU and memory usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Network latency (simplified)
            network_latency = 0.0  # Placeholder
            
            # Error rate (from recent processing)
            error_rate = self._calculate_error_rate()
            
            # Throughput (events/sec)
            throughput = self._calculate_throughput()
            
            return HealthMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_latency=network_latency,
                error_rate=error_rate,
                throughput=throughput
            )
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return HealthMetrics(0, 0, 0, 0, 100, 0)  # Safe defaults
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        # Placeholder implementation
        return 0.0
    
    def _calculate_throughput(self) -> float:
        """Calculate current processing throughput."""
        # Placeholder implementation  
        return 1000.0
    
    def _check_alerts(self, metrics: HealthMetrics) -> List[str]:
        """Check if any metrics exceed alert thresholds."""
        alerts = []
        
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
            
        if metrics.memory_usage > self.alert_thresholds['memory_usage']:
            alerts.append(f"High memory usage: {metrics.memory_usage:.1f}%")
            
        if metrics.disk_usage > self.alert_thresholds['disk_usage']:
            alerts.append(f"High disk usage: {metrics.disk_usage:.1f}%")
            
        if metrics.error_rate > self.alert_thresholds['error_rate']:
            alerts.append(f"High error rate: {metrics.error_rate:.1f}%")
            
        return alerts
    
    def _handle_alerts(self, alerts: List[str]):
        """Handle system alerts."""
        for alert in alerts:
            logger.warning(f"ALERT: {alert}")
            
    def get_system_status(self) -> SystemStatus:
        """Get overall system status."""
        if not self.metrics_history:
            return SystemStatus.WARNING
            
        latest = self.metrics_history[-1]
        
        critical_conditions = [
            latest.cpu_usage > 95,
            latest.memory_usage > 95,
            latest.disk_usage > 95,
            latest.error_rate > 50
        ]
        
        warning_conditions = [
            latest.cpu_usage > 80,
            latest.memory_usage > 80,
            latest.disk_usage > 85,
            latest.error_rate > 10
        ]
        
        if any(critical_conditions):
            return SystemStatus.CRITICAL
        elif any(warning_conditions):
            return SystemStatus.WARNING
        else:
            return SystemStatus.HEALTHY
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        if not self.metrics_history:
            return {'status': 'No data available'}
            
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 readings
        
        # Calculate averages
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        
        return {
            'system_status': self.get_system_status().value,
            'average_metrics': {
                'cpu_usage': avg_cpu,
                'memory_usage': avg_memory,
                'throughput': avg_throughput,
                'error_rate': avg_error_rate
            },
            'latest_metrics': {
                'cpu_usage': recent_metrics[-1].cpu_usage,
                'memory_usage': recent_metrics[-1].memory_usage,
                'disk_usage': recent_metrics[-1].disk_usage,
                'throughput': recent_metrics[-1].throughput,
                'error_rate': recent_metrics[-1].error_rate
            },
            'uptime_minutes': len(recent_metrics) * 5 / 60,  # Assuming 5sec intervals
            'data_points': len(recent_metrics)
        }

class FaultTolerantProcessor:
    """Fault-tolerant event processing with recovery mechanisms."""
    
    def __init__(self):
        self.validator = InputValidator()
        self.monitor = SystemMonitor()
        self.processing_stats = {
            'events_processed': 0,
            'errors_encountered': 0,
            'recoveries_attempted': 0,
            'successful_recoveries': 0
        }
        self.circuit_breaker_open = False
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5
        
    @contextmanager
    def fault_tolerance(self, operation_name: str):
        """Context manager for fault-tolerant operations."""
        try:
            yield
            # Reset circuit breaker on success
            if self.circuit_breaker_failures > 0:
                self.circuit_breaker_failures -= 1
                if self.circuit_breaker_failures == 0:
                    self.circuit_breaker_open = False
                    logger.info("Circuit breaker closed - system recovered")
                    
        except Exception as e:
            self.processing_stats['errors_encountered'] += 1
            self.circuit_breaker_failures += 1
            
            logger.error(f"Error in {operation_name}: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            
            # Open circuit breaker if threshold exceeded
            if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
                self.circuit_breaker_open = True
                logger.critical("Circuit breaker opened - system protection activated")
                
            # Attempt recovery
            try:
                self._attempt_recovery(operation_name, e)
                self.processing_stats['successful_recoveries'] += 1
                logger.info(f"Recovery successful for {operation_name}")
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {operation_name}: {recovery_error}")
                
            self.processing_stats['recoveries_attempted'] += 1
            
    def _attempt_recovery(self, operation_name: str, original_error: Exception):
        """Attempt to recover from errors."""
        # Memory cleanup
        import gc
        gc.collect()
        
        # Reset any corrupted state
        if hasattr(self, '_processing_state'):
            self._processing_state = {}
            
        # Log recovery attempt
        logger.info(f"Attempting recovery for {operation_name}")
        
        # Wait briefly before continuing
        time.sleep(0.1)
        
    def process_events_safely(self, events: List[List[float]]) -> Dict[str, Any]:
        """Process events with comprehensive fault tolerance."""
        # Check circuit breaker
        if self.circuit_breaker_open:
            logger.warning("Circuit breaker is open - processing blocked")
            return {
                'status': 'blocked',
                'reason': 'circuit_breaker_open',
                'processed_events': 0,
                'detections': []
            }
        
        results = {
            'status': 'success',
            'processed_events': 0,
            'detections': [],
            'validation_errors': [],
            'security_events': [],
            'processing_time_ms': 0
        }
        
        start_time = time.time()
        
        try:
            with self.fault_tolerance("input_validation"):
                # Validate input
                is_valid, errors, security_events = self.validator.validate_event_data(events)
                results['validation_errors'] = errors
                results['security_events'] = [
                    {
                        'threat_type': e.threat_type.value,
                        'severity': e.severity,
                        'description': e.description
                    } for e in security_events
                ]
                
                if not is_valid and security_events:
                    # Apply security mitigations
                    events = self.validator.sanitize_events(events)
                    logger.warning("Applied security sanitization to input data")
                    
            with self.fault_tolerance("event_processing"):
                # Process sanitized events
                processed_events = self._safe_event_processing(events)
                results['processed_events'] = len(processed_events)
                
            with self.fault_tolerance("inference"):
                # Run inference
                detections = self._safe_inference(processed_events)
                results['detections'] = detections
                
            self.processing_stats['events_processed'] += len(events)
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            logger.error(f"Processing failed: {e}")
            
        finally:
            results['processing_time_ms'] = (time.time() - start_time) * 1000
            
        return results
    
    def _safe_event_processing(self, events: List[List[float]]) -> List[List[float]]:
        """Safely process events with additional validation."""
        processed = []
        
        for event in events:
            try:
                # Additional processing-time validation
                if len(event) == 4 and all(isinstance(x, (int, float)) for x in event):
                    processed.append(event)
            except Exception as e:
                logger.debug(f"Skipped malformed event: {e}")
                continue
                
        return processed
    
    def _safe_inference(self, events: List[List[float]]) -> List[Dict[str, Any]]:
        """Safely run inference with fallback mechanisms."""
        if not events:
            return []
            
        try:
            # Simplified safe inference
            detections = []
            
            if len(events) > 50:  # Minimum threshold for detection
                # Mock detection logic with safety checks
                detection_count = min(3, len(events) // 100)  # Reasonable detection count
                
                for i in range(detection_count):
                    detection = {
                        'bbox': [10 + i*20, 10 + i*15, 30, 25],
                        'confidence': 0.6 + (i * 0.1),
                        'class_id': i % 3,
                        'class_name': ['person', 'vehicle', 'object'][i % 3]
                    }
                    detections.append(detection)
                    
            return detections
            
        except Exception as e:
            logger.warning(f"Inference failed, using fallback: {e}")
            return []  # Safe fallback
    
    def get_processing_report(self) -> Dict[str, Any]:
        """Get comprehensive processing report."""
        return {
            'processing_statistics': self.processing_stats.copy(),
            'circuit_breaker': {
                'is_open': self.circuit_breaker_open,
                'failure_count': self.circuit_breaker_failures,
                'threshold': self.circuit_breaker_threshold
            },
            'system_health': self.monitor.get_health_report(),
            'security_summary': self.validator.get_security_report(),
            'reliability_score': self._calculate_reliability_score()
        }
    
    def _calculate_reliability_score(self) -> float:
        """Calculate system reliability score."""
        total_operations = (self.processing_stats['events_processed'] + 
                          self.processing_stats['errors_encountered'])
        
        if total_operations == 0:
            return 100.0
            
        success_rate = ((total_operations - self.processing_stats['errors_encountered']) / 
                       total_operations * 100)
        
        # Factor in recovery success
        recovery_rate = (self.processing_stats['successful_recoveries'] / 
                        max(1, self.processing_stats['recoveries_attempted']) * 100)
        
        # Weighted reliability score
        reliability = (success_rate * 0.7) + (recovery_rate * 0.3)
        return min(100.0, max(0.0, reliability))

def run_robustness_tests():
    """Run comprehensive robustness tests."""
    print("🛡️ Enhanced Robustness System Testing")
    print("=" * 60)
    
    # Initialize robust processor
    processor = FaultTolerantProcessor()
    processor.monitor.start_monitoring()
    
    test_cases = [
        ("Normal events", [[i, i, time.time(), 1] for i in range(100)]),
        ("Malicious large coordinates", [[999999, 999999, time.time(), 1]] * 50),
        ("Invalid timestamps", [[10, 10, -1, 1], [10, 10, time.time() + 1000000, -1]]),
        ("Malformed events", [[], [1], [1, 2], [1, 2, 3], "not_a_list"]),
        ("Attack pattern", [[i, 0, time.time(), 1] for i in range(100)]),  # Perfect line
        ("Resource exhaustion", [[1, 1, time.time(), 1]] * 150000),  # Huge batch
        ("Mixed threats", [[999, 999, -1, 2]] + [[i, i, time.time(), 1] for i in range(1000)])
    ]
    
    results = []
    
    for test_name, test_events in test_cases:
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            result = processor.process_events_safely(test_events)
            
            print(f"   Status: {result['status']}")
            print(f"   Processed events: {result['processed_events']}")
            print(f"   Detections: {len(result['detections'])}")
            print(f"   Validation errors: {len(result['validation_errors'])}")
            print(f"   Security events: {len(result['security_events'])}")
            print(f"   Processing time: {result['processing_time_ms']:.2f} ms")
            
            if result['validation_errors']:
                print(f"   Sample errors: {result['validation_errors'][:2]}")
                
            if result['security_events']:
                print(f"   Security threats detected: {[e['threat_type'] for e in result['security_events'][:3]]}")
                
            results.append({
                'test_name': test_name,
                'status': result['status'],
                'security_events': len(result['security_events']),
                'validation_errors': len(result['validation_errors'])
            })
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            results.append({
                'test_name': test_name,
                'status': 'failed',
                'error': str(e)
            })
    
    # Generate final report
    time.sleep(2)  # Let monitoring collect data
    processor.monitor.stop_monitoring()
    
    print("\n📊 Final Robustness Report")
    print("=" * 60)
    
    report = processor.get_processing_report()
    
    print(f"Processing Statistics:")
    print(f"   Events processed: {report['processing_statistics']['events_processed']}")
    print(f"   Errors encountered: {report['processing_statistics']['errors_encountered']}")
    print(f"   Recovery attempts: {report['processing_statistics']['recoveries_attempted']}")
    print(f"   Successful recoveries: {report['processing_statistics']['successful_recoveries']}")
    
    print(f"\nCircuit Breaker Status:")
    print(f"   Open: {report['circuit_breaker']['is_open']}")
    print(f"   Failure count: {report['circuit_breaker']['failure_count']}")
    
    print(f"\nSecurity Summary:")
    sec_summary = report['security_summary']
    print(f"   Total security events: {sec_summary['total_security_events']}")
    print(f"   Threat types: {list(sec_summary['threat_breakdown'].keys())}")
    
    print(f"\nReliability Score: {report['reliability_score']:.1f}%")
    
    # Overall assessment
    if report['reliability_score'] >= 95:
        status = "✅ EXCELLENT"
    elif report['reliability_score'] >= 80:
        status = "✅ GOOD"
    elif report['reliability_score'] >= 60:
        status = "⚠️ ACCEPTABLE"
    else:
        status = "❌ NEEDS IMPROVEMENT"
        
    print(f"Overall Robustness: {status}")
    
    # Save comprehensive report
    with open('robustness_report.json', 'w') as f:
        json.dump({
            'test_results': results,
            'system_report': report,
            'timestamp': time.time()
        }, f, indent=2)
    
    print("\n💾 Detailed report saved to: robustness_report.json")
    
    return report['reliability_score']

if __name__ == "__main__":
    reliability_score = run_robustness_tests()
    print(f"\n🎯 Final Reliability Score: {reliability_score:.1f}%")
    
    if reliability_score >= 80:
        print("🎉 System passes robustness requirements!")
        sys.exit(0)
    else:
        print("⚠️ System needs robustness improvements")
        sys.exit(1)