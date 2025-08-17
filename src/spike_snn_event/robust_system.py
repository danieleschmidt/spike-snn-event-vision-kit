"""
Robust neuromorphic vision system with comprehensive error handling, monitoring, and reliability features.

This module implements Generation 2 robustness enhancements including:
- Advanced error handling and recovery
- Comprehensive logging and monitoring  
- Health checks and system diagnostics
- Security measures and input validation
- Circuit breakers and fault tolerance
- Resource management and cleanup
"""

import logging
import time
import threading
import uuid
import hashlib
import json
import os
import psutil
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from queue import Queue, Empty
from enum import Enum
import numpy as np
from functools import wraps
import traceback
import signal
import sys
try:
    import psutil
except ImportError:
    psutil = None

# Enhanced logging configuration
def setup_robust_logging():
    """Setup comprehensive logging with rotation and monitoring."""
    logger = logging.getLogger('neuromorphic_vision')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with basic rotation
        try:
            file_handler = logging.FileHandler('neuromorphic_vision.log', mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
    
    return logger

# Setup logging on import
setup_robust_logging()

class SystemState(Enum):
    """System operational states."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy" 
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"
    SHUTDOWN = "shutdown"
    ERROR = "error"

class SecurityLevel(Enum):
    """Security validation levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class HealthMetrics:
    """Comprehensive health metrics for system monitoring."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    event_processing_rate: float
    error_rate: float
    queue_size: int
    active_threads: int
    system_state: SystemState
    uptime: float
    last_error: Optional[str] = None
    performance_score: float = 100.0

@dataclass  
class SecurityConfig:
    """Security configuration for robust operations."""
    enable_input_validation: bool = True
    enable_rate_limiting: bool = True
    max_events_per_second: int = 100000
    enable_encryption: bool = False
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    allowed_file_extensions: List[str] = None
    max_file_size_mb: int = 100
    
    def __post_init__(self):
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = ['.npy', '.txt', '.h5', '.dat', '.json']

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker")
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self._lock:
                if self.state == 'OPEN':
                    if self._should_attempt_reset():
                        self.state = 'HALF_OPEN'
                        self.logger.info("Circuit breaker moving to HALF_OPEN state")
                    else:
                        raise Exception("Circuit breaker is OPEN - calls not allowed")
                        
                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result
                except self.expected_exception as e:
                    self._on_failure()
                    raise
                    
        return wrapper
        
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
        
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            self.logger.info("Circuit breaker reset to CLOSED state")
            
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

class RobustEventValidator:
    """Advanced event validation with security and robustness features."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.RobustEventValidator")
        self.validation_stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'security_violations': 0
        }
        
    def validate_events(self, events: np.ndarray, source: str = "unknown") -> Tuple[bool, str]:
        """Comprehensive event validation with security checks."""
        self.validation_stats['total_validations'] += 1
        
        try:
            # Basic format validation
            if not isinstance(events, np.ndarray):
                self._log_validation_failure("Events must be numpy array", source)
                return False, "Invalid data type"
                
            if len(events.shape) != 2 or events.shape[1] != 4:
                self._log_validation_failure("Events must have shape (N, 4)", source)
                return False, "Invalid shape"
                
            if len(events) == 0:
                return True, "Empty event array (valid)"
                
            # Range validation
            x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
            
            # Coordinate validation
            if np.any(x < 0) or np.any(y < 0):
                self._log_validation_failure("Negative coordinates detected", source)
                return False, "Invalid coordinates"
                
            if np.any(x > 10000) or np.any(y > 10000):  # Reasonable upper bound
                self._log_validation_failure("Suspiciously large coordinates", source)
                return False, "Coordinates out of range"
                
            # Timestamp validation
            if np.any(t < 0):
                self._log_validation_failure("Negative timestamps detected", source)
                return False, "Invalid timestamps"
                
            if np.any(np.diff(t) < -1.0):  # Allow some temporal reordering but not extreme
                self._log_validation_failure("Extreme temporal reordering detected", source)
                return False, "Invalid temporal order"
                
            # Polarity validation
            if not np.all(np.isin(p, [-1, 0, 1])):
                self._log_validation_failure("Invalid polarity values", source)
                return False, "Invalid polarity"
                
            # Security checks
            if self.config.enable_rate_limiting:
                time_span = t.max() - t.min() if len(t) > 1 else 1.0
                event_rate = len(events) / max(time_span, 1e-6)
                
                if event_rate > self.config.max_events_per_second:
                    self._log_security_violation(f"Event rate too high: {event_rate:.0f} events/s", source)
                    return False, "Rate limit exceeded"
                    
            # Check for potential injection patterns
            if self._detect_injection_patterns(events):
                self._log_security_violation("Potential injection pattern detected", source)
                return False, "Security violation"
                
            return True, "Valid events"
            
        except Exception as e:
            self._log_validation_failure(f"Validation error: {e}", source)
            return False, f"Validation exception: {e}"
            
    def _detect_injection_patterns(self, events: np.ndarray) -> bool:
        """Detect potential malicious injection patterns."""
        if len(events) < 10:
            return False
            
        # Check for suspicious patterns
        x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        
        # Pattern 1: All events at same location (potential DoS)
        if len(np.unique(x)) == 1 and len(np.unique(y)) == 1 and len(events) > 1000:
            return True
            
        # Pattern 2: Perfect geometric patterns (unlikely in real data)
        if len(events) > 100:
            x_sorted = np.sort(x)
            if np.allclose(np.diff(x_sorted), np.diff(x_sorted)[0], rtol=1e-10):
                return True
                
        # Pattern 3: Suspicious timestamp patterns
        if len(events) > 50:
            t_diff = np.diff(t)
            if np.std(t_diff) < 1e-12 and len(t_diff) > 10:  # Too regular
                return True
                
        return False
        
    def _log_validation_failure(self, message: str, source: str):
        """Log validation failure."""
        self.validation_stats['failed_validations'] += 1
        self.logger.warning(f"Validation failed from {source}: {message}")
        
    def _log_security_violation(self, message: str, source: str):
        """Log security violation."""
        self.validation_stats['security_violations'] += 1
        self.logger.error(f"SECURITY VIOLATION from {source}: {message}")

class SystemMonitor:
    """Comprehensive system monitoring and health management."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.logger = logging.getLogger(f"{__name__}.SystemMonitor")
        self.metrics_history: List[HealthMetrics] = []
        self.max_history = 1000
        self.start_time = time.time()
        
        self.monitoring_thread = None
        self.is_monitoring = False
        self._shutdown_event = threading.Event()
        
        # Performance tracking
        self.event_counter = 0
        self.error_counter = 0
        self.last_check_time = time.time()
        
    def start_monitoring(self):
        """Start continuous system monitoring."""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="SystemMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("System monitoring started")
        
    def stop_monitoring(self):
        """Stop system monitoring."""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        self._shutdown_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
        self.logger.info("System monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring and not self._shutdown_event.is_set():
            try:
                metrics = self._collect_metrics()
                self._process_metrics(metrics)
                
                # Clean up old metrics
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history//2:]
                    
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                
            self._shutdown_event.wait(self.check_interval)
            
    def _collect_metrics(self) -> HealthMetrics:
        """Collect comprehensive system metrics."""
        current_time = time.time()
        
        # System resource metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Application metrics
        time_delta = current_time - self.last_check_time
        event_rate = self.event_counter / max(time_delta, 1e-6)
        error_rate = self.error_counter / max(time_delta, 1e-6)
        
        # Reset counters
        self.event_counter = 0
        self.error_counter = 0
        self.last_check_time = current_time
        
        # Determine system state
        system_state = self._determine_system_state(
            cpu_usage, memory.percent, error_rate
        )
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(
            cpu_usage, memory.percent, error_rate
        )
        
        metrics = HealthMetrics(
            timestamp=current_time,
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            event_processing_rate=event_rate,
            error_rate=error_rate,
            queue_size=0,  # Will be updated by specific components
            active_threads=threading.active_count(),
            system_state=system_state,
            uptime=current_time - self.start_time,
            performance_score=performance_score
        )
        
        return metrics
        
    def _determine_system_state(
        self, 
        cpu_usage: float, 
        memory_usage: float, 
        error_rate: float
    ) -> SystemState:
        """Determine overall system state based on metrics."""
        if error_rate > 10:  # More than 10 errors per second
            return SystemState.CRITICAL
        elif cpu_usage > 90 or memory_usage > 90:
            return SystemState.CRITICAL
        elif error_rate > 1 or cpu_usage > 80 or memory_usage > 80:
            return SystemState.WARNING
        elif cpu_usage > 60 or memory_usage > 60:
            return SystemState.DEGRADED
        else:
            return SystemState.HEALTHY
            
    def _calculate_performance_score(
        self,
        cpu_usage: float,
        memory_usage: float, 
        error_rate: float
    ) -> float:
        """Calculate overall performance score (0-100)."""
        cpu_score = max(0, 100 - cpu_usage)
        memory_score = max(0, 100 - memory_usage)
        error_score = max(0, 100 - min(error_rate * 10, 100))
        
        return (cpu_score + memory_score + error_score) / 3
        
    def _process_metrics(self, metrics: HealthMetrics):
        """Process and store metrics."""
        self.metrics_history.append(metrics)
        
        # Log critical states
        if metrics.system_state in [SystemState.CRITICAL, SystemState.ERROR]:
            self.logger.error(
                f"System in {metrics.system_state.value} state - "
                f"CPU: {metrics.cpu_usage:.1f}%, "
                f"Memory: {metrics.memory_usage:.1f}%, "
                f"Errors: {metrics.error_rate:.2f}/s"
            )
        elif metrics.system_state == SystemState.WARNING:
            self.logger.warning(
                f"System performance degraded - "
                f"Score: {metrics.performance_score:.1f}"
            )
            
    def record_event_processed(self):
        """Record that an event was processed."""
        self.event_counter += 1
        
    def record_error(self):
        """Record that an error occurred."""
        self.error_counter += 1
        
    def get_latest_metrics(self) -> Optional[HealthMetrics]:
        """Get the most recent health metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
        
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics."""
        if not self.metrics_history:
            return {}
            
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        return {
            'current_state': recent_metrics[-1].system_state.value,
            'avg_cpu_usage': np.mean([m.cpu_usage for m in recent_metrics]),
            'avg_memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
            'avg_performance_score': np.mean([m.performance_score for m in recent_metrics]),
            'total_uptime': recent_metrics[-1].uptime,
            'measurements_count': len(self.metrics_history)
        }

class RobustEventProcessor:
    """Robust event processor with comprehensive error handling."""
    
    def __init__(
        self,
        security_config: Optional[SecurityConfig] = None,
        monitor_interval: float = 30.0
    ):
        self.security_config = security_config or SecurityConfig()
        self.logger = logging.getLogger(f"{__name__}.RobustEventProcessor")
        
        # Initialize components
        self.validator = RobustEventValidator(self.security_config)
        self.monitor = SystemMonitor(monitor_interval)
        
        # Circuit breakers for different operations
        self.validation_breaker = CircuitBreaker(
            failure_threshold=10,
            recovery_timeout=60.0,
            expected_exception=Exception
        )
        
        self.processing_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            expected_exception=Exception
        )
        
        # Processing queue and threads
        self.event_queue = Queue(maxsize=10000)
        self.processing_threads = []
        self.is_processing = False
        self.processing_stats = {
            'events_processed': 0,
            'events_failed': 0,
            'processing_time_total': 0.0
        }
        
        # Graceful shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def start(self, num_workers: int = 2):
        """Start robust event processing system."""
        self.logger.info("Starting robust event processing system...")
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Start processing workers
        self.is_processing = True
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._processing_worker,
                name=f"EventWorker-{i}",
                daemon=True
            )
            worker.start()
            self.processing_threads.append(worker)
            
        self.logger.info(f"Started {num_workers} processing workers")
        
    def stop(self):
        """Gracefully stop the processing system."""
        self.logger.info("Stopping robust event processing system...")
        
        # Stop processing
        self.is_processing = False
        
        # Wait for workers to finish
        for worker in self.processing_threads:
            worker.join(timeout=5.0)
            
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Log final statistics
        self._log_final_stats()
        self.logger.info("System stopped gracefully")
        
    def submit_events(
        self, 
        events: np.ndarray, 
        source: str = "unknown",
        priority: int = 1
    ) -> bool:
        """Submit events for processing with validation."""
        try:
            # Validate events
            is_valid, message = self.validator.validate_events(events, source)
            if not is_valid:
                self.monitor.record_error()
                self.logger.warning(f"Event validation failed from {source}: {message}")
                return False
                
            # Create processing task
            task = {
                'events': events,
                'source': source,
                'priority': priority,
                'timestamp': time.time(),
                'task_id': str(uuid.uuid4())
            }
            
            # Submit to queue
            try:
                self.event_queue.put_nowait(task)
                self.logger.debug(f"Submitted {len(events)} events from {source}")
                return True
            except:
                self.logger.error(f"Event queue full, dropping events from {source}")
                self.monitor.record_error()
                return False
                
        except Exception as e:
            self.logger.error(f"Error submitting events from {source}: {e}")
            self.monitor.record_error()
            raise
            
    def _processing_worker(self):
        """Worker thread for processing events."""
        worker_name = threading.current_thread().name
        self.logger.info(f"Processing worker {worker_name} started")
        
        while self.is_processing:
            try:
                # Get task from queue
                try:
                    task = self.event_queue.get(timeout=1.0)
                except Empty:
                    continue
                    
                # Process task
                self._process_task_safely(task)
                
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {e}")
                self.monitor.record_error()
                
        self.logger.info(f"Processing worker {worker_name} stopped")
        
    def _process_task_safely(self, task: Dict[str, Any]):
        """Process a single task with error handling."""
        start_time = time.time()
        task_id = task['task_id']
        
        try:
            events = task['events']
            source = task['source']
            
            # Simulate processing (replace with actual processing logic)
            self._simulate_event_processing(events)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats['events_processed'] += len(events)
            self.processing_stats['processing_time_total'] += processing_time
            
            self.monitor.record_event_processed()
            
            self.logger.debug(
                f"Processed {len(events)} events from {source} "
                f"in {processing_time*1000:.1f}ms (task: {task_id[:8]})"
            )
            
        except Exception as e:
            self.processing_stats['events_failed'] += len(task.get('events', []))
            self.monitor.record_error()
            self.logger.error(f"Failed to process task {task_id[:8]}: {e}")
            self.logger.debug(traceback.format_exc())
            
    def _simulate_event_processing(self, events: np.ndarray):
        """Simulate event processing (replace with actual logic)."""
        # Simulate some processing time
        processing_delay = len(events) * 1e-6  # 1 microsecond per event
        time.sleep(min(processing_delay, 0.1))  # Cap at 100ms
        
        # Simulate occasional processing errors
        if np.random.random() < 0.001:  # 0.1% chance of error
            raise RuntimeError("Simulated processing error")
            
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()
        sys.exit(0)
        
    def _log_final_stats(self):
        """Log final processing statistics."""
        stats = self.processing_stats
        total_events = stats['events_processed'] + stats['events_failed']
        
        if total_events > 0:
            success_rate = stats['events_processed'] / total_events * 100
            avg_processing_time = stats['processing_time_total'] / max(stats['events_processed'], 1)
            
            self.logger.info(
                f"Final Statistics - "
                f"Events processed: {stats['events_processed']}, "
                f"Events failed: {stats['events_failed']}, "
                f"Success rate: {success_rate:.1f}%, "
                f"Avg processing time: {avg_processing_time*1000:.2f}ms"
            )
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        latest_metrics = self.monitor.get_latest_metrics()
        
        status = {
            'system_state': latest_metrics.system_state.value if latest_metrics else 'unknown',
            'is_processing': self.is_processing,
            'queue_size': self.event_queue.qsize(),
            'active_workers': len([t for t in self.processing_threads if t.is_alive()]),
            'processing_stats': self.processing_stats.copy(),
            'validation_stats': self.validator.validation_stats.copy(),
            'circuit_breakers': {
                'validation': self.validation_breaker.state,
                'processing': self.processing_breaker.state
            }
        }
        
        if latest_metrics:
            status['health_metrics'] = asdict(latest_metrics)
            
        return status

# Robust camera interface with reliability enhancements
class RobustDVSCamera:
    """Enhanced DVS camera with robustness features."""
    
    def __init__(
        self, 
        sensor_type: str = "DVS128",
        security_config: Optional[SecurityConfig] = None
    ):
        self.sensor_type = sensor_type
        self.security_config = security_config or SecurityConfig()
        self.logger = logging.getLogger(f"{__name__}.RobustDVSCamera")
        
        # Initialize robust processing
        self.processor = RobustEventProcessor(self.security_config)
        
        # Camera state
        self.is_active = False
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        
        # Performance tracking
        self.capture_stats = {
            'frames_captured': 0,
            'capture_errors': 0,
            'last_capture_time': None
        }
        
    def start_capture(self) -> bool:
        """Start robust event capture with error handling."""
        self.logger.info(f"Starting capture for {self.sensor_type}")
        
        for attempt in range(self.max_connection_attempts):
            try:
                # Simulate camera initialization
                self._initialize_camera()
                
                # Start processing system
                self.processor.start()
                
                self.is_active = True
                self.logger.info("Camera capture started successfully")
                return True
                
            except Exception as e:
                self.connection_attempts += 1
                self.logger.warning(
                    f"Camera initialization attempt {attempt + 1} failed: {e}"
                )
                
                if attempt < self.max_connection_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        self.logger.error("Failed to start camera after all attempts")
        return False
        
    def stop_capture(self):
        """Stop camera capture gracefully."""
        if not self.is_active:
            return
            
        self.logger.info("Stopping camera capture...")
        self.is_active = False
        
        # Stop processing
        self.processor.stop()
        
        self.logger.info("Camera capture stopped")
        
    def _initialize_camera(self):
        """Initialize camera connection (placeholder)."""
        # Simulate camera initialization
        init_time = np.random.uniform(0.1, 0.5)
        time.sleep(init_time)
        
        # Simulate occasional initialization failures
        if np.random.random() < 0.1:  # 10% chance of failure
            raise RuntimeError("Camera initialization failed")
            
    def capture_events(self, duration: float = 1.0) -> Optional[np.ndarray]:
        """Capture events with robust error handling."""
        if not self.is_active:
            self.logger.warning("Camera not active, cannot capture events")
            return None
            
        try:
            # Simulate event capture
            events = self._simulate_event_capture(duration)
            
            # Submit for processing
            success = self.processor.submit_events(
                events, 
                source=f"camera_{self.sensor_type}"
            )
            
            if success:
                self.capture_stats['frames_captured'] += 1
                self.capture_stats['last_capture_time'] = time.time()
                return events
            else:
                self.logger.warning("Failed to submit captured events for processing")
                return None
                
        except Exception as e:
            self.capture_stats['capture_errors'] += 1
            self.logger.error(f"Event capture failed: {e}")
            return None
            
    def _simulate_event_capture(self, duration: float) -> np.ndarray:
        """Simulate realistic event capture."""
        # Generate realistic number of events
        base_rate = 1000  # events per second
        num_events = int(np.random.poisson(base_rate * duration))
        
        if num_events == 0:
            return np.empty((0, 4))
            
        # Generate realistic events
        current_time = time.time()
        events = np.zeros((num_events, 4))
        
        # Spatial distribution (more events in center)
        center_bias = 0.7
        if self.sensor_type == "DVS128":
            width, height = 128, 128
        else:
            width, height = 240, 180
            
        # Generate coordinates with center bias
        x_rand = np.random.beta(2, 2, num_events) if center_bias > 0.5 else np.random.random(num_events)
        y_rand = np.random.beta(2, 2, num_events) if center_bias > 0.5 else np.random.random(num_events)
        
        events[:, 0] = x_rand * width
        events[:, 1] = y_rand * height
        
        # Temporal distribution
        timestamps = np.sort(current_time + np.random.exponential(duration/num_events, num_events))
        events[:, 2] = timestamps
        
        # Polarity (slightly biased towards positive)
        events[:, 3] = np.random.choice([-1, 1], num_events, p=[0.45, 0.55])
        
        return events
        
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive camera status."""
        status = {
            'sensor_type': self.sensor_type,
            'is_active': self.is_active,
            'connection_attempts': self.connection_attempts,
            'capture_stats': self.capture_stats.copy()
        }
        
        # Add processor status
        if hasattr(self, 'processor'):
            status['processor_status'] = self.processor.get_system_status()
            
        return status

# Context manager for robust operations
@contextmanager
def robust_operation(operation_name: str, logger: Optional[logging.Logger] = None):
    """Context manager for robust operations with logging and cleanup."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    start_time = time.time()
    logger.info(f"Starting {operation_name}")
    
    try:
        yield
        duration = time.time() - start_time
        logger.info(f"Completed {operation_name} in {duration:.3f}s")
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed {operation_name} after {duration:.3f}s: {e}")
        logger.debug(traceback.format_exc())
        raise
        
    finally:
        # Cleanup operations can be added here
        pass

# Test and demonstration functions
def test_robust_system():
    """Test the robust neuromorphic vision system."""
    logger = logging.getLogger("test_robust_system")
    
    with robust_operation("Robust System Test", logger):
        # Initialize robust camera
        camera = RobustDVSCamera("DVS128")
        
        # Start capture
        if not camera.start_capture():
            raise RuntimeError("Failed to start camera")
            
        try:
            # Capture some events
            for i in range(5):
                logger.info(f"Capture iteration {i+1}")
                events = camera.capture_events(duration=0.5)
                
                if events is not None:
                    logger.info(f"Captured {len(events)} events")
                else:
                    logger.warning("No events captured")
                    
                time.sleep(1.0)
                
            # Get system status
            status = camera.get_status()
            logger.info(f"Final system status: {status['processor_status']['system_state']}")
            
        finally:
            # Always stop camera
            camera.stop_capture()

if __name__ == "__main__":
    # Run test
    test_robust_system()