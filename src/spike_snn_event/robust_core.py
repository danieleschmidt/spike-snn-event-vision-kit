"""
Robust core functionality with enterprise-grade error handling and resilience.

Generation 2: MAKE IT ROBUST
- Circuit breaker patterns
- Comprehensive error recovery
- Health monitoring and alerting
- Thread-safe operations
- Graceful shutdown handling
"""

import time
import threading
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import functools
import traceback
from contextlib import contextmanager


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failure state - rejecting requests
    HALF_OPEN = "half_open"  # Testing state - allowing limited requests


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    timeout: float = 60.0       # Seconds before attempting reset
    success_threshold: int = 3   # Successes to close from half-open


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count = 0
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.config.timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self._reset()
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _on_success(self):
        """Handle successful operation."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._reset()
            else:
                self.failure_count = 0
    
    def _on_failure(self, error: Exception):
        """Handle failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.logger.error(f"Circuit breaker OPENED after {self.failure_count} failures: {error}")
    
    def _reset(self):
        """Reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.logger.info("Circuit breaker CLOSED - normal operation resumed")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time
        }


class HealthMonitor:
    """System health monitoring and alerting."""
    
    def __init__(self):
        self.checks = {}
        self.alerts = []
        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._monitor_thread = None
        
    def register_check(self, name: str, check_func: Callable[[], bool], interval: float = 60.0):
        """Register a health check."""
        self.checks[name] = {
            'func': check_func,
            'interval': interval,
            'last_check': 0,
            'status': 'unknown',
            'last_error': None
        }
        
    def start_monitoring(self):
        """Start health monitoring in background thread."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Health monitoring started")
        
    def stop_monitoring(self):
        """Stop health monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Health monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            current_time = time.time()
            
            for name, check in self.checks.items():
                if current_time - check['last_check'] >= check['interval']:
                    try:
                        result = check['func']()
                        check['status'] = 'healthy' if result else 'unhealthy'
                        check['last_error'] = None
                        
                        if not result:
                            self._trigger_alert(name, "Health check failed")
                            
                    except Exception as e:
                        check['status'] = 'error'
                        check['last_error'] = str(e)
                        self._trigger_alert(name, f"Health check error: {e}")
                        
                    check['last_check'] = current_time
            
            time.sleep(10)  # Check every 10 seconds
            
    def _trigger_alert(self, check_name: str, message: str):
        """Trigger health alert."""
        alert = {
            'timestamp': time.time(),
            'check': check_name,
            'message': message,
            'severity': 'warning'
        }
        
        self.alerts.append(alert)
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts.pop(0)
            
        self.logger.warning(f"Health alert: {check_name} - {message}")
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        healthy_count = sum(1 for c in self.checks.values() if c['status'] == 'healthy')
        total_count = len(self.checks)
        
        overall_status = 'healthy' if healthy_count == total_count else 'degraded'
        if healthy_count == 0 and total_count > 0:
            overall_status = 'unhealthy'
            
        return {
            'overall': overall_status,
            'healthy_checks': healthy_count,
            'total_checks': total_count,
            'checks': {name: {'status': check['status'], 'last_error': check['last_error']} 
                      for name, check in self.checks.items()},
            'recent_alerts': self.alerts[-10:]  # Last 10 alerts
        }


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logging.error(f"Function {func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    
                    logging.warning(f"Attempt {attempt} failed for {func.__name__}: {e}, retrying in {current_delay}s")
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
                    
        return wrapper
    return decorator


def timeout(seconds: float):
    """Timeout decorator for functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = []
            exception = []
            
            def target():
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    exception.append(e)
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                # Can't actually kill the thread, but we can timeout
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            if exception:
                raise exception[0]
            
            return result[0] if result else None
            
        return wrapper
    return decorator


class GracefulShutdown:
    """Graceful shutdown handler for clean system exit."""
    
    def __init__(self):
        self.shutdown_handlers = []
        self.is_shutting_down = False
        self.logger = logging.getLogger(__name__)
        
    def register_handler(self, handler: Callable[[], None], priority: int = 0):
        """Register shutdown handler with priority (higher = earlier)."""
        self.shutdown_handlers.append((priority, handler))
        self.shutdown_handlers.sort(key=lambda x: x[0], reverse=True)
        
    def shutdown(self):
        """Execute graceful shutdown."""
        if self.is_shutting_down:
            return
            
        self.is_shutting_down = True
        self.logger.info("Initiating graceful shutdown...")
        
        for priority, handler in self.shutdown_handlers:
            try:
                self.logger.debug(f"Executing shutdown handler (priority {priority})")
                handler()
            except Exception as e:
                self.logger.error(f"Shutdown handler failed: {e}")
                
        self.logger.info("Graceful shutdown completed")


class RobustEventProcessor:
    """Event processor with enhanced robustness features."""
    
    def __init__(self, camera_class=None):
        self.camera_class = camera_class
        self.camera = None
        self.circuit_breaker = CircuitBreaker()
        self.health_monitor = HealthMonitor()
        self.shutdown_handler = GracefulShutdown()
        self.logger = logging.getLogger(__name__)
        
        # Setup health checks
        self.health_monitor.register_check(
            "camera_connection", 
            self._check_camera_health, 
            interval=30.0
        )
        
        # Setup shutdown handlers
        self.shutdown_handler.register_handler(self._cleanup_camera, priority=10)
        self.shutdown_handler.register_handler(self._stop_monitoring, priority=5)
        
    def initialize(self, camera_config: Dict[str, Any]):
        """Initialize with robust error handling."""
        try:
            self.logger.info("Initializing robust event processor...")
            
            if self.camera_class:
                self.camera = self.circuit_breaker.call(
                    self.camera_class, **camera_config
                )
            
            self.health_monitor.start_monitoring()
            self.logger.info("Robust event processor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize event processor: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def process_events(self, events):
        """Process events with retry logic."""
        if not events or len(events) == 0:
            return []
            
        # Validate events
        if not self._validate_events(events):
            raise ValueError("Invalid event data")
            
        # Process with circuit breaker protection
        return self.circuit_breaker.call(self._do_process_events, events)
        
    def _do_process_events(self, events):
        """Internal event processing implementation."""
        # Simulate processing
        processed_events = []
        for event in events:
            # Basic processing
            processed_event = {
                'x': event.get('x', 0),
                'y': event.get('y', 0), 
                'timestamp': event.get('timestamp', time.time()),
                'polarity': event.get('polarity', 1),
                'processed_at': time.time()
            }
            processed_events.append(processed_event)
            
        return processed_events
    
    def _validate_events(self, events) -> bool:
        """Validate event data structure."""
        if not isinstance(events, (list, tuple)):
            return False
            
        for event in events:
            if not isinstance(event, dict):
                return False
            if 'x' not in event or 'y' not in event:
                return False
                
        return True
    
    def _check_camera_health(self) -> bool:
        """Check camera health status."""
        if not self.camera:
            return False
            
        try:
            # Attempt to get camera status
            if hasattr(self.camera, 'health_check'):
                status = self.camera.health_check()
                return status.get('status') == 'healthy'
            return True
        except Exception as e:
            self.logger.error(f"Camera health check failed: {e}")
            return False
    
    def _cleanup_camera(self):
        """Cleanup camera resources."""
        if self.camera and hasattr(self.camera, 'stop_streaming'):
            try:
                self.camera.stop_streaming()
                self.logger.info("Camera streaming stopped")
            except Exception as e:
                self.logger.error(f"Failed to stop camera streaming: {e}")
    
    def _stop_monitoring(self):
        """Stop health monitoring."""
        self.health_monitor.stop_monitoring()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'circuit_breaker': self.circuit_breaker.get_state(),
            'health': self.health_monitor.get_health_status(),
            'is_shutting_down': self.shutdown_handler.is_shutting_down,
            'camera_initialized': self.camera is not None
        }
    
    def shutdown(self):
        """Gracefully shutdown the processor."""
        self.shutdown_handler.shutdown()


@contextmanager
def error_handling(operation_name: str, logger: logging.Logger = None):
    """Context manager for consistent error handling."""
    if not logger:
        logger = logging.getLogger(__name__)
        
    try:
        logger.debug(f"Starting operation: {operation_name}")
        yield
        logger.debug(f"Operation completed: {operation_name}")
        
    except Exception as e:
        logger.error(f"Operation failed: {operation_name} - {e}")
        logger.error(traceback.format_exc())
        raise


# Example usage and demonstration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example of robust event processing
    processor = RobustEventProcessor()
    
    try:
        processor.initialize({})
        
        # Simulate event processing
        test_events = [
            {'x': 10, 'y': 20, 'timestamp': time.time(), 'polarity': 1},
            {'x': 15, 'y': 25, 'timestamp': time.time(), 'polarity': -1}
        ]
        
        with error_handling("event_processing"):
            results = processor.process_events(test_events)
            print(f"Processed {len(results)} events successfully")
            
        # Check system status
        status = processor.get_status()
        print(f"System status: {status}")
        
    finally:
        processor.shutdown()
        print("Processor shutdown completed")