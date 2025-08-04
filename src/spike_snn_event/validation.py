"""
Input validation and error handling utilities for spike-snn-event-vision-kit.

Provides comprehensive validation functions and custom exceptions
for robust error handling throughout the system.
"""

import numpy as np
import torch
from typing import Union, Tuple, List, Dict, Any, Optional
from pathlib import Path
import logging
from functools import wraps
import time


class SpikeNNError(Exception):
    """Base exception for spike-snn-event-vision-kit."""
    pass


class ValidationError(SpikeNNError):
    """Raised when input validation fails."""
    pass


class ModelError(SpikeNNError):
    """Raised when model operations fail."""
    pass


class DataError(SpikeNNError):
    """Raised when data processing fails."""
    pass


class ConfigurationError(SpikeNNError):
    """Raised when configuration is invalid."""
    pass


class HardwareError(SpikeNNError):
    """Raised when hardware operations fail."""
    pass


def validate_events(events: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Validate event array format and content.
    
    Args:
        events: Event array with shape [N, 4] containing [x, y, timestamp, polarity]
        
    Returns:
        Validated numpy array
        
    Raises:
        ValidationError: If events format is invalid
    """
    if events is None:
        raise ValidationError("Events cannot be None")
        
    # Convert to numpy if torch tensor
    if isinstance(events, torch.Tensor):
        events = events.detach().cpu().numpy()
        
    if not isinstance(events, np.ndarray):
        try:
            events = np.array(events)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Cannot convert events to numpy array: {e}")
    
    # Check shape
    if events.ndim != 2:
        raise ValidationError(f"Events must be 2D array, got {events.ndim}D")
        
    if events.shape[1] != 4:
        raise ValidationError(f"Events must have 4 columns [x, y, t, p], got {events.shape[1]}")
    
    # Check for invalid values
    if len(events) > 0:
        # Check for NaN or infinite values
        if np.any(~np.isfinite(events)):
            raise ValidationError("Events contain NaN or infinite values")
            
        # Check coordinate ranges (basic sanity check)
        if np.any(events[:, 0] < 0) or np.any(events[:, 1] < 0):
            raise ValidationError("Event coordinates cannot be negative")
            
        if np.any(events[:, 0] > 10000) or np.any(events[:, 1] > 10000):
            raise ValidationError("Event coordinates are unreasonably large")
            
        # Check timestamp ordering (should be non-decreasing)
        if len(events) > 1:
            timestamps = events[:, 2]
            if not np.all(timestamps[1:] >= timestamps[:-1]):
                # Sort events by timestamp if not ordered
                sorted_indices = np.argsort(timestamps)
                events = events[sorted_indices]
                logging.warning("Events were not time-ordered, sorted automatically")
                
        # Check polarity values
        unique_polarities = np.unique(events[:, 3])
        valid_polarities = {-1, 0, 1}
        if not set(unique_polarities).issubset(valid_polarities):
            raise ValidationError(f"Invalid polarity values: {unique_polarities}. Must be in {valid_polarities}")
    
    return events


def validate_image_dimensions(
    width: int, 
    height: int, 
    min_size: int = 1, 
    max_size: int = 10000
) -> Tuple[int, int]:
    """
    Validate image dimensions.
    
    Args:
        width: Image width
        height: Image height
        min_size: Minimum allowed dimension
        max_size: Maximum allowed dimension
        
    Returns:
        Validated (width, height) tuple
        
    Raises:
        ValidationError: If dimensions are invalid
    """
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValidationError("Width and height must be integers")
        
    if width < min_size or height < min_size:
        raise ValidationError(f"Dimensions must be at least {min_size}x{min_size}")
        
    if width > max_size or height > max_size:
        raise ValidationError(f"Dimensions cannot exceed {max_size}x{max_size}")
        
    return width, height


def validate_time_window(time_window: float) -> float:
    """
    Validate time window parameter.
    
    Args:
        time_window: Time window in seconds
        
    Returns:
        Validated time window
        
    Raises:
        ValidationError: If time window is invalid
    """
    if not isinstance(time_window, (int, float)):
        raise ValidationError("Time window must be a number")
        
    if time_window <= 0:
        raise ValidationError("Time window must be positive")
        
    if time_window > 10.0:  # 10 second max
        raise ValidationError("Time window cannot exceed 10 seconds")
        
    return float(time_window)


def validate_threshold(threshold: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Validate threshold parameter.
    
    Args:
        threshold: Threshold value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Validated threshold
        
    Raises:
        ValidationError: If threshold is invalid
    """
    if not isinstance(threshold, (int, float)):
        raise ValidationError("Threshold must be a number")
        
    if threshold < min_val or threshold > max_val:
        raise ValidationError(f"Threshold must be between {min_val} and {max_val}")
        
    return float(threshold)


def validate_model_input(
    input_tensor: torch.Tensor,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Validate model input tensor.
    
    Args:
        input_tensor: Input tensor to validate
        expected_shape: Expected tensor shape (None values ignored)
        expected_dtype: Expected tensor dtype
        
    Returns:
        Validated tensor
        
    Raises:
        ValidationError: If input is invalid
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise ValidationError("Input must be a torch.Tensor")
        
    # Check for NaN or infinite values
    if torch.any(~torch.isfinite(input_tensor)):
        raise ValidationError("Input tensor contains NaN or infinite values")
        
    # Check shape
    if expected_shape is not None:
        actual_shape = input_tensor.shape
        if len(actual_shape) != len(expected_shape):
            raise ValidationError(
                f"Expected {len(expected_shape)}D tensor, got {len(actual_shape)}D"
            )
            
        for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            if expected is not None and actual != expected:
                raise ValidationError(
                    f"Dimension {i}: expected {expected}, got {actual}"
                )
    
    # Check dtype
    if expected_dtype is not None and input_tensor.dtype != expected_dtype:
        raise ValidationError(
            f"Expected dtype {expected_dtype}, got {input_tensor.dtype}"
        )
    
    return input_tensor


def validate_file_path(
    file_path: Union[str, Path],
    must_exist: bool = False,
    must_be_file: bool = False,
    allowed_extensions: Optional[List[str]] = None
) -> Path:
    """
    Validate file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        must_be_file: Whether path must be a file (not directory)
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid
    """
    if not isinstance(file_path, (str, Path)):
        raise ValidationError("File path must be string or Path object")
        
    path = Path(file_path)
    
    if must_exist and not path.exists():
        raise ValidationError(f"File does not exist: {path}")
        
    if must_be_file and path.exists() and not path.is_file():
        raise ValidationError(f"Path is not a file: {path}")
        
    if allowed_extensions:
        extension = path.suffix.lower()
        if extension not in allowed_extensions:
            raise ValidationError(
                f"Invalid file extension '{extension}'. "
                f"Allowed: {allowed_extensions}"
            )
    
    return path


def validate_device(device: Union[str, torch.device]) -> torch.device:
    """
    Validate and normalize device specification.
    
    Args:
        device: Device specification
        
    Returns:
        Validated torch.device
        
    Raises:
        ValidationError: If device is invalid
        HardwareError: If CUDA requested but not available
    """
    if isinstance(device, str):
        device = device.lower()
        if device not in ['cpu', 'cuda', 'gpu']:
            raise ValidationError(f"Invalid device string: {device}")
            
        if device in ['cuda', 'gpu']:
            if not torch.cuda.is_available():
                raise HardwareError("CUDA requested but not available")
            device = 'cuda'
            
    try:
        device = torch.device(device)
    except Exception as e:
        raise ValidationError(f"Invalid device specification: {e}")
        
    return device


def validate_config_dict(
    config: Dict[str, Any],
    required_keys: List[str],
    optional_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: List of required keys
        optional_keys: List of optional keys (if provided, extra keys raise error)
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration must be a dictionary")
        
    # Check required keys
    missing_keys = set(required_keys) - set(config.keys())
    if missing_keys:
        raise ConfigurationError(f"Missing required configuration keys: {missing_keys}")
        
    # Check for unexpected keys if optional_keys is specified
    if optional_keys is not None:
        allowed_keys = set(required_keys) | set(optional_keys)
        extra_keys = set(config.keys()) - allowed_keys
        if extra_keys:
            raise ConfigurationError(f"Unexpected configuration keys: {extra_keys}")
    
    return config


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (Exception,)
):
    """
    Decorator to retry function calls on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logging.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise
                    
                    logging.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                    
            # Should never reach here, but just in case
            raise last_exception
            
        return wrapper
    return decorator


def validate_and_sanitize_events(
    events: Union[np.ndarray, List, torch.Tensor],
    sensor_width: int = 640,
    sensor_height: int = 480,
    time_window: Optional[float] = None
) -> np.ndarray:
    """
    Comprehensive validation and sanitization of event data.
    
    Args:
        events: Raw event data
        sensor_width: Sensor width for coordinate validation
        sensor_height: Sensor height for coordinate validation
        time_window: Time window for temporal filtering
        
    Returns:
        Validated and sanitized event array
        
    Raises:
        ValidationError: If events cannot be validated/sanitized
    """
    # Basic validation
    events = validate_events(events)
    
    if len(events) == 0:
        return events
    
    # Coordinate sanitization
    events = events[
        (events[:, 0] >= 0) & (events[:, 0] < sensor_width) &
        (events[:, 1] >= 0) & (events[:, 1] < sensor_height)
    ]
    
    if len(events) == 0:
        logging.warning("All events filtered out due to coordinate bounds")
        return events
    
    # Temporal filtering
    if time_window is not None:
        time_window = validate_time_window(time_window)
        max_time = events[:, 2].max()
        min_time = max_time - time_window
        events = events[events[:, 2] >= min_time]
        
    # Sort by timestamp
    events = events[np.argsort(events[:, 2])]
    
    return events


class SafetyMonitor:
    """Monitor system safety and resource usage."""
    
    def __init__(self):
        self.max_memory_mb = 1000  # 1GB default
        self.max_events_per_batch = 100000
        self.max_processing_time_s = 30.0
        
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        try:
            import psutil
            current_process = psutil.Process()
            memory_mb = current_process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.max_memory_mb:
                logging.warning(f"High memory usage: {memory_mb:.1f}MB > {self.max_memory_mb}MB")
                return False
                
        except ImportError:
            logging.debug("psutil not available for memory monitoring")
            
        return True
        
    def check_batch_size(self, batch_size: int) -> bool:
        """Check if batch size is safe."""
        if batch_size > self.max_events_per_batch:
            logging.warning(f"Large batch size: {batch_size} > {self.max_events_per_batch}")
            return False
        return True
        
    def check_processing_time(self, start_time: float) -> bool:
        """Check if processing time is within limits."""
        elapsed = time.time() - start_time
        if elapsed > self.max_processing_time_s:
            logging.warning(f"Long processing time: {elapsed:.1f}s > {self.max_processing_time_s}s")
            return False
        return True


# Global safety monitor instance
safety_monitor = SafetyMonitor()


def safe_operation(func):
    """Decorator to monitor operations for safety."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Pre-checks
        if not safety_monitor.check_memory_usage():
            raise HardwareError("Memory usage too high")
            
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            # Log the error with context
            logging.error(f"Operation {func.__name__} failed: {e}")
            raise
        finally:
            # Post-checks
            if not safety_monitor.check_processing_time(start_time):
                logging.warning(f"Operation {func.__name__} took too long")
                
        return result
    return wrapper