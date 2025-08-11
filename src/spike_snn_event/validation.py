"""
Input validation and error handling for Spike SNN Event Vision Kit.

This module provides comprehensive validation and error handling to ensure
robust operation in production environments.
"""

import logging
import re
import os
import ipaddress
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass
import time
import hashlib
import json
from pathlib import Path
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SpikeNNError(Exception):
    """Base exception class for Spike SNN Event Vision Kit."""
    pass


class ValidationError:
    """Validation error information."""
    
    def __init__(self, code: str, message: str, field: Optional[str] = None, value: Any = None, severity: str = "error"):
        self.code = code
        self.message = message
        self.field = field
        self.value = value
        self.severity = severity


class HardwareError(SpikeNNError):
    """Raised when hardware-related errors occur."""
    pass


class DataIntegrityError(SpikeNNError):
    """Raised when data integrity checks fail."""
    pass


class CircuitBreakerError(SpikeNNError):
    """Raised when circuit breaker is open."""
    pass
    

class ValidationResult:
    """Result of validation operation."""
    
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.is_valid = True
        
    def add_error(self, code: str, message: str, field: Optional[str] = None, value: Any = None):
        """Add validation error."""
        error = ValidationError(code, message, field, value, "error")
        self.errors.append(error)
        self.is_valid = False
        
    def add_warning(self, code: str, message: str, field: Optional[str] = None, value: Any = None):
        """Add validation warning."""
        warning = ValidationError(code, message, field, value, "warning")
        self.warnings.append(warning)
        
    def get_all_issues(self) -> List[ValidationError]:
        """Get all validation issues."""
        return self.errors + self.warnings
        
    def format_errors(self) -> str:
        """Format errors for display."""
        if not self.errors and not self.warnings:
            return "No validation issues"
            
        lines = []
        
        if self.errors:
            lines.append("ERRORS:")
            for error in self.errors:
                line = f"  [{error.code}] {error.message}"
                if error.field:
                    line += f" (field: {error.field})"
                if error.value is not None:
                    line += f" (value: {error.value})"
                lines.append(line)
                
        if self.warnings:
            lines.append("WARNINGS:")
            for warning in self.warnings:
                line = f"  [{warning.code}] {warning.message}"
                if warning.field:
                    line += f" (field: {warning.field})"
                if warning.value is not None:
                    line += f" (value: {warning.value})"
                lines.append(line)
                
        return "\n".join(lines)


class SecurityValidator:
    """Security-focused input validation."""
    
    # Common malicious patterns
    SQL_INJECTION_PATTERNS = [
        r"(\s*(union|select|insert|update|delete|drop|create|alter|exec|execute)\s+)",
        r"(\s*;\s*(select|insert|update|delete|drop|create|alter)\s+)",
        r"(\s*--\s*)",
        r"(\s*/\*.*?\*/\s*)",
        r"(\s*'\s*(or|and)\s+['\d])",
    ]
    
    XSS_PATTERNS = [
        r"<\s*script[^>]*>.*?</\s*script\s*>",
        r"javascript\s*:",
        r"on\w+\s*=\s*[\"'][^\"']*[\"']",
        r"<\s*iframe[^>]*>",
        r"<\s*object[^>]*>",
        r"<\s*embed[^>]*>",
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"[\|;&`\$\(\)\{\}]",
        r"\.\./",
        r"/etc/passwd",
        r"/bin/sh",
        r"/bin/bash",
        r"cmd\.exe",
        r"powershell",
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_string_security(self, value: str, field_name: str = "input") -> ValidationResult:
        """Validate string for security threats."""
        result = ValidationResult()
        
        if not isinstance(value, str):
            result.add_error("TYPE_ERROR", f"Expected string, got {type(value).__name__}", field_name, type(value).__name__)
            return result
            
        # Check for SQL injection
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                result.add_error("SECURITY_SQL_INJECTION", "Potential SQL injection detected", field_name, value[:50])
                break
                
        # Check for XSS
        for pattern in self.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                result.add_error("SECURITY_XSS", "Potential XSS attack detected", field_name, value[:50])
                break
                
        # Check for command injection
        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value):
                result.add_error("SECURITY_COMMAND_INJECTION", "Potential command injection detected", field_name, value[:50])
                break
                
        # Check string length
        if len(value) > 10000:
            result.add_error("SIZE_LIMIT_EXCEEDED", "String too long (max 10000 characters)", field_name, len(value))
            
        # Check for excessive repeating characters (potential DoS)
        if len(set(value)) == 1 and len(value) > 100:
            result.add_error("SECURITY_DOS_PATTERN", "Excessive repeating characters detected", field_name)
            
        return result
        
    def validate_file_path(self, path: str, allowed_dirs: Optional[List[str]] = None) -> ValidationResult:
        """Validate file path for security."""
        result = ValidationResult()
        
        if not isinstance(path, str):
            result.add_error("TYPE_ERROR", f"Path must be string, got {type(path).__name__}", "path", type(path).__name__)
            return result
            
        # Check for path traversal
        if ".." in path:
            result.add_error("SECURITY_PATH_TRAVERSAL", "Path traversal detected", "path", path)
            
        # Check for absolute paths to sensitive locations
        sensitive_paths = ["/etc/", "/root/", "/home/", "/usr/bin/", "/bin/", "C:\\Windows\\", "C:\\Users\\"]
        for sensitive in sensitive_paths:
            if path.lower().startswith(sensitive.lower()):
                result.add_error("SECURITY_SENSITIVE_PATH", f"Access to sensitive path {sensitive} denied", "path", path)
                break
                
        # Check allowed directories
        if allowed_dirs:
            path_obj = Path(path)
            allowed = False
            for allowed_dir in allowed_dirs:
                try:
                    path_obj.relative_to(allowed_dir)
                    allowed = True
                    break
                except ValueError:
                    continue
            
            if not allowed:
                result.add_error("SECURITY_UNAUTHORIZED_PATH", f"Path not in allowed directories: {allowed_dirs}", "path", path)
                
        return result
        
    def validate_network_input(self, host: str, port: int = None) -> ValidationResult:
        """Validate network input for security."""
        result = ValidationResult()
        
        # Validate host
        try:
            # Check if it's an IP address
            ip = ipaddress.ip_address(host)
            
            # Block private and loopback addresses in production
            if ip.is_private or ip.is_loopback:
                result.add_warning("SECURITY_PRIVATE_IP", "Private/loopback IP address", "host", host)
                
        except ValueError:
            # It's a hostname, validate format
            if not re.match(r'^[a-zA-Z0-9.-]+$', host):
                result.add_error("SECURITY_INVALID_HOSTNAME", "Invalid hostname format", "host", host)
                
            # Check for suspicious hostnames
            suspicious_patterns = ['localhost', '127.', '10.', '192.168.', 'internal', 'admin']
            for pattern in suspicious_patterns:
                if pattern in host.lower():
                    result.add_warning("SECURITY_SUSPICIOUS_HOSTNAME", f"Suspicious hostname pattern: {pattern}", "host", host)
                    break
                    
        # Validate port
        if port is not None:
            if not isinstance(port, int):
                result.add_error("TYPE_ERROR", f"Port must be integer, got {type(port).__name__}", "port", type(port).__name__)
            elif port < 1 or port > 65535:
                result.add_error("VALUE_ERROR", "Port must be between 1 and 65535", "port", port)
            elif port in [22, 23, 135, 139, 445, 1433, 3306, 5432]:  # Sensitive ports
                result.add_warning("SECURITY_SENSITIVE_PORT", f"Connection to sensitive port {port}", "port", port)
                
        return result
        
    def validate_json_input(self, json_str: str, max_depth: int = 10) -> ValidationResult:
        """Validate JSON input for security."""
        result = ValidationResult()
        
        if not isinstance(json_str, str):
            result.add_error("TYPE_ERROR", f"JSON input must be string, got {type(json_str).__name__}", "json_input")
            return result
            
        # Check size
        if len(json_str) > 1_000_000:  # 1MB limit
            result.add_error("SIZE_LIMIT_EXCEEDED", "JSON input too large (max 1MB)", "json_input", len(json_str))
            return result
            
        try:
            data = json.loads(json_str)
            
            # Check depth
            def check_depth(obj, current_depth=0):
                if current_depth > max_depth:
                    return False
                if isinstance(obj, dict):
                    return all(check_depth(v, current_depth + 1) for v in obj.values())
                elif isinstance(obj, list):
                    return all(check_depth(item, current_depth + 1) for item in obj)
                return True
                
            if not check_depth(data):
                result.add_error("SECURITY_JSON_DEPTH", f"JSON depth exceeds limit of {max_depth}", "json_input")
                
        except json.JSONDecodeError as e:
            result.add_error("FORMAT_ERROR", f"Invalid JSON format: {e}", "json_input")
            
        return result


class DataValidator:
    """Advanced data validation for various data types."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.security_validator = SecurityValidator()
        
    def validate_numeric_range(self, value: Union[int, float], min_val: Optional[float] = None, 
                             max_val: Optional[float] = None, field_name: str = "value") -> ValidationResult:
        """Validate numeric value within range."""
        result = ValidationResult()
        
        if not isinstance(value, (int, float)):
            result.add_error("TYPE_ERROR", f"Expected numeric value, got {type(value).__name__}", field_name, type(value).__name__)
            return result
            
        if not np.isfinite(value):
            result.add_error("VALUE_ERROR", "Value is not finite (NaN or infinity)", field_name, value)
            return result
            
        if min_val is not None and value < min_val:
            result.add_error("RANGE_ERROR", f"Value {value} below minimum {min_val}", field_name, value)
            
        if max_val is not None and value > max_val:
            result.add_error("RANGE_ERROR", f"Value {value} above maximum {max_val}", field_name, value)
            
        return result
        
    def validate_array_shape(self, array: np.ndarray, expected_shape: Tuple[int, ...] = None, 
                           min_dims: int = None, max_dims: int = None, field_name: str = "array") -> ValidationResult:
        """Validate numpy array shape and properties."""
        result = ValidationResult()
        
        if not isinstance(array, np.ndarray):
            result.add_error("TYPE_ERROR", f"Expected numpy array, got {type(array).__name__}", field_name, type(array).__name__)
            return result
            
        # Check dimensions
        if min_dims is not None and array.ndim < min_dims:
            result.add_error("DIMENSION_ERROR", f"Array has {array.ndim} dimensions, minimum required: {min_dims}", field_name, array.ndim)
            
        if max_dims is not None and array.ndim > max_dims:
            result.add_error("DIMENSION_ERROR", f"Array has {array.ndim} dimensions, maximum allowed: {max_dims}", field_name, array.ndim)
            
        # Check exact shape
        if expected_shape is not None:
            if array.shape != expected_shape:
                result.add_error("SHAPE_ERROR", f"Array shape {array.shape} does not match expected {expected_shape}", field_name, array.shape)
                
        # Check for invalid values
        if np.any(np.isnan(array)):
            result.add_error("VALUE_ERROR", "Array contains NaN values", field_name)
            
        if np.any(np.isinf(array)):
            result.add_error("VALUE_ERROR", "Array contains infinite values", field_name)
            
        # Check memory size (prevent DoS)
        memory_mb = array.nbytes / (1024 * 1024)
        if memory_mb > 1000:  # 1GB limit
            result.add_error("SIZE_LIMIT_EXCEEDED", f"Array too large: {memory_mb:.1f}MB (max 1000MB)", field_name, memory_mb)
            
        return result
        
    def validate_tensor(self, tensor: Any, expected_device: str = None, 
                       expected_dtype: Any = None, field_name: str = "tensor") -> ValidationResult:
        """Validate PyTorch tensor."""
        result = ValidationResult()
        
        if not TORCH_AVAILABLE:
            result.add_warning("TORCH_NOT_AVAILABLE", "PyTorch not available, skipping tensor validation", field_name)
            return result
            
        if not isinstance(tensor, torch.Tensor):
            result.add_error("TYPE_ERROR", f"Expected torch.Tensor, got {type(tensor).__name__}", field_name, type(tensor).__name__)
            return result
            
        # Check device
        if expected_device is not None and str(tensor.device) != expected_device:
            result.add_error("DEVICE_ERROR", f"Tensor on device {tensor.device}, expected {expected_device}", field_name, str(tensor.device))
            
        # Check dtype
        if expected_dtype is not None and tensor.dtype != expected_dtype:
            result.add_error("DTYPE_ERROR", f"Tensor dtype {tensor.dtype}, expected {expected_dtype}", field_name, tensor.dtype)
            
        # Check for invalid values
        if torch.any(torch.isnan(tensor)):
            result.add_error("VALUE_ERROR", "Tensor contains NaN values", field_name)
            
        if torch.any(torch.isinf(tensor)):
            result.add_error("VALUE_ERROR", "Tensor contains infinite values", field_name)
            
        # Check memory usage
        if tensor.is_cuda:
            memory_mb = tensor.element_size() * tensor.numel() / (1024 * 1024)
            if memory_mb > 5000:  # 5GB limit for GPU tensors
                result.add_error("SIZE_LIMIT_EXCEEDED", f"GPU tensor too large: {memory_mb:.1f}MB (max 5000MB)", field_name, memory_mb)
                
        return result
        
    def validate_config_dict(self, config: Dict[str, Any], schema: Dict[str, Dict] = None) -> ValidationResult:
        """Validate configuration dictionary against schema."""
        result = ValidationResult()
        
        if not isinstance(config, dict):
            result.add_error("TYPE_ERROR", f"Config must be dict, got {type(config).__name__}", "config", type(config).__name__)
            return result
            
        if schema is None:
            return result
            
        # Check required fields
        for field, field_schema in schema.items():
            if field_schema.get('required', False) and field not in config:
                result.add_error("MISSING_FIELD", f"Required field '{field}' missing", field)
                continue
                
            if field not in config:
                continue
                
            value = config[field]
            
            # Check type
            expected_type = field_schema.get('type')
            if expected_type and not isinstance(value, expected_type):
                result.add_error("TYPE_ERROR", f"Field '{field}' must be {expected_type.__name__}, got {type(value).__name__}", field, type(value).__name__)
                continue
                
            # Check choices
            choices = field_schema.get('choices')
            if choices and value not in choices:
                result.add_error("VALUE_ERROR", f"Field '{field}' must be one of {choices}, got {value}", field, value)
                
            # Check range for numeric fields
            if isinstance(value, (int, float)):
                min_val = field_schema.get('min')
                max_val = field_schema.get('max')
                range_result = self.validate_numeric_range(value, min_val, max_val, field)
                result.errors.extend(range_result.errors)
                result.warnings.extend(range_result.warnings)
                if range_result.errors:
                    result.is_valid = False
                    
            # Security validation for string fields
            if isinstance(value, str):
                security_result = self.security_validator.validate_string_security(value, field)
                result.errors.extend(security_result.errors)
                result.warnings.extend(security_result.warnings)
                if security_result.errors:
                    result.is_valid = False
                    
        # Check for unknown fields
        for field in config:
            if field not in schema:
                result.add_warning("UNKNOWN_FIELD", f"Unknown field '{field}' in config", field, config[field])
                
        return result


class EventValidator:
    """Validator for event data structures."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_events(self, events: List[List[float]]) -> ValidationResult:
        """Validate event array structure and values."""
        result = ValidationResult()
        
        # Check if events is a list
        if not isinstance(events, list):
            result.add_error(
                "TYPE_ERROR", 
                f"Events must be a list, got {type(events).__name__}",
                "events",
                type(events).__name__
            )
            return result
            
        # Check if empty
        if len(events) == 0:
            result.add_warning(
                "EMPTY_DATA", 
                "Event list is empty",
                "events",
                len(events)
            )
            return result
            
        # Validate each event (sample first 10 for performance)
        sample_size = min(10, len(events))
        for i in range(sample_size):
            event = events[i]
            event_result = self.validate_single_event(event, i)
            result.errors.extend(event_result.errors)
            result.warnings.extend(event_result.warnings)
            if not event_result.is_valid:
                result.is_valid = False
                
        return result
        
    def validate_single_event(self, event: List[float], index: int) -> ValidationResult:
        """Validate a single event."""
        result = ValidationResult()
        
        # Check if event is a list
        if not isinstance(event, list):
            result.add_error(
                "EVENT_TYPE_ERROR",
                f"Event must be a list, got {type(event).__name__}",
                f"events[{index}]",
                type(event).__name__
            )
            return result
            
        # Check event length
        if len(event) != 4:
            result.add_error(
                "EVENT_LENGTH_ERROR",
                f"Event must have 4 elements [x, y, t, p], got {len(event)}",
                f"events[{index}]",
                len(event)
            )
            return result
            
        x, y, t, p = event
        
        # Validate x coordinate
        if not isinstance(x, (int, float)):
            result.add_error(
                "X_TYPE_ERROR",
                f"X coordinate must be numeric, got {type(x).__name__}",
                f"events[{index}].x",
                x
            )
        elif x < 0:
            result.add_error(
                "X_NEGATIVE_ERROR",
                f"X coordinate must be non-negative, got {x}",
                f"events[{index}].x",
                x
            )
            
        # Validate y coordinate
        if not isinstance(y, (int, float)):
            result.add_error(
                "Y_TYPE_ERROR",
                f"Y coordinate must be numeric, got {type(y).__name__}",
                f"events[{index}].y",
                y
            )
        elif y < 0:
            result.add_error(
                "Y_NEGATIVE_ERROR",
                f"Y coordinate must be non-negative, got {y}",
                f"events[{index}].y",
                y
            )
            
        # Validate timestamp
        if not isinstance(t, (int, float)):
            result.add_error(
                "T_TYPE_ERROR",
                f"Timestamp must be numeric, got {type(t).__name__}",
                f"events[{index}].t",
                t
            )
        elif t < 0:
            result.add_error(
                "T_NEGATIVE_ERROR",
                f"Timestamp must be non-negative, got {t}",
                f"events[{index}].t",
                t
            )
            
        # Validate polarity
        if not isinstance(p, (int, float)):
            result.add_error(
                "P_TYPE_ERROR",
                f"Polarity must be numeric, got {type(p).__name__}",
                f"events[{index}].p",
                p
            )
        elif p not in [-1, 1]:
            result.add_error(
                "P_VALUE_ERROR",
                f"Polarity must be -1 or 1, got {p}",
                f"events[{index}].p",
                p
            )
            
        return result
        
    def validate_image_dimensions(self, width: int, height: int) -> ValidationResult:
        """Validate image dimensions."""
        result = ValidationResult()
        
        if not isinstance(width, int) or not isinstance(height, int):
            result.add_error(
                "DIMENSION_TYPE_ERROR",
                f"Width and height must be integers, got {type(width).__name__}, {type(height).__name__}",
                "dimensions"
            )
            return result
            
        if width <= 0 or height <= 0:
            result.add_error(
                "DIMENSION_VALUE_ERROR", 
                f"Width and height must be positive, got {width}x{height}",
                "dimensions"
            )
            
        if width > 10000 or height > 10000:
            result.add_warning(
                "DIMENSION_LARGE_WARNING",
                f"Very large dimensions: {width}x{height}",
                "dimensions"
            )
            
        return result


class StreamIntegrityValidator:
    """Real-time validation of event stream integrity and quality."""
    
    def __init__(self, window_size: int = 1000, max_frequency_hz: float = 10000):
        self.logger = logging.getLogger(__name__)
        self.window_size = window_size
        self.max_frequency_hz = max_frequency_hz
        self.event_history = []
        self.integrity_stats = {
            'total_events': 0,
            'corrupted_events': 0,
            'out_of_order_events': 0,
            'duplicate_events': 0,
            'frequency_violations': 0,
            'last_check_time': time.time()
        }
        
    def validate_event_stream_integrity(self, events: List[List[float]]) -> ValidationResult:
        """Validate event stream for real-time integrity."""
        result = ValidationResult()
        
        if not events:
            result.add_warning("EMPTY_STREAM", "Event stream is empty")
            return result
            
        # Update statistics
        self.integrity_stats['total_events'] += len(events)
        
        # Check temporal consistency
        temporal_result = self._validate_temporal_consistency(events)
        result.errors.extend(temporal_result.errors)
        result.warnings.extend(temporal_result.warnings)
        if not temporal_result.is_valid:
            result.is_valid = False
            
        # Check for duplicates
        duplicate_result = self._detect_duplicate_events(events)
        result.errors.extend(duplicate_result.errors)
        result.warnings.extend(duplicate_result.warnings)
        if not duplicate_result.is_valid:
            result.is_valid = False
            
        # Check event frequency
        frequency_result = self._validate_event_frequency(events)
        result.errors.extend(frequency_result.errors)
        result.warnings.extend(frequency_result.warnings)
        if not frequency_result.is_valid:
            result.is_valid = False
            
        # Check data quality
        quality_result = self._validate_data_quality(events)
        result.errors.extend(quality_result.errors)
        result.warnings.extend(quality_result.warnings)
        if not quality_result.is_valid:
            result.is_valid = False
            
        # Update event history for sliding window analysis
        self._update_event_history(events)
        
        return result
        
    def _validate_temporal_consistency(self, events: List[List[float]]) -> ValidationResult:
        """Check that events are temporally ordered."""
        result = ValidationResult()
        
        for i in range(1, len(events)):
            prev_time = events[i-1][2]
            curr_time = events[i][2]
            
            if curr_time < prev_time:
                self.integrity_stats['out_of_order_events'] += 1
                result.add_error(
                    "TEMPORAL_DISORDER",
                    f"Event {i} timestamp {curr_time} < previous timestamp {prev_time}",
                    f"events[{i}]",
                    curr_time
                )
                break  # Stop after first violation for performance
                
        return result
        
    def _detect_duplicate_events(self, events: List[List[float]]) -> ValidationResult:
        """Detect duplicate events in the stream."""
        result = ValidationResult()
        
        seen_events = set()
        duplicates = 0
        
        for i, event in enumerate(events[:100]):  # Check first 100 for performance
            event_tuple = tuple(event)
            if event_tuple in seen_events:
                duplicates += 1
                if duplicates == 1:  # Report only first duplicate
                    result.add_warning(
                        "DUPLICATE_EVENT",
                        f"Duplicate event detected at index {i}",
                        f"events[{i}]",
                        event
                    )
            else:
                seen_events.add(event_tuple)
                
        if duplicates > 0:
            self.integrity_stats['duplicate_events'] += duplicates
            
        return result
        
    def _validate_event_frequency(self, events: List[List[float]]) -> ValidationResult:
        """Check event frequency for abnormal patterns."""
        result = ValidationResult()
        
        if len(events) < 2:
            return result
            
        # Calculate time span
        time_span = events[-1][2] - events[0][2]
        if time_span <= 0:
            result.add_error("INVALID_TIME_SPAN", "Invalid time span in event stream")
            return result
            
        # Calculate frequency
        frequency = len(events) / time_span
        
        if frequency > self.max_frequency_hz:
            self.integrity_stats['frequency_violations'] += 1
            result.add_warning(
                "HIGH_FREQUENCY",
                f"Event frequency {frequency:.1f} Hz exceeds maximum {self.max_frequency_hz} Hz",
                "frequency",
                frequency
            )
            
        # Check for suspicious clustering
        if len(events) >= 10:
            cluster_result = self._detect_event_clustering(events)
            result.errors.extend(cluster_result.errors)
            result.warnings.extend(cluster_result.warnings)
            
        return result
        
    def _detect_event_clustering(self, events: List[List[float]]) -> ValidationResult:
        """Detect abnormal event clustering."""
        result = ValidationResult()
        
        # Check if too many events happen in a short time window
        window_duration = 0.001  # 1ms window
        max_events_per_window = 50
        
        for i in range(len(events) - max_events_per_window):
            window_start = events[i][2]
            window_end = events[i + max_events_per_window - 1][2]
            
            if window_end - window_start < window_duration:
                result.add_warning(
                    "EVENT_CLUSTERING",
                    f"Excessive event clustering: {max_events_per_window} events in {window_end - window_start:.6f}s",
                    f"events[{i}:{i+max_events_per_window}]"
                )
                break
                
        return result
        
    def _validate_data_quality(self, events: List[List[float]]) -> ValidationResult:
        """Validate data quality metrics."""
        result = ValidationResult()
        
        # Check for noise patterns
        noise_result = self._detect_noise_patterns(events)
        result.errors.extend(noise_result.errors)
        result.warnings.extend(noise_result.warnings)
        
        # Check spatial distribution
        spatial_result = self._validate_spatial_distribution(events)
        result.errors.extend(spatial_result.errors)
        result.warnings.extend(spatial_result.warnings)
        
        return result
        
    def _detect_noise_patterns(self, events: List[List[float]]) -> ValidationResult:
        """Detect noise patterns in event data."""
        result = ValidationResult()
        
        if len(events) < 10:
            return result
            
        # Check for excessive hot pixels
        pixel_counts = {}
        for event in events[:1000]:  # Sample first 1000 events
            pixel = (int(event[0]), int(event[1]))
            pixel_counts[pixel] = pixel_counts.get(pixel, 0) + 1
            
        max_pixel_count = max(pixel_counts.values()) if pixel_counts else 0
        if max_pixel_count > len(events) * 0.1:  # Single pixel > 10% of events
            result.add_warning(
                "HOT_PIXEL_DETECTED",
                f"Hot pixel detected with {max_pixel_count} events",
                "spatial_distribution"
            )
            
        return result
        
    def _validate_spatial_distribution(self, events: List[List[float]]) -> ValidationResult:
        """Validate spatial distribution of events."""
        result = ValidationResult()
        
        if len(events) < 10:
            return result
            
        # Extract coordinates
        x_coords = [event[0] for event in events[:100]]
        y_coords = [event[1] for event in events[:100]]
        
        # Check for reasonable coordinate ranges
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        if x_range == 0 and y_range == 0:
            result.add_warning(
                "STATIC_EVENTS",
                "All events at same spatial location",
                "spatial_distribution"
            )
        elif x_range > 10000 or y_range > 10000:
            result.add_error(
                "UNREASONABLE_COORDINATES",
                f"Coordinate range too large: x={x_range}, y={y_range}",
                "spatial_distribution"
            )
            
        return result
        
    def _update_event_history(self, events: List[List[float]]):
        """Update sliding window of event history."""
        self.event_history.extend(events)
        
        # Keep only recent events
        if len(self.event_history) > self.window_size:
            self.event_history = self.event_history[-self.window_size:]
            
    def get_integrity_stats(self) -> Dict[str, Any]:
        """Get integrity statistics."""
        stats = self.integrity_stats.copy()
        stats['integrity_ratio'] = 1.0 - (stats['corrupted_events'] / max(1, stats['total_events']))
        stats['check_duration'] = time.time() - stats['last_check_time']
        return stats
        
    def reset_stats(self):
        """Reset integrity statistics."""
        self.integrity_stats = {
            'total_events': 0,
            'corrupted_events': 0,
            'out_of_order_events': 0,
            'duplicate_events': 0,
            'frequency_violations': 0,
            'last_check_time': time.time()
        }


class ModelOutputValidator:
    """Validator for SNN model outputs with sanity checks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_snn_output(self, output: Any, expected_shape: Optional[Tuple] = None,
                          expected_range: Optional[Tuple[float, float]] = None,
                          model_name: str = "SNN") -> ValidationResult:
        """Validate SNN model output."""
        result = ValidationResult()
        
        # Type validation
        if TORCH_AVAILABLE and isinstance(output, torch.Tensor):
            tensor_result = self._validate_tensor_output(output, expected_shape, expected_range)
            result.errors.extend(tensor_result.errors)
            result.warnings.extend(tensor_result.warnings)
            if not tensor_result.is_valid:
                result.is_valid = False
        elif isinstance(output, np.ndarray):
            array_result = self._validate_array_output(output, expected_shape, expected_range)
            result.errors.extend(array_result.errors)
            result.warnings.extend(array_result.warnings)
            if not array_result.is_valid:
                result.is_valid = False
        else:
            result.add_error(
                "UNSUPPORTED_OUTPUT_TYPE",
                f"Unsupported output type: {type(output)}",
                "output_type",
                type(output).__name__
            )
            return result
            
        # SNN-specific validations
        snn_result = self._validate_snn_characteristics(output)
        result.errors.extend(snn_result.errors)
        result.warnings.extend(snn_result.warnings)
        if not snn_result.is_valid:
            result.is_valid = False
            
        return result
        
    def _validate_tensor_output(self, tensor: 'torch.Tensor', expected_shape: Optional[Tuple],
                              expected_range: Optional[Tuple[float, float]]) -> ValidationResult:
        """Validate PyTorch tensor output."""
        result = ValidationResult()
        
        # Shape validation
        if expected_shape and tensor.shape != expected_shape:
            result.add_error(
                "SHAPE_MISMATCH",
                f"Output shape {tensor.shape} != expected {expected_shape}",
                "shape"
            )
            
        # Check for invalid values
        if torch.any(torch.isnan(tensor)):
            result.add_error("NAN_VALUES", "Output contains NaN values", "values")
            
        if torch.any(torch.isinf(tensor)):
            result.add_error("INF_VALUES", "Output contains infinite values", "values")
            
        # Range validation
        if expected_range:
            min_val, max_val = expected_range
            if torch.any(tensor < min_val) or torch.any(tensor > max_val):
                actual_min = torch.min(tensor).item()
                actual_max = torch.max(tensor).item()
                result.add_error(
                    "RANGE_VIOLATION",
                    f"Output range [{actual_min:.6f}, {actual_max:.6f}] outside expected {expected_range}",
                    "range"
                )
                
        return result
        
    def _validate_array_output(self, array: np.ndarray, expected_shape: Optional[Tuple],
                             expected_range: Optional[Tuple[float, float]]) -> ValidationResult:
        """Validate numpy array output."""
        result = ValidationResult()
        
        # Shape validation
        if expected_shape and array.shape != expected_shape:
            result.add_error(
                "SHAPE_MISMATCH",
                f"Output shape {array.shape} != expected {expected_shape}",
                "shape"
            )
            
        # Check for invalid values
        if np.any(np.isnan(array)):
            result.add_error("NAN_VALUES", "Output contains NaN values", "values")
            
        if np.any(np.isinf(array)):
            result.add_error("INF_VALUES", "Output contains infinite values", "values")
            
        # Range validation
        if expected_range:
            min_val, max_val = expected_range
            if np.any(array < min_val) or np.any(array > max_val):
                actual_min = np.min(array)
                actual_max = np.max(array)
                result.add_error(
                    "RANGE_VIOLATION",
                    f"Output range [{actual_min:.6f}, {actual_max:.6f}] outside expected {expected_range}",
                    "range"
                )
                
        return result
        
    def _validate_snn_characteristics(self, output: Union['torch.Tensor', np.ndarray]) -> ValidationResult:
        """Validate SNN-specific output characteristics."""
        result = ValidationResult()
        
        # For spike trains, values should typically be 0 or 1
        if TORCH_AVAILABLE and isinstance(output, torch.Tensor):
            unique_values = torch.unique(output)
            non_binary = torch.any((unique_values != 0) & (unique_values != 1))
        else:
            unique_values = np.unique(output)
            non_binary = np.any((unique_values != 0) & (unique_values != 1))
            
        if non_binary:
            if len(unique_values) > 10:  # Too many unique values for spike data
                result.add_warning(
                    "NON_BINARY_SPIKES",
                    f"Output contains {len(unique_values)} unique values, expected binary spikes",
                    "spike_characteristics"
                )
            else:
                result.add_warning(
                    "ANALOG_OUTPUT",
                    f"Analog output detected with values: {unique_values}",
                    "spike_characteristics"
                )
                
        # Check spike rate
        if TORCH_AVAILABLE and isinstance(output, torch.Tensor):
            spike_rate = torch.mean(output.float()).item()
        else:
            spike_rate = np.mean(output.astype(float))
            
        if spike_rate > 0.5:
            result.add_warning(
                "HIGH_SPIKE_RATE",
                f"High spike rate: {spike_rate:.3f} (>50%)",
                "spike_rate",
                spike_rate
            )
        elif spike_rate < 0.001:
            result.add_warning(
                "LOW_SPIKE_RATE",
                f"Very low spike rate: {spike_rate:.6f} (<0.1%)",
                "spike_rate",
                spike_rate
            )
            
        return result


class CircuitBreaker:
    """Circuit breaker pattern for hardware backend failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 expected_call_frequency: float = 1.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_call_frequency = expected_call_frequency
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_call_time = time.time()
        self.logger = logging.getLogger(__name__)
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        current_time = time.time()
        
        # Check if circuit should be closed due to inactivity
        if (current_time - self.last_call_time) > (1.0 / self.expected_call_frequency * 10):
            self._reset()
            
        self.last_call_time = current_time
        
        if self.state == "OPEN":
            if current_time - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker OPEN. Retry after {self.recovery_timeout - (current_time - self.last_failure_time):.1f}s"
                )
                
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise
            
    def _on_success(self):
        """Handle successful call."""
        if self.state == "HALF_OPEN":
            self._reset()
            self.logger.info("Circuit breaker reset to CLOSED after successful call")
            
    def _on_failure(self, error: Exception):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.error(
                f"Circuit breaker OPEN after {self.failure_count} failures. "
                f"Last error: {error}"
            )
        else:
            self.logger.warning(
                f"Circuit breaker failure {self.failure_count}/{self.failure_threshold}: {error}"
            )
            
    def _reset(self):
        """Reset circuit breaker to closed state."""
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = 0
        
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "time_until_retry": max(0, self.recovery_timeout - (time.time() - self.last_failure_time)) if self.state == "OPEN" else 0,
            "last_failure_time": self.last_failure_time
        }


def get_event_validator():
    """Get event validator instance."""
    return EventValidator()


def get_stream_integrity_validator():
    """Get stream integrity validator instance."""
    return StreamIntegrityValidator()


def get_model_output_validator():
    """Get model output validator instance."""
    return ModelOutputValidator()


def validate_and_handle(data, validator_func, operation: str, strict: bool = True) -> bool:
    """Validate data and handle errors in one call."""
    try:
        result = validator_func(data)
        
        if not result.is_valid:
            if strict:
                raise ValueError(f"Validation failed for {operation}: {result.format_errors()}")
            return False
            
        return True
        
    except Exception as e:
        if strict:
            raise
        return False


# Convenient standalone validation functions for easy import
def validate_events(events) -> ValidationResult:
    """Validate event array - standalone function."""
    validator = EventValidator()
    return validator.validate_events(events)


def validate_image_dimensions(width: int, height: int) -> ValidationResult:
    """Validate image dimensions - standalone function."""
    validator = EventValidator()
    return validator.validate_image_dimensions(width, height)


def safe_operation(operation_func, *args, **kwargs):
    """Execute operation with error handling."""
    try:
        return operation_func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Operation failed: {e}")
        return None