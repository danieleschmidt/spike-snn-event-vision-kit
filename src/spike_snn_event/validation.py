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


class ValidationError:
    """Validation error information."""
    
    def __init__(self, code: str, message: str, field: Optional[str] = None, value: Any = None, severity: str = "error"):
        self.code = code
        self.message = message
        self.field = field
        self.value = value
        self.severity = severity
    

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


def get_event_validator():
    """Get event validator instance."""
    return EventValidator()


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