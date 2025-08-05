"""
Input validation and error handling for Spike SNN Event Vision Kit.

This module provides comprehensive validation and error handling to ensure
robust operation in production environments.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time


@dataclass
class ValidationError:
    """Validation error information."""
    code: str
    message: str
    field: Optional[str] = None
    value: Any = None
    severity: str = "error"  # error, warning, info
    

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


class ValidationError(Exception):
    """Custom validation error exception."""
    pass