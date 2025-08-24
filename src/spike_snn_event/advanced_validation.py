"""
Advanced validation system with comprehensive error checking and recovery.

Generation 2: Enhanced validation for robust operation
- Multi-layer validation
- Automatic error correction
- Performance monitoring
- Data integrity checks
"""

import time
import logging
import hashlib
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import functools


class ValidationLevel(Enum):
    """Validation severity levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class ValidationResult(Enum):
    """Validation result types."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    CORRECTED = "corrected"


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    result: ValidationResult
    message: str
    details: Dict[str, Any]
    corrected_data: Optional[Any] = None
    execution_time_ms: float = 0.0
    

class DataValidator:
    """Enhanced data validator with automatic correction capabilities."""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        self.level = level
        self.logger = logging.getLogger(__name__)
        self.validation_stats = {
            'total_validations': 0,
            'passes': 0,
            'warnings': 0,
            'failures': 0,
            'corrections': 0
        }
    
    def validate_events(self, events: Union[np.ndarray, List[Dict]], auto_correct: bool = True) -> ValidationReport:
        """Validate event data with optional auto-correction."""
        start_time = time.time()
        
        try:
            self.validation_stats['total_validations'] += 1
            
            # Convert to standard format if needed
            if isinstance(events, list):
                events = self._list_to_array(events)
            
            # Basic structure validation
            if not isinstance(events, np.ndarray):
                return ValidationReport(
                    ValidationResult.FAIL,
                    "Events must be numpy array or list of dictionaries",
                    {'type': type(events).__name__},
                    execution_time_ms=self._elapsed_ms(start_time)
                )
            
            # Shape validation
            shape_report = self._validate_shape(events)
            if shape_report.result == ValidationResult.FAIL:
                self.validation_stats['failures'] += 1
                return shape_report
            
            # Data range validation
            range_report = self._validate_ranges(events, auto_correct)
            if range_report.result == ValidationResult.FAIL:
                self.validation_stats['failures'] += 1
                return range_report
            elif range_report.result == ValidationResult.CORRECTED:
                self.validation_stats['corrections'] += 1
                events = range_report.corrected_data
            
            # Temporal validation
            temporal_report = self._validate_temporal(events, auto_correct)
            if temporal_report.result == ValidationResult.FAIL:
                self.validation_stats['failures'] += 1
                return temporal_report
            elif temporal_report.result == ValidationResult.CORRECTED:
                self.validation_stats['corrections'] += 1
                events = temporal_report.corrected_data
            
            # Statistical validation (higher levels)
            if self.level.value in ['strict', 'paranoid']:
                stats_report = self._validate_statistics(events)
                if stats_report.result == ValidationResult.WARNING:
                    self.validation_stats['warnings'] += 1
                    return stats_report
            
            # Data integrity (paranoid level)
            if self.level == ValidationLevel.PARANOID:
                integrity_report = self._validate_integrity(events)
                if integrity_report.result != ValidationResult.PASS:
                    self.validation_stats['warnings'] += 1
                    return integrity_report
            
            self.validation_stats['passes'] += 1
            
            return ValidationReport(
                ValidationResult.PASS,
                f"Events validation passed ({len(events)} events)",
                {
                    'event_count': len(events),
                    'validation_level': self.level.value,
                    'shape': events.shape if hasattr(events, 'shape') else 'N/A'
                },
                corrected_data=events,
                execution_time_ms=self._elapsed_ms(start_time)
            )
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            self.validation_stats['failures'] += 1
            
            return ValidationReport(
                ValidationResult.FAIL,
                f"Validation exception: {e}",
                {'exception': str(e), 'type': type(e).__name__},
                execution_time_ms=self._elapsed_ms(start_time)
            )
    
    def _list_to_array(self, events: List[Dict]) -> np.ndarray:
        """Convert list of event dictionaries to numpy array."""
        if not events:
            return np.empty((0, 4))
        
        # Extract standard fields
        array_data = []
        for event in events:
            x = event.get('x', 0)
            y = event.get('y', 0)
            t = event.get('timestamp', event.get('t', 0))
            p = event.get('polarity', event.get('p', 1))
            array_data.append([x, y, t, p])
        
        return np.array(array_data, dtype=np.float64)
    
    def _validate_shape(self, events: np.ndarray) -> ValidationReport:
        """Validate array shape and dimensions."""
        if len(events.shape) != 2:
            return ValidationReport(
                ValidationResult.FAIL,
                f"Events must be 2D array, got {len(events.shape)}D",
                {'shape': events.shape}
            )
        
        if events.shape[1] not in [4, 5]:  # Standard event format: [x, y, t, p] or [x, y, t, p, extra]
            return ValidationReport(
                ValidationResult.FAIL,
                f"Events must have 4 or 5 columns, got {events.shape[1]}",
                {'shape': events.shape, 'expected_columns': '[x, y, t, p]'}
            )
        
        if events.shape[0] == 0:
            return ValidationReport(
                ValidationResult.WARNING,
                "Empty event array",
                {'shape': events.shape}
            )
        
        return ValidationReport(
            ValidationResult.PASS,
            "Shape validation passed",
            {'shape': events.shape}
        )
    
    def _validate_ranges(self, events: np.ndarray, auto_correct: bool) -> ValidationReport:
        """Validate data ranges and correct if possible."""
        issues = []
        corrections = []
        
        if len(events) == 0:
            return ValidationReport(ValidationResult.PASS, "No events to validate", {})
        
        # Check for invalid coordinates
        x_invalid = np.isnan(events[:, 0]) | np.isinf(events[:, 0]) | (events[:, 0] < 0) | (events[:, 0] > 10000)
        y_invalid = np.isnan(events[:, 1]) | np.isinf(events[:, 1]) | (events[:, 1] < 0) | (events[:, 1] > 10000)
        
        if np.any(x_invalid) or np.any(y_invalid):
            issues.append(f"Invalid coordinates: {np.sum(x_invalid)} x, {np.sum(y_invalid)} y")
            
            if auto_correct:
                # Clip coordinates to valid ranges
                events[:, 0] = np.clip(events[:, 0], 0, 1000)
                events[:, 1] = np.clip(events[:, 1], 0, 1000)
                # Replace NaN/inf with zeros
                events[:, 0] = np.nan_to_num(events[:, 0], nan=0.0, posinf=1000.0, neginf=0.0)
                events[:, 1] = np.nan_to_num(events[:, 1], nan=0.0, posinf=1000.0, neginf=0.0)
                corrections.append("Clipped coordinates to valid range [0, 1000]")
        
        # Check timestamps
        t_invalid = np.isnan(events[:, 2]) | np.isinf(events[:, 2]) | (events[:, 2] < 0)
        if np.any(t_invalid):
            issues.append(f"Invalid timestamps: {np.sum(t_invalid)}")
            
            if auto_correct:
                # Use current time for invalid timestamps
                current_time = time.time()
                events[t_invalid, 2] = current_time
                corrections.append("Replaced invalid timestamps with current time")
        
        # Check polarity
        p_invalid = ~np.isin(events[:, 3], [-1, 0, 1])
        if np.any(p_invalid):
            issues.append(f"Invalid polarity values: {np.sum(p_invalid)}")
            
            if auto_correct:
                # Normalize polarity to -1 or 1
                events[p_invalid, 3] = np.sign(events[p_invalid, 3])
                events[events[:, 3] == 0, 3] = 1  # Zero polarity becomes positive
                corrections.append("Normalized polarity to {-1, 1}")
        
        if issues and not auto_correct:
            return ValidationReport(
                ValidationResult.FAIL,
                f"Range validation failed: {'; '.join(issues)}",
                {'issues': issues}
            )
        elif corrections:
            return ValidationReport(
                ValidationResult.CORRECTED,
                f"Range issues corrected: {'; '.join(corrections)}",
                {'corrections': corrections, 'original_issues': issues},
                corrected_data=events
            )
        
        return ValidationReport(
            ValidationResult.PASS,
            "Range validation passed",
            {'x_range': [float(events[:, 0].min()), float(events[:, 0].max())],
             'y_range': [float(events[:, 1].min()), float(events[:, 1].max())],
             't_range': [float(events[:, 2].min()), float(events[:, 2].max())]}
        )
    
    def _validate_temporal(self, events: np.ndarray, auto_correct: bool) -> ValidationReport:
        """Validate temporal ordering and consistency."""
        if len(events) <= 1:
            return ValidationReport(ValidationResult.PASS, "Insufficient events for temporal validation", {})
        
        timestamps = events[:, 2]
        
        # Check for non-monotonic timestamps
        time_diffs = np.diff(timestamps)
        negative_diffs = np.sum(time_diffs < 0)
        
        if negative_diffs > 0:
            if auto_correct:
                # Sort events by timestamp
                sort_idx = np.argsort(timestamps)
                events = events[sort_idx]
                
                return ValidationReport(
                    ValidationResult.CORRECTED,
                    f"Corrected {negative_diffs} temporal ordering issues",
                    {'sorted_events': True, 'negative_diffs': int(negative_diffs)},
                    corrected_data=events
                )
            else:
                return ValidationReport(
                    ValidationResult.FAIL,
                    f"Temporal ordering violation: {negative_diffs} negative time differences",
                    {'negative_diffs': int(negative_diffs)}
                )
        
        # Check for suspicious time gaps
        large_gaps = np.sum(time_diffs > 1.0)  # Gaps > 1 second
        if large_gaps > len(events) * 0.1:  # More than 10% large gaps
            return ValidationReport(
                ValidationResult.WARNING,
                f"Suspicious temporal patterns: {large_gaps} large time gaps",
                {'large_gaps': int(large_gaps), 'threshold': 1.0}
            )
        
        return ValidationReport(
            ValidationResult.PASS,
            "Temporal validation passed",
            {'time_span_s': float(timestamps[-1] - timestamps[0]),
             'avg_rate_hz': len(events) / max(timestamps[-1] - timestamps[0], 1e-9)}
        )
    
    def _validate_statistics(self, events: np.ndarray) -> ValidationReport:
        """Validate statistical properties of events."""
        if len(events) == 0:
            return ValidationReport(ValidationResult.PASS, "No events for statistical validation", {})
        
        warnings = []
        
        # Check event rate
        if len(events) > 1:
            time_span = events[-1, 2] - events[0, 2]
            event_rate = len(events) / max(time_span, 1e-9)
            
            if event_rate > 1e6:  # More than 1MHz
                warnings.append(f"Very high event rate: {event_rate:.0f} Hz")
            elif event_rate < 1:  # Less than 1Hz
                warnings.append(f"Very low event rate: {event_rate:.2f} Hz")
        
        # Check spatial distribution
        x_std = np.std(events[:, 0])
        y_std = np.std(events[:, 1])
        
        if x_std < 1.0 or y_std < 1.0:
            warnings.append(f"Low spatial variance: x_std={x_std:.2f}, y_std={y_std:.2f}")
        
        # Check polarity balance
        if len(events) > 100:
            positive = np.sum(events[:, 3] > 0)
            negative = np.sum(events[:, 3] < 0)
            balance = min(positive, negative) / max(positive, negative) if max(positive, negative) > 0 else 0
            
            if balance < 0.1:  # Very imbalanced
                warnings.append(f"Imbalanced polarity: {balance:.2f} ratio")
        
        if warnings:
            return ValidationReport(
                ValidationResult.WARNING,
                f"Statistical validation warnings: {'; '.join(warnings)}",
                {'warnings': warnings}
            )
        
        return ValidationReport(
            ValidationResult.PASS,
            "Statistical validation passed",
            {'event_rate_hz': event_rate if 'event_rate' in locals() else 0,
             'spatial_std': [float(x_std), float(y_std)]}
        )
    
    def _validate_integrity(self, events: np.ndarray) -> ValidationReport:
        """Validate data integrity using checksums."""
        if len(events) == 0:
            return ValidationReport(ValidationResult.PASS, "No events for integrity check", {})
        
        # Calculate checksum
        data_bytes = events.tobytes()
        checksum = hashlib.md5(data_bytes).hexdigest()
        
        # Check for duplicate events
        if len(events) > 1:
            # Simple duplicate check on coordinates and time (rounded)
            rounded_events = np.round(events[:, :3], 6)  # Round to microsecond precision
            unique_events, unique_indices = np.unique(rounded_events, axis=0, return_index=True)
            duplicate_count = len(events) - len(unique_events)
            
            if duplicate_count > 0:
                return ValidationReport(
                    ValidationResult.WARNING,
                    f"Data integrity warning: {duplicate_count} potential duplicate events",
                    {'checksum': checksum, 'duplicates': int(duplicate_count)}
                )
        
        return ValidationReport(
            ValidationResult.PASS,
            "Data integrity check passed",
            {'checksum': checksum, 'data_size_bytes': len(data_bytes)}
        )
    
    def _elapsed_ms(self, start_time: float) -> float:
        """Calculate elapsed time in milliseconds."""
        return (time.time() - start_time) * 1000.0
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validation_stats['total_validations']
        return {
            **self.validation_stats,
            'success_rate': self.validation_stats['passes'] / max(total, 1),
            'correction_rate': self.validation_stats['corrections'] / max(total, 1),
            'failure_rate': self.validation_stats['failures'] / max(total, 1)
        }


def validation_required(level: ValidationLevel = ValidationLevel.STANDARD):
    """Decorator to add validation to functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            validator = DataValidator(level)
            
            # Find and validate event data in arguments
            validated_args = []
            for arg in args:
                if isinstance(arg, (np.ndarray, list)) and len(arg) > 0:
                    # Check if this looks like event data
                    if isinstance(arg, np.ndarray) and len(arg.shape) == 2 and arg.shape[1] in [4, 5]:
                        report = validator.validate_events(arg)
                        if report.result == ValidationResult.FAIL:
                            raise ValueError(f"Validation failed: {report.message}")
                        elif report.result == ValidationResult.CORRECTED:
                            validated_args.append(report.corrected_data)
                            continue
                
                validated_args.append(arg)
            
            return func(*validated_args, **kwargs)
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    import random
    
    # Create test data
    test_events = np.array([
        [10.5, 20.3, 1000.0, 1],
        [15.2, 25.1, 1000.1, -1], 
        [12.0, 22.5, 999.9, 1],  # Out of order timestamp
        [np.nan, 30.0, 1000.2, 1],  # Invalid coordinate
        [20.0, 35.0, 1000.3, 2]   # Invalid polarity
    ])
    
    validator = DataValidator(ValidationLevel.STRICT)
    
    print("Testing validation with auto-correction:")
    report = validator.validate_events(test_events, auto_correct=True)
    print(f"Result: {report.result.value}")
    print(f"Message: {report.message}")
    print(f"Details: {report.details}")
    print(f"Execution time: {report.execution_time_ms:.2f}ms")
    
    if report.corrected_data is not None:
        print(f"Corrected events shape: {report.corrected_data.shape}")
    
    print(f"\nValidation stats: {validator.get_validation_stats()}")