"""
Spike SNN Event Vision Kit - Production-ready neuromorphic vision processing.

A comprehensive toolkit for event-camera object detection with spiking neural networks.
"""

__version__ = "0.3.0"

# Core functionality
try:
    from .validation import ValidationResult, EventValidator, SecurityValidator, DataValidator
    from .optimization import get_memory_tracker, get_tensor_optimizer
    from .scaling import AutoScaler, ScalingPolicy
except ImportError as e:
    # Graceful degradation for testing
    pass

__all__ = [
    'ValidationResult',
    'EventValidator',
    'SecurityValidator', 
    'DataValidator',
    'get_memory_tracker',
    'get_tensor_optimizer',
    'AutoScaler',
    'ScalingPolicy'
]