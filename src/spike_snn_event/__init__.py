"""
Spike SNN Event Vision Kit - Production-Ready Neuromorphic Vision Processing

A comprehensive toolkit for event-based vision processing with spiking neural networks,
featuring robust error handling, high-performance processing, and adaptive intelligence.

Generations of Enhancement:
- Generation 1: Basic functionality with syntax fixes and security
- Generation 2: Robustness with fault tolerance and error recovery  
- Generation 3: Performance with scaling, caching, and adaptive intelligence
"""

__version__ = "1.0.0"
__author__ = "Terragon Labs"

# Core imports for convenience
try:
    from .config import SystemConfiguration, load_configuration
    from .security import InputSanitizer, SecurityError
    from .robust_core import RobustEventProcessor, CircuitBreaker
    from .advanced_validation import DataValidator, ValidationLevel
    from .high_performance_core import HighPerformanceProcessor, IntelligentCache
    from .adaptive_intelligence import AdaptiveIntelligenceEngine, AdaptationStrategy
    
    __all__ = [
        'SystemConfiguration',
        'load_configuration', 
        'InputSanitizer',
        'SecurityError',
        'RobustEventProcessor',
        'CircuitBreaker',
        'DataValidator',
        'ValidationLevel', 
        'HighPerformanceProcessor',
        'IntelligentCache',
        'AdaptiveIntelligenceEngine',
        'AdaptationStrategy'
    ]
    
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}", ImportWarning)
    __all__ = []