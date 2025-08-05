"""
Spike-SNN Event Vision Kit

Production-ready toolkit for event-camera object detection with spiking neural networks.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "1.0.0-dev"

__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"
__license__ = "MIT"

# Core imports for easy access
try:
    from .core import (
        DVSCamera, 
        EventPreprocessor, 
        SpatioTemporalPreprocessor,
        EventDataset,
        EventVisualizer,
        CameraConfig,
        HotPixelFilter,
        load_events_from_file,
        save_events_to_file
    )
    CORE_AVAILABLE = True
except ImportError:
    # Fallback to lightweight core
    from .lite_core import (
        DVSCamera,
        EventPreprocessor,
        SpatioTemporalPreprocessor,
        EventVisualizer,
        CameraConfig,
        HotPixelFilter,
        load_events_from_file,
        save_events_to_file,
        LiteEventSNN
    )
    CORE_AVAILABLE = False

# Model imports (require PyTorch)
try:
    from .models import (
        EventSNN, 
        SpikingYOLO, 
        CustomSNN, 
        LIFNeuron,
        SpikingConv2d,
        TrainingConfig,
        SurrogateGradient,
        SpikingLayer
    )
    from .training import (
        SpikingTrainer,
        EventDataLoader,
        create_training_config
    )
    MODELS_AVAILABLE = True
except ImportError:
    # Add lightweight models if core not available
    if not CORE_AVAILABLE:
        from .lite_core import LiteEventSNN
        # Create aliases for compatibility
        EventSNN = LiteEventSNN
        CustomSNN = LiteEventSNN
    MODELS_AVAILABLE = False

# Optional ROS2 imports
try:
    from .ros2_nodes import (
        EventCameraNode,
        SNNDetectionNode,
        EventVisualizationNode
    )
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# Build __all__ dynamically based on available features
__all__ = [
    # Core functionality (always available)
    "DVSCamera",
    "EventPreprocessor", 
    "SpatioTemporalPreprocessor",
    "EventVisualizer",
    "CameraConfig",
    "HotPixelFilter",
    "load_events_from_file",
    "save_events_to_file",
]

# Add core dataset if available
if CORE_AVAILABLE:
    __all__.append("EventDataset")
else:
    __all__.append("LiteEventSNN")

# Add models if available
if MODELS_AVAILABLE:
    __all__.extend([
        "EventSNN",
        "SpikingYOLO",
        "CustomSNN", 
        "LIFNeuron",
        "SpikingConv2d",
        "TrainingConfig",
        "SurrogateGradient",
        "SpikingLayer",
        "SpikingTrainer",
        "EventDataLoader", 
        "create_training_config",
    ])
else:
    # Add lightweight alternatives
    __all__.extend([
        "EventSNN",  # Alias to LiteEventSNN
        "CustomSNN", # Alias to LiteEventSNN
    ])

# Add ROS2 classes if available
if ROS2_AVAILABLE:
    __all__.extend([
        "EventCameraNode",
        "SNNDetectionNode",
        "EventVisualizationNode",
    ])

# Advanced functionality imports (Generation 2 & 3) - temporarily disabled due to syntax issues
try:
    # Skip advanced imports for now
    # from .optimization import (...)
    # from .concurrency import (...)
    # from .scaling import (...)
    
    ADVANCED_FEATURES_AVAILABLE = False
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False