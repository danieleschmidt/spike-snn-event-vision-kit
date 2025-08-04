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

__all__ = [
    # Core functionality
    "DVSCamera",
    "EventPreprocessor", 
    "SpatioTemporalPreprocessor",
    "EventDataset",
    "EventVisualizer",
    "CameraConfig",
    "HotPixelFilter",
    "load_events_from_file",
    "save_events_to_file",
    
    # Models
    "EventSNN",
    "SpikingYOLO",
    "CustomSNN",
    "LIFNeuron",
    "SpikingConv2d",
    "TrainingConfig",
    "SurrogateGradient",
    "SpikingLayer",
    
    # Training
    "SpikingTrainer",
    "EventDataLoader",
    "create_training_config",
]

# Add ROS2 classes if available
if ROS2_AVAILABLE:
    __all__.extend([
        "EventCameraNode",
        "SNNDetectionNode",
        "EventVisualizationNode",
    ])

# Advanced functionality imports (Generation 2 & 3)
try:
    from .optimization import (
        LRUCache,
        ModelCache,
        MemoryOptimizer,
        GPUAccelerator,
        get_optimizer
    )
    from .concurrency import (
        ConcurrentProcessor,
        ModelPool,
        AsyncProcessor,
        EventStreamProcessor,
        get_concurrent_processor,
        parallel_map
    )
    from .scaling import (
        AutoScaler,
        LoadBalancer,
        ScalingOrchestrator,
        get_auto_scaler,
        get_load_balancer
    )
    
    __all__.extend([
        # Optimization
        "LRUCache", "ModelCache", "MemoryOptimizer", "GPUAccelerator", "get_optimizer",
        # Concurrency
        "ConcurrentProcessor", "ModelPool", "AsyncProcessor", "EventStreamProcessor",
        "get_concurrent_processor", "parallel_map", 
        # Scaling
        "AutoScaler", "LoadBalancer", "ScalingOrchestrator", 
        "get_auto_scaler", "get_load_balancer"
    ])
    
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False