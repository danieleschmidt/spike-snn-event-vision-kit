"""
Spike-SNN Event Vision Kit

Production-ready toolkit for event-camera object detection with spiking neural networks.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"
__license__ = "MIT"

# Core imports for easy access
from .core import DVSCamera, EventPreprocessor, SpatioTemporalPreprocessor
from .models import EventSNN, SpikingYOLO, CustomSNN, LIFNeuron

__all__ = [
    "DVSCamera",
    "EventPreprocessor", 
    "SpatioTemporalPreprocessor",
    "EventSNN",
    "SpikingYOLO",
    "CustomSNN",
    "LIFNeuron",
]