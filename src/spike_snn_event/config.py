"""
Configuration management system for spike-snn-event-vision-kit.

Provides robust configuration loading, validation, and management
with support for environment variables, defaults, and schema validation.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List, Type
from pathlib import Path
from dataclasses import dataclass, field, fields
from abc import ABC, abstractmethod
import logging
from enum import Enum

from .validation import ValidationError, ConfigurationError, validate_file_path
from .security import get_input_sanitizer


class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    GPU = "gpu"  # Alias for CUDA


class ModelType(Enum):
    """Supported model types."""
    SPIKING_YOLO = "spiking_yolo"
    CUSTOM_SNN = "custom_snn"


class SensorType(Enum):
    """Supported sensor types."""
    DVS128 = "DVS128"
    DVS240 = "DVS240"
    DAVIS346 = "DAVIS346"
    PROPHESEE = "Prophesee"


class LogLevel(Enum):
    """Supported log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class CameraConfiguration:
    """Camera configuration parameters."""
    sensor_type: SensorType = SensorType.DVS128
    noise_filter: bool = True
    refractory_period: float = 1e-3
    hot_pixel_threshold: int = 1000
    background_activity_filter: bool = True
    publish_rate: float = 30.0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.refractory_period <= 0:
            raise ConfigurationError("Refractory period must be positive")
        if self.hot_pixel_threshold <= 0:
            raise ConfigurationError("Hot pixel threshold must be positive")
        if self.publish_rate <= 0:
            raise ConfigurationError("Publish rate must be positive")


@dataclass
class ModelConfiguration:
    """Model configuration parameters."""
    model_type: ModelType = ModelType.SPIKING_YOLO
    model_path: Optional[str] = None
    pretrained_model: str = "yolo_v4_spiking_dvs"
    input_width: int = 128
    input_height: int = 128
    num_classes: int = 80
    time_steps: int = 10
    integration_time_ms: float = 10.0
    detection_threshold: float = 0.5
    device: DeviceType = DeviceType.CPU
    use_gpu: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.input_width <= 0 or self.input_height <= 0:
            raise ConfigurationError("Input dimensions must be positive")
        if self.num_classes <= 0:
            raise ConfigurationError("Number of classes must be positive")
        if self.time_steps <= 0:
            raise ConfigurationError("Time steps must be positive")
        if not (0.0 <= self.detection_threshold <= 1.0):
            raise ConfigurationError("Detection threshold must be between 0 and 1")
        if self.integration_time_ms <= 0:
            raise ConfigurationError("Integration time must be positive")


@dataclass
class SystemConfiguration:
    """Complete system configuration."""
    camera: CameraConfiguration = field(default_factory=CameraConfiguration)
    model: ModelConfiguration = field(default_factory=ModelConfiguration)
    
    # Global settings
    output_dir: str = "./output"
    data_dir: str = "./data"
    temp_dir: str = "./temp"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure directories are valid paths
        for dir_field in ['output_dir', 'data_dir', 'temp_dir']:
            dir_path = getattr(self, dir_field)
            if not isinstance(dir_path, str) or not dir_path.strip():
                raise ConfigurationError(f"{dir_field} must be a non-empty string")


class ConfigurationError(Exception):
    """Configuration validation error."""
    pass


def load_configuration(
    config_path: Optional[Union[str, Path]] = None,
    validate: bool = True
) -> SystemConfiguration:
    """Load and validate configuration."""
    if config_path:
        config_path = Path(config_path)
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration file: {e}")
            
        if not isinstance(config_data, dict):
            raise ConfigurationError("Configuration file must contain a dictionary")
    else:
        config_data = {}
    
    return SystemConfiguration(**config_data)


def save_configuration(config: SystemConfiguration, output_path: Union[str, Path]):
    """Save configuration to file."""
    output_path = Path(output_path)
    
    config_dict = {
        'camera': {
            'sensor_type': config.camera.sensor_type.value,
            'noise_filter': config.camera.noise_filter,
            'refractory_period': config.camera.refractory_period,
            'hot_pixel_threshold': config.camera.hot_pixel_threshold,
            'background_activity_filter': config.camera.background_activity_filter,
            'publish_rate': config.camera.publish_rate
        },
        'model': {
            'model_type': config.model.model_type.value,
            'model_path': config.model.model_path,
            'pretrained_model': config.model.pretrained_model,
            'input_width': config.model.input_width,
            'input_height': config.model.input_height,
            'num_classes': config.model.num_classes,
            'time_steps': config.model.time_steps,
            'integration_time_ms': config.model.integration_time_ms,
            'detection_threshold': config.model.detection_threshold,
            'device': config.model.device.value,
            'use_gpu': config.model.use_gpu
        },
        'output_dir': config.output_dir,
        'data_dir': config.data_dir,
        'temp_dir': config.temp_dir
    }
    
    with open(output_path, 'w') as f:
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            json.dump(config_dict, f, indent=2)
            
    logging.info(f"Configuration saved to {output_path}")


def create_default_config(output_path: Union[str, Path]):
    """Create a default configuration file."""
    default_config = SystemConfiguration()
    save_configuration(default_config, output_path)
    logging.info(f"Default configuration created at {output_path}")