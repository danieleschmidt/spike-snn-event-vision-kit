"""Basic tests for spike-snn-event-vision-kit."""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

import spike_snn_event
from spike_snn_event import (
    DVSCamera, 
    SpikingYOLO, 
    CustomSNN, 
    EventVisualizer,
    TrainingConfig,
    SpikingTrainer,
    load_events_from_file,
    save_events_to_file
)
from spike_snn_event.core import CameraConfig, SpatioTemporalPreprocessor


def test_version():
    """Test that version is accessible."""
    assert hasattr(spike_snn_event, '__version__')
    assert isinstance(spike_snn_event.__version__, str)


def test_package_attributes():
    """Test that basic package attributes are present."""
    assert hasattr(spike_snn_event, '__author__')
    assert hasattr(spike_snn_event, '__email__')
    assert hasattr(spike_snn_event, '__license__')
    
    assert spike_snn_event.__license__ == "MIT"


class TestDVSCamera:
    """Test DVS camera functionality."""
    
    def test_camera_initialization(self):
        """Test camera initialization with different sensors."""
        camera = DVSCamera(sensor_type="DVS128")
        assert camera.width == 128
        assert camera.height == 128
        assert camera.sensor_type == "DVS128"
        
    def test_camera_config(self):
        """Test camera with custom configuration."""
        config = CameraConfig(
            width=240,
            height=180,
            noise_filter=True,
            refractory_period=2e-3
        )
        camera = DVSCamera(sensor_type="DVS240", config=config)
        assert camera.config.refractory_period == 2e-3
        
    def test_event_generation(self):
        """Test synthetic event generation."""
        camera = DVSCamera()
        events = camera._generate_synthetic_events(100)
        
        assert len(events) == 100
        assert events.shape[1] == 4  # x, y, t, p
        assert np.all(events[:, 0] >= 0) and np.all(events[:, 0] < camera.width)
        assert np.all(events[:, 1] >= 0) and np.all(events[:, 1] < camera.height)
        assert np.all(np.abs(events[:, 3]) == 1)  # Polarity should be ±1


class TestSpikingModels:
    """Test spiking neural network models."""
    
    def test_spiking_yolo_initialization(self):
        """Test SpikingYOLO model initialization."""
        model = SpikingYOLO(
            input_size=(128, 128),
            num_classes=10,
            time_steps=5
        )
        
        assert model.input_size == (128, 128)
        assert model.num_classes == 10
        assert model.time_steps == 5
        
    def test_custom_snn_initialization(self):
        """Test CustomSNN model initialization."""
        model = CustomSNN(
            input_size=(64, 64),
            hidden_channels=[32, 64],
            output_classes=5
        )
        
        assert model.input_size == (64, 64)
        assert model.output_classes == 5
        
    def test_events_to_tensor_conversion(self):
        """Test conversion of events to tensor format."""
        model = SpikingYOLO(input_size=(32, 32))
        
        events = np.array([
            [10, 10, 0.001, 1],
            [15, 15, 0.005, -1],
            [20, 20, 0.009, 1]
        ])
        
        tensor = model.events_to_tensor(events, time_window=0.01)
        
        assert tensor.shape[0] == 1  # batch size
        assert tensor.shape[1] == 2  # polarities
        assert tensor.shape[2:4] == (32, 32)  # spatial dimensions
        assert tensor.shape[4] >= 1  # time bins


class TestFileIO:
    """Test file input/output functionality."""
    
    def test_save_load_events_npy(self, tmp_path):
        """Test saving and loading events in numpy format."""
        events = np.random.rand(100, 4)
        filepath = tmp_path / "test_events.npy"
        
        # Save events
        save_events_to_file(events, str(filepath))
        assert filepath.exists()
        
        # Load events
        loaded_events, metadata = load_events_from_file(str(filepath))
        
        np.testing.assert_array_equal(events, loaded_events)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_sensor_type(self):
        """Test handling of invalid sensor type."""
        with pytest.raises(ValueError, match="Unknown sensor type"):
            DVSCamera(sensor_type="INVALID_SENSOR")
            
    def test_empty_events_processing(self):
        """Test processing of empty event arrays."""
        camera = DVSCamera()
        empty_events = np.empty((0, 4))
        
        # Should handle empty events gracefully
        filtered = camera._apply_noise_filter(empty_events)
        assert len(filtered) == 0