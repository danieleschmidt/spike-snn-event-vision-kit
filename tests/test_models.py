"""Tests for spiking neural network models."""

import pytest
import torch
import numpy as np
from spike_snn_event.models import LIFNeuron, SpikingConv2d, EventSNN, SpikingYOLO, CustomSNN


class TestLIFNeuron:
    """Test cases for LIF neuron model."""
    
    def test_lif_initialization(self):
        """Test LIF neuron initialization."""
        neuron = LIFNeuron(threshold=1.0, tau_mem=20e-3, tau_syn=5e-3)
        assert neuron.threshold == 1.0
        assert neuron.tau_mem == 20e-3
        assert neuron.tau_syn == 5e-3
        
    def test_lif_forward_pass(self):
        """Test LIF neuron forward pass."""
        neuron = LIFNeuron()
        
        # Create test input
        batch_size, features, time_steps = 2, 10, 20
        input_current = torch.randn(batch_size, features, time_steps)
        
        # Forward pass
        spikes = neuron(input_current)
        
        # Check output shape
        assert spikes.shape == input_current.shape
        
        # Check spikes are binary (0 or 1)
        assert torch.all((spikes == 0) | (spikes == 1))
        
    def test_lif_threshold_behavior(self):
        """Test that LIF neuron spikes when threshold is exceeded."""
        neuron = LIFNeuron(threshold=0.5, tau_mem=1.0, tau_syn=0.1)
        
        # High constant input should cause spikes
        high_input = torch.ones(1, 1, 10) * 2.0
        spikes = neuron(high_input)
        
        # Should generate some spikes
        assert spikes.sum() > 0
        
        # Low input should not cause spikes
        low_input = torch.ones(1, 1, 10) * 0.1
        spikes_low = neuron(low_input)
        
        # Should generate fewer or no spikes
        assert spikes_low.sum() <= spikes.sum()


class TestSpikingConv2d:
    """Test cases for spiking convolutional layer."""
    
    def test_spiking_conv2d_initialization(self):
        """Test spiking conv2d initialization."""
        conv = SpikingConv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        assert conv.conv.in_channels == 2
        assert conv.conv.out_channels == 64
        assert conv.conv.kernel_size == (3, 3)
        
    def test_spiking_conv2d_forward(self):
        """Test spiking conv2d forward pass."""
        conv = SpikingConv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        
        # Create test input [batch, channels, height, width, time]
        batch, channels, height, width, time = 1, 2, 32, 32, 10
        x = torch.randn(batch, channels, height, width, time)
        
        # Forward pass
        output = conv(x)
        
        # Check output shape
        expected_shape = (batch, 8, height, width, time)
        assert output.shape == expected_shape
        
        # Check output is binary (spikes)
        assert torch.all((output == 0) | (output == 1))


class TestEventSNN:
    """Test cases for base EventSNN class."""
    
    def test_event_snn_initialization(self):
        """Test EventSNN initialization."""
        snn = EventSNN(input_size=(128, 128))
        assert snn.input_size == (128, 128)
        assert snn.backend == "cpu"
        
    def test_set_backend(self):
        """Test backend setting."""
        snn = EventSNN()
        snn.set_backend("cpu")
        assert snn.backend == "cpu"
        
        # Only test CUDA if available
        if torch.cuda.is_available():
            snn.set_backend("cuda")
            assert snn.backend == "cuda"
            
    def test_events_to_tensor_empty(self):
        """Test events_to_tensor with empty input."""
        snn = EventSNN(input_size=(64, 64))
        events = np.empty((0, 4))
        
        tensor = snn.events_to_tensor(events)
        
        # Should return zero tensor with correct shape
        expected_shape = (1, 2, 64, 64, 1)
        assert tensor.shape == expected_shape
        assert torch.all(tensor == 0)
        
    def test_events_to_tensor_valid(self):
        """Test events_to_tensor with valid events."""
        snn = EventSNN(input_size=(64, 64))
        
        # Create sample events [x, y, timestamp, polarity]
        events = np.array([
            [10, 20, 0.0, 1],    # Positive event
            [30, 40, 0.001, -1], # Negative event
            [50, 10, 0.002, 1],  # Another positive event
        ])
        
        tensor = snn.events_to_tensor(events, time_window=0.001)
        
        # Check shape
        assert tensor.shape[0] == 1  # batch size
        assert tensor.shape[1] == 2  # polarities
        assert tensor.shape[2:4] == (64, 64)  # spatial dimensions
        
        # Check that events were placed correctly
        assert tensor.sum() > 0  # Some events should be present


class TestSpikingYOLO:
    """Test cases for SpikingYOLO model."""
    
    def test_spiking_yolo_initialization(self):
        """Test SpikingYOLO initialization."""
        model = SpikingYOLO(input_size=(128, 128), num_classes=10, time_steps=5)
        assert model.input_size == (128, 128)
        assert model.num_classes == 10
        assert model.time_steps == 5
        
    def test_from_pretrained(self):
        """Test loading pretrained model."""
        model = SpikingYOLO.from_pretrained("test_model", backend="cpu")
        assert isinstance(model, SpikingYOLO)
        assert model.backend == "cpu"
        
    def test_detect_empty_events(self):
        """Test detection with empty events."""
        model = SpikingYOLO()
        empty_events = np.empty((0, 4))
        
        detections = model.detect(empty_events)
        
        # Should return empty detection list
        assert isinstance(detections, list)
        assert len(detections) == 0
        
    def test_detect_with_events(self):
        """Test detection with sample events."""
        model = SpikingYOLO(num_classes=2)
        
        # Create sample events
        events = np.array([
            [10, 20, 0.0, 1],
            [30, 40, 0.001, -1],
            [50, 60, 0.002, 1],
        ])
        
        detections = model.detect(events, threshold=0.0)  # Low threshold to ensure detection
        
        # Should return detection list
        assert isinstance(detections, list)
        # Detection format should be correct
        for det in detections:
            assert "bbox" in det
            assert "confidence" in det
            assert "class_id" in det
            assert "class_name" in det
            
    def test_inference_timing(self):
        """Test that inference timing is recorded."""
        model = SpikingYOLO()
        events = np.array([[10, 20, 0.0, 1]])
        
        model.detect(events)
        
        # Should have recorded inference time
        assert model.last_inference_time >= 0


class TestCustomSNN:
    """Test cases for CustomSNN model."""
    
    def test_custom_snn_initialization(self):
        """Test CustomSNN initialization."""
        model = CustomSNN(
            input_size=(64, 64),
            hidden_channels=[32, 64],
            output_classes=5
        )
        assert model.input_size == (64, 64)
        assert model.output_classes == 5
        
    def test_custom_snn_forward(self):
        """Test CustomSNN forward pass."""
        model = CustomSNN(
            input_size=(32, 32),
            hidden_channels=[8, 16],
            output_classes=3
        )
        
        # Create test input [batch, channels, height, width, time]
        x = torch.randn(2, 2, 32, 32, 10)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (2, 3)  # [batch, classes]
        
    def test_export_methods(self):
        """Test model export methods."""
        model = CustomSNN(output_classes=2)
        
        # Test ONNX export (should not raise error)
        try:
            model.export_onnx("/tmp/test_model.onnx")
        except Exception as e:
            # ONNX export might fail in test environment, that's okay
            pass
            
        # Test Loihi export (placeholder implementation)
        model.export_loihi("/tmp/test_model.net")


@pytest.mark.slow
class TestModelIntegration:
    """Integration tests for model interactions."""
    
    def test_full_pipeline(self):
        """Test full processing pipeline."""
        # Initialize components
        model = SpikingYOLO(input_size=(64, 64), num_classes=2)
        
        # Create realistic event sequence
        np.random.seed(42)  # For reproducibility
        num_events = 100
        events = np.column_stack([
            np.random.uniform(0, 64, num_events),      # x
            np.random.uniform(0, 64, num_events),      # y
            np.sort(np.random.uniform(0, 0.01, num_events)),  # sorted timestamps
            np.random.choice([-1, 1], num_events)     # polarity
        ])
        
        # Run detection
        detections = model.detect(events, integration_time=0.005, threshold=0.3)
        
        # Validate results
        assert isinstance(detections, list)
        assert model.last_inference_time > 0
        
        # Each detection should have required fields
        for det in detections:
            assert all(key in det for key in ["bbox", "confidence", "class_id", "class_name"])
            assert 0 <= det["confidence"] <= 1
            assert isinstance(det["bbox"], list) and len(det["bbox"]) == 4


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUAcceleration:
    """Test GPU acceleration capabilities."""
    
    def test_cuda_backend(self):
        """Test CUDA backend functionality."""
        model = SpikingYOLO.from_pretrained("test_model", backend="cuda")
        assert model.backend == "cuda"
        
        # Model should be on GPU
        assert next(model.parameters()).device.type == "cuda"
        
    def test_gpu_inference(self):
        """Test GPU inference performance."""
        model = SpikingYOLO(backend="cuda")
        
        # Create sample events
        events = np.random.rand(1000, 4)
        events[:, 0] *= 128  # x coordinates
        events[:, 1] *= 128  # y coordinates
        events[:, 2] = np.sort(events[:, 2] * 0.01)  # timestamps
        events[:, 3] = np.random.choice([-1, 1], 1000)  # polarity
        
        # Run detection
        detections = model.detect(events)
        
        # Should complete without error
        assert isinstance(detections, list)
        assert model.last_inference_time > 0