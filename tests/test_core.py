"""Tests for core event processing functionality."""

import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock

from spike_snn_event.core import (
    DVSCamera, 
    EventPreprocessor,
    SpatioTemporalPreprocessor, 
    HotPixelFilter,
    EventDataset,
    load_events_from_file,
    save_events_to_file
)


class TestDVSCamera:
    """Test cases for DVS camera interface."""
    
    def test_dvs_camera_initialization(self):
        """Test DVS camera initialization."""
        camera = DVSCamera(sensor_type="DVS128", noise_filter=True)
        assert camera.sensor_type == "DVS128"
        assert camera.noise_filter is True
        assert camera.width == 128
        assert camera.height == 128
        
    def test_unsupported_sensor_type(self):
        """Test error handling for unsupported sensor types."""
        with pytest.raises(ValueError, match="Unknown sensor type"):
            DVSCamera(sensor_type="INVALID_SENSOR")
            
    def test_sensor_specifications(self):
        """Test different sensor specifications."""
        # Test DVS240
        camera_240 = DVSCamera(sensor_type="DVS240")
        assert camera_240.width == 240
        assert camera_240.height == 180
        
        # Test DAVIS346
        camera_346 = DVSCamera(sensor_type="DAVIS346")
        assert camera_346.width == 346
        assert camera_346.height == 240
        
        # Test Prophesee
        camera_prophesee = DVSCamera(sensor_type="Prophesee")
        assert camera_prophesee.width == 640
        assert camera_prophesee.height == 480
        
    def test_stream_duration_limit(self):
        """Test streaming with duration limit."""
        camera = DVSCamera(sensor_type="DVS128")
        
        start_time = time.time()
        event_batches = []
        
        # Stream for short duration
        for events in camera.stream(duration=0.05):  # 50ms
            event_batches.append(events)
            if len(event_batches) >= 10:  # Safety limit
                break
                
        end_time = time.time()
        
        # Should respect duration approximately
        assert end_time - start_time >= 0.04  # Allow some tolerance
        assert len(event_batches) > 0
        
    def test_synthetic_event_generation(self):
        """Test synthetic event generation."""
        camera = DVSCamera(sensor_type="DVS128")
        events = camera._generate_synthetic_events(100)
        
        assert events.shape == (100, 4)
        
        # Check coordinate bounds
        assert np.all(events[:, 0] >= 0) and np.all(events[:, 0] < 128)  # x
        assert np.all(events[:, 1] >= 0) and np.all(events[:, 1] < 128)  # y
        
        # Check polarity values
        assert np.all(np.isin(events[:, 3], [-1, 1]))  # polarity
        
        # Check timestamp ordering
        assert np.all(events[1:, 2] >= events[:-1, 2])  # timestamps sorted
        
    def test_noise_filtering(self):
        """Test refractory period noise filtering."""
        camera = DVSCamera(sensor_type="DVS128", refractory_period=0.001)
        
        # Create events with some at same pixel location
        events = np.array([
            [10, 20, 0.000, 1],  # First event
            [10, 20, 0.0005, 1], # Same pixel, within refractory period
            [10, 20, 0.002, 1],  # Same pixel, after refractory period
            [30, 40, 0.001, -1], # Different pixel
        ])
        
        filtered = camera._apply_noise_filter(events)
        
        # Should filter out the second event (within refractory period)
        assert len(filtered) == 3
        
        # Check that correct events are kept
        expected_times = [0.000, 0.002, 0.001]
        assert np.allclose(filtered[:, 2], expected_times, atol=1e-6)
        
    def test_visualization_placeholder(self):
        """Test visualization method (placeholder implementation)."""
        camera = DVSCamera(sensor_type="DVS128")
        events = np.array([[10, 20, 0.0, 1]])
        detections = [{"bbox": [5, 5, 10, 10], "confidence": 0.8, "class_name": "test"}]
        
        # Should not raise error
        camera.visualize_detections(events, detections)


class TestSpatioTemporalPreprocessor:
    """Test cases for spatiotemporal preprocessing."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        processor = SpatioTemporalPreprocessor(
            spatial_size=(64, 64), 
            time_bins=10
        )
        assert processor.spatial_size == (64, 64)
        assert processor.time_bins == 10
        
    def test_process_empty_events(self):
        """Test processing empty event array."""
        processor = SpatioTemporalPreprocessor()
        empty_events = np.empty((0, 4))
        
        result = processor.process(empty_events)
        
        # Should handle empty input gracefully
        assert isinstance(result, np.ndarray)
        
    def test_spatial_downsampling(self):
        """Test spatial downsampling functionality."""
        processor = SpatioTemporalPreprocessor(spatial_size=(32, 32))
        
        # Create events with coordinates up to 64x64
        events = np.array([
            [63, 63, 0.0, 1],  # Max coordinates
            [0, 0, 0.001, -1], # Min coordinates
            [32, 32, 0.002, 1], # Middle coordinates
        ])
        
        downsampled = processor.spatial_downsample(events, (32, 32))
        
        # Check that coordinates are within new bounds
        assert np.all(downsampled[:, 0] < 32)  # x coordinates
        assert np.all(downsampled[:, 1] < 32)  # y coordinates
        assert np.all(downsampled[:, 0] >= 0)
        assert np.all(downsampled[:, 1] >= 0)
        
    def test_events_to_frames(self):
        """Test event to frame conversion."""
        processor = SpatioTemporalPreprocessor(spatial_size=(32, 32), time_bins=5)
        
        # Create events spanning some time
        events = np.array([
            [10, 15, 0.000, 1],
            [20, 25, 0.002, -1],
            [15, 20, 0.004, 1],
        ])
        
        frames = processor.events_to_frames(events, num_bins=3)
        
        # Check frame shape
        expected_shape = (3, 32, 32, 2)  # (time_bins, height, width, polarities)
        assert frames.shape == expected_shape
        
        # Should have some non-zero values where events occurred
        assert frames.sum() > 0
        
    def test_frames_to_spikes_rate_encoding(self):
        """Test frame to spike conversion with rate encoding."""
        processor = SpatioTemporalPreprocessor()
        
        # Create sample frames
        frames = np.random.rand(5, 32, 32, 2) * 10  # Random frame data
        
        spikes = processor.frames_to_spikes(frames, encoding="rate")
        
        # Check shape preservation
        assert spikes.shape == frames.shape
        
        # Rate encoding should normalize values to [0, 1]
        assert np.all(spikes >= 0) and np.all(spikes <= 1)
        
    def test_frames_to_spikes_unsupported_encoding(self):
        """Test error for unsupported encoding."""
        processor = SpatioTemporalPreprocessor()
        frames = np.zeros((5, 32, 32, 2))
        
        with pytest.raises(NotImplementedError):
            processor.frames_to_spikes(frames, encoding="unsupported")


class TestHotPixelFilter:
    """Test cases for hot pixel filtering."""
    
    def test_hot_pixel_filter_initialization(self):
        """Test hot pixel filter initialization."""
        filter_obj = HotPixelFilter(threshold=500)
        assert filter_obj.threshold == 500
        assert len(filter_obj.pixel_counts) == 0
        
    def test_hot_pixel_filtering(self):
        """Test hot pixel filtering functionality."""
        filter_obj = HotPixelFilter(threshold=2)
        
        # Create events with repeated hot pixel
        events = np.array([
            [10, 20, 0.000, 1],  # First event at (10, 20)
            [10, 20, 0.001, 1],  # Second event at (10, 20)
            [10, 20, 0.002, 1],  # Third event at (10, 20) - should be filtered
            [30, 40, 0.003, -1], # Different pixel, should pass
        ])
        
        filtered = filter_obj(events)
        
        # Should filter out the third event (exceeds threshold)
        assert len(filtered) == 3
        
        # Check pixel counts were updated
        assert filter_obj.pixel_counts[(10, 20)] == 3
        assert filter_obj.pixel_counts[(30, 40)] == 1
        
    def test_empty_events_handling(self):
        """Test handling of empty event arrays."""
        filter_obj = HotPixelFilter()
        empty_events = np.empty((0, 4))
        
        result = filter_obj(empty_events)
        
        assert result.shape == (0, 4)


class TestEventDataset:
    """Test cases for event dataset loading."""
    
    def test_load_known_dataset(self):
        """Test loading known datasets."""
        # These are placeholder implementations
        dataset = EventDataset.load("N-CARS")
        assert isinstance(dataset, EventDataset)
        
        dataset = EventDataset.load("N-Caltech101")
        assert isinstance(dataset, EventDataset)
        
    def test_load_unknown_dataset(self):
        """Test error handling for unknown datasets."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            EventDataset.load("UNKNOWN_DATASET")
            
    def test_get_loaders(self):
        """Test data loader creation."""
        dataset = EventDataset.load("N-CARS")
        train_loader, val_loader = dataset.get_loaders(batch_size=16)
        
        # Placeholder implementation returns None
        assert train_loader is None
        assert val_loader is None


class TestFileOperations:
    """Test cases for file I/O operations."""
    
    def test_load_save_npy_format(self, tmp_path):
        """Test loading and saving NPY format."""
        # Create test events
        events = np.array([
            [10, 20, 0.0, 1],
            [30, 40, 0.001, -1],
        ])
        
        # Save to file
        filepath = tmp_path / "test_events.npy"
        save_events_to_file(events, str(filepath))
        
        # Load from file
        loaded_events = load_events_from_file(str(filepath))
        
        # Should match original
        np.testing.assert_array_equal(events, loaded_events)
        
    def test_load_save_txt_format(self, tmp_path):
        """Test loading and saving TXT format."""
        events = np.array([
            [10, 20, 0.0, 1],
            [30, 40, 0.001, -1],
        ])
        
        # Save to file
        filepath = tmp_path / "test_events.txt"
        save_events_to_file(events, str(filepath))
        
        # Load from file
        loaded_events = load_events_from_file(str(filepath))
        
        # Should match original (with potential floating point precision)
        np.testing.assert_array_almost_equal(events, loaded_events)
        
    def test_unsupported_format_load(self, tmp_path):
        """Test error handling for unsupported file formats."""
        filepath = tmp_path / "test_events.unknown"
        filepath.write_text("dummy content")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_events_from_file(str(filepath))
            
    def test_unsupported_format_save(self):
        """Test error handling for unsupported save formats."""
        events = np.array([[1, 2, 3, 4]])
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            save_events_to_file(events, "test.unknown")
            
    def test_dat_format_not_implemented(self, tmp_path):
        """Test DAT format raises NotImplementedError."""
        filepath = tmp_path / "test_events.dat"
        filepath.write_bytes(b"dummy binary content")
        
        with pytest.raises(NotImplementedError, match="DAT format not yet implemented"):
            load_events_from_file(str(filepath))


@pytest.mark.integration
class TestPreprocessingPipeline:
    """Integration tests for preprocessing pipeline."""
    
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing workflow."""
        # Initialize components
        camera = DVSCamera(sensor_type="DVS128", noise_filter=True)
        processor = SpatioTemporalPreprocessor(spatial_size=(64, 64), time_bins=5)
        
        # Generate synthetic events
        events = camera._generate_synthetic_events(200)
        
        # Apply preprocessing
        processed = processor.process(events)
        
        # Validate output
        assert isinstance(processed, np.ndarray)
        assert processed.shape == (5, 64, 64, 2)  # time_bins, height, width, polarities
        
        # Should have some activity
        assert processed.sum() > 0
        
    def test_preprocessing_with_real_event_structure(self):
        """Test preprocessing with realistic event patterns."""
        processor = SpatioTemporalPreprocessor(spatial_size=(32, 32), time_bins=3)
        
        # Create events simulating a moving object
        t_start = 0.0
        events = []
        
        # Moving object trace
        for i in range(20):
            x = 5 + i  # Moving horizontally
            y = 10 + np.sin(i * 0.5) * 3  # Sinusoidal vertical movement
            t = t_start + i * 0.001
            
            events.append([x, y, t, 1])  # Positive polarity
            
        events = np.array(events)
        
        # Process events
        result = processor.process(events)
        
        # Should produce reasonable output
        assert result.shape == (3, 32, 32, 2)
        assert result.sum() > 0
        
        # Should have activity in the expected spatial region
        activity_region = result[:, 5:25, 5:15, :]  # Region where movement occurred
        assert activity_region.sum() > 0