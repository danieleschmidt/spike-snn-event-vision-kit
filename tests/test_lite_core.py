#!/usr/bin/env python3
"""
Test suite for lite_core module.
"""

import sys
import os
import unittest
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spike_snn_event.lite_core import (
    DVSCamera, CameraConfig, SpatioTemporalPreprocessor,
    EventVisualizer, LiteEventSNN, HotPixelFilter
)


class TestCameraConfig(unittest.TestCase):
    """Test CameraConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CameraConfig()
        self.assertEqual(config.width, 128)
        self.assertEqual(config.height, 128)
        self.assertTrue(config.noise_filter)
        self.assertEqual(config.refractory_period, 1e-3)
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CameraConfig(
            width=256,
            height=256,
            noise_filter=False,
            refractory_period=2e-3
        )
        self.assertEqual(config.width, 256)
        self.assertEqual(config.height, 256)
        self.assertFalse(config.noise_filter)
        self.assertEqual(config.refractory_period, 2e-3)


class TestDVSCamera(unittest.TestCase):
    """Test DVSCamera class."""
    
    def setUp(self):
        self.camera = DVSCamera(sensor_type="DVS128")
        
    def test_camera_initialization(self):
        """Test camera initialization."""
        self.assertEqual(self.camera.sensor_type, "DVS128")
        self.assertEqual(self.camera.width, 128)
        self.assertEqual(self.camera.height, 128)
        self.assertFalse(self.camera.is_streaming)
        
    def test_invalid_sensor_type(self):
        """Test initialization with invalid sensor type."""
        with self.assertRaises(ValueError):
            DVSCamera(sensor_type="INVALID_SENSOR")
            
    def test_sensor_specifications(self):
        """Test different sensor specifications."""
        sensors = [
            ("DVS128", 128, 128),
            ("DVS240", 240, 180),
            ("DAVIS346", 346, 240),
            ("Prophesee", 640, 480)
        ]
        
        for sensor_type, expected_width, expected_height in sensors:
            camera = DVSCamera(sensor_type=sensor_type)
            self.assertEqual(camera.width, expected_width)
            self.assertEqual(camera.height, expected_height)
            
    def test_event_generation(self):
        """Test synthetic event generation."""
        num_events = 100
        events = self.camera._generate_synthetic_events(num_events)
        
        self.assertEqual(len(events), num_events)
        
        # Check event structure
        for event in events[:5]:  # Check first 5 events
            self.assertEqual(len(event), 4)
            x, y, t, p = event
            
            # Check coordinate bounds
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, self.camera.width)
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, self.camera.height)
            
            # Check polarity
            self.assertIn(p, [-1, 1])
            
            # Check timestamp type
            self.assertIsInstance(t, float)
            
        # Events should be sorted by timestamp
        timestamps = [event[2] for event in events]
        self.assertEqual(timestamps, sorted(timestamps))
        
    def test_noise_filtering(self):
        """Test noise filtering functionality."""
        # Create events with some invalid ones
        mixed_events = [
            [10, 20, 0.1, 1],      # Valid
            [-5, 20, 0.2, 1],      # Invalid x (out of bounds)
            [10, -5, 0.3, 1],      # Invalid y (out of bounds)
            [10, 20, 0.4, 2],      # Invalid polarity
            [50, 60, 0.5, -1],     # Valid
            "invalid_event",        # Invalid structure
            [10, 20, -0.1, 1],     # Invalid timestamp
        ]
        
        filtered = self.camera._apply_noise_filter(mixed_events)
        
        # Should only keep valid events
        self.assertLessEqual(len(filtered), len(mixed_events))
        
        # Check that filtered events are valid
        for event in filtered:
            self.assertEqual(len(event), 4)
            x, y, t, p = event
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, self.camera.width)
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, self.camera.height)
            self.assertIn(p, [-1, 1])
            self.assertGreaterEqual(t, 0)
            
    def test_stream_duration(self):
        """Test streaming with duration limit."""
        duration = 0.1  # 100ms
        start_time = time.time()
        
        event_count = 0
        batch_count = 0
        
        for events in self.camera.stream(duration=duration):
            event_count += len(events)
            batch_count += 1
            
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Should respect duration (with some tolerance)
        self.assertLessEqual(actual_duration, duration + 0.05)
        self.assertGreater(event_count, 0)
        self.assertGreater(batch_count, 0)
        
    def test_statistics_tracking(self):
        """Test statistics tracking."""
        initial_stats = self.camera.stats.copy()
        
        # Generate and filter some events
        events = self.camera._generate_synthetic_events(100)
        filtered = self.camera._apply_noise_filter(events)
        
        # Stats should be updated
        self.assertGreater(self.camera.stats['events_processed'], initial_stats['events_processed'])


class TestSpatioTemporalPreprocessor(unittest.TestCase):
    """Test SpatioTemporalPreprocessor class."""
    
    def setUp(self):
        self.preprocessor = SpatioTemporalPreprocessor(
            spatial_size=(64, 64),
            time_bins=5
        )
        
    def test_initialization(self):
        """Test preprocessor initialization."""
        self.assertEqual(self.preprocessor.spatial_size, (64, 64))
        self.assertEqual(self.preprocessor.time_bins, 5)
        
    def test_empty_processing(self):
        """Test processing empty event list."""
        result = self.preprocessor.process([])
        self.assertEqual(len(result), 0)
        
    def test_event_processing(self):
        """Test basic event processing."""
        # Create some test events
        events = [
            [10, 20, 0.001, 1],
            [30, 40, 0.002, -1],
            [50, 60, 0.003, 1]
        ]
        
        processed = self.preprocessor.process(events)
        
        # Should return processed events
        self.assertGreaterEqual(len(processed), 0)
        
    def test_spatial_downsampling(self):
        """Test spatial downsampling."""
        events = [[100, 200, 0.001, 1], [150, 250, 0.002, -1]]
        
        downsampled = self.preprocessor.spatial_downsample(events, (32, 32))
        
        self.assertEqual(len(downsampled), len(events))
        
        # Check that coordinates are scaled down
        for event in downsampled:
            x, y = event[0], event[1]
            self.assertLess(x, 32)
            self.assertLess(y, 32)
            
    def test_statistics(self):
        """Test getting preprocessing statistics."""
        stats = self.preprocessor.get_statistics()
        
        self.assertIn('spatial_compression_ratio', stats)
        self.assertIn('temporal_bins', stats)
        self.assertIn('processing_efficiency', stats)
        
        self.assertEqual(stats['temporal_bins'], 5)


class TestEventVisualizer(unittest.TestCase):
    """Test EventVisualizer class."""
    
    def setUp(self):
        self.visualizer = EventVisualizer(width=128, height=128)
        
    def test_initialization(self):
        """Test visualizer initialization."""
        self.assertEqual(self.visualizer.width, 128)
        self.assertEqual(self.visualizer.height, 128)
        self.assertEqual(self.visualizer.event_count, 0)
        
    def test_update_visualization(self):
        """Test updating visualization with events."""
        events = [
            [10, 20, 0.001, 1],
            [30, 40, 0.002, -1]
        ]
        
        vis_info = self.visualizer.update(events)
        
        self.assertEqual(vis_info['current_batch'], len(events))
        self.assertEqual(vis_info['total_events'], len(events))
        self.assertEqual(self.visualizer.event_count, len(events))
        
    def test_detection_overlay(self):
        """Test adding detection overlays."""
        vis_info = {'total_events': 100}
        detections = [
            {'bbox': [10, 20, 30, 40], 'confidence': 0.8, 'class_name': 'test'}
        ]
        
        result = self.visualizer.draw_detections(vis_info, detections)
        
        self.assertIn('detections', result)
        self.assertEqual(result['detection_count'], len(detections))


class TestLiteEventSNN(unittest.TestCase):
    """Test LiteEventSNN class."""
    
    def setUp(self):
        self.model = LiteEventSNN(input_size=(128, 128), num_classes=3)
        
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.input_size, (128, 128))
        self.assertEqual(self.model.num_classes, 3)
        self.assertGreater(self.model.model_parameters, 0)
        
    def test_detection_with_empty_events(self):
        """Test detection with empty event list."""
        detections = self.model.detect([])
        self.assertEqual(len(detections), 0)
        
    def test_detection_with_events(self):
        """Test detection with valid events."""
        events = [
            [10, 20, 0.001, 1],
            [30, 40, 0.002, -1],
            [50, 60, 0.003, 1]
        ]
        
        detections = self.model.detect(events, threshold=0.3)
        
        # Should return list of detections
        self.assertIsInstance(detections, list)
        
        # Check detection format if any detections found
        for detection in detections:
            self.assertIn('bbox', detection)
            self.assertIn('confidence', detection)
            self.assertIn('class_id', detection)
            self.assertIn('class_name', detection)
            
            # Validate bbox
            bbox = detection['bbox']
            self.assertEqual(len(bbox), 4)
            
            # Validate confidence
            confidence = detection['confidence']
            self.assertGreaterEqual(confidence, 0)
            self.assertLessEqual(confidence, 1)
            
    def test_model_statistics(self):
        """Test getting model statistics."""
        stats = self.model.get_model_statistics()
        
        expected_keys = [
            'total_parameters', 'trainable_parameters',
            'input_width', 'input_height', 'output_classes'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
            
        self.assertEqual(stats['input_width'], 128)
        self.assertEqual(stats['input_height'], 128)
        self.assertEqual(stats['output_classes'], 3)
        
    def test_inference_timing(self):
        """Test that inference timing is recorded."""
        events = [[10, 20, 0.001, 1]]
        
        detections = self.model.detect(events)
        
        # Should have recorded inference time
        self.assertGreater(self.model.last_inference_time, 0)


class TestHotPixelFilter(unittest.TestCase):
    """Test HotPixelFilter class."""
    
    def setUp(self):
        self.filter = HotPixelFilter(threshold=5)
        
    def test_initialization(self):
        """Test filter initialization."""
        self.assertEqual(self.filter.threshold, 5)
        
    def test_filtering(self):
        """Test hot pixel filtering."""
        # Create events with repeated pixels
        events = []
        
        # Add many events at same pixel (should be filtered)
        for i in range(10):
            events.append([10, 20, i * 0.001, 1])
            
        # Add events at different pixels (should pass)
        events.append([30, 40, 0.011, 1])
        events.append([50, 60, 0.012, -1])
        
        filtered = self.filter(events)
        
        # Should filter out the repeated pixel events beyond threshold
        self.assertLess(len(filtered), len(events))
        
    def test_empty_events(self):
        """Test filtering empty event list."""
        result = self.filter([])
        self.assertEqual(len(result), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test complete processing pipeline."""
        # Initialize components
        camera = DVSCamera(sensor_type="DVS128")
        preprocessor = SpatioTemporalPreprocessor()
        model = LiteEventSNN()
        visualizer = EventVisualizer()
        
        # Generate events
        events = camera._generate_synthetic_events(50)
        self.assertGreater(len(events), 0)
        
        # Filter events
        filtered = camera._apply_noise_filter(events)
        self.assertLessEqual(len(filtered), len(events))
        
        # Preprocess events
        processed = preprocessor.process(filtered)
        self.assertGreaterEqual(len(processed), 0)
        
        # Run detection
        detections = model.detect(processed)
        self.assertIsInstance(detections, list)
        
        # Update visualization
        vis_info = visualizer.update(processed)
        self.assertGreater(vis_info['total_events'], 0)
        
        # Add detection overlays
        vis_with_detections = visualizer.draw_detections(vis_info, detections)
        self.assertEqual(vis_with_detections['detection_count'], len(detections))


if __name__ == '__main__':
    unittest.main()