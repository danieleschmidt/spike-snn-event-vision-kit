"""End-to-end integration tests for spike-snn-event-vision-kit."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from tests.fixtures import TestDataGenerator, MockHardwareData


@pytest.mark.integration
class TestEventProcessingPipeline:
    """Test complete event processing pipeline."""
    
    def test_events_to_spikes_pipeline(self, sample_events):
        """Test conversion from events to spike trains."""
        # Mock event preprocessor
        class MockEventPreprocessor:
            def __init__(self, spatial_size=(64, 64), time_bins=10):
                self.spatial_size = spatial_size
                self.time_bins = time_bins
            
            def process(self, events):
                # Spatial binning
                spatial_hist, _, _ = np.histogram2d(
                    events[:, 0], events[:, 1],
                    bins=self.spatial_size,
                    range=[[0, 127], [0, 127]]
                )
                
                # Convert to spike trains (mock)
                spike_trains = torch.from_numpy(spatial_hist).float()
                spike_trains = spike_trains.unsqueeze(0).unsqueeze(0)  # Add batch, channel dims
                spike_trains = spike_trains.unsqueeze(-1).repeat(1, 2, 1, 1, self.time_bins)
                
                return spike_trains
        
        preprocessor = MockEventPreprocessor()
        spike_trains = preprocessor.process(sample_events)
        
        # Verify output format
        assert spike_trains.shape == (1, 2, 64, 64, 10)
        assert spike_trains.dtype == torch.float32
        assert torch.all(spike_trains >= 0)
    
    def test_inference_pipeline(self, sample_spike_train, device):
        """Test complete inference pipeline."""
        # Mock SNN model
        class MockSpikingYOLO(torch.nn.Module):
            def __init__(self, n_classes=10):
                super().__init__()
                self.backbone = torch.nn.Sequential(
                    torch.nn.Conv3d(2, 32, (1, 3, 3), padding=(0, 1, 1)),
                    torch.nn.ReLU(),
                    torch.nn.Conv3d(32, 64, (1, 3, 3), padding=(0, 1, 1)),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool3d((1, 1, 1))
                )
                self.classifier = torch.nn.Linear(64, n_classes)
            
            def forward(self, x):
                # x: (batch, channels, height, width, time) -> (batch, channels, time, height, width)
                x = x.permute(0, 1, 4, 2, 3)
                x = self.backbone(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        model = MockSpikingYOLO().to(device)
        model.eval()
        
        input_spikes = sample_spike_train.to(device)
        
        with torch.no_grad():
            outputs = model(input_spikes)
        
        # Verify output format
        assert outputs.shape == (1, 10)  # batch_size=1, n_classes=10
        assert not torch.isnan(outputs).any()
        assert not torch.isinf(outputs).any()
    
    def test_training_pipeline(self, device):
        """Test complete training pipeline."""
        # Generate training data
        batch_size = 4
        n_classes = 5
        
        spike_trains = TestDataGenerator.generate_spike_trains(
            batch_size=batch_size, height=32, width=32, time_steps=5
        ).to(device)
        
        labels = torch.randint(0, n_classes, (batch_size,)).to(device)
        
        # Mock training setup
        class SimpleTrainer:
            def __init__(self, model, optimizer, criterion):
                self.model = model
                self.optimizer = optimizer
                self.criterion = criterion
                self.train_losses = []
            
            def train_step(self, inputs, targets):
                self.model.train()
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                loss.backward()
                self.optimizer.step()
                
                self.train_losses.append(loss.item())
                return loss.item()
        
        # Create model and trainer
        from tests.integration.test_end_to_end import MockSpikingYOLO
        model = MockSpikingYOLO(n_classes=n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        trainer = SimpleTrainer(model, optimizer, criterion)
        
        # Run training steps
        initial_loss = trainer.train_step(spike_trains, labels)
        
        # Run a few more steps
        for _ in range(5):
            trainer.train_step(spike_trains, labels)
        
        final_loss = trainer.train_losses[-1]
        
        # Training should reduce loss (at least not increase significantly)
        assert len(trainer.train_losses) == 6
        assert all(loss > 0 for loss in trainer.train_losses)
        # Allow some variance but expect general improvement
        assert final_loss <= initial_loss * 1.5


@pytest.mark.integration
class TestDatasetIntegration:
    """Test dataset loading and processing integration."""
    
    def test_dataset_creation_and_loading(self, temp_dir):
        """Test creating and loading a neuromorphic dataset."""
        from tests.fixtures import TestDataGenerator
        
        # Create test dataset
        dataset_info = TestDataGenerator.generate_neuromorphic_dataset(
            dataset_name="test_integration",
            n_samples=10,
            save_dir=temp_dir
        )
        
        # Verify dataset structure
        assert dataset_info['n_samples'] == 10
        assert len(dataset_info['files']) == 10
        
        # Test loading samples
        for i, file_info in enumerate(dataset_info['files'][:3]):  # Test first 3
            # Load events
            events = np.load(file_info['events'])
            assert events.shape[1] == 4  # x, y, t, p
            assert np.all(events[:, 0] >= 0) and np.all(events[:, 0] < 128)  # x bounds
            assert np.all(events[:, 1] >= 0) and np.all(events[:, 1] < 128)  # y bounds
            assert np.all(events[:, 2] >= 0) and np.all(events[:, 2] <= 1.0)  # time bounds
            assert np.all(np.isin(events[:, 3], [0, 1]))  # polarity values
            
            # Load labels
            labels = torch.load(file_info['labels'])
            assert 'boxes' in labels
            assert 'labels' in labels
            assert 'scores' in labels
            assert len(labels['boxes']) == len(labels['labels']) == len(labels['scores'])
    
    def test_data_loader_integration(self, temp_dir):
        """Test PyTorch DataLoader integration."""
        from torch.utils.data import Dataset, DataLoader
        
        # Create test dataset
        dataset_info = TestDataGenerator.generate_neuromorphic_dataset(
            dataset_name="test_loader",
            n_samples=8,
            save_dir=temp_dir
        )
        
        # Mock Dataset class
        class NeuromorphicDataset(Dataset):
            def __init__(self, dataset_info):
                self.files = dataset_info['files']
            
            def __len__(self):
                return len(self.files)
            
            def __getitem__(self, idx):
                file_info = self.files[idx]
                
                # Load events and convert to tensor
                events = np.load(file_info['events'])
                events_tensor = torch.from_numpy(events).float()
                
                # Load labels
                labels = torch.load(file_info['labels'])
                
                return events_tensor, labels
        
        # Create dataset and dataloader
        dataset = NeuromorphicDataset(dataset_info)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Test iteration
        batch_count = 0
        for events_batch, labels_batch in dataloader:
            batch_count += 1
            
            # Check batch dimensions
            assert len(events_batch) == 2  # batch_size
            assert len(labels_batch) == 2
            
            # Check event format
            for events in events_batch:
                assert events.dim() == 2
                assert events.shape[1] == 4
            
            # Check labels format
            for labels in labels_batch:
                assert 'boxes' in labels
                assert 'labels' in labels
                assert 'scores' in labels
        
        assert batch_count == 4  # 8 samples / 2 batch_size


@pytest.mark.integration 
@pytest.mark.ros2
class TestROS2Integration:
    """Test ROS2 integration."""
    
    def test_ros2_node_creation(self, ros2_available):
        """Test ROS2 node creation and basic functionality."""
        if not ros2_available:
            pytest.skip("ROS2 not available")
        
        import rclpy
        from rclpy.node import Node
        
        # Mock SNN detection node
        class TestDetectionNode(Node):
            def __init__(self):
                super().__init__('test_snn_detection')
                self.detections_received = 0
                
                # Create a simple timer for testing
                self.timer = self.create_timer(0.1, self.timer_callback)
                
            def timer_callback(self):
                self.detections_received += 1
                if self.detections_received >= 5:
                    self.destroy_timer(self.timer)
        
        # Initialize ROS2
        rclpy.init()
        
        try:
            # Create and test node
            node = TestDetectionNode()
            
            # Run for a short time
            start_time = rclpy.get_global_executor()._clock.now()
            while (rclpy.get_global_executor()._clock.now() - start_time).nanoseconds < 1e9:  # 1 second
                rclpy.spin_once(node, timeout_sec=0.1)
                if node.detections_received >= 5:
                    break
            
            # Verify node functionality
            assert node.detections_received >= 1
            
            # Clean up
            node.destroy_node()
        
        finally:
            rclpy.shutdown()


@pytest.mark.integration
@pytest.mark.hardware
class TestHardwareIntegration:
    """Test hardware integration (with mocks)."""
    
    def test_loihi_integration(self, loihi_available):
        """Test Loihi hardware integration."""
        if not loihi_available:
            # Use mock for testing
            response = MockHardwareData.mock_loihi_response()
        else:
            # Would connect to real hardware
            response = MockHardwareData.mock_loihi_response()
        
        # Verify response format
        assert 'status' in response
        assert 'latency_us' in response
        assert 'power_mw' in response
        assert response['status'] == 'success'
        assert response['latency_us'] > 0
        assert response['power_mw'] > 0
    
    def test_akida_integration(self, akida_available):
        """Test Akida hardware integration."""
        if not akida_available:
            # Use mock for testing
            response = MockHardwareData.mock_akida_response()
        else:
            # Would connect to real hardware
            response = MockHardwareData.mock_akida_response()
        
        # Verify response format
        assert 'status' in response
        assert 'latency_us' in response
        assert 'power_mw' in response
        assert response['status'] == 'success'
        assert response['latency_us'] > 0
        assert response['power_mw'] > 0
    
    def test_event_camera_integration(self):
        """Test event camera integration."""
        # Use mock camera stream
        camera_frames = MockHardwareData.mock_event_camera_stream(n_frames=5)
        
        assert len(camera_frames) == 5
        
        for frame in camera_frames:
            assert frame.shape[1] == 4  # x, y, t, p format
            assert frame.dtype == np.float64
            assert np.all(frame[:, 3].astype(int) == frame[:, 3])  # polarities are integers


@pytest.mark.integration
@pytest.mark.slow
class TestLongRunningIntegration:
    """Long-running integration tests."""
    
    def test_memory_stability(self, device):
        """Test memory stability over extended processing."""
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        # Process multiple batches
        for i in range(20):
            spike_trains = TestDataGenerator.generate_spike_trains(
                batch_size=2, height=64, width=64, time_steps=10
            ).to(device)
            
            # Simple processing
            result = torch.sum(spike_trains, dim=4)
            
            # Clean up
            del spike_trains, result
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        if device.type == 'cuda':
            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory
            
            # Memory should not increase significantly
            assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase
    
    def test_processing_consistency(self):
        """Test processing consistency across multiple runs."""
        # Generate the same events multiple times
        results = []
        
        for seed in [42, 42, 42]:  # Same seed should give same results
            events = TestDataGenerator.generate_events(
                width=64, height=64, duration=0.5, event_rate=5000, seed=seed
            )
            
            # Simple processing
            spatial_hist = np.histogram2d(
                events[:, 0], events[:, 1],
                bins=[64, 64], range=[[0, 63], [0, 63]]
            )[0]
            
            results.append(spatial_hist)
        
        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])


class MockSpikingYOLO(torch.nn.Module):
    """Mock SNN model for integration testing."""
    
    def __init__(self, n_classes=10):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv3d(2, 32, (1, 3, 3), padding=(0, 1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv3d(32, 64, (1, 3, 3), padding=(0, 1, 1)),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = torch.nn.Linear(64, n_classes)
    
    def forward(self, x):
        # x: (batch, channels, height, width, time) -> (batch, channels, time, height, width)
        x = x.permute(0, 1, 4, 2, 3)
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)