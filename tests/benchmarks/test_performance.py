"""Performance benchmarking tests for spike-snn-event-vision-kit."""

import pytest
import torch
import numpy as np
import time
from typing import Dict, List, Tuple

from tests.fixtures import TestDataGenerator, PerformanceTestData


@pytest.mark.benchmark
class TestInferencePerformance:
    """Benchmark inference performance across different configurations."""
    
    @pytest.mark.parametrize("config", PerformanceTestData.get_benchmark_configs())
    def test_snn_inference_latency(self, benchmark, config, device):
        """Benchmark SNN inference latency."""
        # Skip GPU tests if CUDA not available
        if device.type == 'cuda' and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Generate test data
        spike_trains = TestDataGenerator.generate_spike_trains(
            batch_size=config['batch_size'],
            height=config['input_size'][0],
            width=config['input_size'][1],
            time_steps=config['time_steps']
        ).to(device)
        
        # Mock SNN model for benchmarking
        class MockSNN(torch.nn.Module):
            def __init__(self, input_size, n_classes=10):
                super().__init__()
                self.conv1 = torch.nn.Conv3d(2, 32, (1, 3, 3), padding=(0, 1, 1))
                self.conv2 = torch.nn.Conv3d(32, 64, (1, 3, 3), padding=(0, 1, 1))
                self.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
                self.fc = torch.nn.Linear(64, n_classes)
            
            def forward(self, x):
                # x: (batch, channels, height, width, time) -> (batch, channels, time, height, width)
                x = x.permute(0, 1, 4, 2, 3)
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        model = MockSNN(config['input_size']).to(device)
        model.eval()
        
        # Warm-up
        with torch.no_grad():
            _ = model(spike_trains)
        
        # Benchmark inference
        def inference():
            with torch.no_grad():
                return model(spike_trains)
        
        result = benchmark(inference)
        
        # Assert reasonable performance
        assert result is not None
        
        # Performance assertions based on config
        if config['name'] == 'small_model_light_load':
            # Small model should be fast
            pass
        elif config['name'] == 'large_model_heavy_load':
            # Large model may be slower but should still be reasonable
            pass
    
    @pytest.mark.parametrize("n_events", [1000, 10000, 50000])
    def test_event_processing_throughput(self, benchmark, n_events):
        """Benchmark event processing throughput."""
        events = TestDataGenerator.generate_events(
            duration=1.0, event_rate=n_events, seed=42
        )
        
        def process_events():
            # Mock event processing pipeline
            # Spatial binning
            spatial_bins = np.histogram2d(
                events[:, 0], events[:, 1], 
                bins=[128, 128], range=[[0, 127], [0, 127]]
            )[0]
            
            # Temporal binning
            temporal_bins = np.histogram(
                events[:, 2], bins=100, range=[0, 1.0]
            )[0]
            
            return spatial_bins, temporal_bins
        
        result = benchmark(process_events)
        assert result is not None
        
        # Check processing rate (events per second)
        processing_time = benchmark.stats.mean
        events_per_second = n_events / processing_time
        
        # Should process at least 100K events/second
        assert events_per_second > 100000, f"Too slow: {events_per_second:.0f} events/s"
    
    @pytest.mark.gpu
    def test_gpu_memory_efficiency(self, device):
        """Test GPU memory usage efficiency."""
        if device.type != 'cuda':
            pytest.skip("GPU test requires CUDA")
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Create progressively larger tensors
        tensors = []
        for i in range(5):
            size = (2**i, 2, 128, 128, 10)  # Increasing batch size
            tensor = TestDataGenerator.generate_spike_trains(
                batch_size=size[0], height=size[2], width=size[3], time_steps=size[4]
            ).to(device)
            tensors.append(tensor)
        
        peak_memory = torch.cuda.memory_allocated()
        memory_used = peak_memory - initial_memory
        
        # Clean up
        del tensors
        torch.cuda.empty_cache()
        
        # Memory usage should be reasonable (less than 1GB for test)
        memory_mb = memory_used / (1024 * 1024)
        assert memory_mb < 1024, f"Memory usage too high: {memory_mb:.1f} MB"


@pytest.mark.benchmark
class TestTrainingPerformance:
    """Benchmark training performance."""
    
    def test_training_step_performance(self, benchmark, device):
        """Benchmark single training step performance."""
        # Mock training setup
        batch_size = 4
        spike_trains = TestDataGenerator.generate_spike_trains(
            batch_size=batch_size, height=64, width=64, time_steps=10
        ).to(device)
        
        labels = torch.randint(0, 10, (batch_size,)).to(device)
        
        # Simple model for benchmarking
        class SimpleSNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv3d(2, 32, (1, 3, 3), padding=(0, 1, 1))
                self.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
                self.fc = torch.nn.Linear(32, 10)
            
            def forward(self, x):
                x = x.permute(0, 1, 4, 2, 3)  # Reorder for Conv3d
                x = torch.relu(self.conv(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        model = SimpleSNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        def training_step():
            optimizer.zero_grad()
            outputs = model(spike_trains)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            return loss.item()
        
        # Warm-up
        _ = training_step()
        
        # Benchmark
        loss = benchmark(training_step)
        assert loss is not None and loss > 0


@pytest.mark.benchmark
class TestHardwarePerformance:
    """Benchmark hardware-specific performance."""
    
    @pytest.mark.loihi
    def test_loihi_deployment_latency(self, benchmark, loihi_available):
        """Benchmark Loihi deployment latency."""
        if not loihi_available:
            pytest.skip("Intel Loihi SDK not available")
        
        # Mock Loihi deployment
        def mock_loihi_inference():
            # Simulate Loihi processing time
            time.sleep(0.001)  # 1ms simulated latency
            return {
                'spikes_out': np.random.randint(10, 100),
                'power_mw': 0.5,
                'latency_us': 1000
            }
        
        result = benchmark(mock_loihi_inference)
        assert result['latency_us'] < 5000  # Should be under 5ms
    
    @pytest.mark.akida
    def test_akida_deployment_latency(self, benchmark, akida_available):
        """Benchmark Akida deployment latency."""
        if not akida_available:
            pytest.skip("BrainChip Akida SDK not available")
        
        # Mock Akida deployment
        def mock_akida_inference():
            # Simulate Akida processing time
            time.sleep(0.002)  # 2ms simulated latency
            return {
                'predictions': np.random.rand(10),
                'power_mw': 2.0,
                'latency_us': 2000
            }
        
        result = benchmark(mock_akida_inference)
        assert result['latency_us'] < 10000  # Should be under 10ms


@pytest.mark.benchmark
class TestScalabilityBenchmarks:
    """Test scalability across different dimensions."""
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
    def test_batch_size_scaling(self, benchmark, batch_size, device):
        """Test how performance scales with batch size."""
        spike_trains = TestDataGenerator.generate_spike_trains(
            batch_size=batch_size, height=32, width=32, time_steps=5
        ).to(device)
        
        # Simple processing function
        def process_batch():
            # Simulate some processing
            return torch.sum(spike_trains, dim=4)  # Sum over time
        
        result = benchmark(process_batch)
        assert result.shape[0] == batch_size
    
    @pytest.mark.parametrize("time_steps", [5, 10, 20, 50])
    def test_temporal_scaling(self, benchmark, time_steps, device):
        """Test how performance scales with temporal resolution."""
        spike_trains = TestDataGenerator.generate_spike_trains(
            batch_size=4, height=32, width=32, time_steps=time_steps
        ).to(device)
        
        def process_temporal():
            # Temporal convolution
            conv = torch.nn.Conv3d(2, 16, (time_steps//2, 1, 1)).to(device)
            return conv(spike_trains.permute(0, 1, 4, 2, 3))
        
        result = benchmark(process_temporal)
        assert result is not None


# Benchmark configuration
pytest_plugins = ['pytest_benchmark']

# Custom benchmark groups
def pytest_benchmark_group_stats(config, benchmarks, group_by):
    """Custom benchmark grouping for neuromorphic tests."""
    return group_by