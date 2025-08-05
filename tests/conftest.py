"""Global pytest configuration and fixtures for spike-snn-event-vision-kit."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Check if CUDA is available for GPU tests."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def device(cuda_available) -> torch.device:
    """Get the appropriate device for testing."""
    return torch.device("cuda" if cuda_available else "cpu")


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_events() -> np.ndarray:
    """Generate sample event data for testing.
    
    Returns:
        np.ndarray: Events in format (x, y, timestamp, polarity)
    """
    np.random.seed(42)
    n_events = 1000
    
    # Generate sample events
    x = np.random.randint(0, 128, size=n_events)
    y = np.random.randint(0, 128, size=n_events)
    t = np.sort(np.random.uniform(0, 1.0, size=n_events))  # 1 second
    p = np.random.choice([0, 1], size=n_events)
    
    return np.column_stack([x, y, t, p])


@pytest.fixture
def sample_spike_train() -> torch.Tensor:
    """Generate sample spike train for SNN testing.
    
    Returns:
        torch.Tensor: Spike train of shape (batch, channels, height, width, time)
    """
    torch.manual_seed(42)
    batch_size, channels, height, width, time_steps = 1, 2, 32, 32, 10
    
    # Generate sparse spike train (10% sparsity)
    spike_train = torch.rand(batch_size, channels, height, width, time_steps)
    spike_train = (spike_train > 0.9).float()
    
    return spike_train


@pytest.fixture
def model_config() -> dict:
    """Basic SNN model configuration for testing."""
    return {
        "input_size": (128, 128),
        "hidden_channels": [32, 64],
        "output_classes": 10,
        "neuron_type": "LIF",
        "threshold": 1.0,
        "tau_mem": 20e-3,
        "tau_syn": 5e-3,
        "time_steps": 10,
    }


@pytest.fixture
def training_config() -> dict:
    """Training configuration for testing."""
    return {
        "batch_size": 4,
        "learning_rate": 1e-3,
        "epochs": 2,
        "device": "cpu",
        "optimizer": "adam",
        "loss_function": "cross_entropy",
    }


@pytest.fixture(scope="session")
def ros2_available() -> bool:
    """Check if ROS2 is available for integration tests."""
    try:
        import rclpy
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def loihi_available() -> bool:
    """Check if Intel Loihi SDK is available."""
    try:
        import nxsdk
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def akida_available() -> bool:
    """Check if BrainChip Akida SDK is available."""
    try:
        import akida
        return True
    except ImportError:
        return False


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "hardware: Hardware-dependent tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "gpu: GPU-required tests")
    config.addinivalue_line("markers", "ros2: ROS2-dependent tests")
    config.addinivalue_line("markers", "loihi: Intel Loihi tests")
    config.addinivalue_line("markers", "akida: BrainChip Akida tests")
    config.addinivalue_line("markers", "benchmark: Performance benchmarks")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle hardware dependencies."""
    cuda_available = torch.cuda.is_available()
    
    # Check for hardware availability
    try:
        import rclpy
        ros2_available = True
    except ImportError:
        ros2_available = False
    
    try:
        import nxsdk
        loihi_available = True
    except ImportError:
        loihi_available = False
    
    try:
        import akida
        akida_available = True
    except ImportError:
        akida_available = False
    
    # Skip markers for unavailable hardware
    skip_gpu = pytest.mark.skip(reason="CUDA GPU not available")
    skip_ros2 = pytest.mark.skip(reason="ROS2 not available")
    skip_loihi = pytest.mark.skip(reason="Intel Loihi SDK not available")
    skip_akida = pytest.mark.skip(reason="BrainChip Akida SDK not available")
    
    for item in items:
        # GPU tests
        if "gpu" in item.keywords and not cuda_available:
            item.add_marker(skip_gpu)
        
        # ROS2 tests
        if "ros2" in item.keywords and not ros2_available:
            item.add_marker(skip_ros2)
        
        # Loihi tests
        if "loihi" in item.keywords and not loihi_available:
            item.add_marker(skip_loihi)
        
        # Akida tests
        if "akida" in item.keywords and not akida_available:
            item.add_marker(skip_akida)


@pytest.fixture
def mock_event_camera(monkeypatch):
    """Mock event camera for testing without hardware."""
    class MockEventCamera:
        def __init__(self, sensor_type="DVS128"):
            self.sensor_type = sensor_type
            self.is_connected = True
        
        def stream(self):
            """Mock event stream."""
            for _ in range(10):  # Generate 10 event frames
                yield self._generate_events()
        
        def _generate_events(self):
            """Generate mock events."""
            np.random.seed(42)
            n_events = np.random.randint(100, 1000)
            x = np.random.randint(0, 128, size=n_events)
            y = np.random.randint(0, 128, size=n_events)
            t = np.sort(np.random.uniform(0, 0.1, size=n_events))
            p = np.random.choice([0, 1], size=n_events)
            return np.column_stack([x, y, t, p])
        
        def visualize_detections(self, events, detections):
            """Mock visualization."""
            pass
    
    return MockEventCamera


@pytest.fixture
def benchmark_data():
    """Data for performance benchmarking."""
    return {
        "event_counts": [1000, 5000, 10000, 50000],
        "model_sizes": ["small", "medium", "large"],
        "batch_sizes": [1, 4, 8, 16],
        "time_windows": [10e-3, 50e-3, 100e-3],
    }


# Performance benchmarks (requires pytest-benchmark plugin)
# def pytest_benchmark_group_stats(config, benchmarks, group_by):
#     """Custom benchmark grouping."""
#     return group_by


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Clean up CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    yield