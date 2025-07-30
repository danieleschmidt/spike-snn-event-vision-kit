"""Pytest configuration and fixtures."""

import pytest
import numpy as np


@pytest.fixture
def mock_events():
    """Generate mock event data for testing."""
    return np.array([
        [100, 50, 0.001, 1],  # x, y, timestamp, polarity
        [101, 51, 0.002, 0],
        [99, 49, 0.003, 1],
    ])


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {
            "architecture": "spiking_yolo",
            "input_size": (128, 128),
            "num_classes": 10,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-3,
            "epochs": 100,
        }
    }