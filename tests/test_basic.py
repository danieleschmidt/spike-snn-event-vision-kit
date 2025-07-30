"""Basic tests to ensure package imports correctly."""

import pytest


def test_package_import():
    """Test that the main package can be imported."""
    import spike_snn_event
    assert spike_snn_event.__version__ == "0.1.0"


def test_mock_events_fixture(mock_events):
    """Test that mock events fixture works."""
    assert mock_events.shape == (3, 4)
    assert mock_events[0][3] == 1  # First event polarity


def test_sample_config_fixture(sample_config):
    """Test that sample config fixture works."""
    assert "model" in sample_config
    assert sample_config["model"]["architecture"] == "spiking_yolo"