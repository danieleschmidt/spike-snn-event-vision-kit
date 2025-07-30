"""Basic tests for spike_snn_event package."""

import pytest
import spike_snn_event


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