"""Tests for the spike-snn-event-vision-kit package."""

import math
import numpy as np
import pytest
import sys
import os

# Ensure the repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from snn.neuron import LIFNeuron
from snn.event_frame import EventFrame
from snn.conv import SpikingConv2D
from snn.network import SpikingNetwork


# ---------------------------------------------------------------------------
# LIFNeuron tests
# ---------------------------------------------------------------------------

class TestLIFNeuron:
    def test_spikes_when_input_exceeds_threshold(self):
        """Neuron fires when input drives V above threshold in one step."""
        neuron = LIFNeuron(threshold=1.0)
        # Single large step: V = 0 * decay + 2.0 = 2.0 >= 1.0 → spike
        spike = neuron.step(2.0)
        assert spike == 1

    def test_no_spike_below_threshold(self):
        """Neuron does NOT fire when input keeps V below threshold."""
        neuron = LIFNeuron(threshold=1.0)
        spike = neuron.step(0.1)
        assert spike == 0

    def test_membrane_potential_decays(self):
        """After a sub-threshold input, V should be positive but decayed."""
        neuron = LIFNeuron(threshold=10.0, tau=20.0, dt=1.0)
        neuron.step(0.5)          # V = 0 * decay + 0.5 = 0.5
        v_after_one = neuron.V
        neuron.step(0.0)          # V = v_after_one * decay + 0 < v_after_one
        assert neuron.V < v_after_one
        assert neuron.V > 0.0

    def test_resets_after_spike(self):
        """After spiking, membrane potential is reset to V_reset."""
        neuron = LIFNeuron(threshold=1.0, V_reset=0.0)
        neuron.step(2.0)  # fires
        assert neuron.V == pytest.approx(0.0)

    def test_decay_value(self):
        """Decay constant matches exp(-dt/tau)."""
        neuron = LIFNeuron(tau=20.0, dt=1.0)
        expected = math.exp(-1.0 / 20.0)
        assert neuron.decay == pytest.approx(expected)

    def test_temporal_dynamics_over_multiple_steps(self):
        """Running over many steps produces valid spike train (0s and 1s)."""
        neuron = LIFNeuron(threshold=1.0)
        inputs = [0.3] * 50  # constant sub-threshold; may accumulate and spike
        spikes = neuron.run(inputs)
        assert len(spikes) == 50
        assert all(s in (0, 1) for s in spikes)

    def test_run_returns_list(self):
        """run() returns a list of the same length as input."""
        neuron = LIFNeuron()
        result = neuron.run([0.5, 0.5, 0.5])
        assert isinstance(result, list)
        assert len(result) == 3

    def test_reset_state(self):
        """reset_state() restores membrane potential to V_rest."""
        neuron = LIFNeuron(threshold=10.0)
        neuron.step(5.0)
        assert neuron.V > 0
        neuron.reset_state()
        assert neuron.V == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# EventFrame tests
# ---------------------------------------------------------------------------

class TestEventFrame:
    def test_shape_is_correct(self):
        """convert() returns array with shape (H, W)."""
        ef = EventFrame(height=8, width=10)
        events = [(3, 2, 0, 1)]
        frame = ef.convert(events)
        assert frame.shape == (8, 10)

    def test_sums_polarities_correctly(self):
        """Polarities at the same pixel are summed before normalizing."""
        ef = EventFrame(height=5, width=5)
        # Three +1 events at (2,2) and one -1 event at (0,0)
        events = [(2, 2, 0, 1), (2, 2, 1, 1), (2, 2, 2, 1), (0, 0, 3, -1)]
        frame = ef.convert(events)
        # After normalization the max should be at [2, 2]
        assert frame[2, 2] == pytest.approx(1.0)

    def test_handles_empty_events(self):
        """Empty event list returns all-zero frame."""
        ef = EventFrame(height=4, width=4)
        frame = ef.convert([])
        assert frame.shape == (4, 4)
        assert np.all(frame == 0.0)

    def test_normalized_range(self):
        """All values in returned frame are in [0, 1]."""
        ef = EventFrame(height=8, width=8)
        rng = np.random.default_rng(1)
        events = [
            (rng.integers(0, 8), rng.integers(0, 8), i, rng.choice([-1, 1]))
            for i in range(100)
        ]
        frame = ef.convert(events)
        assert frame.min() >= 0.0
        assert frame.max() <= 1.0


# ---------------------------------------------------------------------------
# SpikingConv2D tests
# ---------------------------------------------------------------------------

class TestSpikingConv2D:
    def _make_input(self, in_c=1, H=8, W=8, seed=0):
        rng = np.random.default_rng(seed)
        return rng.random((in_c, H, W))

    def test_output_shape_is_correct(self):
        """Output shape follows (out_channels, H-k+1, W-k+1) for stride=1."""
        layer = SpikingConv2D(in_channels=1, out_channels=4, kernel_size=3)
        x = self._make_input(in_c=1, H=8, W=8)
        out = layer.forward(x)
        assert out.shape == (4, 6, 6)

    def test_produces_spike_outputs(self):
        """All output values are 0 or 1."""
        layer = SpikingConv2D(in_channels=1, out_channels=2, kernel_size=3, threshold=0.01)
        x = self._make_input(in_c=1, H=6, W=6)
        out = layer.forward(x)
        assert set(np.unique(out)).issubset({0, 1})

    def test_stdp_update_changes_weights(self):
        """STDP update modifies weights when pre and post spikes are present."""
        # Use a very low threshold to guarantee post-synaptic spikes
        layer = SpikingConv2D(in_channels=1, out_channels=2, kernel_size=3, threshold=0.001)
        x = np.ones((1, 8, 8))
        spikes = layer.forward(x)
        # Ensure we got at least some post-synaptic spikes
        assert spikes.sum() > 0, "Need spikes for STDP update to have effect"
        weights_before = layer.weights.copy()
        layer.stdp_update(
            (x > 0).astype(np.int32),
            spikes,
        )
        assert not np.allclose(layer.weights, weights_before)


# ---------------------------------------------------------------------------
# SpikingNetwork tests
# ---------------------------------------------------------------------------

class TestSpikingNetwork:
    def test_forward_pass_returns_spikes(self):
        """Network forward pass returns spike array."""
        net = SpikingNetwork([
            SpikingConv2D(1, 4, kernel_size=3, threshold=0.1),
            SpikingConv2D(4, 2, kernel_size=3, threshold=0.1),
        ])
        x = np.random.default_rng(5).random((1, 12, 12))
        out = net.forward(x)
        assert out.ndim == 3
        assert set(np.unique(out)).issubset({0, 1})

    def test_forward_accepts_2d_input(self):
        """Network automatically promotes 2-D input to (1, H, W)."""
        net = SpikingNetwork([
            SpikingConv2D(1, 2, kernel_size=3, threshold=0.1),
        ])
        x = np.random.default_rng(6).random((8, 8))
        out = net.forward(x)
        assert out.shape == (2, 6, 6)


# ---------------------------------------------------------------------------
# Demo integration test
# ---------------------------------------------------------------------------

class TestDemo:
    def test_demo_classification_completes(self):
        """Demo script runs end-to-end without error."""
        import importlib.util, pathlib
        demo_path = pathlib.Path(__file__).parent.parent / "demo.py"
        spec = importlib.util.spec_from_file_location("demo", demo_path)
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        accuracy = demo.main()
        # Just check it completed and returned a numeric accuracy
        assert 0.0 <= accuracy <= 1.0
