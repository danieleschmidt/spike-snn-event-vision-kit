"""Leaky Integrate-and-Fire (LIF) neuron model."""

import math
import numpy as np


class LIFNeuron:
    """Leaky Integrate-and-Fire neuron.

    State:
        V  — membrane potential (float), starts at V_rest = 0.0

    Dynamics per time step:
        V = V * decay + input_current
        decay = exp(-dt / tau)

    Spike condition:
        if V >= threshold: emit spike (output=1), reset V to V_reset
        else: output=0
    """

    def __init__(
        self,
        threshold: float = 1.0,
        tau: float = 20.0,
        dt: float = 1.0,
        V_rest: float = 0.0,
        V_reset: float = 0.0,
    ):
        self.threshold = threshold
        self.tau = tau
        self.dt = dt
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.decay = math.exp(-dt / tau)
        self.V = V_rest

    def reset_state(self):
        """Reset membrane potential to resting state."""
        self.V = self.V_rest

    def step(self, input_current: float) -> int:
        """Advance one time step with the given input current.

        Args:
            input_current: Synaptic input current for this step.

        Returns:
            1 if the neuron spiked, 0 otherwise.
        """
        self.V = self.V * self.decay + input_current
        if self.V >= self.threshold:
            self.V = self.V_reset
            return 1
        return 0

    def run(self, input_currents) -> list:
        """Run neuron over a sequence of input currents.

        Args:
            input_currents: Iterable of input current values.

        Returns:
            List of spike outputs (0 or 1) for each time step.
        """
        return [self.step(i) for i in input_currents]
