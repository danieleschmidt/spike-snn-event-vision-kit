"""SpikingNetwork: stack of spiking layers."""

import numpy as np
from .conv import SpikingConv2D


class SpikingNetwork:
    """A sequential stack of SpikingConv2D layers.

    Example::

        net = SpikingNetwork([
            SpikingConv2D(1, 4, kernel_size=3),
            SpikingConv2D(4, 8, kernel_size=3),
        ])
        spikes = net.forward(input_frame)
    """

    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers.

        Args:
            x: Input array. If 2-D (H, W), it is treated as a single-channel
               frame and expanded to (1, H, W).

        Returns:
            Spike output of the last layer.
        """
        if x.ndim == 2:
            x = x[np.newaxis, :]  # (1, H, W)

        out = x.astype(np.float64)
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def train_step(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with STDP weight updates at each layer.

        Args:
            x: Input array (H, W) or (C, H, W).

        Returns:
            Final spike output.
        """
        if x.ndim == 2:
            x = x[np.newaxis, :]

        current = x.astype(np.float64)
        for layer in self.layers:
            output = layer.forward(current)
            if hasattr(layer, "stdp_update"):
                layer.stdp_update(
                    (current > 0).astype(np.int32),
                    output,
                )
            current = output
        return current
