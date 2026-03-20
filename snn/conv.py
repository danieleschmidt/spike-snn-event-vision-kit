"""Spiking convolutional layer with STDP-like weight updates."""

import numpy as np
from .neuron import LIFNeuron


class SpikingConv2D:
    """2D convolutional layer with LIF neurons and STDP-like learning.

    Attributes:
        weights: shape (out_channels, in_channels, kH, kW)
        neurons: grid of LIFNeuron instances, shape (out_channels, out_H, out_W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        threshold: float = 1.0,
        tau: float = 20.0,
        dt: float = 1.0,
        learning_rate: float = 0.01,
        seed: int = 42,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.learning_rate = learning_rate

        rng = np.random.default_rng(seed)
        self.weights = rng.normal(
            0.0, 0.1, (out_channels, in_channels, kernel_size, kernel_size)
        )

        # Neuron pool is initialised per forward call based on output shape
        self._neuron_params = dict(threshold=threshold, tau=tau, dt=dt)
        self._neurons = None  # lazy init
        self._output_shape = None

    def _init_neurons(self, out_h: int, out_w: int):
        self._output_shape = (self.out_channels, out_h, out_w)
        self._neurons = [
            [
                [LIFNeuron(**self._neuron_params) for _ in range(out_w)]
                for _ in range(out_h)
            ]
            for _ in range(self.out_channels)
        ]

    def _conv2d(self, x: np.ndarray) -> np.ndarray:
        """Manual convolution: x shape (in_channels, H, W) → (out_channels, out_H, out_W)."""
        in_c, H, W = x.shape
        kH = kW = self.kernel_size
        out_H = (H - kH) // self.stride + 1
        out_W = (W - kW) // self.stride + 1

        out = np.zeros((self.out_channels, out_H, out_W))
        for oc in range(self.out_channels):
            for i in range(out_H):
                for j in range(out_W):
                    si = i * self.stride
                    sj = j * self.stride
                    patch = x[:, si : si + kH, sj : sj + kW]  # (in_c, kH, kW)
                    out[oc, i, j] = np.sum(self.weights[oc] * patch)
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            x: Input of shape (in_channels, H, W).

        Returns:
            Spike output of shape (out_channels, out_H, out_W), values 0 or 1.
        """
        conv_out = self._conv2d(x)
        out_c, out_H, out_W = conv_out.shape

        if self._neurons is None or self._output_shape != (out_c, out_H, out_W):
            self._init_neurons(out_H, out_W)

        spikes = np.zeros((out_c, out_H, out_W), dtype=np.int32)
        for oc in range(out_c):
            for i in range(out_H):
                for j in range(out_W):
                    spikes[oc, i, j] = self._neurons[oc][i][j].step(conv_out[oc, i, j])

        return spikes

    def stdp_update(self, pre_spikes: np.ndarray, post_spikes: np.ndarray):
        """Spike-Timing-Dependent Plasticity weight update.

        Increases weights where pre-synaptic activity (input) preceded
        post-synaptic spiking (output).

        Args:
            pre_spikes: shape (in_channels, H, W) — input spike frame
            post_spikes: shape (out_channels, out_H, out_W) — output spikes
        """
        in_c, H, W = pre_spikes.shape
        out_c, out_H, out_W = post_spikes.shape
        kH = kW = self.kernel_size

        for oc in range(out_c):
            for i in range(out_H):
                for j in range(out_W):
                    if post_spikes[oc, i, j]:
                        si = i * self.stride
                        sj = j * self.stride
                        pre_patch = pre_spikes[:, si : si + kH, sj : sj + kW]
                        self.weights[oc] += self.learning_rate * pre_patch
