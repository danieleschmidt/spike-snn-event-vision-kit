# spike-snn-event-vision-kit

A lightweight **Spiking Neural Network (SNN)** framework for event-based vision, built entirely on NumPy.

---

## What's inside

| Module | Description |
|--------|-------------|
| `snn/neuron.py` | `LIFNeuron` — leaky integrate-and-fire neuron with configurable threshold, τ, and reset |
| `snn/event_frame.py` | `EventFrame` — converts a raw event stream `(x, y, t, polarity)` → normalized 2-D frame |
| `snn/conv.py` | `SpikingConv2D` — 2-D convolutional layer with LIF neurons and STDP-like weight updates |
| `snn/network.py` | `SpikingNetwork` — sequential stack of spiking layers |

---

## Quick start

```python
from snn import LIFNeuron, EventFrame, SpikingConv2D, SpikingNetwork
import numpy as np

# Single neuron
neuron = LIFNeuron(threshold=1.0, tau=20.0, dt=1.0)
spikes = neuron.run([0.3] * 100)

# Event stream → frame
ef = EventFrame(height=32, width=32)
events = [(x, y, t, p) for x, y, t, p in zip(
    np.random.randint(0, 32, 200),
    np.random.randint(0, 32, 200),
    np.arange(200),
    np.random.choice([-1, 1], 200),
)]
frame = ef.convert(events)          # shape (32, 32), values in [0, 1]

# Two-layer spiking network
net = SpikingNetwork([
    SpikingConv2D(1, 8, kernel_size=3),
    SpikingConv2D(8, 4, kernel_size=3),
])
output_spikes = net.forward(frame)  # shape (4, 26, 26)
```

---

## LIF neuron dynamics

```
V(t) = V(t-1) × exp(-dt/τ) + I(t)

if V(t) ≥ θ:
    emit spike (output = 1)
    V ← V_reset
```

Default parameters: `threshold = 1.0`, `τ = 20 ms`, `dt = 1 ms`, `V_reset = 0.0`.

---

## STDP learning

`SpikingConv2D.stdp_update(pre_spikes, post_spikes)` implements a simple
Hebbian rule: weights are increased proportionally to the pre-synaptic
activation whenever a post-synaptic spike is detected.

---

## Run the demo

```bash
python demo.py
```

Generates synthetic event data for 3 spatial classes, converts each to a
frame, runs it through a two-layer SpikingNetwork, and reports classification
accuracy using a winner-takes-all readout.

---

## Tests

```bash
pip install pytest
pytest tests/
```

---

## Requirements

- Python 3.8+
- NumPy

```bash
pip install -r requirements.txt
```

---

## License

MIT — see `LICENSE`.
