#!/usr/bin/env python3
"""Demo: toy 3-class classification on synthetic event-camera data.

Generates random event streams for 3 classes, processes them through a
SpikingNetwork, and uses a simple winner-takes-all rule on total spike count
per output channel to predict the class.
"""

import numpy as np
from snn import LIFNeuron, EventFrame, SpikingConv2D, SpikingNetwork

RNG = np.random.default_rng(0)

H, W = 16, 16
N_CLASSES = 3
N_SAMPLES = 30
N_EVENTS = 50


def generate_events(class_id: int, n_events: int = N_EVENTS):
    """Generate synthetic events biased towards a class region."""
    # Each class gets events concentrated in a different quadrant
    cx = [4, 12, 8][class_id]
    cy = [4, 4, 12][class_id]

    events = []
    for _ in range(n_events):
        x = int(np.clip(RNG.normal(cx, 2), 0, W - 1))
        y = int(np.clip(RNG.normal(cy, 2), 0, H - 1))
        t = RNG.integers(0, 100)
        polarity = RNG.choice([-1, 1])
        events.append((x, y, t, polarity))
    return events


def build_network():
    return SpikingNetwork([
        SpikingConv2D(1, 4, kernel_size=3, threshold=0.5),
        SpikingConv2D(4, N_CLASSES, kernel_size=3, threshold=0.5),
    ])


def predict(net: SpikingNetwork, frame: np.ndarray) -> int:
    spikes = net.forward(frame)          # (N_CLASSES, out_H, out_W)
    totals = spikes.sum(axis=(1, 2))     # spike count per output channel
    return int(np.argmax(totals))


def main():
    ef = EventFrame(H, W)
    net = build_network()

    correct = 0
    total = 0

    for class_id in range(N_CLASSES):
        for _ in range(N_SAMPLES // N_CLASSES):
            events = generate_events(class_id)
            frame = ef.convert(events)
            pred = predict(net, frame)
            if pred == class_id:
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"Demo completed — {total} samples, accuracy: {accuracy:.2%}")
    print("(Untrained baseline; STDP learning not applied in this demo)")
    return accuracy


if __name__ == "__main__":
    main()
