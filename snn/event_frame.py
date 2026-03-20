"""Event frame conversion: event stream → 2D spatial frame."""

import numpy as np


class EventFrame:
    """Convert a stream of events into a 2D spatial frame.

    Events are tuples of (x, y, t, polarity) where polarity is +1 or -1.
    Polarities at each pixel location are summed and the result is
    normalized to [0, 1].
    """

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def convert(self, events) -> np.ndarray:
        """Convert event list to a 2D frame.

        Args:
            events: List of (x, y, t, polarity) tuples.
                    x in [0, width-1], y in [0, height-1].

        Returns:
            numpy array of shape (H, W), dtype float64, values in [0, 1].
        """
        frame = np.zeros((self.height, self.width), dtype=np.float64)

        for event in events:
            x, y, _t, polarity = event
            if 0 <= x < self.width and 0 <= y < self.height:
                frame[y, x] += polarity

        # Normalize to [0, 1]
        min_val = frame.min()
        max_val = frame.max()
        if max_val > min_val:
            frame = (frame - min_val) / (max_val - min_val)
        else:
            # All values equal — return zeros
            frame = np.zeros_like(frame)

        return frame
