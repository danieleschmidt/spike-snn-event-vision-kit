#!/usr/bin/env python3
"""Basic event-based object detection example.

This example demonstrates how to use the Spike-SNN Event Vision Kit
for real-time object detection with event cameras.
"""

import numpy as np
from typing import List, Tuple


def mock_event_stream() -> List[np.ndarray]:
    """Generate mock event data for demonstration.
    
    Returns:
        List of event arrays with shape (N, 4) containing x, y, timestamp, polarity
    """
    # Simulate moving object creating events
    events = []
    for t in range(10):
        # Generate events for a moving blob
        x_center = 64 + t * 5  # Moving right
        y_center = 64
        
        # Create circular pattern of events
        event_list = []
        for i in range(20):
            angle = (i / 20) * 2 * np.pi
            x = int(x_center + 10 * np.cos(angle))
            y = int(y_center + 10 * np.sin(angle))
            
            # Ensure coordinates are within bounds
            if 0 <= x < 128 and 0 <= y < 128:
                timestamp = t * 0.01 + i * 0.0001  # 10ms intervals
                polarity = 1 if i % 2 == 0 else 0
                event_list.append([x, y, timestamp, polarity])
        
        events.append(np.array(event_list))
    
    return events


def mock_detect(events: np.ndarray, threshold: float = 0.5) -> List[dict]:
    """Mock detection function.
    
    Args:
        events: Event array with shape (N, 4)
        threshold: Detection confidence threshold
        
    Returns:
        List of detection dictionaries
    """
    if len(events) == 0:
        return []
    
    # Simple mock: detect object based on event density
    x_coords = events[:, 0]
    y_coords = events[:, 1]
    
    if len(x_coords) > 10:  # Minimum events for detection
        # Calculate bounding box from events
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Mock confidence based on number of events
        confidence = min(len(events) / 50.0, 1.0)
        
        if confidence > threshold:
            return [{
                'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                'confidence': confidence,
                'class_id': 0,
                'class_name': 'object'
            }]
    
    return []


def main():
    """Main demonstration function."""
    print("🎥 Spike-SNN Event Vision Kit - Basic Detection Demo")
    print("=" * 50)
    
    # Generate mock event stream
    print("📊 Generating mock event data...")
    event_stream = mock_event_stream()
    
    print(f"📈 Processing {len(event_stream)} time windows...")
    
    # Process each time window
    total_detections = 0
    for i, events in enumerate(event_stream):
        print(f"\n⏱️  Time window {i+1}:")
        print(f"   Events: {len(events)}")
        
        # Run mock detection
        detections = mock_detect(events, threshold=0.3)
        print(f"   Detections: {len(detections)}")
        
        for det in detections:
            print(f"   📍 {det['class_name']}: "
                  f"bbox={det['bbox']}, conf={det['confidence']:.2f}")
            total_detections += 1
    
    print(f"\n✅ Processing complete!")
    print(f"🎯 Total detections: {total_detections}")
    print(f"⚡ Average latency: <1ms (mock)")
    
    print("\n💡 Next steps:")
    print("   1. Connect real event camera")
    print("   2. Load pre-trained SNN model")
    print("   3. Run real-time detection")


if __name__ == "__main__":
    main()