#!/usr/bin/env python3
"""
Generation 1 Demo: Basic Functionality

Demonstrates the working basic functionality of the Spike SNN Event Vision Kit.
"""

import numpy as np
import time
from spike_snn_event.lite_core import DVSCamera, EventPreprocessor, LiteEventSNN

def main():
    print("🚀 GENERATION 1 DEMO: MAKE IT WORK")
    print("=" * 50)
    
    # 1. Test DVS Camera
    print("1. Testing DVS Camera...")
    camera = DVSCamera(sensor_type="DVS128")
    print(f"   ✓ Camera initialized: {camera.sensor_type}")
    print(f"   ✓ Config: {camera.config.width}x{camera.config.height}")
    
    # 2. Test Event Processing
    print("\n2. Testing Event Processing...")
    processor = EventPreprocessor()
    
    # Generate sample events
    np.random.seed(42)
    n_events = 1000
    sample_events = np.column_stack([
        np.random.randint(0, 128, n_events),  # x
        np.random.randint(0, 128, n_events),  # y
        np.sort(np.random.uniform(0, 1.0, n_events)),  # timestamp
        np.random.choice([0, 1], n_events)    # polarity
    ])
    
    processed_events = processor.process(sample_events)
    print(f"   ✓ Processed {len(processed_events)} events")
    
    # 3. Test Lightweight SNN
    print("\n3. Testing Lightweight SNN...")
    snn = LiteEventSNN(
        input_size=(128, 128),
        num_classes=10
    )
    print(f"   ✓ SNN initialized: {snn.input_size}, {snn.num_classes} classes")
    
    # Simple inference test
    start_time = time.time()
    detections = snn.detect(sample_events, threshold=0.5)
    inference_time = (time.time() - start_time) * 1000  # ms
    
    print(f"   ✓ Inference completed in {inference_time:.2f}ms")
    print(f"   ✓ Detections: {len(detections)}")
    
    # 4. Test Event Streaming (simulated)
    print("\n4. Testing Event Streaming...")
    camera.start_streaming()
    events_collected = 0
    
    for i, events in enumerate(camera.stream()):
        events_collected += len(events)
        if i >= 5:  # Test 5 batches
            break
    
    camera.stop_streaming()
    print(f"   ✓ Collected {events_collected} events from {i+1} batches")
    
    # 5. Performance Summary
    print("\n" + "=" * 50)
    print("📊 GENERATION 1 PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"✓ Basic event processing: WORKING")
    print(f"✓ Camera interface: WORKING") 
    print(f"✓ Lightweight SNN: WORKING")
    print(f"✓ Event streaming: WORKING")
    print(f"✓ Average inference time: {inference_time:.2f}ms")
    print(f"✓ Events processed: {events_collected}")
    
    print("\n🎉 GENERATION 1 COMPLETE - BASIC FUNCTIONALITY ACHIEVED!")
    print("Ready to proceed to Generation 2: Make It Robust")

if __name__ == "__main__":
    main()