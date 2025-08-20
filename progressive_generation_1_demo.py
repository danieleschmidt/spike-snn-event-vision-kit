#!/usr/bin/env python3
"""
Progressive Quality Gates Generation 1 Demo
============================================

Demonstrates the "MAKE IT WORK" principle with incremental validation.
Each component is validated before building the next layer.
"""

import sys
import os
sys.path.append('/root/repo/src')

def progressive_quality_gate_1():
    """Generation 1: MAKE IT WORK with progressive quality gates."""
    print("🚀 PROGRESSIVE QUALITY GATES - GENERATION 1: MAKE IT WORK")
    print("=" * 60)
    
    # Gate 1: Core Dependencies
    print("\n📋 QUALITY GATE 1: Core Dependencies")
    try:
        import numpy as np
        import torch
        print(f"✅ NumPy {np.__version__}")
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ Dependency error: {e}")
        return False
    
    # Gate 2: Module Imports
    print("\n📋 QUALITY GATE 2: Module Imports")
    try:
        from spike_snn_event.core import DVSCamera, EventPreprocessor, validate_events
        from spike_snn_event.models import SpikingYOLO, CustomSNN
        print("✅ Core modules imported successfully")
    except ImportError as e:
        print(f"❌ Module import error: {e}")
        return False
    
    # Gate 3: Basic Event Processing
    print("\n📋 QUALITY GATE 3: Basic Event Processing")
    try:
        # Generate synthetic events
        events = np.array([
            [10.0, 20.0, 0.001, 1],    # x, y, timestamp, polarity
            [15.0, 25.0, 0.002, -1],
            [30.0, 40.0, 0.003, 1]
        ])
        
        # Validate events
        validated_events = validate_events(events)
        print(f"✅ Event validation: {len(validated_events)} events processed")
        
        # Test camera initialization
        camera = DVSCamera(sensor_type="DVS128")
        print(f"✅ Camera initialized: {camera.sensor_type} ({camera.width}x{camera.height})")
        
    except Exception as e:
        print(f"❌ Event processing error: {e}")
        return False
    
    # Gate 4: Event Stream Generation
    print("\n📋 QUALITY GATE 4: Event Stream Generation")
    try:
        camera = DVSCamera(sensor_type="DVS128")
        event_count = 0
        batch_count = 0
        
        # Test short stream
        for i, batch in enumerate(camera.stream(duration=0.1)):  # 100ms test
            event_count += len(batch)
            batch_count = i + 1
            if i >= 2:  # Test first few batches
                break
                
        print(f"✅ Event stream generated: {event_count} events in {batch_count} batches")
        
        # Test health check
        health = camera.health_check()
        print(f"✅ Camera health: {health['status']}")
        
    except Exception as e:
        print(f"❌ Event stream error: {e}")
        return False
    
    # Gate 5: Neural Network Instantiation
    print("\n📋 QUALITY GATE 5: Neural Network Instantiation")
    try:
        # Test SNN model creation
        model = SpikingYOLO.from_pretrained("yolo_v4_spiking_dvs", backend="cpu")
        print("✅ SpikingYOLO model instantiated")
        
        # Test custom SNN
        custom_model = CustomSNN(
            input_size=(128, 128),
            hidden_channels=[32, 64],
            output_classes=2
        )
        print("✅ CustomSNN model instantiated")
        
        # Get model stats
        stats = custom_model.get_model_statistics()
        print(f"✅ Model parameters: {stats['trainable_parameters']:,}")
        
    except Exception as e:
        print(f"❌ Neural network error: {e}")
        return False
    
    # Gate 6: End-to-End Processing Pipeline
    print("\n📋 QUALITY GATE 6: End-to-End Pipeline")
    try:
        # Initialize components
        camera = DVSCamera(sensor_type="DVS128")
        model = SpikingYOLO.from_pretrained("yolo_v4_spiking_dvs", backend="cpu")
        
        # Generate events
        test_events = np.array([
            [64.0, 64.0, 0.001, 1],
            [65.0, 65.0, 0.002, -1],
            [66.0, 66.0, 0.003, 1],
            [67.0, 67.0, 0.004, -1]
        ])
        
        # Run detection
        detections = model.detect(
            test_events,
            integration_time=10e-3,
            threshold=0.5
        )
        
        print(f"✅ End-to-end pipeline: {len(detections)} detections")
        print(f"✅ Inference time: {model.last_inference_time:.2f}ms")
        
        # Test preprocessing
        preprocessor = EventPreprocessor()
        print("✅ Event preprocessor instantiated")
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return False
    
    # Gate 7: Performance Baseline
    print("\n📋 QUALITY GATE 7: Performance Baseline")
    try:
        camera = DVSCamera(sensor_type="DVS128")
        model = SpikingYOLO.from_pretrained("yolo_v4_spiking_dvs", backend="cpu")
        
        # Measure event generation rate
        import time
        start_time = time.time()
        total_events = 0
        
        for batch in camera.stream(duration=0.2):  # 200ms test
            total_events += len(batch)
            
        elapsed = time.time() - start_time
        event_rate = total_events / elapsed
        
        print(f"✅ Event generation rate: {event_rate:.0f} events/sec")
        print(f"✅ Processing efficiency: {total_events} events in {elapsed:.3f}s")
        
        # Test detection latency
        test_events = np.random.rand(100, 4) * [128, 128, 0.01, 2] - [0, 0, 0, 1]
        detections = model.detect(test_events)
        
        print(f"✅ Detection latency: {model.last_inference_time:.2f}ms")
        
        # Memory usage check
        import psutil
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"✅ Memory usage: {memory_mb:.1f} MB")
        
    except Exception as e:
        print(f"❌ Performance baseline error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 GENERATION 1 QUALITY GATES: ALL PASSED")
    print("✅ Core functionality working")
    print("✅ Event processing pipeline operational")
    print("✅ Neural network inference functional")
    print("✅ Performance baseline established")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    # Activate virtual environment programmatically
    venv_path = "/root/repo/venv/bin/activate_this.py"
    if os.path.exists(venv_path):
        exec(open(venv_path).read(), {'__file__': venv_path})
    
    success = progressive_quality_gate_1()
    exit(0 if success else 1)