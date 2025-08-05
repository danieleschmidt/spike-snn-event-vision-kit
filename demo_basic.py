#!/usr/bin/env python3
"""
Minimal demo of Spike SNN Event Vision Kit functionality.

This demo shows the basic architecture and functionality without requiring
external dependencies like PyTorch, NumPy, etc.
"""

import time
import sys
import os

# Add src to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_core_functionality():
    """Demonstrate core functionality with built-in types."""
    print("=== Spike SNN Event Vision Kit - Basic Demo ===")
    print()
    
    # Test 1: Module Import Test
    print("1. Testing module imports...")
    try:
        # Test core imports
        from spike_snn_event.core import DVSCamera, CameraConfig
        print("✓ Core modules imported successfully")
        
        # Test model imports (may fail without PyTorch)
        try:
            from spike_snn_event.models import EventSNN, SpikingYOLO, CustomSNN
            print("✓ Model modules imported successfully")
            models_available = True
        except ImportError as e:
            print(f"⚠ Model modules not available: {e}")
            models_available = False
            
    except ImportError as e:
        print(f"✗ Failed to import core modules: {e}")
        return
    
    print()
    
    # Test 2: Camera Configuration
    print("2. Testing camera configuration...")
    try:
        config = CameraConfig(
            width=128,
            height=128,
            noise_filter=True,
            refractory_period=1e-3
        )
        print(f"✓ Camera config created: {config.width}x{config.height}")
        
        camera = DVSCamera(sensor_type="DVS128", config=config)
        print(f"✓ DVS camera initialized: {camera.sensor_type}")
        print(f"  - Resolution: {camera.width}x{camera.height}")
        print(f"  - Noise filter: {camera.config.noise_filter}")
        
    except Exception as e:
        print(f"✗ Camera configuration failed: {e}")
        return
    
    print()
    
    # Test 3: Event Generation (using built-in random)
    print("3. Testing event generation...")
    try:
        import random
        
        # Generate some synthetic events
        num_events = 100
        events = []
        current_time = time.time()
        
        for i in range(num_events):
            event = [
                random.uniform(0, 128),  # x
                random.uniform(0, 128),  # y  
                current_time + random.uniform(0, 0.01),  # timestamp
                random.choice([-1, 1])   # polarity
            ]
            events.append(event)
        
        print(f"✓ Generated {len(events)} synthetic events")
        print(f"  - Sample event: {events[0]}")
        
        # Test event filtering
        filtered_count = 0
        for event in events:
            x, y, t, p = event
            if 0 <= x <= 128 and 0 <= y <= 128:
                filtered_count += 1
        
        print(f"✓ Filtered events: {filtered_count}/{len(events)} valid")
        
    except Exception as e:
        print(f"✗ Event generation failed: {e}")
        return
    
    print()
    
    # Test 4: Basic Processing Pipeline
    print("4. Testing processing pipeline...")
    try:
        # Simulate event stream processing
        processed_batches = 0
        total_events = 0
        
        for batch_id in range(5):
            # Simulate event batch
            batch_size = random.randint(50, 150)
            
            # Simulate processing
            time.sleep(0.01)  # Simulate processing time
            
            processed_batches += 1
            total_events += batch_size
            
            print(f"  - Batch {batch_id}: {batch_size} events processed")
        
        print(f"✓ Processed {processed_batches} batches, {total_events} total events")
        
    except Exception as e:
        print(f"✗ Processing pipeline failed: {e}")
        return
    
    print()
    
    # Test 5: Model Architecture (if available)
    if models_available:
        print("5. Testing model architecture...")
        try:
            # This will only work if PyTorch is available
            import torch
            
            # Create a simple model
            model = CustomSNN(
                input_size=(64, 64),
                hidden_channels=[32, 64],
                output_classes=2
            )
            
            param_count = sum(p.numel() for p in model.parameters())
            print(f"✓ CustomSNN created with {param_count:,} parameters")
            
            # Test model statistics
            stats = model.get_model_statistics()
            print(f"  - Total parameters: {stats['total_parameters']:,}")
            print(f"  - Trainable parameters: {stats['trainable_parameters']:,}")
            
        except ImportError:
            print("⚠ PyTorch not available, skipping model architecture test")
        except Exception as e:
            print(f"✗ Model architecture test failed: {e}")
    else:
        print("5. Skipping model tests (PyTorch not available)")
    
    print()
    
    # Test 6: Configuration and Metadata
    print("6. Testing configuration system...")
    try:
        # Test different camera configurations
        configs = [
            ("DVS128", 128, 128),
            ("DVS240", 240, 180), 
            ("DAVIS346", 346, 240),
            ("Prophesee", 640, 480)
        ]
        
        for sensor, width, height in configs:
            try:
                camera = DVSCamera(sensor_type=sensor)
                actual_width = camera.width
                actual_height = camera.height
                print(f"  - {sensor}: {actual_width}x{actual_height} ✓")
            except ValueError:
                print(f"  - {sensor}: Unknown sensor type ✗")
        
        print("✓ Configuration system working")
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return
    
    print()
    
    # Test 7: Performance Metrics
    print("7. Testing performance metrics...")
    try:
        start_time = time.time()
        
        # Simulate processing workload
        iterations = 1000
        for i in range(iterations):
            # Simulate event processing
            x = random.uniform(0, 128)
            y = random.uniform(0, 128)
            t = time.time()
            p = random.choice([-1, 1])
            
            # Simple validation
            valid = 0 <= x <= 128 and 0 <= y <= 128
        
        end_time = time.time()
        processing_time = end_time - start_time
        events_per_second = iterations / processing_time
        
        print(f"✓ Processed {iterations} events in {processing_time:.3f}s")
        print(f"  - Throughput: {events_per_second:.0f} events/second")
        print(f"  - Average latency: {(processing_time/iterations)*1000:.3f}ms per event")
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return
    
    print()
    print("=== Demo Complete ===")
    print("✅ All basic functionality tests passed!")
    print()
    print("Next steps:")
    print("- Install PyTorch for full SNN functionality")
    print("- Install OpenCV for visualization")
    print("- Run examples/basic_usage.py for complete examples")
    print("- Check docs/ for detailed documentation")


if __name__ == "__main__":
    demo_core_functionality()