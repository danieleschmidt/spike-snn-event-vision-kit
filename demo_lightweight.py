#!/usr/bin/env python3
"""
Lightweight demo of Spike SNN Event Vision Kit functionality.

This demo shows the basic architecture and functionality using only
Python standard library, no external dependencies required.
"""

import time
import sys
import os

# Add src to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_core_functionality():
    """Demonstrate core functionality with built-in types."""
    print("=== Spike SNN Event Vision Kit - Lightweight Demo ===")
    print()
    
    # Test 1: Module Import Test
    print("1. Testing lightweight module imports...")
    try:
        from spike_snn_event.lite_core import (
            DVSCamera, CameraConfig, SpatioTemporalPreprocessor,
            EventVisualizer, LiteEventSNN
        )
        print("✓ Lightweight modules imported successfully")
        
    except ImportError as e:
        print(f"✗ Failed to import modules: {e}")
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
        
        # Test different sensor types
        sensors = ["DVS128", "DVS240", "DAVIS346", "Prophesee"]
        for sensor in sensors:
            test_camera = DVSCamera(sensor_type=sensor)
            print(f"  - {sensor}: {test_camera.width}x{test_camera.height} ✓")
        
    except Exception as e:
        print(f"✗ Camera configuration failed: {e}")
        return
    
    print()
    
    # Test 3: Event Stream Generation
    print("3. Testing event stream generation...")
    try:
        camera = DVSCamera(sensor_type="DVS128")
        
        total_events = 0
        batch_count = 0
        
        print("  Starting event stream (2 seconds)...")
        for events in camera.stream(duration=2.0):
            if events:
                total_events += len(events)
                batch_count += 1
                
                if batch_count <= 3:  # Show first 3 batches
                    print(f"    Batch {batch_count}: {len(events)} events")
                    if events:
                        sample_event = events[0]
                        print(f"      Sample: x={sample_event[0]:.1f}, y={sample_event[1]:.1f}, "
                              f"t={sample_event[2]:.6f}, p={sample_event[3]}")
        
        print(f"✓ Generated {total_events} events in {batch_count} batches")
        print(f"  - Stats: {camera.stats}")
        
    except Exception as e:
        print(f"✗ Event stream generation failed: {e}")
        return
    
    print()
    
    # Test 4: Event Preprocessing
    print("4. Testing event preprocessing...")
    try:
        preprocessor = SpatioTemporalPreprocessor(
            spatial_size=(64, 64),
            time_bins=10
        )
        
        # Generate sample events
        camera = DVSCamera(sensor_type="DVS128")
        sample_events = camera._generate_synthetic_events(200)
        print(f"  Generated {len(sample_events)} events for preprocessing")
        
        # Process events
        processed_events = preprocessor.process(sample_events)
        print(f"✓ Processed events: {len(processed_events)} output events")
        
        # Get statistics
        stats = preprocessor.get_statistics()
        print(f"  - Processing stats: {stats}")
        
    except Exception as e:
        print(f"✗ Event preprocessing failed: {e}")
        return
    
    print()
    
    # Test 5: SNN Model Inference
    print("5. Testing SNN model inference...")
    try:
        # Create lightweight SNN model
        model = LiteEventSNN(
            input_size=(128, 128),
            num_classes=5
        )
        
        stats = model.get_model_statistics()
        print(f"✓ Created SNN model with {stats['total_parameters']:,} parameters")
        print(f"  - Input size: {stats['input_width']}x{stats['input_height']}")
        print(f"  - Output classes: {stats['output_classes']}")
        
        # Generate events for inference
        camera = DVSCamera(sensor_type="DVS128")
        test_events = camera._generate_synthetic_events(100)
        
        # Run detection
        detections = model.detect(
            test_events,
            integration_time=10e-3,
            threshold=0.5
        )
        
        print(f"✓ Inference completed in {model.last_inference_time:.2f}ms")
        print(f"  - Found {len(detections)} detections")
        
        for i, detection in enumerate(detections):
            print(f"    Detection {i}: {detection['class_name']} "
                  f"({detection['confidence']:.2f}) at {detection['bbox']}")
        
    except Exception as e:
        print(f"✗ SNN model inference failed: {e}")
        return
    
    print()
    
    # Test 6: Event Visualization
    print("6. Testing event visualization...")
    try:
        visualizer = EventVisualizer(width=128, height=128)
        
        # Generate events for visualization
        camera = DVSCamera(sensor_type="DVS128")
        vis_events = camera._generate_synthetic_events(150)
        
        # Update visualization
        vis_info = visualizer.update(vis_events)
        print(f"✓ Visualization updated")
        print(f"  - Current batch: {vis_info['current_batch']} events")
        print(f"  - Total events: {vis_info['total_events']} events")
        
        # Add detections
        fake_detections = [
            {
                'bbox': [20, 30, 40, 50],
                'confidence': 0.85,
                'class_name': 'car'
            },
            {
                'bbox': [60, 70, 30, 25], 
                'confidence': 0.72,
                'class_name': 'person'
            }
        ]
        
        vis_with_detections = visualizer.draw_detections(vis_info, fake_detections)
        print(f"✓ Added {vis_with_detections['detection_count']} detection overlays")
        
    except Exception as e:
        print(f"✗ Event visualization failed: {e}")
        return
    
    print()
    
    # Test 7: Performance Benchmark
    print("7. Running performance benchmark...")
    try:
        camera = DVSCamera(sensor_type="DVS128")
        model = LiteEventSNN(input_size=(128, 128), num_classes=3)
        
        # Benchmark parameters
        num_iterations = 50
        total_events_processed = 0
        total_inference_time = 0
        
        print(f"  Running {num_iterations} iterations...")
        
        start_time = time.time()
        
        for i in range(num_iterations):
            # Generate events
            events = camera._generate_synthetic_events(100)
            total_events_processed += len(events)
            
            # Run inference
            detections = model.detect(events, threshold=0.3)
            total_inference_time += model.last_inference_time
            
            if (i + 1) % 10 == 0:
                print(f"    Completed {i + 1}/{num_iterations} iterations")
        
        total_time = time.time() - start_time
        
        print(f"✓ Benchmark completed in {total_time:.2f} seconds")
        print(f"  - Events processed: {total_events_processed:,}")
        print(f"  - Average events/second: {total_events_processed/total_time:.0f}")
        print(f"  - Average inference time: {total_inference_time/num_iterations:.2f}ms")
        print(f"  - Throughput: {num_iterations/total_time:.1f} inferences/second")
        
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        return
    
    print()
    
    # Test 8: End-to-End Pipeline
    print("8. Testing end-to-end pipeline...")
    try:
        # Create complete pipeline
        camera = DVSCamera(sensor_type="DAVIS346", config=CameraConfig(noise_filter=True))
        preprocessor = SpatioTemporalPreprocessor(spatial_size=(128, 128), time_bins=5)
        model = LiteEventSNN(input_size=(128, 128), num_classes=10)
        visualizer = EventVisualizer(width=346, height=240)
        
        print("✓ Pipeline components initialized")
        print(f"  - Camera: {camera.sensor_type} ({camera.width}x{camera.height})")
        print(f"  - Preprocessor: {preprocessor.spatial_size} spatial, {preprocessor.time_bins} time bins")
        print(f"  - Model: {model.get_model_statistics()['total_parameters']:,} parameters")
        
        # Run pipeline
        pipeline_results = []
        
        for i, events in enumerate(camera.stream(duration=1.0)):
            if not events:
                continue
                
            # Preprocess
            processed_events = preprocessor.process(events)
            
            # Inference
            detections = model.detect(processed_events, threshold=0.4)
            
            # Visualize
            vis_info = visualizer.update(processed_events)
            vis_with_detections = visualizer.draw_detections(vis_info, detections)
            
            pipeline_results.append({
                'batch': i,
                'input_events': len(events),
                'processed_events': len(processed_events),
                'detections': len(detections),
                'inference_time': model.last_inference_time
            })
            
            if i >= 5:  # Process 5 batches
                break
        
        print(f"✓ Pipeline processed {len(pipeline_results)} batches")
        
        # Summary statistics
        total_input = sum(r['input_events'] for r in pipeline_results)
        total_processed = sum(r['processed_events'] for r in pipeline_results)
        total_detections = sum(r['detections'] for r in pipeline_results)
        avg_inference_time = sum(r['inference_time'] for r in pipeline_results) / len(pipeline_results)
        
        print(f"  - Total input events: {total_input}")
        print(f"  - Total processed events: {total_processed}")
        print(f"  - Total detections: {total_detections}")
        print(f"  - Average inference time: {avg_inference_time:.2f}ms")
        
    except Exception as e:
        print(f"✗ End-to-end pipeline failed: {e}")
        return
    
    print()
    print("=== Demo Complete ===")
    print("🎉 All functionality tests passed!")
    print()
    print("Architecture Components Verified:")
    print("✅ Event camera simulation and streaming")  
    print("✅ Event preprocessing and filtering")
    print("✅ Spiking neural network inference")
    print("✅ Object detection pipeline")
    print("✅ Visualization framework")
    print("✅ Performance benchmarking")
    print("✅ End-to-end integration")
    print()
    print("Next steps:")
    print("- Install PyTorch: pip install torch torchvision")
    print("- Install OpenCV: pip install opencv-python") 
    print("- Install NumPy: pip install numpy matplotlib")
    print("- Run examples/basic_usage.py for full examples")
    print("- Check docs/ for detailed documentation")
    print("- Deploy on neuromorphic hardware (Loihi, Akida)")


if __name__ == "__main__":
    demo_core_functionality()