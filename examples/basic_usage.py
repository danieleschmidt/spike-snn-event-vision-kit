#!/usr/bin/env python3
"""
Basic Usage Example for Spike-SNN Event Vision Kit

This example demonstrates:
1. Event camera setup and streaming
2. Basic spiking neural network inference
3. Real-time event processing
4. Detection visualization

Run with: python examples/basic_usage.py
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from spike_snn_event import (
    DVSCamera, 
    EventDataset,
    SpatioTemporalPreprocessor,
    EventVisualizer,
    PYTORCH_MODELS_AVAILABLE
)

if PYTORCH_MODELS_AVAILABLE:
    from spike_snn_event import SpikingYOLO, CustomSNN


def basic_event_detection_demo():
    """Demonstrate basic event-based object detection."""
    print("🚀 Basic Event Detection Demo")
    print("=" * 50)
    
    if not PYTORCH_MODELS_AVAILABLE:
        print("⚠️  PyTorch models not available, skipping neural network demo")
        return
    
    # 1. Initialize event camera
    print("📷 Initializing DVS camera...")
    camera = DVSCamera(
        sensor_type="DVS128",
        noise_filter=True,
        refractory_period=1e-3  # 1ms refractory period
    )
    
    # 2. Load pre-trained spiking YOLO model
    print("🧠 Loading pre-trained Spiking YOLO model...")
    model = SpikingYOLO.from_pretrained(
        "yolo_v4_spiking_dvs",
        backend="cpu",  # Use CPU for compatibility
        time_steps=10
    )
    
    # 3. Setup event visualizer
    visualizer = EventVisualizer(width=640, height=480)
    
    print("\n🎬 Starting event detection (10 seconds)...")
    print("Real-time statistics:")
    
    # 4. Real-time detection loop
    total_events = 0
    total_detections = 0
    start_time = time.time()
    
    for frame_idx, events in enumerate(camera.stream(duration=10.0)):
        if len(events) == 0:
            continue
            
        total_events += len(events)
        
        # Run spiking inference
        detections = model.detect(
            events,
            integration_time=10e-3,  # 10ms integration window
            threshold=0.5
        )
        
        total_detections += len(detections)
        
        # Update visualization
        vis_image = visualizer.update(events)
        vis_image = visualizer.draw_detections(vis_image, detections)
        
        # Print statistics every 10 frames
        if frame_idx % 10 == 0:
            elapsed = time.time() - start_time
            fps = frame_idx / elapsed if elapsed > 0 else 0
            
            print(f"Frame {frame_idx:3d} | "
                  f"Events: {len(events):4d} | "
                  f"Detections: {len(detections):2d} | "
                  f"Latency: {model.last_inference_time:.1f}ms | "
                  f"FPS: {fps:.1f}")
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"\n📊 Final Statistics:")
    print(f"   Total runtime:    {total_time:.1f}s")
    print(f"   Total events:     {total_events:,}")
    print(f"   Total detections: {total_detections}")
    print(f"   Events/second:    {total_events / total_time:.0f}")
    print(f"   Detection rate:   {total_detections / total_time:.1f}/s")


def custom_snn_training_demo():
    """Demonstrate training a custom SNN on synthetic data."""
    print("\n🧪 Custom SNN Training Demo")
    print("=" * 50)
    
    if not PYTORCH_MODELS_AVAILABLE:
        print("⚠️  PyTorch models not available, skipping training demo")
        return
    
    # 1. Create custom SNN model
    print("🧠 Creating custom SNN model...")
    model = CustomSNN(
        input_size=(128, 128),
        hidden_channels=[64, 128, 256],
        output_classes=2,  # Binary classification
        neuron_type="LIF",
        surrogate_gradient="fast_sigmoid"
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. Load synthetic dataset
    print("📁 Loading synthetic dataset...")
    dataset = EventDataset.load("N-CARS")  # This will create synthetic data
    train_loader, val_loader = dataset.get_loaders(
        batch_size=16,
        time_window=50e-3,
        augmentation=True
    )
    
    if train_loader is None:
        print("⚠️  PyTorch not available, skipping training demo")
        return
    
    # 3. Setup training configuration
    from spike_snn_event.training import SpikingTrainer, create_training_config
    
    config = create_training_config(
        learning_rate=1e-3,
        epochs=5,  # Short demo
        batch_size=16,
        loss_function="cross_entropy"
    )
    
    # 4. Train model
    print("🏃 Starting training...")
    device = torch.device("cpu")  # Use CPU for compatibility
    trainer = SpikingTrainer(model, config, device)
    
    try:
        history = trainer.fit(
            train_loader,
            val_loader,
            save_dir="./demo_checkpoints"
        )
        
        print("✅ Training completed!")
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"Final train accuracy: {history['train_accuracy'][-1]:.4f}")
        
    except Exception as e:
        print(f"⚠️  Training demo failed: {e}")


def event_preprocessing_demo():
    """Demonstrate event preprocessing pipeline."""
    print("\n🔧 Event Preprocessing Demo")
    print("=" * 50)
    
    # 1. Create preprocessor
    preprocessor = SpatioTemporalPreprocessor(
        spatial_size=(256, 256),
        time_bins=5
    )
    
    # 2. Generate synthetic events
    print("🎲 Generating synthetic events...")
    num_events = 1000
    events = np.zeros((num_events, 4))
    events[:, 0] = np.random.uniform(0, 128, num_events)    # x
    events[:, 1] = np.random.uniform(0, 128, num_events)    # y
    events[:, 2] = np.cumsum(np.random.exponential(1e-3, num_events))  # timestamps
    events[:, 3] = np.random.choice([-1, 1], num_events)   # polarity
    
    print(f"Generated {len(events)} events")
    print(f"Time span: {events[:, 2].max() - events[:, 2].min():.3f}s")
    
    # 3. Process events
    print("⚙️  Processing events...")
    processed_events = preprocessor.process(events)
    
    print(f"Processed shape: {processed_events.shape}")
    
    # 4. Show preprocessing statistics
    stats = preprocessor.get_statistics()
    print("📊 Preprocessing Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")


def file_io_demo():
    """Demonstrate event file I/O operations."""
    print("\n💾 Event File I/O Demo")
    print("=" * 50)
    
    from spike_snn_event.core import save_events_to_file, load_events_from_file
    
    # 1. Generate sample events
    print("🎲 Generating sample events...")
    num_events = 500
    events = np.zeros((num_events, 4))
    events[:, 0] = np.random.uniform(0, 240, num_events)  # x
    events[:, 1] = np.random.uniform(0, 180, num_events)  # y
    events[:, 2] = np.cumsum(np.random.exponential(2e-3, num_events))  # timestamps
    events[:, 3] = np.random.choice([-1, 1], num_events)  # polarity
    
    # 2. Save to different formats
    print("💾 Saving events to different formats...")
    metadata = {
        "sensor_type": "DVS240",
        "recording_duration": events[:, 2].max(),
        "total_events": len(events)
    }
    
    formats = [".npy", ".txt", ".dat"]
    for fmt in formats:
        filepath = f"demo_events{fmt}"
        try:
            save_events_to_file(events, filepath, metadata)
            print(f"   ✅ Saved to {filepath}")
        except Exception as e:
            print(f"   ⚠️  Failed to save {filepath}: {e}")
    
    # 3. Load and verify
    print("📂 Loading and verifying events...")
    for fmt in formats:
        filepath = f"demo_events{fmt}"
        try:
            loaded_events, loaded_metadata = load_events_from_file(filepath)
            
            if np.allclose(events, loaded_events, rtol=1e-5):
                print(f"   ✅ {filepath}: Events match!")
            else:
                print(f"   ⚠️  {filepath}: Events don't match")
                
            if loaded_metadata:
                print(f"   📊 Metadata keys: {list(loaded_metadata.keys())}")
                
        except Exception as e:
            print(f"   ⚠️  Failed to load {filepath}: {e}")


def performance_analysis_demo():
    """Demonstrate model performance analysis."""
    print("\n⚡ Performance Analysis Demo")
    print("=" * 50)
    
    if not PYTORCH_MODELS_AVAILABLE:
        print("⚠️  PyTorch models not available, skipping performance analysis demo")
        return
    
    # 1. Create models for comparison
    models = {
        "Small SNN": CustomSNN(
            input_size=(64, 64),
            hidden_channels=[32, 64],
            output_classes=2
        ),
        "Medium SNN": CustomSNN(
            input_size=(128, 128),
            hidden_channels=[64, 128, 256],
            output_classes=2
        ),
        "Large SNN": CustomSNN(
            input_size=(256, 256),
            hidden_channels=[128, 256, 512],
            output_classes=2
        )
    }
    
    # 2. Analyze each model
    print("🔍 Analyzing model performance...")
    
    for name, model in models.items():
        print(f"\n📊 {name}:")
        
        # Model statistics
        stats = model.get_model_statistics()
        print(f"   Parameters: {stats['total_parameters']:,}")
        
        # Create sample input
        h, w = model.input_size
        sample_input = torch.randn(1, 2, h, w, 10)
        
        # Profile inference
        try:
            profile_results = model.profile_inference(sample_input)
            print(f"   Mean latency: {profile_results['mean_latency_ms']:.2f}ms")
            print(f"   Throughput: {profile_results['throughput_fps']:.1f} FPS")
        except Exception as e:
            print(f"   ⚠️  Profiling failed: {e}")


def main():
    """Run all demo functions."""
    print("🎯 Spike-SNN Event Vision Kit - Basic Usage Examples")
    print("=" * 60)
    
    demos = [
        ("Basic Event Detection", basic_event_detection_demo),
        ("Custom SNN Training", custom_snn_training_demo),
        ("Event Preprocessing", event_preprocessing_demo),
        ("File I/O Operations", file_io_demo),
        ("Performance Analysis", performance_analysis_demo),
    ]
    
    for demo_name, demo_func in demos:
        try:
            demo_func()
            print(f"\n✅ {demo_name} completed successfully!\n")
        except Exception as e:
            print(f"\n❌ {demo_name} failed: {e}")
            import traceback
            traceback.print_exc()
            print()
        
        # Pause between demos
        time.sleep(1)
    
    print("🎉 All demos completed!")
    print("\nNext steps:")
    print("- Try the CLI: python -m spike_snn_event.cli demo")
    print("- Train a model: python -m spike_snn_event.cli train --epochs 20")
    print("- Run detection: python -m spike_snn_event.cli detect --input demo")


if __name__ == "__main__":
    main()