#!/usr/bin/env python3
"""
Basic usage examples for Spike SNN Event Vision Kit.

This script demonstrates the fundamental capabilities of the toolkit including:
- Event camera setup and streaming
- Basic event processing
- SNN model inference
- Visualization
"""

import numpy as np
import time
import logging
from pathlib import Path

# Import the spike-snn-event toolkit
import spike_snn_event as snn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_camera_setup():
    """Example 1: Basic event camera setup and streaming."""
    logger.info("=== Example 1: Basic Camera Setup ===")
    
    # Create DVS camera instance
    camera = snn.DVSCamera(sensor_type="DVS128")
    
    # Configure camera settings
    config = snn.CameraConfig(
        width=128,
        height=128,
        noise_filter=True,
        refractory_period=1e-3,
        hot_pixel_threshold=1000
    )
    camera.config = config
    
    # Start streaming and collect some events
    logger.info("Starting event stream...")
    event_count = 0
    
    for events in camera.stream(duration=2.0):  # Stream for 2 seconds
        if len(events) > 0:
            event_count += len(events)
            logger.info(f"Received batch of {len(events)} events")
            
            # Print first few events for inspection
            if event_count < 100:
                print(f"Sample events: {events[:3]}")
    
    logger.info(f"Total events collected: {event_count}")
    logger.info("Camera streaming completed\n")


def example_2_event_preprocessing():
    """Example 2: Event preprocessing and filtering."""
    logger.info("=== Example 2: Event Preprocessing ===")
    
    # Create synthetic event data for demonstration
    num_events = 1000
    events = np.random.rand(num_events, 4)
    events[:, 0] *= 128  # x coordinates (0-128)
    events[:, 1] *= 128  # y coordinates (0-128)
    events[:, 2] = np.sort(np.random.rand(num_events) * 0.1)  # sorted timestamps
    events[:, 3] = np.random.choice([-1, 1], num_events)  # polarity
    
    logger.info(f"Created {len(events)} synthetic events")
    
    # Initialize preprocessor
    preprocessor = snn.SpatioTemporalPreprocessor(
        spatial_size=(64, 64),  # Downsample to 64x64
        time_bins=10
    )
    
    # Process events
    logger.info("Processing events...")
    processed_events = preprocessor.process(events)
    
    logger.info(f"Processed events shape: {processed_events.shape}")
    
    # Get processing statistics
    stats = preprocessor.get_statistics()
    logger.info(f"Processing stats: {stats}")
    logger.info("Event preprocessing completed\n")


def example_3_snn_inference():
    """Example 3: SNN model inference."""
    logger.info("=== Example 3: SNN Model Inference ===")
    
    try:
        # Create a simple SNN model
        model = snn.CustomSNN(
            input_size=(128, 128),
            hidden_channels=[32, 64],
            output_classes=2,
            neuron_type="LIF"
        )
        
        logger.info(f"Created SNN model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Generate synthetic event data
        events = np.random.rand(500, 4)
        events[:, 0] *= 128
        events[:, 1] *= 128
        events[:, 2] = np.sort(np.random.rand(500) * 0.05)
        events[:, 3] = np.random.choice([-1, 1], 500)
        
        # Convert events to tensor format
        event_tensor = model.events_to_tensor(events, time_window=5e-3)
        logger.info(f"Event tensor shape: {event_tensor.shape}")
        
        # Run inference
        logger.info("Running inference...")
        model.eval()
        
        import torch
        with torch.no_grad():
            output = model(event_tensor)
            prediction = torch.softmax(output, dim=1)
        
        logger.info(f"Model output: {output}")
        logger.info(f"Predictions: {prediction}")
        
        # Get model statistics
        stats = model.get_model_statistics()
        logger.info(f"Model stats: {stats}")
        
    except ImportError:
        logger.warning("PyTorch not available, skipping SNN inference example")
    
    logger.info("SNN inference completed\n")


def example_4_yolo_detection():
    """Example 4: Event-based object detection with SpikingYOLO."""
    logger.info("=== Example 4: SpikingYOLO Detection ===")
    
    try:
        # Create SpikingYOLO model
        yolo_model = snn.SpikingYOLO(
            input_size=(128, 128),
            num_classes=10,
            time_steps=10
        )
        yolo_model.set_backend("cpu")  # Use CPU for this example
        
        logger.info("Created SpikingYOLO model")
        
        # Generate synthetic event data
        events = np.random.rand(1000, 4)
        events[:, 0] *= 128
        events[:, 1] *= 128
        events[:, 2] = np.sort(np.random.rand(1000) * 0.1)
        events[:, 3] = np.random.choice([-1, 1], 1000)
        
        # Run detection
        logger.info("Running object detection...")
        detections = yolo_model.detect(
            events,
            integration_time=10e-3,  # 10ms integration window
            threshold=0.3
        )
        
        logger.info(f"Found {len(detections)} detections")
        for i, detection in enumerate(detections):
            logger.info(f"Detection {i}: {detection}")
        
        logger.info(f"Inference time: {yolo_model.last_inference_time:.2f}ms")
        
    except ImportError:
        logger.warning("PyTorch not available, skipping YOLO detection example")
    
    logger.info("YOLO detection completed\n")


def example_5_visualization():
    """Example 5: Event visualization."""
    logger.info("=== Example 5: Event Visualization ===")
    
    # Create visualizer
    visualizer = snn.EventVisualizer(width=128, height=128)
    
    # Generate events for visualization
    events = np.random.rand(200, 4)
    events[:, 0] *= 128
    events[:, 1] *= 128
    events[:, 2] = np.sort(np.random.rand(200) * 0.01)
    events[:, 3] = np.random.choice([-1, 1], 200)
    
    logger.info(f"Visualizing {len(events)} events")
    
    # Update visualization
    vis_image = visualizer.update(events)
    logger.info(f"Generated visualization image with shape: {vis_image.shape}")
    
    # Simulate detections for visualization
    fake_detections = [
        {
            'bbox': [20, 30, 40, 50],
            'confidence': 0.85,
            'class_name': 'object1'
        },
        {
            'bbox': [60, 70, 30, 25],
            'confidence': 0.72,
            'class_name': 'object2'
        }
    ]
    
    # Draw detections on image
    vis_with_detections = visualizer.draw_detections(vis_image, fake_detections)
    logger.info(f"Added {len(fake_detections)} detection overlays")
    
    logger.info("Event visualization completed\n")


def example_6_file_operations():
    """Example 6: Loading and saving events."""
    logger.info("=== Example 6: File Operations ===")
    
    # Generate sample events
    events = np.random.rand(1000, 4)
    events[:, 0] *= 128
    events[:, 1] *= 128
    events[:, 2] = np.sort(np.random.rand(1000) * 0.1)
    events[:, 3] = np.random.choice([-1, 1], 1000)
    
    # Metadata
    metadata = {
        'sensor_type': 'DVS128',
        'resolution': [128, 128],
        'duration': 0.1,
        'event_count': len(events)
    }
    
    # Save events to different formats
    output_dir = Path("example_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save as NumPy format
    npy_path = output_dir / "events.npy"
    snn.save_events_to_file(events, str(npy_path), metadata)
    logger.info(f"Saved events to {npy_path}")
    
    # Save as text format
    txt_path = output_dir / "events.txt"
    snn.save_events_to_file(events, str(txt_path), metadata)
    logger.info(f"Saved events to {txt_path}")
    
    # Load events back
    loaded_events, loaded_metadata = snn.load_events_from_file(str(npy_path))
    logger.info(f"Loaded {len(loaded_events)} events")
    logger.info(f"Loaded metadata: {loaded_metadata}")
    
    # Verify data integrity
    if np.allclose(events, loaded_events):
        logger.info("✓ Data integrity check passed")
    else:
        logger.error("✗ Data integrity check failed")
    
    logger.info("File operations completed\n")


def example_7_training_pipeline():
    """Example 7: Basic training pipeline setup."""
    logger.info("=== Example 7: Training Pipeline ===")
    
    try:
        # Create training configuration
        config = snn.create_training_config(
            learning_rate=1e-3,
            epochs=10,
            batch_size=16,
            early_stopping_patience=5
        )
        logger.info(f"Created training config: {config}")
        
        # Create model for training
        model = snn.CustomSNN(
            input_size=(64, 64),
            hidden_channels=[32, 64],
            output_classes=2
        )
        
        # Initialize trainer
        trainer = snn.SpikingTrainer(model, config)
        logger.info("Initialized SNN trainer")
        
        # Create synthetic dataset
        dataset = snn.EventDataset.load("synthetic")  # This creates synthetic data
        if dataset:
            train_loader, val_loader = dataset.get_loaders(batch_size=config.batch_size)
            
            if train_loader and val_loader:
                logger.info(f"Created data loaders with batch size {config.batch_size}")
                logger.info(f"Training batches: {len(train_loader)}")
                logger.info(f"Validation batches: {len(val_loader)}")
                
                # Note: Actual training would be run here with trainer.fit()
                # trainer.fit(train_loader, val_loader, save_dir="checkpoints")
                logger.info("Training pipeline setup completed (training not run in example)")
            else:
                logger.warning("Could not create data loaders")
        else:
            logger.warning("Could not create dataset")
        
    except ImportError:
        logger.warning("PyTorch not available, skipping training pipeline example")
    
    logger.info("Training pipeline example completed\n")


def main():
    """Run all examples."""
    logger.info("Starting Spike SNN Event Vision Kit Examples")
    logger.info("=" * 50)
    
    examples = [
        example_1_basic_camera_setup,
        example_2_event_preprocessing,
        example_3_snn_inference,
        example_4_yolo_detection,
        example_5_visualization,
        example_6_file_operations,
        example_7_training_pipeline
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            logger.error(f"Example {i} failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Small delay between examples
        time.sleep(0.5)
    
    logger.info("All examples completed!")
    logger.info("Check the 'example_output' directory for generated files.")


if __name__ == "__main__":
    main()