"""
Command Line Interface for Spike-SNN Event Vision Kit.

Provides command-line tools for training, detection, and benchmarking
of spiking neural networks on event camera data.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import json
import time

import torch
import numpy as np

from .models import SpikingYOLO, CustomSNN, TrainingConfig
from .training import SpikingTrainer, EventDataLoader, create_training_config
from .core import DVSCamera, EventDataset, load_events_from_file, save_events_to_file


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('spike_snn.log')
        ]
    )


def train_command(args):
    """Training command implementation."""
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")
    
    # Validate environment
    if not torch.cuda.is_available() and not args.cpu:
        logger.warning("CUDA not available, falling back to CPU")
    
    # Load configuration
    if args.config:
        try:
            with open(args.config, 'r') as f:
                if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
            config = create_training_config(**config_dict)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            return 1
    else:
        config = create_training_config(
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        logger.info("Using default configuration")
    
    # Create model
    if args.model == "spiking_yolo":
        model = SpikingYOLO(
            input_size=(args.input_height, args.input_width),
            num_classes=args.num_classes,
            time_steps=args.time_steps,
            config=config
        )
    elif args.model == "custom_snn":
        model = CustomSNN(
            input_size=(args.input_height, args.input_width),
            output_classes=args.num_classes,
            config=config
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    logger.info(f"Created {args.model} with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create data loaders
    try:
        train_loader, val_loader, _ = EventDataLoader.create_loaders(
            args.dataset,
            batch_size=config.batch_size
        )
        logger.info(f"Loaded dataset: {args.dataset}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1
    
    # Create trainer
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    trainer = SpikingTrainer(model, config, device)
    
    # Train model
    try:
        history = trainer.fit(
            train_loader,
            val_loader,
            save_dir=args.output_dir
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        
        # Save final results
        if args.output_dir:
            results_path = Path(args.output_dir) / "results.json"
            with open(results_path, 'w') as f:
                json.dump({
                    'best_val_loss': trainer.best_val_loss,
                    'total_epochs': len(history['train_loss']),
                    'model_parameters': sum(p.numel() for p in model.parameters()),
                    'config': config.__dict__ if hasattr(config, '__dict__') else str(config)
                }, f, indent=2)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def detect_command(args):
    """Detection command implementation."""
    logger = logging.getLogger(__name__)
    logger.info("Starting detection...")
    
    # Load model
    if args.model_path:
        # Load pretrained model
        if args.model == "spiking_yolo":
            model = SpikingYOLO(
                input_size=(args.input_height, args.input_width),
                num_classes=args.num_classes
            )
        else:
            model = CustomSNN(
                input_size=(args.input_height, args.input_width),
                output_classes=args.num_classes
            )
        
        try:
            model.load_checkpoint(args.model_path)
            logger.info(f"Loaded model from {args.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return 1
    else:
        # Use pretrained model
        model = SpikingYOLO.from_pretrained(
            args.pretrained_model,
            backend="cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        )
        logger.info(f"Loaded pretrained model: {args.pretrained_model}")
    
    # Setup detection source
    if args.input_file:
        # Process from file
        try:
            events, metadata = load_events_from_file(args.input_file)
            logger.info(f"Loaded {len(events)} events from {args.input_file}")
            
            # Run detection
            detections = model.detect(
                events,
                integration_time=args.integration_time / 1000.0,  # Convert ms to s
                threshold=args.threshold
            )
            
            logger.info(f"Detected {len(detections)} objects")
            
            # Save results
            if args.output_file:
                results = {
                    'detections': detections,
                    'metadata': metadata,
                    'model_info': {
                        'model_type': args.model,
                        'integration_time_ms': args.integration_time,
                        'threshold': args.threshold
                    }
                }
                
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Saved results to {args.output_file}")
            
            # Print detections
            for i, detection in enumerate(detections):
                print(f"Detection {i}: {detection}")
                
        except Exception as e:
            logger.error(f"Detection from file failed: {e}")
            return 1
            
    elif args.camera:
        # Real-time detection from camera
        try:
            camera = DVSCamera(sensor_type=args.camera)
            camera.start_streaming()
            
            logger.info(f"Started {args.camera} camera, press Ctrl+C to stop")
            
            total_detections = 0
            start_time = time.time()
            
            try:
                while True:
                    events = camera.get_events(timeout=0.1)
                    if events is not None and len(events) > 0:
                        detections = model.detect(
                            events,
                            integration_time=args.integration_time / 1000.0,
                            threshold=args.threshold
                        )
                        
                        total_detections += len(detections)
                        
                        if detections:
                            print(f"Frame detections: {len(detections)}")
                            for detection in detections:
                                print(f"  {detection}")
                                
            except KeyboardInterrupt:
                logger.info("Detection stopped by user")
                
            finally:
                camera.stop_streaming()
                runtime = time.time() - start_time
                logger.info(f"Processed for {runtime:.1f}s, total detections: {total_detections}")
                
        except Exception as e:
            logger.error(f"Real-time detection failed: {e}")
            return 1
    else:
        logger.error("Must specify either --input-file or --camera")
        return 1
    
    return 0


def benchmark_command(args):
    """Benchmarking command implementation."""
    logger = logging.getLogger(__name__)
    logger.info("Starting benchmark...")
    
    # Load model
    if args.model == "spiking_yolo":
        model = SpikingYOLO(
            input_size=(args.input_height, args.input_width),
            num_classes=args.num_classes
        )
    else:
        model = CustomSNN(
            input_size=(args.input_height, args.input_width),
            output_classes=args.num_classes
        )
    
    if args.model_path:
        try:
            model.load_checkpoint(args.model_path)
            logger.info(f"Loaded model from {args.model_path}")
        except Exception as e:
            logger.warning(f"Could not load model: {e}, using random weights")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    
    # Create synthetic input for benchmarking
    batch_size = args.batch_size
    height, width = args.input_height, args.input_width
    time_steps = 10
    
    # Event tensor format: [batch, channels=2, height, width, time]
    sample_input = torch.randn(batch_size, 2, height, width, time_steps).to(device)
    
    logger.info(f"Benchmarking on {device} with input shape: {sample_input.shape}")
    
    # Profile inference
    try:
        profile_results = model.profile_inference(sample_input)
        
        # Display results
        print("\n" + "="*50)
        print("BENCHMARK RESULTS")
        print("="*50)
        print(f"Model: {args.model}")
        print(f"Device: {device}")
        print(f"Input shape: {sample_input.shape}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Memory usage: {torch.cuda.memory_allocated() / 1e6:.1f} MB" if device.type == "cuda" else "N/A")
        print("-"*50)
        print(f"Mean latency: {profile_results['mean_latency_ms']:.2f} ± {profile_results['std_latency_ms']:.2f} ms")
        print(f"Min latency:  {profile_results['min_latency_ms']:.2f} ms")
        print(f"Max latency:  {profile_results['max_latency_ms']:.2f} ms")
        print(f"Throughput:   {profile_results['throughput_fps']:.1f} FPS")
        print("="*50)
        
        # Save results
        if args.output_file:
            benchmark_results = {
                'model': args.model,
                'device': str(device),
                'input_shape': list(sample_input.shape),
                'parameters': sum(p.numel() for p in model.parameters()),
                'performance': profile_results,
                'timestamp': time.time()
            }
            
            with open(args.output_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            logger.info(f"Saved benchmark results to {args.output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        return 1


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Spike-SNN Event Vision Kit CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train a spiking neural network")
    train_parser.add_argument("--model", default="spiking_yolo", choices=["spiking_yolo", "custom_snn"])
    train_parser.add_argument("--dataset", default="synthetic", help="Dataset to use")
    train_parser.add_argument("--config", help="Configuration file (YAML or JSON)")
    train_parser.add_argument("--output-dir", default="./checkpoints", help="Output directory")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--num-classes", type=int, default=2)
    train_parser.add_argument("--input-height", type=int, default=128)
    train_parser.add_argument("--input-width", type=int, default=128)
    train_parser.add_argument("--time-steps", type=int, default=10)
    train_parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    # Detection command
    detect_parser = subparsers.add_parser("detect", help="Run object detection")
    detect_parser.add_argument("--model", default="spiking_yolo", choices=["spiking_yolo", "custom_snn"])
    detect_parser.add_argument("--model-path", help="Path to trained model checkpoint")
    detect_parser.add_argument("--pretrained-model", default="yolo_v4_spiking_dvs", help="Pretrained model name")
    detect_parser.add_argument("--input-file", help="Input event file")
    detect_parser.add_argument("--output-file", help="Output detection file")
    detect_parser.add_argument("--camera", help="Camera type (DVS128, DVS240, DAVIS346, Prophesee)")
    detect_parser.add_argument("--integration-time", type=float, default=10.0, help="Integration time in ms")
    detect_parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    detect_parser.add_argument("--num-classes", type=int, default=80)
    detect_parser.add_argument("--input-height", type=int, default=128)
    detect_parser.add_argument("--input-width", type=int, default=128)
    detect_parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark model performance")
    benchmark_parser.add_argument("--model", default="spiking_yolo", choices=["spiking_yolo", "custom_snn"])
    benchmark_parser.add_argument("--model-path", help="Path to trained model checkpoint")
    benchmark_parser.add_argument("--batch-size", type=int, default=1)
    benchmark_parser.add_argument("--num-classes", type=int, default=80)
    benchmark_parser.add_argument("--input-height", type=int, default=128)
    benchmark_parser.add_argument("--input-width", type=int, default=128)
    benchmark_parser.add_argument("--output-file", help="Output benchmark results file")
    benchmark_parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Dispatch to command
    if args.command == "train":
        return train_command(args)
    elif args.command == "detect":
        return detect_command(args)
    elif args.command == "benchmark":
        return benchmark_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())