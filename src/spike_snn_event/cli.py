```python
"""
Command-line interface for the spike-snn-event-vision-kit.

Provides comprehensive CLI tools for training, detection, benchmarking,
and demonstration of spiking neural networks on event camera data.
"""

import argparse
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .core import DVSCamera, EventDataset, SpatioTemporalPreprocessor, load_events_from_file

# Import PyTorch-dependent classes conditionally
if TORCH_AVAILABLE:
    try:
        from .models import SpikingYOLO, CustomSNN, TrainingConfig
        from .training import SpikingTrainer, EventDataLoader, create_training_config
        PYTORCH_MODELS_AVAILABLE = True
    except ImportError:
        PYTORCH_MODELS_AVAILABLE = False
else:
    PYTORCH_MODELS_AVAILABLE = False


def setup_logging(verbose: bool):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def train_command(args, device):
    """Train command implementation."""
    if not PYTORCH_MODELS_AVAILABLE:
        print("❌ PyTorch models not available. Please install PyTorch to use training functionality.")
        return
        
    logger = logging.getLogger(__name__)
    print(f"🚀 Training {args.model} model on {args.dataset} dataset")
    print(f"📊 Config: Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}")
    print(f"💾 Save directory: {args.save_dir}")
    
    # Validate environment
    if not torch.cuda.is_available() and device.type == "cuda":
        logger.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    
    # Load configuration
    if args.config:
        try:
            with open(args.config, 'r') as f:
                if YAML_AVAILABLE and (args.config.endswith('.yaml') or args.config.endswith('.yml')):
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
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        logger.info("Using default configuration")
    
    # Create model
    print("🧠 Creating model...")
    if args.model == "yolo":
        model = SpikingYOLO(input_size=(128, 128), num_classes=2)
    else:
        model = CustomSNN(input_size=(128, 128), output_classes=2)
    
    logger.info(f"Created {args.model} with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create data loaders
    print("📁 Loading dataset...")
    try:
        train_loader, val_loader, _ = EventDataLoader.create_loaders(
            args.dataset,
            batch_size=config.batch_size
        )
        logger.info(f"Loaded dataset: {args.dataset}")
        
        if train_loader is None:
            print("❌ Failed to create data loaders. Check dependencies.")
            return 1
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1
    
    # Create trainer
    trainer = SpikingTrainer(model, config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"🔄 Resuming from checkpoint: {args.resume}")
        model.load_checkpoint(args.resume)
    
    # Train model
    print("🏃 Starting training...")
    try:
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=args.save_dir
        )
        
        print("✅ Training completed successfully!")
        print(f"📈 Final metrics:")
        print(f"   Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"   Train Acc:  {history['train_accuracy'][-1]:.4f}")
        if history.get('val_loss'):
            print(f"   Val Loss:   {history['val_loss'][-1]:.4f}")
            print(f"   Val Acc:    {history['val_accuracy'][-1]:.4f}")
        
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        
        # Save final results
        if args.save_dir:
            results_path = Path(args.save_dir) / "results.json"
            with open(results_path, 'w') as f:
                json.dump({
                    'best_val_loss': trainer.best_val_loss,
                    'total_epochs': len(history['train_loss']),
                    'model_parameters': sum(p.numel() for p in model.parameters()),
                    'config': config.__dict__ if hasattr(config, '__dict__') else str(config),
                    'final_metrics': {
                        'train_loss': history['train_loss'][-1],
                        'train_accuracy': history['train_accuracy'][-1],
                        'val_loss': history.get('val_loss', [None])[-1],
                        'val_accuracy': history.get('val_accuracy', [None])[-1]
                    }
                }, f, indent=2)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        return 1


def detect_command(args, device):
    """Detect command implementation."""
    if not PYTORCH_MODELS_AVAILABLE:
        print("❌ PyTorch models not available. Please install PyTorch to use detection functionality.")
        return
        
    print(f"🔍 Running object detection")
    print(f"📹 Input: {args.input}")
    print(f"🎯 Threshold: {args.threshold}")
    print(f"⏱️  Duration: {args.duration}s")
    
    try:
        # Load or create model
        if args.model_path:
            print(f"📥 Loading model from {args.model_path}")
            model = SpikingYOLO()
            model.load_checkpoint(args.model_path)
        else:
            print("🧠 Using pretrained model (demo weights)")
            model = SpikingYOLO.from_pretrained("yolo_v4_spiking_dvs", backend=device.type)
        
        model.to(device)
        model.eval()
        
        all_detections = []
        start_time = time.time()
        
        if args.input == "camera":
            print("📷 Starting camera stream...")
            camera = DVSCamera(sensor_type="DVS128")
            
            for events in camera.stream(duration=args.duration):
                if len(events) == 0:
                    continue
                    
                detections = model.detect(
                    events,
                    integration_time=args.integration_time,
                    threshold=args.threshold
                )
                
                current_time = time.time() - start_time
                print(f"⏰ {current_time:6.2f}s | Events: {len(events):4d} | "
                      f"Detections: {len(detections):2d} | "
                      f"Latency: {model.last_inference_time:.2f}ms")
                
                all_detections.extend(detections)
                
                if current_time >= args.duration:
                    break
                    
        elif args.input == "demo":
            print("🎮 Running demo mode with synthetic events...")
            camera = DVSCamera(sensor_type="DVS128")
            
            for i, events in enumerate(camera.stream(duration=args.duration)):
                if len(events) == 0:
                    continue
                    
                detections = model.detect(
                    events,
                    integration_time=args.integration_time,
                    threshold=args.threshold
                )
                
                if detections:
                    print(f"🎯 Frame {i:3d}: {len(detections)} detections")
                    for j, det in enumerate(detections):
                        bbox = det['bbox']
                        conf = det['confidence']
                        cls = det['class_name']
                        print(f"   [{j}] {cls}: {conf:.3f} at ({bbox[0]:.0f},{bbox[1]:.0f})")
                
                all_detections.extend(detections)
                time.sleep(0.1)  # Demo pace
                
        else:
            print(f"📁 Loading events from file: {args.input}")
            events, metadata = load_events_from_file(args.input)
            print(f"📊 Loaded {len(events)} events")
            
            detections = model.detect(
                events,
                integration_time=args.integration_time,
                threshold=args.threshold
            )
            
            all_detections.extend(detections)
            print(f"🎯 Found {len(detections)} detections")
        
        # Save results if output specified
        if args.output:
            print(f"💾 Saving results to {args.output}")
            with open(args.output, 'w') as f:
                json.dump(all_detections, f, indent=2)
        
        print(f"✅ Detection completed! Total detections: {len(all_detections)}")
        
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        raise


def benchmark_command(args, device):
    """Benchmark command implementation."""
    if not PYTORCH_MODELS_AVAILABLE:
        print("❌ PyTorch models not available. Please install PyTorch to use benchmark functionality.")
        return
    print(f"⚡ Benchmarking model performance")
    print(f"🔢 Iterations: {args.iterations}")
    print(f"📦 Batch size: {args.batch_size}")
    
    try:
        # Load or create model
        if args.model_path:
            print(f"📥 Loading model from {args.model_path}")
            model = CustomSNN()
            model.load_checkpoint(args.model_path)
        else:
            print("🧠 Using default model for benchmarking")
            model = CustomSNN(input_size=(128, 128), output_classes=2)
        
        model.to(device)
        model.eval()
        
        # Create sample input
        print("📊 Preparing benchmark data...")
        sample_input = torch.randn(args.batch_size, 2, 128, 128, 10).to(device)
        
        # Profile inference
        if args.profile:
            print("🔍 Running detailed profiling...")
            profile_results = model.profile_inference(sample_input)
            
            print("📈 Profiling Results:")
            print(f"   Mean Latency:  {profile_results['mean_latency_ms']:.2f} ± {profile_results['std_latency_ms']:.2f} ms")
            print(f"   Min Latency:   {profile_results['min_latency_ms']:.2f} ms")
            print(f"   Max Latency:   {profile_results['max_latency_ms']:.2f} ms")
            print(f"   Throughput:    {profile_results['throughput_fps']:.1f} FPS")
        
        # Standard benchmarking
        print(f"🏃 Running {args.iterations} inference iterations...")
        latencies = []
        
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(sample_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Actual benchmarking
            for i in range(args.iterations):
                start_time = time.time()
                _ = model(sample_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                latency = (time.time() - start_time) * 1000  # ms
                latencies.append(latency)
                
                if (i + 1) % 20 == 0:
                    avg_latency = np.mean(latencies[-20:])
                    print(f"   Progress: {i+1:3d}/{args.iterations} | Avg latency: {avg_latency:.2f}ms")
        
        # Results
        latencies = np.array(latencies)
        
        print("\n📊 Benchmark Results:")
        print(f"   Mean latency:    {latencies.mean():.2f} ± {latencies.std():.2f} ms")
        print(f"   Median latency:  {np.median(latencies):.2f} ms")
        print(f"   Min latency:     {latencies.min():.2f} ms")
        print(f"   Max latency:     {latencies.max():.2f} ms")
        print(f"   95th percentile: {np.percentile(latencies, 95):.2f} ms")
        print(f"   99th percentile: {np.percentile(latencies, 99):.2f} ms")
        print(f"   Throughput:      {args.batch_size / (latencies.mean() / 1000):.1f} FPS")
        
        # Model statistics
        stats = model.get_model_statistics()
        print(f"\n🔧 Model Statistics:")
        print(f"   Total parameters:     {stats['total_parameters']:,}")
        print(f"   Trainable parameters: {stats['trainable_parameters']:,}")
        print(f"   Average firing rate:  {stats['average_firing_rate']:.4f}")
        
        print("✅ Benchmarking completed!")
        
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        raise


def demo_command(args, device):
    """Interactive demo command."""
    if not PYTORCH_MODELS_AVAILABLE:
        print("❌ PyTorch models not available. Please install PyTorch to use demo functionality.")
        return
        
    print(f"🎮 Starting interactive demo")
    print(f"⏱️  Duration: {args.duration}s")
    print(f"🧠 Model: {args.model}")
    
    try:
        # Create model
        if args.model == "yolo":
            model = SpikingYOLO.from_pretrained("yolo_v4_spiking_dvs", backend=device.type)
        else:
            model = CustomSNN()
        
        model.to(device)
        model.eval()
        
        # Create camera
        camera = DVSCamera(sensor_type="DVS128")
        preprocessor = SpatioTemporalPreprocessor()
        
        print("\n🚀 Demo started! Press Ctrl+C to stop early.")
        print("📊 Real-time statistics:")
        
        stats = {
            'total_events': 0,
            'total_detections': 0,
            'total_frames': 0,
            'avg_latency': 0.0
        }
        
        start_time = time.time()
        
        try:
            for frame_idx, events in enumerate(camera.stream(duration=args.duration)):
                if len(events) == 0:
                    continue
                
                # Process events
                processed_events = preprocessor.process(events)
                
                # Run detection
                if args.model == "yolo":
                    detections = model.detect(events, threshold=0.3)
                    latency = model.last_inference_time
                else:
                    # Classification demo
                    event_tensor = model.events_to_tensor(events)
                    if device.type == "cuda":
                        event_tensor = event_tensor.cuda()
                    
                    inference_start = time.time()
                    with torch.no_grad():
                        output = model(event_tensor)
                        prediction = torch.softmax(output, dim=1)
                    latency = (time.time() - inference_start) * 1000
                    
                    # Convert to detection format
                    detections = []
                    if prediction.max() > 0.7:
                        detections.append({
                            'class_id': prediction.argmax().item(),
                            'confidence': prediction.max().item(),
                            'class_name': f'class_{prediction.argmax().item()}'
                        })
                
                # Update statistics
                stats['total_events'] += len(events)
                stats['total_detections'] += len(detections)
                stats['total_frames'] += 1
                stats['avg_latency'] = (stats['avg_latency'] * (frame_idx) + latency) / (frame_idx + 1)
                
                # Print live stats every 10 frames
                if frame_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = stats['total_frames'] / elapsed
                    
                    print(f"\r⏰ {elapsed:6.1f}s | "
                          f"📊 {stats['total_events']:6d} events | "
                          f"🎯 {stats['total_detections']:3d} detections | "
                          f"🖼️  {fps:5.1f} FPS | "
                          f"⚡ {stats['avg_latency']:5.1f}ms", end="")
                
                if elapsed >= args.duration:
                    break
                    
        except KeyboardInterrupt:
            print("\n\n⏹️  Demo stopped by user")
        
        # Final statistics
        total_time = time.time() - start_time
        print(f"\n\n📈 Final Demo Statistics:")
        print(f"   Duration:         {total_time:.1f}s")
        print(f"   Total events:     {stats['total_events']:,}")
        print(f"   Total frames:     {stats['total_frames']:,}")
        print(f"   Total detections: {stats['total_detections']:,}")
        print(f"   Average FPS:      {stats['total_frames'] / total_time:.1f}")
        print(f"   Average latency:  {stats['avg_latency']:.1f}ms")
        print(f"   Events per second: {stats['total_events'] / total_time:.0f}")
        
        print("✅ Demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Spike SNN Event Vision Kit CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --model custom --epochs 50 --save-dir ./models
  %(prog)s detect --model-path ./models/best_model.pth --input camera
  %(prog)s benchmark --model-path ./models/final_model.pth --iterations 200
  %(prog)s demo --model yolo --duration 30
        """
    )
    
    # Global arguments
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto", help="Computation device")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a spiking neural network")
    train_parser.add_argument("--model", choices=["yolo", "custom"], default="custom", help="Model type")
    train_parser.add_argument("--dataset", default="synthetic", help="Dataset name")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save model")
    train_parser.add_argument("--config", type=str, help="Training config JSON file")
    train_parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Run object detection")
    detect_parser.add_argument("--model-path", help="Path to trained model (optional for pretrained)")
    detect_parser.add_argument("--input", default="camera", help="Input: 'camera', event file path, or 'demo'")
    detect_parser.add_argument("--output", help="Output file for detections")
    detect_parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    detect_parser.add_argument("--duration", type=float, default=10.0, help="Detection duration (seconds)")
    detect_parser.add_argument("--integration-time", type=float, default=10e-3, help="Integration time (seconds)")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark model performance")
    benchmark_parser.add_argument("--model-path", help="Path to model (optional for pretrained)")
    benchmark_parser.add_argument("--dataset", default="synthetic", help="Dataset for benchmarking")
    benchmark_parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    benchmark_parser.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmarking")
    benchmark_parser.add_argument("--profile", action="store_true", help="Enable detailed profiling")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    demo_parser.add_argument("--model", choices=["yolo", "custom"], default="yolo", help="Model type")
    demo_parser.add_argument("--duration", type=float, default=30.0, help="Demo duration (seconds)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Determine device
    if TORCH_AVAILABLE:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        print(f"Using device: {device}")
    else:
        device = "cpu"  # String fallback when torch not available
        print(f"Using device: {device} (PyTorch not available)")
    
    try:
        if args.command == "train":
            train_command(args, device)
        elif args.command == "detect":
            detect_command(args, device)
        elif args.command == "benchmark":
            benchmark_command(args, device)
        elif args.command == "demo":
            demo_command(args, device)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


# Entry points for console scripts
def train():
    """Training entry point."""
    sys.argv = ["spike-snn-train", "train"] + sys.argv[1:]
    main()

def detect():
    """Detection entry point."""
    sys.argv = ["spike-snn-detect", "detect"] + sys.argv[1:]
    main()

def benchmark():
    """Benchmark entry point."""
    sys.argv = ["spike-snn-benchmark", "benchmark"] + sys.argv[1:]
    main()

def demo():
    """Demo entry point."""
    sys.argv = ["spike-snn-demo", "demo"] + sys.argv[1:]
    main()


if __name__ == "__main__":
    main()
```
