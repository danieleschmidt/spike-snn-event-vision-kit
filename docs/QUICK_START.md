# Quick Start Guide

Get up and running with the Spike SNN Event Vision Kit in minutes.

## Installation

### Basic Installation
```bash
pip install spike-snn-event
```

### Development Installation
```bash
git clone https://github.com/danieleschmidt/spike-snn-event-vision-kit.git
cd spike-snn-event-vision-kit
pip install -e ".[dev,cuda,monitoring]"
```

### Docker Installation
```bash
# Pull the latest image
docker pull spike-snn-event:latest

# Run with GPU support
docker run --gpus all -it spike-snn-event:latest
```

## Basic Usage

### 1. Event Camera Setup

```python
import spike_snn_event as snn

# Create and configure DVS camera
camera = snn.DVSCamera(sensor_type="DVS128")
config = snn.CameraConfig(
    width=128,
    height=128,
    noise_filter=True,
    refractory_period=1e-3
)
camera.config = config

# Start streaming events
for events in camera.stream(duration=5.0):
    print(f"Received {len(events)} events")
```

### 2. Event Processing

```python
# Create preprocessor
preprocessor = snn.SpatioTemporalPreprocessor(
    spatial_size=(64, 64),
    time_bins=10
)

# Process events
processed = preprocessor.process(events)
print(f"Processed events shape: {processed.shape}")
```

### 3. SNN Inference

```python
# Create and configure SNN model
model = snn.SpikingYOLO(
    input_size=(128, 128),
    num_classes=10
)

# Run detection
detections = model.detect(
    events,
    integration_time=10e-3,
    threshold=0.5
)

print(f"Found {len(detections)} objects")
```

### 4. Training

```python
# Create training configuration
config = snn.create_training_config(
    learning_rate=1e-3,
    epochs=50,
    batch_size=32
)

# Initialize trainer
trainer = snn.SpikingTrainer(model, config)

# Load dataset and train
dataset = snn.EventDataset.load("N-CARS")
train_loader, val_loader = dataset.get_loaders(batch_size=32)
history = trainer.fit(train_loader, val_loader)
```

## Command Line Interface

### Training
```bash
# Train a model
python -m spike_snn_event.cli train \
    --dataset N-CARS \
    --model SpikingYOLO \
    --epochs 100 \
    --batch-size 32

# Resume training from checkpoint
python -m spike_snn_event.cli train \
    --resume checkpoints/model_epoch_50.pth
```

### Inference
```bash
# Run inference on video file
python -m spike_snn_event.cli detect \
    --input video.aedat4 \
    --model models/spiking_yolo.pth \
    --output detections.json

# Real-time detection from camera
python -m spike_snn_event.cli detect \
    --camera DVS128 \
    --model models/spiking_yolo.pth \
    --display
```

### Benchmarking
```bash
# Benchmark model performance
python -m spike_snn_event.cli benchmark \
    --model models/spiking_yolo.pth \
    --dataset test_data/ \
    --metrics accuracy latency throughput
```

## ROS2 Integration

### Setup ROS2 Environment
```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Build workspace
colcon build --packages-select spike_snn_event

# Source workspace
source install/setup.bash
```

### Launch ROS2 Nodes
```bash
# Start event camera node
ros2 run spike_snn_event event_camera_node

# Start SNN detection node
ros2 run spike_snn_event snn_detection_node

# Start visualization node
ros2 run spike_snn_event event_visualization_node
```

## Docker Deployment

### Local Development
```bash
# Build development image
docker build --target development -t spike-snn-dev .

# Run with volume mounting
docker run -it --gpus all \
    -v $(pwd):/workspace \
    -p 8888:8888 \
    spike-snn-dev
```

### Production Deployment
```bash
# Build production image
docker build --target production -t spike-snn-prod .

# Run production service
docker run -d --gpus all \
    -p 8000:8000 \
    -v /data/models:/app/models \
    spike-snn-prod
```

### Kubernetes Deployment
```bash
# Deploy using Helm
helm install spike-snn-event ./deploy/helm/spike-snn-event \
    --namespace spike-snn \
    --create-namespace

# Check deployment status
kubectl get pods -n spike-snn
```

## Configuration Options

### Model Configuration
```python
# Custom SNN configuration
model_config = {
    'input_size': (128, 128),
    'hidden_channels': [64, 128, 256],
    'output_classes': 10,
    'neuron_type': 'LIF',
    'threshold': 1.0,
    'tau_mem': 20e-3,
    'tau_syn': 5e-3
}

model = snn.CustomSNN(**model_config)
```

### Training Configuration
```python
# Advanced training settings
training_config = snn.TrainingConfig(
    learning_rate=1e-3,
    epochs=100,
    batch_size=32,
    early_stopping_patience=10,
    gradient_clip_value=1.0,
    loss_function="spike_count",
    optimizer="adamw",
    lr_scheduler="cosine",
    surrogate_gradient="fast_sigmoid"
)
```

### Scaling Configuration
```python
# Auto-scaling setup
scaling_policy = snn.scaling.ScalingPolicy(
    min_instances=2,
    max_instances=20,
    cpu_scale_up_threshold=70.0,
    memory_scale_up_threshold=80.0,
    scale_up_cooldown=60.0
)

auto_scaler = snn.AutoScaler(policy=scaling_policy)
auto_scaler.start()
```

## Performance Optimization

### GPU Acceleration
```python
# Enable GPU acceleration
model.set_backend("cuda")

# Use GPU optimizer
optimizer = snn.get_optimizer()
optimizer.enable_gpu_acceleration()
```

### Concurrent Processing
```python
# Use concurrent processor for high throughput
processor = snn.get_concurrent_processor()

# Submit parallel tasks
task_ids = []
for batch in event_batches:
    task_id = processor.submit_task(
        f"process_{len(task_ids)}",
        process_function,
        batch,
        execution_mode="thread"
    )
    task_ids.append(task_id)

# Get results
results = processor.wait_for_completion(task_ids)
```

### Memory Optimization
```python
# Enable memory optimization
memory_optimizer = snn.MemoryOptimizer()
memory_optimizer.optimize_model(model)

# Use model caching
cache = snn.ModelCache(max_size=5)
cached_model = cache.get_or_create("yolo_v1", create_model_fn)
```

## Monitoring and Debugging

### Enable Logging
```python
import logging
logging.basicConfig(level=logging.INFO)

# Enable metrics collection
metrics = snn.get_metrics_collector()
metrics.start_collection()
```

### Health Checks
```python
# System health monitoring
health_checker = snn.HealthChecker()
status = health_checker.check_system_health()
print(f"System status: {status}")
```

### Performance Profiling
```python
# Profile model inference
profile_results = model.profile_inference(sample_input)
print(f"Mean latency: {profile_results['mean_latency_ms']:.2f}ms")
print(f"Throughput: {profile_results['throughput_fps']:.1f} FPS")
```

## Next Steps

1. **Explore Examples**: Check out `/examples/` for comprehensive usage examples
2. **Read Documentation**: Visit `/docs/` for detailed guides and API reference
3. **Join Community**: Contribute to the project on GitHub
4. **Production Deployment**: Use Terraform and Kubernetes configs in `/deploy/`

## Common Issues

### CUDA Not Available
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
```python
# Reduce batch size
config.batch_size = 16

# Enable gradient checkpointing
model.enable_gradient_checkpointing()

# Use memory optimizer
optimizer = snn.MemoryOptimizer()
optimizer.optimize_model(model)
```

### ROS2 Integration Issues
```bash
# Check ROS2 installation
ros2 --version

# Source ROS2 setup
source /opt/ros/humble/setup.bash

# Install ROS2 dependencies
sudo apt install ros-humble-sensor-msgs ros-humble-cv-bridge
```

For more detailed information, see the [full documentation](docs/index.rst).