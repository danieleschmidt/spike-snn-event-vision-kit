# API Reference

Complete API reference for the Spike SNN Event Vision Kit.

## Core Module (`spike_snn_event.core`)

### DVSCamera

Dynamic Vision Sensor camera interface for event capture.

```python
class DVSCamera:
    def __init__(self, sensor_type: str = "DVS128", config: Optional[CameraConfig] = None)
```

**Parameters:**
- `sensor_type`: Type of DVS sensor ("DVS128", "DVS240", "DAVIS346", "Prophesee")
- `config`: Camera configuration object

**Methods:**

#### `stream(duration: Optional[float] = None) -> Iterator[np.ndarray]`
Stream events from camera.

**Returns:** Iterator yielding event arrays [N, 4] with columns [x, y, timestamp, polarity]

#### `start_streaming(duration: Optional[float] = None)`
Start asynchronous event streaming.

#### `stop_streaming()`
Stop event streaming.

#### `get_events(timeout: float = 0.1) -> Optional[np.ndarray]`
Get next batch of events from queue.

**Example:**
```python
camera = snn.DVSCamera("DVS128")
for events in camera.stream(duration=5.0):
    print(f"Received {len(events)} events")
```

### CameraConfig

Configuration for event cameras.

```python
@dataclass
class CameraConfig:
    width: int = 128
    height: int = 128
    noise_filter: bool = True
    refractory_period: float = 1e-3
    hot_pixel_threshold: int = 1000
    background_activity_filter: bool = True
```

### EventPreprocessor

Base class for event preprocessing.

```python
class EventPreprocessor:
    def process(self, events: np.ndarray) -> np.ndarray
```

### SpatioTemporalPreprocessor

Advanced spatiotemporal event preprocessing.

```python
class SpatioTemporalPreprocessor(EventPreprocessor):
    def __init__(self, spatial_size: Tuple[int, int] = (256, 256), time_bins: int = 5)
```

**Methods:**

#### `process(events: np.ndarray) -> np.ndarray`
Process events through spatiotemporal pipeline.

#### `get_statistics() -> Dict[str, float]`
Get preprocessing statistics.

### EventVisualizer

Real-time event visualization.

```python
class EventVisualizer:
    def __init__(self, width: int = 640, height: int = 480)
```

**Methods:**

#### `update(events: np.ndarray) -> np.ndarray`
Update visualization with new events.

#### `draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray`
Draw detection bounding boxes on image.

### File Operations

#### `load_events_from_file(filepath: str) -> Tuple[np.ndarray, Optional[Dict]]`
Load events from file with metadata.

**Supported formats:** .npy, .txt, .h5, .dat

#### `save_events_to_file(events: np.ndarray, filepath: str, metadata: Optional[Dict] = None)`
Save events to file with optional metadata.

## Models Module (`spike_snn_event.models`)

### LIFNeuron

Leaky Integrate-and-Fire neuron implementation.

```python
class LIFNeuron(nn.Module):
    def __init__(self, threshold: float = 1.0, tau_mem: float = 20e-3, 
                 tau_syn: float = 5e-3, reset: str = "subtract", dt: float = 1e-3)
```

**Parameters:**
- `threshold`: Firing threshold
- `tau_mem`: Membrane time constant
- `tau_syn`: Synaptic time constant  
- `reset`: Reset mechanism ("subtract" or "zero")
- `dt`: Time step

**Methods:**

#### `forward(input_current: torch.Tensor) -> torch.Tensor`
Forward pass through LIF neuron.

**Input shape:** [batch, features, time]
**Output shape:** [batch, features, time]

### SpikingConv2d

Spiking convolutional layer.

```python
class SpikingConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, **lif_kwargs)
```

### EventSNN

Base class for event-based spiking neural networks.

```python
class EventSNN(nn.Module):
    def __init__(self, input_size: Tuple[int, int] = (128, 128), 
                 config: Optional[TrainingConfig] = None)
```

**Methods:**

#### `events_to_tensor(events: np.ndarray, time_window: float = 10e-3) -> torch.Tensor`
Convert event array to tensor representation.

#### `compute_loss(outputs: torch.Tensor, targets: torch.Tensor, loss_type: str = "cross_entropy") -> torch.Tensor`
Compute training loss with SNN-specific regularization.

#### `get_model_statistics() -> Dict[str, float]`
Get model statistics for monitoring.

### SpikingYOLO

Spiking YOLO for event-based object detection.

```python
class SpikingYOLO(EventSNN):
    def __init__(self, input_size: Tuple[int, int] = (128, 128), 
                 num_classes: int = 80, time_steps: int = 10)
```

**Methods:**

#### `detect(events: np.ndarray, integration_time: float = 10e-3, threshold: float = 0.5) -> List[Dict]`
Detect objects in event stream.

**Returns:** List of detection dictionaries with keys: bbox, confidence, class_id, class_name

#### `from_pretrained(model_name: str, backend: str = "cpu", **kwargs) -> "SpikingYOLO"`
Load pre-trained model.

### CustomSNN

Customizable spiking neural network.

```python
class CustomSNN(EventSNN):
    def __init__(self, input_size: Tuple[int, int] = (128, 128),
                 hidden_channels: List[int] = [64, 128, 256],
                 output_classes: int = 2, neuron_type: str = "LIF")
```

**Methods:**

#### `export_onnx(filepath: str)`
Export model to ONNX format.

#### `export_loihi(filepath: str)`
Export model for Intel Loihi deployment.

#### `save_checkpoint(filepath: str, epoch: int, loss: float)`
Save training checkpoint.

#### `load_checkpoint(filepath: str)`
Load training checkpoint.

#### `profile_inference(sample_input: torch.Tensor) -> Dict[str, float]`
Profile model inference performance.

### TrainingConfig

Configuration for SNN training.

```python
@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 10
    gradient_clip_value: float = 1.0
    loss_function: str = "cross_entropy"
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"
    surrogate_gradient: str = "fast_sigmoid"
```

### SurrogateGradient

Collection of surrogate gradient functions.

**Static Methods:**
- `fast_sigmoid(input_tensor: torch.Tensor, alpha: float = 10.0) -> torch.Tensor`
- `straight_through_estimator(input_tensor: torch.Tensor) -> torch.Tensor`
- `triangle(input_tensor: torch.Tensor, alpha: float = 1.0) -> torch.Tensor`
- `arctan(input_tensor: torch.Tensor, alpha: float = 2.0) -> torch.Tensor`

## Training Module (`spike_snn_event.training`)

### SpikingTrainer

Advanced trainer for spiking neural networks.

```python
class SpikingTrainer:
    def __init__(self, model: EventSNN, config: Optional[TrainingConfig] = None,
                 device: Optional[torch.device] = None)
```

**Methods:**

#### `fit(train_loader: DataLoader, val_loader: Optional[DataLoader] = None, save_dir: Optional[str] = None) -> Dict[str, List[float]]`
Full training loop with early stopping and checkpointing.

**Returns:** Training history dictionary

#### `train_epoch(train_loader: DataLoader, epoch: int) -> Dict[str, float]`
Train for one epoch.

#### `validate_epoch(val_loader: DataLoader, epoch: int) -> Dict[str, float]`
Validate for one epoch.

#### `evaluate(test_loader: DataLoader, metrics: Optional[List[str]] = None) -> Dict[str, float]`
Evaluate model on test set.

#### `plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None)`
Plot training history.

### EventDataLoader

Specialized data loader for event datasets.

**Static Methods:**

#### `create_loaders(dataset_name: str, batch_size: int = 32, split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Tuple[DataLoader, DataLoader, DataLoader]`
Create train/val/test data loaders.

#### `collate_events(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]`
Custom collate function for variable-length event sequences.

## Advanced Features

### Optimization Module (`spike_snn_event.optimization`)

#### LRUCache
Least Recently Used cache implementation.

```python
class LRUCache:
    def __init__(self, max_size: int = 100)
    def get(self, key: str) -> Optional[Any]
    def put(self, key: str, value: Any) -> None
```

#### ModelCache
Intelligent caching for neural network models.

```python
class ModelCache:
    def __init__(self, max_models: int = 5, memory_limit_gb: float = 8.0)
    def get_or_create(self, model_key: str, create_fn: Callable) -> nn.Module
    def preload_model(self, model_key: str, model: nn.Module) -> None
```

#### MemoryOptimizer
Memory optimization utilities.

```python
class MemoryOptimizer:
    def optimize_model(self, model: nn.Module) -> Dict[str, Any]
    def enable_gradient_checkpointing(self, model: nn.Module) -> None
    def optimize_dataloader(self, dataloader: DataLoader) -> DataLoader
```

### Concurrency Module (`spike_snn_event.concurrency`)

#### ConcurrentProcessor
High-level concurrent processor.

```python
class ConcurrentProcessor:
    def __init__(self, max_threads: int = 8, max_processes: int = 4, enable_gpu: bool = True)
    def submit_task(self, task_id: str, func: Callable, *args, **kwargs) -> str
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult
    def wait_for_completion(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, TaskResult]
```

#### ModelPool
Pool for managing neural network models.

```python
class ModelPool:
    def __init__(self, model_factory: Callable, pool_size: int = 4)
    def acquire(self, timeout: Optional[float] = None) -> Any
    def release(self, model: Any) -> None
    def get_model(self, timeout: Optional[float] = None)  # Context manager
```

#### AsyncProcessor
Asynchronous processor using asyncio.

```python
class AsyncProcessor:
    def __init__(self, max_concurrent: int = 100)
    async def submit_async_task(self, task_id: str, coro: Callable, *args, **kwargs) -> str
    async def get_result_async(self, task_id: str) -> Any
    async def wait_for_all(self, task_ids: List[str]) -> Dict[str, Any]
```

### Scaling Module (`spike_snn_event.scaling`)

#### AutoScaler
Intelligent auto-scaling system.

```python
class AutoScaler:
    def __init__(self, policy: Optional[ScalingPolicy] = None)
    def start(self) -> None
    def stop(self) -> None
    def get_scaling_status(self) -> Dict[str, Any]
```

#### LoadBalancer
Load balancer for distributing requests.

```python
class LoadBalancer:
    def __init__(self, config: Optional[LoadBalancerConfig] = None)
    def add_instance(self, instance_id: str, endpoint: str, weight: float = 1.0) -> None
    def remove_instance(self, instance_id: str) -> None
    def get_next_instance(self, request_context: Optional[Dict] = None) -> Optional[str]
```

#### ScalingOrchestrator
High-level orchestrator for scaling and load balancing.

```python
class ScalingOrchestrator:
    def __init__(self, scaling_policy: Optional[ScalingPolicy] = None, 
                 lb_config: Optional[LoadBalancerConfig] = None)
    def start(self) -> None
    def stop(self) -> None
    def get_status(self) -> Dict[str, Any]
```

## Utility Functions

### Global Instances

```python
# Get global instances
processor = snn.get_concurrent_processor()
optimizer = snn.get_optimizer()
auto_scaler = snn.get_auto_scaler()
load_balancer = snn.get_load_balancer()
```

### Parallel Processing

```python
# Parallel map function
results = snn.parallel_map(
    func=process_function,
    iterable=data_list,
    max_workers=8,
    execution_mode="thread"
)

# Parallel batch processing
results = snn.parallel_batch_process(
    func=process_function,
    data_list=large_dataset,
    batch_size=32,
    max_workers=4
)
```

### Configuration Helpers

```python
# Create training configuration
config = snn.create_training_config(
    learning_rate=1e-3,
    epochs=100,
    batch_size=32
)

# Create scaling policy
policy = snn.scaling.create_scaling_policy(
    min_instances=2,
    max_instances=10,
    cpu_threshold=70.0
)
```

## Error Handling

### Custom Exceptions

- `SpikeNNError`: Base exception class
- `ValidationError`: Input validation errors
- `ModelError`: Model-related errors
- `TrainingError`: Training-related errors
- `InferenceError`: Inference-related errors

### Example Error Handling

```python
try:
    model = snn.SpikingYOLO()
    detections = model.detect(events)
except snn.ValidationError as e:
    print(f"Validation error: {e}")
except snn.InferenceError as e:
    print(f"Inference error: {e}")
except snn.SpikeNNError as e:
    print(f"SpikeNN error: {e}")
```

## Type Hints

The library provides comprehensive type hints for better IDE support:

```python
from typing import List, Dict, Tuple, Optional, Union, Iterator, Callable, Any
import numpy as np
import torch

# Event array type
EventArray = np.ndarray  # Shape: [N, 4] with columns [x, y, timestamp, polarity]

# Detection type
Detection = Dict[str, Union[List[float], float, int, str]]

# Tensor type
EventTensor = torch.Tensor  # Shape: [batch, channels, height, width, time]
```

## Constants and Enums

```python
# Task priorities
class TaskPriority:
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

# Sensor types
SUPPORTED_SENSORS = ["DVS128", "DVS240", "DAVIS346", "Prophesee"]

# File formats
SUPPORTED_FORMATS = [".npy", ".txt", ".h5", ".dat"]
```

For more detailed examples and usage patterns, see the [examples directory](../examples/) and [Quick Start Guide](QUICK_START.md).