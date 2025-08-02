# Architecture Documentation

## System Overview

The Spike-SNN Event Vision Kit is designed as a modular, production-ready framework for neuromorphic vision processing. The architecture enables real-time, ultra-low-power object detection using spiking neural networks on event-based camera data.

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Event Camera  │    │   Preprocessing │    │  Spiking Neural │
│   (DVS/DAVIS)   │───▶│     Pipeline    │───▶│    Network      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐              │
│   Visualization │◀───│   Post-process  │◀─────────────┘
│   & ROS2 Output │    │   & Detection   │
└─────────────────┘    └─────────────────┘
```

## Core Components

### 1. Event Processing Pipeline

**Location**: `src/spike_snn_event/core.py`

- **Event Capture**: Interface with DVS128, DAVIS346, Prophesee sensors
- **Noise Filtering**: Hot pixel removal, background activity filtering
- **Spatial Preprocessing**: Resolution adaptation, region of interest extraction
- **Temporal Binning**: Event accumulation windows, frame reconstruction

```python
class EventProcessor:
    - spatial_filter()    # Remove noise, outliers
    - temporal_bin()      # Create time windows
    - normalize()         # Standardize input ranges
    - to_spike_train()    # Convert to neural encoding
```

### 2. Spiking Neural Network Models

**Location**: `src/spike_snn_event/models.py`

- **LIF Neurons**: Leaky Integrate-and-Fire dynamics
- **Surrogate Gradients**: Differentiable spike functions for backprop
- **Spiking Convolutions**: Temporal convolution with membrane dynamics
- **Hardware Mapping**: Translation to neuromorphic chips

```python
class SpikingYOLO:
    - backbone: SpikingResNet
    - neck: SpikingFPN
    - head: SpikingDetectionHead
    - temporal_integration: membrane_potential_accumulation
```

### 3. Hardware Backends

**Supported Platforms**:
- **CUDA GPU**: High-throughput training and inference
- **Intel Loihi 2**: Ultra-low power neuromorphic processing
- **BrainChip Akida**: Edge AI acceleration
- **CPU**: Development and testing fallback

### 4. Training Infrastructure

**Components**:
- **Surrogate Gradient Learning**: Enables backpropagation through spikes
- **Temporal BPTT**: Backpropagation through time for sequence learning
- **Event-based Augmentation**: Spatial jitter, temporal shift, polarity flip
- **Multi-objective Optimization**: Accuracy vs. energy vs. latency trade-offs

## Data Flow

### Input Processing
1. **Event Stream**: Raw (x, y, t, polarity) tuples from camera
2. **Preprocessing**: Noise filtering and spatial/temporal binning
3. **Spike Encoding**: Rate coding or temporal coding conversion
4. **Batch Formation**: Mini-batch creation for efficient processing

### Neural Processing
1. **Membrane Integration**: Leaky integration of input currents
2. **Spike Generation**: Threshold-based spike emission
3. **Lateral Connections**: Recurrent processing for temporal context
4. **Hierarchical Feature Extraction**: Multi-scale spike pattern recognition

### Output Generation
1. **Spike Rate Decoding**: Convert output spikes to detection scores
2. **Non-Maximum Suppression**: Remove duplicate detections
3. **Coordinate Mapping**: Transform to original image coordinates
4. **Confidence Thresholding**: Filter low-confidence predictions

## Performance Characteristics

### Latency Analysis
- **Event Capture**: <1ms (hardware dependent)
- **Preprocessing**: 0.1-0.5ms (optimized C++/CUDA)
- **SNN Inference**: 0.2-2ms (network size dependent)
- **Post-processing**: 0.1-0.3ms
- **Total Pipeline**: 0.4-3.8ms (vs. 20-50ms for frame-based)

### Memory Footprint
- **Model Weights**: 1-50MB (sparse connectivity)
- **Activation Memory**: Minimal (sparse spikes)
- **Event Buffer**: 1-10MB (configurable window)
- **Total Runtime**: 10-100MB (vs. 500MB-2GB for CNNs)

### Power Consumption
- **GPU Implementation**: 5-15W
- **Loihi Implementation**: 0.1-1W
- **Akida Implementation**: 0.05-0.5W
- **Energy per Inference**: 0.1-10mJ (vs. 100-1000mJ for CNNs)

## Scalability Considerations

### Horizontal Scaling
- **Multi-Camera**: Synchronized processing of multiple event streams
- **Distributed Processing**: Camera-edge-cloud architecture
- **Load Balancing**: Dynamic workload distribution

### Vertical Scaling
- **Model Complexity**: From simple classifiers to complex detection networks
- **Resolution Scaling**: Adaptive processing based on scene complexity
- **Temporal Windows**: Variable integration times based on scene dynamics

## Integration Points

### ROS2 Integration
- **Event Topics**: Standardized message types for event streams
- **Detection Topics**: Bounding box and classification outputs
- **Parameter Services**: Dynamic reconfiguration of processing pipeline
- **Lifecycle Management**: Proper node startup/shutdown sequencing

### External APIs
- **Model Zoo**: Pre-trained model downloading and caching
- **Hardware Abstraction**: Unified interface across different backends
- **Monitoring**: Metrics collection and performance monitoring
- **Configuration**: YAML-based parameter management

## Security Considerations

### Data Protection
- **Event Stream Encryption**: Secure transmission of camera data
- **Model Protection**: Encrypted weight storage and transfer
- **Access Control**: Authentication for hardware resources

### Privacy
- **On-Device Processing**: Minimize data transmission
- **Differential Privacy**: Noise injection for sensitive applications
- **Data Minimization**: Process only necessary event information

## Future Architecture Evolution

### Planned Enhancements
- **Multi-Modal Fusion**: Integration with RGB, thermal, radar
- **Adaptive Networks**: Dynamic model selection based on scene
- **Federated Learning**: Distributed model training across devices
- **Quantum-Enhanced**: Hybrid classical-quantum processing

### Research Directions
- **Neuromorphic Memory**: Integration with memristive devices
- **Bio-Inspired Plasticity**: Online learning and adaptation
- **3D Event Processing**: Temporal-spatial-depth integration
- **Edge-Cloud Continuum**: Seamless processing across deployment tiers