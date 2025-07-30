# Development Guide

This guide provides detailed instructions for setting up and working with the Spike-SNN Event Vision Kit development environment.

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU (recommended for training)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space for datasets and models

### Hardware Support

#### Event Cameras
- DVS128, DVS240, DVS346
- DAVIS240C, DAVIS346
- Prophesee EVK series
- CelePixel CeleX-V

#### Neuromorphic Processors
- Intel Loihi 2 (requires Intel NRC access)
- BrainChip Akida AKD1000
- SpiNNaker boards

## Development Environment Setup

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/spike-snn-event-vision-kit.git
cd spike-snn-event-vision-kit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Development installation with all extras
pip install -e ".[dev,docs,cuda,ros2,monitoring]"

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Installation

```bash
# Run basic tests
make test

# Check code style
make lint

# Build documentation
make docs
```

## Development Workflow

### Code Organization

```
src/spike_snn_event/
├── __init__.py              # Package initialization
├── cameras/                 # Event camera interfaces
│   ├── __init__.py
│   ├── base.py             # Base camera class
│   ├── dvs.py              # DVS camera support
│   └── prophesee.py        # Prophesee camera support
├── models/                 # SNN model implementations  
│   ├── __init__.py
│   ├── base.py             # Base model class
│   ├── spiking_yolo.py     # Spiking YOLO detector
│   └── custom_snn.py       # Custom SNN architectures
├── training/               # Training utilities
│   ├── __init__.py
│   ├── trainer.py          # Training loop
│   └── losses.py           # SNN-specific losses
├── hardware/               # Hardware backend support
│   ├── __init__.py
│   ├── loihi.py           # Intel Loihi backend
│   └── akida.py           # BrainChip Akida backend
├── datasets/               # Dataset loaders
│   ├── __init__.py
│   ├── nmnist.py          # N-MNIST dataset
│   └── ncars.py           # N-CARS dataset
└── utils/                  # Utility functions
    ├── __init__.py
    ├── events.py           # Event processing
    └── visualization.py    # Plotting utilities
```

### Coding Standards

#### Python Style Guide

We follow PEP 8 with these specific guidelines:

```python
# Good: Clear function names and type hints
def process_event_stream(
    events: torch.Tensor,
    time_window: float = 10e-3,
    spatial_resolution: tuple[int, int] = (128, 128)
) -> torch.Tensor:
    """Process event stream into spike trains.
    
    Args:
        events: Event tensor (N, 4) with [x, y, t, p]
        time_window: Integration time window in seconds
        spatial_resolution: Output spatial dimensions
        
    Returns:
        Spike tensor (H, W, T) with binary spikes
    """
    # Implementation here
    pass

# Good: Descriptive variable names
spike_threshold = 1.0
membrane_potential = torch.zeros(batch_size, num_neurons)
refractory_period = 2e-3

# Bad: Unclear abbreviations
th = 1.0  # What threshold?
v = torch.zeros(bs, n)  # Unclear dimensions
rp = 2e-3  # What is rp?
```

#### Docstring Format

Use Google-style docstrings:

```python
class SpikingConv2d(nn.Module):
    """Spiking convolutional layer with temporal dynamics.
    
    This layer implements a convolutional operation followed by
    leaky integrate-and-fire (LIF) neurons with learnable thresholds.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels  
        kernel_size: Size of convolving kernel
        threshold: Spike firing threshold (default: 1.0)
        tau_mem: Membrane time constant in seconds (default: 20e-3)
        tau_syn: Synaptic time constant in seconds (default: 5e-3)
        
    Example:
        >>> layer = SpikingConv2d(64, 128, kernel_size=3)
        >>> spikes = layer(input_spikes)  # (B, C, H, W, T)
        
    Note:
        Input tensors should have shape (batch, channels, height, width, time).
        Time dimension represents the number of simulation time steps.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        threshold: float = 1.0,
        tau_mem: float = 20e-3,
        tau_syn: float = 5e-3
    ):
        # Implementation
        pass
```

### Testing

#### Test Structure

```bash
tests/
├── __init__.py
├── test_basic.py              # Basic package tests
├── cameras/
│   ├── test_dvs.py           # DVS camera tests
│   └── test_prophesee.py     # Prophesee tests
├── models/
│   ├── test_spiking_yolo.py  # YOLO model tests
│   └── test_custom_snn.py    # Custom model tests
├── training/
│   └── test_trainer.py       # Training tests
├── hardware/
│   ├── test_loihi.py         # Loihi tests (require hardware)
│   └── test_akida.py         # Akida tests (require hardware)
└── fixtures/
    ├── sample_events.pt       # Sample event data
    └── test_models.pt         # Small test models
```

#### Writing Tests

```python
import pytest
import torch
from spike_snn_event.models import SpikingYOLO

class TestSpikingYOLO:
    """Test suite for SpikingYOLO model."""
    
    @pytest.fixture
    def sample_events(self):
        """Generate sample event data for testing."""
        return torch.rand(100, 4)  # [x, y, t, p]
    
    @pytest.fixture  
    def model(self):
        """Create test model instance."""
        return SpikingYOLO(
            input_size=(128, 128),
            num_classes=2,
            time_steps=10
        )
    
    def test_model_forward(self, model, sample_events):
        """Test forward pass with sample data."""
        # Convert events to input tensor
        input_tensor = model.preprocess_events(sample_events)
        
        # Forward pass
        output = model(input_tensor)
        
        # Check output shape
        expected_shape = (1, 2, 10)  # (batch, classes, time)
        assert output.shape == expected_shape
        
    @pytest.mark.slow
    def test_model_training(self, model, sample_events):
        """Test training step (marked as slow test)."""
        # Training test implementation
        pass
        
    @pytest.mark.hardware
    def test_loihi_deployment(self, model):
        """Test deployment to Loihi hardware."""
        pytest.importorskip("nxsdk")  # Skip if Loihi SDK not available
        # Hardware deployment test
        pass
```

#### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Skip slow tests
pytest -m "not slow"

# Skip hardware tests  
pytest -m "not hardware"

# Specific test file
pytest tests/models/test_spiking_yolo.py

# Specific test method
pytest tests/models/test_spiking_yolo.py::TestSpikingYOLO::test_model_forward
```

### Documentation

#### Building Documentation

```bash
# Build HTML documentation
make docs

# Serve documentation locally
make docs-serve
# Open http://localhost:8000 in browser

# Clean and rebuild
cd docs && make clean && make html
```

#### Writing Documentation

- Use reStructuredText (.rst) for main documentation
- Use Markdown (.md) for guides and READMEs
- Include docstrings in all public functions/classes
- Add examples for complex functionality
- Keep documentation up-to-date with code changes

### Performance Profiling

#### Memory Profiling

```python
# Use memory_profiler for memory usage analysis
from memory_profiler import profile

@profile
def train_epoch(model, dataloader):
    """Profile memory usage during training."""
    for batch in dataloader:
        # Training step
        loss = model.training_step(batch)
        loss.backward()
```

#### Timing Analysis

```python
import time
import torch

# CUDA timing
torch.cuda.synchronize()
start_time = time.time()

# Your code here
output = model(input_data)

torch.cuda.synchronize()
elapsed_time = time.time() - start_time
print(f"Inference time: {elapsed_time*1000:.2f}ms")
```

#### Profiling Tools

```bash
# Python profiler
python -m cProfile -o profile.stats train.py

# PyTorch profiler
python -c "
import torch.profiler
with torch.profiler.profile() as prof:
    model(sample_input)
prof.export_chrome_trace('trace.json')
"

# Memory usage
mprof run train.py
mprof plot
```

## Debugging

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size or use gradient checkpointing
torch.cuda.empty_cache()

# Monitor GPU memory
print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
```

#### Event Camera Connection
```bash
# Check device permissions
ls -la /dev/event*
sudo usermod -a -G dialout $USER  # Add user to dialout group

# Test camera connection
python -c "
from spike_snn_event.cameras import DVSCamera
camera = DVSCamera(device='/dev/event0')
print('Camera connected successfully')
"
```

#### ROS2 Integration
```bash
# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Check ROS2 topics
ros2 topic list
ros2 topic echo /dvs/events
```

### IDE Configuration

#### VS Code Settings

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.associations": {
        "*.cu": "cuda-cpp"
    }
}
```

#### PyCharm Configuration

1. Set Python interpreter to `./venv/bin/python`
2. Enable pytest as test runner
3. Configure code style to use Black formatter
4. Set up run configurations for common tasks

## Release Process

### Version Management

We use semantic versioning (SemVer):
- **Major** (X.0.0): Breaking changes
- **Minor** (X.Y.0): New features, backward compatible  
- **Patch** (X.Y.Z): Bug fixes

### Release Checklist

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite** including hardware tests
4. **Build and test package** locally
5. **Create release PR** and get approval
6. **Tag release** and push to GitHub
7. **Build and upload** to PyPI
8. **Update documentation** on ReadTheDocs

### Automated Releases

```bash
# Build package
make build

# Upload to PyPI (requires authentication)
make upload

# Create GitHub release
gh release create v1.0.0 --title "Version 1.0.0" --notes "Release notes here"
```

This development guide provides a comprehensive foundation for contributing to the Spike-SNN Event Vision Kit. For specific questions, consult the API documentation or reach out to the development team.