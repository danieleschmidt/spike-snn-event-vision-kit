# Development Guide

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/spike-snn-event-vision-kit.git
cd spike-snn-event-vision-kit
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements-dev.txt
pre-commit install

# 3. Run tests
pytest

# 4. Start developing!
```

## Project Structure

```
spike-snn-event-vision-kit/
├── src/spike_snn_event/          # Main package
│   ├── models/                   # SNN architectures
│   ├── preprocessing/            # Event processing
│   ├── training/                 # Training utilities
│   ├── hardware/                 # Hardware backends
│   └── ros/                      # ROS2 integration
├── tests/                        # Test suite
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
└── examples/                     # Usage examples
```

## Development Workflow

### 1. Environment Setup

```bash
# Create isolated environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"
```

### 2. Code Quality

```bash
# Format code
black .
isort .

# Check style
flake8
mypy

# Run all checks
pre-commit run --all-files
```

### 3. Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov

# Run specific test
pytest tests/test_models.py::test_spiking_yolo
```

## Hardware Development

### Event Camera Testing

```bash
# Mock camera for testing
export USE_MOCK_CAMERA=1
pytest tests/test_cameras.py

# Real hardware (requires camera)
pytest tests/test_integration.py -m hardware
```

### Neuromorphic Hardware

```bash
# Intel Loihi (requires NRC access)
pip install nxsdk
export LOIHI_ENABLED=1

# BrainChip Akida
pip install akida
export AKIDA_ENABLED=1
```

## Performance Profiling

```bash
# Profile inference speed
python scripts/benchmark_models.py

# Memory profiling
python -m memory_profiler examples/basic_detection.py

# CUDA profiling (if available)
python scripts/profile_gpu.py
```

## Documentation

```bash
# Build docs locally
cd docs
make html
# Open docs/_build/html/index.html

# Auto-rebuild on changes
sphinx-autobuild . _build/html
```

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Run full test suite: `pytest`
4. Build package: `python -m build`
5. Test package: `twine check dist/*`
6. Create release PR

## Troubleshooting

### Common Issues

**Import Error**: Ensure you installed in development mode (`pip install -e .`)

**CUDA Issues**: Check PyTorch CUDA compatibility with your driver version

**Camera Permissions**: Add user to video group on Linux (`sudo usermod -a -G video $USER`)

**ROS2 Integration**: Source ROS2 setup before testing (`source /opt/ros/humble/setup.bash`)

### Debug Mode

```bash
# Enable verbose logging
export SPIKE_SNN_DEBUG=1
python your_script.py

# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Resources

- [SNN Tutorial](docs/tutorials/01_snn_basics.md)
- [Event Camera Guide](docs/tutorials/02_event_cameras.md)
- [Hardware Deployment](docs/tutorials/03_hardware.md)
- [API Reference](https://spike-snn-event-vision.readthedocs.io)