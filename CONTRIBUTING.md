# Contributing to Spike-SNN Event Vision Kit

Thank you for your interest in contributing to the Spike-SNN Event Vision Kit! This document provides guidelines for contributing to this neuromorphic vision processing toolkit.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Git

### Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/spike-snn-event-vision-kit.git
cd spike-snn-event-vision-kit
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev,docs]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Development Workflow

### Code Style

We follow PEP 8 and use automated tools for code formatting:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking

Run these tools before committing:
```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Testing

We use pytest for testing. Run tests with:
```bash
# All tests
pytest

# With coverage
pytest --cov=spike_snn_event

# Skip slow tests
pytest -m "not slow"

# Skip hardware-dependent tests
pytest -m "not hardware"
```

### Documentation

Documentation is built with Sphinx. To build locally:
```bash
cd docs/
make html
```

## Contribution Guidelines

### Areas of Contribution

We welcome contributions in these areas:

1. **Event Camera Support**: Additional sensor integrations
2. **SNN Architectures**: New spiking neural network models
3. **Hardware Backends**: Neuromorphic chip support
4. **Datasets**: Neuromorphic vision dataset loaders
5. **Documentation**: Tutorials and examples
6. **Testing**: Unit and integration tests

### Bug Reports

When reporting bugs, please include:

- Python and package versions
- Hardware specifications (GPU, event camera)
- Minimal code example
- Full error traceback
- Expected vs. actual behavior

Use our bug report template on GitHub Issues.

### Feature Requests

For new features:

- Check existing issues and discussions
- Describe the use case and benefit
- Consider implementation complexity
- Provide code examples if possible

### Pull Requests

1. **Create a feature branch**: `git checkout -b feature/your-feature`
2. **Make changes**: Follow code style and add tests
3. **Update documentation**: Add docstrings and examples
4. **Run tests**: Ensure all tests pass
5. **Commit changes**: Use clear, descriptive commit messages
6. **Push and create PR**: Include description and link issues

#### PR Requirements

- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] All CI checks pass
- [ ] Backward compatibility maintained (unless breaking change)

### Code Review Process

1. Maintainers review PRs within 1-2 weeks
2. Address feedback and discussions
3. Maintainer approval required for merge
4. Squash merge preferred for clean history

## Specialized Contributions

### Adding Event Camera Support

To add support for a new event camera:

1. Implement camera interface in `src/spike_snn_event/cameras/`
2. Add device-specific configuration
3. Include calibration and noise filtering
4. Add tests with mock data
5. Update documentation

### Implementing SNN Models

For new spiking neural network architectures:

1. Inherit from base SNN classes
2. Implement forward pass with temporal dynamics
3. Add surrogate gradient functions
4. Include training and inference modes
5. Provide pre-trained weights if available

### Hardware Backend Integration

To add neuromorphic hardware support:

1. Create backend in `src/spike_snn_event/hardware/`
2. Implement model compilation and deployment
3. Add profiling and benchmarking tools
4. Include hardware-specific optimizations
5. Document setup and requirements

## Community

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Discord**: Real-time community chat (coming soon)

### Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behavior to the maintainers.

### Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Documentation acknowledgments
- Conference presentations

## Resources

### Learning Resources

- [Neuromorphic Vision Tutorial](docs/tutorials/neuromorphic_vision.md)
- [SNN Basics](docs/tutorials/snn_basics.md)
- [Event Camera Programming](docs/tutorials/event_cameras.md)

### External Resources

- [Neuromorphic Engineering Handbook](https://link.springer.com/book/10.1007/978-3-642-34487-9)
- [Event-based Vision Survey](https://arxiv.org/abs/1904.07368)
- [SNN Training Methods](https://arxiv.org/abs/1906.08165)

## Release Process

Releases follow semantic versioning (SemVer):

- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes

### Release Schedule

- Minor releases: Every 2-3 months
- Patch releases: As needed for critical bugs
- Major releases: Annual or for significant architecture changes

Thank you for contributing to advancing neuromorphic vision processing!