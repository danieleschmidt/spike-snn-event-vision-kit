# Contributing to Spike-SNN Event Vision Kit

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/spike-snn-event-vision-kit.git
   cd spike-snn-event-vision-kit
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards (Black, isort, flake8)
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and checks**
   ```bash
   # Run tests
   pytest
   
   # Check code formatting
   black --check .
   isort --check-only .
   flake8
   mypy
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

- **Python Style**: Follow PEP 8, enforced by Black (line length: 88)
- **Import Sorting**: Use isort with Black profile
- **Type Hints**: Required for all public functions
- **Docstrings**: Use Google-style docstrings
- **Testing**: Aim for >90% code coverage

## Priority Areas

- Additional event camera support (DAVIS, Prophesee, iniVation)
- New SNN architectures (ResNet, Vision Transformer variants)
- Neuromorphic hardware backends (Intel Loihi 2, BrainChip Akida)
- ROS2 packages and launch files
- Performance optimization and benchmarking

## Testing Guidelines

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test function names
- Include unit, integration, and end-to-end tests
- Mock external dependencies (cameras, hardware)

## Documentation

- Update README.md for significant changes
- Add docstrings for all public APIs
- Include examples in docstrings
- Update tutorials for new features

## Questions?

- Open an issue for bug reports or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.