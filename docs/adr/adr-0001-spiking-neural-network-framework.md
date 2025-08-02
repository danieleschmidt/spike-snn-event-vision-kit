# ADR-0001: Spiking Neural Network Framework Selection

## Status
Accepted

## Context

The project requires a robust framework for implementing spiking neural networks (SNNs) that can:
- Support multiple neuron models (LIF, ALIF, Izhikevich)
- Enable gradient-based training through surrogate gradients
- Interface with neuromorphic hardware (Loihi, Akida)
- Integrate with PyTorch ecosystem for seamless development
- Provide efficient CUDA implementations for GPU acceleration

Available options evaluated:
1. **SNNTorch**: PyTorch-based, active development, good documentation
2. **Norse**: Clean API, research-focused, limited hardware support
3. **SpyTorch**: Older, less maintained, limited features
4. **Custom Implementation**: Full control, significant development overhead

## Decision

We will use **SNNTorch** as the primary framework with custom extensions for:
- Event camera preprocessing pipelines
- Hardware-specific optimizations
- Neuromorphic backend interfaces

SNNTorch provides:
- Native PyTorch integration with automatic differentiation
- Multiple neuron models with customizable dynamics
- Surrogate gradient functions for backpropagation
- Active community and regular updates
- Comprehensive tutorials and documentation

## Consequences

### Positive
- Rapid development with proven framework
- Access to PyTorch ecosystem (optimizers, schedulers, distributed training)
- Built-in visualization and analysis tools
- Strong community support and examples
- Hardware acceleration through PyTorch CUDA backend

### Negative
- Dependency on external framework evolution
- May require custom patches for specialized neuromorphic features
- Learning curve for team members unfamiliar with SNNTorch
- Potential vendor lock-in to PyTorch ecosystem

### Mitigation Strategies
- Implement hardware abstraction layer to reduce SNNTorch coupling
- Contribute improvements back to SNNTorch community
- Maintain compatibility layers for potential future framework changes
- Document custom extensions thoroughly for maintainability