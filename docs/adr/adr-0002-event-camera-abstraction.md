# ADR-0002: Event Camera Hardware Abstraction

## Status
Accepted

## Context

The project needs to support multiple event camera hardware types with different:
- Data formats and streaming protocols
- Resolution and dynamic range capabilities
- SDK requirements and licensing constraints
- Performance characteristics and latency profiles

Supported hardware includes:
- iniVation DVS128/DAVIS346 (libcaer)
- Prophesee sensors (OpenEB SDK)
- Samsung DVS (proprietary SDK)
- Future: CelePixel, Insightness sensors

Design requirements:
- Unified API across different hardware vendors
- Efficient zero-copy data handling
- Real-time streaming with minimal latency
- Hot-swappable sensor support
- Simulation/playback capabilities for development

## Decision

Implement a **Hardware Abstraction Layer (HAL)** with:

1. **Common Interface**: `EventCamera` base class with standardized methods
2. **Vendor-Specific Drivers**: Separate modules for each hardware SDK
3. **Event Format Standardization**: Internal representation as (x, y, t, polarity) tuples
4. **Plugin Architecture**: Dynamic loading of camera drivers at runtime
5. **Simulation Backend**: File-based and synthetic event generation

```python
class EventCamera:
    def stream() -> Iterator[EventBatch]
    def configure(params: CameraConfig) -> None
    def calibrate() -> CalibrationData
    def get_info() -> CameraInfo
```

## Consequences

### Positive
- Simplified application development across different hardware
- Easy testing with simulation backend
- Future-proof for new camera vendors
- Consistent performance monitoring across platforms
- Reduced vendor lock-in

### Negative
- Abstraction overhead may impact performance
- Common denominator approach may limit hardware-specific features
- Additional complexity in driver management
- Testing burden across multiple hardware platforms

### Implementation Strategy
- Start with DVS128 and Prophesee support (most common)
- Implement performance-critical paths with minimal abstraction
- Use factory pattern for driver instantiation
- Provide escape hatches for hardware-specific optimizations
- Comprehensive simulation backend for CI/CD testing