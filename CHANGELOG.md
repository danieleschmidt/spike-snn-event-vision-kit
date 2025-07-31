# Changelog

All notable changes to spike-snn-event-vision-kit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and core functionality
- Event camera integration support (DVS128, DAVIS346, Prophesee)
- Spiking neural network implementations
- Hardware acceleration backends (CUDA, Intel Loihi 2, BrainChip Akida)
- ROS2 integration for robotics applications
- Comprehensive testing suite with hardware mocking
- Docker containerization with multi-stage builds
- Monitoring and observability stack (Prometheus, Grafana)
- Advanced SDLC configurations and templates

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- N/A (initial release)

---

## Release Template

When creating releases, use this template:

### Added
- New features and capabilities

### Changed
- Changes to existing functionality

### Deprecated
- Features that will be removed in future versions

### Removed
- Features removed in this version

### Fixed
- Bug fixes

### Security
- Security improvements and vulnerability fixes

---

## Release Guidelines

### Version Numbers
- **Major** (X.0.0): Breaking changes, major new features
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, small improvements

### Release Process
1. Update CHANGELOG.md with new version
2. Commit changes to main branch
3. Create and push git tag: `git tag vX.Y.Z && git push origin vX.Y.Z`
4. GitHub Actions will automatically create release with artifacts
5. Publish to PyPI (manual approval required)

### Breaking Changes
Breaking changes should be clearly documented with:
- **What changed**: Clear description of the change
- **Why it changed**: Rationale for the breaking change
- **Migration guide**: Step-by-step instructions for users

### Hardware Compatibility
Document hardware compatibility changes:
- New hardware support
- Deprecated hardware
- Performance improvements
- Driver requirements

### Dependencies
Track significant dependency changes:
- New required dependencies
- Version requirement changes
- Optional dependency additions