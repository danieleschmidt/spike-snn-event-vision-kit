# Deployment Guide

This directory contains comprehensive deployment documentation and configurations for the Spike-SNN Event Vision Kit.

## Quick Start

### Local Development
```bash
# Start development environment
make docker-up

# Access Jupyter Lab
open http://localhost:8888

# Run tests in container
make docker-test
```

### Production Deployment
```bash
# Build production image
make docker-build-prod

# Run security scan
make docker-security-scan

# Start production services
docker-compose --profile production up -d
```

## Deployment Options

### 🐳 [Docker Deployment](docker.md)
- Multi-stage Dockerfiles for development, production, CPU-only, and ROS2
- Docker Compose orchestration
- Container security and optimization

### ☸️ [Kubernetes Deployment](kubernetes.md)
- Production-ready Kubernetes manifests
- Helm charts for easy deployment
- Auto-scaling and monitoring integration

### ☁️ [Cloud Deployment](cloud.md)
- AWS, Azure, GCP deployment guides
- Serverless functions for inference
- Managed neuromorphic computing services

### 🤖 [Edge Deployment](edge.md)
- NVIDIA Jetson deployment
- Intel NUC with neuromorphic accelerators
- Raspberry Pi CPU-only deployment

### 🔧 [ROS2 Integration](ros2.md)
- ROS2 node deployment
- Launch files and configurations
- Multi-robot coordination

## Security and Compliance

### 🔒 [Security Guide](security.md)
- Container security best practices
- Secrets management
- Network security configuration
- SBOM generation and vulnerability scanning

### 📋 [Compliance](compliance.md)
- GDPR compliance for event data
- Industry-specific requirements
- Audit logging and monitoring

## Monitoring and Observability

### 📊 [Monitoring Setup](monitoring.md)
- Prometheus metrics collection  
- Grafana dashboards
- Alerting configuration
- Performance monitoring

### 🔍 [Observability](observability.md)
- Distributed tracing
- Structured logging
- Application performance monitoring
- Debug and troubleshooting guides

## Hardware-Specific Deployments

### 🧠 [Neuromorphic Hardware](neuromorphic.md)
- Intel Loihi 2 deployment
- BrainChip Akida integration
- Performance optimization guides

### 🎮 [GPU Deployment](gpu.md)
- NVIDIA GPU optimization
- Multi-GPU scaling
- Memory management

## Maintenance and Operations

### 🔄 [CI/CD Pipeline](cicd.md)
- Automated testing and deployment
- GitOps workflows
- Release management

### 🛠️ [Maintenance](maintenance.md)
- Updates and upgrades
- Backup and recovery
- Performance tuning
- Troubleshooting common issues

## Getting Help

- 📖 Check the specific deployment guide for your use case
- 🐛 Report issues on [GitHub Issues](https://github.com/yourusername/spike-snn-event-vision-kit/issues)
- 💬 Join discussions on [GitHub Discussions](https://github.com/yourusername/spike-snn-event-vision-kit/discussions)
- 📚 Read the [API Documentation](https://spike-snn-event-vision.readthedocs.io)