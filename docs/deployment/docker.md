# Docker Deployment Guide

This guide covers deploying Spike-SNN Event Vision Kit using Docker containers.

## Container Images

The project provides multiple Docker targets optimized for different use cases:

### Available Images

| Target | Use Case | Size | GPU Support | Description |
|--------|----------|------|-------------|-------------|
| `development` | Local development | ~3GB | ✅ CUDA | Full development environment with Jupyter |
| `production` | Production inference | ~2GB | ✅ CUDA | Optimized for deployment |
| `cpu-only` | CPU-only environments | ~1.5GB | ❌ CPU only | Lightweight for edge devices |
| `ros2` | Robotics integration | ~2.5GB | ✅ CUDA | ROS2 Humble integration |

## Quick Start

### Development Environment

```bash
# Start development container with Jupyter Lab
docker-compose up -d spike-snn-dev

# Access Jupyter Lab
open http://localhost:8888

# Open shell in container
docker-compose exec spike-snn-dev bash

# Run tests
docker-compose run --rm spike-snn-dev pytest tests/
```

### Production Deployment

```bash
# Build production image
docker build --target production -t spike-snn-event-vision:latest .

# Run inference service
docker run -d \
  --name snn-inference \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/home/snnuser/models:ro \
  spike-snn-event-vision:latest
```

## Docker Compose Profiles

Use profiles to run different service combinations:

```bash
# Development (default)
docker-compose up -d

# Production services
docker-compose --profile production up -d

# ROS2 integration
docker-compose --profile ros2 up -d

# CPU-only deployment
docker-compose --profile cpu-only up -d

# Monitoring stack
docker-compose --profile monitoring up -d

# Simulation environment
docker-compose --profile simulation up -d
```

## Service Configuration

### Environment Variables

Configure services using environment variables:

```bash
# GPU configuration
CUDA_VISIBLE_DEVICES=0,1  # Use specific GPUs
NVIDIA_VISIBLE_DEVICES=all  # Use all GPUs

# Model configuration
MODEL_PATH=/home/snnuser/models
DEFAULT_MODEL=spiking_yolo
MODEL_QUANTIZATION=fp16

# Performance tuning
BATCH_SIZE=8
TIME_STEPS=10
INFERENCE_THREADS=4

# Logging
LOG_LEVEL=INFO
ENABLE_PROFILING=false

# ROS2 configuration
ROS_DOMAIN_ID=42
RMW_IMPLEMENTATION=rmw_cyclonedx_cpp
```

### Volume Mounts

The containers use several volumes for persistent data:

```yaml
volumes:
  spike-snn-cache:      # Package cache and temporary files
  spike-snn-models:     # Pre-trained models and checkpoints
  spike-snn-data:       # Datasets and input data
  spike-snn-logs:       # Application logs and metrics
  spike-snn-wandb:      # Weights & Biases local data
```

Mount local directories for development:

```bash
docker run -v $(pwd):/workspace spike-snn-event-vision:dev
```

## Building Images

### Single Platform Build

```bash
# Development image
docker build --target development -t spike-snn-event-vision:dev .

# Production image
docker build --target production -t spike-snn-event-vision:latest .

# CPU-only image
docker build --target cpu-only -t spike-snn-event-vision:cpu .

# ROS2 image
docker build --target ros2 -t spike-snn-event-vision:ros2 .
```

### Multi-Platform Build

For deployment across different architectures:

```bash
# Setup buildx (one-time)
docker buildx create --name multibuilder --use
docker buildx inspect --bootstrap

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --target production \
  -t spike-snn-event-vision:latest \
  --push .
```

## Production Deployment

### Basic Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  snn-inference:
    image: spike-snn-event-vision:latest
    deploy:
      replicas: 3
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          memory: 4G
          cpus: '2'
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models
      - LOG_LEVEL=INFO
    volumes:
      - models:/models:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - snn-inference
    restart: unless-stopped

volumes:
  models:
    driver: local
```

Deploy with:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Load Balancing with NGINX

```nginx
# nginx.conf
upstream snn_backend {
    least_conn;
    server snn-inference-1:8000 max_fails=3 fail_timeout=30s;
    server snn-inference-2:8000 max_fails=3 fail_timeout=30s;
    server snn-inference-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name spike-snn.example.com;
    
    location / {
        proxy_pass http://snn_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout configuration for inference requests
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    location /health {
        access_log off;
        proxy_pass http://snn_backend/health;
    }
}
```

## Security Configuration

### Container Security

```dockerfile
# Use non-root user
USER snnuser

# Read-only root filesystem (where possible)
# Note: Some ML libraries require write access
COPY --chown=snnuser:snnuser . /app

# Drop unnecessary capabilities
# Use seccomp profiles in docker-compose
```

### Secrets Management

```bash
# Use Docker secrets for sensitive data
echo "your-api-key" | docker secret create wandb-api-key -

# Reference in compose file
secrets:
  - wandb-api-key

environment:
  - WANDB_API_KEY_FILE=/run/secrets/wandb-api-key
```

### Network Security

```yaml
# docker-compose.security.yml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No external access

services:
  snn-inference:
    networks:
      - backend
    # No direct external access
  
  nginx:
    networks:
      - frontend
      - backend
    ports:
      - "443:443"
```

## Monitoring and Logging

### Container Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import spike_snn_event; \
                  import requests; \
                  requests.get('http://localhost:8000/health').raise_for_status()" \
    || exit 1
```

### Logging Configuration

```yaml
services:
  snn-inference:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"
        labels: "service=snn-inference"
```

### Prometheus Metrics

```yaml
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--web.enable-lifecycle'
```

## Troubleshooting

### Common Issues

#### GPU Not Available

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Verify CUDA in container
docker run --rm --gpus all spike-snn-event-vision:latest python -c "import torch; print(torch.cuda.is_available())"
```

#### Memory Issues

```bash
# Check container memory usage
docker stats

# Increase shared memory size
docker run --shm-size=2g spike-snn-event-vision:latest

# Or in compose:
shm_size: 2gb
```

#### Permission Issues

```bash
# Fix volume permissions
sudo chown -R 1000:1000 ./models ./data

# Or use init container:
docker run --rm -v spike-snn-models:/models alpine chown -R 1000:1000 /models
```

### Debug Mode

Enable debug logging and additional diagnostics:

```bash
docker run -e DEBUG=true -e LOG_LEVEL=DEBUG spike-snn-event-vision:latest
```

### Container Shell Access

```bash
# Development container
docker-compose exec spike-snn-dev bash

# Production container (debugging)
docker run -it --entrypoint bash spike-snn-event-vision:latest
```

## Performance Optimization

### Multi-GPU Setup

```yaml
services:
  snn-inference:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2  # Use 2 GPUs
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
```

### Memory Optimization

```yaml
services:
  snn-inference:
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
    shm_size: 2gb  # Increase shared memory for PyTorch
```

### CPU Optimization

```yaml
services:
  snn-inference:
    deploy:
      resources:
        limits:
          cpus: '4.0'
        reservations:
          cpus: '2.0'
    environment:
      - OMP_NUM_THREADS=4
      - MKL_NUM_THREADS=4
```

## Maintenance

### Updates

```bash
# Pull latest images
docker-compose pull

# Recreate containers
docker-compose up -d --force-recreate

# Clean old images
docker image prune -f
```

### Backup

```bash
# Backup volumes
docker run --rm -v spike-snn-models:/data -v $(pwd):/backup \
  alpine tar czf /backup/models-backup.tar.gz /data

# Restore volumes
docker run --rm -v spike-snn-models:/data -v $(pwd):/backup \
  alpine tar xzf /backup/models-backup.tar.gz -C /
```

### Log Rotation

```yaml
services:
  snn-inference:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
```

## Next Steps

- [Kubernetes Deployment](kubernetes.md) for orchestration at scale
- [Monitoring Setup](monitoring.md) for production observability
- [Security Guide](security.md) for hardening your deployment