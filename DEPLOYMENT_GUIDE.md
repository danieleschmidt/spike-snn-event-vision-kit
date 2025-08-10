# Production Deployment Guide

This guide provides step-by-step instructions for deploying the Spike SNN Event Vision Kit to production.

## Prerequisites

- Docker 20.10+
- Kubernetes 1.20+
- kubectl configured
- Helm 3.0+ (optional)

## Quick Start

### 1. Build Production Container

```bash
# Build the production-optimized container
docker build -f Dockerfile.production -t spike-snn-event:v1.0.0 .

# Test the container locally
docker run -p 8080:8080 spike-snn-event:v1.0.0
```

### 2. Deploy with Docker Compose (Simple)

```bash
# Deploy the full stack locally
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f spike-snn-app
```

### 3. Deploy to Kubernetes (Production)

```bash
# Create namespace
kubectl create namespace spike-snn-production

# Apply all manifests
kubectl apply -k k8s/production/

# Check deployment status
kubectl get pods -n spike-snn-production
kubectl get services -n spike-snn-production

# Check application logs
kubectl logs -f deployment/spike-snn-app -n spike-snn-production
```

## Monitoring Setup

### 1. Deploy Monitoring Stack

```bash
# Create monitoring namespace
kubectl create namespace monitoring

# Deploy Prometheus and Grafana
kubectl apply -f monitoring/production/monitoring-deployment.yaml

# Access Grafana dashboard
kubectl port-forward svc/grafana 3000:3000 -n monitoring
# Open http://localhost:3000 (admin/admin)
```

### 2. Import Dashboard

1. Open Grafana at http://localhost:3000
2. Import the dashboard from `monitoring/production/grafana-dashboard.json`
3. Configure Prometheus data source: http://prometheus:9090

## Health Checks

The application provides several health check endpoints:

- `/health` - Overall application health
- `/ready` - Readiness for traffic
- `/metrics` - Prometheus metrics

## Scaling

### Horizontal Pod Autoscaler

The HPA is configured to scale between 2-10 replicas based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl scale deployment spike-snn-app --replicas=5 -n spike-snn-production
```

## Security Considerations

1. **Container Security**:
   - Runs as non-root user
   - Read-only root filesystem
   - No privilege escalation

2. **Network Security**:
   - All communication over HTTPS
   - Rate limiting enabled
   - Network policies configured

3. **Secrets Management**:
   - Use Kubernetes secrets for sensitive data
   - Enable secret encryption at rest

## Troubleshooting

### Common Issues

1. **Pod not starting**:
   ```bash
   kubectl describe pod <pod-name> -n spike-snn-production
   kubectl logs <pod-name> -n spike-snn-production
   ```

2. **High memory usage**:
   - Check Grafana dashboard
   - Adjust resource limits
   - Consider horizontal scaling

3. **Performance issues**:
   - Monitor response times
   - Check CPU utilization
   - Scale if necessary

### Recovery Procedures

1. **Rolling restart**:
   ```bash
   kubectl rollout restart deployment/spike-snn-app -n spike-snn-production
   ```

2. **Rollback deployment**:
   ```bash
   kubectl rollout undo deployment/spike-snn-app -n spike-snn-production
   ```

## Maintenance

### Updates

1. Build new container with updated tag
2. Update image in deployment manifest
3. Apply rolling update:
   ```bash
   kubectl set image deployment/spike-snn-app spike-snn-app=spike-snn-event:v1.1.0 -n spike-snn-production
   ```

### Backup

- Application is stateless
- Configuration is stored in ConfigMaps
- Logs are centralized in monitoring system

## Support

For production support:
- Check monitoring dashboards first
- Review application logs
- Check Kubernetes events
- Contact development team with specific error details
