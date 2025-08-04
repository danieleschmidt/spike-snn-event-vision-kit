#!/bin/bash
# Automated deployment script for Spike SNN Event Vision Kit
# Supports multiple deployment targets: local, staging, production

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_DIR="${PROJECT_ROOT}/deploy"

# Default values
ENVIRONMENT="${ENVIRONMENT:-local}"
BUILD_CACHE="${BUILD_CACHE:-true}"
PUSH_IMAGES="${PUSH_IMAGES:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_DEPLOY="${FORCE_DEPLOY:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Help function
show_help() {
    cat << EOF
Spike SNN Event Vision Kit Deployment Script

Usage: $0 [OPTIONS] ENVIRONMENT

ENVIRONMENT:
    local       Deploy to local Docker/Kind cluster
    staging     Deploy to staging Kubernetes cluster  
    production  Deploy to production Kubernetes cluster

OPTIONS:
    -h, --help          Show this help message
    -b, --build         Force rebuild Docker images
    -p, --push          Push images to registry
    -t, --skip-tests    Skip running tests
    -d, --dry-run       Show what would be deployed without executing
    -f, --force         Force deployment even if validation fails
    --no-cache          Build images without cache

EXAMPLES:
    $0 local                    # Deploy to local environment
    $0 staging --build --push   # Rebuild and deploy to staging
    $0 production --dry-run     # Preview production deployment

ENVIRONMENT VARIABLES:
    REGISTRY_URL        Docker registry URL (default: localhost:5000)
    IMAGE_TAG          Docker image tag (default: latest)
    KUBECONFIG         Path to kubeconfig file
    BUILD_CACHE        Enable/disable build cache (default: true)
    SKIP_TESTS         Skip test execution (default: false)

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -b|--build)
                BUILD_IMAGES="true"
                shift
                ;;
            -p|--push)
                PUSH_IMAGES="true"
                shift
                ;;
            -t|--skip-tests)
                SKIP_TESTS="true"
                shift
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -f|--force)
                FORCE_DEPLOY="true"
                shift
                ;;
            --no-cache)
                BUILD_CACHE="false"
                shift
                ;;
            -*)
                log_error "Unknown option $1"
                show_help
                exit 1
                ;;
            *)
                ENVIRONMENT="$1"
                shift
                ;;
        esac
    done
}

# Validation functions
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    for tool in docker kubectl helm terraform; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install missing tools and try again"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Environment-specific checks
    case "$ENVIRONMENT" in
        local)
            if ! kind version &> /dev/null; then
                log_warning "Kind not found. Installing..."
                install_kind
            fi
            ;;
        staging|production)
            if [[ -z "${KUBECONFIG:-}" ]]; then
                log_error "KUBECONFIG environment variable not set for $ENVIRONMENT"
                exit 1
            fi
            
            if ! kubectl cluster-info &> /dev/null; then
                log_error "Cannot connect to Kubernetes cluster"
                exit 1
            fi
            ;;
    esac
    
    log_success "Prerequisites check passed"
}

install_kind() {
    log_info "Installing Kind..."
    curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
    chmod +x ./kind
    sudo mv ./kind /usr/local/bin/kind
    log_success "Kind installed successfully"
}

# Build functions
build_images() {
    log_info "Building Docker images..."
    
    local cache_flag=""
    if [[ "$BUILD_CACHE" == "false" ]]; then
        cache_flag="--no-cache"
    fi
    
    local image_tag="${IMAGE_TAG:-latest}"
    local registry_url="${REGISTRY_URL:-localhost:5000}"
    
    # Build production image
    log_info "Building production image..."
    docker build $cache_flag \
        --target production \
        -t "${registry_url}/spike-snn-event:${image_tag}" \
        -t "${registry_url}/spike-snn-event:latest" \
        "$PROJECT_ROOT"
    
    # Build inference image
    log_info "Building inference image..."
    docker build $cache_flag \
        --target inference \
        -t "${registry_url}/spike-snn-event:${image_tag}-inference" \
        -t "${registry_url}/spike-snn-event:inference" \
        "$PROJECT_ROOT"
    
    # Build training image
    log_info "Building training image..."
    docker build $cache_flag \
        --target training \
        -t "${registry_url}/spike-snn-event:${image_tag}-training" \
        -t "${registry_url}/spike-snn-event:training" \
        "$PROJECT_ROOT"
    
    # Build ROS2 image if ROS2 integration is needed
    if [[ "$ENVIRONMENT" == "ros2" ]] || [[ "${ENABLE_ROS2:-false}" == "true" ]]; then
        log_info "Building ROS2 image..."
        docker build $cache_flag \
            --target ros2 \
            -t "${registry_url}/spike-snn-event:${image_tag}-ros2" \
            -t "${registry_url}/spike-snn-event:ros2" \
            "$PROJECT_ROOT"
    fi
    
    log_success "Docker images built successfully"
}

push_images() {
    if [[ "$PUSH_IMAGES" == "true" ]]; then
        log_info "Pushing images to registry..."
        
        local image_tag="${IMAGE_TAG:-latest}"
        local registry_url="${REGISTRY_URL:-localhost:5000}"
        
        docker push "${registry_url}/spike-snn-event:${image_tag}"
        docker push "${registry_url}/spike-snn-event:latest"
        docker push "${registry_url}/spike-snn-event:${image_tag}-inference"
        docker push "${registry_url}/spike-snn-event:inference"
        docker push "${registry_url}/spike-snn-event:${image_tag}-training"
        docker push "${registry_url}/spike-snn-event:training"
        
        if [[ "${ENABLE_ROS2:-false}" == "true" ]]; then
            docker push "${registry_url}/spike-snn-event:${image_tag}-ros2"
            docker push "${registry_url}/spike-snn-event:ros2"
        fi
        
        log_success "Images pushed successfully"
    fi
}

# Test functions
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests"
        return 0
    fi
    
    log_info "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run unit tests
    log_info "Running unit tests..."
    if ! python -m pytest tests/unit/ -v; then
        log_error "Unit tests failed"
        return 1
    fi
    
    # Run integration tests
    log_info "Running integration tests..."
    if ! python -m pytest tests/integration/ -v; then
        log_error "Integration tests failed"
        return 1
    fi
    
    # Run security tests
    log_info "Running security tests..."
    if command -v bandit &> /dev/null; then
        bandit -r src/
    else
        log_warning "Bandit not found, skipping security tests"
    fi
    
    log_success "All tests passed"
}

# Deployment functions
deploy_local() {
    log_info "Deploying to local environment..."
    
    # Create kind cluster if it doesn't exist
    if ! kind get clusters | grep -q "spike-snn-local"; then
        log_info "Creating local Kind cluster..."
        cat << EOF | kind create cluster --name spike-snn-local --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  - containerPort: 443
    hostPort: 443
    protocol: TCP
- role: worker
- role: worker
EOF
    fi
    
    # Load images into kind cluster
    log_info "Loading images into Kind cluster..."
    kind load docker-image "localhost:5000/spike-snn-event:latest" --name spike-snn-local
    
    # Deploy using Helm
    log_info "Deploying with Helm..."
    helm upgrade --install spike-snn-event \
        "$DEPLOY_DIR/helm/spike-snn-event" \
        --namespace spike-snn-event \
        --create-namespace \
        --values "$DEPLOY_DIR/helm/spike-snn-event/values-local.yaml" \
        --wait
    
    log_success "Local deployment completed"
    
    # Show access information
    log_info "Application will be available at: http://localhost"
    log_info "Use 'kubectl port-forward' for direct access to services"
}

deploy_staging() {
    log_info "Deploying to staging environment..."
    
    # Deploy using Helm
    helm upgrade --install spike-snn-event-staging \
        "$DEPLOY_DIR/helm/spike-snn-event" \
        --namespace spike-snn-staging \
        --create-namespace \
        --values "$DEPLOY_DIR/helm/spike-snn-event/values-staging.yaml" \
        --set image.tag="${IMAGE_TAG:-latest}" \
        --wait
    
    log_success "Staging deployment completed"
}

deploy_production() {
    log_info "Deploying to production environment..."
    
    # Additional validation for production
    if [[ "$FORCE_DEPLOY" != "true" ]]; then
        log_info "Running production pre-deployment validation..."
        
        # Check if staging tests pass
        if ! run_staging_smoke_tests; then
            log_error "Staging smoke tests failed. Use --force to override."
            exit 1
        fi
        
        # Check resource quotas
        if ! check_resource_quotas; then
            log_error "Insufficient resources for production deployment"
            exit 1
        fi
    fi
    
    # Deploy using Helm with production values
    helm upgrade --install spike-snn-event-prod \
        "$DEPLOY_DIR/helm/spike-snn-event" \
        --namespace spike-snn-production \
        --create-namespace \
        --values "$DEPLOY_DIR/helm/spike-snn-event/values-production.yaml" \
        --set image.tag="${IMAGE_TAG:-latest}" \
        --wait \
        --timeout=10m
    
    log_success "Production deployment completed"
}

run_staging_smoke_tests() {
    log_info "Running staging smoke tests..."
    # Placeholder for staging smoke tests
    return 0
}

check_resource_quotas() {
    log_info "Checking resource quotas..."
    # Placeholder for resource quota checks
    return 0
}

# Infrastructure deployment
deploy_infrastructure() {
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_info "Deploying infrastructure with Terraform..."
        
        cd "$DEPLOY_DIR/terraform"
        
        # Initialize Terraform
        terraform init
        
        # Plan deployment
        terraform plan -out=tfplan
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "Dry run - infrastructure plan created but not applied"
            return 0
        fi
        
        # Apply infrastructure changes
        terraform apply -auto-approve tfplan
        
        # Get cluster credentials
        aws eks update-kubeconfig --region us-west-2 --name spike-snn-cluster
        
        log_success "Infrastructure deployment completed"
    fi
}

# Post-deployment functions
post_deployment_checks() {
    log_info "Running post-deployment checks..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/name=spike-snn-event \
        --namespace="spike-snn-${ENVIRONMENT}" \
        --timeout=300s
    
    # Run health checks
    log_info "Running health checks..."
    
    # Get service endpoint
    local service_name="spike-snn-event-service"
    local namespace="spike-snn-${ENVIRONMENT}"
    
    if kubectl get service "$service_name" -n "$namespace" &> /dev/null; then
        # Port forward for health check
        kubectl port-forward service/"$service_name" 8080:80 -n "$namespace" &
        local port_forward_pid=$!
        
        sleep 5
        
        # Health check
        if curl -f http://localhost:8080/health &> /dev/null; then
            log_success "Health check passed"
        else
            log_error "Health check failed"
            kill $port_forward_pid
            return 1
        fi
        
        kill $port_forward_pid
    fi
    
    log_success "Post-deployment checks completed"
}

# Cleanup functions
cleanup() {
    log_info "Cleaning up temporary resources..."
    
    # Kill any background processes
    jobs -p | xargs -r kill
    
    # Remove temporary files
    rm -f /tmp/spike-snn-*
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting deployment for environment: $ENVIRONMENT"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Check prerequisites
    check_prerequisites
    
    # Run tests
    run_tests
    
    # Build images if needed
    if [[ "${BUILD_IMAGES:-false}" == "true" ]] || [[ "$ENVIRONMENT" == "local" ]]; then
        build_images
    fi
    
    # Push images if needed
    push_images
    
    # Deploy infrastructure (for production)
    deploy_infrastructure
    
    # Deploy application
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run - would deploy to $ENVIRONMENT environment"
        return 0
    fi
    
    case "$ENVIRONMENT" in
        local)
            deploy_local
            ;;
        staging)
            deploy_staging
            ;;
        production)
            deploy_production
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    # Post-deployment checks
    post_deployment_checks
    
    log_success "Deployment completed successfully!"
}

# Parse arguments and run main function
parse_args "$@"
main