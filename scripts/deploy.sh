#!/bin/bash

# Deployment script for spike-snn-event-vision-kit
# Supports Docker Compose and Kubernetes deployments

set -e

# Default values
DEPLOYMENT_TYPE="docker-compose"
ENVIRONMENT="development"
REGISTRY=""
TAG="latest"
NAMESPACE="default"
DRY_RUN=false
VERBOSE=false
CONFIG_FILE=""
WAIT_TIMEOUT=300

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Deployment Script for Spike-SNN Event Vision Kit

Usage: $0 [OPTIONS] COMMAND

COMMANDS:
    deploy      Deploy the application
    stop        Stop the application
    restart     Restart the application
    status      Show deployment status
    logs        Show application logs
    clean       Clean up deployment

OPTIONS:
    -t, --type TYPE           Deployment type (docker-compose|k8s) [default: docker-compose]
    -e, --env ENVIRONMENT     Environment (development|production|staging) [default: development]
    -r, --registry REGISTRY   Docker registry
    --tag TAG                 Image tag [default: latest]
    -n, --namespace NAMESPACE Kubernetes namespace [default: default]
    -c, --config CONFIG       Configuration file
    --dry-run                 Show what would be deployed without executing
    --timeout SECONDS         Wait timeout in seconds [default: 300]
    -v, --verbose             Verbose output
    -h, --help                Show this help

EXAMPLES:
    # Docker Compose deployments
    $0 deploy                                    # Deploy development environment
    $0 deploy -e production                      # Deploy production environment
    $0 stop                                      # Stop all services
    $0 logs                                      # Show logs

    # Kubernetes deployments
    $0 deploy -t k8s -e production -n spike-snn # Deploy to Kubernetes
    $0 status -t k8s -n spike-snn               # Check Kubernetes status

EOF
}

# Parse command line arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        deploy|stop|restart|status|logs|clean)
            COMMAND="$1"
            shift
            ;;
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --timeout)
            WAIT_TIMEOUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate command
if [[ -z "$COMMAND" ]]; then
    log_error "No command specified"
    show_help
    exit 1
fi

# Check requirements
check_requirements() {
    log_info "Checking deployment requirements..."
    
    case $DEPLOYMENT_TYPE in
        docker-compose)
            if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
                log_error "Docker Compose is not installed"
                exit 1
            fi
            
            if [[ ! -f "docker-compose.yml" ]]; then
                log_error "docker-compose.yml not found"
                exit 1
            fi
            ;;
        k8s)
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl is not installed"
                exit 1
            fi
            
            # Check cluster connectivity
            if ! kubectl cluster-info &> /dev/null; then
                log_error "Cannot connect to Kubernetes cluster"
                exit 1
            fi
            ;;
        *)
            log_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    log_success "Requirements check passed"
}

# Docker Compose functions
docker_compose_deploy() {
    local compose_file="docker-compose.yml"
    local profile_args=""
    
    # Set profiles based on environment
    case $ENVIRONMENT in
        development)
            profile_args="--profile development"
            ;;
        production)
            profile_args="--profile production --profile monitoring"
            ;;
        staging)
            profile_args="--profile production"
            ;;
    esac
    
    # Set environment variables
    export COMPOSE_PROJECT_NAME="spike-snn-${ENVIRONMENT}"
    
    if [[ -n "$REGISTRY" ]]; then
        export REGISTRY_PREFIX="${REGISTRY}/"
    fi
    
    export IMAGE_TAG="$TAG"
    
    log_info "Deploying with Docker Compose..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Project: $COMPOSE_PROJECT_NAME"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run - would execute:"
        echo "docker-compose -f $compose_file $profile_args up -d"
        return 0
    fi
    
    # Pull latest images
    log_info "Pulling latest images..."
    docker-compose -f "$compose_file" $profile_args pull
    
    # Deploy services
    log_info "Starting services..."
    docker-compose -f "$compose_file" $profile_args up -d
    
    # Wait for health checks
    wait_for_services_docker_compose "$compose_file" "$profile_args"
    
    log_success "Docker Compose deployment completed"
}

docker_compose_stop() {
    local compose_file="docker-compose.yml"
    
    log_info "Stopping Docker Compose services..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run - would execute:"
        echo "docker-compose -f $compose_file down"
        return 0
    fi
    
    docker-compose -f "$compose_file" down
    log_success "Services stopped"
}

docker_compose_status() {
    local compose_file="docker-compose.yml"
    
    log_info "Docker Compose service status:"
    docker-compose -f "$compose_file" ps
}

docker_compose_logs() {
    local compose_file="docker-compose.yml"
    
    log_info "Docker Compose logs:"
    docker-compose -f "$compose_file" logs -f --tail=100
}

wait_for_services_docker_compose() {
    local compose_file="$1"
    local profile_args="$2"
    
    log_info "Waiting for services to be healthy..."
    
    local timeout=$WAIT_TIMEOUT
    local interval=5
    
    while [[ $timeout -gt 0 ]]; do
        local unhealthy_services
        unhealthy_services=$(docker-compose -f "$compose_file" $profile_args ps --filter "health=unhealthy" -q)
        
        if [[ -z "$unhealthy_services" ]]; then
            log_success "All services are healthy"
            return 0
        fi
        
        log_info "Waiting for services to become healthy... (${timeout}s remaining)"
        sleep $interval
        timeout=$((timeout - interval))
    done
    
    log_warning "Timeout waiting for services to become healthy"
    docker-compose -f "$compose_file" $profile_args ps
}

# Kubernetes functions
k8s_deploy() {
    local manifests_dir="k8s"
    
    if [[ ! -d "$manifests_dir" ]]; then
        log_error "Kubernetes manifests directory not found: $manifests_dir"
        exit 1
    fi
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace: $NAMESPACE"
        if [[ "$DRY_RUN" == "false" ]]; then
            kubectl create namespace "$NAMESPACE"
        fi
    fi
    
    log_info "Deploying to Kubernetes..."
    log_info "Namespace: $NAMESPACE"
    log_info "Environment: $ENVIRONMENT"
    
    # Apply manifests
    local manifest_files
    manifest_files=$(find "$manifests_dir" -name "*.yaml" -o -name "*.yml" | sort)
    
    for manifest in $manifest_files; do
        # Skip environment-specific files if they don't match
        if [[ "$manifest" == *"production"* && "$ENVIRONMENT" != "production" ]]; then
            continue
        fi
        if [[ "$manifest" == *"development"* && "$ENVIRONMENT" != "development" ]]; then
            continue
        fi
        
        log_info "Applying: $manifest"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            kubectl apply --dry-run=client -f "$manifest" -n "$NAMESPACE"
        else
            kubectl apply -f "$manifest" -n "$NAMESPACE"
        fi
    done
    
    if [[ "$DRY_RUN" == "false" ]]; then
        wait_for_deployment_k8s
    fi
    
    log_success "Kubernetes deployment completed"
}

k8s_stop() {
    log_info "Stopping Kubernetes deployment..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run - would delete resources in namespace: $NAMESPACE"
        return 0
    fi
    
    # Scale down deployments
    kubectl scale deployment --all --replicas=0 -n "$NAMESPACE"
    
    # Delete resources (optional - uncomment if you want full cleanup)
    # kubectl delete all --all -n "$NAMESPACE"
    
    log_success "Kubernetes deployment stopped"
}

k8s_status() {
    log_info "Kubernetes deployment status:"
    
    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    echo ""
    echo "Services:"
    kubectl get services -n "$NAMESPACE"
    
    echo ""
    echo "Deployments:"
    kubectl get deployments -n "$NAMESPACE"
}

k8s_logs() {
    log_info "Kubernetes logs:"
    
    # Get all pods with spike-snn label
    local pods
    pods=$(kubectl get pods -n "$NAMESPACE" -l app=spike-snn -o name)
    
    if [[ -z "$pods" ]]; then
        log_warning "No pods found with label app=spike-snn"
        return 1
    fi
    
    # Follow logs from all pods
    kubectl logs -f -l app=spike-snn -n "$NAMESPACE" --max-log-requests=10
}

wait_for_deployment_k8s() {
    log_info "Waiting for deployment to be ready..."
    
    # Wait for deployments to be ready
    local deployments
    deployments=$(kubectl get deployments -n "$NAMESPACE" -o name)
    
    for deployment in $deployments; do
        log_info "Waiting for $deployment..."
        kubectl wait --for=condition=available --timeout="${WAIT_TIMEOUT}s" "$deployment" -n "$NAMESPACE"
    done
    
    log_success "All deployments are ready"
}

# Health check functions
health_check() {
    case $DEPLOYMENT_TYPE in
        docker-compose)
            health_check_docker_compose
            ;;
        k8s)
            health_check_k8s
            ;;
    esac
}

health_check_docker_compose() {
    log_info "Running health checks..."
    
    # Check if main services are running
    local services=("spike-snn-dev" "spike-snn-inference")
    
    for service in "${services[@]}"; do
        if docker ps --filter "name=$service" --filter "status=running" | grep -q "$service"; then
            log_success "$service is running"
        else
            log_warning "$service is not running"
        fi
    done
}

health_check_k8s() {
    log_info "Running Kubernetes health checks..."
    
    # Check pod status
    local failed_pods
    failed_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running -o name)
    
    if [[ -z "$failed_pods" ]]; then
        log_success "All pods are running"
    else
        log_warning "Some pods are not running:"
        echo "$failed_pods"
    fi
}

# Main execution functions
execute_command() {
    case $DEPLOYMENT_TYPE in
        docker-compose)
            case $COMMAND in
                deploy)
                    docker_compose_deploy
                    ;;
                stop)
                    docker_compose_stop
                    ;;
                restart)
                    docker_compose_stop
                    sleep 2
                    docker_compose_deploy
                    ;;
                status)
                    docker_compose_status
                    ;;
                logs)
                    docker_compose_logs
                    ;;
                clean)
                    docker_compose_stop
                    docker system prune -f
                    ;;
            esac
            ;;
        k8s)
            case $COMMAND in
                deploy)
                    k8s_deploy
                    ;;
                stop)
                    k8s_stop
                    ;;
                restart)
                    k8s_stop
                    sleep 5
                    k8s_deploy
                    ;;
                status)
                    k8s_status
                    ;;
                logs)
                    k8s_logs
                    ;;
                clean)
                    k8s_stop
                    kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
                    ;;
            esac
            ;;
    esac
}

# Main function
main() {
    log_info "Starting deployment process..."
    log_info "Command: $COMMAND"
    log_info "Type: $DEPLOYMENT_TYPE"
    log_info "Environment: $ENVIRONMENT"
    
    check_requirements
    execute_command
    
    if [[ "$COMMAND" == "deploy" && "$DRY_RUN" == "false" ]]; then
        health_check
    fi
    
    log_success "Deployment process completed!"
}

# Run main function
main "$@"