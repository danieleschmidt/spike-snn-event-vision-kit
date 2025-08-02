#!/bin/bash

# Build script for spike-snn-event-vision-kit
# Supports multiple build targets and configurations

set -e

# Default values
BUILD_TARGET="development"
BUILD_PLATFORM="linux/amd64"
PUSH_IMAGES=false
REGISTRY=""
TAG="latest"
NO_CACHE=false
VERBOSE=false
PARALLEL_BUILD=true

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
Build Script for Spike-SNN Event Vision Kit

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --target TARGET       Build target (development|production|ros2|cpu-only) [default: development]
    -p, --platform PLATFORM  Target platform [default: linux/amd64]
    -r, --registry REGISTRY  Docker registry to use
    --tag TAG                 Image tag [default: latest]
    --push                    Push images to registry
    --no-cache                Build without cache
    --no-parallel             Disable parallel building
    -v, --verbose             Verbose output
    -h, --help                Show this help

EXAMPLES:
    $0                                    # Build development image
    $0 -t production --tag v1.0.0        # Build production image with tag
    $0 -t ros2 --push -r myregistry.com  # Build and push ROS2 image
    $0 --platform linux/arm64            # Build for ARM64

TARGETS:
    development     Full development environment with Jupyter and debugging
    production      Optimized production image
    ros2           ROS2 integration image
    cpu-only       CPU-only image (no CUDA)
    all            Build all targets

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        -p|--platform)
            BUILD_PLATFORM="$2"
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
        --push)
            PUSH_IMAGES=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --no-parallel)
            PARALLEL_BUILD=false
            shift
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

# Validate build target
case $BUILD_TARGET in
    development|production|ros2|cpu-only|all)
        ;;
    *)
        log_error "Invalid build target: $BUILD_TARGET"
        log_error "Valid targets: development, production, ros2, cpu-only, all"
        exit 1
        ;;
esac

# Set registry prefix
if [[ -n "$REGISTRY" ]]; then
    REGISTRY_PREFIX="${REGISTRY}/"
else
    REGISTRY_PREFIX=""
fi

# Build options
BUILD_OPTS=""
if [[ "$NO_CACHE" == "true" ]]; then
    BUILD_OPTS="$BUILD_OPTS --no-cache"
fi

if [[ "$VERBOSE" == "true" ]]; then
    BUILD_OPTS="$BUILD_OPTS --progress=plain"
fi

# Check requirements
check_requirements() {
    log_info "Checking build requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check Dockerfile
    if [[ ! -f "Dockerfile" ]]; then
        log_error "Dockerfile not found in current directory"
        exit 1
    fi
    
    # Check buildx for multi-platform builds
    if [[ "$BUILD_PLATFORM" != "linux/amd64" ]]; then
        if ! docker buildx version &> /dev/null; then
            log_error "Docker buildx is required for multi-platform builds"
            exit 1
        fi
    fi
    
    log_success "All requirements satisfied"
}

# Build image function
build_image() {
    local target=$1
    local image_name="${REGISTRY_PREFIX}spike-snn-event-vision:${target}-${TAG}"
    
    log_info "Building image: $image_name (target: $target, platform: $BUILD_PLATFORM)"
    
    # Build command
    local build_cmd="docker build"
    
    # Use buildx for multi-platform or if requested
    if [[ "$BUILD_PLATFORM" != "linux/amd64" ]] || command -v docker buildx &> /dev/null; then
        build_cmd="docker buildx build"
        BUILD_OPTS="$BUILD_OPTS --platform $BUILD_PLATFORM"
    fi
    
    # Add target and tag
    BUILD_OPTS="$BUILD_OPTS --target $target -t $image_name"
    
    # Execute build
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Build command: $build_cmd $BUILD_OPTS ."
    fi
    
    eval "$build_cmd $BUILD_OPTS ." || {
        log_error "Failed to build $target image"
        return 1
    }
    
    log_success "Successfully built: $image_name"
    
    # Push if requested
    if [[ "$PUSH_IMAGES" == "true" ]]; then
        log_info "Pushing image: $image_name"
        docker push "$image_name" || {
            log_error "Failed to push $image_name"
            return 1
        }
        log_success "Successfully pushed: $image_name"
    fi
}

# Build all targets
build_all_targets() {
    local targets=("development" "production" "ros2" "cpu-only")
    local failed_builds=()
    
    if [[ "$PARALLEL_BUILD" == "true" ]]; then
        log_info "Building all targets in parallel..."
        local pids=()
        
        for target in "${targets[@]}"; do
            build_image "$target" &
            pids+=($!)
        done
        
        # Wait for all builds to complete
        for i in "${!pids[@]}"; do
            if ! wait "${pids[$i]}"; then
                failed_builds+=("${targets[$i]}")
            fi
        done
    else
        log_info "Building all targets sequentially..."
        for target in "${targets[@]}"; do
            if ! build_image "$target"; then
                failed_builds+=("$target")
            fi
        done
    fi
    
    # Report results
    if [[ ${#failed_builds[@]} -eq 0 ]]; then
        log_success "All targets built successfully"
    else
        log_error "Failed builds: ${failed_builds[*]}"
        exit 1
    fi
}

# Pre-build checks
pre_build_checks() {
    log_info "Running pre-build checks..."
    
    # Check for large files
    local large_files
    large_files=$(find . -name "*.pth" -o -name "*.onnx" -o -name "*.h5" | head -5)
    if [[ -n "$large_files" ]]; then
        log_warning "Found large model files that will be excluded from build context:"
        echo "$large_files"
    fi
    
    # Check for secrets
    local potential_secrets
    potential_secrets=$(find . -name "*.key" -o -name "*.pem" -o -name "*secret*" -o -name "*credential*" | head -5)
    if [[ -n "$potential_secrets" ]]; then
        log_warning "Found potential secret files (will be excluded):"
        echo "$potential_secrets"
    fi
    
    # Estimate build context size
    local context_size
    context_size=$(du -sh . 2>/dev/null | cut -f1)
    log_info "Build context size: $context_size"
    
    log_success "Pre-build checks completed"
}

# Post-build checks
post_build_checks() {
    log_info "Running post-build checks..."
    
    # List built images
    log_info "Built images:"
    docker images | grep "spike-snn-event-vision" | grep "$TAG"
    
    # Check image sizes
    log_info "Image sizes:"
    docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" | grep "spike-snn-event-vision.*$TAG"
    
    log_success "Post-build checks completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up build artifacts..."
    
    # Remove dangling images
    local dangling_images
    dangling_images=$(docker images -f "dangling=true" -q)
    if [[ -n "$dangling_images" ]]; then
        log_info "Removing dangling images..."
        docker rmi $dangling_images || log_warning "Failed to remove some dangling images"
    fi
    
    # Prune build cache if requested
    if [[ "$NO_CACHE" == "true" ]]; then
        log_info "Pruning build cache..."
        docker builder prune -f || log_warning "Failed to prune build cache"
    fi
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    log_info "Starting build process..."
    log_info "Target: $BUILD_TARGET"
    log_info "Platform: $BUILD_PLATFORM"
    log_info "Tag: $TAG"
    
    check_requirements
    pre_build_checks
    
    # Build based on target
    case $BUILD_TARGET in
        all)
            build_all_targets
            ;;
        *)
            build_image "$BUILD_TARGET"
            ;;
    esac
    
    post_build_checks
    cleanup
    
    log_success "Build process completed successfully!"
}

# Trap for cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"