#!/bin/bash

# Security scanning script for spike-snn-event-vision-kit
# Performs comprehensive security analysis

set -e

# Default values
SCAN_TYPE="all"
OUTPUT_FORMAT="json"
OUTPUT_DIR="security-reports"
FAIL_ON_HIGH=true
FAIL_ON_CRITICAL=true
VERBOSE=false
CONTAINER_SCAN=false
SBOM_GENERATE=true

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
Security Scanning Script for Spike-SNN Event Vision Kit

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --type TYPE           Scan type (all|dependencies|code|secrets|container) [default: all]
    -f, --format FORMAT       Output format (json|sarif|table) [default: json]
    -o, --output-dir DIR      Output directory [default: security-reports]
    --fail-on-high            Fail on high severity vulnerabilities [default: true]
    --fail-on-critical        Fail on critical severity vulnerabilities [default: true]
    --container-scan          Scan container images
    --no-sbom                 Skip SBOM generation
    -v, --verbose             Verbose output
    -h, --help                Show this help

SCAN TYPES:
    all           Run all security scans
    dependencies  Scan dependencies for vulnerabilities
    code          Static code analysis for security issues
    secrets       Scan for secrets and sensitive data
    container     Scan container images for vulnerabilities

EXAMPLES:
    $0                              # Run all security scans
    $0 -t dependencies             # Scan only dependencies
    $0 --container-scan            # Include container scanning
    $0 -f sarif -o reports         # Output SARIF format to reports dir

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            SCAN_TYPE="$2"
            shift 2
            ;;
        -f|--format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --fail-on-high)
            FAIL_ON_HIGH=true
            shift
            ;;
        --no-fail-on-high)
            FAIL_ON_HIGH=false
            shift
            ;;
        --fail-on-critical)
            FAIL_ON_CRITICAL=true
            shift
            ;;
        --no-fail-on-critical)
            FAIL_ON_CRITICAL=false
            shift
            ;;
        --container-scan)
            CONTAINER_SCAN=true
            shift
            ;;
        --no-sbom)
            SBOM_GENERATE=false
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

# Validate scan type
case $SCAN_TYPE in
    all|dependencies|code|secrets|container)
        ;;
    *)
        log_error "Invalid scan type: $SCAN_TYPE"
        exit 1
        ;;
esac

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check requirements
check_requirements() {
    log_info "Checking security scanning requirements..."
    
    local missing_tools=()
    
    # Check for required tools based on scan type
    case $SCAN_TYPE in
        all|dependencies)
            if ! command -v safety &> /dev/null; then
                missing_tools+=("safety")
            fi
            if ! command -v pip-audit &> /dev/null; then
                missing_tools+=("pip-audit")
            fi
            ;;
    esac
    
    case $SCAN_TYPE in
        all|code)
            if ! command -v bandit &> /dev/null; then
                missing_tools+=("bandit")
            fi
            if ! command -v semgrep &> /dev/null; then
                missing_tools+=("semgrep")
            fi
            ;;
    esac
    
    case $SCAN_TYPE in
        all|secrets)
            if ! command -v detect-secrets &> /dev/null; then
                missing_tools+=("detect-secrets")
            fi
            if ! command -v truffleHog &> /dev/null; then
                missing_tools+=("truffleHog")
            fi
            ;;
    esac
    
    case $SCAN_TYPE in
        all|container)
            if [[ "$CONTAINER_SCAN" == "true" ]]; then
                if ! command -v trivy &> /dev/null; then
                    missing_tools+=("trivy")
                fi
            fi
            ;;
    esac
    
    # SBOM tools
    if [[ "$SBOM_GENERATE" == "true" ]]; then
        if ! command -v syft &> /dev/null; then
            missing_tools+=("syft")
        fi
    fi
    
    # Install missing tools
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_warning "Missing security tools: ${missing_tools[*]}"
        log_info "Installing missing tools..."
        
        for tool in "${missing_tools[@]}"; do
            install_security_tool "$tool"
        done
    fi
    
    log_success "Security tools check completed"
}

# Install security tool
install_security_tool() {
    local tool=$1
    
    case $tool in
        safety)
            pip install safety
            ;;
        pip-audit)
            pip install pip-audit
            ;;
        bandit)
            pip install bandit
            ;;
        semgrep)
            pip install semgrep
            ;;
        detect-secrets)
            pip install detect-secrets
            ;;
        truffleHog)
            # Install truffleHog
            if command -v go &> /dev/null; then
                go install github.com/trufflesecurity/trufflehog/v3@latest
            else
                log_warning "Go not available, skipping truffleHog installation"
            fi
            ;;
        trivy)
            # Install Trivy
            if command -v curl &> /dev/null; then
                curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
            else
                log_warning "curl not available, skipping Trivy installation"
            fi
            ;;
        syft)
            # Install Syft
            if command -v curl &> /dev/null; then
                curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
            else
                log_warning "curl not available, skipping Syft installation"
            fi
            ;;
    esac
}

# Dependency vulnerability scanning
scan_dependencies() {
    log_info "Scanning dependencies for vulnerabilities..."
    
    local exit_code=0
    
    # Safety scan
    if command -v safety &> /dev/null; then
        log_info "Running Safety scan..."
        local safety_output="$OUTPUT_DIR/safety-report.$OUTPUT_FORMAT"
        
        if [[ "$OUTPUT_FORMAT" == "json" ]]; then
            safety check --json --output "$safety_output" || exit_code=$?
        else
            safety check --output text > "$safety_output" || exit_code=$?
        fi
        
        log_success "Safety scan completed: $safety_output"
    fi
    
    # pip-audit scan
    if command -v pip-audit &> /dev/null; then
        log_info "Running pip-audit scan..."
        local audit_output="$OUTPUT_DIR/pip-audit-report.$OUTPUT_FORMAT"
        
        case $OUTPUT_FORMAT in
            json)
                pip-audit --format=json --output="$audit_output" || exit_code=$?
                ;;
            sarif)
                pip-audit --format=sarif --output="$audit_output" || exit_code=$?
                ;;
            *)
                pip-audit --output="$audit_output" || exit_code=$?
                ;;
        esac
        
        log_success "pip-audit scan completed: $audit_output"
    fi
    
    return $exit_code
}

# Static code analysis
scan_code() {
    log_info "Running static code analysis..."
    
    local exit_code=0
    
    # Bandit scan
    if command -v bandit &> /dev/null; then
        log_info "Running Bandit scan..."
        local bandit_output="$OUTPUT_DIR/bandit-report.$OUTPUT_FORMAT"
        
        case $OUTPUT_FORMAT in
            json)
                bandit -r src/ -f json -o "$bandit_output" || exit_code=$?
                ;;
            sarif)
                bandit -r src/ -f sarif -o "$bandit_output" || exit_code=$?
                ;;
            *)
                bandit -r src/ -f txt -o "$bandit_output" || exit_code=$?
                ;;
        esac
        
        log_success "Bandit scan completed: $bandit_output"
    fi
    
    # Semgrep scan
    if command -v semgrep &> /dev/null; then
        log_info "Running Semgrep scan..."
        local semgrep_output="$OUTPUT_DIR/semgrep-report.$OUTPUT_FORMAT"
        
        case $OUTPUT_FORMAT in
            json)
                semgrep --config=auto --json --output="$semgrep_output" src/ || exit_code=$?
                ;;
            sarif)
                semgrep --config=auto --sarif --output="$semgrep_output" src/ || exit_code=$?
                ;;
            *)
                semgrep --config=auto --output="$semgrep_output" src/ || exit_code=$?
                ;;
        esac
        
        log_success "Semgrep scan completed: $semgrep_output"
    fi
    
    return $exit_code
}

# Secrets scanning
scan_secrets() {
    log_info "Scanning for secrets and sensitive data..."
    
    local exit_code=0
    
    # detect-secrets scan
    if command -v detect-secrets &> /dev/null; then
        log_info "Running detect-secrets scan..."
        local secrets_output="$OUTPUT_DIR/detect-secrets-report.json"
        
        detect-secrets scan --all-files --baseline .secrets.baseline > "$secrets_output" || exit_code=$?
        
        log_success "detect-secrets scan completed: $secrets_output"
    fi
    
    # TruffleHog scan
    if command -v truffleHog &> /dev/null; then
        log_info "Running TruffleHog scan..."
        local trufflehog_output="$OUTPUT_DIR/trufflehog-report.json"
        
        truffleHog filesystem . --json > "$trufflehog_output" || exit_code=$?
        
        log_success "TruffleHog scan completed: $trufflehog_output"
    fi
    
    return $exit_code
}

# Container image scanning
scan_containers() {
    log_info "Scanning container images..."
    
    if [[ "$CONTAINER_SCAN" != "true" ]]; then
        log_info "Container scanning disabled, skipping..."
        return 0
    fi
    
    local exit_code=0
    
    # Find container images
    local images
    images=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "spike-snn-event-vision" || true)
    
    if [[ -z "$images" ]]; then
        log_warning "No spike-snn-event-vision images found to scan"
        return 0
    fi
    
    # Trivy scan
    if command -v trivy &> /dev/null; then
        log_info "Running Trivy container scans..."
        
        while IFS= read -r image; do
            log_info "Scanning image: $image"
            local safe_name=${image//[^a-zA-Z0-9]/-}
            local trivy_output="$OUTPUT_DIR/trivy-$safe_name.$OUTPUT_FORMAT"
            
            case $OUTPUT_FORMAT in
                json)
                    trivy image --format json --output "$trivy_output" "$image" || exit_code=$?
                    ;;
                sarif)
                    trivy image --format sarif --output "$trivy_output" "$image" || exit_code=$?
                    ;;
                *)
                    trivy image --format table --output "$trivy_output" "$image" || exit_code=$?
                    ;;
            esac
            
            log_success "Container scan completed: $trivy_output"
        done <<< "$images"
    fi
    
    return $exit_code
}

# Generate SBOM
generate_sbom() {
    if [[ "$SBOM_GENERATE" != "true" ]]; then
        log_info "SBOM generation disabled, skipping..."
        return 0
    fi
    
    log_info "Generating Software Bill of Materials (SBOM)..."
    
    if command -v syft &> /dev/null; then
        local sbom_output="$OUTPUT_DIR/sbom.spdx.json"
        syft . -o spdx-json="$sbom_output"
        log_success "SBOM generated: $sbom_output"
        
        # Also generate CycloneDX format
        local cyclonedx_output="$OUTPUT_DIR/sbom.cyclonedx.json"
        syft . -o cyclonedx-json="$cyclonedx_output"
        log_success "CycloneDX SBOM generated: $cyclonedx_output"
    else
        log_warning "Syft not available, skipping SBOM generation"
    fi
}

# Analyze results
analyze_results() {
    log_info "Analyzing security scan results..."
    
    local critical_issues=0
    local high_issues=0
    local medium_issues=0
    local low_issues=0
    
    # Count issues from different scan results
    local json_files
    json_files=$(find "$OUTPUT_DIR" -name "*.json" -type f)
    
    for file in $json_files; do
        if [[ -f "$file" ]]; then
            # Simple issue counting (would need more sophisticated parsing for each tool)
            local file_issues
            file_issues=$(grep -c "CRITICAL\|HIGH\|MEDIUM\|LOW" "$file" 2>/dev/null || echo "0")
            
            case $(basename "$file") in
                *safety*)
                    log_info "Safety scan found $file_issues potential issues"
                    ;;
                *bandit*)
                    log_info "Bandit scan found $file_issues potential issues"
                    ;;
                *semgrep*)
                    log_info "Semgrep scan found $file_issues potential issues"
                    ;;
                *trivy*)
                    log_info "Trivy scan found $file_issues potential issues"
                    ;;
            esac
        fi
    done
    
    # Generate summary report
    local summary_file="$OUTPUT_DIR/security-summary.json"
    cat > "$summary_file" << EOF
{
  "scan_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "scan_type": "$SCAN_TYPE",
  "output_format": "$OUTPUT_FORMAT",
  "summary": {
    "critical_issues": $critical_issues,
    "high_issues": $high_issues,
    "medium_issues": $medium_issues,
    "low_issues": $low_issues
  },
  "files_scanned": [
    $(find "$OUTPUT_DIR" -name "*-report.*" -type f | sed 's/.*/"&"/' | paste -sd, -)
  ]
}
EOF
    
    log_success "Security summary generated: $summary_file"
    
    # Determine exit code based on severity thresholds
    local final_exit_code=0
    
    if [[ "$FAIL_ON_CRITICAL" == "true" && $critical_issues -gt 0 ]]; then
        log_error "Critical security issues found: $critical_issues"
        final_exit_code=1
    fi
    
    if [[ "$FAIL_ON_HIGH" == "true" && $high_issues -gt 0 ]]; then
        log_error "High severity security issues found: $high_issues"
        final_exit_code=1
    fi
    
    return $final_exit_code
}

# Main execution
main() {
    log_info "Starting security scan..."
    log_info "Scan type: $SCAN_TYPE"
    log_info "Output directory: $OUTPUT_DIR"
    log_info "Output format: $OUTPUT_FORMAT"
    
    check_requirements
    
    local overall_exit_code=0
    
    # Run scans based on type
    case $SCAN_TYPE in
        all)
            scan_dependencies || overall_exit_code=$?
            scan_code || overall_exit_code=$?
            scan_secrets || overall_exit_code=$?
            scan_containers || overall_exit_code=$?
            ;;
        dependencies)
            scan_dependencies || overall_exit_code=$?
            ;;
        code)
            scan_code || overall_exit_code=$?
            ;;
        secrets)
            scan_secrets || overall_exit_code=$?
            ;;
        container)
            scan_containers || overall_exit_code=$?
            ;;
    esac
    
    # Generate SBOM
    generate_sbom
    
    # Analyze results
    analyze_results || overall_exit_code=$?
    
    if [[ $overall_exit_code -eq 0 ]]; then
        log_success "Security scan completed successfully!"
    else
        log_error "Security scan completed with issues (exit code: $overall_exit_code)"
    fi
    
    log_info "Security reports available in: $OUTPUT_DIR"
    
    exit $overall_exit_code
}

# Run main function
main "$@"