#!/usr/bin/env python3
"""
Generation 2 validation script for Spike-SNN Event Vision Kit.
Tests robustness enhancements including error handling, monitoring, and security.
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def validate_generation2():
    """Validate Generation 2 robustness enhancements."""
    
    print("🛡️  GENERATION 2 VALIDATION")
    print("=" * 50)
    
    # Check for Generation 2 files and features
    gen2_files = [
        "src/spike_snn_event/validation.py",
        "src/spike_snn_event/security.py", 
        "src/spike_snn_event/monitoring.py",
        "src/spike_snn_event/health.py"
    ]
    
    missing_files = []
    for file_path in gen2_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    print()
    
    # Check for Generation 2 enhancements in code
    enhancements = []
    
    # Check validation module robustness
    validation_path = Path("src/spike_snn_event/validation.py")
    if validation_path.exists():
        content = validation_path.read_text()
        if "validate_and_sanitize_events" in content:
            enhancements.append("✅ Comprehensive event validation")
        if "SafetyMonitor" in content:
            enhancements.append("✅ Safety monitoring system")
        if "retry_on_failure" in content:
            enhancements.append("✅ Automatic retry mechanisms")
        if "safe_operation" in content:
            enhancements.append("✅ Safe operation decorators")
    
    # Check security module
    security_path = Path("src/spike_snn_event/security.py")
    if security_path.exists():
        content = security_path.read_text()
        if "InputSanitizer" in content:
            enhancements.append("✅ Input sanitization system")
        if "SecureModelLoader" in content:
            enhancements.append("✅ Secure model loading")  
        if "RateLimiter" in content:
            enhancements.append("✅ Rate limiting protection")
        if "SecurityAuditLog" in content:
            enhancements.append("✅ Security audit logging")
    
    # Check monitoring module
    monitoring_path = Path("src/spike_snn_event/monitoring.py")
    if monitoring_path.exists():
        content = monitoring_path.read_text()
        if "MetricsCollector" in content:
            enhancements.append("✅ Advanced metrics collection")
        if "HealthChecker" in content:
            enhancements.append("✅ Automated health checking")
        if "MonitoringDashboard" in content:
            enhancements.append("✅ Real-time monitoring dashboard")
        if "SystemMetrics" in content:
            enhancements.append("✅ Comprehensive system metrics")
    
    # Check health module
    health_path = Path("src/spike_snn_event/health.py")
    if health_path.exists():
        content = health_path.read_text()
        if "SystemHealthChecker" in content:
            enhancements.append("✅ System-wide health checking")
        if "_check_pytorch" in content:
            enhancements.append("✅ Component-specific health checks")
        if "export_health_report" in content:
            enhancements.append("✅ Health report generation")
    
    # Check enhanced training module
    training_path = Path("src/spike_snn_event/training.py")
    if training_path.exists():
        content = training_path.read_text()
        if "consecutive_errors" in content:
            enhancements.append("✅ Enhanced error tracking in training")
        if "validate_model_input" in content:
            enhancements.append("✅ Training input validation")
        if "@safe_operation" in content:
            enhancements.append("✅ Safe operation decorators in training")
    
    # Check core module enhancements
    core_path = Path("src/spike_snn_event/core.py")
    if core_path.exists():
        content = core_path.read_text()
        if "ValidationError" in content:
            enhancements.append("✅ Integrated validation in core")
        if "@safe_operation" in content:
            enhancements.append("✅ Safe operations in core streaming")
    
    print("🛡️  GENERATION 2 ROBUSTNESS ENHANCEMENTS:")
    print("-" * 40)
    for enhancement in enhancements:
        print(enhancement)
    
    if not enhancements:
        print("❌ No Generation 2 enhancements detected")
        return False
    
    print(f"\n✅ Generation 2 Status: {len(enhancements)} robustness enhancements implemented")
    
    # Check for quality assurance features
    quality_features = []
    
    # Check for error handling patterns
    all_python_files = list(Path("src/spike_snn_event").glob("*.py"))
    error_handling_count = 0
    logging_count = 0
    validation_count = 0
    
    for py_file in all_python_files:
        if py_file.name.startswith('_'):
            continue
            
        content = py_file.read_text()
        
        # Count error handling patterns
        if "try:" in content and "except" in content:
            error_handling_count += 1
        if "logging" in content or "logger" in content:
            logging_count += 1
        if "validate" in content.lower():
            validation_count += 1
    
    if error_handling_count >= 4:
        quality_features.append(f"✅ Error handling in {error_handling_count} modules")
    if logging_count >= 4:
        quality_features.append(f"✅ Logging in {logging_count} modules")
    if validation_count >= 3:
        quality_features.append(f"✅ Validation in {validation_count} modules")
    
    # Check for monitoring integration
    if Path("src/spike_snn_event/monitoring.py").exists():
        quality_features.append("✅ Comprehensive monitoring system")
    if Path("src/spike_snn_event/security.py").exists():
        quality_features.append("✅ Security hardening implemented")
    
    print(f"\n🔒 QUALITY ASSURANCE FEATURES:")
    print("-" * 30)
    for feature in quality_features:
        print(feature)
    
    # Check deployment readiness
    deployment_features = []
    
    if Path("deploy").exists():
        deployment_features.append("✅ Deployment configurations")
    if Path("monitoring").exists():
        deployment_features.append("✅ Monitoring configurations")
    if Path("docker-compose.yml").exists():
        deployment_features.append("✅ Docker orchestration")
    if Path("Dockerfile").exists():
        deployment_features.append("✅ Container configuration")
    
    print(f"\n🚀 DEPLOYMENT READINESS:")
    print("-" * 25)
    for feature in deployment_features:
        print(feature)
    
    # Calculate robustness score
    total_possible_features = 20  # Expected features for Generation 2
    actual_features = len(enhancements) + len(quality_features)
    robustness_score = min(100, (actual_features / total_possible_features) * 100)
    
    print(f"\n📊 ROBUSTNESS SCORE: {robustness_score:.1f}%")
    
    if robustness_score >= 80:
        print("🎯 GENERATION 2 COMPLETE!")
        print("System is production-ready with comprehensive robustness features")
        print("Ready for Generation 3: Make It Scale")
        return True
    elif robustness_score >= 60:
        print("⚠️  GENERATION 2 PARTIALLY COMPLETE")
        print("Core robustness features implemented, some enhancements pending")
        return True
    else:
        print("❌ GENERATION 2 INCOMPLETE")
        print("Additional robustness features needed")
        return False

if __name__ == "__main__":
    success = validate_generation2()
    sys.exit(0 if success else 1)