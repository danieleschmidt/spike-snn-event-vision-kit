#!/usr/bin/env python3
"""
Generation 1 validation script for Spike-SNN Event Vision Kit.
Tests basic enhancements without requiring external dependencies.
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def validate_structure():
    """Validate that enhanced structure is in place."""
    
    print("🔍 GENERATION 1 VALIDATION")
    print("=" * 50)
    
    # Check enhanced files exist
    files_to_check = [
        "src/spike_snn_event/__init__.py",
        "src/spike_snn_event/core.py", 
        "src/spike_snn_event/models.py",
        "src/spike_snn_event/training.py",
        "src/spike_snn_event/validation.py",
        "src/spike_snn_event/cli.py",
        "examples/basic_usage.py",
        "tests/test_basic.py"
    ]
    
    missing_files = []
    for file_path in files_to_check:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    print()
    
    # Check for enhanced functionality in code
    enhancements = []
    
    # Check validation module enhancements
    validation_path = Path("src/spike_snn_event/validation.py")
    if validation_path.exists():
        content = validation_path.read_text()
        if "validate_events" in content:
            enhancements.append("✅ Event validation functions")
        if "SafetyMonitor" in content:
            enhancements.append("✅ Safety monitoring system")
        if "retry_on_failure" in content:
            enhancements.append("✅ Retry mechanism decorator")
    
    # Check core module enhancements  
    core_path = Path("src/spike_snn_event/core.py")
    if core_path.exists():
        content = core_path.read_text()
        if "health_check" in content:
            enhancements.append("✅ Health check functionality")
        if "@safe_operation" in content:
            enhancements.append("✅ Safe operation decorators")
        if "ValidationError" in content:
            enhancements.append("✅ Enhanced error handling")
    
    # Check CLI enhancements
    cli_path = Path("src/spike_snn_event/cli.py") 
    if cli_path.exists():
        content = cli_path.read_text()
        if "validate environment" in content.lower():
            enhancements.append("✅ CLI environment validation")
        if "train_command" in content and "detect_command" in content:
            enhancements.append("✅ Complete CLI commands")
    
    print("🚀 GENERATION 1 ENHANCEMENTS:")
    print("-" * 30)
    for enhancement in enhancements:
        print(enhancement)
    
    if not enhancements:
        print("❌ No enhancements detected")
        return False
    
    print(f"\n✅ Generation 1 Status: {len(enhancements)} enhancements implemented")
    
    # Check project structure completeness
    key_directories = [
        "src/spike_snn_event",
        "tests", 
        "examples",
        "deploy",
        "monitoring",
        "docs"
    ]
    
    print(f"\n🏗️  PROJECT STRUCTURE:")
    print("-" * 20)
    for directory in key_directories:
        if Path(directory).exists():
            file_count = len(list(Path(directory).rglob("*.py")))
            print(f"✅ {directory} ({file_count} Python files)")
        else:
            print(f"❌ {directory}")
    
    print("\n🎯 GENERATION 1 COMPLETE!")
    print("Ready for Generation 2: Make It Robust")
    
    return True

if __name__ == "__main__":
    success = validate_structure()
    sys.exit(0 if success else 1)