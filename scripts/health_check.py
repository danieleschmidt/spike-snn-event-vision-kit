#!/usr/bin/env python3
"""Health check script for container monitoring."""

import sys
import time
import requests
import json
from pathlib import Path

def check_application_health():
    """Check application health endpoints."""
    try:
        # Check main health endpoint
        response = requests.get('http://localhost:8080/health', timeout=5)
        if response.status_code != 200:
            print(f"Health endpoint returned {response.status_code}")
            return False
        
        health_data = response.json()
        if health_data.get('status') != 'healthy':
            print(f"Application reports unhealthy: {health_data}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def check_dependencies():
    """Check critical dependencies."""
    try:
        # Check if we can import main modules
        import spike_snn_event
        from spike_snn_event.core import DVSCamera
        return True
        
    except ImportError as e:
        print(f"Dependency check failed: {e}")
        return False

def main():
    """Run comprehensive health check."""
    checks = [
        ("Application Health", check_application_health),
        ("Dependencies", check_dependencies)
    ]
    
    for check_name, check_func in checks:
        if not check_func():
            print(f"HEALTH CHECK FAILED: {check_name}")
            sys.exit(1)
    
    print("HEALTH CHECK PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()
