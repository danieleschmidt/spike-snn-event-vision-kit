#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spike_snn_event.validation import ValidationError, ValidationResult, SecurityValidator

# Test ValidationError
error = ValidationError("TEST_CODE", "Test message")
print(f"Error code: {error.code}")
print(f"Error message: {error.message}")
print(f"Error has code attribute: {hasattr(error, 'code')}")
print(f"Error dir: {dir(error)}")

# Test ValidationResult
result = ValidationResult()
result.add_error("TEST_ERROR", "Test error message")
print(f"Errors: {len(result.errors)}")
if result.errors:
    first_error = result.errors[0]
    print(f"First error type: {type(first_error)}")
    print(f"First error has code: {hasattr(first_error, 'code')}")
    print(f"First error code: {getattr(first_error, 'code', 'NO CODE ATTR')}")

# Test SecurityValidator
validator = SecurityValidator()
security_result = validator.validate_string_security("'; DROP TABLE users; --")
print(f"Security validation errors: {len(security_result.errors)}")
if security_result.errors:
    error = security_result.errors[0]
    print(f"Security error type: {type(error)}")
    print(f"Security error has code: {hasattr(error, 'code')}")
    print(f"Security error code: {getattr(error, 'code', 'NO CODE ATTR')}")