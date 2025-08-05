#!/usr/bin/env python3
"""
Test suite for validation module.
"""

import sys
import os
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spike_snn_event.validation import EventValidator, ValidationResult, ValidationError


class TestValidationResult(unittest.TestCase):
    """Test ValidationResult class."""
    
    def setUp(self):
        self.result = ValidationResult()
        
    def test_initial_state(self):
        """Test initial state of ValidationResult."""
        self.assertTrue(self.result.is_valid)
        self.assertEqual(len(self.result.errors), 0)
        self.assertEqual(len(self.result.warnings), 0)
        
    def test_add_error(self):
        """Test adding errors."""
        self.result.add_error("TEST_ERROR", "Test error message", "test_field", "test_value")
        
        self.assertFalse(self.result.is_valid)
        self.assertEqual(len(self.result.errors), 1)
        
        error = self.result.errors[0]
        self.assertEqual(error.code, "TEST_ERROR")
        self.assertEqual(error.message, "Test error message")
        self.assertEqual(error.field, "test_field")
        self.assertEqual(error.value, "test_value")
        
    def test_add_warning(self):
        """Test adding warnings."""
        self.result.add_warning("TEST_WARNING", "Test warning message")
        
        self.assertTrue(self.result.is_valid)  # Warnings don't affect validity
        self.assertEqual(len(self.result.warnings), 1)
        
    def test_format_errors(self):
        """Test error formatting."""
        # Test with no issues
        self.assertEqual(self.result.format_errors(), "No validation issues")
        
        # Add error and warning
        self.result.add_error("ERR1", "Error message", "field1", "value1")
        self.result.add_warning("WARN1", "Warning message", "field2", "value2")
        
        formatted = self.result.format_errors()
        self.assertIn("ERRORS:", formatted)
        self.assertIn("WARNINGS:", formatted)
        self.assertIn("ERR1", formatted)
        self.assertIn("WARN1", formatted)


class TestEventValidator(unittest.TestCase):
    """Test EventValidator class."""
    
    def setUp(self):
        self.validator = EventValidator()
        
    def test_valid_events(self):
        """Test validation of valid events."""
        valid_events = [
            [10.0, 20.0, 0.001, 1],
            [30.0, 40.0, 0.002, -1],
            [50.0, 60.0, 0.003, 1]
        ]
        
        result = self.validator.validate_events(valid_events)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        
    def test_empty_events(self):
        """Test validation of empty event list."""
        result = self.validator.validate_events([])
        self.assertTrue(result.is_valid)  # Empty is valid but generates warning
        self.assertEqual(len(result.warnings), 1)
        self.assertEqual(result.warnings[0].code, "EMPTY_DATA")
        
    def test_invalid_type(self):
        """Test validation of non-list input."""
        result = self.validator.validate_events("not a list")
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].code, "TYPE_ERROR")
        
    def test_malformed_events(self):
        """Test validation of malformed events."""
        malformed_events = [
            [10, 20],  # Too few elements
            [10, 20, 0.1, 1, 5],  # Too many elements
            ["x", 20, 0.1, 1],  # Invalid type
            [10, 20, 0.1, 2],  # Invalid polarity
            [-10, 20, 0.1, 1],  # Negative coordinate
        ]
        
        result = self.validator.validate_events(malformed_events)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
        
    def test_single_event_validation(self):
        """Test validation of single event."""
        # Valid event
        valid_event = [10.0, 20.0, 0.001, 1]
        result = self.validator.validate_single_event(valid_event, 0)
        self.assertTrue(result.is_valid)
        
        # Invalid event - wrong length
        invalid_event = [10.0, 20.0]
        result = self.validator.validate_single_event(invalid_event, 0)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors[0].code, "EVENT_LENGTH_ERROR")
        
        # Invalid event - wrong type
        invalid_event = "not a list"
        result = self.validator.validate_single_event(invalid_event, 0)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors[0].code, "EVENT_TYPE_ERROR")
        
    def test_coordinate_validation(self):
        """Test coordinate validation."""
        # Negative x
        result = self.validator.validate_single_event([-1, 20, 0.1, 1], 0)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(e.code == "X_NEGATIVE_ERROR" for e in result.errors))
        
        # Negative y
        result = self.validator.validate_single_event([10, -1, 0.1, 1], 0)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(e.code == "Y_NEGATIVE_ERROR" for e in result.errors))
        
    def test_timestamp_validation(self):
        """Test timestamp validation."""
        # Negative timestamp
        result = self.validator.validate_single_event([10, 20, -0.1, 1], 0)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(e.code == "T_NEGATIVE_ERROR" for e in result.errors))
        
        # Invalid type
        result = self.validator.validate_single_event([10, 20, "time", 1], 0)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(e.code == "T_TYPE_ERROR" for e in result.errors))
        
    def test_polarity_validation(self):
        """Test polarity validation."""
        # Invalid polarity value
        result = self.validator.validate_single_event([10, 20, 0.1, 2], 0)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(e.code == "P_VALUE_ERROR" for e in result.errors))
        
        # Invalid polarity type
        result = self.validator.validate_single_event([10, 20, 0.1, "polarity"], 0)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(e.code == "P_TYPE_ERROR" for e in result.errors))


class TestValidationIntegration(unittest.TestCase):
    """Integration tests for validation with other components."""
    
    def test_validation_with_camera(self):
        """Test validation integration with camera."""
        try:
            from spike_snn_event.lite_core import DVSCamera
            
            camera = DVSCamera()
            
            # Generate some events
            events = camera._generate_synthetic_events(10)
            
            # Validate them
            validator = EventValidator()
            result = validator.validate_events(events)
            
            # Should be valid since camera generates proper events
            self.assertTrue(result.is_valid)
            
        except ImportError:
            self.skipTest("lite_core not available")
            
    def test_validation_helper(self):
        """Test validation helper function."""
        from spike_snn_event.validation import validate_and_handle
        
        validator = EventValidator()
        valid_events = [[10, 20, 0.1, 1]]
        
        # Should pass in strict mode
        result = validate_and_handle(
            valid_events, 
            validator.validate_events, 
            "test_operation", 
            strict=True
        )
        self.assertTrue(result)
        
        # Test with invalid events in non-strict mode
        invalid_events = [["invalid"]]
        result = validate_and_handle(
            invalid_events,
            validator.validate_events,
            "test_operation",
            strict=False
        )
        self.assertFalse(result)
        
        # Test with invalid events in strict mode (should raise)
        with self.assertRaises(ValueError):
            validate_and_handle(
                invalid_events,
                validator.validate_events,
                "test_operation", 
                strict=True
            )


if __name__ == '__main__':
    unittest.main()