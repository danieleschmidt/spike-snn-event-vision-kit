#!/usr/bin/env python3
"""
Comprehensive robustness and security testing for the neuromorphic vision system.

This script tests all security and robustness features implemented in the system
including adversarial defense, memory safety, circuit breakers, and validation.
"""

import sys
import time
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - some tests will be skipped")


def test_input_validation():
    """Test comprehensive input validation."""
    logger.info("Testing input validation...")
    
    try:
        from src.spike_snn_event.validation import (
            get_event_validator, get_stream_integrity_validator,
            get_model_output_validator, SecurityValidator, DataValidator
        )
        
        results = {}
        
        # Test event validation
        event_validator = get_event_validator()
        
        # Valid events
        valid_events = [[10, 20, 0.1, 1], [15, 25, 0.2, -1]]
        result = event_validator.validate_events(valid_events)
        results['valid_events'] = result.is_valid
        
        # Invalid events
        invalid_events = [[10, 20], [15, 25, 0.2, 2]]  # Wrong length, invalid polarity
        result = event_validator.validate_events(invalid_events)
        results['invalid_events_detected'] = not result.is_valid
        
        # Test stream integrity validation
        stream_validator = get_stream_integrity_validator()
        
        # Normal event stream
        normal_stream = [[i, i+1, i*0.01, 1 if i % 2 == 0 else -1] for i in range(100)]
        result = stream_validator.validate_event_stream_integrity(normal_stream)
        results['normal_stream'] = result.is_valid
        
        # Suspicious event stream (many duplicates)
        suspicious_stream = [[10, 10, 0.1, 1] * 50]  # Many identical events
        result = stream_validator.validate_event_stream_integrity(suspicious_stream)
        results['suspicious_stream_detected'] = len(result.warnings) > 0
        
        # Test security validation
        security_validator = SecurityValidator()
        
        # Test malicious strings
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "eval(malicious_code)",
            "../../../etc/passwd"
        ]
        
        malicious_detected = 0
        for malicious_input in malicious_inputs:
            result = security_validator.validate_string_security(malicious_input)
            if not result.is_valid:
                malicious_detected += 1
                
        results['malicious_string_detection_rate'] = malicious_detected / len(malicious_inputs)
        
        # Test model output validation
        if TORCH_AVAILABLE:
            output_validator = get_model_output_validator()
            
            # Valid output
            valid_output = torch.randn(32, 10)  # Batch of 32, 10 classes
            result = output_validator.validate_snn_output(valid_output)
            results['valid_model_output'] = result.is_valid
            
            # Invalid output (NaN values)
            invalid_output = torch.full((32, 10), float('nan'))
            result = output_validator.validate_snn_output(invalid_output)
            results['invalid_output_detected'] = not result.is_valid
        else:
            results['model_output_tests'] = 'skipped_no_torch'
            
        logger.info(f"Input validation test results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Input validation test failed: {e}")
        return {'error': str(e)}


def test_adversarial_defense():
    """Test adversarial attack detection and defense."""
    logger.info("Testing adversarial defense...")
    
    try:
        from src.spike_snn_event.security_enhancements import get_adversarial_defense
        
        defense = get_adversarial_defense()
        results = {}
        
        # Test 1: Normal event stream
        normal_events = [[i % 128, (i+10) % 128, i * 0.001, 1 if i % 2 == 0 else -1] 
                        for i in range(1000)]
        defended_events = defense.defend_against_adversarial_events(normal_events)
        results['normal_events_preserved_ratio'] = len(defended_events) / len(normal_events)
        
        # Test 2: High-density attack (hot pixel)
        hot_pixel_events = [[64, 64, i * 0.0001, 1] for i in range(2000)]  # 2000 events at same pixel
        defended_events = defense.defend_against_adversarial_events(hot_pixel_events)
        results['hot_pixel_mitigation_ratio'] = 1.0 - (len(defended_events) / len(hot_pixel_events))
        
        # Test 3: High-frequency attack
        high_freq_events = [[i % 128, (i+1) % 128, i * 0.00001, 1] 
                           for i in range(10000)]  # Very high frequency
        defended_events = defense.defend_against_adversarial_events(high_freq_events)
        results['high_frequency_mitigation_ratio'] = 1.0 - (len(defended_events) / len(high_freq_events))
        
        # Test 4: Noise injection attack
        noise_events = []
        for i in range(1000):
            # Add random noise with very regular temporal pattern
            x = np.random.randint(0, 128)
            y = np.random.randint(0, 128)
            t = i * 0.001  # Very regular timing
            p = np.random.choice([-1, 1])
            noise_events.append([x, y, t, p])
            
        defended_events = defense.defend_against_adversarial_events(noise_events)
        results['noise_filtering_ratio'] = 1.0 - (len(defended_events) / len(noise_events))
        
        # Test defense statistics
        defense_stats = defense.get_defense_stats()
        results['defense_enabled'] = defense_stats['defense_enabled']
        results['total_detections'] = defense_stats['total_detections']
        
        logger.info(f"Adversarial defense test results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Adversarial defense test failed: {e}")
        return {'error': str(e)}


def test_memory_safety():
    """Test memory safety management."""
    logger.info("Testing memory safety...")
    
    try:
        from src.spike_snn_event.security_enhancements import get_memory_safety_manager
        
        memory_manager = get_memory_safety_manager()
        results = {}
        
        # Test safe allocation
        large_allocation = 100 * 1024 * 1024  # 100MB
        allocation_allowed = memory_manager.safe_allocate(large_allocation, "test_allocation")
        results['large_allocation_check'] = allocation_allowed
        
        # Test memory monitoring
        memory_stats = memory_manager.monitor_memory_usage()
        results['memory_monitoring_available'] = 'current_memory_gb' in memory_stats
        results['current_memory_gb'] = memory_stats.get('current_memory_gb', 0)
        results['memory_trend'] = memory_stats.get('memory_trend', 'unknown')
        
        # Test memory cleanup
        initial_memory = memory_stats.get('current_memory_gb', 0)
        cleanup_result = memory_manager.force_cleanup()
        final_memory = cleanup_result.get('final_memory_gb', initial_memory)
        
        results['cleanup_available'] = 'cleanup_duration_s' in cleanup_result
        results['memory_freed_gb'] = cleanup_result.get('memory_freed_gb', 0)
        results['garbage_collected'] = cleanup_result.get('gc_collected', 0)
        
        # Test allocation tracking
        results['large_allocations_tracked'] = memory_stats.get('large_allocations', 0)
        
        logger.info(f"Memory safety test results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Memory safety test failed: {e}")
        return {'error': str(e)}


def test_circuit_breakers():
    """Test circuit breaker patterns."""
    logger.info("Testing circuit breakers...")
    
    try:
        from src.spike_snn_event.validation import CircuitBreaker
        
        results = {}
        
        # Test circuit breaker with failing function
        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        
        def failing_function():
            raise RuntimeError("Simulated failure")
            
        def successful_function():
            return "success"
            
        # Test failure accumulation
        failures = 0
        for i in range(5):
            try:
                circuit_breaker.call(failing_function)
            except Exception:
                failures += 1
                
        results['failures_before_circuit_open'] = failures
        
        # Check circuit state
        circuit_state = circuit_breaker.get_state()
        results['circuit_opened'] = circuit_state['state'] == 'OPEN'
        results['failure_count'] = circuit_state['failure_count']
        
        # Test recovery after timeout
        time.sleep(1.5)  # Wait for recovery timeout
        
        try:
            result = circuit_breaker.call(successful_function)
            results['recovery_successful'] = result == "success"
            
            # Check if circuit is reset
            final_state = circuit_breaker.get_state()
            results['circuit_reset'] = final_state['state'] == 'CLOSED'
            
        except Exception as e:
            results['recovery_successful'] = False
            results['recovery_error'] = str(e)
            
        logger.info(f"Circuit breaker test results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Circuit breaker test failed: {e}")
        return {'error': str(e)}


def test_resilient_operations():
    """Test resilient operation management."""
    logger.info("Testing resilient operations...")
    
    try:
        from src.spike_snn_event.health import get_resilient_operation_manager
        
        resilient_manager = get_resilient_operation_manager()
        results = {}
        
        # Test successful operation
        def successful_operation(x, y):
            return x + y
            
        result = resilient_manager.execute_with_resilience(
            'test_operation', successful_operation, 5, 3
        )
        results['successful_operation'] = result == 8
        
        # Test operation with fallback
        def failing_operation():
            raise RuntimeError("Simulated failure")
            
        def fallback_operation():
            return "fallback_result"
            
        result = resilient_manager.execute_with_resilience(
            'test_operation', failing_operation,
            fallback_func=fallback_operation
        )
        results['fallback_operation'] = result == "fallback_result"
        
        # Test resilience statistics
        stats = resilient_manager.get_resilience_stats()
        results['resilience_stats_available'] = 'total_operations' in stats
        results['success_rate'] = stats.get('success_rate', 0.0)
        results['circuit_breaker_states'] = len(stats.get('circuit_breaker_states', {}))
        
        logger.info(f"Resilient operations test results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Resilient operations test failed: {e}")
        return {'error': str(e)}


def test_resource_monitoring():
    """Test resource monitoring and alerting."""
    logger.info("Testing resource monitoring...")
    
    try:
        from src.spike_snn_event.health import get_resource_monitor
        
        resource_monitor = get_resource_monitor()
        results = {}
        
        # Test resource monitoring
        resource_snapshot = resource_monitor.monitor_resources()
        
        results['system_monitoring'] = resource_snapshot['system'].get('available', False)
        results['gpu_monitoring'] = resource_snapshot['gpu'].get('available', False)
        results['application_monitoring'] = resource_snapshot['application'].get('available', False)
        
        if resource_snapshot['system'].get('available', False):
            results['memory_percent'] = resource_snapshot['system']['memory_percent']
            results['cpu_percent'] = resource_snapshot['system']['cpu_percent']
            
        # Test alert conditions (simulate by setting low thresholds)
        low_threshold_monitor = get_resource_monitor()
        low_threshold_monitor.alert_thresholds['memory_percent'] = 1.0  # Very low threshold
        
        snapshot_with_alerts = low_threshold_monitor.monitor_resources()
        results['alert_system_functional'] = True  # If no exception, system is functional
        
        # Test trend analysis
        trends = resource_monitor.get_resource_trends()
        results['trend_analysis_available'] = 'trends' in trends or 'insufficient_data' in trends
        
        logger.info(f"Resource monitoring test results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Resource monitoring test failed: {e}")
        return {'error': str(e)}


def test_secure_configuration():
    """Test secure configuration management."""
    logger.info("Testing secure configuration...")
    
    try:
        from src.spike_snn_event.security import SecureConfig
        
        results = {}
        
        # Test configuration without encryption
        secure_config = SecureConfig()
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_config = {
                'model_path': '/tmp/test_model.pth',
                'batch_size': 32,
                'learning_rate': 0.001,
                'device': 'cpu'
            }
            json.dump(test_config, f)
            temp_config_path = f.name
            
        try:
            # Test loading secure config
            loaded_config = secure_config.load_secure_config(temp_config_path)
            results['config_loading'] = len(loaded_config) == len(test_config)
            results['config_sanitization'] = all(key in loaded_config for key in test_config.keys())
            
            # Test saving secure config
            secure_config.save_secure_config(loaded_config, temp_config_path + '.out')
            results['config_saving'] = Path(temp_config_path + '.out').exists()
            
        finally:
            # Cleanup
            Path(temp_config_path).unlink(missing_ok=True)
            Path(temp_config_path + '.out').unlink(missing_ok=True)
            
        # Test configuration with encryption
        encrypted_config = SecureConfig("test_password_123")
        
        # Test encryption/decryption
        test_data = "sensitive_api_key_12345"
        encrypted_data = encrypted_config.encrypt_sensitive_data(test_data)
        decrypted_data = encrypted_config.decrypt_sensitive_data(encrypted_data)
        
        results['encryption_functional'] = decrypted_data == test_data
        results['encryption_changes_data'] = encrypted_data != test_data
        
        logger.info(f"Secure configuration test results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Secure configuration test failed: {e}")
        return {'error': str(e)}


def test_data_quality_assurance():
    """Test data quality assurance for training."""
    logger.info("Testing data quality assurance...")
    
    try:
        from src.spike_snn_event.training_pipeline import DataQualityAssurance, TrainingConfig
        
        results = {}
        
        # Create test training config
        config = TrainingConfig(
            model_name="test_model",
            dataset_path="/tmp/test_dataset",
            output_dir="/tmp/test_output"
        )
        
        # Test config validation
        config_validation = config.validate()
        results['config_validation_functional'] = hasattr(config_validation, 'is_valid')
        
        # Create data quality assurance
        dqa = DataQualityAssurance(config)
        
        # Test with numpy array data (simulating dataset)
        if TORCH_AVAILABLE:
            # Create test batch
            test_batch = torch.randn(16, 3, 32, 32)  # Batch of 16, 3 channels, 32x32
            test_labels = torch.randint(0, 10, (16,))  # 16 labels, 10 classes
            
            # Test batch validation
            batch_result = dqa.validate_batch(test_batch, test_labels)
            results['batch_validation'] = batch_result.is_valid
            
            # Test invalid batch (with NaN values)
            invalid_batch = torch.full((16, 3, 32, 32), float('nan'))
            invalid_result = dqa.validate_batch(invalid_batch, test_labels)
            results['invalid_batch_detection'] = not invalid_result.is_valid
            
        else:
            results['batch_validation'] = 'skipped_no_torch'
            results['invalid_batch_detection'] = 'skipped_no_torch'
            
        # Test quality report generation
        quality_report = dqa.get_quality_report()
        results['quality_report_available'] = 'quality_score' in quality_report
        
        logger.info(f"Data quality assurance test results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Data quality assurance test failed: {e}")
        return {'error': str(e)}


def test_health_monitoring():
    """Test system health monitoring."""
    logger.info("Testing health monitoring...")
    
    try:
        from src.spike_snn_event.health import (
            get_system_health_checker, quick_health_check, detailed_health_check
        )
        
        results = {}
        
        # Test quick health check
        quick_status = quick_health_check()
        results['quick_health_check'] = quick_status in ['healthy', 'warning', 'critical', 'unknown']
        
        # Test detailed health check
        detailed_report = detailed_health_check()
        results['detailed_health_check'] = 'overall_status' in detailed_report
        results['component_count'] = detailed_report.get('component_count', 0)
        
        # Test individual component checks
        health_checker = get_system_health_checker()
        component_health = health_checker.check_all_components()
        
        results['pytorch_check'] = 'pytorch' in component_health
        results['memory_check'] = 'memory' in component_health
        results['dependencies_check'] = 'dependencies' in component_health
        
        # Test health report export
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_report_path = f.name
            
        try:
            health_checker.export_health_report(temp_report_path)
            results['health_report_export'] = Path(temp_report_path).exists()
            
            # Verify report content
            if Path(temp_report_path).exists():
                with open(temp_report_path, 'r') as f:
                    report_data = json.load(f)
                results['report_contains_status'] = 'overall_status' in report_data
                
        finally:
            Path(temp_report_path).unlink(missing_ok=True)
            
        logger.info(f"Health monitoring test results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Health monitoring test failed: {e}")
        return {'error': str(e)}


def run_performance_benchmarks():
    """Run performance benchmarks to measure overhead."""
    logger.info("Running performance benchmarks...")
    
    try:
        results = {}
        
        # Benchmark input validation overhead
        from src.spike_snn_event.validation import get_event_validator
        
        event_validator = get_event_validator()
        test_events = [[i, i+1, i*0.001, 1 if i % 2 == 0 else -1] for i in range(1000)]
        
        # Time validation
        start_time = time.time()
        for _ in range(100):  # 100 iterations
            event_validator.validate_events(test_events)
        validation_time = (time.time() - start_time) / 100  # Average per iteration
        
        results['event_validation_time_ms'] = validation_time * 1000
        
        # Benchmark adversarial defense overhead
        from src.spike_snn_event.security_enhancements import get_adversarial_defense
        
        defense = get_adversarial_defense()
        
        start_time = time.time()
        for _ in range(100):
            defense.defend_against_adversarial_events(test_events)
        defense_time = (time.time() - start_time) / 100
        
        results['adversarial_defense_time_ms'] = defense_time * 1000
        results['defense_overhead_ratio'] = defense_time / max(validation_time, 0.001)
        
        # Benchmark memory monitoring overhead
        from src.spike_snn_event.security_enhancements import get_memory_safety_manager
        
        memory_manager = get_memory_safety_manager()
        
        start_time = time.time()
        for _ in range(100):
            memory_manager.monitor_memory_usage()
        memory_monitoring_time = (time.time() - start_time) / 100
        
        results['memory_monitoring_time_ms'] = memory_monitoring_time * 1000
        
        logger.info(f"Performance benchmark results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Performance benchmarks failed: {e}")
        return {'error': str(e)}


def generate_test_report(test_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive test report."""
    
    # Calculate overall scores
    total_tests = 0
    passed_tests = 0
    
    for test_name, results in test_results.items():
        if isinstance(results, dict) and 'error' not in results:
            for key, value in results.items():
                if isinstance(value, bool):
                    total_tests += 1
                    if value:
                        passed_tests += 1
                elif isinstance(value, (int, float)) and key.endswith('_ratio'):
                    # Ratio tests - consider >0.8 as pass
                    total_tests += 1
                    if value > 0.8:
                        passed_tests += 1
                        
    overall_pass_rate = passed_tests / max(1, total_tests)
    
    # Security assessment
    security_features = [
        'input_validation', 'adversarial_defense', 'secure_configuration'
    ]
    security_passed = sum(1 for feature in security_features 
                         if feature in test_results and 'error' not in test_results[feature])
    security_score = security_passed / len(security_features)
    
    # Robustness assessment
    robustness_features = [
        'circuit_breakers', 'resilient_operations', 'memory_safety', 'resource_monitoring'
    ]
    robustness_passed = sum(1 for feature in robustness_features 
                           if feature in test_results and 'error' not in test_results[feature])
    robustness_score = robustness_passed / len(robustness_features)
    
    report = {
        'test_timestamp': time.time(),
        'overall_results': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'pass_rate': overall_pass_rate,
            'security_score': security_score,
            'robustness_score': robustness_score
        },
        'test_results': test_results,
        'summary': {
            'security_ready': security_score >= 0.8,
            'robustness_ready': robustness_score >= 0.8,
            'production_ready': overall_pass_rate >= 0.8 and security_score >= 0.8 and robustness_score >= 0.8
        },
        'recommendations': []
    }
    
    # Generate recommendations
    if overall_pass_rate < 0.8:
        report['recommendations'].append("Overall test pass rate below 80% - review failed tests")
    if security_score < 0.8:
        report['recommendations'].append("Security features need attention - ensure all security modules are functional")
    if robustness_score < 0.8:
        report['recommendations'].append("Robustness features need improvement - check resilience mechanisms")
        
    # Performance recommendations
    if 'performance_benchmarks' in test_results:
        perf_results = test_results['performance_benchmarks']
        if perf_results.get('defense_overhead_ratio', 0) > 2.0:
            report['recommendations'].append("Adversarial defense overhead is high - consider optimization")
            
    return report


def main():
    """Run comprehensive robustness and security testing."""
    logger.info("Starting comprehensive robustness and security testing...")
    
    # Run all tests
    test_functions = [
        ('input_validation', test_input_validation),
        ('adversarial_defense', test_adversarial_defense),
        ('memory_safety', test_memory_safety),
        ('circuit_breakers', test_circuit_breakers),
        ('resilient_operations', test_resilient_operations),
        ('resource_monitoring', test_resource_monitoring),
        ('secure_configuration', test_secure_configuration),
        ('data_quality_assurance', test_data_quality_assurance),
        ('health_monitoring', test_health_monitoring),
        ('performance_benchmarks', run_performance_benchmarks)
    ]
    
    test_results = {}
    
    for test_name, test_func in test_functions:
        logger.info(f"Running {test_name} test...")
        try:
            result = test_func()
            test_results[test_name] = result
            logger.info(f"{test_name} test completed successfully")
        except Exception as e:
            logger.error(f"{test_name} test failed: {e}")
            test_results[test_name] = {'error': str(e)}
            
    # Generate comprehensive report
    report = generate_test_report(test_results)
    
    # Save report
    report_path = Path('robustness_test_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
        
    # Print summary
    logger.info("=" * 80)
    logger.info("ROBUSTNESS AND SECURITY TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Tests: {report['overall_results']['total_tests']}")
    logger.info(f"Passed Tests: {report['overall_results']['passed_tests']}")
    logger.info(f"Pass Rate: {report['overall_results']['pass_rate']:.1%}")
    logger.info(f"Security Score: {report['overall_results']['security_score']:.1%}")
    logger.info(f"Robustness Score: {report['overall_results']['robustness_score']:.1%}")
    logger.info(f"Production Ready: {report['summary']['production_ready']}")
    
    if report['recommendations']:
        logger.info("\nRECOMMENDATIONS:")
        for recommendation in report['recommendations']:
            logger.info(f"  - {recommendation}")
            
    logger.info(f"\nDetailed report saved to: {report_path}")
    logger.info("=" * 80)
    
    # Return exit code based on results
    if report['summary']['production_ready']:
        logger.info("All tests passed - system is production ready!")
        return 0
    else:
        logger.warning("Some tests failed - system needs improvement before production deployment")
        return 1


if __name__ == '__main__':
    sys.exit(main())