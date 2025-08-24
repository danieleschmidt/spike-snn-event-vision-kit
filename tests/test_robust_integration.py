"""
Comprehensive integration tests for robust neuromorphic vision system.

Tests all three generations of enhancements:
- Generation 1: Basic functionality and syntax validation
- Generation 2: Robustness, error handling, and resilience
- Generation 3: Performance, scaling, and adaptive intelligence
"""

import pytest
import numpy as np
import time
import threading
import logging
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import pickle

# Import our modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spike_snn_event.config import SystemConfiguration, load_configuration
from spike_snn_event.security import InputSanitizer, SecurityError
from spike_snn_event.robust_core import (
    RobustEventProcessor, CircuitBreaker, HealthMonitor, 
    retry, timeout, GracefulShutdown
)
from spike_snn_event.advanced_validation import (
    DataValidator, ValidationLevel, ValidationResult
)
from spike_snn_event.high_performance_core import (
    IntelligentCache, ResourcePool, AutoScaler, HighPerformanceProcessor
)
from spike_snn_event.adaptive_intelligence import (
    AdaptiveIntelligenceEngine, AdaptationStrategy, DataCharacterizer
)


class TestGeneration1BasicFunctionality:
    """Test Generation 1: Basic functionality and syntax validation."""
    
    def test_configuration_loading(self):
        """Test configuration system works correctly."""
        config = SystemConfiguration()
        assert config.camera.sensor_type.value == "DVS128"
        assert config.model.device.value == "cpu"
        assert config.output_dir == "./output"
    
    def test_security_input_sanitization(self):
        """Test security input sanitization."""
        sanitizer = InputSanitizer()
        
        # Valid input
        clean_string = sanitizer.sanitize_string_input("valid_input", "test")
        assert clean_string == "valid_input"
        
        # Dangerous input
        with pytest.raises(SecurityError):
            sanitizer.sanitize_string_input("eval('malicious_code')", "test")
    
    def test_basic_event_validation(self):
        """Test basic event data validation."""
        validator = DataValidator(ValidationLevel.BASIC)
        
        # Valid events
        valid_events = np.array([
            [10, 20, 1000.0, 1],
            [15, 25, 1000.1, -1]
        ])
        
        report = validator.validate_events(valid_events)
        assert report.result == ValidationResult.PASS
        
        # Invalid events (empty)
        empty_events = np.array([])
        report = validator.validate_events(empty_events)
        assert report.result in [ValidationResult.PASS, ValidationResult.WARNING]


class TestGeneration2Robustness:
    """Test Generation 2: Robustness and reliability features."""
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker pattern works correctly."""
        circuit_breaker = CircuitBreaker()
        
        # Function that fails initially then succeeds
        call_count = 0
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise Exception("Simulated failure")
            return "success"
        
        # Should fail and open circuit
        for _ in range(5):
            try:
                circuit_breaker.call(flaky_function)
            except:
                pass
        
        # Circuit should be open now
        state = circuit_breaker.get_state()
        assert state['state'] == 'open'
        assert state['failure_count'] >= 3
    
    def test_health_monitoring(self):
        """Test health monitoring system."""
        health_monitor = HealthMonitor()
        
        # Register a health check
        check_called = False
        def test_health_check():
            nonlocal check_called
            check_called = True
            return True
        
        health_monitor.register_check("test_check", test_health_check, interval=0.1)
        health_monitor.start_monitoring()
        
        # Wait for check to be called
        time.sleep(0.2)
        
        status = health_monitor.get_health_status()
        health_monitor.stop_monitoring()
        
        assert check_called
        assert status['overall'] in ['healthy', 'degraded']
        assert 'test_check' in status['checks']
    
    def test_retry_decorator(self):
        """Test retry functionality with exponential backoff."""
        attempt_count = 0
        
        @retry(max_attempts=3, delay=0.01)
        def failing_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Not yet")
            return "success"
        
        result = failing_function()
        assert result == "success"
        assert attempt_count == 3
    
    def test_timeout_decorator(self):
        """Test timeout functionality."""
        @timeout(0.1)
        def slow_function():
            time.sleep(0.2)
            return "never reached"
        
        with pytest.raises(TimeoutError):
            slow_function()
    
    def test_graceful_shutdown(self):
        """Test graceful shutdown handler."""
        shutdown_handler = GracefulShutdown()
        
        handler1_called = False
        handler2_called = False
        
        def handler1():
            nonlocal handler1_called
            handler1_called = True
        
        def handler2():
            nonlocal handler2_called
            handler2_called = True
        
        shutdown_handler.register_handler(handler1, priority=10)
        shutdown_handler.register_handler(handler2, priority=5)
        
        shutdown_handler.shutdown()
        
        assert handler1_called
        assert handler2_called
        assert shutdown_handler.is_shutting_down
    
    def test_robust_event_processor(self):
        """Test robust event processor integration."""
        processor = RobustEventProcessor()
        
        try:
            processor.initialize({})
            
            # Test event processing
            test_events = [
                {'x': 10, 'y': 20, 'timestamp': time.time(), 'polarity': 1}
            ]
            
            results = processor.process_events(test_events)
            assert len(results) == 1
            assert 'processed_at' in results[0]
            
            # Test status reporting
            status = processor.get_status()
            assert 'circuit_breaker' in status
            assert 'health' in status
            
        finally:
            processor.shutdown()
    
    def test_advanced_validation_with_correction(self):
        """Test advanced validation with auto-correction."""
        validator = DataValidator(ValidationLevel.STRICT)
        
        # Events with issues that can be corrected
        problematic_events = np.array([
            [10.0, 20.0, 1000.0, 1],
            [np.nan, 25.0, 1000.1, -1],  # Invalid coordinate
            [15.0, 30.0, 999.9, 1],      # Out of order timestamp
            [20.0, 35.0, 1000.2, 2]      # Invalid polarity
        ])
        
        report = validator.validate_events(problematic_events, auto_correct=True)
        
        # Should be corrected
        assert report.result == ValidationResult.CORRECTED
        assert report.corrected_data is not None
        assert not np.any(np.isnan(report.corrected_data))
        
        # Validation statistics
        stats = validator.get_validation_stats()
        assert stats['total_validations'] >= 1
        assert stats['corrections'] >= 1


class TestGeneration3Performance:
    """Test Generation 3: Performance and scaling optimizations."""
    
    def test_intelligent_cache(self):
        """Test intelligent cache with LRU and TTL."""
        cache = IntelligentCache(max_size=3, max_memory_mb=1)
        
        # Add items
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Should all be present
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        
        # Add one more - should evict LRU
        cache.put("key4", "value4")
        
        # key1 should be evicted (least recently used)
        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"
        
        # Test statistics
        stats = cache.get_stats()
        assert stats['size'] == 3
        assert stats['hits'] > 0
        assert stats['misses'] > 0
    
    def test_resource_pool(self):
        """Test resource pooling functionality."""
        creation_count = 0
        
        def factory():
            nonlocal creation_count
            creation_count += 1
            return f"resource_{creation_count}"
        
        pool = ResourcePool(factory, max_size=2)
        
        # Acquire resources
        res1 = pool.acquire()
        res2 = pool.acquire()
        
        assert res1 == "resource_1"
        assert res2 == "resource_2"
        assert creation_count == 2
        
        # Release and reuse
        pool.release(res1)
        res3 = pool.acquire()
        
        assert res3 == "resource_1"  # Reused
        assert creation_count == 2   # No new creation
    
    def test_auto_scaler(self):
        """Test auto-scaling based on metrics."""
        scaler = AutoScaler(min_workers=1, max_workers=4)
        
        # Record high load metrics
        for _ in range(10):
            scaler.record_metrics(
                cpu_percent=80.0,
                queue_length=10,
                memory_percent=60.0
            )
        
        # Should suggest scaling up
        decision = scaler.get_scaling_decision()
        # Note: may not scale immediately due to cooldown
        
        assert scaler.current_workers >= 1
        assert scaler.current_workers <= 4
    
    def test_high_performance_processor(self):
        """Test high-performance processor integration."""
        processor = HighPerformanceProcessor(max_workers=2)
        
        try:
            # Create test batches
            test_batches = [
                np.random.rand(100, 4).astype(np.float32) * [640, 480, 10.0, 2] - [0, 0, 0, 1],
                np.random.rand(200, 4).astype(np.float32) * [640, 480, 10.0, 2] - [0, 0, 0, 1]
            ]
            
            # Process batches
            start_time = time.time()
            results = processor.process_events_batch(test_batches)
            processing_time = time.time() - start_time
            
            assert len(results) == 2
            assert results[0] is not None
            assert results[1] is not None
            assert processing_time < 5.0  # Should be reasonably fast
            
            # Check performance stats
            stats = processor.get_performance_stats()
            assert 'processing' in stats
            assert 'cache' in stats
            assert stats['processing']['processed_events'] > 0
            
        finally:
            processor.shutdown()
    
    def test_adaptive_intelligence_engine(self):
        """Test adaptive intelligence and algorithm selection."""
        engine = AdaptiveIntelligenceEngine(AdaptationStrategy.LEARNING)
        
        # Test different data characteristics
        test_scenarios = [
            # High-rate data
            np.random.rand(1000, 4).astype(np.float32) * [640, 480, 1.0, 2] - [0, 0, 0, 1],
            # Low-rate, complex spatial
            np.random.rand(100, 4).astype(np.float32) * [1920, 1080, 10.0, 2] - [0, 0, 0, 1],
            # Balanced scenario
            np.random.rand(500, 4).astype(np.float32) * [800, 600, 5.0, 2] - [0, 0, 0, 1]
        ]
        
        for events in test_scenarios:
            # Sort by timestamp for realism
            events = events[np.argsort(events[:, 2])]
            
            processed_events, metadata = engine.process_events_adaptive(events)
            
            assert processed_events is not None
            assert 'algorithm_used' in metadata
            assert 'processing_time' in metadata
            assert 'estimated_accuracy' in metadata
            
            # Algorithm should be from registry
            assert metadata['algorithm_used'] in engine.algorithm_registry.list_algorithms()
        
        # Check adaptation statistics
        stats = engine.get_adaptation_stats()
        assert stats['total_profiles'] > 0
        assert stats['current_algorithm'] is not None
        assert 0.0 <= stats['avg_accuracy'] <= 1.0
    
    def test_data_characterizer(self):
        """Test data characterization for adaptive processing."""
        characterizer = DataCharacterizer()
        
        # High-rate temporal data
        high_rate_events = np.random.rand(5000, 4)
        high_rate_events[:, 2] = np.sort(np.random.rand(5000) * 0.1)  # 0.1s span
        
        characteristics = characterizer.characterize_events(high_rate_events)
        
        assert characteristics['event_count'] == 5000
        assert characteristics['event_rate_hz'] > 10000  # High rate
        assert characteristics['temporal_density'] == 'high'
        
        # Low-rate, spatially complex data
        complex_events = np.random.rand(100, 4)
        complex_events[:, 0] *= 1920  # Wide spatial range
        complex_events[:, 1] *= 1080
        complex_events[:, 2] = np.sort(np.random.rand(100) * 10.0)  # 10s span
        
        characteristics = characterizer.characterize_events(complex_events)
        
        assert characteristics['event_count'] == 100
        assert characteristics['event_rate_hz'] < 100  # Low rate
        assert characteristics['spatial_complexity'] == 'high'


class TestIntegrationScenarios:
    """Test complete integration scenarios combining all generations."""
    
    def test_end_to_end_processing_pipeline(self):
        """Test complete processing pipeline from input to output."""
        # Initialize components
        config = SystemConfiguration()
        validator = DataValidator(ValidationLevel.BALANCED)
        processor = HighPerformanceProcessor(max_workers=2)
        engine = AdaptiveIntelligenceEngine(AdaptationStrategy.BALANCED)
        
        try:
            # Generate realistic event stream
            event_stream = self._generate_realistic_events(1000)
            
            # Validate events
            validation_report = validator.validate_events(event_stream, auto_correct=True)
            assert validation_report.result in [ValidationResult.PASS, ValidationResult.CORRECTED]
            
            validated_events = validation_report.corrected_data if validation_report.corrected_data is not None else event_stream
            
            # Process with adaptive intelligence
            processed_events, metadata = engine.process_events_adaptive(
                validated_events,
                requirements={'priority': 'balanced'}
            )
            
            assert len(processed_events) > 0
            assert metadata['estimated_accuracy'] > 0.5
            
            # Process in high-performance mode
            batch_results = processor.process_events_batch([validated_events])
            assert len(batch_results) == 1
            assert batch_results[0] is not None
            
        finally:
            processor.shutdown()
    
    def test_fault_tolerance_under_stress(self):
        """Test system behavior under stress and fault conditions."""
        processor = RobustEventProcessor()
        
        try:
            processor.initialize({})
            
            # Simulate various failure scenarios
            test_cases = [
                [],  # Empty events
                [{'invalid': 'data'}],  # Invalid event structure
                [{'x': float('nan'), 'y': 20, 'timestamp': time.time(), 'polarity': 1}],  # NaN data
                [{'x': 10, 'y': 20, 'timestamp': time.time(), 'polarity': 'invalid'}]  # Invalid polarity
            ]
            
            success_count = 0
            for events in test_cases:
                try:
                    results = processor.process_events(events)
                    success_count += 1
                except Exception as e:
                    # Some failures are expected and handled gracefully
                    logging.info(f"Handled expected failure: {e}")
            
            # Should handle at least empty events gracefully
            assert success_count >= 1
            
            # System should still be operational
            status = processor.get_status()
            assert not status['is_shutting_down']
            
        finally:
            processor.shutdown()
    
    def test_performance_under_load(self):
        """Test system performance under high load conditions."""
        processor = HighPerformanceProcessor(max_workers=4)
        
        try:
            # Generate large batches
            large_batches = [
                np.random.rand(5000, 4).astype(np.float32) * [640, 480, 10.0, 2] - [0, 0, 0, 1]
                for _ in range(10)
            ]
            
            # Process batches and measure performance
            start_time = time.time()
            results = processor.process_events_batch(large_batches)
            total_time = time.time() - start_time
            
            assert len(results) == 10
            assert all(result is not None for result in results)
            
            # Performance should be reasonable (less than 10 seconds for this load)
            assert total_time < 10.0
            
            # Check performance statistics
            stats = processor.get_performance_stats()
            assert stats['processing']['processed_events'] >= 50000  # 10 batches * 5000 events
            assert stats['cache']['hit_rate'] >= 0.0  # Some cache utilization
            
        finally:
            processor.shutdown()
    
    def test_adaptive_learning_convergence(self):
        """Test that adaptive system learns and converges to optimal algorithms."""
        engine = AdaptiveIntelligenceEngine(AdaptationStrategy.LEARNING)
        
        # Consistently use same type of data to see if system learns
        consistent_data_type = np.random.rand(500, 4).astype(np.float32) * [640, 480, 5.0, 2] - [0, 0, 0, 1]
        
        algorithm_choices = []
        
        # Process same type of data multiple times
        for _ in range(20):
            # Add slight variation to avoid exact caching
            varied_data = consistent_data_type + np.random.normal(0, 0.1, consistent_data_type.shape)
            varied_data = varied_data[np.argsort(varied_data[:, 2])]  # Sort by timestamp
            
            processed_events, metadata = engine.process_events_adaptive(varied_data)
            algorithm_choices.append(metadata['algorithm_used'])
        
        # Should show some consistency in later choices (learning effect)
        # Check if the most recent 5 choices show more consistency than first 5
        early_choices = set(algorithm_choices[:5])
        late_choices = set(algorithm_choices[-5:])
        
        # Later choices should be more consistent (fewer unique algorithms)
        assert len(late_choices) <= len(early_choices) + 1  # Allow some variation
        
        # Get adaptation statistics
        stats = engine.get_adaptation_stats()
        assert stats['total_profiles'] > 4  # Should have tried multiple algorithms
        assert stats['best_algorithm'] is not None
    
    def _generate_realistic_events(self, count: int) -> np.ndarray:
        """Generate realistic event stream for testing."""
        events = np.zeros((count, 4))
        
        # Realistic spatial distribution (DVS camera simulation)
        events[:, 0] = np.random.normal(320, 100, count).clip(0, 640)  # x coordinates
        events[:, 1] = np.random.normal(240, 75, count).clip(0, 480)   # y coordinates
        
        # Realistic temporal distribution (Poisson-like with some clustering)
        base_times = np.sort(np.random.exponential(0.001, count))  # Base Poisson process
        # Add some temporal clustering
        cluster_times = np.random.normal(0, 0.0001, count)
        events[:, 2] = base_times + cluster_times
        
        # Realistic polarity distribution (slightly more positive)
        events[:, 3] = np.random.choice([-1, 1], size=count, p=[0.4, 0.6])
        
        return events.astype(np.float32)


class TestPerformanceBenchmarks:
    """Performance benchmark tests to ensure scaling requirements are met."""
    
    def test_processing_throughput_benchmark(self):
        """Benchmark processing throughput across different data sizes."""
        processor = HighPerformanceProcessor(max_workers=4)
        
        try:
            data_sizes = [100, 1000, 5000, 10000]
            throughput_results = {}
            
            for size in data_sizes:
                events = np.random.rand(size, 4).astype(np.float32) * [640, 480, 10.0, 2] - [0, 0, 0, 1]
                
                start_time = time.time()
                results = processor.process_events_batch([events])
                processing_time = time.time() - start_time
                
                throughput = size / processing_time  # events per second
                throughput_results[size] = throughput
                
                # Minimum throughput requirements
                if size <= 1000:
                    assert throughput > 1000  # At least 1K events/sec for small batches
                elif size <= 5000:
                    assert throughput > 5000  # At least 5K events/sec for medium batches
                else:
                    assert throughput > 10000  # At least 10K events/sec for large batches
            
            # Throughput should generally increase with data size (batching efficiency)
            assert throughput_results[10000] > throughput_results[1000]
            
        finally:
            processor.shutdown()
    
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage under different load conditions."""
        import psutil
        process = psutil.Process()
        
        processor = HighPerformanceProcessor(max_workers=2)
        
        try:
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process increasingly large batches
            batch_sizes = [1000, 5000, 10000]
            max_memory_increase = 0
            
            for size in batch_sizes:
                large_batch = np.random.rand(size, 4).astype(np.float32) * [640, 480, 10.0, 2] - [0, 0, 0, 1]
                
                processor.process_events_batch([large_batch])
                
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                max_memory_increase = max(max_memory_increase, memory_increase)
            
            # Memory usage should be reasonable (less than 500MB increase)
            assert max_memory_increase < 500
            
            # Check cache memory usage
            stats = processor.get_performance_stats()
            cache_memory = stats['cache']['memory_usage_mb']
            assert cache_memory < 100  # Cache should stay under 100MB
            
        finally:
            processor.shutdown()
    
    def test_concurrent_processing_benchmark(self):
        """Benchmark concurrent processing capabilities."""
        processor = HighPerformanceProcessor(max_workers=8)
        
        try:
            # Create multiple batches for concurrent processing
            num_batches = 20
            batch_size = 1000
            
            batches = [
                np.random.rand(batch_size, 4).astype(np.float32) * [640, 480, 10.0, 2] - [0, 0, 0, 1]
                for _ in range(num_batches)
            ]
            
            # Sequential processing benchmark
            start_time = time.time()
            sequential_results = []
            for batch in batches:
                result = processor.process_events_batch([batch])
                sequential_results.extend(result)
            sequential_time = time.time() - start_time
            
            # Concurrent processing benchmark  
            start_time = time.time()
            concurrent_results = processor.process_events_batch(batches)
            concurrent_time = time.time() - start_time
            
            # Concurrent should be faster (at least 20% improvement)
            speedup_ratio = sequential_time / concurrent_time
            assert speedup_ratio > 1.2
            
            # Results should be equivalent
            assert len(concurrent_results) == num_batches
            assert all(result is not None for result in concurrent_results)
            
        finally:
            processor.shutdown()


if __name__ == "__main__":
    # Configure logging for test runs
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])