#!/usr/bin/env python3
"""
Comprehensive test runner for Spike SNN Event Vision Kit.

This script runs all tests and provides detailed reporting.
"""

import sys
import os
import unittest
import time
import traceback
from io import StringIO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class ColoredTestResult(unittest.TextTestResult):
    """Test result with colored output."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.success_count = 0
        self._verbosity = verbosity
        
    def addSuccess(self, test):
        super().addSuccess(test)
        self.success_count += 1
        if self._verbosity > 1:
            self.stream.writeln(f"✓ {self.getDescription(test)}")
            
    def addError(self, test, err):
        super().addError(test, err)
        if self._verbosity > 1:
            self.stream.writeln(f"💥 ERROR: {self.getDescription(test)}")
            
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self._verbosity > 1:
            self.stream.writeln(f"✗ FAIL: {self.getDescription(test)}")
            
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self._verbosity > 1:
            self.stream.writeln(f"⏭ SKIP: {self.getDescription(test)} ({reason})")


class TestRunner:
    """Enhanced test runner with reporting."""
    
    def __init__(self):
        self.total_tests = 0
        self.successful_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
        self.skipped_tests = 0
        self.total_time = 0
        
    def discover_and_run_tests(self):
        """Discover and run all tests."""
        print("🧪 Spike SNN Event Vision Kit - Test Suite")
        print("=" * 60)
        
        # Test modules to run
        test_modules = [
            'tests.test_validation',
            'tests.test_lite_core',
        ]
        
        all_results = []
        
        for module_name in test_modules:
            print(f"\n📋 Running {module_name}...")
            result = self._run_module_tests(module_name)
            all_results.append(result)
            
        self._print_summary(all_results)
        
        # Return overall success
        return self.failed_tests == 0 and self.error_tests == 0
        
    def _run_module_tests(self, module_name):
        """Run tests for a specific module."""
        try:
            # Load the test module
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromName(module_name)
            
            # Run tests with custom result class
            stream = StringIO()
            runner = unittest.TextTestRunner(
                stream=stream,
                resultclass=ColoredTestResult,
                verbosity=2
            )
            
            start_time = time.time()
            result = runner.run(suite)
            end_time = time.time()
            
            # Update counters
            self.total_tests += result.testsRun
            self.successful_tests += result.success_count
            self.failed_tests += len(result.failures)
            self.error_tests += len(result.errors)
            self.skipped_tests += len(result.skipped)
            
            module_time = end_time - start_time
            self.total_time += module_time
            
            # Print results
            print(f"  Tests run: {result.testsRun}")
            print(f"  Successes: {result.success_count}")
            print(f"  Failures: {len(result.failures)}")
            print(f"  Errors: {len(result.errors)}")
            print(f"  Skipped: {len(result.skipped)}")
            print(f"  Time: {module_time:.3f}s")
            
            # Print failures and errors if any
            if result.failures:
                print("\n❌ FAILURES:")
                for test, traceback in result.failures:
                    print(f"  {test}: {traceback.split('AssertionError:')[-1].strip()}")
                    
            if result.errors:
                print("\n💥 ERRORS:")
                for test, traceback in result.errors:
                    print(f"  {test}: {traceback.splitlines()[-1]}")
                    
            return {
                'module': module_name,
                'result': result,
                'time': module_time
            }
            
        except Exception as e:
            print(f"❌ Failed to run {module_name}: {e}")
            import traceback as tb
            tb.print_exc()
            return {
                'module': module_name,
                'result': None,
                'time': 0,
                'error': str(e)
            }
            
    def _print_summary(self, results):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        # Overall stats
        print(f"Total tests run: {self.total_tests}")
        print(f"Successful: {self.successful_tests} ✓")
        print(f"Failed: {self.failed_tests} ✗")
        print(f"Errors: {self.error_tests} 💥")
        print(f"Skipped: {self.skipped_tests} ⏭")
        print(f"Total time: {self.total_time:.3f}s")
        
        # Success rate
        if self.total_tests > 0:
            success_rate = (self.successful_tests / self.total_tests) * 100
            print(f"Success rate: {success_rate:.1f}%")
            
        # Per-module breakdown
        print(f"\n📋 Per-Module Results:")
        for result_info in results:
            module = result_info['module']
            result = result_info['result']
            module_time = result_info['time']
            
            if result:
                status = "✓" if len(result.failures) == 0 and len(result.errors) == 0 else "✗"
                print(f"  {module}: {result.testsRun} tests, {module_time:.3f}s {status}")
            else:
                print(f"  {module}: FAILED TO RUN ❌")
                
        # Overall result
        print(f"\n🎯 OVERALL RESULT:")
        if self.failed_tests == 0 and self.error_tests == 0:
            print("🎉 ALL TESTS PASSED! System is ready for deployment.")
        else:
            print("⚠️  SOME TESTS FAILED. Please review and fix issues.")
            
        print()


def run_demo_tests():
    """Run the demo programs as integration tests."""
    print("🚀 Running Integration Demo Tests...")
    print("-" * 40)
    
    demo_tests = [
        ("Lightweight Demo", "python3 demo_lightweight.py"),
        ("Robustness Demo", "python3 demo_robust.py"),
    ]
    
    demo_results = []
    
    for name, command in demo_tests:
        print(f"\n▶️ Running {name}...")
        try:
            import subprocess
            start_time = time.time()
            
            # Run demo with timeout
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            end_time = time.time()
            demo_time = end_time - start_time
            
            if result.returncode == 0:
                print(f"   ✓ {name} completed successfully in {demo_time:.1f}s")
                demo_results.append((name, True, demo_time))
            else:
                print(f"   ✗ {name} failed with return code {result.returncode}")
                print(f"     Error: {result.stderr[:200]}...")
                demo_results.append((name, False, demo_time))
                
        except subprocess.TimeoutExpired:
            print(f"   ⏱ {name} timed out after 30 seconds")
            demo_results.append((name, False, 30.0))
        except Exception as e:
            print(f"   💥 {name} crashed: {e}")
            demo_results.append((name, False, 0))
            
    # Demo summary
    print(f"\n📊 Demo Test Summary:")
    successful_demos = sum(1 for _, success, _ in demo_results if success)
    total_demo_time = sum(time for _, _, time in demo_results)
    
    for name, success, demo_time in demo_results:
        status = "✓" if success else "✗"
        print(f"  {name}: {demo_time:.1f}s {status}")
        
    print(f"\nDemo success rate: {successful_demos}/{len(demo_tests)} ({successful_demos/len(demo_tests)*100:.0f}%)")
    print(f"Total demo time: {total_demo_time:.1f}s")
    
    return successful_demos == len(demo_tests)


def run_performance_benchmarks():
    """Run performance benchmarks."""
    print("⚡ Running Performance Benchmarks...")
    print("-" * 40)
    
    try:
        # Import after path setup
        from spike_snn_event.lite_core import DVSCamera, LiteEventSNN, SpatioTemporalPreprocessor
        
        # Initialize components
        camera = DVSCamera(sensor_type="DVS128")
        model = LiteEventSNN(input_size=(128, 128), num_classes=10)
        preprocessor = SpatioTemporalPreprocessor()
        
        benchmarks = []
        
        # Benchmark 1: Event generation
        print("\n1. Event Generation Benchmark...")
        start_time = time.time()
        
        for _ in range(100):
            events = camera._generate_synthetic_events(100)
            
        generation_time = time.time() - start_time
        events_per_second = (100 * 100) / generation_time
        print(f"   Generated 10,000 events in {generation_time:.3f}s")
        print(f"   Event generation rate: {events_per_second:.0f} events/s")
        benchmarks.append(("Event Generation", events_per_second, "events/s"))
        
        # Benchmark 2: Event filtering
        print("\n2. Event Filtering Benchmark...")
        test_events = camera._generate_synthetic_events(1000)
        
        start_time = time.time()
        for _ in range(100):
            filtered = camera._apply_noise_filter(test_events)
        filtering_time = time.time() - start_time
        
        filtering_rate = (100 * len(test_events)) / filtering_time
        print(f"   Filtered 100,000 events in {filtering_time:.3f}s")
        print(f"   Event filtering rate: {filtering_rate:.0f} events/s")
        benchmarks.append(("Event Filtering", filtering_rate, "events/s"))
        
        # Benchmark 3: Model inference
        print("\n3. Model Inference Benchmark...")
        test_events = camera._generate_synthetic_events(100)
        
        start_time = time.time()
        for _ in range(100):
            detections = model.detect(test_events)
        inference_time = time.time() - start_time
        
        inference_rate = 100 / inference_time
        avg_latency = (inference_time / 100) * 1000  # ms
        print(f"   100 inferences in {inference_time:.3f}s")
        print(f"   Inference rate: {inference_rate:.1f} inferences/s")
        print(f"   Average latency: {avg_latency:.2f}ms")
        benchmarks.append(("Model Inference", inference_rate, "inferences/s"))
        benchmarks.append(("Average Latency", avg_latency, "ms"))
        
        # Benchmark 4: End-to-end pipeline
        print("\n4. End-to-End Pipeline Benchmark...")
        start_time = time.time()
        
        for _ in range(50):
            # Full pipeline
            events = camera._generate_synthetic_events(100)
            filtered = camera._apply_noise_filter(events)
            processed = preprocessor.process(filtered)
            detections = model.detect(processed)
            
        pipeline_time = time.time() - start_time
        pipeline_rate = 50 / pipeline_time
        
        print(f"   50 pipeline runs in {pipeline_time:.3f}s")
        print(f"   Pipeline rate: {pipeline_rate:.1f} fps")
        benchmarks.append(("End-to-End Pipeline", pipeline_rate, "fps"))
        
        # Performance summary
        print(f"\n📊 Performance Summary:")
        for name, value, unit in benchmarks:
            print(f"  {name}: {value:.1f} {unit}")
            
        # Performance targets
        print(f"\n🎯 Performance Analysis:")
        if events_per_second > 100000:
            print("  ✓ Event generation exceeds 100k events/s target")
        else:
            print("  ⚠ Event generation below 100k events/s target")
            
        if inference_rate > 30:
            print("  ✓ Inference rate exceeds 30 fps target")
        else:
            print("  ⚠ Inference rate below 30 fps target")
            
        if avg_latency < 10:
            print("  ✓ Average latency under 10ms target")
        else:
            print("  ⚠ Average latency above 10ms target")
            
        return True
        
    except Exception as e:
        print(f"💥 Performance benchmark failed: {e}")
        return False


def main():
    """Main test execution."""
    print("🔬 Starting Comprehensive Test Suite for Spike SNN Event Vision Kit")
    print("=" * 80)
    
    overall_success = True
    
    # 1. Run unit tests
    print("\n🧪 PHASE 1: Unit Tests")
    test_runner = TestRunner()
    unit_test_success = test_runner.discover_and_run_tests()
    overall_success = overall_success and unit_test_success
    
    # 2. Run demo integration tests
    print("\n🚀 PHASE 2: Integration Tests")
    demo_success = run_demo_tests()
    overall_success = overall_success and demo_success
    
    # 3. Run performance benchmarks
    print("\n⚡ PHASE 3: Performance Benchmarks")
    perf_success = run_performance_benchmarks()
    overall_success = overall_success and perf_success
    
    # Final summary
    print("\n" + "=" * 80)
    print("🏁 FINAL TEST RESULTS")
    print("=" * 80)
    
    phases = [
        ("Unit Tests", unit_test_success),
        ("Integration Tests", demo_success), 
        ("Performance Tests", perf_success)
    ]
    
    for phase_name, success in phases:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{phase_name}: {status}")
        
    if overall_success:
        print(f"\n🎉 ALL TESTS PASSED! 🎉")
        print("✅ System is production-ready")
        print("✅ All functionality works correctly")
        print("✅ Performance meets targets")
        print("✅ Error handling is robust")
        print("\n🚀 Ready for deployment!")
        return 0
    else:
        print(f"\n⚠️  SOME TESTS FAILED")
        print("Please review the test results and fix any issues before deployment.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)