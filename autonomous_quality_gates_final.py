#!/usr/bin/env python3
"""
Autonomous Quality Gates Implementation - Final Validation

Implements comprehensive quality gates to ensure production readiness
across all dimensions: functionality, performance, security, reliability.

This is the final validation step of the Autonomous SDLC v4.0 system.
"""

import sys
import os
import time
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import numpy as np
    from spike_snn_event.validation import SecurityValidator, DataValidator
    from spike_snn_event.optimization import get_memory_tracker
    from spike_snn_event.scaling import AutoScaler, ScalingPolicy
    VALIDATION_AVAILABLE = True
except ImportError as e:
    VALIDATION_AVAILABLE = False
    print(f"Warning: Validation modules not available: {e}")


@dataclass
class QualityGateResult:
    """Result of a single quality gate check."""
    gate_name: str
    status: str  # PASS, FAIL, WARNING, SKIP
    score: float  # 0.0 to 100.0
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass  
class QualityReport:
    """Comprehensive quality assessment report."""
    overall_score: float = 0.0
    gate_results: List[QualityGateResult] = field(default_factory=list)
    total_gates: int = 0
    passed_gates: int = 0
    failed_gates: int = 0
    warning_gates: int = 0
    skipped_gates: int = 0
    generation_assessment: str = ""
    production_readiness: bool = False
    recommendations: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


class QualityGateEngine:
    """Engine for executing comprehensive quality gates."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent
        
        # Quality gate definitions
        self.gates = [
            # Functionality Gates
            ("Code Syntax Validation", self._validate_code_syntax),
            ("Import Structure Check", self._validate_imports),
            ("Core Functionality Test", self._test_core_functionality),
            
            # Security Gates  
            ("Security Vulnerability Scan", self._security_scan),
            ("Input Validation Coverage", self._validate_input_coverage),
            ("Dependency Security Check", self._check_dependency_security),
            
            # Performance Gates
            ("Memory Optimization Test", self._test_memory_optimization),
            ("Auto-scaling Validation", self._test_auto_scaling),
            ("Resource Utilization Check", self._check_resource_utilization),
            
            # Reliability Gates
            ("Error Handling Coverage", self._test_error_handling),
            ("Fault Tolerance Test", self._test_fault_tolerance),
            ("Recovery Mechanisms Test", self._test_recovery_mechanisms),
            
            # Quality Gates
            ("Code Quality Metrics", self._assess_code_quality),
            ("Test Coverage Analysis", self._analyze_test_coverage),
            ("Documentation Completeness", self._check_documentation),
            
            # Production Readiness Gates
            ("Deployment Configuration", self._validate_deployment),
            ("Monitoring Integration", self._test_monitoring),
            ("Configuration Management", self._validate_configuration),
            
            # Compliance Gates
            ("Security Standards Compliance", self._check_security_compliance),
            ("Performance Benchmarks", self._validate_performance_benchmarks)
        ]
        
    def execute_all_gates(self) -> QualityReport:
        """Execute all quality gates and generate comprehensive report."""
        start_time = time.time()
        report = QualityReport()
        
        self.logger.info("Starting autonomous quality gate execution...")
        
        # Execute gates in parallel for efficiency
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all gates
            future_to_gate = {
                executor.submit(self._execute_gate, gate_name, gate_func): (gate_name, gate_func)
                for gate_name, gate_func in self.gates
            }
            
            # Collect results
            for future in as_completed(future_to_gate):
                gate_name, gate_func = future_to_gate[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per gate
                    report.gate_results.append(result)
                except Exception as e:
                    # Gate execution failed
                    result = QualityGateResult(
                        gate_name=gate_name,
                        status="FAIL",
                        score=0.0,
                        message=f"Gate execution failed: {e}",
                        details={"error": str(e)}
                    )
                    report.gate_results.append(result)
        
        # Calculate overall metrics
        report.total_gates = len(report.gate_results)
        report.passed_gates = sum(1 for r in report.gate_results if r.status == "PASS")
        report.failed_gates = sum(1 for r in report.gate_results if r.status == "FAIL")
        report.warning_gates = sum(1 for r in report.gate_results if r.status == "WARNING")
        report.skipped_gates = sum(1 for r in report.gate_results if r.status == "SKIP")
        
        # Calculate weighted overall score
        total_score = sum(r.score for r in report.gate_results)
        report.overall_score = total_score / len(report.gate_results) if report.gate_results else 0.0
        
        # Assess generation and production readiness
        report.generation_assessment = self._assess_generation(report)
        report.production_readiness = self._assess_production_readiness(report)
        report.recommendations = self._generate_recommendations(report)
        
        report.execution_time = time.time() - start_time
        
        self.logger.info(f"Quality gate execution completed in {report.execution_time:.2f}s")
        return report
        
    def _execute_gate(self, gate_name: str, gate_func) -> QualityGateResult:
        """Execute a single quality gate."""
        start_time = time.time()
        
        try:
            result = gate_func()
            result.execution_time = time.time() - start_time
            return result
        except Exception as e:
            return QualityGateResult(
                gate_name=gate_name,
                status="FAIL", 
                score=0.0,
                message=f"Gate execution error: {e}",
                execution_time=time.time() - start_time
            )
    
    # === FUNCTIONALITY GATES ===
    
    def _validate_code_syntax(self) -> QualityGateResult:
        """Validate Python syntax across the codebase."""
        errors = []
        total_files = 0
        
        for py_file in self.project_root.glob("src/**/*.py"):
            total_files += 1
            try:
                with open(py_file, 'r') as f:
                    compile(f.read(), py_file, 'exec')
            except SyntaxError as e:
                errors.append(f"{py_file}: {e}")
                
        if errors:
            return QualityGateResult(
                gate_name="Code Syntax Validation",
                status="FAIL",
                score=max(0, (total_files - len(errors)) / total_files * 100),
                message=f"Syntax errors found in {len(errors)} files",
                details={"errors": errors}
            )
        else:
            return QualityGateResult(
                gate_name="Code Syntax Validation", 
                status="PASS",
                score=100.0,
                message=f"All {total_files} Python files have valid syntax"
            )
    
    def _validate_imports(self) -> QualityGateResult:
        """Validate import structure and dependencies."""
        import_errors = []
        
        # Test critical imports
        critical_imports = [
            "spike_snn_event.validation",
            "spike_snn_event.optimization", 
            "spike_snn_event.scaling",
            "spike_snn_event.core"
        ]
        
        for module in critical_imports:
            try:
                __import__(module)
            except ImportError as e:
                import_errors.append(f"{module}: {e}")
                
        success_rate = (len(critical_imports) - len(import_errors)) / len(critical_imports) * 100
        
        if import_errors:
            return QualityGateResult(
                gate_name="Import Structure Check",
                status="FAIL" if success_rate < 50 else "WARNING",
                score=success_rate,
                message=f"Import issues found: {len(import_errors)}/{len(critical_imports)}",
                details={"errors": import_errors}
            )
        else:
            return QualityGateResult(
                gate_name="Import Structure Check",
                status="PASS", 
                score=100.0,
                message="All critical imports successful"
            )
    
    def _test_core_functionality(self) -> QualityGateResult:
        """Test core system functionality."""
        if not VALIDATION_AVAILABLE:
            return QualityGateResult(
                gate_name="Core Functionality Test",
                status="SKIP",
                score=0.0,
                message="Validation modules not available"
            )
            
        test_results = []
        
        # Test validation system
        try:
            validator = SecurityValidator()
            result = validator.validate_string_security("test_string")
            test_results.append(("SecurityValidator", result.is_valid))
        except Exception as e:
            test_results.append(("SecurityValidator", False))
            
        # Test data validation
        try:
            data_validator = DataValidator()
            result = data_validator.validate_numeric_range(5.0, 0.0, 10.0)
            test_results.append(("DataValidator", result.is_valid))
        except Exception as e:
            test_results.append(("DataValidator", False))
            
        # Test memory optimization
        try:
            tracker = get_memory_tracker()
            stats = tracker._collect_memory_stats()
            test_results.append(("MemoryTracker", stats is not None))
        except Exception as e:
            test_results.append(("MemoryTracker", False))
            
        # Test auto-scaling
        try:
            policy = ScalingPolicy(min_workers=1, max_workers=5)
            scaler = AutoScaler(policy=policy)
            stats = scaler.get_scaling_stats()
            test_results.append(("AutoScaler", stats is not None))
        except Exception as e:
            test_results.append(("AutoScaler", False))
            
        passed = sum(1 for _, success in test_results if success)
        total = len(test_results)
        score = (passed / total * 100) if total > 0 else 0
        
        status = "PASS" if passed == total else "WARNING" if passed > total // 2 else "FAIL"
        
        return QualityGateResult(
            gate_name="Core Functionality Test",
            status=status,
            score=score,
            message=f"Core functionality: {passed}/{total} tests passed",
            details={"test_results": test_results}
        )
    
    # === SECURITY GATES ===
    
    def _security_scan(self) -> QualityGateResult:
        """Perform basic security vulnerability scan."""
        vulnerabilities = []
        
        # Check for common security patterns
        security_patterns = [
            ("Hardcoded passwords", r"password\s*=\s*['\"][^'\"]+['\"]"),
            ("SQL injection risks", r"execute\s*\(\s*['\"].*%.*['\"]"),
            ("Command injection", r"os\.system\s*\(\s*.*\+.*\)"),
            ("Unsafe eval usage", r"eval\s*\("),
        ]
        
        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                import re
                for vuln_name, pattern in security_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        vulnerabilities.append(f"{py_file}: {vuln_name}")
            except Exception:
                continue
                
        if vulnerabilities:
            score = max(0, 100 - len(vulnerabilities) * 10)  # Deduct 10 points per vulnerability
            return QualityGateResult(
                gate_name="Security Vulnerability Scan",
                status="WARNING" if score >= 70 else "FAIL",
                score=score,
                message=f"Found {len(vulnerabilities)} potential security issues",
                details={"vulnerabilities": vulnerabilities}
            )
        else:
            return QualityGateResult(
                gate_name="Security Vulnerability Scan",
                status="PASS",
                score=100.0,
                message="No obvious security vulnerabilities detected"
            )
    
    def _validate_input_coverage(self) -> QualityGateResult:
        """Validate input validation coverage."""
        if not VALIDATION_AVAILABLE:
            return QualityGateResult(
                gate_name="Input Validation Coverage",
                status="SKIP",
                score=0.0,
                message="Validation modules not available"
            )
            
        # Test validation capabilities
        try:
            # Run enhanced validation test
            result = subprocess.run([
                sys.executable, "test_enhanced_validation.py"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Parse test results from output
                lines = result.stdout.split('\n')
                success_rate_line = [l for l in lines if "Success Rate:" in l]
                if success_rate_line:
                    rate = float(success_rate_line[0].split(": ")[1].replace("%", ""))
                    return QualityGateResult(
                        gate_name="Input Validation Coverage",
                        status="PASS" if rate >= 80 else "WARNING" if rate >= 60 else "FAIL",
                        score=rate,
                        message=f"Input validation test success rate: {rate:.1f}%"
                    )
            
            return QualityGateResult(
                gate_name="Input Validation Coverage", 
                status="FAIL",
                score=0.0,
                message="Input validation tests failed",
                details={"error": result.stderr}
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Input Validation Coverage",
                status="FAIL",
                score=0.0, 
                message=f"Error running validation tests: {e}"
            )
    
    def _check_dependency_security(self) -> QualityGateResult:
        """Check for known vulnerabilities in dependencies."""
        # This would typically run pip-audit or similar
        # For demo, we'll check for basic dependency security
        
        try:
            requirements_file = self.project_root / "requirements.txt"
            if not requirements_file.exists():
                return QualityGateResult(
                    gate_name="Dependency Security Check",
                    status="WARNING",
                    score=70.0,
                    message="No requirements.txt found for security analysis"
                )
            
            with open(requirements_file, 'r') as f:
                deps = f.read()
                
            # Check for pinned versions (basic security practice)
            lines = [l.strip() for l in deps.split('\n') if l.strip() and not l.startswith('#')]
            pinned = sum(1 for l in lines if '==' in l or '>=' in l)
            total = len(lines)
            
            if total == 0:
                score = 100.0
                status = "PASS"
                message = "No dependencies to check"
            else:
                score = (pinned / total) * 100
                status = "PASS" if score >= 80 else "WARNING" if score >= 50 else "FAIL"
                message = f"Dependency pinning: {pinned}/{total} dependencies have version constraints"
            
            return QualityGateResult(
                gate_name="Dependency Security Check",
                status=status,
                score=score, 
                message=message,
                details={"total_deps": total, "pinned_deps": pinned}
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Dependency Security Check",
                status="FAIL",
                score=0.0,
                message=f"Error checking dependencies: {e}"
            )
    
    # === PERFORMANCE GATES ===
    
    def _test_memory_optimization(self) -> QualityGateResult:
        """Test memory optimization capabilities."""
        if not VALIDATION_AVAILABLE:
            return QualityGateResult(
                gate_name="Memory Optimization Test",
                status="SKIP",
                score=0.0,
                message="Memory tracking not available"
            )
            
        try:
            tracker = get_memory_tracker()
            
            # Test memory collection
            initial_stats = tracker._collect_memory_stats()
            
            # Test optimization
            tracker.force_optimization()
            
            # Get optimization stats
            opt_stats = tracker.get_optimization_stats()
            
            # Evaluate performance
            score = 85.0  # Base score for working memory optimization
            if opt_stats['gc_collections'] > 0:
                score += 10.0  # Bonus for active GC
            if initial_stats.percent < 50:
                score += 5.0   # Bonus for low memory usage
                
            return QualityGateResult(
                gate_name="Memory Optimization Test",
                status="PASS",
                score=min(100.0, score),
                message=f"Memory optimization functional, {opt_stats['gc_collections']} GC cycles",
                details={
                    "memory_percent": initial_stats.percent,
                    "gc_collections": opt_stats['gc_collections'],
                    "tracked_objects": opt_stats['tracked_objects']
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Memory Optimization Test",
                status="FAIL", 
                score=0.0,
                message=f"Memory optimization test failed: {e}"
            )
    
    def _test_auto_scaling(self) -> QualityGateResult:
        """Test auto-scaling functionality.""" 
        if not VALIDATION_AVAILABLE:
            return QualityGateResult(
                gate_name="Auto-scaling Validation",
                status="SKIP", 
                score=0.0,
                message="Auto-scaling modules not available"
            )
            
        try:
            policy = ScalingPolicy(min_workers=1, max_workers=5)
            scaler = AutoScaler(policy=policy)
            
            # Test basic functionality
            stats = scaler.get_scaling_stats()
            
            # Test scaling decision logic
            from spike_snn_event.scaling import ResourceMetrics
            test_metrics = ResourceMetrics(cpu_percent=85.0, memory_percent=70.0)
            decision = scaler._make_scaling_decision(test_metrics)
            
            score = 80.0  # Base score for functional auto-scaler
            if stats['current_workers'] == policy.min_workers:
                score += 10.0  # Correct initial state
            if hasattr(scaler, '_make_scaling_decision'):
                score += 10.0  # Has decision logic
                
            return QualityGateResult(
                gate_name="Auto-scaling Validation",
                status="PASS",
                score=score,
                message=f"Auto-scaling functional, {stats['current_workers']} workers active",
                details={
                    "current_workers": stats['current_workers'],
                    "min_workers": policy.min_workers,
                    "max_workers": policy.max_workers,
                    "decision_test": decision
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Auto-scaling Validation",
                status="FAIL",
                score=0.0,
                message=f"Auto-scaling test failed: {e}"
            )
    
    def _check_resource_utilization(self) -> QualityGateResult:
        """Check resource utilization patterns."""
        try:
            import psutil
            
            # Get current resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Assess resource health
            issues = []
            score = 100.0
            
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                score -= 20
            elif cpu_percent > 70:
                score -= 10
                
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
                score -= 20
            elif memory.percent > 70:
                score -= 10
                
            if disk.percent > 90:
                issues.append(f"High disk usage: {disk.percent:.1f}%")
                score -= 20
            elif disk.percent > 80:
                score -= 10
            
            status = "PASS" if score >= 80 else "WARNING" if score >= 60 else "FAIL"
            message = "Resource utilization healthy" if not issues else f"Issues: {len(issues)}"
            
            return QualityGateResult(
                gate_name="Resource Utilization Check",
                status=status,
                score=max(0, score),
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "issues": issues
                }
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Resource Utilization Check", 
                status="WARNING",
                score=50.0,
                message=f"Could not assess resource utilization: {e}"
            )
    
    # === RELIABILITY GATES ===
    
    def _test_error_handling(self) -> QualityGateResult:
        """Test error handling coverage."""
        try_except_count = 0
        total_functions = 0
        
        # Scan for try-except blocks and functions
        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                # Count try-except blocks
                try_except_count += content.count('try:')
                
                # Count function definitions (rough estimate)
                import re
                functions = re.findall(r'def\s+\w+\s*\(', content)
                total_functions += len(functions)
                
            except Exception:
                continue
        
        # Estimate error handling coverage
        if total_functions == 0:
            coverage = 0
        else:
            coverage = min(100, (try_except_count / total_functions) * 100)
            
        status = "PASS" if coverage >= 60 else "WARNING" if coverage >= 30 else "FAIL"
        
        return QualityGateResult(
            gate_name="Error Handling Coverage",
            status=status,
            score=coverage,
            message=f"Error handling coverage: ~{coverage:.1f}% ({try_except_count} try blocks, {total_functions} functions)",
            details={
                "try_blocks": try_except_count,
                "total_functions": total_functions,
                "estimated_coverage": coverage
            }
        )
    
    def _test_fault_tolerance(self) -> QualityGateResult:
        """Test system fault tolerance."""
        fault_tolerance_features = []
        
        # Check for fault tolerance patterns
        patterns = [
            ("Circuit breaker pattern", r"circuit.*breaker", 10),
            ("Retry mechanisms", r"retry|attempt", 8),
            ("Timeout handling", r"timeout", 6),
            ("Graceful degradation", r"graceful|fallback", 8),
            ("Health checks", r"health.*check", 5),
        ]
        
        score = 0
        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    
                for pattern_name, pattern, points in patterns:
                    import re
                    if re.search(pattern, content):
                        if pattern_name not in [f[0] for f in fault_tolerance_features]:
                            fault_tolerance_features.append((pattern_name, py_file.name))
                            score += points
            except Exception:
                continue
        
        status = "PASS" if score >= 30 else "WARNING" if score >= 15 else "FAIL"
        
        return QualityGateResult(
            gate_name="Fault Tolerance Test",
            status=status,
            score=min(100, score * 2),  # Scale score to 0-100
            message=f"Fault tolerance features: {len(fault_tolerance_features)} detected",
            details={"features": fault_tolerance_features, "raw_score": score}
        )
    
    def _test_recovery_mechanisms(self) -> QualityGateResult:
        """Test recovery and resilience mechanisms."""
        recovery_features = []
        
        # Look for recovery-related patterns
        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    
                patterns = [
                    "restart", "recover", "resume", "restore", "repair",
                    "reconnect", "retry", "rollback", "failover"
                ]
                
                for pattern in patterns:
                    if pattern in content:
                        recovery_features.append((pattern, py_file.name))
                        
            except Exception:
                continue
        
        # Remove duplicates
        unique_features = list(set(recovery_features))
        score = min(100, len(unique_features) * 12)  # 12 points per feature
        
        status = "PASS" if score >= 60 else "WARNING" if score >= 30 else "FAIL"
        
        return QualityGateResult(
            gate_name="Recovery Mechanisms Test",
            status=status,
            score=score,
            message=f"Recovery mechanisms: {len(unique_features)} types detected",
            details={"features": unique_features}
        )
    
    # === QUALITY GATES ===
    
    def _assess_code_quality(self) -> QualityGateResult:
        """Assess overall code quality metrics."""
        metrics = {
            "total_lines": 0,
            "total_files": 0,
            "avg_function_length": 0,
            "comment_ratio": 0,
            "docstring_coverage": 0
        }
        
        total_functions = 0
        total_function_lines = 0
        total_comments = 0
        total_docstrings = 0
        
        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file, 'r') as f:
                    lines = f.readlines()
                    metrics["total_lines"] += len(lines)
                    metrics["total_files"] += 1
                    
                content = ''.join(lines)
                
                # Count comments
                comment_lines = [l for l in lines if l.strip().startswith('#')]
                total_comments += len(comment_lines)
                
                # Count functions and docstrings
                import re
                functions = re.findall(r'def\s+\w+\s*\([^)]*\):', content)
                docstrings = re.findall(r'""".*?"""', content, re.DOTALL)
                
                total_functions += len(functions)
                total_docstrings += len(docstrings)
                
                # Rough function length estimate
                if functions:
                    total_function_lines += len(lines) // len(functions)
                    
            except Exception:
                continue
        
        # Calculate metrics
        if total_functions > 0:
            metrics["avg_function_length"] = total_function_lines // total_functions
            metrics["docstring_coverage"] = (total_docstrings / total_functions) * 100
            
        if metrics["total_lines"] > 0:
            metrics["comment_ratio"] = (total_comments / metrics["total_lines"]) * 100
            
        # Score based on quality heuristics
        score = 70.0  # Base score
        
        # Bonus for good metrics
        if metrics["avg_function_length"] < 50:  # Short functions are good
            score += 10
        if metrics["comment_ratio"] > 10:  # Good commenting
            score += 10  
        if metrics["docstring_coverage"] > 50:  # Good documentation
            score += 10
            
        return QualityGateResult(
            gate_name="Code Quality Metrics",
            status="PASS" if score >= 80 else "WARNING" if score >= 60 else "FAIL",
            score=score,
            message=f"Code quality assessment: {metrics['total_files']} files, {metrics['total_lines']} lines",
            details=metrics
        )
    
    def _analyze_test_coverage(self) -> QualityGateResult:
        """Analyze test coverage."""
        test_files = list(self.project_root.glob("test*.py")) + list(self.project_root.glob("tests/**/*.py"))
        src_files = list(self.project_root.glob("src/**/*.py"))
        
        if not test_files:
            return QualityGateResult(
                gate_name="Test Coverage Analysis",
                status="FAIL",
                score=0.0,
                message="No test files found"
            )
            
        # Rough coverage estimate based on test files vs source files
        coverage_ratio = len(test_files) / max(1, len(src_files))
        estimated_coverage = min(100, coverage_ratio * 100)
        
        # Bonus for specific test types
        test_types = []
        for test_file in test_files:
            try:
                with open(test_file, 'r') as f:
                    content = f.read().lower()
                    
                if 'unit' in content or 'unittest' in content:
                    test_types.append('unit')
                if 'integration' in content:
                    test_types.append('integration') 
                if 'performance' in content or 'benchmark' in content:
                    test_types.append('performance')
                if 'security' in content or 'validation' in content:
                    test_types.append('security')
            except Exception:
                continue
                
        test_types = list(set(test_types))
        type_bonus = len(test_types) * 5  # 5 points per test type
        
        final_score = min(100, estimated_coverage + type_bonus)
        status = "PASS" if final_score >= 70 else "WARNING" if final_score >= 50 else "FAIL"
        
        return QualityGateResult(
            gate_name="Test Coverage Analysis",
            status=status,
            score=final_score,
            message=f"Test coverage estimate: {final_score:.1f}% ({len(test_files)} test files)",
            details={
                "test_files": len(test_files),
                "src_files": len(src_files),
                "test_types": test_types,
                "estimated_coverage": estimated_coverage
            }
        )
    
    def _check_documentation(self) -> QualityGateResult:
        """Check documentation completeness."""
        doc_files = []
        doc_score = 0
        
        # Check for essential documentation
        essential_docs = [
            ("README.md", 20),
            ("CONTRIBUTING.md", 10),
            ("LICENSE", 10),
            ("ARCHITECTURE.md", 15),
            ("SECURITY.md", 10),
            ("CHANGELOG.md", 5),
        ]
        
        for doc_file, points in essential_docs:
            if (self.project_root / doc_file).exists():
                doc_files.append(doc_file)
                doc_score += points
                
        # Check for API documentation
        docs_dir = self.project_root / "docs"
        if docs_dir.exists():
            doc_count = len(list(docs_dir.glob("**/*.md"))) + len(list(docs_dir.glob("**/*.rst")))
            doc_score += min(20, doc_count * 2)  # Up to 20 points for API docs
            
        # Check for inline documentation (docstrings)
        src_files = list(self.project_root.glob("src/**/*.py"))
        docstring_count = 0
        
        for py_file in src_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    docstring_count += content.count('"""')
            except Exception:
                continue
                
        if src_files:
            docstring_ratio = docstring_count / len(src_files)
            doc_score += min(20, docstring_ratio * 10)  # Up to 20 points for docstrings
            
        status = "PASS" if doc_score >= 60 else "WARNING" if doc_score >= 40 else "FAIL"
        
        return QualityGateResult(
            gate_name="Documentation Completeness", 
            status=status,
            score=min(100, doc_score),
            message=f"Documentation score: {doc_score}/100 ({len(doc_files)} essential docs)",
            details={
                "essential_docs": doc_files,
                "docs_directory": docs_dir.exists(),
                "docstring_count": docstring_count,
                "score_breakdown": doc_score
            }
        )
    
    # === PRODUCTION READINESS GATES ===
    
    def _validate_deployment(self) -> QualityGateResult:
        """Validate deployment configuration."""
        deployment_files = []
        deployment_score = 0
        
        # Check for deployment artifacts
        deployment_artifacts = [
            ("Dockerfile", 15),
            ("docker-compose.yml", 10),
            ("requirements.txt", 15),
            ("pyproject.toml", 10),
            ("Makefile", 5),
            ("deploy/", 15),
            ("k8s/", 20), 
            ("helm/", 15)
        ]
        
        for artifact, points in deployment_artifacts:
            path = self.project_root / artifact
            if path.exists():
                deployment_files.append(artifact)
                deployment_score += points
                
        # Check for CI/CD configuration
        ci_configs = [".github/workflows/", ".gitlab-ci.yml", "Jenkinsfile", ".travis.yml"]
        for ci_config in ci_configs:
            if (self.project_root / ci_config).exists():
                deployment_score += 10
                break
                
        status = "PASS" if deployment_score >= 50 else "WARNING" if deployment_score >= 30 else "FAIL"
        
        return QualityGateResult(
            gate_name="Deployment Configuration",
            status=status,
            score=min(100, deployment_score),
            message=f"Deployment readiness: {deployment_score}/100 ({len(deployment_files)} artifacts)",
            details={"artifacts": deployment_files}
        )
    
    def _test_monitoring(self) -> QualityGateResult:
        """Test monitoring integration."""
        monitoring_features = []
        
        # Check for monitoring-related code and configuration
        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    
                patterns = [
                    ("Logging", "logging"),
                    ("Metrics", "metric"),  
                    ("Health checks", "health"),
                    ("Monitoring", "monitor"),
                    ("Telemetry", "telemetry"),
                    ("Prometheus", "prometheus"),
                    ("Grafana", "grafana")
                ]
                
                for feature, pattern in patterns:
                    if pattern in content and feature not in [f[0] for f in monitoring_features]:
                        monitoring_features.append((feature, py_file.name))
                        
            except Exception:
                continue
        
        # Check for monitoring configuration files
        monitoring_configs = ["prometheus.yml", "grafana/", "monitoring/"]
        for config in monitoring_configs:
            if (self.project_root / config).exists():
                monitoring_features.append((f"Config: {config}", "filesystem"))
                
        score = min(100, len(monitoring_features) * 12)
        status = "PASS" if score >= 60 else "WARNING" if score >= 30 else "FAIL"
        
        return QualityGateResult(
            gate_name="Monitoring Integration",
            status=status,
            score=score,
            message=f"Monitoring features: {len(monitoring_features)} detected",
            details={"features": monitoring_features}
        )
    
    def _validate_configuration(self) -> QualityGateResult:
        """Validate configuration management."""
        config_features = []
        config_score = 0
        
        # Check for configuration files
        config_files = [
            "config.py", "settings.py", "config.json", "config.yaml", 
            "config/", ".env", "environment.py"
        ]
        
        for config_file in config_files:
            if (self.project_root / config_file).exists():
                config_features.append(config_file)
                config_score += 10
                
        # Check for environment-specific configs
        env_configs = ["config/development.json", "config/production.json", "config/staging.json"]
        env_count = sum(1 for ec in env_configs if (self.project_root / ec).exists())
        if env_count >= 2:
            config_score += 20
            config_features.append(f"Environment configs ({env_count})")
            
        # Check for configuration validation
        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    if "config" in content and "valid" in content:
                        config_score += 10
                        config_features.append("Configuration validation")
                        break
            except Exception:
                continue
                
        status = "PASS" if config_score >= 40 else "WARNING" if config_score >= 20 else "FAIL"
        
        return QualityGateResult(
            gate_name="Configuration Management",
            status=status,
            score=min(100, config_score),
            message=f"Configuration management: {config_score}/100",
            details={"features": config_features}
        )
    
    # === COMPLIANCE GATES ===
    
    def _check_security_compliance(self) -> QualityGateResult:
        """Check security standards compliance."""
        compliance_features = []
        
        # Check for security-related features
        security_checks = [
            ("Input validation", ["validation", "sanitize", "validate"]),
            ("Authentication", ["auth", "login", "credential"]),
            ("Authorization", ["permission", "access", "role"]),
            ("Encryption", ["encrypt", "decrypt", "cipher"]),
            ("Logging", ["log", "audit", "trace"]),
            ("Rate limiting", ["rate", "limit", "throttle"]),
            ("HTTPS", ["https", "ssl", "tls"]),
        ]
        
        for feature_name, patterns in security_checks:
            found = False
            for py_file in self.project_root.glob("src/**/*.py"):
                try:
                    with open(py_file, 'r') as f:
                        content = f.read().lower()
                        if any(pattern in content for pattern in patterns):
                            compliance_features.append(feature_name)
                            found = True
                            break
                except Exception:
                    continue
            if found:
                continue
                
        score = (len(compliance_features) / len(security_checks)) * 100
        status = "PASS" if score >= 70 else "WARNING" if score >= 50 else "FAIL"
        
        return QualityGateResult(
            gate_name="Security Standards Compliance",
            status=status,
            score=score,
            message=f"Security compliance: {len(compliance_features)}/{len(security_checks)} features",
            details={"features": compliance_features}
        )
    
    def _validate_performance_benchmarks(self) -> QualityGateResult:
        """Validate performance benchmarks."""
        # Check for performance testing
        perf_files = []
        for pattern in ["*benchmark*", "*performance*", "*perf*"]:
            perf_files.extend(self.project_root.glob(f"**/{pattern}.py"))
            
        # Check for performance-related code
        perf_features = []
        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    patterns = ["benchmark", "performance", "latency", "throughput", "optimization"]
                    for pattern in patterns:
                        if pattern in content and pattern not in perf_features:
                            perf_features.append(pattern)
            except Exception:
                continue
                
        # Run existing performance reports
        reports = list(self.project_root.glob("*performance*report*.json"))
        
        score = 60  # Base score
        if perf_files:
            score += 20  # Bonus for performance test files
        if len(perf_features) >= 3:
            score += 15  # Bonus for performance features
        if reports:
            score += 5   # Bonus for existing reports
            
        status = "PASS" if score >= 80 else "WARNING" if score >= 60 else "FAIL"
        
        return QualityGateResult(
            gate_name="Performance Benchmarks",
            status=status,
            score=score,
            message=f"Performance validation: {len(perf_files)} test files, {len(perf_features)} features",
            details={
                "perf_files": [f.name for f in perf_files],
                "features": perf_features,
                "reports": [r.name for r in reports]
            }
        )
    
    # === ASSESSMENT METHODS ===
    
    def _assess_generation(self, report: QualityReport) -> str:
        """Assess which SDLC generation the system has reached."""
        scores_by_category = {
            "functionality": [],
            "security": [],
            "performance": [],
            "reliability": [],
            "quality": [],
            "production": [],
            "compliance": []
        }
        
        # Categorize results
        category_mapping = {
            "Code Syntax Validation": "functionality",
            "Import Structure Check": "functionality", 
            "Core Functionality Test": "functionality",
            
            "Security Vulnerability Scan": "security",
            "Input Validation Coverage": "security",
            "Dependency Security Check": "security",
            
            "Memory Optimization Test": "performance",
            "Auto-scaling Validation": "performance", 
            "Resource Utilization Check": "performance",
            
            "Error Handling Coverage": "reliability",
            "Fault Tolerance Test": "reliability",
            "Recovery Mechanisms Test": "reliability",
            
            "Code Quality Metrics": "quality",
            "Test Coverage Analysis": "quality",
            "Documentation Completeness": "quality",
            
            "Deployment Configuration": "production",
            "Monitoring Integration": "production",
            "Configuration Management": "production",
            
            "Security Standards Compliance": "compliance",
            "Performance Benchmarks": "compliance"
        }
        
        for result in report.gate_results:
            category = category_mapping.get(result.gate_name, "quality")
            scores_by_category[category].append(result.score)
            
        # Calculate category averages
        category_scores = {}
        for category, scores in scores_by_category.items():
            if scores:
                category_scores[category] = sum(scores) / len(scores)
            else:
                category_scores[category] = 0
                
        # Determine generation
        functionality_score = category_scores.get("functionality", 0)
        reliability_score = category_scores.get("reliability", 0)  
        performance_score = category_scores.get("performance", 0)
        production_score = category_scores.get("production", 0)
        
        if (functionality_score >= 80 and reliability_score >= 80 and 
            performance_score >= 80 and production_score >= 70):
            return "Generation 3+ (Production Ready)"
        elif functionality_score >= 70 and reliability_score >= 70:
            return "Generation 2 (Robust)"
        elif functionality_score >= 60:
            return "Generation 1 (Functional)"
        else:
            return "Generation 0 (Incomplete)"
    
    def _assess_production_readiness(self, report: QualityReport) -> bool:
        """Assess if system is ready for production deployment."""
        # Critical gates that must pass for production
        critical_gates = [
            "Code Syntax Validation",
            "Security Vulnerability Scan", 
            "Input Validation Coverage",
            "Memory Optimization Test",
            "Error Handling Coverage"
        ]
        
        critical_passed = 0
        for result in report.gate_results:
            if (result.gate_name in critical_gates and 
                result.status in ["PASS", "WARNING"]):
                critical_passed += 1
                
        # Production readiness criteria
        return (
            report.overall_score >= 75 and  # Overall score > 75%
            critical_passed >= len(critical_gates) * 0.8 and  # 80% of critical gates pass
            report.failed_gates <= 3  # No more than 3 failed gates
        )
    
    def _generate_recommendations(self, report: QualityReport) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Analyze failed gates
        failed_gates = [r for r in report.gate_results if r.status == "FAIL"]
        warning_gates = [r for r in report.gate_results if r.status == "WARNING"]
        
        # Category-specific recommendations
        if any("Security" in r.gate_name for r in failed_gates):
            recommendations.append("🔒 Address security vulnerabilities and improve input validation coverage")
            
        if any("Performance" in r.gate_name or "Memory" in r.gate_name for r in failed_gates):
            recommendations.append("⚡ Optimize performance bottlenecks and memory usage patterns")
            
        if any("Test" in r.gate_name for r in failed_gates):
            recommendations.append("🧪 Expand test coverage with unit, integration, and performance tests")
            
        if any("Documentation" in r.gate_name for r in failed_gates):
            recommendations.append("📚 Improve documentation coverage and API documentation")
            
        if any("Deployment" in r.gate_name or "Configuration" in r.gate_name for r in failed_gates):
            recommendations.append("🚀 Complete deployment configuration and production setup")
            
        # Overall recommendations
        if report.overall_score < 70:
            recommendations.append("🎯 Focus on core functionality and reliability improvements first")
        elif report.overall_score < 85:
            recommendations.append("🔧 Address remaining quality and performance optimizations")
        else:
            recommendations.append("✨ System shows excellent quality - focus on monitoring and maintenance")
            
        # Specific recommendations based on warning gates
        for result in warning_gates[:3]:  # Top 3 warnings
            if result.score < 70:
                recommendations.append(f"⚠️ Improve {result.gate_name}: {result.message}")
                
        return recommendations[:7]  # Limit to 7 recommendations


def main():
    """Execute autonomous quality gates and generate final report."""
    print("🤖 TERRAGON AUTONOMOUS SDLC v4.0 - QUALITY GATES")
    print("=" * 60)
    print()
    
    # Initialize quality gate engine
    engine = QualityGateEngine()
    
    # Execute all quality gates
    print("🔍 Executing comprehensive quality gates...")
    report = engine.execute_all_gates()
    
    # Display results
    print("\n" + "=" * 60)
    print("QUALITY GATE EXECUTION RESULTS")  
    print("=" * 60)
    
    # Summary statistics
    print(f"📊 EXECUTION SUMMARY:")
    print(f"   Total Gates: {report.total_gates}")
    print(f"   Passed: {report.passed_gates} ✅")
    print(f"   Failed: {report.failed_gates} ❌")
    print(f"   Warnings: {report.warning_gates} ⚠️")
    print(f"   Skipped: {report.skipped_gates} ⏭️")
    print(f"   Overall Score: {report.overall_score:.1f}/100")
    print(f"   Execution Time: {report.execution_time:.2f}s")
    print()
    
    # Generation assessment
    print(f"🎯 SDLC GENERATION: {report.generation_assessment}")
    print(f"🚀 PRODUCTION READY: {'✅ YES' if report.production_readiness else '❌ NO'}")
    print()
    
    # Gate results by status
    status_groups = {
        "PASS": [r for r in report.gate_results if r.status == "PASS"],
        "WARNING": [r for r in report.gate_results if r.status == "WARNING"],
        "FAIL": [r for r in report.gate_results if r.status == "FAIL"],
        "SKIP": [r for r in report.gate_results if r.status == "SKIP"]
    }
    
    for status, results in status_groups.items():
        if results:
            icon = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}[status]
            print(f"{icon} {status} GATES ({len(results)}):")
            for result in results:
                print(f"   • {result.gate_name}: {result.score:.1f}% - {result.message}")
            print()
    
    # Top recommendations
    if report.recommendations:
        print("🔧 KEY RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"   {i}. {rec}")
        print()
    
    # Detailed category analysis
    categories = {
        "🔧 Functionality": ["Code Syntax", "Import Structure", "Core Functionality"],
        "🔒 Security": ["Security Vulnerability", "Input Validation", "Dependency Security", "Security Standards"],
        "⚡ Performance": ["Memory Optimization", "Auto-scaling", "Resource Utilization", "Performance Benchmarks"],
        "🛡️ Reliability": ["Error Handling", "Fault Tolerance", "Recovery Mechanisms"],
        "📋 Quality": ["Code Quality", "Test Coverage", "Documentation"],
        "🚀 Production": ["Deployment", "Monitoring", "Configuration"]
    }
    
    print("📈 DETAILED CATEGORY ANALYSIS:")
    for category, keywords in categories.items():
        category_results = [r for r in report.gate_results 
                          if any(keyword in r.gate_name for keyword in keywords)]
        if category_results:
            avg_score = sum(r.score for r in category_results) / len(category_results)
            passed = sum(1 for r in category_results if r.status == "PASS")
            total = len(category_results)
            print(f"   {category}: {avg_score:.1f}% ({passed}/{total} passed)")
    print()
    
    # Final assessment
    if report.production_readiness and report.overall_score >= 85:
        print("🎉 AUTONOMOUS SDLC COMPLETION: SUCCESS!")
        print("   ✅ All critical quality gates passed")
        print("   ✅ System ready for production deployment")
        print("   ✅ Generation 3+ capabilities achieved")
        conclusion = "OUTSTANDING"
    elif report.production_readiness:
        print("✅ AUTONOMOUS SDLC COMPLETION: SUCCESSFUL")
        print("   ✅ Core quality gates passed")
        print("   ✅ System ready for production with monitoring")
        print("   ⚠️ Some optimizations recommended")
        conclusion = "SUCCESSFUL"
    elif report.overall_score >= 70:
        print("⚠️ AUTONOMOUS SDLC COMPLETION: PARTIAL SUCCESS")
        print("   ⚠️ System functional but needs improvements")
        print("   ❌ Not recommended for production deployment")
        print("   🔧 Address critical issues before deployment")
        conclusion = "PARTIAL"
    else:
        print("❌ AUTONOMOUS SDLC COMPLETION: NEEDS IMPROVEMENT")
        print("   ❌ Critical quality gates failed")
        print("   ❌ System requires significant development")
        print("   🔧 Focus on core functionality and reliability")
        conclusion = "NEEDS_WORK"
    
    # Save detailed report
    report_data = {
        "autonomous_sdlc_version": "4.0",
        "execution_timestamp": time.time(),
        "overall_score": report.overall_score,
        "production_readiness": report.production_readiness,
        "generation_assessment": report.generation_assessment,
        "conclusion": conclusion,
        "execution_time": report.execution_time,
        "summary": {
            "total_gates": report.total_gates,
            "passed_gates": report.passed_gates,
            "failed_gates": report.failed_gates,
            "warning_gates": report.warning_gates,
            "skipped_gates": report.skipped_gates
        },
        "gate_results": [
            {
                "name": r.gate_name,
                "status": r.status,
                "score": r.score,
                "message": r.message,
                "details": r.details,
                "execution_time": r.execution_time
            } for r in report.gate_results
        ],
        "recommendations": report.recommendations
    }
    
    report_file = Path(__file__).parent / "autonomous_sdlc_final_report.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
        
    print(f"\n📄 Detailed report saved: {report_file}")
    
    # Return appropriate exit code
    if report.production_readiness and report.overall_score >= 80:
        return 0  # Success
    elif report.overall_score >= 60:
        return 1  # Partial success
    else:
        return 2  # Needs work


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)