"""
Intelligent Auto-Scaling Infrastructure with Resource Prediction.

Provides cutting-edge auto-scaling capabilities with machine learning-based
workload prediction, intelligent resource allocation, and proactive scaling
for neuromorphic vision processing systems.
"""

import time
import threading
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import json
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .monitoring import get_metrics_collector
from .scaling import ResourceMetrics, ScalingPolicy, WorkloadPrediction, ScalingDecision
from .gpu_distributed_processor import get_distributed_gpu_processor
from .async_event_processor import get_async_event_pipeline
from .intelligent_cache_system import get_intelligent_cache


@dataclass
class PredictionFeatures:
    """Feature vector for workload prediction."""
    # Time-based features
    hour_of_day: float
    day_of_week: float
    minute_of_hour: float
    
    # Historical resource metrics
    cpu_trend: float
    memory_trend: float
    throughput_trend: float
    latency_trend: float
    
    # Workload characteristics
    event_rate: float
    processing_complexity: float
    cache_efficiency: float
    
    # System state
    current_workers: int
    queue_depth: float
    gpu_utilization: float
    
    # External factors
    external_load_indicator: float = 0.0
    seasonal_factor: float = 1.0


@dataclass
class ResourceUtilizationTarget:
    """Target resource utilization levels for optimization."""
    cpu_target_percent: float = 70.0
    memory_target_percent: float = 75.0
    gpu_target_percent: float = 80.0
    queue_target_depth: int = 50
    latency_target_ms: float = 10.0
    throughput_target_eps: float = 10000.0
    
    # Tolerance ranges
    cpu_tolerance: float = 10.0
    memory_tolerance: float = 10.0
    gpu_tolerance: float = 15.0


class WorkloadPredictor:
    """Machine learning-based workload prediction system."""
    
    def __init__(self, history_window_hours: int = 24):
        self.history_window_hours = history_window_hours
        self.max_history_points = history_window_hours * 60  # Assume 1-minute intervals
        
        # ML models for different metrics
        self.cpu_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.memory_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.throughput_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.latency_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Feature scaling
        self.feature_scaler = StandardScaler()
        
        # Training data
        self.training_features = deque(maxlen=self.max_history_points)
        self.training_targets = deque(maxlen=self.max_history_points)
        
        # Model state
        self.models_trained = False
        self.last_training_time = 0.0
        self.min_training_samples = 50
        self.training_interval = 3600.0  # Retrain every hour
        
        self.logger = logging.getLogger(__name__)
        
    def collect_training_sample(self, features: PredictionFeatures, metrics: ResourceMetrics):
        """Collect training sample for model improvement."""
        # Convert features to array
        feature_array = np.array([
            features.hour_of_day,
            features.day_of_week,
            features.minute_of_hour,
            features.cpu_trend,
            features.memory_trend,
            features.throughput_trend,
            features.latency_trend,
            features.event_rate,
            features.processing_complexity,
            features.cache_efficiency,
            features.current_workers,
            features.queue_depth,
            features.gpu_utilization,
            features.external_load_indicator,
            features.seasonal_factor
        ])
        
        # Target values (what we want to predict)
        targets = {
            'cpu': metrics.cpu_percent,
            'memory': metrics.memory_percent,
            'throughput': metrics.event_throughput_eps,
            'latency': metrics.average_inference_time
        }
        
        self.training_features.append(feature_array)
        self.training_targets.append(targets)
        
        # Trigger retraining if needed
        current_time = time.time()
        if (len(self.training_features) >= self.min_training_samples and
            current_time - self.last_training_time > self.training_interval):
            self._retrain_models()
            
    def _retrain_models(self):
        """Retrain prediction models with collected data."""
        try:
            if len(self.training_features) < self.min_training_samples:
                return
                
            self.logger.info("Retraining workload prediction models...")
            
            # Prepare training data
            X = np.array(list(self.training_features))
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Prepare targets for each model
            cpu_targets = [t['cpu'] for t in self.training_targets]
            memory_targets = [t['memory'] for t in self.training_targets]
            throughput_targets = [t['throughput'] for t in self.training_targets]
            latency_targets = [t['latency'] for t in self.training_targets]
            
            # Train models
            self.cpu_model.fit(X_scaled, cpu_targets)
            self.memory_model.fit(X_scaled, memory_targets)
            self.throughput_model.fit(X_scaled, throughput_targets)
            self.latency_model.fit(X_scaled, latency_targets)
            
            self.models_trained = True
            self.last_training_time = time.time()
            
            # Calculate and log training accuracy
            cpu_score = self.cpu_model.score(X_scaled, cpu_targets)
            memory_score = self.memory_model.score(X_scaled, memory_targets)
            throughput_score = self.throughput_model.score(X_scaled, throughput_targets)
            latency_score = self.latency_model.score(X_scaled, latency_targets)
            
            self.logger.info(
                f"Model training completed - Scores: "
                f"CPU: {cpu_score:.3f}, Memory: {memory_score:.3f}, "
                f"Throughput: {throughput_score:.3f}, Latency: {latency_score:.3f}"
            )
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            
    def predict_workload(
        self, 
        current_features: PredictionFeatures,
        prediction_horizon_seconds: float = 300.0
    ) -> WorkloadPrediction:
        """Predict future workload characteristics."""
        if not self.models_trained:
            # Return conservative prediction
            return WorkloadPrediction(
                predicted_cpu_percent=current_features.cpu_trend,
                predicted_memory_percent=current_features.memory_trend,
                predicted_throughput_eps=current_features.throughput_trend,
                predicted_latency_ms=current_features.latency_trend,
                confidence_score=0.1,
                prediction_horizon_seconds=prediction_horizon_seconds
            )
            
        try:
            # Prepare feature array
            feature_array = np.array([[
                current_features.hour_of_day,
                current_features.day_of_week,
                current_features.minute_of_hour,
                current_features.cpu_trend,
                current_features.memory_trend,
                current_features.throughput_trend,
                current_features.latency_trend,
                current_features.event_rate,
                current_features.processing_complexity,
                current_features.cache_efficiency,
                current_features.current_workers,
                current_features.queue_depth,
                current_features.gpu_utilization,
                current_features.external_load_indicator,
                current_features.seasonal_factor
            ]])
            
            # Scale features
            if hasattr(self.feature_scaler, 'transform'):
                feature_array_scaled = self.feature_scaler.transform(feature_array)
            else:
                feature_array_scaled = feature_array
                
            # Make predictions
            cpu_pred = self.cpu_model.predict(feature_array_scaled)[0]
            memory_pred = self.memory_model.predict(feature_array_scaled)[0]
            throughput_pred = self.throughput_model.predict(feature_array_scaled)[0]
            latency_pred = self.latency_model.predict(feature_array_scaled)[0]
            
            # Calculate confidence based on model uncertainty
            # Using ensemble variance as confidence proxy
            confidence = self._calculate_prediction_confidence(feature_array_scaled)
            
            return WorkloadPrediction(
                predicted_cpu_percent=max(0.0, min(100.0, cpu_pred)),
                predicted_memory_percent=max(0.0, min(100.0, memory_pred)),
                predicted_throughput_eps=max(0.0, throughput_pred),
                predicted_latency_ms=max(0.0, latency_pred),
                confidence_score=confidence,
                prediction_horizon_seconds=prediction_horizon_seconds
            )
            
        except Exception as e:
            self.logger.error(f"Workload prediction failed: {e}")
            return WorkloadPrediction(
                confidence_score=0.0,
                prediction_horizon_seconds=prediction_horizon_seconds
            )
            
    def _calculate_prediction_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence score for predictions."""
        try:
            # Use ensemble variance as confidence indicator
            # Lower variance = higher confidence
            
            if len(self.training_features) < 10:
                return 0.1
                
            # Get predictions from individual trees
            cpu_preds = np.array([
                tree.predict(features)[0] 
                for tree in self.cpu_model.estimators_[:10]  # Sample first 10 trees
            ])
            
            # Calculate coefficient of variation (std/mean) as uncertainty measure
            cv = np.std(cpu_preds) / (np.mean(cpu_preds) + 1e-6)
            
            # Convert to confidence (0-1, higher is better)
            confidence = max(0.1, min(1.0, 1.0 / (1.0 + cv)))
            
            return confidence
            
        except Exception:
            return 0.5  # Default moderate confidence
            
    def save_models(self, path: Path):
        """Save trained models to disk."""
        try:
            model_data = {
                'cpu_model': self.cpu_model,
                'memory_model': self.memory_model,
                'throughput_model': self.throughput_model,
                'latency_model': self.latency_model,
                'feature_scaler': self.feature_scaler,
                'models_trained': self.models_trained,
                'last_training_time': self.last_training_time
            }
            
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
                
            self.logger.info(f"Models saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
            
    def load_models(self, path: Path) -> bool:
        """Load trained models from disk."""
        try:
            if not path.exists():
                return False
                
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.cpu_model = model_data['cpu_model']
            self.memory_model = model_data['memory_model']
            self.throughput_model = model_data['throughput_model']
            self.latency_model = model_data['latency_model']
            self.feature_scaler = model_data['feature_scaler']
            self.models_trained = model_data['models_trained']
            self.last_training_time = model_data['last_training_time']
            
            self.logger.info(f"Models loaded from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False


class IntelligentScalingDecisionEngine:
    """Intelligent decision engine for auto-scaling decisions."""
    
    def __init__(self, utilization_targets: ResourceUtilizationTarget = None):
        self.utilization_targets = utilization_targets or ResourceUtilizationTarget()
        self.logger = logging.getLogger(__name__)
        
        # Decision history for learning
        self.decision_history = deque(maxlen=100)
        
        # Economic factors
        self.cost_per_worker_hour = 1.0  # Placeholder cost
        self.sla_violation_cost = 10.0   # Cost of SLA violations
        
    def make_scaling_decision(
        self,
        current_metrics: ResourceMetrics,
        prediction: WorkloadPrediction,
        current_workers: int,
        policy: ScalingPolicy
    ) -> ScalingDecision:
        """Make intelligent scaling decision based on current and predicted metrics."""
        
        reasoning = []
        expected_improvement = {}
        risk_assessment = {}
        
        # Analyze current state
        current_utilization = self._analyze_current_utilization(current_metrics)
        predicted_utilization = self._analyze_predicted_utilization(prediction)
        
        # Economic analysis
        economic_factors = self._analyze_economic_factors(
            current_metrics, prediction, current_workers
        )
        
        # Risk analysis
        risks = self._analyze_scaling_risks(current_metrics, prediction, current_workers)
        
        # Decision logic
        if prediction.confidence_score < 0.3:
            # Low confidence in predictions - use conservative reactive approach
            decision = self._make_reactive_decision(
                current_metrics, current_workers, policy
            )
            reasoning.append("Low prediction confidence - using reactive scaling")
            
        else:
            # High confidence - use predictive scaling
            decision = self._make_predictive_decision(
                current_metrics, prediction, current_workers, policy
            )
            reasoning.append("High prediction confidence - using predictive scaling")
            
        # Add economic reasoning
        if economic_factors['scale_up_beneficial']:
            reasoning.append("Economic analysis favors scaling up")
        elif economic_factors['scale_down_beneficial']:
            reasoning.append("Economic analysis favors scaling down")
            
        # Add risk reasoning
        if risks['high_latency_risk'] > 0.7:
            reasoning.append("High risk of latency SLA violations")
        if risks['resource_exhaustion_risk'] > 0.8:
            reasoning.append("High risk of resource exhaustion")
            
        # Calculate expected improvements
        if decision['action'] == 'scale_up':
            expected_improvement = {
                'latency_reduction_percent': min(30.0, risks['high_latency_risk'] * 40),
                'throughput_increase_percent': 15.0,
                'queue_depth_reduction_percent': 25.0
            }
        elif decision['action'] == 'scale_down':
            expected_improvement = {
                'cost_reduction_percent': 20.0,
                'resource_efficiency_increase': 10.0
            }
            
        # Build final decision
        scaling_decision = ScalingDecision(
            action=decision['action'],
            target_workers=decision['target_workers'],
            confidence=decision['confidence'],
            reasoning=reasoning,
            expected_improvement=expected_improvement,
            risk_assessment=risks
        )
        
        # Record decision for learning
        self.decision_history.append(scaling_decision)
        
        return scaling_decision
        
    def _analyze_current_utilization(self, metrics: ResourceMetrics) -> Dict[str, Any]:
        """Analyze current resource utilization."""
        return {
            'cpu_above_target': metrics.cpu_percent > self.utilization_targets.cpu_target_percent,
            'memory_above_target': metrics.memory_percent > self.utilization_targets.memory_target_percent,
            'latency_above_target': metrics.average_inference_time > self.utilization_targets.latency_target_ms,
            'queue_above_target': metrics.inference_queue_size > self.utilization_targets.queue_target_depth,
            'overall_pressure': self._calculate_resource_pressure(metrics)
        }
        
    def _analyze_predicted_utilization(self, prediction: WorkloadPrediction) -> Dict[str, Any]:
        """Analyze predicted resource utilization."""
        return {
            'cpu_will_exceed': prediction.predicted_cpu_percent > self.utilization_targets.cpu_target_percent,
            'memory_will_exceed': prediction.predicted_memory_percent > self.utilization_targets.memory_target_percent,
            'latency_will_exceed': prediction.predicted_latency_ms > self.utilization_targets.latency_target_ms,
            'throughput_will_drop': prediction.predicted_throughput_eps < self.utilization_targets.throughput_target_eps * 0.8
        }
        
    def _calculate_resource_pressure(self, metrics: ResourceMetrics) -> float:
        """Calculate overall resource pressure score (0-1)."""
        cpu_pressure = metrics.cpu_percent / 100.0
        memory_pressure = metrics.memory_percent / 100.0
        queue_pressure = min(1.0, metrics.inference_queue_size / 200.0)
        latency_pressure = min(1.0, metrics.average_inference_time / 1000.0)  # Normalize to 1s
        
        # Weighted average
        overall_pressure = (
            0.3 * cpu_pressure +
            0.3 * memory_pressure +
            0.2 * queue_pressure +
            0.2 * latency_pressure
        )
        
        return overall_pressure
        
    def _analyze_economic_factors(
        self, 
        current_metrics: ResourceMetrics,
        prediction: WorkloadPrediction,
        current_workers: int
    ) -> Dict[str, Any]:
        """Analyze economic factors for scaling decisions."""
        
        # Estimate costs
        current_cost_per_hour = current_workers * self.cost_per_worker_hour
        
        # Estimate SLA violation costs
        latency_violation_prob = max(0, (current_metrics.average_inference_time - 100) / 100)
        sla_violation_cost_per_hour = latency_violation_prob * self.sla_violation_cost
        
        # Scale up analysis
        scale_up_cost = (current_workers + 1) * self.cost_per_worker_hour
        scale_up_sla_violation_cost = max(0, sla_violation_cost_per_hour * 0.5)  # 50% reduction
        scale_up_total_cost = scale_up_cost + scale_up_sla_violation_cost
        
        # Scale down analysis
        scale_down_cost = max(1, current_workers - 1) * self.cost_per_worker_hour
        scale_down_sla_violation_cost = sla_violation_cost_per_hour * 1.5  # 50% increase
        scale_down_total_cost = scale_down_cost + scale_down_sla_violation_cost
        
        current_total_cost = current_cost_per_hour + sla_violation_cost_per_hour
        
        return {
            'scale_up_beneficial': scale_up_total_cost < current_total_cost,
            'scale_down_beneficial': scale_down_total_cost < current_total_cost,
            'current_total_cost': current_total_cost,
            'scale_up_savings': current_total_cost - scale_up_total_cost,
            'scale_down_savings': current_total_cost - scale_down_total_cost
        }
        
    def _analyze_scaling_risks(
        self,
        current_metrics: ResourceMetrics,
        prediction: WorkloadPrediction,
        current_workers: int
    ) -> Dict[str, float]:
        """Analyze risks associated with scaling decisions."""
        
        risks = {}
        
        # Latency SLA violation risk
        current_latency_risk = min(1.0, current_metrics.average_inference_time / 1000.0)
        predicted_latency_risk = min(1.0, prediction.predicted_latency_ms / 1000.0)
        risks['high_latency_risk'] = max(current_latency_risk, predicted_latency_risk)
        
        # Resource exhaustion risk
        cpu_risk = current_metrics.cpu_percent / 100.0
        memory_risk = current_metrics.memory_percent / 100.0
        queue_risk = min(1.0, current_metrics.inference_queue_size / 500.0)
        risks['resource_exhaustion_risk'] = max(cpu_risk, memory_risk, queue_risk)
        
        # Thrashing risk (scaling up/down too frequently)
        recent_decisions = list(self.decision_history)[-10:]
        scale_changes = sum(1 for d in recent_decisions if d.action != 'no_action')
        risks['thrashing_risk'] = min(1.0, scale_changes / 5.0)
        
        # Underutilization risk
        if current_workers > 1:
            avg_utilization = (current_metrics.cpu_percent + current_metrics.memory_percent) / 200.0
            risks['underutilization_risk'] = max(0.0, 0.5 - avg_utilization)
        else:
            risks['underutilization_risk'] = 0.0
            
        return risks
        
    def _make_reactive_decision(
        self,
        current_metrics: ResourceMetrics,
        current_workers: int,
        policy: ScalingPolicy
    ) -> Dict[str, Any]:
        """Make reactive scaling decision based on current metrics."""
        
        pressure = self._calculate_resource_pressure(current_metrics)
        
        if pressure > 0.8 and current_workers < policy.max_workers:
            return {
                'action': 'scale_up',
                'target_workers': min(current_workers + policy.scale_step_size, policy.max_workers),
                'confidence': 0.7
            }
        elif pressure < 0.3 and current_workers > policy.min_workers:
            return {
                'action': 'scale_down',
                'target_workers': max(current_workers - policy.scale_step_size, policy.min_workers),
                'confidence': 0.6
            }
        else:
            return {
                'action': 'no_action',
                'target_workers': current_workers,
                'confidence': 0.8
            }
            
    def _make_predictive_decision(
        self,
        current_metrics: ResourceMetrics,
        prediction: WorkloadPrediction,
        current_workers: int,
        policy: ScalingPolicy
    ) -> Dict[str, Any]:
        """Make predictive scaling decision based on predictions."""
        
        # Combine current and predicted pressure
        current_pressure = self._calculate_resource_pressure(current_metrics)
        
        # Estimate predicted pressure
        predicted_pressure = (
            prediction.predicted_cpu_percent / 100.0 * 0.3 +
            prediction.predicted_memory_percent / 100.0 * 0.3 +
            min(1.0, prediction.predicted_latency_ms / 1000.0) * 0.4
        )
        
        # Weight current and predicted
        combined_pressure = 0.6 * current_pressure + 0.4 * predicted_pressure
        confidence = prediction.confidence_score
        
        if combined_pressure > 0.75 and current_workers < policy.max_workers:
            return {
                'action': 'scale_up',
                'target_workers': min(current_workers + policy.scale_step_size, policy.max_workers),
                'confidence': confidence
            }
        elif combined_pressure < 0.25 and current_workers > policy.min_workers:
            return {
                'action': 'scale_down',
                'target_workers': max(current_workers - policy.scale_step_size, policy.min_workers),
                'confidence': confidence * 0.8  # Be more cautious about scaling down
            }
        else:
            return {
                'action': 'no_action',
                'target_workers': current_workers,
                'confidence': confidence
            }


class IntelligentAutoScaler:
    """Advanced intelligent auto-scaler with predictive capabilities."""
    
    def __init__(
        self,
        policy: ScalingPolicy = None,
        utilization_targets: ResourceUtilizationTarget = None,
        enable_predictions: bool = True
    ):
        self.policy = policy or ScalingPolicy()
        self.utilization_targets = utilization_targets or ResourceUtilizationTarget()
        self.enable_predictions = enable_predictions
        
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.workload_predictor = WorkloadPredictor() if enable_predictions else None
        self.decision_engine = IntelligentScalingDecisionEngine(utilization_targets)
        
        # System integration
        self.gpu_processor = get_distributed_gpu_processor()
        self.async_pipeline = get_async_event_pipeline()
        self.cache_system = get_intelligent_cache()
        self.metrics_collector = get_metrics_collector()
        
        # Auto-scaling state
        self.current_workers = self.policy.min_workers
        self.is_running = False
        self.scaling_thread = None
        self.last_scaling_action = 0.0
        
        # Performance tracking
        self.scaling_history = deque(maxlen=100)
        self.performance_improvements = deque(maxlen=50)
        
        # Feature extraction
        self.metrics_history = deque(maxlen=1440)  # 24 hours of minute-level data
        
    def start_intelligent_scaling(self):
        """Start intelligent auto-scaling system."""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Load pre-trained models if available
        if self.workload_predictor:
            model_path = Path("workload_models.pkl")
            self.workload_predictor.load_models(model_path)
            
        # Start scaling loop
        self.scaling_thread = threading.Thread(
            target=self._intelligent_scaling_loop,
            daemon=True
        )
        self.scaling_thread.start()
        
        self.logger.info("Intelligent auto-scaling started")
        
    def stop_intelligent_scaling(self):
        """Stop intelligent auto-scaling system."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10.0)
            
        # Save trained models
        if self.workload_predictor:
            model_path = Path("workload_models.pkl")
            self.workload_predictor.save_models(model_path)
            
        self.logger.info("Intelligent auto-scaling stopped")
        
    def _intelligent_scaling_loop(self):
        """Main intelligent scaling loop."""
        while self.is_running:
            try:
                # Collect current metrics
                current_metrics = self._collect_comprehensive_metrics()
                self.metrics_history.append(current_metrics)
                
                # Extract features for prediction
                if self.enable_predictions and self.workload_predictor:
                    features = self._extract_prediction_features(current_metrics)
                    
                    # Make workload prediction
                    prediction = self.workload_predictor.predict_workload(features)
                    
                    # Collect training sample
                    self.workload_predictor.collect_training_sample(features, current_metrics)
                    
                else:
                    prediction = WorkloadPrediction(confidence_score=0.0)
                    
                # Make scaling decision
                scaling_decision = self.decision_engine.make_scaling_decision(
                    current_metrics, prediction, self.current_workers, self.policy
                )
                
                # Execute scaling action if needed
                if (scaling_decision.action != 'no_action' and
                    scaling_decision.confidence > 0.5 and
                    time.time() - self.last_scaling_action > self.policy.scale_cooldown):
                    
                    self._execute_intelligent_scaling_action(scaling_decision, current_metrics)
                    
                # Sleep until next evaluation
                time.sleep(self.policy.scale_check_interval)
                
            except Exception as e:
                self.logger.error(f"Intelligent scaling loop error: {e}")
                time.sleep(self.policy.scale_check_interval)
                
    def _collect_comprehensive_metrics(self) -> ResourceMetrics:
        """Collect comprehensive system metrics."""
        # Get GPU processor stats
        gpu_stats = self.gpu_processor.get_processing_stats()
        
        # Get pipeline stats  
        pipeline_stats = self.async_pipeline.get_comprehensive_stats()
        
        # Get cache stats
        cache_stats = self.cache_system.get_comprehensive_stats()
        
        # Get system metrics
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Calculate advanced metrics
        event_throughput = pipeline_stats['pipeline'].get('current_throughput_eps', 0.0)
        pipeline_latency = pipeline_stats['pipeline'].get('average_latency_ns', 0.0)
        cache_hit_rate = cache_stats['global'].get('overall_hit_rate', 0.0)
        
        # GPU metrics
        gpu_memory_percent = 0.0
        gpu_utilization = 0.0
        
        if gpu_stats['resource_stats']['total_devices'] > 0:
            gpu_memory_percent = (
                gpu_stats['resource_stats']['used_memory_mb'] / 
                gpu_stats['resource_stats']['total_memory_mb'] * 100
            )
            gpu_utilization = gpu_stats['resource_stats']['average_utilization']
            
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_memory_percent=gpu_memory_percent,
            gpu_utilization=gpu_utilization,
            inference_queue_size=gpu_stats['processing_stats']['tasks_submitted'] - 
                               gpu_stats['processing_stats']['tasks_completed'],
            average_inference_time=gpu_stats['processing_stats'].get('average_throughput', 0.0),
            event_throughput_eps=event_throughput,
            cache_hit_rate=cache_hit_rate,
            pipeline_latency_ns=pipeline_latency
        )
        
    def _extract_prediction_features(self, current_metrics: ResourceMetrics) -> PredictionFeatures:
        """Extract features for workload prediction."""
        current_time = time.time()
        local_time = time.localtime(current_time)
        
        # Time-based features
        hour_of_day = local_time.tm_hour / 24.0
        day_of_week = local_time.tm_wday / 7.0
        minute_of_hour = local_time.tm_min / 60.0
        
        # Historical trends (if enough history available)
        if len(self.metrics_history) >= 10:
            recent_metrics = list(self.metrics_history)[-10:]
            
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_percent for m in recent_metrics]
            throughput_values = [m.event_throughput_eps for m in recent_metrics]
            latency_values = [m.average_inference_time for m in recent_metrics]
            
            cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]
            memory_trend = np.polyfit(range(len(memory_values)), memory_values, 1)[0]
            throughput_trend = np.polyfit(range(len(throughput_values)), throughput_values, 1)[0]
            latency_trend = np.polyfit(range(len(latency_values)), latency_values, 1)[0]
        else:
            cpu_trend = 0.0
            memory_trend = 0.0
            throughput_trend = 0.0
            latency_trend = 0.0
            
        # Workload characteristics
        event_rate = current_metrics.event_throughput_eps
        processing_complexity = current_metrics.average_inference_time / 100.0  # Normalized
        cache_efficiency = current_metrics.cache_hit_rate
        
        # System state
        queue_depth = current_metrics.inference_queue_size / 100.0  # Normalized
        gpu_utilization = current_metrics.gpu_utilization / 100.0
        
        return PredictionFeatures(
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            minute_of_hour=minute_of_hour,
            cpu_trend=cpu_trend,
            memory_trend=memory_trend,
            throughput_trend=throughput_trend,
            latency_trend=latency_trend,
            event_rate=event_rate,
            processing_complexity=processing_complexity,
            cache_efficiency=cache_efficiency,
            current_workers=self.current_workers,
            queue_depth=queue_depth,
            gpu_utilization=gpu_utilization
        )
        
    def _execute_intelligent_scaling_action(
        self, 
        decision: ScalingDecision, 
        current_metrics: ResourceMetrics
    ):
        """Execute intelligent scaling action."""
        try:
            old_workers = self.current_workers
            new_workers = decision.target_workers
            
            # Record pre-scaling metrics
            pre_scaling_metrics = current_metrics
            
            # Execute scaling
            if decision.action == 'scale_up':
                self._scale_up_intelligent(new_workers - old_workers)
            elif decision.action == 'scale_down':
                self._scale_down_intelligent(old_workers - new_workers)
                
            self.current_workers = new_workers
            self.last_scaling_action = time.time()
            
            # Record scaling decision
            self.scaling_history.append(decision)
            
            self.logger.info(
                f"Executed {decision.action}: {old_workers} -> {new_workers} workers. "
                f"Confidence: {decision.confidence:.2f}. "
                f"Reasoning: {'; '.join(decision.reasoning)}"
            )
            
            # Schedule performance evaluation
            threading.Timer(60.0, self._evaluate_scaling_performance, 
                          args=(decision, pre_scaling_metrics)).start()
            
        except Exception as e:
            self.logger.error(f"Failed to execute scaling action: {e}")
            
    def _scale_up_intelligent(self, worker_count: int):
        """Intelligent scale up with optimal resource allocation."""
        # Pre-warm caches
        self.cache_system.clear_all()  # Clear to ensure fresh state
        
        # Configure GPU processor for additional workers
        # This would typically involve starting new worker threads/processes
        
        self.logger.info(f"Scaled up by {worker_count} workers")
        
    def _scale_down_intelligent(self, worker_count: int):
        """Intelligent scale down with graceful resource cleanup."""
        # Graceful shutdown of workers
        # This would typically involve stopping worker threads/processes
        
        # Optimize remaining resources
        self.gpu_processor.stop_processing()
        time.sleep(1.0)  # Brief pause
        self.gpu_processor.start_processing()
        
        self.logger.info(f"Scaled down by {worker_count} workers")
        
    def _evaluate_scaling_performance(
        self, 
        decision: ScalingDecision, 
        pre_metrics: ResourceMetrics
    ):
        """Evaluate performance after scaling action."""
        try:
            # Wait for system to stabilize
            time.sleep(30.0)
            
            # Collect post-scaling metrics
            post_metrics = self._collect_comprehensive_metrics()
            
            # Calculate improvements
            improvements = {
                'cpu_improvement': pre_metrics.cpu_percent - post_metrics.cpu_percent,
                'memory_improvement': pre_metrics.memory_percent - post_metrics.memory_percent,
                'latency_improvement': pre_metrics.average_inference_time - post_metrics.average_inference_time,
                'throughput_improvement': post_metrics.event_throughput_eps - pre_metrics.event_throughput_eps
            }
            
            # Compare with expected improvements
            accuracy_score = 0.0
            for metric, actual in improvements.items():
                expected_key = metric.replace('_improvement', '_reduction_percent')
                if expected_key in decision.expected_improvement:
                    expected = decision.expected_improvement[expected_key] / 100.0 * getattr(pre_metrics, metric.split('_')[0] + '_percent')
                    accuracy_score += 1.0 - abs(actual - expected) / max(abs(expected), 1.0)
                    
            accuracy_score /= len(improvements)
            
            # Record performance improvement
            performance_record = {
                'decision': decision,
                'pre_metrics': pre_metrics,
                'post_metrics': post_metrics,
                'actual_improvements': improvements,
                'accuracy_score': accuracy_score,
                'timestamp': time.time()
            }
            
            self.performance_improvements.append(performance_record)
            
            self.logger.info(
                f"Scaling performance evaluation - Accuracy: {accuracy_score:.2f}, "
                f"Improvements: {improvements}"
            )
            
        except Exception as e:
            self.logger.error(f"Performance evaluation failed: {e}")
            
    def get_intelligent_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive intelligent scaling statistics."""
        
        # Calculate decision accuracy
        if self.performance_improvements:
            avg_accuracy = np.mean([r['accuracy_score'] for r in self.performance_improvements])
        else:
            avg_accuracy = 0.0
            
        # Prediction model stats
        predictor_stats = {}
        if self.workload_predictor:
            predictor_stats = {
                'models_trained': self.workload_predictor.models_trained,
                'training_samples': len(self.workload_predictor.training_features),
                'last_training': self.workload_predictor.last_training_time
            }
            
        return {
            'intelligent_scaling': {
                'is_running': self.is_running,
                'current_workers': self.current_workers,
                'scaling_decisions_made': len(self.scaling_history),
                'average_decision_accuracy': avg_accuracy,
                'last_scaling_action': self.last_scaling_action,
                'predictions_enabled': self.enable_predictions
            },
            'predictor': predictor_stats,
            'recent_decisions': [
                {
                    'action': d.action,
                    'target_workers': d.target_workers,
                    'confidence': d.confidence,
                    'reasoning': d.reasoning[:2],  # First 2 reasons
                    'timestamp': d.timestamp
                }
                for d in list(self.scaling_history)[-5:]  # Last 5 decisions
            ],
            'utilization_targets': {
                'cpu_target': self.utilization_targets.cpu_target_percent,
                'memory_target': self.utilization_targets.memory_target_percent,
                'latency_target': self.utilization_targets.latency_target_ms,
                'throughput_target': self.utilization_targets.throughput_target_eps
            }
        }


# Global instance
_global_intelligent_autoscaler = None


def get_intelligent_autoscaler() -> IntelligentAutoScaler:
    """Get global intelligent auto-scaler instance."""
    global _global_intelligent_autoscaler
    if _global_intelligent_autoscaler is None:
        _global_intelligent_autoscaler = IntelligentAutoScaler()
    return _global_intelligent_autoscaler