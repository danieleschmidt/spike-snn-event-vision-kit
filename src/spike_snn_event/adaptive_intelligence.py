"""
Adaptive Intelligence System for Dynamic Algorithm Selection and Self-Optimization.

Generation 3: Scaling Intelligence
- Dynamic algorithm adaptation based on data characteristics
- Machine learning-based performance prediction
- Self-tuning hyperparameters
- Intelligent resource allocation
- Real-time optimization feedback loops
"""

import time
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import pickle
from threading import RLock
import statistics


class AdaptationStrategy(Enum):
    """Algorithm adaptation strategies."""
    CONSERVATIVE = "conservative"    # Slow adaptation, high stability
    BALANCED = "balanced"           # Balanced adaptation rate
    AGGRESSIVE = "aggressive"       # Fast adaptation, may be unstable
    LEARNING = "learning"           # ML-based adaptation


@dataclass
class PerformanceProfile:
    """Performance profile for algorithm/configuration combinations."""
    algorithm_id: str
    config_hash: str
    avg_processing_time: float = 0.0
    avg_memory_usage: float = 0.0
    avg_accuracy: float = 0.0
    execution_count: int = 0
    total_processing_time: float = 0.0
    last_used: float = field(default_factory=time.time)
    stability_score: float = 1.0  # Higher = more stable
    
    def update_performance(self, processing_time: float, memory_usage: float, accuracy: float = 1.0):
        """Update performance metrics with new measurement."""
        self.execution_count += 1
        self.total_processing_time += processing_time
        self.last_used = time.time()
        
        # Exponential moving average for responsive adaptation
        alpha = 0.3  # Learning rate
        if self.execution_count == 1:
            self.avg_processing_time = processing_time
            self.avg_memory_usage = memory_usage
            self.avg_accuracy = accuracy
        else:
            self.avg_processing_time = alpha * processing_time + (1 - alpha) * self.avg_processing_time
            self.avg_memory_usage = alpha * memory_usage + (1 - alpha) * self.avg_memory_usage
            self.avg_accuracy = alpha * accuracy + (1 - alpha) * self.avg_accuracy
        
        # Update stability score based on variance
        self._update_stability_score(processing_time, accuracy)
    
    def _update_stability_score(self, processing_time: float, accuracy: float):
        """Update stability score based on performance variance."""
        # Calculate coefficient of variation for processing time
        time_deviation = abs(processing_time - self.avg_processing_time) / max(self.avg_processing_time, 1e-6)
        accuracy_deviation = abs(accuracy - self.avg_accuracy) / max(self.avg_accuracy, 1e-6)
        
        # Lower deviation = higher stability
        stability_factor = 1.0 / (1.0 + time_deviation + accuracy_deviation)
        
        # Update stability with exponential moving average
        self.stability_score = 0.7 * self.stability_score + 0.3 * stability_factor
    
    def get_performance_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted performance score."""
        if weights is None:
            weights = {'speed': 0.4, 'memory': 0.2, 'accuracy': 0.3, 'stability': 0.1}
        
        # Normalize metrics (higher is better for score)
        speed_score = 1.0 / max(self.avg_processing_time, 1e-6)
        memory_score = 1.0 / max(self.avg_memory_usage, 1e-6)
        accuracy_score = self.avg_accuracy
        stability_score = self.stability_score
        
        # Weighted combination
        total_score = (
            weights['speed'] * speed_score +
            weights['memory'] * memory_score +
            weights['accuracy'] * accuracy_score +
            weights['stability'] * stability_score
        )
        
        return total_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'algorithm_id': self.algorithm_id,
            'config_hash': self.config_hash,
            'avg_processing_time': self.avg_processing_time,
            'avg_memory_usage': self.avg_memory_usage,
            'avg_accuracy': self.avg_accuracy,
            'execution_count': self.execution_count,
            'total_processing_time': self.total_processing_time,
            'last_used': self.last_used,
            'stability_score': self.stability_score
        }


class DataCharacterizer:
    """Analyzes data characteristics for algorithm selection."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def characterize_events(self, events: np.ndarray) -> Dict[str, Any]:
        """Extract characteristics from event data."""
        if len(events) == 0:
            return {'empty': True, 'event_count': 0}
        
        characteristics = {'empty': False, 'event_count': len(events)}
        
        try:
            # Temporal characteristics
            if events.shape[1] >= 3:  # Has timestamp
                timestamps = events[:, 2]
                if len(timestamps) > 1:
                    time_span = timestamps[-1] - timestamps[0]
                    event_rate = len(events) / max(time_span, 1e-9)
                    
                    characteristics.update({
                        'time_span_s': float(time_span),
                        'event_rate_hz': float(event_rate),
                        'temporal_density': 'high' if event_rate > 10000 else 'medium' if event_rate > 1000 else 'low'
                    })
                    
                    # Temporal regularity
                    if len(timestamps) > 2:
                        time_diffs = np.diff(timestamps)
                        time_regularity = 1.0 / (1.0 + np.std(time_diffs) / max(np.mean(time_diffs), 1e-9))
                        characteristics['temporal_regularity'] = float(time_regularity)
            
            # Spatial characteristics
            if events.shape[1] >= 2:  # Has x, y
                x_coords = events[:, 0]
                y_coords = events[:, 1]
                
                x_range = float(np.ptp(x_coords)) if len(x_coords) > 1 else 0.0
                y_range = float(np.ptp(y_coords)) if len(y_coords) > 1 else 0.0
                
                x_density = len(events) / max(x_range, 1.0)
                y_density = len(events) / max(y_range, 1.0)
                
                characteristics.update({
                    'spatial_range_x': x_range,
                    'spatial_range_y': y_range,
                    'spatial_density_x': float(x_density),
                    'spatial_density_y': float(y_density),
                    'spatial_complexity': 'high' if x_range * y_range > 100000 else 'medium' if x_range * y_range > 10000 else 'low'
                })
                
                # Spatial distribution
                if len(events) > 10:
                    x_std = float(np.std(x_coords))
                    y_std = float(np.std(y_coords))
                    spatial_spread = np.sqrt(x_std**2 + y_std**2)
                    characteristics['spatial_spread'] = spatial_spread
            
            # Polarity characteristics
            if events.shape[1] >= 4:  # Has polarity
                polarities = events[:, 3]
                positive = np.sum(polarities > 0)
                negative = np.sum(polarities < 0)
                polarity_balance = min(positive, negative) / max(positive, negative) if max(positive, negative) > 0 else 1.0
                
                characteristics.update({
                    'polarity_balance': float(polarity_balance),
                    'positive_events': int(positive),
                    'negative_events': int(negative)
                })
            
            # Data quality metrics
            characteristics.update({
                'data_completeness': 1.0 - (np.sum(np.isnan(events)) + np.sum(np.isinf(events))) / events.size,
                'data_size_mb': events.nbytes / (1024 * 1024)
            })
            
        except Exception as e:
            self.logger.error(f"Error characterizing events: {e}")
            characteristics['error'] = str(e)
        
        return characteristics


class AlgorithmRegistry:
    """Registry of available algorithms with metadata."""
    
    def __init__(self):
        self.algorithms = {}
        self.logger = logging.getLogger(__name__)
        
        # Register built-in algorithms
        self._register_builtin_algorithms()
    
    def _register_builtin_algorithms(self):
        """Register built-in processing algorithms."""
        
        # Fast algorithm for high-rate data
        self.register_algorithm(
            'fast_vectorized',
            self._fast_vectorized_processing,
            metadata={
                'description': 'Fast vectorized processing for high event rates',
                'best_for': ['high_temporal_density', 'large_datasets'],
                'complexity': 'O(n)',
                'memory_usage': 'low'
            }
        )
        
        # Accurate algorithm for complex spatial patterns
        self.register_algorithm(
            'spatial_accurate',
            self._spatial_accurate_processing,
            metadata={
                'description': 'High-accuracy spatial processing',
                'best_for': ['high_spatial_complexity', 'accuracy_critical'],
                'complexity': 'O(n log n)',
                'memory_usage': 'medium'
            }
        )
        
        # Balanced algorithm for general use
        self.register_algorithm(
            'balanced_general',
            self._balanced_general_processing,
            metadata={
                'description': 'Balanced speed and accuracy for general use',
                'best_for': ['medium_complexity', 'balanced_requirements'],
                'complexity': 'O(n)',
                'memory_usage': 'medium'
            }
        )
        
        # Memory-efficient for large datasets
        self.register_algorithm(
            'memory_efficient',
            self._memory_efficient_processing,
            metadata={
                'description': 'Memory-efficient processing for large datasets',
                'best_for': ['large_datasets', 'memory_constrained'],
                'complexity': 'O(n)',
                'memory_usage': 'low'
            }
        )
    
    def register_algorithm(self, algorithm_id: str, func: Callable, metadata: Dict[str, Any]):
        """Register a new algorithm."""
        self.algorithms[algorithm_id] = {
            'function': func,
            'metadata': metadata,
            'registered_at': time.time()
        }
        self.logger.info(f"Registered algorithm: {algorithm_id}")
    
    def get_algorithm(self, algorithm_id: str) -> Optional[Callable]:
        """Get algorithm function by ID."""
        algo_info = self.algorithms.get(algorithm_id)
        return algo_info['function'] if algo_info else None
    
    def get_metadata(self, algorithm_id: str) -> Dict[str, Any]:
        """Get algorithm metadata."""
        algo_info = self.algorithms.get(algorithm_id)
        return algo_info['metadata'] if algo_info else {}
    
    def list_algorithms(self) -> List[str]:
        """Get list of available algorithm IDs."""
        return list(self.algorithms.keys())
    
    # Built-in algorithm implementations
    def _fast_vectorized_processing(self, events: np.ndarray, config: Dict[str, Any] = None) -> np.ndarray:
        """Fast vectorized processing implementation."""
        if len(events) == 0:
            return events
        
        # Simple vectorized operations
        processed = events.copy()
        
        # Normalize coordinates if needed
        if config and config.get('normalize_coords', False):
            max_x = config.get('max_x', 640)
            max_y = config.get('max_y', 480)
            processed[:, 0] = np.clip(processed[:, 0] / max_x, 0, 1)
            processed[:, 1] = np.clip(processed[:, 1] / max_y, 0, 1)
        
        # Simple temporal smoothing
        if len(processed) > 1 and config and config.get('temporal_smoothing', False):
            kernel_size = config.get('kernel_size', 3)
            if kernel_size > 1 and len(processed) >= kernel_size:
                # Simple moving average on timestamps
                for i in range(kernel_size//2, len(processed) - kernel_size//2):
                    start_idx = i - kernel_size//2
                    end_idx = i + kernel_size//2 + 1
                    processed[i, 2] = np.mean(processed[start_idx:end_idx, 2])
        
        return processed
    
    def _spatial_accurate_processing(self, events: np.ndarray, config: Dict[str, Any] = None) -> np.ndarray:
        """High-accuracy spatial processing implementation."""
        if len(events) == 0:
            return events
        
        processed = events.copy()
        
        # Spatial clustering and filtering
        if config and config.get('spatial_clustering', False):
            cluster_radius = config.get('cluster_radius', 2.0)
            
            # Simple spatial clustering (for demonstration)
            clustered_events = []
            used_indices = set()
            
            for i, event in enumerate(events):
                if i in used_indices:
                    continue
                
                # Find nearby events
                x, y = event[0], event[1]
                distances = np.sqrt((events[:, 0] - x)**2 + (events[:, 1] - y)**2)
                nearby_indices = np.where(distances <= cluster_radius)[0]
                
                # Average the cluster
                if len(nearby_indices) > 1:
                    cluster_center = np.mean(events[nearby_indices], axis=0)
                    clustered_events.append(cluster_center)
                else:
                    clustered_events.append(event)
                
                used_indices.update(nearby_indices)
            
            processed = np.array(clustered_events) if clustered_events else events
        
        return processed
    
    def _balanced_general_processing(self, events: np.ndarray, config: Dict[str, Any] = None) -> np.ndarray:
        """Balanced general processing implementation."""
        if len(events) == 0:
            return events
        
        # Combine fast vectorized ops with some accuracy improvements
        processed = self._fast_vectorized_processing(events, config)
        
        # Add noise filtering
        if config and config.get('noise_filtering', True):
            # Simple noise filter based on temporal consistency
            if len(processed) > 2:
                time_diffs = np.diff(processed[:, 2])
                median_dt = np.median(time_diffs)
                outlier_threshold = config.get('outlier_threshold', 5.0)
                
                # Remove events with very large time jumps
                outliers = np.concatenate([[False], np.abs(time_diffs - median_dt) > outlier_threshold * median_dt])
                processed = processed[~outliers]
        
        return processed
    
    def _memory_efficient_processing(self, events: np.ndarray, config: Dict[str, Any] = None) -> np.ndarray:
        """Memory-efficient processing implementation."""
        if len(events) == 0:
            return events
        
        # Process in chunks to save memory
        chunk_size = config.get('chunk_size', 1000) if config else 1000
        
        processed_chunks = []
        for i in range(0, len(events), chunk_size):
            chunk = events[i:i+chunk_size]
            
            # Basic processing on chunk
            processed_chunk = chunk.copy()
            
            # Simple operations that don't require cross-chunk information
            if config and config.get('coordinate_quantization', False):
                quantize_factor = config.get('quantize_factor', 1.0)
                processed_chunk[:, 0] = np.round(processed_chunk[:, 0] / quantize_factor) * quantize_factor
                processed_chunk[:, 1] = np.round(processed_chunk[:, 1] / quantize_factor) * quantize_factor
            
            processed_chunks.append(processed_chunk)
        
        return np.vstack(processed_chunks) if processed_chunks else events


class AdaptiveIntelligenceEngine:
    """Main adaptive intelligence engine for algorithm selection and optimization."""
    
    def __init__(self, strategy: AdaptationStrategy = AdaptationStrategy.BALANCED):
        self.strategy = strategy
        self.characterizer = DataCharacterizer()
        self.algorithm_registry = AlgorithmRegistry()
        
        # Performance tracking
        self.performance_profiles = {}
        self._performance_lock = RLock()
        
        # Adaptation history
        self.adaptation_history = deque(maxlen=1000)
        
        # Current configuration
        self.current_algorithm = 'balanced_general'
        self.current_config = {}
        
        # Learning parameters
        self.exploration_rate = 0.1  # Probability of trying non-optimal algorithm
        self.min_samples_for_confidence = 10
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize with default performance profiles
        self._initialize_performance_profiles()
    
    def _initialize_performance_profiles(self):
        """Initialize performance profiles for all algorithms."""
        for algo_id in self.algorithm_registry.list_algorithms():
            profile_key = f"{algo_id}_default"
            self.performance_profiles[profile_key] = PerformanceProfile(
                algorithm_id=algo_id,
                config_hash="default"
            )
    
    def process_events_adaptive(self, events: np.ndarray, requirements: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process events with adaptive algorithm selection."""
        start_time = time.time()
        
        # Characterize input data
        characteristics = self.characterizer.characterize_events(events)
        
        # Select best algorithm and configuration
        algorithm_id, config = self._select_algorithm(characteristics, requirements)
        
        # Execute processing
        algorithm_func = self.algorithm_registry.get_algorithm(algorithm_id)
        if not algorithm_func:
            raise ValueError(f"Algorithm not found: {algorithm_id}")
        
        processing_start = time.time()
        processed_events = algorithm_func(events, config)
        processing_time = time.time() - processing_start
        
        # Measure performance
        memory_usage = processed_events.nbytes / (1024 * 1024) if hasattr(processed_events, 'nbytes') else 1.0
        accuracy = self._estimate_accuracy(events, processed_events, characteristics)
        
        # Update performance profile
        self._update_performance_profile(algorithm_id, config, processing_time, memory_usage, accuracy)
        
        # Record adaptation
        adaptation_record = {
            'timestamp': time.time(),
            'algorithm': algorithm_id,
            'config': config.copy(),
            'characteristics': characteristics.copy(),
            'performance': {
                'processing_time': processing_time,
                'memory_usage': memory_usage,
                'accuracy': accuracy
            }
        }
        self.adaptation_history.append(adaptation_record)
        
        # Update current configuration
        self.current_algorithm = algorithm_id
        self.current_config = config
        
        total_time = time.time() - start_time
        
        # Return results with metadata
        metadata = {
            'algorithm_used': algorithm_id,
            'config_used': config,
            'characteristics': characteristics,
            'processing_time': processing_time,
            'total_time': total_time,
            'memory_usage': memory_usage,
            'estimated_accuracy': accuracy
        }
        
        return processed_events, metadata
    
    def _select_algorithm(self, characteristics: Dict[str, Any], requirements: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """Select best algorithm based on data characteristics and requirements."""
        if characteristics.get('empty', False):
            return 'fast_vectorized', {}
        
        # Get performance profiles
        with self._performance_lock:
            profiles = self.performance_profiles.copy()
        
        # Calculate scores for each algorithm
        algorithm_scores = {}
        
        for profile_key, profile in profiles.items():
            if profile.execution_count < 1:
                # No performance data yet, use heuristic scoring
                score = self._heuristic_score(profile.algorithm_id, characteristics, requirements)
            else:
                # Use learned performance data
                score = profile.get_performance_score()
                
                # Adjust score based on suitability to characteristics
                suitability = self._calculate_suitability(profile.algorithm_id, characteristics, requirements)
                score *= suitability
                
                # Apply exploration bonus for less-used algorithms
                if self.strategy == AdaptationStrategy.LEARNING:
                    exploration_bonus = 1.0 + self.exploration_rate / max(profile.execution_count, 1)
                    score *= exploration_bonus
            
            algorithm_scores[profile.algorithm_id] = score
        
        # Select best algorithm
        if self.strategy == AdaptationStrategy.AGGRESSIVE or np.random.random() < self.exploration_rate:
            # Always choose best or explore randomly
            best_algorithm = max(algorithm_scores.keys(), key=lambda k: algorithm_scores[k])
        else:
            # Weighted random selection for more stable adaptation
            algorithms = list(algorithm_scores.keys())
            scores = [algorithm_scores[algo] for algo in algorithms]
            
            # Softmax for probability distribution
            exp_scores = np.exp(np.array(scores) - max(scores))
            probabilities = exp_scores / np.sum(exp_scores)
            
            best_algorithm = np.random.choice(algorithms, p=probabilities)
        
        # Generate configuration for selected algorithm
        config = self._generate_config(best_algorithm, characteristics, requirements)
        
        self.logger.debug(f"Selected algorithm: {best_algorithm} with config: {config}")
        
        return best_algorithm, config
    
    def _heuristic_score(self, algorithm_id: str, characteristics: Dict[str, Any], requirements: Dict[str, Any] = None) -> float:
        """Calculate heuristic score for algorithm based on characteristics."""
        metadata = self.algorithm_registry.get_metadata(algorithm_id)
        score = 1.0  # Base score
        
        # Match algorithm strengths to data characteristics
        best_for = metadata.get('best_for', [])
        
        # Temporal density matching
        temporal_density = characteristics.get('temporal_density', 'medium')
        if 'high_temporal_density' in best_for and temporal_density == 'high':
            score *= 1.5
        
        # Spatial complexity matching
        spatial_complexity = characteristics.get('spatial_complexity', 'medium')
        if 'high_spatial_complexity' in best_for and spatial_complexity == 'high':
            score *= 1.3
        
        # Memory constraints
        data_size = characteristics.get('data_size_mb', 1.0)
        memory_usage = metadata.get('memory_usage', 'medium')
        
        if data_size > 100 and memory_usage == 'low':  # Large dataset, prefer low memory
            score *= 1.4
        elif data_size < 10 and memory_usage == 'high':  # Small dataset, can afford high memory
            score *= 0.8
        
        # Requirements matching
        if requirements:
            priority = requirements.get('priority', 'balanced')
            
            if priority == 'speed' and 'complexity' in metadata:
                if 'O(n)' in metadata['complexity']:
                    score *= 1.3
                elif 'O(n log n)' in metadata['complexity']:
                    score *= 0.8
            
            elif priority == 'accuracy' and 'accuracy_critical' in best_for:
                score *= 1.4
        
        return score
    
    def _calculate_suitability(self, algorithm_id: str, characteristics: Dict[str, Any], requirements: Dict[str, Any] = None) -> float:
        """Calculate how suitable an algorithm is for given characteristics."""
        # This is a simplified suitability calculation
        # In a real system, this could be learned from historical data
        
        metadata = self.algorithm_registry.get_metadata(algorithm_id)
        suitability = 1.0
        
        # Data size suitability
        data_size = characteristics.get('data_size_mb', 1.0)
        memory_usage = metadata.get('memory_usage', 'medium')
        
        if data_size > 50:  # Large dataset
            if memory_usage == 'low':
                suitability *= 1.2
            elif memory_usage == 'high':
                suitability *= 0.7
        
        # Event rate suitability
        event_rate = characteristics.get('event_rate_hz', 1000)
        complexity = metadata.get('complexity', 'O(n)')
        
        if event_rate > 10000:  # High rate
            if complexity == 'O(n)':
                suitability *= 1.1
            elif 'O(n log n)' in complexity:
                suitability *= 0.8
        
        return suitability
    
    def _generate_config(self, algorithm_id: str, characteristics: Dict[str, Any], requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate optimal configuration for selected algorithm."""
        config = {}
        
        # Algorithm-specific configuration
        if algorithm_id == 'fast_vectorized':
            config.update({
                'normalize_coords': characteristics.get('spatial_range_x', 0) > 1000,
                'temporal_smoothing': characteristics.get('temporal_regularity', 1.0) < 0.8,
                'kernel_size': 3 if characteristics.get('event_count', 0) > 1000 else 1
            })
            
        elif algorithm_id == 'spatial_accurate':
            spatial_density = characteristics.get('spatial_density_x', 1.0)
            config.update({
                'spatial_clustering': spatial_density > 100,
                'cluster_radius': max(2.0, 10.0 / spatial_density)
            })
            
        elif algorithm_id == 'balanced_general':
            config.update({
                'noise_filtering': True,
                'outlier_threshold': 3.0 if characteristics.get('temporal_regularity', 1.0) > 0.8 else 5.0,
                'normalize_coords': characteristics.get('spatial_range_x', 0) > 500
            })
            
        elif algorithm_id == 'memory_efficient':
            event_count = characteristics.get('event_count', 1000)
            config.update({
                'chunk_size': min(1000, max(100, event_count // 10)),
                'coordinate_quantization': characteristics.get('spatial_complexity', 'medium') == 'high',
                'quantize_factor': 0.5 if characteristics.get('spatial_spread', 0) > 100 else 1.0
            })
        
        # Apply requirements
        if requirements:
            if requirements.get('priority') == 'speed':
                # Optimize for speed
                config['chunk_size'] = config.get('chunk_size', 1000) * 2
                config['temporal_smoothing'] = False
                
            elif requirements.get('priority') == 'accuracy':
                # Optimize for accuracy
                config['noise_filtering'] = True
                config['spatial_clustering'] = True
                config['outlier_threshold'] = config.get('outlier_threshold', 3.0) * 0.8
        
        return config
    
    def _update_performance_profile(self, algorithm_id: str, config: Dict[str, Any], processing_time: float, memory_usage: float, accuracy: float):
        """Update performance profile for algorithm/config combination."""
        config_hash = self._hash_config(config)
        profile_key = f"{algorithm_id}_{config_hash}"
        
        with self._performance_lock:
            if profile_key not in self.performance_profiles:
                self.performance_profiles[profile_key] = PerformanceProfile(
                    algorithm_id=algorithm_id,
                    config_hash=config_hash
                )
            
            profile = self.performance_profiles[profile_key]
            profile.update_performance(processing_time, memory_usage, accuracy)
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate hash for configuration dictionary."""
        # Simple hash based on sorted config items
        config_str = json.dumps(config, sort_keys=True, default=str)
        return str(hash(config_str))
    
    def _estimate_accuracy(self, original_events: np.ndarray, processed_events: np.ndarray, characteristics: Dict[str, Any]) -> float:
        """Estimate processing accuracy (simplified for demonstration)."""
        if len(original_events) == 0:
            return 1.0
        
        if len(processed_events) == 0:
            return 0.0
        
        # Simple accuracy estimation based on event count preservation
        count_preservation = len(processed_events) / len(original_events)
        
        # Penalize for over-reduction or over-expansion
        if count_preservation > 1.2 or count_preservation < 0.8:
            accuracy = 0.8
        else:
            accuracy = 0.9 + 0.1 * (1.0 - abs(1.0 - count_preservation))
        
        return min(1.0, max(0.0, accuracy))
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics."""
        with self._performance_lock:
            profiles = list(self.performance_profiles.values())
        
        if not profiles:
            return {'total_profiles': 0}
        
        # Algorithm usage statistics
        algo_usage = defaultdict(int)
        for profile in profiles:
            algo_usage[profile.algorithm_id] += profile.execution_count
        
        # Performance statistics
        avg_processing_time = statistics.mean(p.avg_processing_time for p in profiles if p.execution_count > 0)
        avg_accuracy = statistics.mean(p.avg_accuracy for p in profiles if p.execution_count > 0)
        
        # Best performing algorithm
        best_profile = max(profiles, key=lambda p: p.get_performance_score() if p.execution_count > 0 else 0)
        
        return {
            'total_profiles': len(profiles),
            'algorithm_usage': dict(algo_usage),
            'current_algorithm': self.current_algorithm,
            'current_config': self.current_config.copy(),
            'strategy': self.strategy.value,
            'exploration_rate': self.exploration_rate,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'avg_accuracy': avg_accuracy,
            'best_algorithm': best_profile.algorithm_id if profiles else None,
            'best_algorithm_score': best_profile.get_performance_score() if profiles else 0,
            'adaptation_history_size': len(self.adaptation_history)
        }
    
    def save_learned_profiles(self, filepath: str):
        """Save learned performance profiles to file."""
        try:
            with self._performance_lock:
                profiles_data = {
                    key: profile.to_dict() 
                    for key, profile in self.performance_profiles.items()
                }
            
            with open(filepath, 'wb') as f:
                pickle.dump(profiles_data, f)
            
            self.logger.info(f"Saved {len(profiles_data)} performance profiles to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save profiles: {e}")
    
    def load_learned_profiles(self, filepath: str):
        """Load learned performance profiles from file."""
        try:
            with open(filepath, 'rb') as f:
                profiles_data = pickle.load(f)
            
            with self._performance_lock:
                for key, profile_dict in profiles_data.items():
                    profile = PerformanceProfile(
                        algorithm_id=profile_dict['algorithm_id'],
                        config_hash=profile_dict['config_hash'],
                        avg_processing_time=profile_dict['avg_processing_time'],
                        avg_memory_usage=profile_dict['avg_memory_usage'],
                        avg_accuracy=profile_dict['avg_accuracy'],
                        execution_count=profile_dict['execution_count'],
                        total_processing_time=profile_dict['total_processing_time'],
                        last_used=profile_dict['last_used'],
                        stability_score=profile_dict['stability_score']
                    )
                    self.performance_profiles[key] = profile
            
            self.logger.info(f"Loaded {len(profiles_data)} performance profiles from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load profiles: {e}")


# Example usage and demonstration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create adaptive intelligence engine
    engine = AdaptiveIntelligenceEngine(AdaptationStrategy.LEARNING)
    
    # Generate test scenarios
    test_scenarios = [
        # High-rate, simple spatial pattern
        {
            'events': np.random.rand(5000, 4) * [640, 480, 10.0, 2] - [0, 0, 0, 1],
            'requirements': {'priority': 'speed'}
        },
        # Low-rate, complex spatial pattern
        {
            'events': np.random.rand(500, 4) * [1920, 1080, 100.0, 2] - [0, 0, 0, 1],
            'requirements': {'priority': 'accuracy'}
        },
        # Medium complexity, balanced requirements
        {
            'events': np.random.rand(2000, 4) * [800, 600, 5.0, 2] - [0, 0, 0, 1],
            'requirements': {'priority': 'balanced'}
        }
    ]
    
    print("Running adaptive intelligence demonstration...")
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n--- Scenario {i+1} ---")
        
        events = scenario['events'].astype(np.float32)
        # Sort by timestamp for realistic temporal ordering
        events = events[np.argsort(events[:, 2])]
        
        processed_events, metadata = engine.process_events_adaptive(
            events, 
            scenario['requirements']
        )
        
        print(f"Algorithm used: {metadata['algorithm_used']}")
        print(f"Processing time: {metadata['processing_time']*1000:.2f}ms")
        print(f"Memory usage: {metadata['memory_usage']:.2f}MB")
        print(f"Estimated accuracy: {metadata['estimated_accuracy']:.3f}")
        print(f"Events: {len(events)} → {len(processed_events)}")
    
    # Show adaptation statistics
    print(f"\n--- Adaptation Statistics ---")
    stats = engine.get_adaptation_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nAdaptive intelligence demonstration completed!")