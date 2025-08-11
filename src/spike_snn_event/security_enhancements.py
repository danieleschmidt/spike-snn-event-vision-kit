"""
Advanced security enhancements for the neuromorphic vision system.

This module provides adversarial defense mechanisms and memory safety
management for production-ready deployment.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np

from .security import InputSanitizer, SecurityError


class AdversarialDefense:
    """Defense mechanisms against adversarial attacks on SNNs."""
    
    def __init__(self, enable_preprocessing: bool = True, enable_detection: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enable_preprocessing = enable_preprocessing
        self.enable_detection = enable_detection
        self.input_sanitizer = InputSanitizer()
        
        # Attack detection thresholds
        self.max_event_density = 1000  # events per spatial region
        self.max_temporal_frequency = 50000  # Hz
        self.noise_threshold = 0.3  # fraction of noisy events
        
        # Statistics for anomaly detection
        self.baseline_stats = None
        self.attack_detection_history = []
        
    def defend_against_adversarial_events(self, events: List[List[float]], 
                                        sensor_width: int = 128, sensor_height: int = 128) -> List[List[float]]:
        """Apply defense mechanisms to event stream."""
        if not events:
            return events
            
        # Input validation
        clean_events = self._validate_event_structure(events)
        
        # Adversarial detection
        if self.enable_detection:
            is_attack = self._detect_adversarial_pattern(clean_events, sensor_width, sensor_height)
            if is_attack:
                self.logger.warning("Potential adversarial attack detected in event stream")
                # Apply stronger filtering under attack
                clean_events = self._apply_attack_mitigation(clean_events, sensor_width, sensor_height)
                
        # Preprocessing defenses
        if self.enable_preprocessing:
            clean_events = self._apply_preprocessing_defenses(clean_events, sensor_width, sensor_height)
            
        return clean_events
        
    def _validate_event_structure(self, events: List[List[float]]) -> List[List[float]]:
        """Validate and sanitize event structure."""
        clean_events = []
        
        for i, event in enumerate(events):
            try:
                # Validate event format
                if len(event) != 4:
                    continue
                    
                x, y, t, p = event
                
                # Sanitize coordinates
                x = self.input_sanitizer.sanitize_numeric_input(x, 0, 10000, f"event[{i}].x")
                y = self.input_sanitizer.sanitize_numeric_input(y, 0, 10000, f"event[{i}].y")
                t = self.input_sanitizer.sanitize_numeric_input(t, 0, None, f"event[{i}].t")
                
                # Validate polarity
                if p not in [-1, 0, 1]:
                    p = 1 if p > 0 else -1
                    
                clean_events.append([x, y, t, p])
                
            except SecurityError as e:
                self.logger.warning(f"Dropping malformed event {i}: {e}")
                continue
                
        return clean_events
        
    def _detect_adversarial_pattern(self, events: List[List[float]], 
                                  sensor_width: int, sensor_height: int) -> bool:
        """Detect adversarial patterns in event stream."""
        if len(events) < 10:
            return False
            
        # Calculate event statistics
        stats = self._calculate_event_statistics(events, sensor_width, sensor_height)
        
        # Check for anomalies
        anomaly_flags = []
        
        # High event density attack
        if stats['max_pixel_density'] > self.max_event_density:
            anomaly_flags.append("HIGH_DENSITY")
            
        # Temporal frequency attack
        if stats['temporal_frequency'] > self.max_temporal_frequency:
            anomaly_flags.append("HIGH_FREQUENCY")
            
        # Noise injection attack
        if stats['noise_ratio'] > self.noise_threshold:
            anomaly_flags.append("HIGH_NOISE")
            
        # Spatial distribution anomaly
        if stats['spatial_entropy'] < 0.5:  # Very low entropy = suspicious concentration
            anomaly_flags.append("SPATIAL_CONCENTRATION")
            
        # Temporal pattern anomaly
        if stats['temporal_regularity'] > 0.9:  # Too regular = synthetic
            anomaly_flags.append("TEMPORAL_REGULARITY")
            
        # Update detection history
        detection_result = {
            'timestamp': time.time(),
            'anomaly_flags': anomaly_flags,
            'stats': stats,
            'is_attack': len(anomaly_flags) >= 2  # Multiple indicators = likely attack
        }
        
        self.attack_detection_history.append(detection_result)
        
        # Keep only recent history
        if len(self.attack_detection_history) > 100:
            self.attack_detection_history = self.attack_detection_history[-100:]
            
        return detection_result['is_attack']
        
    def _calculate_event_statistics(self, events: List[List[float]], 
                                  sensor_width: int, sensor_height: int) -> Dict[str, float]:
        """Calculate statistical features for anomaly detection."""
        if not events:
            return {}
            
        # Extract components
        x_coords = [e[0] for e in events]
        y_coords = [e[1] for e in events]
        timestamps = [e[2] for e in events]
        polarities = [e[3] for e in events]
        
        # Spatial statistics
        spatial_bins = np.zeros((sensor_height, sensor_width))
        for x, y in zip(x_coords, y_coords):
            if 0 <= int(x) < sensor_width and 0 <= int(y) < sensor_height:
                spatial_bins[int(y), int(x)] += 1
                
        max_pixel_density = np.max(spatial_bins)
        spatial_entropy = self._calculate_entropy(spatial_bins.flatten())
        
        # Temporal statistics
        time_span = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
        temporal_frequency = len(events) / max(time_span, 1e-6)
        
        # Calculate temporal regularity (how uniform are inter-event intervals)
        if len(timestamps) > 2:
            intervals = np.diff(sorted(timestamps))
            temporal_regularity = 1.0 - np.std(intervals) / max(np.mean(intervals), 1e-6)
        else:
            temporal_regularity = 0.0
            
        # Noise ratio estimation (based on polarity switching frequency)
        polarity_switches = sum(1 for i in range(1, len(polarities)) 
                              if polarities[i] != polarities[i-1])
        noise_ratio = polarity_switches / max(len(polarities) - 1, 1)
        
        return {
            'max_pixel_density': max_pixel_density,
            'spatial_entropy': spatial_entropy,
            'temporal_frequency': temporal_frequency,
            'temporal_regularity': temporal_regularity,
            'noise_ratio': noise_ratio,
            'total_events': len(events),
            'time_span': time_span
        }
        
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of data distribution."""
        # Add small epsilon to avoid log(0)
        data = data + 1e-10
        probabilities = data / np.sum(data)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
        
    def _apply_attack_mitigation(self, events: List[List[float]], 
                               sensor_width: int, sensor_height: int) -> List[List[float]]:
        """Apply strong filtering when attack is detected."""
        if not events:
            return events
            
        mitigated_events = []
        
        # Track recent events per pixel
        pixel_counts = {}
        recent_time_window = 0.01  # 10ms window
        
        for event in events:
            x, y, t, p = event
            pixel = (int(x), int(y))
            
            # Clean old events from pixel history
            if pixel not in pixel_counts:
                pixel_counts[pixel] = []
                
            pixel_counts[pixel] = [timestamp for timestamp in pixel_counts[pixel] 
                                 if t - timestamp < recent_time_window]
            
            # Apply rate limiting per pixel
            if len(pixel_counts[pixel]) < 10:  # Max 10 events per pixel per 10ms
                mitigated_events.append(event)
                pixel_counts[pixel].append(t)
            else:
                # Drop excessive events from hot pixels
                continue
                
        self.logger.info(f"Attack mitigation: {len(events)} -> {len(mitigated_events)} events")
        return mitigated_events
        
    def _apply_preprocessing_defenses(self, events: List[List[float]], 
                                    sensor_width: int, sensor_height: int) -> List[List[float]]:
        """Apply preprocessing defenses (noise filtering, normalization)."""
        if not events:
            return events
            
        # Apply temporal smoothing
        smoothed_events = self._apply_temporal_smoothing(events)
        
        # Apply spatial denoising
        denoised_events = self._apply_spatial_denoising(smoothed_events, sensor_width, sensor_height)
        
        # Normalize event density
        normalized_events = self._normalize_event_density(denoised_events)
        
        return normalized_events
        
    def _apply_temporal_smoothing(self, events: List[List[float]]) -> List[List[float]]:
        """Apply temporal smoothing to reduce high-frequency noise."""
        if len(events) < 3:
            return events
            
        smoothed = []
        window_size = 3
        
        for i in range(len(events)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(events), i + window_size // 2 + 1)
            
            window_events = events[start_idx:end_idx]
            if len(window_events) >= 2:
                # Keep event if it's consistent with neighbors
                x, y, t, p = events[i]
                neighbor_x = np.mean([e[0] for e in window_events])
                neighbor_y = np.mean([e[1] for e in window_events])
                
                # Check spatial consistency
                spatial_distance = np.sqrt((x - neighbor_x)**2 + (y - neighbor_y)**2)
                if spatial_distance < 5.0:  # Within 5 pixels
                    smoothed.append(events[i])
            else:
                smoothed.append(events[i])
                
        return smoothed
        
    def _apply_spatial_denoising(self, events: List[List[float]], 
                               sensor_width: int, sensor_height: int) -> List[List[float]]:
        """Apply spatial denoising to remove isolated events."""
        if len(events) < 5:
            return events
            
        # Group events by spatial proximity
        spatial_groups = {}
        group_radius = 2  # pixels
        
        for event in events:
            x, y, t, p = event
            grid_x, grid_y = int(x // group_radius), int(y // group_radius)
            grid_key = (grid_x, grid_y)
            
            if grid_key not in spatial_groups:
                spatial_groups[grid_key] = []
            spatial_groups[grid_key].append(event)
            
        # Keep events from groups with sufficient activity
        denoised = []
        min_group_size = 2
        
        for group_events in spatial_groups.values():
            if len(group_events) >= min_group_size:
                denoised.extend(group_events)
                
        return denoised
        
    def _normalize_event_density(self, events: List[List[float]]) -> List[List[float]]:
        """Normalize event density to prevent overwhelming downstream processing."""
        if len(events) <= 1000:  # Small streams don't need normalization
            return events
            
        # Sample events to maintain reasonable density
        target_density = 1000
        sampling_ratio = target_density / len(events)
        
        if sampling_ratio >= 1.0:
            return events
            
        # Uniform sampling
        sampled_events = []
        for i, event in enumerate(events):
            if np.random.random() < sampling_ratio:
                sampled_events.append(event)
                
        self.logger.info(f"Event density normalization: {len(events)} -> {len(sampled_events)} events")
        return sampled_events
        
    def get_defense_stats(self) -> Dict[str, Any]:
        """Get adversarial defense statistics."""
        recent_attacks = [h for h in self.attack_detection_history 
                         if time.time() - h['timestamp'] < 300]  # Last 5 minutes
        
        return {
            'total_detections': len(self.attack_detection_history),
            'recent_attacks': len([h for h in recent_attacks if h['is_attack']]),
            'defense_enabled': self.enable_preprocessing and self.enable_detection,
            'detection_rate': len([h for h in recent_attacks if h['is_attack']]) / max(1, len(recent_attacks)),
            'last_detection_time': self.attack_detection_history[-1]['timestamp'] if self.attack_detection_history else None
        }


class MemorySafetyManager:
    """Memory safety management for large-scale processing."""
    
    def __init__(self, max_memory_gb: float = 8.0, monitor_interval: float = 1.0):
        self.logger = logging.getLogger(__name__)
        self.max_memory_gb = max_memory_gb
        self.monitor_interval = monitor_interval
        self.memory_warnings = []
        self.last_cleanup_time = time.time()
        
        # Memory tracking
        self._memory_history = []
        self._large_allocations = {}
        
    def safe_allocate(self, size_bytes: int, description: str = "unknown") -> bool:
        """Check if allocation is safe before proceeding."""
        current_memory = self._get_memory_usage_gb()
        allocation_gb = size_bytes / (1024**3)
        
        if current_memory + allocation_gb > self.max_memory_gb:
            self.logger.warning(
                f"Memory allocation blocked: {allocation_gb:.2f}GB would exceed "
                f"limit {self.max_memory_gb}GB (current: {current_memory:.2f}GB)"
            )
            return False
            
        # Track large allocations
        if allocation_gb > 0.1:  # Track allocations > 100MB
            allocation_id = f"{description}_{time.time()}"
            self._large_allocations[allocation_id] = {
                'size_gb': allocation_gb,
                'timestamp': time.time(),
                'description': description
            }
            
        return True
        
    def monitor_memory_usage(self) -> Dict[str, Any]:
        """Monitor current memory usage and return statistics."""
        current_memory = self._get_memory_usage_gb()
        
        # Add to history
        self._memory_history.append({
            'timestamp': time.time(),
            'memory_gb': current_memory,
            'usage_percent': (current_memory / self.max_memory_gb) * 100
        })
        
        # Keep only recent history
        cutoff_time = time.time() - 300  # 5 minutes
        self._memory_history = [h for h in self._memory_history if h['timestamp'] > cutoff_time]
        
        # Check for memory warnings
        usage_percent = (current_memory / self.max_memory_gb) * 100
        
        if usage_percent > 90:
            warning = f"Critical memory usage: {usage_percent:.1f}%"
            self.memory_warnings.append(warning)
            self.logger.error(warning)
        elif usage_percent > 80:
            warning = f"High memory usage: {usage_percent:.1f}%"
            self.memory_warnings.append(warning)
            self.logger.warning(warning)
            
        # Cleanup old warnings
        if len(self.memory_warnings) > 50:
            self.memory_warnings = self.memory_warnings[-25:]
            
        return {
            'current_memory_gb': current_memory,
            'max_memory_gb': self.max_memory_gb,
            'usage_percent': usage_percent,
            'memory_available_gb': self.max_memory_gb - current_memory,
            'large_allocations': len(self._large_allocations),
            'recent_warnings': len(self.memory_warnings),
            'memory_trend': self._calculate_memory_trend()
        }
        
    def force_cleanup(self) -> Dict[str, Any]:
        """Force memory cleanup operations."""
        cleanup_start = time.time()
        initial_memory = self._get_memory_usage_gb()
        
        # Python garbage collection
        import gc
        collected = gc.collect()
        
        # Clear old large allocations tracking
        current_time = time.time()
        old_allocations = [
            k for k, v in self._large_allocations.items()
            if current_time - v['timestamp'] > 600  # 10 minutes
        ]
        for k in old_allocations:
            del self._large_allocations[k]
            
        # Torch cache cleanup if available
        cuda_cache_cleared = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                cuda_cache_cleared = True
        except:
            pass
            
        final_memory = self._get_memory_usage_gb()
        cleanup_duration = time.time() - cleanup_start
        memory_freed = initial_memory - final_memory
        
        self.last_cleanup_time = time.time()
        
        result = {
            'initial_memory_gb': initial_memory,
            'final_memory_gb': final_memory,
            'memory_freed_gb': memory_freed,
            'cleanup_duration_s': cleanup_duration,
            'gc_collected': collected,
            'cuda_cache_cleared': cuda_cache_cleared
        }
        
        self.logger.info(f"Memory cleanup: freed {memory_freed:.2f}GB in {cleanup_duration:.2f}s")
        return result
        
    def _get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024**3)  # Convert bytes to GB
        except ImportError:
            # Fallback for systems without psutil
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)  # KB to GB on Linux
            
    def _calculate_memory_trend(self) -> str:
        """Calculate memory usage trend over recent history."""
        if len(self._memory_history) < 2:
            return "insufficient_data"
            
        recent_memory = [h['memory_gb'] for h in self._memory_history[-10:]]  # Last 10 samples
        
        if len(recent_memory) < 2:
            return "stable"
            
        # Simple trend calculation
        start_memory = np.mean(recent_memory[:len(recent_memory)//2])
        end_memory = np.mean(recent_memory[len(recent_memory)//2:])
        
        change_percent = ((end_memory - start_memory) / start_memory) * 100 if start_memory > 0 else 0
        
        if change_percent > 10:
            return "increasing"
        elif change_percent < -10:
            return "decreasing"
        else:
            return "stable"


# Global instances
_global_adversarial_defense = None
_global_memory_safety_manager = None


def get_adversarial_defense() -> AdversarialDefense:
    """Get global adversarial defense instance."""
    global _global_adversarial_defense
    if _global_adversarial_defense is None:
        _global_adversarial_defense = AdversarialDefense()
    return _global_adversarial_defense


def get_memory_safety_manager() -> MemorySafetyManager:
    """Get global memory safety manager instance."""
    global _global_memory_safety_manager
    if _global_memory_safety_manager is None:
        _global_memory_safety_manager = MemorySafetyManager()
    return _global_memory_safety_manager