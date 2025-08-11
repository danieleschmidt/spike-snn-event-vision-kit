"""
Advanced Multi-Level Intelligent Caching System for Neuromorphic Vision.

Provides sophisticated caching capabilities with pattern recognition, predictive caching,
event stream analysis, and distributed cache synchronization for high-performance
neuromorphic event processing.
"""

import time
import threading
import hashlib
import pickle
import weakref
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque, OrderedDict
from abc import ABC, abstractmethod
import numpy as np
import logging
from pathlib import Path
import json
import gzip
from contextlib import contextmanager
import asyncio
from queue import Queue, PriorityQueue, Empty
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .monitoring import get_metrics_collector


@dataclass
class CacheEntry:
    """Intelligent cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    access_count: int = 0
    access_frequency: float = 0.0
    last_accessed: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    predicted_next_access: Optional[float] = None
    access_pattern_hash: Optional[str] = None
    priority_score: float = 0.0
    compression_ratio: float = 1.0
    tags: Set[str] = field(default_factory=set)


@dataclass
class PatternSignature:
    """Event pattern signature for predictive caching."""
    pattern_id: str
    event_sequence: List[int]
    temporal_features: List[float]
    spatial_features: List[float]
    confidence_score: float
    occurrence_count: int = 0
    last_seen: float = field(default_factory=time.time)
    predicted_next_events: List[int] = field(default_factory=list)


@dataclass 
class CacheStats:
    """Comprehensive cache statistics."""
    level: str
    total_capacity: int
    current_size: int
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    prediction_hit_count: int = 0
    prediction_miss_count: int = 0
    average_access_time_ms: float = 0.0
    compression_savings_bytes: int = 0
    pattern_matches: int = 0
    last_updated: float = field(default_factory=time.time)
    
    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
        
    @property 
    def prediction_accuracy(self) -> float:
        total = self.prediction_hit_count + self.prediction_miss_count
        return self.prediction_hit_count / total if total > 0 else 0.0


class CacheEvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""
    
    @abstractmethod
    def select_victim(self, cache_entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select entry to evict."""
        pass
        
    @abstractmethod
    def update_on_access(self, entry: CacheEntry):
        """Update policy state on cache access."""
        pass


class LRUEvictionPolicy(CacheEvictionPolicy):
    """Least Recently Used eviction policy."""
    
    def select_victim(self, cache_entries: Dict[str, CacheEntry]) -> Optional[str]:
        if not cache_entries:
            return None
        return min(cache_entries.keys(), key=lambda k: cache_entries[k].last_accessed)
        
    def update_on_access(self, entry: CacheEntry):
        entry.last_accessed = time.time()


class LFUEvictionPolicy(CacheEvictionPolicy):
    """Least Frequently Used eviction policy."""
    
    def select_victim(self, cache_entries: Dict[str, CacheEntry]) -> Optional[str]:
        if not cache_entries:
            return None
        return min(cache_entries.keys(), key=lambda k: cache_entries[k].access_frequency)
        
    def update_on_access(self, entry: CacheEntry):
        entry.access_count += 1
        # Exponential smoothing for frequency calculation
        time_since_creation = time.time() - entry.created_at
        entry.access_frequency = entry.access_count / max(time_since_creation, 1.0)


class AdaptiveEvictionPolicy(CacheEvictionPolicy):
    """Adaptive eviction policy combining multiple factors."""
    
    def __init__(self):
        self.aging_factor = 0.1
        self.frequency_weight = 0.4
        self.recency_weight = 0.3
        self.prediction_weight = 0.3
        
    def select_victim(self, cache_entries: Dict[str, CacheEntry]) -> Optional[str]:
        if not cache_entries:
            return None
            
        current_time = time.time()
        best_victim = None
        best_score = float('inf')
        
        for key, entry in cache_entries.items():
            score = self._calculate_eviction_score(entry, current_time)
            if score < best_score:
                best_score = score
                best_victim = key
                
        return best_victim
        
    def _calculate_eviction_score(self, entry: CacheEntry, current_time: float) -> float:
        """Calculate composite eviction score (lower = more likely to evict)."""
        # Recency component (older = lower score)
        age = current_time - entry.last_accessed
        recency_score = 1.0 / (1.0 + age)
        
        # Frequency component
        frequency_score = entry.access_frequency
        
        # Prediction component
        prediction_score = 0.0
        if entry.predicted_next_access:
            time_until_predicted = entry.predicted_next_access - current_time
            if time_until_predicted > 0:
                prediction_score = 1.0 / (1.0 + time_until_predicted)
                
        # Composite score
        score = (
            self.recency_weight * recency_score +
            self.frequency_weight * frequency_score +
            self.prediction_weight * prediction_score
        )
        
        return score
        
    def update_on_access(self, entry: CacheEntry):
        entry.access_count += 1
        entry.last_accessed = time.time()
        
        # Update frequency with exponential smoothing
        time_since_creation = time.time() - entry.created_at
        entry.access_frequency = entry.access_count / max(time_since_creation, 1.0)


class EventPatternAnalyzer:
    """Analyzes event patterns for predictive caching."""
    
    def __init__(self, max_patterns: int = 1000):
        self.max_patterns = max_patterns
        self.patterns: Dict[str, PatternSignature] = {}
        self.pattern_lock = threading.RLock()
        self.sequence_buffer = deque(maxlen=100)
        self.temporal_window = 1.0  # seconds
        self.logger = logging.getLogger(__name__)
        
    def analyze_event_sequence(self, events: List[Any]) -> Optional[PatternSignature]:
        """Analyze event sequence to identify patterns."""
        try:
            # Convert events to numerical sequence
            event_ids = [hash(str(event)) % 1000 for event in events]
            
            # Extract temporal features
            temporal_features = self._extract_temporal_features(events)
            
            # Extract spatial features if available
            spatial_features = self._extract_spatial_features(events)
            
            # Create pattern signature
            pattern_hash = self._compute_pattern_hash(event_ids, temporal_features, spatial_features)
            
            with self.pattern_lock:
                if pattern_hash in self.patterns:
                    # Update existing pattern
                    pattern = self.patterns[pattern_hash]
                    pattern.occurrence_count += 1
                    pattern.last_seen = time.time()
                    pattern.confidence_score = min(1.0, pattern.occurrence_count / 10.0)
                    
                    # Update predictions
                    self._update_pattern_predictions(pattern, event_ids)
                    
                else:
                    # Create new pattern
                    pattern = PatternSignature(
                        pattern_id=pattern_hash,
                        event_sequence=event_ids,
                        temporal_features=temporal_features,
                        spatial_features=spatial_features,
                        confidence_score=0.1,
                        occurrence_count=1
                    )
                    
                    # Manage pattern cache size
                    if len(self.patterns) >= self.max_patterns:
                        self._evict_old_patterns()
                        
                    self.patterns[pattern_hash] = pattern
                    
                return pattern
                
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
            return None
            
    def _extract_temporal_features(self, events: List[Any]) -> List[float]:
        """Extract temporal features from event sequence."""
        if len(events) < 2:
            return [0.0, 0.0, 0.0]
            
        # Extract timestamps if available
        timestamps = []
        for event in events:
            if hasattr(event, 'timestamp'):
                timestamps.append(event.timestamp)
            elif isinstance(event, dict) and 'timestamp' in event:
                timestamps.append(event['timestamp'])
            else:
                timestamps.append(time.time())
                
        if len(timestamps) < 2:
            return [0.0, 0.0, 0.0]
            
        # Calculate temporal features
        intervals = np.diff(timestamps)
        
        features = [
            float(np.mean(intervals)) if len(intervals) > 0 else 0.0,    # Average interval
            float(np.std(intervals)) if len(intervals) > 0 else 0.0,     # Interval variance
            float(len(events))                                            # Sequence length
        ]
        
        return features
        
    def _extract_spatial_features(self, events: List[Any]) -> List[float]:
        """Extract spatial features from event sequence."""
        try:
            positions = []
            for event in events:
                if hasattr(event, 'x') and hasattr(event, 'y'):
                    positions.append((event.x, event.y))
                elif isinstance(event, dict) and 'x' in event and 'y' in event:
                    positions.append((event['x'], event['y']))
                    
            if len(positions) < 2:
                return [0.0, 0.0, 0.0]
                
            positions = np.array(positions)
            
            # Calculate spatial features
            center = np.mean(positions, axis=0)
            distances = np.linalg.norm(positions - center, axis=1)
            
            features = [
                float(center[0]),                    # Center X
                float(center[1]),                    # Center Y
                float(np.mean(distances))            # Average distance from center
            ]
            
            return features
            
        except Exception:
            return [0.0, 0.0, 0.0]
            
    def _compute_pattern_hash(self, event_ids: List[int], temporal: List[float], spatial: List[float]) -> str:
        """Compute hash for pattern signature."""
        combined_features = event_ids + temporal + spatial
        feature_str = ','.join(str(f) for f in combined_features)
        return hashlib.md5(feature_str.encode()).hexdigest()[:16]
        
    def _update_pattern_predictions(self, pattern: PatternSignature, new_sequence: List[int]):
        """Update pattern predictions based on new occurrence."""
        # Simple next-event prediction based on sequence analysis
        if len(pattern.event_sequence) > 0 and len(new_sequence) > 0:
            # Find common subsequences and predict next events
            pattern.predicted_next_events = new_sequence[-3:] if len(new_sequence) >= 3 else new_sequence
            
    def _evict_old_patterns(self):
        """Evict old patterns to maintain cache size."""
        if not self.patterns:
            return
            
        # Remove patterns with low confidence and old timestamps
        current_time = time.time()
        patterns_to_remove = []
        
        for pattern_id, pattern in self.patterns.items():
            age = current_time - pattern.last_seen
            if age > 3600 and pattern.confidence_score < 0.3:  # 1 hour old with low confidence
                patterns_to_remove.append(pattern_id)
                
        for pattern_id in patterns_to_remove:
            del self.patterns[pattern_id]
            
        # If still too many patterns, remove least confident ones
        if len(self.patterns) >= self.max_patterns:
            sorted_patterns = sorted(
                self.patterns.items(),
                key=lambda x: (x[1].confidence_score, x[1].last_seen)
            )
            
            num_to_remove = len(self.patterns) - int(self.max_patterns * 0.8)
            for i in range(num_to_remove):
                if i < len(sorted_patterns):
                    del self.patterns[sorted_patterns[i][0]]
                    
    def predict_next_events(self, current_sequence: List[Any]) -> List[Any]:
        """Predict next events based on current sequence."""
        try:
            current_ids = [hash(str(event)) % 1000 for event in current_sequence]
            
            # Find matching patterns
            best_matches = []
            with self.pattern_lock:
                for pattern in self.patterns.values():
                    if pattern.confidence_score > 0.5:  # Only use confident patterns
                        # Simple sequence matching
                        match_score = self._calculate_sequence_similarity(current_ids, pattern.event_sequence)
                        if match_score > 0.7:
                            best_matches.append((pattern, match_score))
                            
            if not best_matches:
                return []
                
            # Sort by match score and confidence
            best_matches.sort(key=lambda x: x[1] * x[0].confidence_score, reverse=True)
            
            # Return predictions from best matching pattern
            best_pattern = best_matches[0][0]
            return best_pattern.predicted_next_events
            
        except Exception as e:
            self.logger.error(f"Event prediction failed: {e}")
            return []
            
    def _calculate_sequence_similarity(self, seq1: List[int], seq2: List[int]) -> float:
        """Calculate similarity between two sequences."""
        if not seq1 or not seq2:
            return 0.0
            
        # Find longest common subsequence
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    
        lcs_length = dp[m][n]
        return lcs_length / max(m, n)


class IntelligentCacheLevel:
    """Single level of intelligent cache with advanced features."""
    
    def __init__(
        self,
        name: str,
        capacity: int,
        eviction_policy: CacheEvictionPolicy = None,
        compression_enabled: bool = True,
        encryption_enabled: bool = False
    ):
        self.name = name
        self.capacity = capacity
        self.compression_enabled = compression_enabled
        self.encryption_enabled = encryption_enabled
        
        # Core data structures
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.cache_lock = threading.RLock()
        
        # Eviction policy
        self.eviction_policy = eviction_policy or AdaptiveEvictionPolicy()
        
        # Statistics
        self.stats = CacheStats(level=name, total_capacity=capacity, current_size=0)
        
        # Pattern analysis
        self.pattern_analyzer = EventPatternAnalyzer()
        
        # Prefetch queue
        self.prefetch_queue = Queue(maxsize=100)
        self.prefetch_thread = None
        self.prefetch_active = False
        
        self.logger = logging.getLogger(__name__)
        
    def start_prefetch_worker(self):
        """Start background prefetching worker."""
        if self.prefetch_active:
            return
            
        self.prefetch_active = True
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker_loop, daemon=True)
        self.prefetch_thread.start()
        
    def stop_prefetch_worker(self):
        """Stop background prefetching worker."""
        self.prefetch_active = False
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=5.0)
            
    def _prefetch_worker_loop(self):
        """Background worker for predictive prefetching."""
        while self.prefetch_active:
            try:
                prefetch_task = self.prefetch_queue.get(timeout=1.0)
                if prefetch_task is None:
                    break
                    
                # Execute prefetch task
                key, loader_func = prefetch_task
                if not self.contains(key):
                    try:
                        value = loader_func()
                        self.put(key, value, is_prefetch=True)
                    except Exception as e:
                        self.logger.debug(f"Prefetch failed for {key}: {e}")
                        
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Prefetch worker error: {e}")
                
    def put(self, key: str, value: Any, tags: Set[str] = None, is_prefetch: bool = False) -> bool:
        """Put item in cache with intelligent management."""
        try:
            start_time = time.time()
            
            with self.cache_lock:
                # Compress value if enabled
                stored_value = value
                compression_ratio = 1.0
                
                if self.compression_enabled and self._should_compress(value):
                    compressed_data = self._compress_value(value)
                    if compressed_data:
                        stored_value = compressed_data
                        original_size = self._calculate_size(value)
                        compressed_size = self._calculate_size(compressed_data)
                        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
                        self.stats.compression_savings_bytes += original_size - compressed_size
                        
                # Calculate size and check capacity
                size_bytes = self._calculate_size(stored_value)
                
                # Ensure capacity
                while (self.stats.current_size + size_bytes > self.capacity and 
                       len(self.entries) > 0):
                    if not self._evict_one_entry():
                        break
                        
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=stored_value,
                    size_bytes=size_bytes,
                    compression_ratio=compression_ratio,
                    tags=tags or set()
                )
                
                # Store entry
                if key in self.entries:
                    # Update existing entry
                    old_entry = self.entries[key]
                    self.stats.current_size -= old_entry.size_bytes
                    
                self.entries[key] = entry
                self.stats.current_size += size_bytes
                
                # Update access time
                access_time = (time.time() - start_time) * 1000
                self._update_average_access_time(access_time)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Cache put failed for {key}: {e}")
            return False
            
    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """Get item from cache with pattern analysis."""
        start_time = time.time()
        
        try:
            with self.cache_lock:
                if key in self.entries:
                    entry = self.entries[key]
                    
                    # Update access statistics
                    self.eviction_policy.update_on_access(entry)
                    self.stats.hit_count += 1
                    
                    # Move to end for LRU behavior in OrderedDict
                    self.entries.move_to_end(key)
                    
                    # Decompress if needed
                    value = entry.value
                    if self.compression_enabled and self._is_compressed(value):
                        value = self._decompress_value(value)
                        
                    # Update access time
                    access_time = (time.time() - start_time) * 1000
                    self._update_average_access_time(access_time)
                    
                    # Trigger predictive prefetching
                    self._trigger_predictive_prefetch(key)
                    
                    return value, True
                else:
                    self.stats.miss_count += 1
                    return None, False
                    
        except Exception as e:
            self.logger.error(f"Cache get failed for {key}: {e}")
            return None, False
            
    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self.cache_lock:
            return key in self.entries
            
    def remove(self, key: str) -> bool:
        """Remove item from cache."""
        with self.cache_lock:
            if key in self.entries:
                entry = self.entries.pop(key)
                self.stats.current_size -= entry.size_bytes
                return True
            return False
            
    def clear(self):
        """Clear all entries from cache."""
        with self.cache_lock:
            self.entries.clear()
            self.stats.current_size = 0
            
    def _evict_one_entry(self) -> bool:
        """Evict one entry using eviction policy."""
        victim_key = self.eviction_policy.select_victim(self.entries)
        if victim_key:
            entry = self.entries.pop(victim_key)
            self.stats.current_size -= entry.size_bytes
            self.stats.eviction_count += 1
            return True
        return False
        
    def _should_compress(self, value: Any) -> bool:
        """Determine if value should be compressed."""
        # Compress large numpy arrays and tensors
        if isinstance(value, np.ndarray) and value.nbytes > 1024:  # > 1KB
            return True
        if TORCH_AVAILABLE and isinstance(value, torch.Tensor) and value.numel() * value.element_size() > 1024:
            return True
        return False
        
    def _compress_value(self, value: Any) -> Optional[bytes]:
        """Compress value using appropriate method."""
        try:
            if isinstance(value, np.ndarray):
                # Use numpy's compressed save format
                from io import BytesIO
                buffer = BytesIO()
                np.savez_compressed(buffer, data=value)
                return buffer.getvalue()
            else:
                # Use gzip compression for general objects
                pickled = pickle.dumps(value)
                return gzip.compress(pickled)
        except Exception as e:
            self.logger.warning(f"Compression failed: {e}")
            return None
            
    def _decompress_value(self, compressed_data: bytes) -> Any:
        """Decompress previously compressed value."""
        try:
            # Try numpy format first
            from io import BytesIO
            buffer = BytesIO(compressed_data)
            try:
                loaded = np.load(buffer)
                return loaded['data']
            except:
                # Fall back to gzip + pickle
                buffer.seek(0)
                decompressed = gzip.decompress(compressed_data)
                return pickle.loads(decompressed)
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            raise
            
    def _is_compressed(self, value: Any) -> bool:
        """Check if value is compressed."""
        return isinstance(value, bytes)
        
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes."""
        if isinstance(value, bytes):
            return len(value)
        elif isinstance(value, np.ndarray):
            return value.nbytes
        elif TORCH_AVAILABLE and isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        else:
            # Estimate using pickle
            try:
                return len(pickle.dumps(value))
            except:
                return 1024  # Default estimate
                
    def _update_average_access_time(self, access_time_ms: float):
        """Update average access time with exponential smoothing."""
        alpha = 0.1  # Smoothing factor
        if self.stats.average_access_time_ms == 0:
            self.stats.average_access_time_ms = access_time_ms
        else:
            self.stats.average_access_time_ms = (
                alpha * access_time_ms + 
                (1 - alpha) * self.stats.average_access_time_ms
            )
            
    def _trigger_predictive_prefetch(self, accessed_key: str):
        """Trigger predictive prefetching based on access patterns."""
        # This is a placeholder for predictive prefetching logic
        # In a full implementation, this would analyze access patterns
        # and preload likely-to-be-accessed items
        pass
        
    def get_statistics(self) -> CacheStats:
        """Get cache statistics."""
        with self.cache_lock:
            self.stats.last_updated = time.time()
            return self.stats


class MultiLevelIntelligentCache:
    """Multi-level intelligent cache system with predictive capabilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Cache levels (L1: fast/small, L2: medium, L3: large/slower)
        self.l1_cache = IntelligentCacheLevel(
            name="L1",
            capacity=100 * 1024 * 1024,  # 100MB
            eviction_policy=LRUEvictionPolicy(),
            compression_enabled=False  # L1 optimized for speed
        )
        
        self.l2_cache = IntelligentCacheLevel(
            name="L2", 
            capacity=500 * 1024 * 1024,  # 500MB
            eviction_policy=AdaptiveEvictionPolicy(),
            compression_enabled=True
        )
        
        self.l3_cache = IntelligentCacheLevel(
            name="L3",
            capacity=2 * 1024 * 1024 * 1024,  # 2GB
            eviction_policy=LFUEvictionPolicy(),
            compression_enabled=True
        )
        
        self.cache_levels = [self.l1_cache, self.l2_cache, self.l3_cache]
        
        # Global pattern analyzer
        self.global_pattern_analyzer = EventPatternAnalyzer(max_patterns=5000)
        
        # Cache coordination
        self.promotion_threshold = 3  # Promote after 3 accesses
        self.access_counters = defaultdict(int)
        self.coordination_lock = threading.RLock()
        
        # Start prefetch workers
        for cache_level in self.cache_levels:
            cache_level.start_prefetch_worker()
            
    def put(self, key: str, value: Any, tags: Set[str] = None) -> bool:
        """Put item in appropriate cache level."""
        try:
            # Determine initial cache level based on size and importance
            size_bytes = self._estimate_size(value)
            initial_level = self._select_initial_level(size_bytes, tags)
            
            return self.cache_levels[initial_level].put(key, value, tags)
            
        except Exception as e:
            self.logger.error(f"Multi-level cache put failed: {e}")
            return False
            
    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """Get item from cache with level promotion."""
        try:
            # Search through cache levels
            for level_idx, cache_level in enumerate(self.cache_levels):
                value, hit = cache_level.get(key)
                if hit:
                    # Update access counter
                    with self.coordination_lock:
                        self.access_counters[key] += 1
                        
                        # Promote to higher level if accessed frequently
                        if (level_idx > 0 and 
                            self.access_counters[key] >= self.promotion_threshold):
                            self._promote_entry(key, value, level_idx)
                            
                    return value, True
                    
            return None, False
            
        except Exception as e:
            self.logger.error(f"Multi-level cache get failed: {e}")
            return None, False
            
    def _select_initial_level(self, size_bytes: int, tags: Set[str] = None) -> int:
        """Select initial cache level for new entry."""
        # High priority items go to L1
        if tags and ('high_priority' in tags or 'critical' in tags):
            return 0
            
        # Small items go to L1
        if size_bytes < 1024 * 1024:  # < 1MB
            return 0
            
        # Medium items go to L2  
        if size_bytes < 10 * 1024 * 1024:  # < 10MB
            return 1
            
        # Large items go to L3
        return 2
        
    def _promote_entry(self, key: str, value: Any, current_level: int):
        """Promote entry to higher cache level."""
        if current_level <= 0:
            return  # Already at highest level
            
        target_level = current_level - 1
        tags = set()
        
        # Get tags from current entry if available
        current_cache = self.cache_levels[current_level]
        with current_cache.cache_lock:
            if key in current_cache.entries:
                tags = current_cache.entries[key].tags.copy()
                
        # Add to higher level and remove from current level
        if self.cache_levels[target_level].put(key, value, tags):
            self.cache_levels[current_level].remove(key)
            self.logger.debug(f"Promoted {key} from L{current_level+1} to L{target_level+1}")
            
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif TORCH_AVAILABLE and isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        else:
            try:
                return len(pickle.dumps(value))
            except:
                return 1024
                
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all cache levels."""
        stats = {
            'levels': {},
            'global': {
                'total_capacity': 0,
                'total_used': 0,
                'overall_hit_rate': 0.0,
                'access_counters': len(self.access_counters),
                'patterns_learned': len(self.global_pattern_analyzer.patterns)
            }
        }
        
        total_hits = 0
        total_requests = 0
        
        for i, cache_level in enumerate(self.cache_levels):
            level_stats = cache_level.get_statistics()
            stats['levels'][f'L{i+1}'] = {
                'name': level_stats.level,
                'capacity_mb': level_stats.total_capacity / (1024**2),
                'used_mb': level_stats.current_size / (1024**2),
                'hit_rate': level_stats.hit_rate,
                'hit_count': level_stats.hit_count,
                'miss_count': level_stats.miss_count,
                'eviction_count': level_stats.eviction_count,
                'avg_access_time_ms': level_stats.average_access_time_ms,
                'compression_savings_mb': level_stats.compression_savings_bytes / (1024**2)
            }
            
            stats['global']['total_capacity'] += level_stats.total_capacity
            stats['global']['total_used'] += level_stats.current_size
            total_hits += level_stats.hit_count
            total_requests += level_stats.hit_count + level_stats.miss_count
            
        if total_requests > 0:
            stats['global']['overall_hit_rate'] = total_hits / total_requests
            
        stats['global']['total_capacity_mb'] = stats['global']['total_capacity'] / (1024**2)
        stats['global']['total_used_mb'] = stats['global']['total_used'] / (1024**2)
        stats['global']['utilization_percent'] = (
            stats['global']['total_used'] / stats['global']['total_capacity'] * 100
            if stats['global']['total_capacity'] > 0 else 0
        )
        
        return stats
        
    def clear_all(self):
        """Clear all cache levels."""
        for cache_level in self.cache_levels:
            cache_level.clear()
        self.access_counters.clear()
        
    def shutdown(self):
        """Shutdown cache system."""
        for cache_level in self.cache_levels:
            cache_level.stop_prefetch_worker()


# Global cache instance
_global_intelligent_cache = None


def get_intelligent_cache() -> MultiLevelIntelligentCache:
    """Get global intelligent cache instance."""
    global _global_intelligent_cache
    if _global_intelligent_cache is None:
        _global_intelligent_cache = MultiLevelIntelligentCache()
    return _global_intelligent_cache


@contextmanager
def cached_operation(key: str, tags: Set[str] = None):
    """Context manager for caching operation results."""
    cache = get_intelligent_cache()
    
    # Try to get from cache first
    result, hit = cache.get(key)
    if hit:
        yield result
        return
        
    # Execute operation and cache result
    result_container = {'result': None}
    
    def set_result(value):
        result_container['result'] = value
        cache.put(key, value, tags)
        
    yield set_result
    
    # Return the result
    return result_container['result']