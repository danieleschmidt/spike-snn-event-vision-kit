"""
Intelligent caching system with ML-based optimization for spike-snn-event-vision-kit.

Provides adaptive caching, predictive prefetching, and intelligent cache
management for high-performance neuromorphic vision processing.
"""

import time
import threading
import hashlib
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict, deque
from abc import ABC, abstractmethod
import logging
import json
from pathlib import Path
import weakref
import gc
from contextlib import contextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .monitoring import get_metrics_collector
from .validation import safe_operation, ValidationError


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    hit_score: float = 0.0
    prediction_confidence: float = 0.0
    ttl: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    
    @property
    def age(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStatistics:
    """Comprehensive cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    average_access_time: float = 0.0
    hit_rate: float = 0.0
    prefetch_accuracy: float = 0.0
    memory_efficiency: float = 0.0


class AccessPatternAnalyzer:
    """Analyzes cache access patterns for predictive optimization."""
    
    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        self.access_history = deque(maxlen=history_size)
        self.access_patterns = defaultdict(list)
        self.sequence_patterns = defaultdict(int)
        self.temporal_patterns = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
    
    def record_access(self, key: str, hit: bool):
        """Record cache access for pattern analysis."""
        access_time = time.time()
        self.access_history.append({
            'key': key,
            'timestamp': access_time,
            'hit': hit
        })
        
        # Update access patterns
        self.access_patterns[key].append(access_time)
        
        # Analyze sequential patterns
        if len(self.access_history) >= 2:
            prev_key = self.access_history[-2]['key']
            sequence = f"{prev_key}->{key}"
            self.sequence_patterns[sequence] += 1
        
        # Update temporal patterns
        hour_of_day = int(access_time % 86400 / 3600)  # Hour of day
        self.temporal_patterns[hour_of_day].append(key)
    
    def predict_next_keys(self, current_key: str, num_predictions: int = 5) -> List[Tuple[str, float]]:
        """Predict next likely cache keys based on patterns."""
        predictions = []
        
        # Sequential pattern predictions
        sequence_predictions = {}
        for sequence, count in self.sequence_patterns.items():
            if sequence.startswith(f"{current_key}->"):
                next_key = sequence.split("->")[1]
                confidence = count / sum(1 for s in self.sequence_patterns.keys() 
                                      if s.startswith(f"{current_key}->"))
                sequence_predictions[next_key] = confidence
        
        # Temporal pattern predictions
        current_hour = int(time.time() % 86400 / 3600)
        temporal_keys = self.temporal_patterns.get(current_hour, [])
        temporal_predictions = {}
        
        if temporal_keys:
            key_counts = defaultdict(int)
            for key in temporal_keys[-100:]:  # Recent temporal patterns
                key_counts[key] += 1
            
            total_temporal = sum(key_counts.values())
            for key, count in key_counts.items():
                temporal_predictions[key] = count / total_temporal
        
        # Combine predictions
        combined_predictions = {}
        
        # Weight sequential patterns higher
        for key, confidence in sequence_predictions.items():
            combined_predictions[key] = confidence * 0.7
        
        # Add temporal patterns
        for key, confidence in temporal_predictions.items():
            if key in combined_predictions:
                combined_predictions[key] += confidence * 0.3
            else:
                combined_predictions[key] = confidence * 0.3
        
        # Sort by confidence and return top predictions
        sorted_predictions = sorted(combined_predictions.items(), 
                                  key=lambda x: x[1], reverse=True)
        
        return sorted_predictions[:num_predictions]
    
    def get_access_frequency(self, key: str) -> float:
        """Get access frequency for a key (accesses per hour)."""
        if key not in self.access_patterns:
            return 0.0
        
        accesses = self.access_patterns[key]
        if len(accesses) < 2:
            return 0.0
        
        time_span = accesses[-1] - accesses[0]
        if time_span == 0:
            return 0.0
        
        return len(accesses) / (time_span / 3600)  # Accesses per hour
    
    def get_recency_score(self, key: str) -> float:
        """Get recency score (0-1, higher = more recent)."""
        if key not in self.access_patterns:
            return 0.0
        
        last_access = self.access_patterns[key][-1]
        age = time.time() - last_access
        
        # Exponential decay with 24-hour half-life
        return np.exp(-age / 86400)


class IntelligentLRUCache:
    """LRU cache with ML-based optimization and predictive prefetching."""
    
    def __init__(
        self, 
        max_size_mb: int = 1000,
        max_entries: int = 10000,
        enable_prefetch: bool = True,
        prefetch_threads: int = 2
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.enable_prefetch = enable_prefetch
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.total_size = 0
        
        # Statistics
        self.stats = CacheStatistics()
        
        # Pattern analysis
        self.pattern_analyzer = AccessPatternAnalyzer()
        
        # Prefetching
        self.prefetch_executor = ThreadPoolExecutor(max_workers=prefetch_threads)
        self.prefetch_queue = asyncio.Queue(maxsize=1000) if enable_prefetch else None
        self.prefetch_callbacks: Dict[str, Callable] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background tasks
        self.cleanup_thread = None
        self.prefetch_thread = None
        self.is_running = False
        
        self.logger = logging.getLogger(__name__)
        
        # Start background processing
        self.start()
    
    def start(self):
        """Start background processing threads."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop, 
            daemon=True
        )
        self.cleanup_thread.start()
        
        # Start prefetch thread if enabled
        if self.enable_prefetch:
            self.prefetch_thread = threading.Thread(
                target=self._prefetch_loop,
                daemon=True
            )
            self.prefetch_thread.start()
        
        self.logger.info("Intelligent cache started")
    
    def stop(self, timeout: float = 5.0):
        """Stop background processing."""
        self.is_running = False
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=timeout)
        
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=timeout)
        
        if self.prefetch_executor:
            self.prefetch_executor.shutdown(wait=True, cancel_futures=True)
        
        self.logger.info("Intelligent cache stopped")
    
    @safe_operation
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with intelligent access tracking."""
        start_time = time.time()
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiration
                if entry.is_expired:
                    del self.cache[key]
                    self.total_size -= entry.size_bytes
                    self.stats.misses += 1
                    self.pattern_analyzer.record_access(key, False)
                    return None
                
                # Update access info
                entry.update_access()
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
                # Update statistics
                self.stats.hits += 1
                access_time = (time.time() - start_time) * 1000
                self._update_average_access_time(access_time)
                
                # Record access pattern
                self.pattern_analyzer.record_access(key, True)
                
                # Trigger predictive prefetching
                if self.enable_prefetch:
                    self._trigger_prefetch(key)
                
                return entry.value
            else:
                self.stats.misses += 1
                self.pattern_analyzer.record_access(key, False)
                return None
    
    @safe_operation
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None
    ):
        """Put item in cache with intelligent management."""
        # Calculate size
        size_bytes = self._calculate_size(value)
        
        with self.lock:
            # Check if key already exists
            if key in self.cache:
                old_entry = self.cache[key]
                self.total_size -= old_entry.size_bytes
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl,
                tags=tags or []
            )
            
            # Check size limits
            if size_bytes > self.max_size_bytes:
                self.logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return
            
            # Evict items if necessary
            self._evict_if_necessary(size_bytes)
            
            # Add to cache
            self.cache[key] = entry
            self.total_size += size_bytes
            
            # Update statistics
            self._update_statistics()
    
    def _evict_if_necessary(self, incoming_size: int):
        """Intelligently evict items to make space."""
        # Check entry count limit
        while len(self.cache) >= self.max_entries:
            self._evict_least_valuable()
        
        # Check size limit
        while self.total_size + incoming_size > self.max_size_bytes:
            if not self.cache:
                break
            self._evict_least_valuable()
    
    def _evict_least_valuable(self):
        """Evict least valuable item based on intelligent scoring."""
        if not self.cache:
            return
        
        min_score = float('inf')
        evict_key = None
        
        current_time = time.time()
        
        for key, entry in self.cache.items():
            # Calculate value score based on multiple factors
            frequency_score = self.pattern_analyzer.get_access_frequency(key)
            recency_score = self.pattern_analyzer.get_recency_score(key)
            size_penalty = entry.size_bytes / (1024 * 1024)  # MB
            
            # Combined score (higher is more valuable)
            value_score = (frequency_score * 0.4 + 
                          recency_score * 0.4 + 
                          entry.access_count * 0.2) - size_penalty * 0.1
            
            if value_score < min_score:
                min_score = value_score
                evict_key = key
        
        # Evict the least valuable item
        if evict_key:
            entry = self.cache.pop(evict_key)
            self.total_size -= entry.size_bytes
            self.stats.evictions += 1
            self.logger.debug(f"Evicted cache entry: {evict_key} (score: {min_score:.3f})")
    
    def _trigger_prefetch(self, current_key: str):
        """Trigger predictive prefetching based on access patterns."""
        if not self.enable_prefetch or not self.prefetch_queue:
            return
        
        predictions = self.pattern_analyzer.predict_next_keys(current_key, 3)
        
        for next_key, confidence in predictions:
            if confidence > 0.3 and next_key not in self.cache:  # Only prefetch if confident
                try:
                    # Add to prefetch queue
                    if self.prefetch_queue.qsize() < 100:  # Avoid queue overflow
                        asyncio.create_task(
                            self.prefetch_queue.put((next_key, confidence))
                        )
                except Exception as e:
                    self.logger.debug(f"Failed to queue prefetch for {next_key}: {e}")
    
    def _prefetch_loop(self):
        """Background prefetch processing loop."""
        async def process_prefetch():
            while self.is_running:
                try:
                    if not self.prefetch_queue:
                        await asyncio.sleep(1)
                        continue
                    
                    # Get prefetch request
                    try:
                        key, confidence = await asyncio.wait_for(
                            self.prefetch_queue.get(), timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        continue
                    
                    # Check if key is still needed and not in cache
                    if key in self.cache:
                        continue
                    
                    # Execute prefetch callback if available
                    if key in self.prefetch_callbacks:
                        try:
                            callback = self.prefetch_callbacks[key]
                            value = callback(key)
                            
                            if value is not None:
                                # Add to cache with short TTL for prefetched items
                                self.put(key, value, ttl=300)  # 5 minute TTL
                                self.stats.prefetch_hits += 1
                                self.logger.debug(f"Prefetched {key} with confidence {confidence:.3f}")
                            else:
                                self.stats.prefetch_misses += 1
                                
                        except Exception as e:
                            self.stats.prefetch_misses += 1
                            self.logger.debug(f"Prefetch failed for {key}: {e}")
                    
                except Exception as e:
                    self.logger.error(f"Prefetch processing error: {e}")
        
        # Run async loop in thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(process_prefetch())
        finally:
            loop.close()
    
    def register_prefetch_callback(self, key_pattern: str, callback: Callable[[str], Any]):
        """Register callback for prefetching specific key patterns."""
        self.prefetch_callbacks[key_pattern] = callback
    
    def invalidate_by_tag(self, tag: str):
        """Invalidate all cache entries with specific tag."""
        with self.lock:
            keys_to_remove = []
            for key, entry in self.cache.items():
                if tag in entry.tags:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                entry = self.cache.pop(key)
                self.total_size -= entry.size_bytes
                
            self.logger.info(f"Invalidated {len(keys_to_remove)} entries with tag '{tag}'")
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.total_size = 0
            self.stats = CacheStatistics()
    
    def _cleanup_loop(self):
        """Background cleanup loop for expired entries."""
        while self.is_running:
            try:
                current_time = time.time()
                expired_keys = []
                
                with self.lock:
                    for key, entry in self.cache.items():
                        if entry.is_expired:
                            expired_keys.append(key)
                
                    for key in expired_keys:
                        entry = self.cache.pop(key)
                        self.total_size -= entry.size_bytes
                
                if expired_keys:
                    self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
                
                # Update statistics
                self._update_statistics()
                
                # Sleep for cleanup interval
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                time.sleep(60)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            elif isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (list, tuple, dict)):
                return len(pickle.dumps(value))
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default size estimate
    
    def _update_average_access_time(self, access_time: float):
        """Update running average of access time."""
        if self.stats.average_access_time == 0:
            self.stats.average_access_time = access_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats.average_access_time = (
                alpha * access_time + 
                (1 - alpha) * self.stats.average_access_time
            )
    
    def _update_statistics(self):
        """Update cache statistics."""
        self.stats.entry_count = len(self.cache)
        self.stats.total_size_bytes = self.total_size
        
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hits / total_requests
        
        total_prefetch = self.stats.prefetch_hits + self.stats.prefetch_misses
        if total_prefetch > 0:
            self.stats.prefetch_accuracy = self.stats.prefetch_hits / total_prefetch
        
        if self.max_size_bytes > 0:
            self.stats.memory_efficiency = self.total_size / self.max_size_bytes
    
    def get_statistics(self) -> CacheStatistics:
        """Get current cache statistics."""
        self._update_statistics()
        return self.stats
    
    def optimize_cache(self):
        """Perform cache optimization based on access patterns."""
        with self.lock:
            # Analyze access patterns and optimize cache configuration
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            
            if total_accesses < 100:
                return  # Not enough data for optimization
            
            # Find optimal cache size based on hit rate
            hit_rates_by_size = {}
            
            # Calculate hit rates for different theoretical cache sizes
            entries_by_value = sorted(
                self.cache.items(),
                key=lambda x: (x[1].access_count * self.pattern_analyzer.get_recency_score(x[0])),
                reverse=True
            )
            
            simulated_hits = 0
            cumulative_size = 0
            
            for i, (key, entry) in enumerate(entries_by_value):
                simulated_hits += entry.access_count
                cumulative_size += entry.size_bytes
                
                if cumulative_size <= self.max_size_bytes:
                    hit_rates_by_size[i + 1] = simulated_hits / total_accesses
            
            # Log optimization insights
            optimal_entries = max(hit_rates_by_size.keys()) if hit_rates_by_size else len(self.cache)
            self.logger.info(
                f"Cache optimization: current hit rate {self.stats.hit_rate:.3f}, "
                f"optimal entries ~{optimal_entries}"
            )


class DistributedIntelligentCache:
    """Distributed intelligent cache using Redis backend."""
    
    def __init__(
        self, 
        redis_host: str = "localhost",
        redis_port: int = 6379,
        local_cache_mb: int = 500
    ):
        self.redis_available = REDIS_AVAILABLE
        self.redis_client = None
        
        if self.redis_available:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host, 
                    port=redis_port, 
                    decode_responses=False
                )
                # Test connection
                self.redis_client.ping()
            except Exception as e:
                logging.warning(f"Redis not available: {e}")
                self.redis_available = False
        
        # Always have local cache as fallback
        self.local_cache = IntelligentLRUCache(max_size_mb=local_cache_mb)
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get from distributed cache with local fallback."""
        # Try local cache first
        value = self.local_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis if available
        if self.redis_available and self.redis_client:
            try:
                redis_value = self.redis_client.get(key)
                if redis_value is not None:
                    value = pickle.loads(redis_value)
                    # Store in local cache for faster future access
                    self.local_cache.put(key, value, ttl=300)  # 5 minute local TTL
                    return value
            except Exception as e:
                self.logger.debug(f"Redis get failed for {key}: {e}")
        
        return None
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ):
        """Put to distributed cache."""
        # Always put in local cache
        self.local_cache.put(key, value, ttl=ttl, tags=tags)
        
        # Put in Redis if available
        if self.redis_available and self.redis_client:
            try:
                serialized_value = pickle.dumps(value)
                if ttl:
                    self.redis_client.setex(key, ttl, serialized_value)
                else:
                    self.redis_client.set(key, serialized_value)
            except Exception as e:
                self.logger.debug(f"Redis put failed for {key}: {e}")
    
    def invalidate(self, key: str):
        """Invalidate key from distributed cache."""
        # Remove from local cache
        if key in self.local_cache.cache:
            with self.local_cache.lock:
                entry = self.local_cache.cache.pop(key, None)
                if entry:
                    self.local_cache.total_size -= entry.size_bytes
        
        # Remove from Redis
        if self.redis_available and self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                self.logger.debug(f"Redis delete failed for {key}: {e}")


# Global cache instances
_global_intelligent_cache = None
_global_distributed_cache = None


def get_intelligent_cache() -> IntelligentLRUCache:
    """Get global intelligent cache instance."""
    global _global_intelligent_cache
    if _global_intelligent_cache is None:
        _global_intelligent_cache = IntelligentLRUCache()
    return _global_intelligent_cache


def get_distributed_cache() -> DistributedIntelligentCache:
    """Get global distributed cache instance."""
    global _global_distributed_cache
    if _global_distributed_cache is None:
        _global_distributed_cache = DistributedIntelligentCache()
    return _global_distributed_cache


# Decorator for automatic caching
def intelligent_cache(
    ttl: Optional[float] = None,
    tags: Optional[List[str]] = None,
    cache_instance: Optional[IntelligentLRUCache] = None
):
    """Decorator for intelligent caching of function results."""
    def decorator(func: Callable) -> Callable:
        cache = cache_instance or get_intelligent_cache()
        
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = f"{func.__module__}.{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = hashlib.sha256(key_data.encode()).hexdigest()[:32]
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl=ttl, tags=tags)
            
            return result
        
        return wrapper
    return decorator


# Convenient wrapper classes for easy usage
class IntelligentCache:
    """Simplified wrapper for IntelligentLRUCache."""
    
    def __init__(self, max_size: int = 1000):
        self._cache = IntelligentLRUCache(max_size_mb=max_size)
        
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set cache value."""
        return self._cache.put(key, value, ttl=ttl)
        
    def get(self, key: str) -> Any:
        """Get cache value."""
        return self._cache.get(key)
        
    def delete(self, key: str):
        """Delete cache entry."""
        return self._cache.invalidate(key)
        
    def clear(self):
        """Clear all cache entries."""
        return self._cache.clear()
        
    def stats(self):
        """Get cache statistics."""
        return self._cache.get_statistics()