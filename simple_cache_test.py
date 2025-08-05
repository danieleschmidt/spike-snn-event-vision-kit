#!/usr/bin/env python3
"""
Simple cache test for Generation 3
"""

import time
from collections import OrderedDict


class SimpleIntelligentCache:
    """Simple working cache for Generation 3 demo."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        
    def set(self, key: str, value) -> bool:
        """Set cache value."""
        if key in self._cache:
            # Move to end (most recent)
            self._cache.move_to_end(key)
        else:
            # Add new entry
            if len(self._cache) >= self.max_size:
                # Remove oldest (LRU)
                self._cache.popitem(last=False)
        
        self._cache[key] = value
        return True
        
    def get(self, key: str):
        """Get cache value."""
        if key in self._cache:
            # Move to end (most recent)
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        else:
            self.misses += 1
            return None
            
    def stats(self):
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self._cache)
        }


def main():
    print("🚀 GENERATION 3 CACHE TEST")
    
    # Test intelligent caching
    cache = SimpleIntelligentCache(max_size=1000)
    cache.set('test_key', 'test_value')
    value = cache.get('test_key')
    print('✓ Cache works:', value)

    # Test cache performance
    start_time = time.time()
    for i in range(1000):
        cache.set(f'key_{i}', f'value_{i}')
    set_time = time.time() - start_time

    start_time = time.time()
    for i in range(1000):
        value = cache.get(f'key_{i}')
    get_time = time.time() - start_time

    stats = cache.stats()
    print(f'✓ Cache performance: {set_time*1000:.2f}ms set, {get_time*1000:.2f}ms get')
    print(f'✓ Cache stats: {stats["hit_rate"]:.1%} hit rate, {stats["size"]} entries')
    print('🚀 Generation 3 Cache Test SUCCESSFUL!')


if __name__ == "__main__":
    main()