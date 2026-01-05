"""Tests for caching module."""

from pathlib import Path

import pytest

from src.cache import (
    EmbeddingCache,
    QueryCache,
    hash_key,
    CacheStats,
)


class TestHashKey:
    """Tests for cache key hashing."""
    
    def test_same_input_same_hash(self):
        """Test identical inputs produce same hash."""
        hash1 = hash_key("hello", "world", key="value")
        hash2 = hash_key("hello", "world", key="value")
        assert hash1 == hash2
    
    def test_different_input_different_hash(self):
        """Test different inputs produce different hashes."""
        hash1 = hash_key("hello")
        hash2 = hash_key("world")
        assert hash1 != hash2
    
    def test_order_matters(self):
        """Test argument order affects hash."""
        hash1 = hash_key("a", "b")
        hash2 = hash_key("b", "a")
        assert hash1 != hash2


class TestCacheStats:
    """Tests for cache statistics."""
    
    def test_initial_stats(self):
        """Test initial statistics are zero."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=7, misses=3)
        assert stats.hit_rate == 0.7
    
    def test_hit_rate_zero_total(self):
        """Test hit rate with no requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = CacheStats(hits=5, misses=5, evictions=2)
        d = stats.to_dict()
        
        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["evictions"] == 2
        assert d["hit_rate"] == 0.5


class TestEmbeddingCache:
    """Tests for embedding cache."""
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = EmbeddingCache(max_size=100)
        result = cache.get("nonexistent")
        
        assert result is None
        assert cache.stats.misses == 1
    
    def test_cache_hit(self):
        """Test cache hit returns stored value."""
        cache = EmbeddingCache(max_size=100)
        embedding = [0.1, 0.2, 0.3]
        
        cache.set("test", embedding)
        result = cache.get("test")
        
        assert result == embedding
        assert cache.stats.hits == 1
    
    def test_get_or_compute(self):
        """Test get_or_compute pattern."""
        cache = EmbeddingCache(max_size=100)
        compute_count = [0]
        
        def compute(text: str) -> list:
            compute_count[0] += 1
            return [0.1, 0.2, 0.3]
        
        # First call computes
        result1 = cache.get_or_compute("test", compute)
        assert compute_count[0] == 1
        
        # Second call uses cache
        result2 = cache.get_or_compute("test", compute)
        assert compute_count[0] == 1  # Not incremented
        
        assert result1 == result2
    
    def test_cache_size_limit(self):
        """Test cache respects size limit."""
        cache = EmbeddingCache(max_size=3)
        
        for i in range(5):
            cache.set(f"key{i}", [float(i)])
        
        assert cache.size <= 3
    
    def test_clear_cache(self):
        """Test clearing cache."""
        cache = EmbeddingCache(max_size=100)
        cache.set("test", [0.1])
        
        cache.clear()
        
        assert cache.size == 0
        assert cache.get("test") is None


class TestQueryCache:
    """Tests for query cache."""
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = QueryCache(max_size=100)
        result = cache.get("nonexistent", 5)
        
        assert result is None
        assert cache.stats.misses == 1
    
    def test_cache_hit(self):
        """Test cache hit returns stored value."""
        cache = QueryCache(max_size=100)
        results = [{"content": "test"}]
        
        cache.set("query", 5, results)
        retrieved = cache.get("query", 5)
        
        assert retrieved == results
        assert cache.stats.hits == 1
    
    def test_different_top_k_different_cache(self):
        """Test different top_k values are cached separately."""
        cache = QueryCache(max_size=100)
        
        cache.set("query", 5, ["result5"])
        cache.set("query", 10, ["result10"])
        
        assert cache.get("query", 5) == ["result5"]
        assert cache.get("query", 10) == ["result10"]
    
    def test_invalidate(self):
        """Test cache invalidation."""
        cache = QueryCache(max_size=100)
        cache.set("query", 5, ["result"])
        
        cache.invalidate()
        
        assert cache.get("query", 5) is None

