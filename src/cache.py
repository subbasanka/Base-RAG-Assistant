"""Caching layer for RAG Assistant.

Provides caching for embeddings and query results to improve performance.
"""

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

from cachetools import TTLCache, LRUCache

from src.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def hash_key(*args, **kwargs) -> str:
    """Generate a hash key from arguments.
    
    Args:
        *args: Positional arguments to hash.
        **kwargs: Keyword arguments to hash.
        
    Returns:
        SHA256 hash string.
    """
    key_data = json.dumps(
        {"args": [str(a) for a in args], "kwargs": {k: str(v) for k, v in sorted(kwargs.items())}},
        sort_keys=True,
    )
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


@dataclass
class CacheStats:
    """Cache statistics tracker."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
        }


class EmbeddingCache:
    """Cache for document embeddings.
    
    Uses a combination of in-memory LRU cache and optional disk persistence.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        cache_dir: Path | str | None = None,
        persist: bool = False,
    ):
        """Initialize embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache in memory.
            cache_dir: Directory for disk cache.
            persist: Whether to persist cache to disk.
        """
        self._memory_cache: LRUCache = LRUCache(maxsize=max_size)
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._persist = persist and cache_dir is not None
        self._stats = CacheStats()
        
        if self._persist and self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
    
    def _get_disk_path(self, key: str) -> Path:
        """Get disk path for a cache key."""
        return self._cache_dir / f"emb_{key}.pkl"
    
    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self._cache_dir or not self._cache_dir.exists():
            return
        
        loaded = 0
        for cache_file in self._cache_dir.glob("emb_*.pkl"):
            try:
                with open(cache_file, "rb") as f:
                    key = cache_file.stem.replace("emb_", "")
                    data = pickle.load(f)
                    self._memory_cache[key] = data
                    loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
        
        if loaded:
            logger.info(f"Loaded {loaded} embeddings from disk cache")
    
    def get(self, text: str) -> list[float] | None:
        """Get embedding from cache.
        
        Args:
            text: Text to get embedding for.
            
        Returns:
            Cached embedding or None if not found.
        """
        key = hash_key(text)
        
        # Check memory cache
        if key in self._memory_cache:
            self._stats.hits += 1
            return self._memory_cache[key]
        
        # Check disk cache
        if self._persist:
            disk_path = self._get_disk_path(key)
            if disk_path.exists():
                try:
                    with open(disk_path, "rb") as f:
                        embedding = pickle.load(f)
                        self._memory_cache[key] = embedding
                        self._stats.hits += 1
                        return embedding
                except Exception:
                    pass
        
        self._stats.misses += 1
        return None
    
    def set(self, text: str, embedding: list[float]) -> None:
        """Store embedding in cache.
        
        Args:
            text: Text that was embedded.
            embedding: Embedding vector.
        """
        key = hash_key(text)
        
        # Store in memory
        if len(self._memory_cache) >= self._memory_cache.maxsize:
            self._stats.evictions += 1
        
        self._memory_cache[key] = embedding
        
        # Persist to disk
        if self._persist:
            disk_path = self._get_disk_path(key)
            try:
                with open(disk_path, "wb") as f:
                    pickle.dump(embedding, f)
            except Exception as e:
                logger.warning(f"Failed to persist embedding to disk: {e}")
    
    def get_or_compute(
        self,
        text: str,
        compute_fn: Callable[[str], list[float]],
    ) -> list[float]:
        """Get embedding from cache or compute it.
        
        Args:
            text: Text to embed.
            compute_fn: Function to compute embedding if not cached.
            
        Returns:
            Embedding vector.
        """
        cached = self.get(text)
        if cached is not None:
            return cached
        
        embedding = compute_fn(text)
        self.set(text, embedding)
        return embedding
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._memory_cache.clear()
        
        if self._persist and self._cache_dir:
            for cache_file in self._cache_dir.glob("emb_*.pkl"):
                cache_file.unlink()
        
        self._stats = CacheStats()
        logger.info("Embedding cache cleared")
    
    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats
    
    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._memory_cache)


class QueryCache:
    """Cache for query results with TTL (time-to-live).
    
    Caches retrieval results to avoid repeated searches for identical queries.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,  # 1 hour default
    ):
        """Initialize query cache.
        
        Args:
            max_size: Maximum number of queries to cache.
            ttl_seconds: Time-to-live for cache entries in seconds.
        """
        self._cache: TTLCache = TTLCache(maxsize=max_size, ttl=ttl_seconds)
        self._stats = CacheStats()
    
    def get(self, query: str, top_k: int) -> Any | None:
        """Get cached query results.
        
        Args:
            query: Query string.
            top_k: Number of results requested.
            
        Returns:
            Cached results or None if not found.
        """
        key = hash_key(query, top_k=top_k)
        
        result = self._cache.get(key)
        if result is not None:
            self._stats.hits += 1
            logger.debug(f"Query cache hit: {key}")
            return result
        
        self._stats.misses += 1
        return None
    
    def set(self, query: str, top_k: int, results: Any) -> None:
        """Cache query results.
        
        Args:
            query: Query string.
            top_k: Number of results.
            results: Results to cache.
        """
        key = hash_key(query, top_k=top_k)
        self._cache[key] = results
        logger.debug(f"Query cached: {key}")
    
    def invalidate(self) -> None:
        """Invalidate all cached queries."""
        self._cache.clear()
        self._stats = CacheStats()
        logger.info("Query cache invalidated")
    
    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


# Global cache instances
_embedding_cache: EmbeddingCache | None = None
_query_cache: QueryCache | None = None


def get_embedding_cache(
    max_size: int = 10000,
    cache_dir: Path | str | None = None,
    persist: bool = False,
) -> EmbeddingCache:
    """Get or create global embedding cache."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(
            max_size=max_size,
            cache_dir=cache_dir,
            persist=persist,
        )
    return _embedding_cache


def get_query_cache(
    max_size: int = 1000,
    ttl_seconds: int = 3600,
) -> QueryCache:
    """Get or create global query cache."""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
        )
    return _query_cache

