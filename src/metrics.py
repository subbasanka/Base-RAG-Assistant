"""Prometheus metrics for RAG Assistant.

Provides observability metrics for monitoring in production.
Handles re-registration gracefully for environments like Streamlit.
"""

import os
from typing import Any

# Check if metrics should be disabled (e.g., in Streamlit)
METRICS_ENABLED = os.environ.get("RAG_METRICS_ENABLED", "true").lower() == "true"

# Flag to track if metrics have been initialized
_metrics_initialized = False


class NoOpMetric:
    """No-op metric that does nothing (used when metrics are disabled)."""
    
    def labels(self, *args, **kwargs) -> "NoOpMetric":
        return self
    
    def inc(self, amount: float = 1) -> None:
        pass
    
    def dec(self, amount: float = 1) -> None:
        pass
    
    def set(self, value: float) -> None:
        pass
    
    def observe(self, amount: float) -> None:
        pass
    
    def info(self, val: dict) -> None:
        pass


class NoOpCollector:
    """No-op collector for context managers."""
    
    def __enter__(self) -> "NoOpCollector":
        return self
    
    def __exit__(self, *args) -> bool:
        return False


# Initialize with no-op metrics by default
app_info: Any = NoOpMetric()
query_counter: Any = NoOpMetric()
query_latency: Any = NoOpMetric()
retrieval_counter: Any = NoOpMetric()
retrieval_latency: Any = NoOpMetric()
chunks_retrieved: Any = NoOpMetric()
llm_request_counter: Any = NoOpMetric()
llm_latency: Any = NoOpMetric()
llm_tokens_used: Any = NoOpMetric()
embedding_counter: Any = NoOpMetric()
embedding_latency: Any = NoOpMetric()
cache_hits: Any = NoOpMetric()
cache_misses: Any = NoOpMetric()
index_size: Any = NoOpMetric()
index_chunks: Any = NoOpMetric()
error_counter: Any = NoOpMetric()
circuit_breaker_state: Any = NoOpMetric()


def _initialize_metrics() -> bool:
    """Initialize Prometheus metrics. Returns True if successful."""
    global _metrics_initialized
    global app_info, query_counter, query_latency, retrieval_counter
    global retrieval_latency, chunks_retrieved, llm_request_counter
    global llm_latency, llm_tokens_used, embedding_counter, embedding_latency
    global cache_hits, cache_misses, index_size, index_chunks
    global error_counter, circuit_breaker_state
    
    if _metrics_initialized:
        return True
    
    if not METRICS_ENABLED:
        _metrics_initialized = True
        return True
    
    try:
        from prometheus_client import Counter, Histogram, Gauge, Info
        
        # Application info
        app_info = Info("rag_assistant", "RAG Assistant application information")
        app_info.info({
            "version": "1.0.0",
            "framework": "langchain",
            "vector_store": "faiss",
        })
        
        # Request metrics
        query_counter = Counter(
            "rag_queries_total",
            "Total number of queries processed",
            ["status"],
        )
        
        query_latency = Histogram(
            "rag_query_latency_seconds",
            "Query processing latency in seconds",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
        )
        
        # Retrieval metrics
        retrieval_counter = Counter(
            "rag_retrievals_total",
            "Total number of document retrievals",
            ["status"],
        )
        
        retrieval_latency = Histogram(
            "rag_retrieval_latency_seconds",
            "Document retrieval latency in seconds",
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )
        
        chunks_retrieved = Histogram(
            "rag_chunks_retrieved",
            "Number of chunks retrieved per query",
            buckets=[1, 2, 3, 5, 10, 20, 50],
        )
        
        # LLM metrics
        llm_request_counter = Counter(
            "rag_llm_requests_total",
            "Total number of LLM requests",
            ["model", "status"],
        )
        
        llm_latency = Histogram(
            "rag_llm_latency_seconds",
            "LLM request latency in seconds",
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        )
        
        llm_tokens_used = Counter(
            "rag_llm_tokens_total",
            "Total tokens used",
            ["type"],
        )
        
        # Embedding metrics
        embedding_counter = Counter(
            "rag_embeddings_total",
            "Total number of embedding operations",
            ["status"],
        )
        
        embedding_latency = Histogram(
            "rag_embedding_latency_seconds",
            "Embedding generation latency in seconds",
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )
        
        # Cache metrics
        cache_hits = Counter(
            "rag_cache_hits_total",
            "Total cache hits",
            ["cache_type"],
        )
        
        cache_misses = Counter(
            "rag_cache_misses_total",
            "Total cache misses",
            ["cache_type"],
        )
        
        # Index metrics
        index_size = Gauge(
            "rag_index_size_documents",
            "Number of documents in the index",
        )
        
        index_chunks = Gauge(
            "rag_index_chunks_total",
            "Total number of chunks in the index",
        )
        
        # Error metrics
        error_counter = Counter(
            "rag_errors_total",
            "Total number of errors",
            ["error_code", "operation"],
        )
        
        # Circuit breaker metrics
        circuit_breaker_state = Gauge(
            "rag_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=half-open, 2=open)",
            ["name"],
        )
        
        _metrics_initialized = True
        return True
        
    except (ValueError, ImportError) as e:
        # Metrics already registered or not available - use no-ops
        _metrics_initialized = True
        return False


def get_metrics() -> bytes:
    """Get all metrics in Prometheus format."""
    if not METRICS_ENABLED:
        return b""
    
    try:
        from prometheus_client import REGISTRY
        from prometheus_client.exposition import generate_latest
        return generate_latest(REGISTRY)
    except Exception:
        return b""


class MetricsCollector:
    """Context manager for collecting operation metrics."""
    
    def __init__(
        self,
        operation: str,
        counter: Any,
        histogram: Any,
        labels: dict | None = None,
    ):
        self.operation = operation
        self.counter = counter
        self.histogram = histogram
        self.labels = labels or {}
        self._start_time: float = 0
    
    def __enter__(self) -> "MetricsCollector":
        import time
        self._start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        import time
        duration = time.perf_counter() - self._start_time
        
        status = "error" if exc_type else "success"
        labels = {**self.labels, "status": status}
        
        try:
            self.counter.labels(**labels).inc()
            self.histogram.observe(duration)
            
            if exc_type:
                error_counter.labels(
                    error_code=str(getattr(exc_val, "code", "unknown")),
                    operation=self.operation,
                ).inc()
        except Exception:
            pass  # Ignore metric errors
        
        return False


# Try to initialize metrics on import (will use no-ops if it fails)
_initialize_metrics()
