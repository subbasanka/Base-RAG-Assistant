"""FAISS index building and management for RAG Assistant.

Handles embedding creation and vector store operations with
production-ready error handling, caching, and metrics.
"""

import time
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.cache import get_embedding_cache
from src.config import Config, get_config, get_project_root
from src.exceptions import EmbeddingError, IndexError, ErrorCode
from src.ingest import ingest_documents
from src.logging_config import get_logger, RequestLogger
from src.metrics import (
    embedding_counter,
    embedding_latency,
    index_chunks,
    index_size,
)
from src.retry import retry_with_backoff

logger = get_logger(__name__)


class CachedEmbeddings(Embeddings):
    """Embeddings wrapper with caching support."""
    
    def __init__(
        self,
        base_embeddings: Embeddings,
        cache_enabled: bool = True,
        cache_dir: Path | None = None,
    ):
        self._base = base_embeddings
        self._cache_enabled = cache_enabled
        self._cache = get_embedding_cache(
            cache_dir=cache_dir,
            persist=cache_dir is not None,
        ) if cache_enabled else None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with caching."""
        if not self._cache_enabled or self._cache is None:
            return self._base.embed_documents(texts)
        
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results.append((i, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Embed uncached texts
        if uncached_texts:
            new_embeddings = self._base.embed_documents(uncached_texts)
            for idx, text, embedding in zip(uncached_indices, uncached_texts, new_embeddings):
                self._cache.set(text, embedding)
                results.append((idx, embedding))
        
        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query with caching."""
        if not self._cache_enabled or self._cache is None:
            return self._base.embed_query(text)
        
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        
        embedding = self._base.embed_query(text)
        self._cache.set(text, embedding)
        return embedding


def get_embeddings(config: Config | None = None) -> Embeddings:
    """Get the configured embeddings model.
    
    Supports two providers:
    - sentence-transformers: Local embeddings using HuggingFace models (free)
    - openai: OpenAI embeddings API (requires OPENAI_API_KEY)
    
    Args:
        config: Configuration object. Uses default if None.
        
    Returns:
        Embeddings instance for the configured provider.
        
    Raises:
        EmbeddingError: If provider is unsupported or configuration is invalid.
    """
    if config is None:
        config = get_config()
    
    provider = config.embeddings.provider
    model = config.embeddings.model
    
    try:
        if provider == "sentence-transformers":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            logger.info(
                "Initializing sentence-transformers embeddings",
                model=model,
            )
            
            base_embeddings = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        
        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            
            config.validate_for_embeddings()
            
            logger.info(
                "Initializing OpenAI embeddings",
                model=model,
            )
            
            base_embeddings = OpenAIEmbeddings(
                model=model,
                openai_api_key=config.openai_api_key,
            )
        
        else:
            raise EmbeddingError(
                f"Unsupported embeddings provider: {provider}",
                code=ErrorCode.EMBEDDING_PROVIDER_ERROR,
                details={"provider": provider, "supported": ["sentence-transformers", "openai"]},
            )
        
        # Wrap with caching
        project_root = get_project_root()
        cache_dir = project_root / config.paths.cache_dir
        
        return CachedEmbeddings(
            base_embeddings=base_embeddings,
            cache_enabled=True,
            cache_dir=cache_dir,
        )
    
    except EmbeddingError:
        raise
    except Exception as e:
        raise EmbeddingError(
            f"Failed to initialize embeddings: {e}",
            code=ErrorCode.EMBEDDING_FAILED,
            cause=e,
        )


@retry_with_backoff(max_attempts=3, min_wait=1.0, max_wait=30.0)
def build_index(
    chunks: List[Document],
    config: Config | None = None,
) -> FAISS:
    """Build a FAISS index from document chunks.
    
    Args:
        chunks: List of Document objects to index.
        config: Configuration object.
        
    Returns:
        FAISS vector store instance.
        
    Raises:
        IndexError: If index building fails.
    """
    if config is None:
        config = get_config()
    
    start_time = time.perf_counter()
    
    try:
        with RequestLogger(logger, "index_building", chunks=len(chunks)):
            embeddings = get_embeddings(config)
            
            # Create FAISS index from documents
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings,
            )
            
            # Update metrics
            duration = time.perf_counter() - start_time
            embedding_latency.observe(duration)
            embedding_counter.labels(status="success").inc()
            index_chunks.set(len(chunks))
            
            logger.info(
                "FAISS index built successfully",
                chunks=len(chunks),
                duration_seconds=round(duration, 2),
            )
            
            return vector_store
    
    except Exception as e:
        embedding_counter.labels(status="error").inc()
        raise IndexError(
            f"Failed to build index: {e}",
            code=ErrorCode.INDEX_BUILD_FAILED,
            cause=e,
        )


def save_index(
    vector_store: FAISS,
    index_path: str | Path | None = None,
    config: Config | None = None,
) -> Path:
    """Save FAISS index to disk.
    
    Args:
        vector_store: FAISS vector store instance.
        index_path: Path to save the index. Uses config default if None.
        config: Configuration object.
        
    Returns:
        Path where the index was saved.
        
    Raises:
        IndexError: If saving fails.
    """
    if config is None:
        config = get_config()
    
    project_root = get_project_root()
    
    if index_path is None:
        index_path = project_root / config.paths.index_dir
    else:
        index_path = Path(index_path)
    
    try:
        # Ensure directory exists
        index_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Saving index", path=str(index_path))
        vector_store.save_local(str(index_path))
        
        logger.info("Index saved successfully", path=str(index_path))
        return index_path
    
    except Exception as e:
        raise IndexError(
            f"Failed to save index: {e}",
            code=ErrorCode.INDEX_SAVE_FAILED,
            details={"path": str(index_path)},
            cause=e,
        )


def load_index(
    index_path: str | Path | None = None,
    config: Config | None = None,
) -> FAISS:
    """Load FAISS index from disk.
    
    Args:
        index_path: Path to the saved index. Uses config default if None.
        config: Configuration object.
        
    Returns:
        FAISS vector store instance.
        
    Raises:
        IndexError: If index doesn't exist or loading fails.
    """
    if config is None:
        config = get_config()
    
    project_root = get_project_root()
    
    if index_path is None:
        index_path = project_root / config.paths.index_dir
    else:
        index_path = Path(index_path)
    
    if not index_path.exists():
        raise IndexError(
            f"FAISS index not found at: {index_path}",
            code=ErrorCode.INDEX_NOT_FOUND,
            details={"path": str(index_path)},
        )
    
    try:
        logger.info("Loading index", path=str(index_path))
        
        embeddings = get_embeddings(config)
        
        vector_store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        
        logger.info("Index loaded successfully", path=str(index_path))
        return vector_store
    
    except IndexError:
        raise
    except Exception as e:
        raise IndexError(
            f"Failed to load index: {e}",
            code=ErrorCode.INDEX_LOAD_FAILED,
            details={"path": str(index_path)},
            cause=e,
        )


def build_and_save_index(
    documents_dir: str | Path | None = None,
    index_path: str | Path | None = None,
    config: Config | None = None,
) -> FAISS:
    """Full pipeline: ingest documents, build index, and save to disk.
    
    Args:
        documents_dir: Path to documents directory.
        index_path: Path to save the index.
        config: Configuration object.
        
    Returns:
        FAISS vector store instance.
    """
    if config is None:
        config = get_config()
    
    logger.info(
        "Starting index build pipeline",
        embeddings_provider=config.embeddings.provider,
        embeddings_model=config.embeddings.model,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
    )
    
    # Ingest documents
    chunks = ingest_documents(documents_dir, config=config)
    
    # Build index
    vector_store = build_index(chunks, config)
    
    # Save index
    save_index(vector_store, index_path, config)
    
    # Update metrics
    index_size.set(len(set(c.metadata.get("source") for c in chunks)))
    
    logger.info("Index build pipeline complete")
    
    return vector_store


if __name__ == "__main__":
    import sys
    from src.logging_config import configure_logging
    
    configure_logging()
    
    try:
        build_and_save_index()
    except (IndexError, EmbeddingError) as e:
        logger.error(f"Index build failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
