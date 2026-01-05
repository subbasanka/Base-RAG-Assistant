"""RAG Assistant - A production-ready Retrieval-Augmented Generation system.

This package provides:
- Document ingestion from PDF, TXT, and Markdown files
- Vector indexing with FAISS
- Retrieval-augmented generation using OpenRouter LLMs
- CLI, REST API, and Streamlit interfaces
"""

__version__ = "1.0.0"
__author__ = "RAG Assistant Team"

from src.config import Config, get_config, load_config
from src.exceptions import (
    RAGException,
    ConfigurationError,
    DocumentError,
    IndexError,
    EmbeddingError,
    LLMError,
    RetrievalError,
    ValidationError,
)
from src.rag_chat import RAGAssistant, RAGResponse, RetrievalResult

__all__ = [
    # Version
    "__version__",
    # Config
    "Config",
    "get_config",
    "load_config",
    # Main classes
    "RAGAssistant",
    "RAGResponse",
    "RetrievalResult",
    # Exceptions
    "RAGException",
    "ConfigurationError",
    "DocumentError",
    "IndexError",
    "EmbeddingError",
    "LLMError",
    "RetrievalError",
    "ValidationError",
]
