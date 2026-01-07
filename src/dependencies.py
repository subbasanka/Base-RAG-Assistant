"""FastAPI dependencies for RAG Assistant.

Provides dependency injection for configuration, assistant, and other services.
"""

from functools import lru_cache
from typing import Generator

from src.config import Config, get_config, get_project_root
from src.rag_chat import RAGAssistant
from src.logging_config import get_logger

logger = get_logger(__name__)


# Cached config dependency
@lru_cache
def get_cached_config() -> Config:
    """Get cached configuration instance."""
    return get_config()


# Global assistant instance
_assistant: RAGAssistant | None = None


def get_assistant() -> RAGAssistant | None:
    """Get the RAG Assistant instance.
    
    Returns:
        RAGAssistant instance or None if not initialized.
    """
    return _assistant


def set_assistant(assistant: RAGAssistant | None) -> None:
    """Set the global RAG Assistant instance."""
    global _assistant
    _assistant = assistant


def init_assistant() -> RAGAssistant | None:
    """Initialize the RAG Assistant.
    
    Returns:
        RAGAssistant instance or None if initialization fails.
    """
    global _assistant
    try:
        _assistant = RAGAssistant()
        logger.info("RAG Assistant initialized successfully")
        return _assistant
    except Exception as e:
        error_msg = str(e)
        if "index" in error_msg.lower() or "E303" in error_msg:
            logger.info("No index found - server ready for document uploads")
        else:
            logger.error(f"Failed to initialize assistant: {e}")
        logger.info("Server starting without index (upload documents to create one)")
        return None


def require_assistant():
    """Dependency that requires an initialized assistant.
    
    Raises:
        HTTPException: If assistant is not initialized.
    """
    from fastapi import HTTPException
    
    assistant = get_assistant()
    if assistant is None:
        raise HTTPException(status_code=503, detail="Service not ready - no index loaded")
    return assistant
