"""Custom exceptions for RAG Assistant.

Provides structured error handling with error codes and context.
"""

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Error codes for categorizing exceptions."""
    
    # Configuration errors (1xx)
    CONFIG_MISSING = "E101"
    CONFIG_INVALID = "E102"
    API_KEY_MISSING = "E103"
    
    # Document errors (2xx)
    DOCUMENTS_NOT_FOUND = "E201"
    DOCUMENT_LOAD_FAILED = "E202"
    DOCUMENT_PARSE_FAILED = "E203"
    UNSUPPORTED_FORMAT = "E204"
    
    # Index errors (3xx)
    INDEX_NOT_FOUND = "E301"
    INDEX_BUILD_FAILED = "E302"
    INDEX_LOAD_FAILED = "E303"
    INDEX_SAVE_FAILED = "E304"
    
    # Embedding errors (4xx)
    EMBEDDING_FAILED = "E401"
    EMBEDDING_PROVIDER_ERROR = "E402"
    
    # LLM errors (5xx)
    LLM_REQUEST_FAILED = "E501"
    LLM_TIMEOUT = "E502"
    LLM_RATE_LIMITED = "E503"
    LLM_INVALID_RESPONSE = "E504"
    
    # Retrieval errors (6xx)
    RETRIEVAL_FAILED = "E601"
    NO_RELEVANT_DOCUMENTS = "E602"
    
    # Validation errors (7xx)
    VALIDATION_ERROR = "E701"
    INPUT_TOO_LONG = "E702"
    INVALID_INPUT = "E703"
    
    # System errors (9xx)
    INTERNAL_ERROR = "E901"
    RESOURCE_EXHAUSTED = "E902"


class RAGException(Exception):
    """Base exception for RAG Assistant."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.cause = cause
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error": self.message,
            "code": self.code.value,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }
    
    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"


class ConfigurationError(RAGException):
    """Configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.CONFIG_INVALID,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)


class DocumentError(RAGException):
    """Document loading/processing errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.DOCUMENT_LOAD_FAILED,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)


class IndexError(RAGException):
    """Index building/loading errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INDEX_BUILD_FAILED,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)


class EmbeddingError(RAGException):
    """Embedding generation errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.EMBEDDING_FAILED,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)


class LLMError(RAGException):
    """LLM request/response errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.LLM_REQUEST_FAILED,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)


class RetrievalError(RAGException):
    """Document retrieval errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.RETRIEVAL_FAILED,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)


class ValidationError(RAGException):
    """Input validation errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.VALIDATION_ERROR,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)

