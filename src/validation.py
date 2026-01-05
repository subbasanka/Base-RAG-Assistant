"""Input validation and sanitization for RAG Assistant.

Provides security measures for user input and configuration.
"""

import re
from typing import Any

import bleach

from src.exceptions import ValidationError, ErrorCode


# Configuration limits
MAX_QUERY_LENGTH = 10000  # Maximum characters for a query
MAX_CHUNK_SIZE = 50000    # Maximum chunk size
MIN_CHUNK_SIZE = 100      # Minimum chunk size
MAX_TOP_K = 100           # Maximum chunks to retrieve
MIN_TOP_K = 1             # Minimum chunks to retrieve
MAX_TEMPERATURE = 2.0     # Maximum LLM temperature
MIN_TEMPERATURE = 0.0     # Minimum LLM temperature


def sanitize_text(text: str) -> str:
    """Sanitize text input by removing potentially harmful content.
    
    Args:
        text: Raw text input.
        
    Returns:
        Sanitized text.
    """
    # Remove HTML tags
    text = bleach.clean(text, tags=[], strip=True)
    
    # Remove null bytes
    text = text.replace("\x00", "")
    
    # Normalize unicode
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    
    return text.strip()


def validate_query(query: str) -> str:
    """Validate and sanitize a user query.
    
    Args:
        query: User's question.
        
    Returns:
        Validated and sanitized query.
        
    Raises:
        ValidationError: If query is invalid.
    """
    if not query:
        raise ValidationError(
            "Query cannot be empty",
            code=ErrorCode.INVALID_INPUT,
        )
    
    # Sanitize
    query = sanitize_text(query)
    
    if not query:
        raise ValidationError(
            "Query is empty after sanitization",
            code=ErrorCode.INVALID_INPUT,
        )
    
    if len(query) > MAX_QUERY_LENGTH:
        raise ValidationError(
            f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters",
            code=ErrorCode.INPUT_TOO_LONG,
            details={"length": len(query), "max_length": MAX_QUERY_LENGTH},
        )
    
    return query


def validate_chunk_params(
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[int, int]:
    """Validate chunking parameters.
    
    Args:
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.
        
    Returns:
        Validated (chunk_size, chunk_overlap) tuple.
        
    Raises:
        ValidationError: If parameters are invalid.
    """
    if not isinstance(chunk_size, int) or chunk_size < MIN_CHUNK_SIZE:
        raise ValidationError(
            f"chunk_size must be at least {MIN_CHUNK_SIZE}",
            code=ErrorCode.VALIDATION_ERROR,
            details={"chunk_size": chunk_size, "min": MIN_CHUNK_SIZE},
        )
    
    if chunk_size > MAX_CHUNK_SIZE:
        raise ValidationError(
            f"chunk_size cannot exceed {MAX_CHUNK_SIZE}",
            code=ErrorCode.VALIDATION_ERROR,
            details={"chunk_size": chunk_size, "max": MAX_CHUNK_SIZE},
        )
    
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValidationError(
            "chunk_overlap must be a non-negative integer",
            code=ErrorCode.VALIDATION_ERROR,
        )
    
    if chunk_overlap >= chunk_size:
        raise ValidationError(
            "chunk_overlap must be less than chunk_size",
            code=ErrorCode.VALIDATION_ERROR,
            details={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        )
    
    return chunk_size, chunk_overlap


def validate_top_k(top_k: int) -> int:
    """Validate top_k parameter.
    
    Args:
        top_k: Number of chunks to retrieve.
        
    Returns:
        Validated top_k value.
        
    Raises:
        ValidationError: If top_k is invalid.
    """
    if not isinstance(top_k, int):
        raise ValidationError(
            "top_k must be an integer",
            code=ErrorCode.VALIDATION_ERROR,
        )
    
    if top_k < MIN_TOP_K or top_k > MAX_TOP_K:
        raise ValidationError(
            f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}",
            code=ErrorCode.VALIDATION_ERROR,
            details={"top_k": top_k, "min": MIN_TOP_K, "max": MAX_TOP_K},
        )
    
    return top_k


def validate_temperature(temperature: float) -> float:
    """Validate LLM temperature parameter.
    
    Args:
        temperature: LLM temperature setting.
        
    Returns:
        Validated temperature value.
        
    Raises:
        ValidationError: If temperature is invalid.
    """
    if not isinstance(temperature, (int, float)):
        raise ValidationError(
            "temperature must be a number",
            code=ErrorCode.VALIDATION_ERROR,
        )
    
    if temperature < MIN_TEMPERATURE or temperature > MAX_TEMPERATURE:
        raise ValidationError(
            f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}",
            code=ErrorCode.VALIDATION_ERROR,
            details={
                "temperature": temperature,
                "min": MIN_TEMPERATURE,
                "max": MAX_TEMPERATURE,
            },
        )
    
    return float(temperature)


def validate_file_path(path: str, allowed_extensions: list[str]) -> bool:
    """Validate a file path for security.
    
    Args:
        path: File path to validate.
        allowed_extensions: List of allowed file extensions.
        
    Returns:
        True if valid.
        
    Raises:
        ValidationError: If path is invalid or unsafe.
    """
    from pathlib import Path
    
    # Check for path traversal attempts
    if ".." in path or path.startswith("/") or path.startswith("\\"):
        raise ValidationError(
            "Invalid file path: path traversal detected",
            code=ErrorCode.VALIDATION_ERROR,
        )
    
    # Check extension
    p = Path(path)
    if p.suffix.lower() not in allowed_extensions:
        raise ValidationError(
            f"Unsupported file extension: {p.suffix}",
            code=ErrorCode.VALIDATION_ERROR,
            details={"extension": p.suffix, "allowed": allowed_extensions},
        )
    
    return True


def validate_api_key(key: str, key_name: str) -> str:
    """Validate an API key format.
    
    Args:
        key: API key to validate.
        key_name: Name of the key for error messages.
        
    Returns:
        Validated API key.
        
    Raises:
        ValidationError: If key is invalid.
    """
    if not key:
        raise ValidationError(
            f"{key_name} is required",
            code=ErrorCode.API_KEY_MISSING,
        )
    
    # Basic format validation (not too short)
    if len(key) < 10:
        raise ValidationError(
            f"{key_name} appears to be invalid (too short)",
            code=ErrorCode.VALIDATION_ERROR,
        )
    
    # Check for placeholder values
    placeholder_patterns = [
        r"^your_.*_here$",
        r"^sk-xxx+$",
        r"^test$",
        r"^placeholder$",
    ]
    
    for pattern in placeholder_patterns:
        if re.match(pattern, key, re.IGNORECASE):
            raise ValidationError(
                f"{key_name} appears to be a placeholder value",
                code=ErrorCode.VALIDATION_ERROR,
            )
    
    return key

