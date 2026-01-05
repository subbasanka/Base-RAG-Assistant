"""Retry logic and circuit breaker patterns for RAG Assistant.

Provides resilient API calls with exponential backoff and circuit breaking.
"""

import functools
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError,
)

from src.exceptions import LLMError, EmbeddingError, ErrorCode
from src.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.
    
    Attributes:
        failure_threshold: Number of failures before opening circuit.
        recovery_timeout: Seconds to wait before trying again.
        half_open_max_calls: Max calls to allow in half-open state.
    """
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: datetime | None = field(default=None)
    half_open_calls: int = field(default=0)
    
    # Track recent failures for logging
    recent_failures: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def can_execute(self) -> bool:
        """Check if a request can be executed."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    return True
            return False
        
        # HALF_OPEN state
        return self.half_open_calls < self.half_open_max_calls
    
    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info("Circuit breaker CLOSED (recovered)")
        else:
            self.failure_count = 0
    
    def record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.recent_failures.append({
            "time": datetime.now().isoformat(),
            "error": str(error),
        })
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker OPEN (half-open test failed)")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPEN (failures: {self.failure_count})",
                failure_count=self.failure_count,
            )
    
    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "recent_failures": list(self.recent_failures),
        }


# Global circuit breakers for different services
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker()
    return _circuit_breakers[name]


def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
) -> Callable:
    """Decorator to apply circuit breaker pattern.
    
    Args:
        name: Name of the circuit breaker.
        failure_threshold: Number of failures before opening.
        recovery_timeout: Seconds before attempting recovery.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Initialize circuit breaker
        cb = get_circuit_breaker(name)
        cb.failure_threshold = failure_threshold
        cb.recovery_timeout = recovery_timeout
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not cb.can_execute():
                raise LLMError(
                    f"Circuit breaker '{name}' is OPEN. Service temporarily unavailable.",
                    code=ErrorCode.LLM_REQUEST_FAILED,
                    details=cb.get_status(),
                )
            
            try:
                if cb.state == CircuitState.HALF_OPEN:
                    cb.half_open_calls += 1
                
                result = func(*args, **kwargs)
                cb.record_success()
                return result
            except Exception as e:
                cb.record_failure(e)
                raise
        
        return wrapper
    return decorator


def retry_with_backoff(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    retry_exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator for retry with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts.
        min_wait: Minimum wait time between retries (seconds).
        max_wait: Maximum wait time between retries (seconds).
        retry_exceptions: Tuple of exception types to retry on.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=min_wait, max=max_wait),
            retry=retry_if_exception_type(retry_exceptions),
            before_sleep=before_sleep_log(logger, log_level=logging.WARNING),
            reraise=True,
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Import logging for tenacity


def create_llm_retry():
    """Create retry decorator for LLM calls."""
    return retry_with_backoff(
        max_attempts=3,
        min_wait=1.0,
        max_wait=30.0,
        retry_exceptions=(Exception,),  # Will be more specific in usage
    )


def create_embedding_retry():
    """Create retry decorator for embedding calls."""
    return retry_with_backoff(
        max_attempts=3,
        min_wait=0.5,
        max_wait=10.0,
        retry_exceptions=(Exception,),
    )

