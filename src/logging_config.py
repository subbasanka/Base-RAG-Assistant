"""Structured logging configuration for RAG Assistant.

Provides consistent, structured logging with proper formatting,
log levels, and file rotation for production use.
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import structlog


def configure_logging(
    log_dir: Path | str = "logs",
    log_level: str = "INFO",
    json_format: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> structlog.BoundLogger:
    """Configure structured logging for the application.
    
    Args:
        log_dir: Directory for log files.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, output logs as JSON (for production).
        max_bytes: Maximum size per log file before rotation.
        backup_count: Number of backup files to keep.
        
    Returns:
        Configured structlog logger.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up standard logging
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    if json_format:
        # JSON format for production (easier to parse)
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
        console_formatter = logging.Formatter("%(message)s")
    else:
        # Human-readable format for development
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # File handler with rotation
    log_file = log_dir / f"rag_assistant_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    ))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()


def get_logger(name: str = "rag_assistant") -> structlog.BoundLogger:
    """Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually module name).
        
    Returns:
        Configured structlog logger.
    """
    return structlog.get_logger(name)


class RequestLogger:
    """Context manager for logging request lifecycle."""
    
    def __init__(
        self,
        logger: structlog.BoundLogger,
        operation: str,
        **context: Any,
    ):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time: datetime | None = None
    
    def __enter__(self) -> "RequestLogger":
        self.start_time = datetime.now()
        self.logger.info(
            f"Starting {self.operation}",
            operation=self.operation,
            **self.context,
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000
        
        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation}",
                operation=self.operation,
                duration_ms=round(duration_ms, 2),
                status="success",
                **self.context,
            )
        else:
            self.logger.error(
                f"Failed {self.operation}",
                operation=self.operation,
                duration_ms=round(duration_ms, 2),
                status="error",
                error=str(exc_val),
                error_type=exc_type.__name__,
                **self.context,
            )
        
        return False  # Don't suppress exceptions

