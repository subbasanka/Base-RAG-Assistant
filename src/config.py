"""Configuration loader for RAG Assistant.

Loads settings from config.yaml and environment variables (.env file).
Uses Pydantic for validation and type safety.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

from src.exceptions import ConfigurationError, ErrorCode


class LLMConfig(BaseModel):
    """LLM configuration settings."""
    
    model_name: str = Field(
        default="openai/gpt-4o-mini",
        description="OpenRouter model identifier",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature (0-2)",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=128000,
        description="Maximum response tokens",
    )
    timeout: float = Field(
        default=60.0,
        ge=1.0,
        description="Request timeout in seconds",
    )


class RetrievalConfig(BaseModel):
    """Retrieval configuration settings."""
    
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of chunks to retrieve",
    )
    score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold",
    )


class ChunkingConfig(BaseModel):
    """Text chunking configuration settings."""
    
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=50000,
        description="Characters per chunk",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between chunks",
    )
    
    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        """Ensure overlap is less than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration settings."""
    
    provider: Literal["sentence-transformers", "openai"] = Field(
        default="sentence-transformers",
        description="Embeddings provider",
    )
    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model name",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=1000,
        description="Batch size for embedding generation",
    )


class PathsConfig(BaseModel):
    """Path configuration settings."""
    
    documents_dir: str = Field(
        default="documents",
        description="Source documents folder",
    )
    index_dir: str = Field(
        default="data/faiss_index",
        description="FAISS index storage",
    )
    logs_dir: str = Field(
        default="logs",
        description="Log files directory",
    )
    cache_dir: str = Field(
        default="data/cache",
        description="Cache directory",
    )


class LoggingConfig(BaseModel):
    """Logging configuration settings."""
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Log level",
    )
    json_format: bool = Field(
        default=False,
        description="Use JSON format for logs (production)",
    )
    max_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Max log file size before rotation",
    )
    backup_count: int = Field(
        default=5,
        description="Number of backup log files",
    )


class Config(BaseModel):
    """Main configuration container."""
    
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # API keys (loaded from environment)
    openrouter_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    
    # Environment
    environment: Literal["development", "production", "testing"] = Field(
        default="development",
        description="Runtime environment",
    )
    
    def validate_for_chat(self) -> None:
        """Validate configuration for chat operations."""
        if not self.openrouter_api_key:
            raise ConfigurationError(
                "OPENROUTER_API_KEY is required for chat operations. "
                "Set it in your .env file.",
                code=ErrorCode.API_KEY_MISSING,
            )
    
    def validate_for_embeddings(self) -> None:
        """Validate configuration for embedding operations."""
        if self.embeddings.provider == "openai" and not self.openai_api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is required when using OpenAI embeddings. "
                "Set it in your .env file or switch to 'sentence-transformers'.",
                code=ErrorCode.API_KEY_MISSING,
            )
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    model_config = {"extra": "ignore"}


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file and environment variables.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Config object with all settings loaded.
        
    Raises:
        ConfigurationError: If configuration is invalid.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Determine project root
    project_root = get_project_root()
    config_file = project_root / config_path
    
    # Load YAML configuration
    yaml_config: dict = {}
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in config file: {e}",
                code=ErrorCode.CONFIG_INVALID,
            )
    else:
        # Log warning but continue with defaults
        import warnings
        warnings.warn(f"Config file {config_file} not found. Using defaults.")
    
    # Get environment
    environment = os.getenv("RAG_ENVIRONMENT", "development")
    
    try:
        config = Config(
            llm=LLMConfig(**yaml_config.get("llm", {})),
            retrieval=RetrievalConfig(**yaml_config.get("retrieval", {})),
            chunking=ChunkingConfig(**yaml_config.get("chunking", {})),
            embeddings=EmbeddingsConfig(**yaml_config.get("embeddings", {})),
            paths=PathsConfig(**yaml_config.get("paths", {})),
            logging=LoggingConfig(**yaml_config.get("logging", {})),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            environment=environment,
        )
    except Exception as e:
        raise ConfigurationError(
            f"Configuration validation failed: {e}",
            code=ErrorCode.CONFIG_INVALID,
            cause=e,
        )
    
    return config


# Cached config getter
@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get cached configuration instance.
    
    Returns:
        Cached Config object.
    """
    return load_config()


def clear_config_cache() -> None:
    """Clear the configuration cache (useful for testing)."""
    get_config.cache_clear()


if __name__ == "__main__":
    # Test configuration loading
    try:
        cfg = load_config()
        print("Configuration loaded successfully!")
        print(f"  Environment: {cfg.environment}")
        print(f"  LLM Model: {cfg.llm.model_name}")
        print(f"  Embeddings: {cfg.embeddings.provider}/{cfg.embeddings.model}")
        print(f"  Chunk Size: {cfg.chunking.chunk_size}")
        print(f"  Top-K: {cfg.retrieval.top_k}")
        print(f"  Log Level: {cfg.logging.level}")
        print(f"  OpenRouter API Key: {'***' + cfg.openrouter_api_key[-4:] if cfg.openrouter_api_key else 'NOT SET'}")
    except ConfigurationError as e:
        print(f"Configuration Error: {e}")
