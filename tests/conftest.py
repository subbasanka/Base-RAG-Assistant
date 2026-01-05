"""Pytest configuration and fixtures for RAG Assistant tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config, clear_config_cache


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config cache before each test."""
    clear_config_cache()
    yield
    clear_config_cache()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_documents(temp_dir: Path) -> Path:
    """Create sample documents for testing."""
    docs_dir = temp_dir / "documents"
    docs_dir.mkdir()
    
    # Create a sample text file
    (docs_dir / "sample.txt").write_text(
        "This is a sample document for testing. "
        "It contains information about AI and machine learning. "
        "RAG systems combine retrieval with generation."
    )
    
    # Create a sample markdown file
    (docs_dir / "readme.md").write_text(
        "# Test Document\n\n"
        "This is a test markdown document.\n\n"
        "## Section 1\n\n"
        "Some content about testing.\n\n"
        "## Section 2\n\n"
        "More content about RAG systems."
    )
    
    return docs_dir


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """Create a test configuration."""
    return Config(
        openrouter_api_key="test_api_key_12345",
        openai_api_key="",
        environment="testing",
    )


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing."""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1] * 384 for _ in range(10)]
    mock.embed_query.return_value = [0.1] * 384
    return mock


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content="This is a test answer based on the documents."
    )
    return mock


@pytest.fixture
def env_vars():
    """Set up environment variables for testing."""
    original = os.environ.copy()
    os.environ["OPENROUTER_API_KEY"] = "test_key_12345"
    os.environ["RAG_ENVIRONMENT"] = "testing"
    yield
    os.environ.clear()
    os.environ.update(original)

