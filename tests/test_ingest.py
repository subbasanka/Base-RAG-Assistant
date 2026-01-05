"""Tests for document ingestion module."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.ingest import (
    clean_text,
    load_documents,
    chunk_documents,
    ingest_documents,
)
from src.exceptions import DocumentError


class TestCleanText:
    """Tests for text cleaning."""
    
    def test_removes_extra_whitespace(self):
        """Test multiple spaces are collapsed."""
        result = clean_text("Hello    World")
        assert result == "Hello World"
    
    def test_removes_newlines(self):
        """Test newlines are converted to spaces."""
        result = clean_text("Hello\n\n\nWorld")
        assert result == "Hello World"
    
    def test_strips_edges(self):
        """Test leading/trailing whitespace is removed."""
        result = clean_text("   Hello World   ")
        assert result == "Hello World"
    
    def test_removes_null_chars(self):
        """Test null characters are removed."""
        result = clean_text("Hello\x00World")
        assert result == "HelloWorld"


class TestLoadDocuments:
    """Tests for document loading."""
    
    def test_load_text_files(self, sample_documents: Path):
        """Test loading text files."""
        docs = load_documents(sample_documents)
        
        assert len(docs) >= 1
        assert any("sample" in d.metadata.get("source", "") for d in docs)
    
    def test_documents_have_metadata(self, sample_documents: Path):
        """Test loaded documents have required metadata."""
        docs = load_documents(sample_documents)
        
        for doc in docs:
            assert "source" in doc.metadata
            assert "page" in doc.metadata
    
    def test_missing_directory_raises(self, temp_dir: Path):
        """Test missing directory raises DocumentError."""
        with pytest.raises(DocumentError) as exc_info:
            load_documents(temp_dir / "nonexistent")
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_empty_directory_raises(self, temp_dir: Path):
        """Test empty directory raises DocumentError."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(DocumentError) as exc_info:
            load_documents(empty_dir)
        
        assert "no documents" in str(exc_info.value).lower()


class TestChunkDocuments:
    """Tests for document chunking."""
    
    def test_chunks_have_metadata(self, sample_documents: Path):
        """Test chunks have chunk_index metadata."""
        docs = load_documents(sample_documents)
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
        
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "source" in chunk.metadata
            assert "page" in chunk.metadata
    
    def test_chunk_size_respected(self, sample_documents: Path):
        """Test chunks don't exceed size limit (with some tolerance)."""
        docs = load_documents(sample_documents)
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
        
        # Most chunks should be around chunk_size
        for chunk in chunks:
            # Allow some tolerance for word boundaries
            assert len(chunk.page_content) <= 250
    
    def test_chunk_indices_increment(self, sample_documents: Path):
        """Test chunk indices increment correctly per source."""
        docs = load_documents(sample_documents)
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
        
        # Group chunks by source
        by_source = {}
        for chunk in chunks:
            source = chunk.metadata["source"]
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(chunk.metadata["chunk_index"])
        
        # Check indices are sequential for each source
        for source, indices in by_source.items():
            sorted_indices = sorted(indices)
            for i, idx in enumerate(sorted_indices):
                assert idx == i, f"Non-sequential index for {source}"


class TestIngestDocuments:
    """Tests for full ingestion pipeline."""
    
    def test_ingest_returns_chunks(self, sample_documents: Path):
        """Test ingestion returns chunked documents."""
        chunks = ingest_documents(sample_documents)
        
        assert len(chunks) > 0
        assert all(hasattr(c, "page_content") for c in chunks)
        assert all(hasattr(c, "metadata") for c in chunks)
    
    def test_ingest_with_custom_params(self, sample_documents: Path):
        """Test ingestion with custom chunk parameters."""
        chunks = ingest_documents(
            sample_documents,
            chunk_size=200,
            chunk_overlap=50,
        )
        
        assert len(chunks) > 0

