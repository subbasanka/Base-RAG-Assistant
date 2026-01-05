"""Tests for validation module."""

import pytest

from src.validation import (
    sanitize_text,
    validate_query,
    validate_chunk_params,
    validate_top_k,
    validate_temperature,
    validate_api_key,
    MAX_QUERY_LENGTH,
)
from src.exceptions import ValidationError


class TestSanitizeText:
    """Tests for text sanitization."""
    
    def test_removes_html_tags(self):
        """Test HTML tag removal."""
        result = sanitize_text("<script>alert('xss')</script>Hello")
        assert "<script>" not in result
        assert "Hello" in result
    
    def test_removes_null_bytes(self):
        """Test null byte removal."""
        result = sanitize_text("Hello\x00World")
        assert "\x00" not in result
        assert "HelloWorld" in result
    
    def test_strips_whitespace(self):
        """Test whitespace stripping."""
        result = sanitize_text("  Hello World  ")
        assert result == "Hello World"
    
    def test_normalizes_unicode(self):
        """Test unicode normalization."""
        # Full-width characters to ASCII
        result = sanitize_text("Ｈｅｌｌｏ")
        assert result == "Hello"


class TestValidateQuery:
    """Tests for query validation."""
    
    def test_valid_query(self):
        """Test valid query passes."""
        query = "What is machine learning?"
        result = validate_query(query)
        assert result == query
    
    def test_empty_query_raises(self):
        """Test empty query raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_query("")
        assert "empty" in str(exc_info.value).lower()
    
    def test_too_long_query_raises(self):
        """Test query exceeding max length raises error."""
        long_query = "a" * (MAX_QUERY_LENGTH + 1)
        
        with pytest.raises(ValidationError) as exc_info:
            validate_query(long_query)
        assert "maximum length" in str(exc_info.value).lower()
    
    def test_query_sanitization(self):
        """Test query is sanitized."""
        query = "  <b>What</b> is AI?  "
        result = validate_query(query)
        assert "<b>" not in result
        assert "What" in result


class TestValidateChunkParams:
    """Tests for chunk parameter validation."""
    
    def test_valid_params(self):
        """Test valid chunk parameters pass."""
        size, overlap = validate_chunk_params(1000, 200)
        assert size == 1000
        assert overlap == 200
    
    def test_chunk_size_too_small(self):
        """Test chunk size below minimum raises error."""
        with pytest.raises(ValidationError):
            validate_chunk_params(50, 10)
    
    def test_chunk_size_too_large(self):
        """Test chunk size above maximum raises error."""
        with pytest.raises(ValidationError):
            validate_chunk_params(100000, 1000)
    
    def test_overlap_greater_than_size(self):
        """Test overlap >= chunk_size raises error."""
        with pytest.raises(ValidationError):
            validate_chunk_params(1000, 1000)
    
    def test_negative_overlap(self):
        """Test negative overlap raises error."""
        with pytest.raises(ValidationError):
            validate_chunk_params(1000, -1)


class TestValidateTopK:
    """Tests for top_k validation."""
    
    def test_valid_top_k(self):
        """Test valid top_k passes."""
        assert validate_top_k(5) == 5
        assert validate_top_k(1) == 1
        assert validate_top_k(100) == 100
    
    def test_top_k_too_small(self):
        """Test top_k below minimum raises error."""
        with pytest.raises(ValidationError):
            validate_top_k(0)
    
    def test_top_k_too_large(self):
        """Test top_k above maximum raises error."""
        with pytest.raises(ValidationError):
            validate_top_k(101)


class TestValidateTemperature:
    """Tests for temperature validation."""
    
    def test_valid_temperature(self):
        """Test valid temperature passes."""
        assert validate_temperature(0.0) == 0.0
        assert validate_temperature(1.0) == 1.0
        assert validate_temperature(2.0) == 2.0
    
    def test_temperature_too_high(self):
        """Test temperature above maximum raises error."""
        with pytest.raises(ValidationError):
            validate_temperature(2.1)
    
    def test_temperature_negative(self):
        """Test negative temperature raises error."""
        with pytest.raises(ValidationError):
            validate_temperature(-0.1)


class TestValidateApiKey:
    """Tests for API key validation."""
    
    def test_valid_api_key(self):
        """Test valid API key passes."""
        key = "sk-abcdefghijklmnop"
        assert validate_api_key(key, "TEST_KEY") == key
    
    def test_empty_api_key(self):
        """Test empty API key raises error."""
        with pytest.raises(ValidationError):
            validate_api_key("", "TEST_KEY")
    
    def test_placeholder_api_key(self):
        """Test placeholder API key raises error."""
        with pytest.raises(ValidationError):
            validate_api_key("your_api_key_here", "TEST_KEY")
    
    def test_too_short_api_key(self):
        """Test too short API key raises error."""
        with pytest.raises(ValidationError):
            validate_api_key("short", "TEST_KEY")

