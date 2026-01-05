"""Tests for configuration module."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import (
    Config,
    LLMConfig,
    ChunkingConfig,
    load_config,
    get_config,
    clear_config_cache,
)
from src.exceptions import ConfigurationError


class TestLLMConfig:
    """Tests for LLM configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.model_name == "openai/gpt-4o-mini"
        assert config.temperature == 0.1
        assert config.max_tokens == 1024
    
    def test_temperature_validation(self):
        """Test temperature bounds validation."""
        # Valid temperature
        config = LLMConfig(temperature=1.0)
        assert config.temperature == 1.0
        
        # Invalid temperature (too high)
        with pytest.raises(ValueError):
            LLMConfig(temperature=3.0)
        
        # Invalid temperature (negative)
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)


class TestChunkingConfig:
    """Tests for chunking configuration."""
    
    def test_default_values(self):
        """Test default chunking values."""
        config = ChunkingConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
    
    def test_overlap_validation(self):
        """Test chunk overlap validation."""
        # Valid overlap
        config = ChunkingConfig(chunk_size=1000, chunk_overlap=200)
        assert config.chunk_overlap < config.chunk_size
        
        # Invalid: overlap >= chunk_size
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=1000, chunk_overlap=1000)
    
    def test_minimum_chunk_size(self):
        """Test minimum chunk size validation."""
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=50)  # Below minimum of 100


class TestConfig:
    """Tests for main configuration."""
    
    def test_validate_for_chat_missing_key(self):
        """Test validation fails without API key."""
        config = Config(openrouter_api_key="")
        
        with pytest.raises(ConfigurationError):
            config.validate_for_chat()
    
    def test_validate_for_chat_with_key(self):
        """Test validation passes with API key."""
        config = Config(openrouter_api_key="test_key_12345")
        config.validate_for_chat()  # Should not raise
    
    def test_is_production(self):
        """Test environment detection."""
        dev_config = Config(environment="development")
        assert not dev_config.is_production
        
        prod_config = Config(environment="production")
        assert prod_config.is_production


class TestLoadConfig:
    """Tests for configuration loading."""
    
    def test_load_from_yaml(self, temp_dir: Path, env_vars):
        """Test loading configuration from YAML file."""
        # Create config file
        config_path = temp_dir / "config.yaml"
        config_data = {
            "llm": {
                "model_name": "test/model",
                "temperature": 0.5,
            },
            "retrieval": {
                "top_k": 10,
            },
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        # Temporarily change working directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            config = load_config("config.yaml")
            assert config.llm.model_name == "test/model"
            assert config.llm.temperature == 0.5
            assert config.retrieval.top_k == 10
        finally:
            os.chdir(original_cwd)
    
    def test_load_missing_file_uses_defaults(self, temp_dir: Path, env_vars):
        """Test loading with missing config file uses defaults."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            config = load_config("nonexistent.yaml")
            assert config.llm.model_name == "openai/gpt-4o-mini"
        finally:
            os.chdir(original_cwd)
    
    def test_config_caching(self, env_vars):
        """Test configuration caching."""
        clear_config_cache()
        
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2  # Same object (cached)

