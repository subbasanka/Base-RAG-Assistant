# RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) assistant that ingests documents, indexes them in FAISS, and answers user questions grounded in retrieved context using an LLM served via OpenRouter.

## Features

- **Document Ingestion**: Load PDF, TXT, and Markdown files with automatic chunking
- **Vector Indexing**: FAISS-based vector store with persistent storage and caching
- **Retrieval + Answering**: Semantic search with LLM-generated answers
- **Citation Support**: Answers include source citations in `[source:filename#page|chunk]` format
- **Multiple Interfaces**: CLI, REST API, and Streamlit web UI
- **Production Ready**: 
  - Structured logging with rotation
  - Retry logic with circuit breaker pattern
  - Input validation and sanitization
  - Prometheus metrics
  - Docker support
  - Comprehensive test suite

## Architecture

```
Documents → Loader → Chunker → Embeddings → FAISS Index
                                    ↓
User Query → Embeddings → Similarity Search → Context + LLM → Answer with Citations
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# Required: OpenRouter API Key
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: OpenAI API Key (only if using OpenAI embeddings)
# OPENAI_API_KEY=your_openai_api_key_here

# Optional: Environment (development, production, testing)
# RAG_ENVIRONMENT=development
```

### 3. Add Documents

Place your documents in the `documents/` folder:
- Supported formats: `.pdf`, `.txt`, `.md`


