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

### 4. Build the Index

```bash
python -m src.build_index
```

### 5. Start Using

**CLI Mode:**
```bash
python -m src.rag_chat
```

**REST API Server:**
```bash
python -m src.server --port 8000
```

**Streamlit UI:**
```bash
streamlit run src/ui_streamlit.py
```

## Commands Reference

| Command | Description |
|---------|-------------|
| `pip install -r requirements.txt` | Install all dependencies |
| `python -m src.build_index` | Ingest documents and build FAISS index |
| `python -m src.rag_chat` | Start interactive CLI chat |
| `python -m src.server` | Start REST API server |
| `python -m src.eval_run` | Run evaluation with sample questions |
| `streamlit run src/ui_streamlit.py` | Start the web UI |
| `pytest` | Run test suite |
| `pytest --cov=src` | Run tests with coverage |

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
# LLM Settings
llm:
  model_name: "openai/gpt-4o-mini"
  temperature: 0.1
  max_tokens: 1024
  timeout: 60.0

# Retrieval Settings
retrieval:
  top_k: 5
  score_threshold: 0.0

# Chunking Settings
chunking:
  chunk_size: 1000
  chunk_overlap: 200

# Embeddings Settings
embeddings:
  provider: "sentence-transformers"  # or "openai"
  model: "all-MiniLM-L6-v2"
  batch_size: 32

# Logging Settings
logging:
  level: "INFO"
  json_format: false  # Set to true for production
```

### Switching LLM Models

Change the model in `config.yaml`:

```yaml
llm:
  model_name: "anthropic/claude-3-haiku"  # Claude
  # or: "openai/gpt-4o"
  # or: "meta-llama/llama-3-70b-instruct"
  # See: https://openrouter.ai/models
```

### Embeddings Providers

**Option 1: Sentence Transformers (Default, Free, Local)**

```yaml
embeddings:
  provider: "sentence-transformers"
  model: "all-MiniLM-L6-v2"
```

**Option 2: OpenAI Embeddings**

```yaml
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"
```

## REST API

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/config` | GET | Current configuration |
| `/query` | POST | Submit a query |
| `/index/rebuild` | POST | Rebuild the index |

### Query Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main topics in the documents?", "top_k": 5}'
```

Response:
```json
{
  "answer": "Based on the documents...",
  "query": "What are the main topics?",
  "sources": [
    {
      "content": "...",
      "source": "document.pdf",
      "page": 1,
      "chunk_index": 0,
      "score": 0.85,
      "citation": "[source:document.pdf#page1|chunk0]"
    }
  ],
  "latency_ms": 1234.56
}
```

## Production Deployment

### Docker

Build and run with Docker:

```bash
# Build image
docker build -t rag-assistant .

# Run container
docker run -p 8000:8000 \
  -e OPENROUTER_API_KEY=your_key \
  -v $(pwd)/documents:/app/documents:ro \
  -v $(pwd)/data:/app/data \
  rag-assistant
```

### Docker Compose

```bash
# Start services
docker-compose up -d

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# View logs
docker-compose logs -f rag-assistant
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key |
| `OPENAI_API_KEY` | No | OpenAI API key (for OpenAI embeddings) |
| `RAG_ENVIRONMENT` | No | Environment: development, production, testing |

### Production Configuration

For production, update `config.yaml`:

```yaml
logging:
  level: "INFO"
  json_format: true  # Structured logs for log aggregation
  max_bytes: 52428800  # 50MB
  backup_count: 10
```

### Monitoring

Prometheus metrics available at `/metrics`:

- `rag_queries_total` - Total queries processed
- `rag_query_latency_seconds` - Query latency histogram
- `rag_retrieval_latency_seconds` - Retrieval latency
- `rag_llm_latency_seconds` - LLM request latency
- `rag_cache_hits_total` - Cache hit count
- `rag_errors_total` - Error count by type

## Project Structure

```
.
├── config.yaml              # Configuration
├── requirements.txt         # Dependencies
├── Dockerfile              # Docker build
├── docker-compose.yml      # Docker Compose
├── prometheus.yml          # Prometheus config
├── documents/              # Source documents
├── data/
│   ├── faiss_index/       # Vector index
│   └── cache/             # Embedding cache
├── logs/                  # Log files
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration loader
│   ├── exceptions.py      # Custom exceptions
│   ├── validation.py      # Input validation
│   ├── logging_config.py  # Structured logging
│   ├── retry.py           # Retry & circuit breaker
│   ├── cache.py           # Caching layer
│   ├── metrics.py         # Prometheus metrics
│   ├── ingest.py          # Document loading
│   ├── build_index.py     # Index building
│   ├── rag_chat.py        # RAG chat (CLI)
│   ├── server.py          # REST API server
│   ├── ui_streamlit.py    # Streamlit UI
│   └── eval_run.py        # Evaluation
└── tests/
    ├── conftest.py        # Test fixtures
    ├── test_config.py
    ├── test_validation.py
    ├── test_ingest.py
    └── test_cache.py
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_validation.py

# Verbose output
pytest -v
```

## Error Handling

The system uses structured error codes:

| Code Range | Category |
|------------|----------|
| E1xx | Configuration errors |
| E2xx | Document errors |
| E3xx | Index errors |
| E4xx | Embedding errors |
| E5xx | LLM errors |
| E6xx | Retrieval errors |
| E7xx | Validation errors |
| E9xx | System errors |

## Common Issues

### 1. "OPENROUTER_API_KEY is not set"

Create a `.env` file with your API key:
```env
OPENROUTER_API_KEY=your_key_here
```

### 2. "FAISS index not found"

Build the index first:
```bash
python -m src.build_index
```

### 3. "No documents found"

Add documents to the `documents/` folder (`.pdf`, `.txt`, `.md`).

### 4. Slow First Run

The first run downloads the embedding model (~90MB). Subsequent runs use the cached model.

### 5. Out of Memory

- Increase `chunk_size` in config
- Use a smaller embedding model
- Process documents in batches

### 6. Rate Limiting

The system includes circuit breaker protection. If you hit rate limits:
- Check OpenRouter quota
- Reduce request frequency
- Circuit breaker will automatically retry after cooldown

### 7. Docker Build Fails

Ensure you have sufficient disk space and memory:
```bash
docker system prune -a  # Clean up
docker build --no-cache -t rag-assistant .
```

## Security Considerations

1. **API Keys**: Never commit `.env` files. Use environment variables in production.
2. **Input Validation**: All user inputs are sanitized and validated.
3. **Docker**: Runs as non-root user with resource limits.
4. **Logging**: Sensitive data is not logged.

## Performance Tuning

### Caching

Embeddings are cached to avoid recomputation:
- Memory cache: LRU with configurable size
- Disk cache: Optional persistence

### Query Caching

Query results are cached with TTL:
- Identical queries return cached results
- Cache invalidated on index rebuild

### Circuit Breaker

Protects against cascading failures:
- Opens after 5 consecutive failures
- Half-open test after 60 seconds
- Automatic recovery on success

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest`
4. Submit a pull request
