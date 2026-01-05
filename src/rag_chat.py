"""RAG Chat module for retrieval-augmented question answering.

Uses OpenRouter for LLM inference and FAISS for document retrieval
with production-ready error handling, retry logic, and metrics.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.build_index import load_index
from src.cache import get_query_cache
from src.config import Config, get_config, get_project_root
from src.exceptions import LLMError, RetrievalError, ErrorCode
from src.logging_config import get_logger, configure_logging, RequestLogger
from src.metrics import (
    query_counter,
    query_latency,
    retrieval_counter,
    retrieval_latency,
    chunks_retrieved,
    llm_request_counter,
    llm_latency,
    MetricsCollector,
)
from src.retry import with_circuit_breaker, retry_with_backoff
from src.validation import validate_query

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Container for a single retrieval result."""
    
    content: str
    source: str
    page: int
    chunk_index: int
    score: float
    
    @property
    def citation(self) -> str:
        """Generate citation string."""
        return f"[source:{self.source}#page{self.page}|chunk{self.chunk_index}]"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "score": self.score,
            "citation": self.citation,
        }
    
    def __str__(self) -> str:
        return f"{self.citation}: {self.content[:100]}..."


@dataclass
class RAGResponse:
    """Container for RAG response with answer and sources."""
    
    answer: str
    sources: List[RetrievalResult]
    query: str
    latency_ms: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "query": self.query,
            "sources": [s.to_dict() for s in self.sources],
            "latency_ms": self.latency_ms,
        }


def get_llm(config: Config | None = None) -> ChatOpenAI:
    """Initialize OpenRouter LLM client.
    
    Uses OpenAI-compatible API with OpenRouter endpoint.
    
    Args:
        config: Configuration object.
        
    Returns:
        ChatOpenAI instance configured for OpenRouter.
        
    Raises:
        LLMError: If API key is not configured.
    """
    if config is None:
        config = get_config()
    
    try:
        config.validate_for_chat()
    except Exception as e:
        raise LLMError(
            "OpenRouter API key not configured",
            code=ErrorCode.API_KEY_MISSING,
            cause=e,
        )
    
    return ChatOpenAI(
        model=config.llm.model_name,
        openai_api_key=config.openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "RAG Assistant",
        },
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        request_timeout=config.llm.timeout,
    )


def retrieve_context(
    query: str,
    vector_store: FAISS,
    top_k: int | None = None,
    config: Config | None = None,
    use_cache: bool = True,
) -> List[RetrievalResult]:
    """Retrieve relevant document chunks for a query.
    
    Args:
        query: User question.
        vector_store: FAISS vector store.
        top_k: Number of chunks to retrieve.
        config: Configuration object.
        use_cache: Whether to use query cache.
        
    Returns:
        List of RetrievalResult objects with retrieved chunks.
        
    Raises:
        RetrievalError: If retrieval fails.
    """
    if config is None:
        config = get_config()
    
    if top_k is None:
        top_k = config.retrieval.top_k
    
    start_time = time.perf_counter()
    
    try:
        # Check cache
        if use_cache:
            cache = get_query_cache()
            cached = cache.get(query, top_k)
            if cached is not None:
                logger.debug("Query cache hit", query=query[:50])
                return cached
        
        # Similarity search with scores
        results: List[Tuple[Document, float]] = vector_store.similarity_search_with_score(
            query, k=top_k
        )
        
        retrieval_results = []
        for doc, score in results:
            # Apply score threshold
            if score > config.retrieval.score_threshold or config.retrieval.score_threshold == 0:
                result = RetrievalResult(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page", 1),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    score=float(score),
                )
                retrieval_results.append(result)
        
        # Update cache
        if use_cache:
            cache.set(query, top_k, retrieval_results)
        
        # Update metrics
        duration = time.perf_counter() - start_time
        retrieval_latency.observe(duration)
        retrieval_counter.labels(status="success").inc()
        chunks_retrieved.observe(len(retrieval_results))
        
        logger.info(
            "Retrieval complete",
            query=query[:50],
            results=len(retrieval_results),
            duration_ms=round(duration * 1000, 2),
        )
        
        return retrieval_results
    
    except Exception as e:
        retrieval_counter.labels(status="error").inc()
        raise RetrievalError(
            f"Failed to retrieve documents: {e}",
            code=ErrorCode.RETRIEVAL_FAILED,
            cause=e,
        )


def build_prompt(query: str, sources: List[RetrievalResult]) -> str:
    """Build the prompt for the LLM with context and instructions.
    
    Args:
        query: User question.
        sources: Retrieved document chunks.
        
    Returns:
        Formatted prompt string.
    """
    if not sources:
        return f"""You are a helpful assistant. The user asked: "{query}"

Unfortunately, no relevant documents were found to answer this question.

Respond with: "I don't know based on the provided documents."
"""
    
    # Build context section with citations
    context_parts = []
    for i, source in enumerate(sources, 1):
        citation = source.citation
        context_parts.append(f"[{i}] {citation}\n{source.content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    return f"""You are a helpful assistant that answers questions based on the provided document context.

IMPORTANT RULES:
1. Only use information from the provided context to answer the question.
2. Include citations in your answer using this format: [source:filename#page|chunk]
3. You MUST cite the specific source for each fact you mention.
4. If the context doesn't contain enough information to answer, say "I don't know based on the provided documents."
5. NEVER fabricate or make up citations. Only cite sources that appear in the context below.

CONTEXT:
{context}

USER QUESTION: {query}

Provide a clear, well-cited answer based ONLY on the context above."""


@with_circuit_breaker("openrouter", failure_threshold=5, recovery_timeout=60.0)
@retry_with_backoff(max_attempts=3, min_wait=1.0, max_wait=30.0)
def call_llm(prompt: str, llm: ChatOpenAI, model_name: str) -> str:
    """Call the LLM with retry and circuit breaker protection.
    
    Args:
        prompt: Formatted prompt.
        llm: LLM instance.
        model_name: Model name for metrics.
        
    Returns:
        LLM response content.
        
    Raises:
        LLMError: If LLM call fails after retries.
    """
    start_time = time.perf_counter()
    
    try:
        response = llm.invoke(prompt)
        
        duration = time.perf_counter() - start_time
        llm_latency.observe(duration)
        llm_request_counter.labels(model=model_name, status="success").inc()
        
        return response.content
    
    except Exception as e:
        llm_request_counter.labels(model=model_name, status="error").inc()
        raise LLMError(
            f"LLM request failed: {e}",
            code=ErrorCode.LLM_REQUEST_FAILED,
            cause=e,
        )


def generate_answer(
    query: str,
    sources: List[RetrievalResult],
    llm: ChatOpenAI,
    config: Config | None = None,
) -> RAGResponse:
    """Generate an answer using the LLM with retrieved context.
    
    Args:
        query: User question.
        sources: Retrieved document chunks.
        llm: LLM instance.
        config: Configuration object.
        
    Returns:
        RAGResponse with answer and sources.
    """
    if config is None:
        config = get_config()
    
    start_time = time.perf_counter()
    
    # Build prompt
    prompt = build_prompt(query, sources)
    
    # Generate response
    answer = call_llm(prompt, llm, config.llm.model_name)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    logger.info(
        "Answer generated",
        query=query[:50],
        sources=len(sources),
        answer_length=len(answer),
        latency_ms=round(latency_ms, 2),
    )
    
    return RAGResponse(
        answer=answer,
        sources=sources,
        query=query,
        latency_ms=latency_ms,
    )


class RAGAssistant:
    """Main RAG Assistant class for interactive question answering."""
    
    def __init__(
        self,
        config: Config | None = None,
        index_path: str | Path | None = None,
    ):
        """Initialize the RAG Assistant.
        
        Args:
            config: Configuration object.
            index_path: Path to the FAISS index.
        """
        self.config = config or get_config()
        self.project_root = get_project_root()
        
        # Configure logging if not already done
        configure_logging(
            log_dir=self.project_root / self.config.paths.logs_dir,
            log_level=self.config.logging.level,
            json_format=self.config.logging.json_format,
        )
        
        logger.info("Initializing RAG Assistant")
        
        # Load components
        self.vector_store = load_index(index_path, self.config)
        self.llm = get_llm(self.config)
        
        logger.info("RAG Assistant ready")
    
    def ask(
        self,
        query: str,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> RAGResponse:
        """Ask a question and get an answer with sources.
        
        Args:
            query: User question.
            top_k: Number of chunks to retrieve.
            use_cache: Whether to use query caching.
            
        Returns:
            RAGResponse with answer and sources.
        """
        start_time = time.perf_counter()
        
        try:
            # Validate and sanitize query
            query = validate_query(query)
            
            with RequestLogger(logger, "query", query=query[:50]):
                # Retrieve relevant chunks
                sources = retrieve_context(
                    query,
                    self.vector_store,
                    top_k=top_k,
                    config=self.config,
                    use_cache=use_cache,
                )
                
                # Generate answer
                response = generate_answer(
                    query, sources, self.llm, self.config
                )
            
            # Update metrics
            duration = time.perf_counter() - start_time
            query_latency.observe(duration)
            query_counter.labels(status="success").inc()
            
            return response
        
        except Exception as e:
            query_counter.labels(status="error").inc()
            raise
    
    def format_response(self, response: RAGResponse) -> str:
        """Format response for display.
        
        Args:
            response: RAGResponse object.
            
        Returns:
            Formatted string for display.
        """
        output_parts = [
            "\n" + "=" * 60,
            "ANSWER:",
            response.answer,
            "\n" + "-" * 60,
            f"SOURCES (latency: {response.latency_ms:.0f}ms):",
        ]
        
        for i, source in enumerate(response.sources, 1):
            output_parts.append(
                f"  [{i}] {source.citation} (similarity: {source.score:.4f})"
            )
        
        output_parts.append("=" * 60)
        
        return "\n".join(output_parts)
    
    def health_check(self) -> dict:
        """Perform health check.
        
        Returns:
            Health status dictionary.
        """
        return {
            "status": "healthy",
            "model": self.config.llm.model_name,
            "embeddings": f"{self.config.embeddings.provider}/{self.config.embeddings.model}",
            "index_loaded": self.vector_store is not None,
        }


def run_cli():
    """Run the interactive CLI chat loop."""
    configure_logging()
    
    print("=" * 60)
    print("RAG Assistant - Interactive Chat")
    print("=" * 60)
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'sources' to show sources from last answer.")
    print("Type 'health' to check system status.")
    print("=" * 60 + "\n")
    
    try:
        assistant = RAGAssistant()
    except Exception as e:
        logger.error(f"Failed to initialize assistant: {e}")
        print(f"\nError: {e}")
        return
    
    last_response: RAGResponse | None = None
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ("quit", "exit"):
                print("\nGoodbye!")
                break
            
            if query.lower() == "health":
                health = assistant.health_check()
                print("\nSystem Status:")
                for key, value in health.items():
                    print(f"  {key}: {value}")
                continue
            
            if query.lower() == "sources" and last_response:
                print("\nSources from last answer:")
                for i, source in enumerate(last_response.sources, 1):
                    print(f"\n[{i}] {source.citation}")
                    print(f"    Score: {source.score:.4f}")
                    print(f"    Content: {source.content[:200]}...")
                continue
            
            # Get answer
            response = assistant.ask(query)
            last_response = response
            
            # Display formatted response
            print(assistant.format_response(response))
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Query error: {e}")
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    run_cli()
