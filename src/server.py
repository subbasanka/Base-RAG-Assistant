"""FastAPI server for RAG Assistant.

Provides REST API endpoints for production deployment with health checks,
metrics, and proper error handling.
"""

import time
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel, Field

from src.build_index import build_and_save_index
from src.config import get_config, get_project_root
from src.exceptions import RAGException, ValidationError, ErrorCode
from src.logging_config import configure_logging, get_logger
from src.metrics import get_metrics
from src.rag_chat import RAGAssistant, RAGResponse, RetrievalResult

logger = get_logger(__name__)

# --- Pydantic Models ---

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of document chunks to retrieve")

class Source(BaseModel):
    content: str
    source: str
    page: int
    score: float
    citation: str

class QueryResponse(BaseModel):
    answer: str
    query: str
    sources: List[Source]
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    model: str
    embeddings: str
    index_loaded: bool

class ConfigResponse(BaseModel):
    llm: dict
    retrieval: dict
    embeddings: dict
    environment: str

class UploadResponse(BaseModel):
    status: str
    filename: str
    message: str
    index_triggered: bool = False

# Supported file types for upload
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# --- Global State & Lifespan ---

class GlobalState:
    assistant: RAGAssistant | None = None

state = GlobalState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    config = get_config()
    project_root = get_project_root()
    
    configure_logging(
        log_dir=project_root / config.paths.logs_dir,
        log_level=config.logging.level,
        json_format=config.is_production,
    )
    
    logger.info("Starting RAG Assistant server", environment=config.environment)
    
    try:
        state.assistant = RAGAssistant()
        logger.info("RAG Assistant initialized successfully")
    except Exception as e:
        # Check if it's a missing index error (expected on first run)
        error_msg = str(e)
        if "index" in error_msg.lower() or "E303" in error_msg:
            logger.info("No index found - server ready for document uploads")
        else:
            logger.error(f"Failed to initialize assistant: {e}")
        logger.info("Server starting without index (upload documents to create one)")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Assistant server")

# --- App Initialization ---

app = FastAPI(
    title="RAG Assistant API",
    description="Production-ready RAG Assistant API with FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    
    logger.info(
        "HTTP request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration * 1000, 2),
    )
    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if not state.assistant:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    try:
        health = state.assistant.health_check()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            **health
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/metrics")
async def metrics():
    try:
        return Response(content=get_metrics(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {e}")

@app.get("/config", response_model=ConfigResponse)
async def get_configuration():
    try:
        config = get_config()
        return {
            "llm": {
                "model_name": config.llm.model_name,
                "temperature": config.llm.temperature,
                "max_tokens": config.llm.max_tokens,
            },
            "retrieval": {
                "top_k": config.retrieval.top_k,
            },
            "embeddings": {
                "provider": config.embeddings.provider,
                "model": config.embeddings.model,
            },
            "environment": config.environment,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {e}")

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not state.assistant:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Note: RAGAssistant.ask is currently synchronous
        # In a real async app, we'd want to make this async or run in a threadpool
        response = state.assistant.ask(request.query, top_k=request.top_k)
        
        # Convert RAGResponse to Pydantic model
        sources_data = [
            Source(
                content=s.content,
                source=s.source,
                page=s.page,
                score=s.score,
                citation=s.citation
            ) for s in response.sources
        ]
        
        return QueryResponse(
            answer=response.answer,
            query=response.query,
            sources=sources_data,
            latency_ms=response.latency_ms
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RAGException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    auto_index: bool = True,
    background_tasks: BackgroundTasks = None,
):
    """Upload a document and optionally trigger index rebuild.
    
    Args:
        file: The document file (PDF, TXT, or MD)
        auto_index: If True, automatically rebuild the index after upload
    
    Returns:
        Upload status with filename and indexing info
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    # Get documents directory
    config = get_config()
    project_root = get_project_root()
    docs_dir = project_root / config.paths.documents_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create safe filename
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in file.filename)
    file_path = docs_dir / safe_name
    
    # Handle duplicate names
    counter = 1
    original_stem = file_path.stem
    while file_path.exists():
        file_path = docs_dir / f"{original_stem}_{counter}{file_path.suffix}"
        counter += 1
    
    # Save file
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"Document uploaded: {file_path.name}")
    except Exception as e:
        logger.exception("File upload failed")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Trigger index rebuild if requested
    index_triggered = False
    if auto_index and background_tasks:
        background_tasks.add_task(run_rebuild_task)
        index_triggered = True
        logger.info("Index rebuild triggered after upload")
    
    return UploadResponse(
        status="success",
        filename=file_path.name,
        message=f"File uploaded successfully" + (" - indexing started" if index_triggered else ""),
        index_triggered=index_triggered,
    )

def run_rebuild_task():
    """Background task to rebuild index."""
    try:
        logger.info("Starting background index rebuild")
        build_and_save_index()
        # Reinitialize assistant to pick up new index
        state.assistant = RAGAssistant()
        logger.info("Index rebuild complete and assistant reloaded")
    except Exception as e:
        logger.exception("Background index rebuild failed")

@app.post("/index/rebuild")
async def rebuild_index(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_rebuild_task)
    return {"status": "accepted", "message": "Index rebuild started in background"}

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the server using uvicorn."""
    uvicorn.run("src.server:app", host=host, port=port, reload=False) # Note: assumes this file is src/server.py

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Assistant API Server (FastAPI)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
