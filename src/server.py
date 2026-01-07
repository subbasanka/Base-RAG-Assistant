"""FastAPI server for RAG Assistant.

Provides REST API endpoints for production deployment with health checks,
metrics, and proper error handling.
"""

import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_config, get_project_root
from src.dependencies import init_assistant
from src.logging_config import configure_logging, get_logger
from src.routes import health_router, query_router, upload_router

logger = get_logger(__name__)


# --- App Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    config = get_config()
    project_root = get_project_root()
    
    configure_logging(
        log_dir=project_root / config.paths.logs_dir,
        log_level=config.logging.level,
        json_format=config.is_production,
    )
    
    logger.info("Starting RAG Assistant server", environment=config.environment)
    
    # Initialize assistant (may fail if no index exists)
    init_assistant()
    
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


# --- Middleware ---

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing."""
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


# --- Register Routes ---

app.include_router(health_router)
app.include_router(query_router)
app.include_router(upload_router)


# --- Entry Point ---

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the server using uvicorn."""
    uvicorn.run("src.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Assistant API Server (FastAPI)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
