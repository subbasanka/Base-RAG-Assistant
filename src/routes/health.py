"""Health and config routes for RAG Assistant API."""

import time

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from src.config import get_config
from src.dependencies import get_assistant
from src.logging_config import get_logger
from src.metrics import get_metrics

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])


# --- Pydantic Models ---

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


# --- Routes ---

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check health status of the server."""
    assistant = get_assistant()
    if not assistant:
        raise HTTPException(status_code=503, detail="Assistant not initialized")
    
    try:
        health = assistant.health_check()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            **health
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/metrics")
async def metrics():
    """Get Prometheus metrics."""
    try:
        return Response(content=get_metrics(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {e}")


@router.get("/config", response_model=ConfigResponse)
async def get_configuration():
    """Get current configuration (non-sensitive)."""
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
