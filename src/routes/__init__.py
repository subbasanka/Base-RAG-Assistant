"""Routes package for RAG Assistant API."""

from src.routes.health import router as health_router
from src.routes.query import router as query_router
from src.routes.upload import router as upload_router

__all__ = ["health_router", "query_router", "upload_router"]
