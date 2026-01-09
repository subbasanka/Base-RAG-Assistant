"""Query routes for RAG Assistant API."""

import json
import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.dependencies import require_assistant
from src.exceptions import RAGException, ValidationError
from src.logging_config import get_logger
from src.rag_chat import RAGAssistant

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


# --- Pydantic Models ---

class Source(BaseModel):
    content: str
    source: str
    page: int
    score: float
    citation: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question")
    top_k: int | None = Field(None, ge=1, le=20, description="Number of documents to retrieve")


class QueryResponse(BaseModel):
    answer: str
    query: str
    sources: List[Source]
    latency_ms: float


# --- Routes ---

@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest, assistant: RAGAssistant = Depends(require_assistant)):
    """Query the RAG assistant with a question."""
    try:
        response = assistant.ask(request.query, top_k=request.top_k)
        
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


@router.post("/stream")
async def query_stream(request: QueryRequest, assistant: RAGAssistant = Depends(require_assistant)):
    """Stream the RAG assistant response as Server-Sent Events."""
    
    async def generate():
        try:
            async for data in assistant.ask_stream(request.query, top_k=request.top_k):
                yield f"data: {json.dumps(data)}\n\n"
        except ValidationError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except RAGException as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            logger.exception("Streaming query failed")
            yield f"data: {json.dumps({'error': f'Internal error: {e}'})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

