"""Upload and index routes for RAG Assistant API."""

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.build_index import build_and_save_index
from src.config import get_config, get_project_root
from src.dependencies import set_assistant
from src.logging_config import get_logger
from src.rag_chat import RAGAssistant

logger = get_logger(__name__)

router = APIRouter(tags=["Documents"])


# --- Constants ---

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


# --- Pydantic Models ---

class UploadResponse(BaseModel):
    status: str
    filename: str
    message: str
    index_triggered: bool = False


# --- Background Tasks ---

def run_rebuild_task():
    """Background task to rebuild index."""
    try:
        logger.info("Starting background index rebuild")
        build_and_save_index()
        # Reinitialize assistant to pick up new index
        set_assistant(RAGAssistant())
        logger.info("Index rebuild complete and assistant reloaded")
    except Exception as e:
        logger.exception("Background index rebuild failed")


# --- Routes ---

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    auto_index: bool = True,
    background_tasks: BackgroundTasks = None,
):
    """Upload a document and optionally trigger index rebuild."""
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


@router.post("/index/rebuild")
async def rebuild_index(background_tasks: BackgroundTasks):
    """Trigger index rebuild in background."""
    background_tasks.add_task(run_rebuild_task)
    return {"status": "accepted", "message": "Index rebuild started in background"}
