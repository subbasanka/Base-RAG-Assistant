"""Document ingestion module for RAG Assistant.

Handles loading documents from various formats (PDF, TXT, MD),
cleaning text, and chunking with metadata attachment.
"""

import re
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import get_config, get_project_root, Config
from src.exceptions import DocumentError, ErrorCode
from src.logging_config import get_logger, RequestLogger
from src.validation import validate_chunk_params, validate_file_path

logger = get_logger(__name__)


# Supported file extensions
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md"]


def clean_text(text: str) -> str:
    """Clean text by removing excessive whitespace and normalizing.
    
    Args:
        text: Raw text content.
        
    Returns:
        Cleaned text with normalized whitespace.
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Remove null characters
    text = text.replace('\x00', '')
    # Normalize unicode
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    return text


def load_pdf(file_path: Path) -> List[Document]:
    """Load a PDF file and extract text with page numbers.
    
    Args:
        file_path: Path to the PDF file.
        
    Returns:
        List of Document objects, one per page.
        
    Raises:
        DocumentError: If PDF loading fails.
    """
    from langchain_community.document_loaders import PyPDFLoader
    
    try:
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        
        # Clean text and ensure metadata
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["source"] = file_path.name
            # PyPDFLoader uses 0-indexed pages, convert to 1-indexed
            if "page" in doc.metadata:
                doc.metadata["page"] = doc.metadata["page"] + 1
        
        logger.info(
            "PDF loaded successfully",
            file=file_path.name,
            pages=len(documents),
        )
        return documents
        
    except Exception as e:
        raise DocumentError(
            f"Failed to load PDF: {file_path.name}",
            code=ErrorCode.DOCUMENT_LOAD_FAILED,
            details={"file": str(file_path)},
            cause=e,
        )


def load_text(file_path: Path) -> List[Document]:
    """Load a text file.
    
    Args:
        file_path: Path to the text file.
        
    Returns:
        List containing a single Document object.
        
    Raises:
        DocumentError: If text file loading fails.
    """
    from langchain_community.document_loaders import TextLoader
    
    try:
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()
        
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["source"] = file_path.name
            doc.metadata["page"] = 1
        
        logger.info(
            "Text file loaded successfully",
            file=file_path.name,
            chars=len(documents[0].page_content) if documents else 0,
        )
        return documents
        
    except Exception as e:
        raise DocumentError(
            f"Failed to load text file: {file_path.name}",
            code=ErrorCode.DOCUMENT_LOAD_FAILED,
            details={"file": str(file_path)},
            cause=e,
        )


def load_markdown(file_path: Path) -> List[Document]:
    """Load a Markdown file.
    
    Args:
        file_path: Path to the Markdown file.
        
    Returns:
        List containing a single Document object.
        
    Raises:
        DocumentError: If Markdown loading fails.
    """
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    
    try:
        loader = UnstructuredMarkdownLoader(str(file_path))
        documents = loader.load()
        
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["source"] = file_path.name
            doc.metadata["page"] = 1
        
        logger.info(
            "Markdown file loaded successfully",
            file=file_path.name,
            chars=len(documents[0].page_content) if documents else 0,
        )
        return documents
        
    except Exception as e:
        raise DocumentError(
            f"Failed to load Markdown file: {file_path.name}",
            code=ErrorCode.DOCUMENT_LOAD_FAILED,
            details={"file": str(file_path)},
            cause=e,
        )


def load_documents(
    documents_dir: str | Path | None = None,
    config: Config | None = None,
) -> List[Document]:
    """Load all documents from the specified directory.
    
    Supports PDF, TXT, and MD files.
    
    Args:
        documents_dir: Path to documents directory. Uses config default if None.
        config: Configuration object.
        
    Returns:
        List of Document objects from all loaded files.
        
    Raises:
        DocumentError: If documents directory doesn't exist or no documents found.
    """
    if config is None:
        config = get_config()
    
    project_root = get_project_root()
    
    if documents_dir is None:
        documents_dir = project_root / config.paths.documents_dir
    else:
        documents_dir = Path(documents_dir)
    
    if not documents_dir.exists():
        raise DocumentError(
            f"Documents directory not found: {documents_dir}",
            code=ErrorCode.DOCUMENTS_NOT_FOUND,
            details={"path": str(documents_dir)},
        )
    
    all_documents: List[Document] = []
    errors: List[dict] = []
    
    # Define supported extensions and their loaders
    loaders = {
        ".pdf": load_pdf,
        ".txt": load_text,
        ".md": load_markdown,
    }
    
    with RequestLogger(logger, "document_loading", directory=str(documents_dir)):
        # Iterate through all files
        for ext, loader_func in loaders.items():
            for file_path in documents_dir.glob(f"*{ext}"):
                try:
                    docs = loader_func(file_path)
                    all_documents.extend(docs)
                except DocumentError as e:
                    logger.warning(
                        f"Failed to load document",
                        file=file_path.name,
                        error=str(e),
                    )
                    errors.append({
                        "file": file_path.name,
                        "error": str(e),
                    })
    
    if not all_documents:
        raise DocumentError(
            f"No documents loaded from {documents_dir}",
            code=ErrorCode.DOCUMENTS_NOT_FOUND,
            details={
                "path": str(documents_dir),
                "errors": errors,
                "supported_formats": SUPPORTED_EXTENSIONS,
            },
        )
    
    logger.info(
        "Documents loaded",
        total_documents=len(all_documents),
        failed=len(errors),
    )
    
    return all_documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    config: Config | None = None,
) -> List[Document]:
    """Split documents into chunks with metadata.
    
    Args:
        documents: List of Document objects to chunk.
        chunk_size: Characters per chunk. Uses config default if None.
        chunk_overlap: Overlap between chunks. Uses config default if None.
        config: Configuration object.
        
    Returns:
        List of chunked Document objects with chunk_index metadata.
    """
    if config is None:
        config = get_config()
    
    if chunk_size is None:
        chunk_size = config.chunking.chunk_size
    if chunk_overlap is None:
        chunk_overlap = config.chunking.chunk_overlap
    
    # Validate parameters
    chunk_size, chunk_overlap = validate_chunk_params(chunk_size, chunk_overlap)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    with RequestLogger(
        logger,
        "chunking",
        documents=len(documents),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ):
        chunks = text_splitter.split_documents(documents)
        
        # Add chunk index to metadata
        source_chunk_counts: dict[str, int] = {}
        
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            if source not in source_chunk_counts:
                source_chunk_counts[source] = 0
            
            chunk.metadata["chunk_index"] = source_chunk_counts[source]
            source_chunk_counts[source] += 1
    
    logger.info(
        "Documents chunked",
        input_documents=len(documents),
        output_chunks=len(chunks),
        sources=len(source_chunk_counts),
    )
    
    return chunks


def ingest_documents(
    documents_dir: str | Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    config: Config | None = None,
) -> List[Document]:
    """Full ingestion pipeline: load documents and chunk them.
    
    Args:
        documents_dir: Path to documents directory.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.
        config: Configuration object.
        
    Returns:
        List of chunked Document objects ready for embedding.
    """
    if config is None:
        config = get_config()
    
    logger.info("Starting document ingestion")
    
    # Load documents
    documents = load_documents(documents_dir, config)
    
    # Chunk documents
    chunks = chunk_documents(documents, chunk_size, chunk_overlap, config)
    
    logger.info(
        "Document ingestion complete",
        total_chunks=len(chunks),
    )
    
    return chunks


if __name__ == "__main__":
    from src.logging_config import configure_logging
    
    configure_logging()
    
    try:
        chunks = ingest_documents()
        
        if chunks:
            print(f"\nIngested {len(chunks)} chunks")
            print("\nSample chunk:")
            print(f"  Source: {chunks[0].metadata.get('source')}")
            print(f"  Page: {chunks[0].metadata.get('page')}")
            print(f"  Chunk Index: {chunks[0].metadata.get('chunk_index')}")
            print(f"  Content preview: {chunks[0].page_content[:200]}...")
    except DocumentError as e:
        print(f"Error: {e}")
