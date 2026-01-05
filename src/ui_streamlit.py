"""Streamlit UI for RAG Assistant.

A modern, interactive web interface for the RAG chat system
with document upload and index management capabilities.
Run with: streamlit run src/ui_streamlit.py
"""

import os
import sys
import time
from pathlib import Path
from typing import List

# Disable Prometheus metrics for Streamlit (prevents re-registration errors)
os.environ["RAG_METRICS_ENABLED"] = "false"

import streamlit as st

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config, get_project_root, clear_config_cache
from src.exceptions import RAGException
from src.logging_config import configure_logging
from src.rag_chat import RAGAssistant, RAGResponse


# Supported file types
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md"]
SUPPORTED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "text/x-markdown": ".md",
}


# Page configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .upload-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 2px dashed #dee2e6;
    }
    
    .file-item {
        background-color: #e9ecef;
        border-radius: 5px;
        padding: 8px 12px;
        margin: 4px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
    }
    
    .warning-message {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 10px;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)


def get_documents_dir() -> Path:
    """Get the documents directory path."""
    config = get_config()
    return get_project_root() / config.paths.documents_dir


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "assistant" not in st.session_state:
        st.session_state.assistant = None
    
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    if "index_needs_rebuild" not in st.session_state:
        st.session_state.index_needs_rebuild = False


def load_assistant(force_reload: bool = False) -> bool:
    """Load or reload the RAG Assistant."""
    if st.session_state.assistant is not None and not force_reload:
        return True
    
    try:
        config = get_config()
        configure_logging(
            log_dir=get_project_root() / config.paths.logs_dir,
            log_level=config.logging.level,
        )
        
        with st.spinner("Loading RAG Assistant..."):
            st.session_state.assistant = RAGAssistant()
            st.session_state.index_needs_rebuild = False
        return True
    except RAGException as e:
        st.session_state.assistant = None
        return False
    except Exception as e:
        st.session_state.assistant = None
        return False


def get_existing_documents() -> List[Path]:
    """Get list of existing documents in the documents directory."""
    docs_dir = get_documents_dir()
    if not docs_dir.exists():
        return []
    
    documents = []
    for ext in SUPPORTED_EXTENSIONS:
        documents.extend(docs_dir.glob(f"*{ext}"))
    
    return sorted(documents, key=lambda x: x.name.lower())


def save_uploaded_file(uploaded_file) -> Path | None:
    """Save an uploaded file to the documents directory.
    
    Args:
        uploaded_file: Streamlit uploaded file object.
        
    Returns:
        Path to saved file or None if failed.
    """
    docs_dir = get_documents_dir()
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine file extension
    file_ext = Path(uploaded_file.name).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        # Try to get from MIME type
        mime_type = uploaded_file.type
        file_ext = SUPPORTED_MIME_TYPES.get(mime_type, "")
        if not file_ext:
            return None
    
    # Create safe filename
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in uploaded_file.name)
    if not safe_name.lower().endswith(tuple(SUPPORTED_EXTENSIONS)):
        safe_name += file_ext
    
    file_path = docs_dir / safe_name
    
    # Handle duplicate names
    counter = 1
    original_stem = file_path.stem
    while file_path.exists():
        file_path = docs_dir / f"{original_stem}_{counter}{file_path.suffix}"
        counter += 1
    
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Failed to save file: {e}")
        return None


def delete_document(file_path: Path) -> bool:
    """Delete a document from the documents directory.
    
    Args:
        file_path: Path to the file to delete.
        
    Returns:
        True if deleted successfully.
    """
    try:
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    except Exception as e:
        st.error(f"Failed to delete file: {e}")
        return False


def rebuild_index() -> bool:
    """Rebuild the FAISS index with current documents."""
    try:
        from src.build_index import build_and_save_index
        
        with st.spinner("Rebuilding index... This may take a moment."):
            build_and_save_index()
        
        # Reload assistant with new index
        st.session_state.assistant = None
        clear_config_cache()
        return load_assistant(force_reload=True)
    except Exception as e:
        st.error(f"Failed to rebuild index: {e}")
        return False


def display_sources(sources):
    """Display retrieved sources in an expandable section."""
    if not sources:
        return
    
    with st.expander(f"📑 Retrieved Sources ({len(sources)})", expanded=False):
        for i, source in enumerate(sources, 1):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"**[{i}] {source.citation}**")
            
            with col2:
                st.markdown(f"Score: `{source.score:.4f}`")
            
            content_preview = source.content[:300] + "..." if len(source.content) > 300 else source.content
            st.markdown(f"*{content_preview}*")
            st.divider()


def display_metrics(response: RAGResponse):
    """Display response metrics."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Latency", f"{response.latency_ms:.0f}ms")
    
    with col2:
        st.metric("Sources", len(response.sources))
    
    with col3:
        if response.sources:
            avg_score = sum(s.score for s in response.sources) / len(response.sources)
            st.metric("Avg Score", f"{avg_score:.3f}")


def render_upload_section():
    """Render the document upload section."""
    st.subheader("📤 Upload Documents")
    
    # Auto-index toggle
    auto_index = st.checkbox(
        "🔄 Auto-build index after upload",
        value=True,
        help="Automatically rebuild the search index when new files are uploaded",
    )
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, Markdown",
        key="file_uploader",
    )
    
    if uploaded_files:
        saved_files = []
        failed_files = []
        
        for uploaded_file in uploaded_files:
            # Check if already processed in this session
            if uploaded_file.name in st.session_state.uploaded_files:
                continue
            
            saved_path = save_uploaded_file(uploaded_file)
            if saved_path:
                saved_files.append(saved_path.name)
                st.session_state.uploaded_files.append(uploaded_file.name)
            else:
                failed_files.append(uploaded_file.name)
        
        if saved_files:
            st.success(f"✅ Uploaded {len(saved_files)} file(s): {', '.join(saved_files)}")
            
            # Auto-build index if enabled
            if auto_index:
                st.info("🔨 Building search index...")
                if rebuild_index():
                    st.success("✅ Index built! You can now search your documents.")
                    st.session_state.index_needs_rebuild = False
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Index build failed. Try manually with the Rebuild button.")
                    st.session_state.index_needs_rebuild = True
            else:
                st.session_state.index_needs_rebuild = True
        
        if failed_files:
            st.error(f"❌ Failed to upload: {', '.join(failed_files)}")
    
    # Show rebuild warning if manual mode
    if st.session_state.index_needs_rebuild and not auto_index:
        st.warning("⚠️ New documents uploaded. Click 'Rebuild Index' to include them in searches.")


def render_document_manager():
    """Render the document management section."""
    st.subheader("📁 Document Manager")
    
    documents = get_existing_documents()
    
    if not documents:
        st.info("No documents found. Upload some documents to get started!")
        return
    
    st.write(f"**{len(documents)} document(s) in library:**")
    
    # Display documents with delete option
    for doc in documents:
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            # File icon based on type
            icon = "📄"
            if doc.suffix.lower() == ".pdf":
                icon = "📕"
            elif doc.suffix.lower() == ".md":
                icon = "📝"
            
            st.write(f"{icon} {doc.name}")
        
        with col2:
            # File size
            size_kb = doc.stat().st_size / 1024
            if size_kb < 1024:
                st.write(f"{size_kb:.1f} KB")
            else:
                st.write(f"{size_kb/1024:.1f} MB")
        
        with col3:
            # Delete button
            if st.button("🗑️", key=f"delete_{doc.name}", help=f"Delete {doc.name}"):
                if delete_document(doc):
                    st.session_state.index_needs_rebuild = True
                    st.rerun()


def render_sidebar():
    """Render the sidebar with controls and settings."""
    with st.sidebar:
        st.title("📚 RAG Assistant")
        
        # Tab selection for sidebar content
        tab = st.radio(
            "Mode",
            ["💬 Chat", "📤 Upload", "📁 Manage"],
            horizontal=True,
            label_visibility="collapsed",
        )
        
        st.markdown("---")
        
        if tab == "📤 Upload":
            render_upload_section()
        elif tab == "📁 Manage":
            render_document_manager()
        else:
            # Chat mode - show config
            try:
                config = get_config()
                st.subheader("⚙️ Configuration")
                st.text(f"Model: {config.llm.model_name}")
                st.text(f"Embeddings: {config.embeddings.provider}")
                st.text(f"Top-K: {config.retrieval.top_k}")
            except Exception:
                pass
            
            st.markdown("---")
            
            # Session stats
            st.subheader("📊 Session Stats")
            st.text(f"Queries: {st.session_state.total_queries}")
            st.text(f"Messages: {len(st.session_state.messages)}")
        
        st.markdown("---")
        
        # Index controls
        st.subheader("🔧 Index Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            rebuild_btn = st.button(
                "🔨 Rebuild",
                use_container_width=True,
                help="Rebuild index with current documents",
                type="primary" if st.session_state.index_needs_rebuild else "secondary",
            )
        
        with col2:
            reload_btn = st.button(
                "🔄 Reload",
                use_container_width=True,
                help="Reload existing index",
            )
        
        if rebuild_btn:
            if rebuild_index():
                st.success("Index rebuilt!")
                time.sleep(1)
                st.rerun()
        
        if reload_btn:
            st.session_state.assistant = None
            if load_assistant(force_reload=True):
                st.success("Index reloaded!")
                st.rerun()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()
        
        st.markdown("---")
        
        # Display options
        st.subheader("👁️ Display Options")
        show_sources = st.checkbox("Show sources", value=True)
        show_metrics = st.checkbox("Show metrics", value=True)
        
        st.markdown("---")
        
        # Status
        docs = get_existing_documents()
        st.caption(f"📄 {len(docs)} documents in library")
        
        if st.session_state.assistant:
            st.caption("✅ Index loaded")
        else:
            st.caption("⚠️ Index not loaded")
        
        if st.session_state.index_needs_rebuild:
            st.caption("🔄 Rebuild needed")
        
        return show_sources, show_metrics


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Render sidebar and get display options
    show_sources, show_metrics = render_sidebar()
    
    # Main content area
    st.title("💬 Ask Your Documents")
    
    # Check if we have documents
    docs = get_existing_documents()
    if not docs:
        st.warning(
            "📭 No documents found! Use the **Upload** tab in the sidebar to add documents, "
            "or place files in the `documents/` folder."
        )
        
        # Show upload section inline as well
        st.markdown("---")
        st.subheader("🚀 Quick Start - Upload Your Documents")
        
        uploaded_files = st.file_uploader(
            "Upload your first documents (PDF, TXT, or Markdown)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            key="main_uploader",
        )
        
        if uploaded_files:
            saved_files = []
            for f in uploaded_files:
                # Avoid re-processing
                if f.name not in st.session_state.uploaded_files:
                    if save_uploaded_file(f):
                        saved_files.append(f.name)
                        st.session_state.uploaded_files.append(f.name)
            
            if saved_files:
                st.success(f"✅ Uploaded {len(saved_files)} file(s): {', '.join(saved_files)}")
                
                # Automatically build the index
                st.info("🔨 Building search index... This may take a moment.")
                if rebuild_index():
                    st.success("✅ Ready! You can now ask questions about your documents.")
                    st.session_state.index_needs_rebuild = False
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Index build failed. Check that documents are valid.")
                    st.session_state.index_needs_rebuild = True
        
        return
    
    # Try to load assistant
    if st.session_state.assistant is None:
        if not load_assistant():
            st.warning(
                "⚠️ Index not built yet. Click **Rebuild** in the sidebar to create the index."
            )
            
            # Offer quick rebuild
            if st.button("🔨 Build Index Now", type="primary"):
                if rebuild_index():
                    st.success("Index built successfully!")
                    st.rerun()
            return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant":
                if show_sources and "sources" in message and message["sources"]:
                    display_sources(message["sources"])
                if show_metrics and "latency_ms" in message:
                    st.caption(
                        f"⏱️ {message['latency_ms']:.0f}ms | "
                        f"📑 {message.get('source_count', 0)} sources"
                    )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response: RAGResponse = st.session_state.assistant.ask(prompt)
                    answer = response.answer
                    sources = response.sources
                    
                    st.markdown(answer)
                    
                    if show_sources and sources:
                        display_sources(sources)
                    
                    if show_metrics:
                        display_metrics(response)
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "latency_ms": response.latency_ms,
                        "source_count": len(sources),
                    })
                    st.session_state.last_sources = sources
                    st.session_state.total_queries += 1
                    
                except RAGException as e:
                    error_msg = f"Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ {error_msg}",
                        "sources": [],
                    })
                except Exception as e:
                    error_msg = f"Unexpected error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ {error_msg}",
                        "sources": [],
                    })


if __name__ == "__main__":
    main()
