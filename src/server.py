"""HTTP API server for RAG Assistant.

Provides REST API endpoints for production deployment with health checks,
metrics, and proper error handling.
"""

import json
import os
import sys
import time
from dataclasses import asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.config import get_config, get_project_root
from src.exceptions import RAGException, ValidationError
from src.logging_config import configure_logging, get_logger
from src.metrics import get_metrics
from src.rag_chat import RAGAssistant, RAGResponse
from src.validation import validate_query

logger = get_logger(__name__)


class RAGRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for RAG Assistant API."""
    
    # Class-level assistant instance (shared across requests)
    assistant: RAGAssistant | None = None
    
    def log_message(self, format: str, *args) -> None:
        """Override to use structured logging."""
        logger.info(
            "HTTP request",
            method=args[0].split()[0] if args else "",
            path=args[0].split()[1] if args and len(args[0].split()) > 1 else "",
            status=args[1] if len(args) > 1 else "",
        )
    
    def send_json_response(
        self,
        data: dict | list,
        status_code: int = 200,
    ) -> None:
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def send_error_response(
        self,
        message: str,
        status_code: int = 500,
        error_code: str | None = None,
    ) -> None:
        """Send error response."""
        error_data = {
            "error": message,
            "status": status_code,
        }
        if error_code:
            error_data["code"] = error_code
        
        self.send_json_response(error_data, status_code)
    
    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == "/health":
            self.handle_health()
        elif path == "/metrics":
            self.handle_metrics()
        elif path == "/config":
            self.handle_config()
        else:
            self.send_error_response("Not Found", 404)
    
    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == "/query":
            self.handle_query()
        elif path == "/index/rebuild":
            self.handle_rebuild_index()
        else:
            self.send_error_response("Not Found", 404)
    
    def handle_health(self) -> None:
        """Health check endpoint."""
        try:
            if self.assistant is None:
                self.send_json_response({
                    "status": "unhealthy",
                    "reason": "Assistant not initialized",
                }, 503)
                return
            
            health = self.assistant.health_check()
            self.send_json_response({
                "status": "healthy",
                "timestamp": time.time(),
                **health,
            })
        except Exception as e:
            self.send_json_response({
                "status": "unhealthy",
                "reason": str(e),
            }, 503)
    
    def handle_metrics(self) -> None:
        """Prometheus metrics endpoint."""
        try:
            metrics = get_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(metrics)
        except Exception as e:
            self.send_error_response(f"Failed to get metrics: {e}", 500)
    
    def handle_config(self) -> None:
        """Get current configuration (non-sensitive)."""
        try:
            config = get_config()
            self.send_json_response({
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
            })
        except Exception as e:
            self.send_error_response(f"Failed to get config: {e}", 500)
    
    def handle_query(self) -> None:
        """Handle query requests."""
        try:
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_error_response("Request body required", 400)
                return
            
            body = self.rfile.read(content_length)
            data = json.loads(body.decode())
            
            # Validate query
            query = data.get("query", "").strip()
            if not query:
                self.send_error_response("Query is required", 400)
                return
            
            try:
                query = validate_query(query)
            except ValidationError as e:
                self.send_error_response(str(e), 400, e.code.value)
                return
            
            # Get optional parameters
            top_k = data.get("top_k")
            
            # Ensure assistant is initialized
            if self.assistant is None:
                self.send_error_response("Service not ready", 503)
                return
            
            # Process query
            response = self.assistant.ask(query, top_k=top_k)
            
            self.send_json_response(response.to_dict())
            
        except json.JSONDecodeError:
            self.send_error_response("Invalid JSON", 400)
        except RAGException as e:
            self.send_error_response(str(e), 500, e.code.value)
        except Exception as e:
            logger.exception("Query failed")
            self.send_error_response(f"Internal error: {e}", 500)
    
    def handle_rebuild_index(self) -> None:
        """Trigger index rebuild."""
        try:
            from src.build_index import build_and_save_index
            
            build_and_save_index()
            
            # Reinitialize assistant
            RAGRequestHandler.assistant = RAGAssistant()
            
            self.send_json_response({
                "status": "success",
                "message": "Index rebuilt successfully",
            })
        except Exception as e:
            logger.exception("Index rebuild failed")
            self.send_error_response(f"Rebuild failed: {e}", 500)


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Run the HTTP API server.
    
    Args:
        host: Host to bind to.
        port: Port to listen on.
    """
    config = get_config()
    project_root = get_project_root()
    
    # Configure logging
    configure_logging(
        log_dir=project_root / config.paths.logs_dir,
        log_level=config.logging.level,
        json_format=config.is_production,
    )
    
    logger.info(
        "Starting RAG Assistant server",
        host=host,
        port=port,
        environment=config.environment,
    )
    
    # Initialize assistant
    try:
        RAGRequestHandler.assistant = RAGAssistant()
    except Exception as e:
        logger.error(f"Failed to initialize assistant: {e}")
        logger.info("Server will start but queries will fail until index is built")
    
    # Start server
    server = HTTPServer((host, port), RAGRequestHandler)
    
    logger.info(f"Server listening on http://{host}:{port}")
    logger.info("Endpoints:")
    logger.info("  GET  /health  - Health check")
    logger.info("  GET  /metrics - Prometheus metrics")
    logger.info("  GET  /config  - Configuration")
    logger.info("  POST /query   - Query endpoint")
    logger.info("  POST /index/rebuild - Rebuild index")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Assistant API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port)

