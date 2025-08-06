# Hebrew Agentic RAG System with Agno - Main FastAPI Application
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import asyncio
import time
import logging
import uuid
from typing import Optional, List, Dict, Any
from pathlib import Path
import aiofiles

from hebrew_rag_system import HebrewAgnoRAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Hebrew Agentic RAG with Agno",
    description="High-performance Hebrew multimodal agentic RAG system using Agno framework",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance
hebrew_rag: Optional[HebrewAgnoRAGSystem] = None

# Pydantic models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="Hebrew question to answer")
    context_id: Optional[str] = Field(None, description="Optional context identifier")
    include_sources: bool = Field(True, description="Include source citations in response")
    max_chunks: int = Field(5, description="Maximum number of chunks to retrieve", ge=1, le=20)

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="Generated Hebrew answer")
    sources: List[Dict] = Field(default_factory=list, description="Source documents and citations")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    processing_time: float = Field(..., description="Processing time in seconds")
    agent_steps: List[Dict] = Field(default_factory=list, description="Agent processing steps")
    method: str = Field(..., description="Processing method used")
    word_count: int = Field(..., description="Number of words in answer")

class DocumentUploadResponse(BaseModel):
    status: str = Field(..., description="Upload status")
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    chunks_generated: int = Field(..., description="Number of chunks generated")
    processing_time: float = Field(..., description="Processing time in seconds")
    processing_method: str = Field(..., description="Method used for processing")

class SystemStatus(BaseModel):
    status: str = Field(..., description="Overall system status")
    agno_version: str = Field(..., description="Agno framework version")
    components: Dict[str, str] = Field(..., description="Component status")
    statistics: Dict[str, Any] = Field(..., description="System statistics")
    uptime: float = Field(..., description="System uptime in seconds")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    components: Dict[str, str] = Field(..., description="Component health")
    timestamp: float = Field(..., description="Health check timestamp")
    message: str = Field(..., description="Health status message")

class DocumentListResponse(BaseModel):
    documents: List[Dict] = Field(..., description="List of processed documents")
    total_documents: int = Field(..., description="Total number of documents")
    total_chunks: int = Field(..., description="Total number of chunks")

# System startup and shutdown
@app.on_event("startup")
async def startup_event():
    """Initialize the Hebrew RAG system on startup"""
    global hebrew_rag
    try:
        logger.info("Starting Hebrew Agentic RAG System...")
        
        # Ensure required directories exist first
        Path("./documents").mkdir(exist_ok=True)
        Path("./chroma_db").mkdir(exist_ok=True)
        Path("./logs").mkdir(exist_ok=True)
        Path("./models").mkdir(exist_ok=True)
        
        # Load configuration (could be from file or environment)
        config = {
            "llm_model": "mistral:7b-instruct",
            "embedding_model": "./models/heBERT",
            "ollama_base_url": "http://localhost:11434",
            "chroma_db_path": "./chroma_db",
            "max_tokens": 2048,
            "temperature": 0.1
        }
        
        # Initialize system with error handling
        try:
            hebrew_rag = HebrewAgnoRAGSystem(config)
            logger.info("Hebrew RAG System started successfully")
        except Exception as init_error:
            logger.warning(f"Failed to initialize with full configuration: {init_error}")
            # Try with minimal configuration
            minimal_config = {
                "llm_model": "mistral:7b-instruct",
                "embedding_model": "./models/multilingual-miniLM",  # Fallback model
                "ollama_base_url": "http://localhost:11434",
                "chroma_db_path": "./chroma_db",
                "max_tokens": 1024,
                "temperature": 0.1
            }
            hebrew_rag = HebrewAgnoRAGSystem(minimal_config)
            logger.info("Hebrew RAG System started with minimal configuration")
        
    except Exception as e:
        logger.error(f"Failed to start Hebrew RAG System: {e}")
        # Don't raise - allow the app to start in degraded mode
        hebrew_rag = None

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully shutdown the system"""
    global hebrew_rag
    if hebrew_rag:
        try:
            await hebrew_rag.shutdown()
            logger.info("Hebrew RAG System shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Dependency to get system instance
async def get_rag_system() -> Optional[HebrewAgnoRAGSystem]:
    """Dependency to get the RAG system instance"""
    if hebrew_rag is None:
        raise HTTPException(
            status_code=503, 
            detail="Hebrew RAG System not initialized. Please check system configuration and restart."
        )
    return hebrew_rag

# API Endpoints

@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    rag_system: Optional[HebrewAgnoRAGSystem] = Depends(get_rag_system)
):
    """Upload and process Hebrew document"""
    start_time = time.time()
    
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Save file
        file_path = Path("./documents") / f"{document_id}_{file.filename}"
        
        async with aiofiles.open(file_path, "wb") as buffer:
            content = await file.read()
            await buffer.write(content)
        
        logger.info(f"File uploaded: {file.filename} -> {file_path}")
        
        # Process document
        result = await rag_system.process_document(str(file_path))
        
        processing_time = time.time() - start_time
        
        if result.get("status") == "success":
            return DocumentUploadResponse(
                status="success",
                document_id=document_id,
                filename=file.filename,
                chunks_generated=result.get("chunks_generated", 0),
                processing_time=processing_time,
                processing_method=result.get("processing_method", "unknown")
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Document processing failed: {result.get('error', 'Unknown error')}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/ask-question", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest,
    rag_system: Optional[HebrewAgnoRAGSystem] = Depends(get_rag_system)
):
    """Process Hebrew question using Agno agent team"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing question: {request.question[:50]}...")
        
        # Answer question using RAG system
        result = await rag_system.answer_question(
            question=request.question,
            context_id=request.context_id
        )
        
        processing_time = time.time() - start_time
        
        if result.get("status") == "success":
            return AnswerResponse(
                answer=result.get("answer", ""),
                sources=result.get("sources", []) if request.include_sources else [],
                confidence=result.get("confidence", 0.0),
                processing_time=processing_time,
                agent_steps=result.get("agent_steps", []),
                method=result.get("method", "unknown"),
                word_count=len(result.get("answer", "").split())
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Question processing failed: {result.get('error', 'Unknown error')}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")

@app.get("/system-status", response_model=SystemStatus)
async def get_system_status():
    """Get Agno system health and performance metrics"""
    try:
        if hebrew_rag is None:
            return SystemStatus(
                status="error",
                agno_version="unknown",
                components={"system": "not_initialized"},
                statistics={"error": "System not initialized"},
                uptime=time.time() - app.extra.get("start_time", time.time())
            )
        
        stats = hebrew_rag.get_system_stats()
        health = await hebrew_rag.health_check()
        
        # Try to get Agno version dynamically
        agno_version = "0.2.75"  # Default
        try:
            import agno
            agno_version = getattr(agno, '__version__', '0.2.75')
        except ImportError:
            pass
        
        return SystemStatus(
            status=health.get("status", "unknown"),
            agno_version=agno_version,
            components=health.get("components", {}),
            statistics=stats,
            uptime=time.time() - app.extra.get("start_time", time.time())
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return SystemStatus(
            status="error",
            agno_version="unknown",
            components={"system": "error"},
            statistics={"error": str(e)},
            uptime=time.time() - app.extra.get("start_time", time.time())
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    try:
        if hebrew_rag is None:
            return HealthResponse(
                status="error",
                components={"system": "not_initialized"},
                timestamp=time.time(),
                message="Hebrew RAG System not initialized"
            )
        
        health = await hebrew_rag.health_check()
        
        return HealthResponse(
            status=health.get("status", "unknown"),
            components=health.get("components", {}),
            timestamp=health.get("timestamp", time.time()),
            message=health.get("message", "Health check completed")
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return HealthResponse(
            status="error",
            components={"system": "error"},
            timestamp=time.time(),
            message=f"Health check failed: {str(e)}"
        )

@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(rag_system: Optional[HebrewAgnoRAGSystem] = Depends(get_rag_system)):
    """Get list of processed documents"""
    try:
        documents = rag_system.get_available_documents()
        total_chunks = sum(doc.get("chunks", 0) for doc in documents)
        
        return DocumentListResponse(
            documents=documents,
            total_documents=len(documents),
            total_chunks=total_chunks
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    rag_system: Optional[HebrewAgnoRAGSystem] = Depends(get_rag_system)
):
    """Delete a processed document"""
    try:
        # Find and delete document file
        documents_dir = Path("./documents")
        matching_files = list(documents_dir.glob(f"{document_id}_*"))
        
        if not matching_files:
            raise HTTPException(status_code=404, detail="Document not found")
        
        for file_path in matching_files:
            file_path.unlink()
            logger.info(f"Deleted document file: {file_path}")
        
        # TODO: Remove from vector database and memory
        # This would require additional implementation in the RAG system
        
        return {"status": "success", "message": f"Document {document_id} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.post("/reindex")
async def reindex_documents(
    background_tasks: BackgroundTasks,
    rag_system: Optional[HebrewAgnoRAGSystem] = Depends(get_rag_system)
):
    """Reindex all documents in the system"""
    try:
        documents_dir = Path("./documents")
        document_files = list(documents_dir.glob("*"))
        
        if not document_files:
            return {"status": "success", "message": "No documents to reindex"}
        
        # Start reindexing in background
        background_tasks.add_task(reindex_documents_task, rag_system, document_files)
        
        return {
            "status": "started",
            "message": f"Reindexing {len(document_files)} documents in background"
        }
        
    except Exception as e:
        logger.error(f"Error starting reindex: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start reindex: {str(e)}")

async def reindex_documents_task(rag_system: Optional[HebrewAgnoRAGSystem], document_files: List[Path]):
    """Background task to reindex documents"""
    try:
        logger.info(f"Starting reindex of {len(document_files)} documents")
        
        for file_path in document_files:
            try:
                await rag_system.process_document(str(file_path))
                logger.info(f"Reindexed: {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to reindex {file_path.name}: {e}")
        
        logger.info("Document reindexing completed")
        
    except Exception as e:
        logger.error(f"Error in reindex task: {e}")

# Additional utility endpoints

@app.get("/")
async def root():
    """Root endpoint with system information"""
    system_status = "running" if hebrew_rag is not None else "degraded"
    
    return {
        "service": "Hebrew Agentic RAG with Agno",
        "version": "1.0.0",
        "status": system_status,
        "docs": "/docs",
        "health": "/health",
        "system_status": "/system-status",
        "api_endpoints": {
            "upload_document": "/upload-document",
            "ask_question": "/ask-question",
            "list_documents": "/documents",
            "test_hebrew": "/test-hebrew"
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Get system metrics for monitoring"""
    try:
        if hebrew_rag is None:
            return {
                "hebrew_rag_documents_total": 0,
                "hebrew_rag_chunks_total": 0,
                "hebrew_rag_agents_active": 0,
                "hebrew_rag_memory_usage_mb": 0,
                "hebrew_rag_cpu_percent": 0,
                "hebrew_rag_system_status": "not_initialized"
            }
        
        stats = hebrew_rag.get_system_stats()
        
        # Format metrics for monitoring systems (Prometheus style)
        metrics = {
            "hebrew_rag_documents_total": stats.get("documents_indexed", 0),
            "hebrew_rag_chunks_total": stats.get("chunks_in_memory", 0),
            "hebrew_rag_agents_active": stats.get("agents_initialized", 0),
            "hebrew_rag_memory_usage_mb": stats.get("memory_usage_mb", 0),
            "hebrew_rag_cpu_percent": stats.get("cpu_percent", 0),
            "hebrew_rag_system_status": "running",
            "hebrew_rag_agno_available": stats.get("agno_available", False)
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {
            "hebrew_rag_system_status": "error",
            "error": str(e)
        }

@app.post("/test-hebrew")
async def test_hebrew_processing():
    """Test endpoint for Hebrew text processing"""
    test_text = "זהו טקסט בדיקה בעברית למערכת עיבוד השפה העברית"
    
    return {
        "original": test_text,
        "contains_hebrew": any(0x0590 <= ord(c) <= 0x05FF for c in test_text),
        "word_count": len(test_text.split()),
        "character_count": len(test_text),
        "rtl_direction": True
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# Store start time for uptime calculation
if not hasattr(app, 'extra'):
    app.extra = {}
app.extra["start_time"] = time.time()

if __name__ == "__main__":
    import uvicorn
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hebrew Agentic RAG System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    args = parser.parse_args()
    
    # Configure logging level
    import logging
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)
    
    try:
        # Configure uvicorn
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            access_log=True,
            reload=args.reload
        )
    except KeyboardInterrupt:
        logger.info("Shutting down Hebrew RAG System...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)