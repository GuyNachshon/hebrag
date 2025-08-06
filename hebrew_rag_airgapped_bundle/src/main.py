# Hebrew Agentic RAG System with Agno
# Main FastAPI application

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import time
from typing import Optional, List, Dict

from hebrew_rag_system import HebrewAgnoRAGSystem

app = FastAPI(title="Hebrew Agentic RAG with Agno")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    context_id: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict]
    confidence: float
    processing_time: float
    agent_steps: List[Dict]

# Initialize Agno-based system
hebrew_rag = HebrewAgnoRAGSystem()

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process Hebrew document"""
    try:
        # Save file
        file_path = f"./documents/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process with Agno agent
        result = await hebrew_rag.process_document(file_path)
        
        return {
            "status": "success",
            "document_id": file.filename,
            "processing_result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-question", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Process Hebrew question using Agno agent team"""
    try:
        start_time = time.time()
        
        # Use Agno team to answer question
        result = await hebrew_rag.answer_question(request.question)
        
        processing_time = time.time() - start_time
        
        return AnswerResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0),
            processing_time=processing_time,
            agent_steps=result.get("agent_steps", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system-status")
async def get_system_status():
    """Get Agno system health and performance metrics"""
    return {
        "status": "healthy",
        "agno_version": "0.2.75",
        **hebrew_rag.get_system_stats()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
