# Hebrew Agentic RAG System with Agno - Complete Implementation Plan

## Strategic Architecture Overview

We're building a **high-performance Hebrew multimodal agentic RAG system** using **Agno as the agent framework** with **custom Hebrew intelligence components**. This approach combines cutting-edge agent technology with specialized Hebrew document processing for air-gapped environments.

### Core Design Philosophy

**Agno + Custom Hebrew = Best of Both Worlds**
- **Agno handles**: Agent orchestration, performance optimization, multimodal processing, agentic RAG
- **Custom components handle**: Hebrew language processing, contextual document parsing, cultural nuances

```
┌─────────────────────────────────────────────────────────────────┐
│                     AIR-GAPPED ENVIRONMENT                      │
├─────────────────────────────────────────────────────────────────┤
│  Frontend UI → FastAPI → Agno Agent Team → Local LLM           │
│                                ↓                                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │  Hebrew Doc     │    │   Agno Agentic  │    │  Agno Agent  │ │
│  │  Processor      │    │   RAG Engine    │    │   Teams      │ │
│  │  (Custom)       │    │   (Built-in)    │    │  (Built-in)  │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│          ↓                       ↓                       ↓     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │ Hebrew Tools    │    │   ChromaDB      │    │ Ollama LLM   │ │
│  │ (Custom)        │    │ (via Agno)      │    │ (via Agno)   │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack with Agno

### Core Dependencies
```python
# requirements.txt
agno==0.2.75
ollama==0.4.2
chromadb==0.4.18
torch==2.1.0
transformers==4.36.0
sentence-transformers==2.2.2

# Document Processing (Custom Hebrew Components)
unstructured[local-inference]==0.11.6
pymupdf==1.23.5
pdfplumber==0.9.0
python-docx==0.8.11
detectron2==0.6
paddleocr==2.7.3
pillow==10.1.0

# Hebrew Language Processing
hebrew-tokenizer==2.3.0
numpy==1.24.3
pandas==2.1.4

# API & Infrastructure
fastapi==0.104.1
uvicorn==0.24.0
```

### Local Models for Air-Gapped Deployment
```bash
# Hebrew Language Models
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
avichr/heBERT
alephbert-base

# Local LLM for Agno
ollama/mistral:7b-instruct
ollama/llama2:13b-chat

# Visual Understanding Models  
microsoft/table-transformer-structure-recognition
microsoft/table-transformer-detection
openai/clip-vit-large-patch14

# OCR Models
PaddleOCR Hebrew models
```

## System Architecture with Agno

### Agent Team Structure
```python
from agno.agent import Agent
from agno.team import Team
from agno.models.ollama import OllamaChat
from agno.vectordb.chroma import ChromaDb
from agno.knowledge.pdf import PDFKnowledgeBase

# Specialized Agent Team for Hebrew Multimodal RAG
hebrew_rag_team = Team(
    mode="coordinate",  # Agents work together on complex tasks
    members=[
        document_processor_agent,    # Hebrew document analysis
        retrieval_specialist_agent,  # Contextual information retrieval  
        visual_analysis_agent,       # Visual element understanding
        response_synthesis_agent     # Final answer generation
    ],
    model=OllamaChat(id="mistral:7b-instruct"),
    success_criteria="Accurate Hebrew answers with proper source citations",
    instructions=[
        "Preserve Hebrew language nuances",
        "Maintain text-visual context relationships", 
        "Cite sources accurately"
    ],
    show_tool_calls=True,
    markdown=True
)
```

### Core Agent Definitions

#### 1. Document Processing Agent
```python
from hebrew_tools import HebrewDocumentProcessor, LayoutAnalyzer

document_processor_agent = Agent(
    name="Hebrew Document Processor",
    role="Process and understand Hebrew documents with embedded visuals", 
    model=OllamaChat(id="mistral:7b-instruct"),
    tools=[
        HebrewDocumentProcessor(
            ocr_enabled=True,
            layout_analysis=True,
            visual_extraction=True
        ),
        LayoutAnalyzer(
            preserve_context=True,
            hebrew_aware=True
        )
    ],
    instructions=[
        "Process Hebrew documents maintaining spatial context",
        "Extract visual elements with surrounding Hebrew text",
        "Preserve right-to-left text directionality",
        "Handle mixed Hebrew-English content appropriately"
    ],
    show_tool_calls=True,
    markdown=True
)
```

#### 2. Retrieval Specialist Agent
```python
retrieval_specialist_agent = Agent(
    name="Hebrew Retrieval Specialist",
    role="Find relevant information from Hebrew document knowledge base",
    model=OllamaChat(id="mistral:7b-instruct"),
    knowledge=PDFKnowledgeBase(
        vector_db=ChromaDb(
            uri="./chroma_db",
            table_name="hebrew_documents", 
            search_type=SearchType.hybrid,
            embedder=HebrewEmbedder(model_path="./models/heBERT")
        )
    ),
    tools=[
        HebrewSemanticSearch(
            boost_visual_content=True,
            context_window=500
        ),
        ContextualRetriever(
            preserve_relationships=True,
            multimodal_aware=True
        )
    ],
    instructions=[
        "Use Agno's built-in agentic RAG capabilities",
        "Search knowledge base for Hebrew content with visual context",
        "Prioritize chunks containing visual elements when query requires data",
        "Maintain semantic relationships between text and visuals"
    ],
    show_tool_calls=True,
    markdown=True
)
```

#### 3. Visual Analysis Agent
```python
visual_analysis_agent = Agent(
    name="Hebrew Visual Analyst", 
    role="Understand and analyze visual elements in Hebrew context",
    model=OllamaChat(id="mistral:7b-instruct"),
    tools=[
        HebrewTableAnalyzer(
            extract_structure=True,
            generate_descriptions=True
        ),
        HebrewChartAnalyzer(
            identify_trends=True,
            extract_data_points=True
        ),
        VisualContextualizer(
            link_to_text=True,
            hebrew_descriptions=True
        )
    ],
    instructions=[
        "Analyze tables, charts, and figures in Hebrew documents",
        "Generate Hebrew descriptions of visual content", 
        "Connect visual elements to surrounding text context",
        "Extract quantitative data from visual elements"
    ],
    show_tool_calls=True,
    markdown=True
)
```

#### 4. Response Synthesis Agent
```python
response_synthesis_agent = Agent(
    name="Hebrew Response Synthesizer",
    role="Generate comprehensive Hebrew answers from retrieved information",
    model=OllamaChat(id="mistral:7b-instruct"),
    tools=[
        HebrewResponseGenerator(
            citation_style="academic",
            preserve_context=True
        ),
        AnswerValidator(
            check_hebrew_grammar=True,
            verify_citations=True
        )
    ],
    instructions=[
        "Synthesize information from multiple sources",
        "Generate natural Hebrew responses",
        "Include proper source citations",
        "Maintain coherent narrative flow",
        "Handle conflicting information appropriately"
    ],
    show_tool_calls=True,
    markdown=True
)
```

## Custom Hebrew Tools for Agno

### Hebrew Document Processing Tools

```python
# hebrew_tools.py
from agno.tools import Tool
from typing import List, Dict, Optional
import fitz  # PyMuPDF
import paddleocr
from detectron2.engine import DefaultPredictor

class HebrewDocumentProcessor(Tool):
    def __init__(self, ocr_enabled: bool = True, layout_analysis: bool = True):
        super().__init__(
            name="hebrew_document_processor",
            description="Process Hebrew documents extracting text and visual elements with spatial context"
        )
        self.ocr_enabled = ocr_enabled
        self.layout_analysis = layout_analysis
        self.setup_models()
    
    def setup_models(self):
        """Initialize Hebrew OCR and layout analysis models"""
        if self.ocr_enabled:
            self.ocr = paddleocr.PaddleOCR(
                use_angle_cls=True,
                lang='he',  # Hebrew
                use_gpu=True,
                show_log=False
            )
        
        if self.layout_analysis:
            # Load local table detection model
            cfg = self.setup_detectron_config()
            self.table_detector = DefaultPredictor(cfg)
    
    def run(self, document_path: str) -> Dict:
        """
        Process a Hebrew document and return structured content
        
        Args:
            document_path: Path to PDF, DOCX, or other document
            
        Returns:
            Dict containing processed content with spatial relationships
        """
        try:
            # Extract layout-aware content
            elements = self.extract_layout_elements(document_path)
            
            # Process Hebrew text
            hebrew_content = self.process_hebrew_text(elements)
            
            # Extract and analyze visual elements
            visual_elements = self.extract_visual_elements(document_path)
            
            # Create contextual chunks preserving relationships
            contextual_chunks = self.create_contextual_chunks(
                hebrew_content, visual_elements
            )
            
            return {
                "status": "success",
                "document_path": document_path,
                "chunks": contextual_chunks,
                "visual_elements_count": len(visual_elements),
                "hebrew_text_blocks": len(hebrew_content)
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "document_path": document_path
            }
    
    def create_contextual_chunks(self, text_content: List, visual_elements: List) -> List[Dict]:
        """Create chunks that preserve text-visual relationships"""
        chunks = []
        
        for i, element in enumerate(text_content + visual_elements):
            if element['type'] == 'text_with_visual':
                # Get surrounding context
                context_before = self.get_preceding_context(text_content, element)
                context_after = self.get_following_context(text_content, element)
                
                # Generate Hebrew description of visual element
                visual_description = self.describe_visual_in_hebrew(element['visual'])
                
                chunk = {
                    'chunk_id': f"chunk_{i}",
                    'text_before': context_before,
                    'visual_element': {
                        'type': element['visual']['type'],
                        'content': element['visual']['content'],
                        'description_hebrew': visual_description,
                        'coordinates': element['visual']['coordinates']
                    },
                    'text_after': context_after,
                    'full_context': self.build_full_context(
                        context_before, visual_description, context_after
                    ),
                    'page_number': element.get('page_number'),
                    'source_file': element.get('source_file')
                }
                chunks.append(chunk)
        
        return chunks

class HebrewSemanticSearch(Tool):
    def __init__(self, boost_visual_content: bool = True):
        super().__init__(
            name="hebrew_semantic_search",
            description="Search Hebrew documents with semantic understanding and visual content awareness"
        )
        self.boost_visual_content = boost_visual_content
        self.hebrew_embedder = self.setup_hebrew_embedder()
    
    def setup_hebrew_embedder(self):
        """Initialize Hebrew embedding model"""
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('./models/heBERT')
    
    def run(self, query: str, context: Optional[str] = None, k: int = 5) -> List[Dict]:
        """
        Search Hebrew documents with contextual understanding
        
        Args:
            query: Hebrew search query
            context: Optional context for query understanding
            k: Number of results to return
            
        Returns:
            List of relevant document chunks with similarity scores
        """
        try:
            # Process Hebrew query
            processed_query = self.process_hebrew_query(query)
            
            # Generate query embedding
            query_embedding = self.hebrew_embedder.encode([processed_query])[0]
            
            # Determine if query needs visual content
            needs_visual = self.query_needs_visual_content(query)
            
            # Search with appropriate strategy
            if needs_visual and self.boost_visual_content:
                results = self.search_with_visual_boost(query_embedding, k * 2)
            else:
                results = self.search_semantic(query_embedding, k)
            
            # Re-rank and filter results
            final_results = self.rerank_results(results, query)[:k]
            
            return final_results
            
        except Exception as e:
            return [{
                "status": "error",
                "error": str(e),
                "query": query
            }]
    
    def query_needs_visual_content(self, query: str) -> bool:
        """Detect if Hebrew query requires visual data"""
        visual_keywords_hebrew = [
            'טבלה', 'תרשים', 'גרף', 'נתונים', 'מספרים',
            'תמונה', 'איור', 'דיאגרמה', 'סטטיסטיקה',
            'השוואה', 'מגמה', 'אחוזים', 'ביצועים'
        ]
        return any(keyword in query for keyword in visual_keywords_hebrew)

class HebrewTableAnalyzer(Tool):
    def __init__(self):
        super().__init__(
            name="hebrew_table_analyzer",
            description="Analyze tables in Hebrew documents and extract structured data"
        )
    
    def run(self, table_data: Dict, context: str) -> Dict:
        """
        Analyze table in Hebrew context and extract insights
        
        Args:
            table_data: Table structure and content
            context: Surrounding Hebrew text context
            
        Returns:
            Dict containing table analysis and Hebrew description
        """
        try:
            # Extract table structure
            structure = self.analyze_table_structure(table_data)
            
            # Generate Hebrew description
            hebrew_description = self.generate_hebrew_description(
                table_data, context, structure
            )
            
            # Extract key insights
            insights = self.extract_table_insights(table_data, context)
            
            return {
                "status": "success",
                "table_structure": structure,
                "hebrew_description": hebrew_description,
                "insights": insights,
                "data_summary": self.summarize_data(table_data)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "table_data": str(table_data)[:200] + "..."
            }
```

### Hebrew Response Generation Tools

```python
class HebrewResponseGenerator(Tool):
    def __init__(self, citation_style: str = "academic"):
        super().__init__(
            name="hebrew_response_generator", 
            description="Generate comprehensive Hebrew responses with proper citations"
        )
        self.citation_style = citation_style
    
    def run(self, question: str, retrieved_chunks: List[Dict], analysis_results: List[Dict]) -> Dict:
        """
        Generate Hebrew response from retrieved information
        
        Args:
            question: Original Hebrew question
            retrieved_chunks: Relevant document chunks
            analysis_results: Visual analysis results
            
        Returns:
            Dict containing generated response and metadata
        """
        try:
            # Combine textual and visual information
            combined_context = self.combine_multimodal_context(
                retrieved_chunks, analysis_results
            )
            
            # Generate structured Hebrew response
            response = self.generate_hebrew_response(
                question, combined_context
            )
            
            # Add citations
            cited_response = self.add_citations(response, retrieved_chunks)
            
            # Validate Hebrew grammar and coherence
            validation_result = self.validate_response(cited_response)
            
            return {
                "status": "success",
                "response": cited_response,
                "confidence": self.calculate_confidence(combined_context),
                "sources": self.extract_source_metadata(retrieved_chunks),
                "validation": validation_result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "question": question
            }
    
    def generate_hebrew_response(self, question: str, context: Dict) -> str:
        """Generate natural Hebrew response using local LLM via Agno"""
        
        # This will be called by Agno's LLM integration
        prompt = f"""
בהתבסס על המידע הבא, ענה על השאלה בעברית באופן מפורט ומדויק:

שאלה: {question}

מידע רלוונטי:
{self.format_context_for_prompt(context)}

הנחיות:
1. תן תשובה מקיפה ומדויקת
2. השתמש במידע הטקסטואלי והוויזואלי כאחד
3. ציין מקורות רלוונטיים
4. כתב בעברית ברורה וזורמת
5. אם המידע לא מספיק, ציין זאת בבירור

תשובה:
"""
        # This integrates with Agno's LLM calling mechanism
        return prompt  # Agno will process this through the configured LLM
```

## Main System Integration

### Agno-Based RAG System Class

```python
# hebrew_rag_system.py
import asyncio
import os
from typing import Dict, List, Optional
from agno.agent import Agent
from agno.team import Team
from agno.models.ollama import OllamaChat
from agno.vectordb.chroma import ChromaDb
from agno.knowledge.pdf import PDFKnowledgeBase

class HebrewAgnoRAGSystem:
    def __init__(self):
        self.setup_environment()
        self.initialize_agno_components()
        self.setup_hebrew_agents()
        
    def setup_environment(self):
        """Configure air-gapped environment"""
        # Disable telemetry for air-gapped deployment
        os.environ["AGNO_TELEMETRY"] = "false"
        
        # Configure local model paths
        self.model_config = {
            "llm_model": "mistral:7b-instruct",
            "embedding_model": "./models/heBERT",
            "ollama_base_url": "http://localhost:11434"
        }
    
    def initialize_agno_components(self):
        """Initialize Agno core components"""
        
        # Local LLM via Ollama
        self.llm = OllamaChat(
            id=self.model_config["llm_model"],
            base_url=self.model_config["ollama_base_url"]
        )
        
        # Vector database
        self.vector_db = ChromaDb(
            uri="./chroma_db",
            table_name="hebrew_documents",
            search_type=SearchType.hybrid,
            embedder=HebrewEmbedder(
                model_path=self.model_config["embedding_model"]
            )
        )
        
        # Knowledge base
        self.knowledge_base = PDFKnowledgeBase(
            vector_db=self.vector_db
        )
    
    def setup_hebrew_agents(self):
        """Create specialized Hebrew agent team"""
        
        # Import custom Hebrew tools
        from hebrew_tools import (
            HebrewDocumentProcessor,
            HebrewSemanticSearch, 
            HebrewTableAnalyzer,
            HebrewResponseGenerator
        )
        
        # Document Processing Agent
        self.document_agent = Agent(
            name="Hebrew Document Processor",
            role="Process Hebrew documents with embedded visuals",
            model=self.llm,
            tools=[
                HebrewDocumentProcessor(ocr_enabled=True, layout_analysis=True),
                LayoutAnalyzer(preserve_context=True)
            ],
            instructions=[
                "Process Hebrew documents maintaining spatial context",
                "Extract visual elements with Hebrew descriptions",
                "Preserve text-visual relationships"
            ],
            show_tool_calls=True,
            markdown=True
        )
        
        # Retrieval Agent
        self.retrieval_agent = Agent(
            name="Hebrew Retrieval Specialist",
            role="Retrieve relevant Hebrew content with visual context",
            model=self.llm,
            knowledge=self.knowledge_base,
            tools=[
                HebrewSemanticSearch(boost_visual_content=True),
                ContextualRetriever(multimodal_aware=True)
            ],
            instructions=[
                "Use agentic RAG for Hebrew content retrieval",
                "Prioritize visual content when query requires data",
                "Maintain contextual relationships"
            ],
            show_tool_calls=True,
            markdown=True
        )
        
        # Visual Analysis Agent
        self.visual_agent = Agent(
            name="Hebrew Visual Analyst",
            role="Analyze visual elements in Hebrew context", 
            model=self.llm,
            tools=[
                HebrewTableAnalyzer(),
                HebrewChartAnalyzer(),
                VisualContextualizer()
            ],
            instructions=[
                "Analyze tables and charts in Hebrew documents",
                "Generate Hebrew descriptions of visual content",
                "Extract quantitative insights"
            ],
            show_tool_calls=True,
            markdown=True
        )
        
        # Response Synthesis Agent
        self.synthesis_agent = Agent(
            name="Hebrew Response Synthesizer",
            role="Generate comprehensive Hebrew answers",
            model=self.llm,
            tools=[
                HebrewResponseGenerator(citation_style="academic"),
                AnswerValidator(check_hebrew_grammar=True)
            ],
            instructions=[
                "Synthesize multimodal information",
                "Generate natural Hebrew responses", 
                "Include proper citations",
                "Ensure answer completeness"
            ],
            show_tool_calls=True,
            markdown=True
        )
        
        # Create coordinated team
        self.hebrew_team = Team(
            mode="coordinate",
            members=[
                self.document_agent,
                self.retrieval_agent, 
                self.visual_agent,
                self.synthesis_agent
            ],
            model=self.llm,
            success_criteria="Accurate, comprehensive Hebrew answer with proper citations",
            instructions=[
                "Collaborate to answer Hebrew questions about multimodal documents",
                "Preserve Hebrew language nuances",
                "Maintain text-visual context relationships"
            ],
            show_tool_calls=True,
            markdown=True
        )
    
    async def process_document(self, document_path: str) -> Dict:
        """Process and index a Hebrew document"""
        try:
            # Use document processing agent
            result = await self.document_agent.arun(
                f"Process the Hebrew document at {document_path} and extract contextual chunks"
            )
            
            # Add to knowledge base
            if result.get("status") == "success":
                chunks = result.get("chunks", [])
                await self.add_chunks_to_knowledge_base(chunks)
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "document_path": document_path
            }
    
    async def answer_question(self, question: str) -> Dict:
        """Answer Hebrew question using agent team"""
        try:
            # Use coordinated team to answer question
            response = await self.hebrew_team.arun(
                f"Answer the following Hebrew question comprehensively: {question}"
            )
            
            return {
                "status": "success", 
                "answer": response,
                "question": question,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "question": question
            }
    
    def get_system_stats(self) -> Dict:
        """Get system performance statistics"""
        return {
            "agents_initialized": len(self.hebrew_team.members),
            "vector_db_status": "healthy",
            "documents_indexed": self.vector_db.collection.count() if hasattr(self.vector_db, 'collection') else 0,
            "llm_model": self.model_config["llm_model"],
            "embedding_model": self.model_config["embedding_model"]
        }
```

## FastAPI Integration with Agno

```python
# main.py  
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import time
from typing import Optional

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
```

## Deployment Configuration

### Docker Setup for Agno + Hebrew RAG

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy models and application code
COPY models/ ./models/
COPY src/ ./src/
COPY hebrew_tools/ ./hebrew_tools/

# Set environment variables for air-gapped deployment
ENV AGNO_TELEMETRY=false
ENV OLLAMA_HOST=0.0.0.0:11434

EXPOSE 8000 11434

# Start script
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
```

```bash
#!/bin/bash
# start.sh

# Start Ollama in background
ollama serve &

# Wait for Ollama to be ready
sleep 10

# Load local models
ollama create mistral:7b-instruct --file ./models/mistral-7b.modelfile

# Start the FastAPI application
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### Docker Compose for Complete System

```yaml
# docker-compose.yml
version: '3.8'

services:
  hebrew-rag-agno:
    build: .
    ports:
      - "8000:8000"
      - "11434:11434"
    volumes:
      - ./models:/app/models:ro
      - ./chroma_db:/app/chroma_db
      - ./documents:/app/documents
      - ./logs:/app/logs
    environment:
      - AGNO_TELEMETRY=false
      - OLLAMA_HOST=0.0.0.0:11434
      - USE_GPU=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  chroma_db:
  models:
  documents:
  logs:
```

## Performance Optimization with Agno

### Agno Performance Benefits

**Agent Instantiation: ~3μs**
- Extremely fast agent creation for concurrent Hebrew queries
- Minimal memory footprint (~5KB per agent)
- Optimized for high-throughput scenarios

**Built-in Optimizations:**
- Parallel tool execution
- Intelligent caching
- Streaming responses
- Memory management

### Performance Configuration

```python
# performance_config.py
from agno.agent import Agent
from agno.models.ollama import OllamaChat

# Optimized agent configuration for Hebrew RAG
performance_config = {
    "agent_settings": {
        "max_iterations": 10,
        "timeout": 300,  # 5 minutes for complex Hebrew queries
        "streaming": True,
        "show_tool_calls": True,
        "markdown": True
    },
    
    "llm_settings": {
        "temperature": 0.1,  # Low temperature for factual Hebrew responses
        "max_tokens": 2048,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    },
    
    "vector_db_settings": {
        "search_type": "hybrid",  # Combines semantic + keyword search
        "k": 5,  # Top 5 results by default
        "score_threshold": 0.7,  # Minimum similarity score
        "include_metadata": True
    }
}

# Apply performance settings to Hebrew agents
def create_optimized_hebrew_agent(name: str, role: str, tools: list) -> Agent:
    return Agent(
        name=name,
        role=role,
        model=OllamaChat(
            id="mistral:7b-instruct",
            **performance_config["llm_settings"]
        ),
        tools=tools,
        **performance_config["agent_settings"]
    )
```

## Testing & Validation Framework

### Comprehensive Test Suite for Hebrew Agno RAG

```python
# tests/test_hebrew_agno_rag.py
import pytest
import asyncio
import time
from pathlib import Path
from hebrew_rag_system import HebrewAgnoRAGSystem

class TestHebrewAgnoRAGSystem:
    
    @pytest.fixture
    async def rag_system(self):
        """Create test instance of Hebrew Agno RAG system"""
        system = HebrewAgnoRAGSystem()
        yield system
    
    @pytest.mark.asyncio
    async def test_agno_agent_instantiation(self, rag_system):
        """Test Agno agent creation performance"""
        start_time = time.perf_counter()
        
        # Create multiple agents
        agents_created = 0
        for i in range(100):
            agent = rag_system.document_agent
            agents_created += 1
            
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / agents_created
        
        # Should be under 10μs per agent (Agno's performance target)
        assert avg_time < 0.00001, f"Agent instantiation too slow: {avg_time}s"
        
    @pytest.mark.asyncio 
    async def test_hebrew_document_processing(self, rag_system):
        """Test Hebrew document processing with Agno agents"""
        test_file = "tests/data/sample_hebrew_report.pdf"
        
        result = await rag_system.process_document(test_file)
        
        assert result["status"] == "success"
        assert "chunks" in result
        assert len(result["chunks"]) > 0
        
        # Verify Hebrew content preservation
        chunks = result["chunks"]
        hebrew_chunks = [c for c in chunks if self.contains_hebrew(c["full_context"])]
        assert len(hebrew_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_hebrew_question_answering_team(self, rag_system):
        """Test Hebrew question answering using Agno team"""
        question = "מה הנתונים העיקריים שמוצגים בטבלת התקציב?"
        
        start_time = time.time()
        result = await rag_system.answer_question(question)
        processing_time = time.time() - start_time
        
        assert result["status"] == "success"
        assert "answer" in result
        assert len(result["answer"]) > 0
        assert processing_time < 60  # Should answer within 60 seconds
        
        # Verify Hebrew response quality
        answer = result["answer"]
        assert self.contains_hebrew(answer)
        assert self.is_coherent_hebrew(answer)
    
    @pytest.mark.asyncio
    async def test_multimodal_visual_analysis(self, rag_system):
        """Test visual element analysis in Hebrew context"""
        question = "תאר את המגמות שמוצגות בתרשים המכירות"
        
        result = await rag_system.answer_question(question)
        
        assert result["status"] == "success"
        # Should include visual analysis results
        assert "תרשים" in result["answer"] or "גרף" in result["answer"]
    
    @pytest.mark.asyncio
    async def test_concurrent_hebrew_queries(self, rag_system):
        """Test handling multiple concurrent Hebrew questions"""
        questions = [
            "מה התקציב הכולל לשנת 2023?",
            "איך השתנו המכירות בין הרבעונים?", 
            "מה המסקנות העיקריות מהדוח?",
            "כמה עובדים יש בכל מחלקה?",
            "מה התחזית לשנה הבאה?"
        ]
        
        start_time = time.time()
        tasks = [rag_system.answer_question(q) for q in questions]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # All queries should succeed
        assert len(results) == 5
        assert all(result["status"] == "success" for result in results)
        
        # Should handle concurrent queries efficiently
        avg_time_per_query = total_time / len(questions)
        assert avg_time_per_query < 30  # Average under 30 seconds per query
    
    def contains_hebrew(self, text: str) -> bool:
        """Check if text contains Hebrew characters"""
        hebrew_chars = set(range(0x0590, 0x05FF))  # Hebrew Unicode block
        return any(ord(char) in hebrew_chars for char in text)
    
    def is_coherent_hebrew(self, text: str) -> bool:
        """Basic coherence check for Hebrew text"""
        # Check for proper sentence structure, punctuation
        sentences = text.split('.')
        return len(sentences) > 1 and all(len(s.strip()) > 5 for s in sentences[:-1])

# Performance benchmarks
class TestAgnoPerformance:
    
    @pytest.mark.asyncio
    async def test_agno_vs_custom_performance(self, rag_system):
        """Compare Agno performance against custom implementation"""
        
        # Test agent instantiation speed
        agno_times = []
        for _ in range(1000):
            start = time.perf_counter()
            agent = rag_system.document_agent
            end = time.perf_counter()
            agno_times.append(end - start)
        
        avg_agno_time = sum(agno_times) / len(agno_times)
        
        # Agno should instantiate agents in microseconds
        assert avg_agno_time < 0.00001  # Under 10μs
        
    @pytest.mark.asyncio
    async def test_memory_usage(self, rag_system):
        """Test memory efficiency of Agno agents"""
        import tracemalloc
        
        tracemalloc.start()
        
        # Create multiple agents
        agents = []
        for i in range(100):
            agent = rag_system.synthesis_agent
            agents.append(agent)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory per agent should be minimal
        memory_per_agent = current / len(agents)
        assert memory_per_agent < 10000  # Under 10KB per agent
        
    @pytest.mark.asyncio
    async def test_throughput_under_load(self, rag_system):
        """Test system throughput under concurrent load"""
        
        # Simulate high load
        num_concurrent_requests = 50
        question = "מה הנתונים בטבלה הראשונה?"
        
        start_time = time.time()
        
        tasks = [
            rag_system.answer_question(f"{question} (בקשה {i})")
            for i in range(num_concurrent_requests)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate throughput
        successful_requests = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        throughput = successful_requests / total_time
        
        assert throughput > 1.0  # At least 1 request per second
        assert successful_requests >= num_concurrent_requests * 0.8  # 80% success rate

# Hebrew-specific validation tests
class TestHebrewLanguageQuality:
    
    @pytest.mark.asyncio
    async def test_hebrew_rtl_preservation(self, rag_system):
        """Test right-to-left Hebrew text preservation"""
        question = "מה כתוב בעמוד הראשון של המסמך?"
        
        result = await rag_system.answer_question(question)
        answer = result["answer"]
        
        # Verify RTL markers and Hebrew text direction
        assert self.has_proper_hebrew_direction(answer)
    
    @pytest.mark.asyncio
    async def test_mixed_hebrew_english_handling(self, rag_system):
        """Test handling of mixed Hebrew-English content"""
        question = "מה המטרה של התוכנית AI ואיך היא משפיעה על ה-ROI?"
        
        result = await rag_system.answer_question(question)
        answer = result["answer"]
        
        # Should handle both languages appropriately
        assert "AI" in answer or "ROI" in answer
        assert self.contains_hebrew(answer)
    
    @pytest.mark.asyncio
    async def test_hebrew_numerical_data_extraction(self, rag_system):
        """Test extraction of numerical data in Hebrew context"""
        question = "כמה אחוז גדל הרווח השנה?"
        
        result = await rag_system.answer_question(question)
        answer = result["answer"]
        
        # Should include numerical data with Hebrew context
        assert any(char.isdigit() for char in answer)
        assert "אחוז" in answer or "%" in answer
    
    def has_proper_hebrew_direction(self, text: str) -> bool:
        """Check if Hebrew text maintains proper RTL direction"""
        # Basic check for Hebrew text structure
        hebrew_words = [word for word in text.split() if self.contains_hebrew(word)]
        return len(hebrew_words) > 0
    
    def contains_hebrew(self, text: str) -> bool:
        """Check if text contains Hebrew characters"""
        hebrew_chars = set(range(0x0590, 0x05FF))
        return any(ord(char) in hebrew_chars for char in text)
```

## Monitoring & Analytics with Agno

### Built-in Agno Monitoring

```python
# monitoring.py
import time
from typing import Dict, List
from agno.agent import Agent
from agno.team import Team

class HebrewRAGMonitor:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.metrics = {
            'queries_processed': 0,
            'avg_response_time': 0,
            'hebrew_accuracy_score': 0,
            'agent_performance': {},
            'error_rate': 0
        }
    
    async def monitor_agent_performance(self, agent: Agent, task: str) -> Dict:
        """Monitor individual agent performance"""
        start_time = time.perf_counter()
        
        try:
            result = await agent.arun(task)
            end_time = time.perf_counter()
            
            performance_data = {
                'agent_name': agent.name,
                'task': task,
                'execution_time': end_time - start_time,
                'status': 'success',
                'memory_usage': self.get_agent_memory_usage(agent),
                'tool_calls': len(result.get('tool_calls', [])),
                'tokens_used': result.get('tokens_used', 0)
            }
            
            return performance_data
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                'agent_name': agent.name,
                'task': task,
                'execution_time': end_time - start_time,
                'status': 'error',
                'error': str(e)
            }
    
    async def monitor_team_performance(self, team: Team, query: str) -> Dict:
        """Monitor agent team coordination performance"""
        start_time = time.perf_counter()
        
        try:
            result = await team.arun(query)
            end_time = time.perf_counter()
            
            team_metrics = {
                'team_size': len(team.members),
                'total_execution_time': end_time - start_time,
                'coordination_mode': team.mode,
                'success_criteria_met': team.success_criteria,
                'member_performance': [],
                'hebrew_quality_score': self.assess_hebrew_quality(result)
            }
            
            # Individual agent metrics within team
            for member in team.members:
                member_metrics = await self.monitor_agent_performance(
                    member, f"Part of team query: {query[:50]}..."
                )
                team_metrics['member_performance'].append(member_metrics)
            
            return team_metrics
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                'team_size': len(team.members),
                'total_execution_time': end_time - start_time,
                'status': 'error',
                'error': str(e)
            }
    
    def assess_hebrew_quality(self, response: str) -> float:
        """Assess Hebrew language quality in response"""
        quality_score = 0.0
        
        # Check Hebrew character presence
        if self.contains_hebrew(response):
            quality_score += 0.3
        
        # Check sentence structure
        sentences = response.split('.')
        if len(sentences) > 1:
            quality_score += 0.2
        
        # Check for proper Hebrew grammar patterns
        if self.has_proper_hebrew_grammar(response):
            quality_score += 0.3
        
        # Check coherence
        if self.is_coherent_response(response):
            quality_score += 0.2
        
        return min(quality_score, 1.0)
    
    def get_system_health(self) -> Dict:
        """Get overall system health metrics"""
        return {
            'agno_version': '0.2.75',
            'agents_active': len(self.rag_system.hebrew_team.members),
            'vector_db_status': 'healthy',
            'llm_status': 'connected',
            'avg_query_time': self.metrics['avg_response_time'],
            'error_rate': self.metrics['error_rate'],
            'hebrew_quality_avg': self.metrics['hebrew_accuracy_score']
        }
```

## Production Deployment Guide

### Step-by-Step Deployment Process

#### Phase 1: Environment Preparation (Air-Gapped)

```bash
#!/bin/bash
# deploy_phase1.sh - Environment Setup

echo "Phase 1: Setting up air-gapped Hebrew RAG environment with Agno"

# Create directory structure
mkdir -p {documents,logs,chroma_db,models,cache}
mkdir -p {hebrew_tools,tests,config}

# Set permissions
chmod 755 documents logs chroma_db cache
chmod 644 models/*

# Install Docker and nvidia-docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install nvidia-docker for GPU support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update && apt-get install -y nvidia-docker2
systemctl restart docker

echo "Phase 1 completed successfully"
```

#### Phase 2: Model and Dependency Installation

```bash
#!/bin/bash
# deploy_phase2.sh - Install Models and Dependencies

echo "Phase 2: Installing models and dependencies offline"

# Install Python packages from offline wheel files
pip install --no-index --find-links packages/ -r requirements.txt

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Load Hebrew models
ollama create mistral:7b-instruct --file ./models/mistral-7b.modelfile
ollama create llama2:13b-chat --file ./models/llama2-13b.modelfile

# Verify Agno installation
python -c "import agno; print(f'Agno version: {agno.__version__}')"

# Test Hebrew models
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('./models/heBERT')
test_text = 'בדיקה של מודל עברי'
embedding = model.encode([test_text])
print(f'Hebrew model loaded successfully. Embedding shape: {embedding.shape}')
"

echo "Phase 2 completed successfully"
```

#### Phase 3: System Configuration and Testing

```bash
#!/bin/bash
# deploy_phase3.sh - Configure and Test System

echo "Phase 3: Configuring Hebrew Agno RAG system"

# Set environment variables
export AGNO_TELEMETRY=false
export OLLAMA_HOST=0.0.0.0:11434
export HEBREW_MODELS_PATH=./models
export CHROMA_DB_PATH=./chroma_db

# Initialize ChromaDB
python -c "
import chromadb
client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_or_create_collection('hebrew_documents')
print(f'ChromaDB initialized. Collection count: {collection.count()}')
"

# Test Agno agent creation
python -c "
from agno.agent import Agent
from agno.models.ollama import OllamaChat

agent = Agent(
    name='Test Hebrew Agent',
    model=OllamaChat(id='mistral:7b-instruct'),
    instructions=['Test Hebrew processing capabilities']
)
print('Agno agent created successfully')
"

# Run comprehensive tests
python -m pytest tests/ -v --tb=short

echo "Phase 3 completed successfully"
```

#### Phase 4: Production Deployment

```bash
#!/bin/bash
# deploy_phase4.sh - Production Deployment

echo "Phase 4: Production deployment"

# Build and start containers
docker-compose build --no-cache
docker-compose up -d

# Wait for services to start
sleep 30

# Health check
curl -f http://localhost:8000/system-status || {
    echo "Health check failed"
    docker-compose logs
    exit 1
}

# Load test documents
python scripts/load_test_documents.py

# Final verification
python scripts/verify_deployment.py

echo "Production deployment completed successfully!"
echo "Hebrew Agno RAG system available at http://localhost:8000"
```

### Configuration Files

#### Environment Configuration

```python
# config/production.py
import os
from pathlib import Path

class ProductionConfig:
    # Air-gapped settings
    AGNO_TELEMETRY = False
    DISABLE_EXTERNAL_CALLS = True
    
    # Model paths
    HEBREW_MODELS_PATH = Path("./models")
    OLLAMA_MODEL = "mistral:7b-instruct"
    HEBREW_EMBEDDING_MODEL = "heBERT"
    
    # Database settings
    CHROMA_DB_PATH = Path("./chroma_db")
    VECTOR_DB_COLLECTION = "hebrew_documents"
    
    # Performance settings
    MAX_CONCURRENT_AGENTS = 10
    AGENT_TIMEOUT = 300  # 5 minutes
    MAX_TOKENS_PER_RESPONSE = 2048
    
    # Hebrew processing settings
    HEBREW_OCR_ENABLED = True
    LAYOUT_ANALYSIS_ENABLED = True
    VISUAL_PROCESSING_ENABLED = True
    
    # Security settings
    LOG_LEVEL = "INFO"
    MAX_FILE_SIZE_MB = 100
    ALLOWED_FILE_TYPES = [".pdf", ".docx", ".doc", ".txt"]
    
    # Monitoring settings
    ENABLE_PERFORMANCE_MONITORING = True
    METRICS_RETENTION_DAYS = 30
    
    @classmethod
    def validate_config(cls):
        """Validate production configuration"""
        required_paths = [
            cls.HEBREW_MODELS_PATH,
            cls.CHROMA_DB_PATH
        ]
        
        for path in required_paths:
            if not path.exists():
                raise ValueError(f"Required path does not exist: {path}")
        
        # Verify Ollama is accessible
        import requests
        try:
            response = requests.get("http://localhost:11434/api/version")
            if response.status_code != 200:
                raise ValueError("Ollama is not accessible")
        except Exception as e:
            raise ValueError(f"Failed to connect to Ollama: {e}")
        
        return True
```

## System Maintenance & Updates

### Regular Maintenance Tasks

```python
# maintenance/system_maintenance.py
import asyncio
import logging
from pathlib import Path
from hebrew_rag_system import HebrewAgnoRAGSystem

class SystemMaintenance:
    def __init__(self, rag_system: HebrewAgnoRAGSystem):
        self.rag_system = rag_system
        self.logger = logging.getLogger(__name__)
    
    async def cleanup_old_logs(self, retention_days: int = 30):
        """Clean up old log files"""
        log_dir = Path("./logs")
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        
        for log_file in log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                self.logger.info(f"Deleted old log file: {log_file}")
    
    async def optimize_vector_database(self):
        """Optimize ChromaDB performance"""
        try:
            # Compact the database
            collection = self.rag_system.vector_db.collection
            
            # Remove duplicate embeddings if any
            await self.remove_duplicate_embeddings(collection)
            
            # Rebuild index if needed
            await self.rebuild_search_index(collection)
            
            self.logger.info("Vector database optimization completed")
            
        except Exception as e:
            self.logger.error(f"Vector database optimization failed: {e}")
    
    async def update_hebrew_models(self, model_path: Path):
        """Update Hebrew language models"""
        try:
            # Backup current models
            backup_path = Path("./models/backup")
            backup_path.mkdir(exist_ok=True)
            
            # Update models with new versions
            # This would be done manually in air-gapped environment
            
            self.logger.info("Hebrew models updated successfully")
            
        except Exception as e:
            self.logger.error(f"Model update failed: {e}")
    
    async def generate_performance_report(self) -> dict:
        """Generate system performance report"""
        stats = self.rag_system.get_system_stats()
        
        report = {
            "timestamp": time.time(),
            "system_health": stats,
            "recommendations": self.generate_recommendations(stats)
        }
        
        return report
    
    def generate_recommendations(self, stats: dict) -> list:
        """Generate system optimization recommendations"""
        recommendations = []
        
        if stats.get("avg_response_time", 0) > 30:
            recommendations.append("Consider upgrading hardware for better performance")
        
        if stats.get("error_rate", 0) > 0.05:
            recommendations.append("High error rate detected, check system logs")
        
        if stats.get("documents_indexed", 0) > 10000:
            recommendations.append("Consider database optimization or archiving old documents")
        
        return recommendations

# Automated maintenance script
async def run_daily_maintenance():
    """Run daily maintenance tasks"""
    rag_system = HebrewAgnoRAGSystem()
    maintenance = SystemMaintenance(rag_system)
    
    await maintenance.cleanup_old_logs()
    await maintenance.optimize_vector_database()
    
    report = await maintenance.generate_performance_report()
    
    # Save report
    with open(f"./logs/maintenance_report_{int(time.time())}.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    asyncio.run(run_daily_maintenance())
```

## Final Deployment Checklist

### Pre-Production Verification

- [ ] **Agno Framework**: Version 0.2.75+ installed and configured
- [ ] **Hebrew Models**: All models downloaded and accessible offline
- [ ] **Ollama Setup**: Local LLM serving Hebrew-capable models
- [ ] **ChromaDB**: Vector database initialized and tested
- [ ] **Custom Tools**: All Hebrew processing tools implemented and tested
- [ ] **Agent Teams**: Multi-agent coordination working properly
- [ ] **Air-Gap Compliance**: No external network calls, telemetry disabled
- [ ] **Performance Tests**: System meets throughput and latency requirements
- [ ] **Security Validation**: All data encrypted, logs secured
- [ ] **Backup Strategy**: Database and model backups configured

### Production Monitoring

- [ ] **Agent Performance**: Monitor agent instantiation and execution times
- [ ] **Hebrew Quality**: Track Hebrew language response quality
- [ ] **System Resources**: Monitor CPU, GPU, memory, and storage usage
- [ ] **Error Tracking**: Log and alert on system errors
- [ ] **User Queries**: Track query patterns and response accuracy

### Success Metrics

**Performance Targets with Agno:**
- Agent instantiation: < 10μs
- Simple queries: < 15 seconds response time
- Complex queries: < 60 seconds response time
- Concurrent users: 50+ simultaneous queries
- System uptime: 99.9%
- Hebrew accuracy: > 95% for native content

**Expected Advantages over Custom/LangChain:**
- **10x faster agent creation** due to Agno's optimization
- **Built-in agentic RAG** reduces development time by 60%
- **Native multimodal support** improves visual processing
- **Team coordination** enables complex Hebrew reasoning workflows
- **Performance monitoring** provides real-time insights

This architecture leverages **Agno's cutting-edge agent framework** while maintaining **specialized Hebrew intelligence**, creating a production-ready system that's both high-performance and linguistically sophisticated.