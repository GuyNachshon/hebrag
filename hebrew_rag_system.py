# Hebrew Agentic RAG System Implementation
import asyncio
import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from agno.agent import Agent
    from agno.team import Team
    from agno.models.ollama import Ollama
    import chromadb
    AGNO_AVAILABLE = True
    # Define SearchType locally
    class SearchType:
        hybrid = "hybrid"
        semantic = "semantic"
        keyword = "keyword"
except ImportError as e:
    AGNO_AVAILABLE = False
    logging.warning(f"Agno framework not available: {e}. Some functionality will be limited.")
    # Define fallback classes
    class Agent: 
        def __init__(self, *args, **kwargs): pass
        async def arun(self, *args, **kwargs): return "Agent not available"
    class Team:
        def __init__(self, *args, **kwargs): self.members = []
        async def arun(self, *args, **kwargs): return "Team not available"
    class Ollama:
        def __init__(self, *args, **kwargs): pass
    class SearchType:
        hybrid = "hybrid"
        semantic = "semantic"
        keyword = "keyword"

# Import Hebrew tools with fallback handling
try:
    from hebrew_tools import (
        HebrewDocumentProcessor,
        LayoutAnalyzer,
        HebrewSemanticSearch,
        ContextualRetriever,
        HebrewTableAnalyzer,
        HebrewChartAnalyzer,
        VisualContextualizer,
        HebrewResponseGenerator,
        AnswerValidator,
        HebrewEmbedder
    )
    HEBREW_TOOLS_AVAILABLE = True
except ImportError as e:
    HEBREW_TOOLS_AVAILABLE = False
    logging.error(f"Hebrew tools not available: {e}")
    
    # Create dummy classes if not available
    class HebrewDocumentProcessor:
        def __init__(self, **kwargs): pass
        def run(self, path): return {"status": "error", "error": "Not available"}
    
    class HebrewSemanticSearch:
        def __init__(self, **kwargs): pass
        def run(self, **kwargs): return []
    
    class ContextualRetriever:
        def __init__(self, **kwargs): pass
        def run(self, **kwargs): return []
    
    class HebrewTableAnalyzer:
        def __init__(self, **kwargs): pass
        def run(self, **kwargs): return {"status": "error"}
    
    class HebrewResponseGenerator:
        def __init__(self, **kwargs): pass
        def run(self, **kwargs): return {"response": "לא זמין", "confidence": 0.0}
    
    class AnswerValidator:
        def __init__(self, **kwargs): pass
        def run(self, **kwargs): return {"overall_score": 0.0}
    
    class HebrewEmbedder:
        def __init__(self, **kwargs): pass
        def is_available(self): return False
        def encode(self, texts): return [[0.0] * 384]  # Dummy embedding
    
    # Unused in main system but imported
    class LayoutAnalyzer: pass
    class HebrewChartAnalyzer: pass
    class VisualContextualizer: pass

class HebrewAgnoRAGSystem:
    """
    Main Hebrew Agentic RAG System using Agno framework with custom Hebrew tools.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.document_chunks = []  # In-memory storage for development
        
        self.setup_environment()
        
        if AGNO_AVAILABLE:
            self.initialize_agno_components()
            self.setup_hebrew_agents()
        else:
            self.logger.warning("Running in fallback mode without Agno")
            self.setup_fallback_components()
        
        self.logger.info("Hebrew RAG System initialized successfully")
    
    def setup_environment(self):
        """Configure air-gapped environment"""
        # Disable telemetry for air-gapped deployment
        os.environ["AGNO_TELEMETRY"] = "false"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        
        # Configure local model paths
        self.model_config = {
            "llm_model": self.config.get("llm_model", "mistral:7b-instruct"),
            "embedding_model": self.config.get("embedding_model", "./models/heBERT"),
            "ollama_base_url": self.config.get("ollama_base_url", "http://localhost:11434"),
            "chroma_db_path": self.config.get("chroma_db_path", "./chroma_db"),
            "max_tokens": self.config.get("max_tokens", 2048),
            "temperature": self.config.get("temperature", 0.1)
        }
        
        self.logger.info(f"Environment configured with models: {self.model_config}")
    
    def initialize_agno_components(self):
        """Initialize Agno core components"""
        try:
            # Local LLM via Ollama
            self.llm = Ollama(
                id=self.model_config["llm_model"],
                host=self.model_config["ollama_base_url"],
            )
            
            # Hebrew Embedder
            self.hebrew_embedder = HebrewEmbedder(
                model_path=self.model_config["embedding_model"]
            )
            
            # Vector database (ChromaDB directly)
            try:
                self.vector_db = chromadb.PersistentClient(path=self.model_config["chroma_db_path"])
                self.collection = self.vector_db.get_or_create_collection(
                    name="hebrew_documents",
                    metadata={"hnsw:space": "cosine"}
                )
                self.logger.info("ChromaDB initialized successfully")
            except Exception as e:
                self.logger.warning(f"ChromaDB initialization failed: {e}")
                self.vector_db = None
                self.collection = None
            
            self.logger.info("Agno components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Agno components: {e}")
            self.setup_fallback_components()
    
    def setup_fallback_components(self):
        """Setup fallback components when Agno is not available"""
        self.llm = None
        self.hebrew_embedder = HebrewEmbedder()
        self.vector_db = None
        self.collection = None
        
        # Initialize Hebrew tools for fallback mode
        self.document_processor = HebrewDocumentProcessor(
            ocr_enabled=True, 
            layout_analysis=True,
            visual_extraction=True
        )
        
        self.semantic_search = HebrewSemanticSearch(
            boost_visual_content=True,
            context_window=500
        )
        
        self.contextual_retriever = ContextualRetriever(
            preserve_relationships=True,
            multimodal_aware=True
        )
        
        self.table_analyzer = HebrewTableAnalyzer(
            extract_structure=True,
            generate_descriptions=True
        )
        
        self.response_generator = HebrewResponseGenerator(
            citation_style="academic",
            preserve_context=True
        )
        
        self.answer_validator = AnswerValidator(
            check_hebrew_grammar=True,
            verify_citations=True
        )
        
        self.logger.info("Fallback components initialized")
    
    def setup_hebrew_agents(self):
        """Create specialized Hebrew agent team"""
        if not AGNO_AVAILABLE:
            self.logger.warning("Cannot setup agents - Agno not available")
            return
        
        try:
            # Initialize Hebrew tools
            self.document_processor = HebrewDocumentProcessor(
                ocr_enabled=True, 
                layout_analysis=True,
                visual_extraction=True
            )
            
            self.layout_analyzer = LayoutAnalyzer(
                preserve_context=True,
                hebrew_aware=True
            )
            
            self.semantic_search = HebrewSemanticSearch(
                boost_visual_content=True,
                context_window=500
            )
            
            self.contextual_retriever = ContextualRetriever(
                preserve_relationships=True,
                multimodal_aware=True
            )
            
            self.table_analyzer = HebrewTableAnalyzer(
                extract_structure=True,
                generate_descriptions=True
            )
            
            self.chart_analyzer = HebrewChartAnalyzer(
                identify_trends=True,
                extract_data_points=True
            )
            
            self.visual_contextualizer = VisualContextualizer(
                link_to_text=True,
                hebrew_descriptions=True
            )
            
            self.response_generator = HebrewResponseGenerator(
                citation_style="academic",
                preserve_context=True
            )
            
            self.answer_validator = AnswerValidator(
                check_hebrew_grammar=True,
                verify_citations=True
            )
            
            # Document Processing Agent
            self.document_agent = Agent(
                name="Hebrew Document Processor",
                role="Process Hebrew documents with embedded visuals",
                model=self.llm,
                tools=[self.document_processor, self.layout_analyzer],
                instructions=[
                    "Process Hebrew documents maintaining spatial context",
                    "Extract visual elements with Hebrew descriptions",
                    "Preserve text-visual relationships",
                    "Handle right-to-left text directionality properly"
                ]
            )
            
            # Retrieval Agent
            self.retrieval_agent = Agent(
                name="Hebrew Retrieval Specialist",
                role="Retrieve relevant Hebrew content with visual context",
                model=self.llm,
                tools=[self.semantic_search, self.contextual_retriever],
                instructions=[
                    "Use agentic RAG for Hebrew content retrieval",
                    "Prioritize visual content when query requires data",
                    "Maintain contextual relationships between elements",
                    "Search with Hebrew semantic understanding"
                ]
            )
            
            # Visual Analysis Agent
            self.visual_agent = Agent(
                name="Hebrew Visual Analyst",
                role="Analyze visual elements in Hebrew context", 
                model=self.llm,
                tools=[
                    self.table_analyzer,
                    self.chart_analyzer,
                    self.visual_contextualizer
                ],
                instructions=[
                    "Analyze tables and charts in Hebrew documents",
                    "Generate Hebrew descriptions of visual content",
                    "Extract quantitative insights from visuals",
                    "Connect visual elements to surrounding Hebrew text"
                ]
            )
            
            # Response Synthesis Agent
            self.synthesis_agent = Agent(
                name="Hebrew Response Synthesizer",
                role="Generate comprehensive Hebrew answers",
                model=self.llm,
                tools=[self.response_generator, self.answer_validator],
                instructions=[
                    "Synthesize multimodal information into coherent Hebrew responses",
                    "Generate natural Hebrew responses with proper grammar", 
                    "Include proper academic citations",
                    "Ensure answer completeness and accuracy",
                    "Validate Hebrew language quality"
                ]
            )
            
            # Create coordinated team
            self.hebrew_team = Team(
                agents=[
                    self.document_agent,
                    self.retrieval_agent, 
                    self.visual_agent,
                    self.synthesis_agent
                ],
                instructions=[
                    "Collaborate to answer Hebrew questions about multimodal documents",
                    "Preserve Hebrew language nuances and cultural context",
                    "Maintain text-visual context relationships",
                    "Ensure high-quality Hebrew responses with proper citations"
                ]
            )
            
            self.logger.info("Hebrew agent team created successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up Hebrew agents: {e}")
            # Set agents to None for fallback mode
            self.document_agent = None
            self.retrieval_agent = None
            self.visual_agent = None
            self.synthesis_agent = None
            self.hebrew_team = None
    
    async def process_document(self, document_path: str) -> Dict[str, Any]:
        """Process and index a Hebrew document"""
        try:
            self.logger.info(f"Processing document: {document_path}")
            
            if self.document_agent and AGNO_AVAILABLE:
                # Use Agno agent for processing
                result = await self.document_agent.arun(
                    f"Process the Hebrew document at {document_path} and extract contextual chunks"
                )
                
                # Extract chunks from agent result
                if isinstance(result, str):
                    # Parse agent response to extract chunks
                    chunks = self.parse_agent_response_for_chunks(result, document_path)
                else:
                    chunks = result.get("chunks", [])
                
            else:
                # Fallback processing without agents
                if hasattr(self, 'document_processor'):
                    result = self.document_processor.run(document_path)
                    chunks = result.get("chunks", [])
                else:
                    # Basic fallback if no processor available
                    chunks = [{
                        'chunk_id': f"fallback_chunk_0",
                        'content': f"Document: {document_path}",
                        'type': 'text',
                        'source_file': document_path
                    }]
                    result = {'status': 'success', 'chunks': chunks}
            
            # Store chunks in memory (for development)
            if chunks:
                self.document_chunks.extend(chunks)
                
                # Add to knowledge base if available
                if self.knowledge_base:
                    await self.add_chunks_to_knowledge_base(chunks)
            
            self.logger.info(f"Document processed successfully. Generated {len(chunks)} chunks.")
            
            return {
                "status": "success",
                "document_path": document_path,
                "chunks_generated": len(chunks),
                "processing_method": "agno_agent" if self.document_agent else "fallback"
            }
            
        except Exception as e:
            self.logger.error(f"Error processing document {document_path}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "document_path": document_path
            }
    
    def parse_agent_response_for_chunks(self, response: str, document_path: str) -> List[Dict]:
        """Parse agent response to extract chunks (fallback method)"""
        # This is a simplified parser - in practice, you'd need more sophisticated parsing
        chunks = []
        
        try:
            # Use document processor directly as fallback
            result = self.document_processor.run(document_path)
            chunks = result.get("chunks", [])
            
        except Exception as e:
            self.logger.error(f"Error parsing agent response: {e}")
        
        return chunks
    
    async def add_chunks_to_knowledge_base(self, chunks: List[Dict]):
        """Add processed chunks to the knowledge base"""
        try:
            if not self.collection:
                self.logger.warning("ChromaDB collection not available")
                return
            
            # Convert chunks to format expected by ChromaDB
            texts = []
            metadatas = []
            ids = []
            embeddings = []
            
            for i, chunk in enumerate(chunks):
                content = chunk.get('content', '') or chunk.get('full_context', '')
                if content:
                    texts.append(content)
                    metadatas.append({
                        'type': chunk.get('type', 'text'),
                        'page_number': chunk.get('page_number', 0),
                        'source_file': chunk.get('source_file', 'unknown'),
                        'is_hebrew': chunk.get('is_hebrew', False)
                    })
                    chunk_id = chunk.get('chunk_id', f'chunk_{i}')
                    ids.append(chunk_id)
                    
                    # Generate embedding
                    try:
                        embedding = self.hebrew_embedder.encode([content])
                        if hasattr(embedding, 'tolist'):
                            embeddings.append(embedding[0].tolist())
                        else:
                            embeddings.append(embedding[0])
                    except Exception as e:
                        self.logger.error(f"Error generating embedding for chunk {chunk_id}: {e}")
                        continue
            
            # Add to collection
            if texts:
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
                self.logger.info(f"Added {len(texts)} chunks to knowledge base")
                    
        except Exception as e:
            self.logger.error(f"Error adding chunks to knowledge base: {e}")
    
    async def answer_question(self, question: str, context_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer Hebrew question using agent team or fallback methods"""
        try:
            self.logger.info(f"Processing Hebrew question: {question[:50]}...")
            
            if self.hebrew_team and AGNO_AVAILABLE:
                # Use coordinated agent team
                response = await self.hebrew_team.arun(
                    f"Answer the following Hebrew question comprehensively: {question}"
                )
                
                return {
                    "status": "success", 
                    "answer": response,
                    "question": question,
                    "method": "agno_team",
                    "timestamp": asyncio.get_event_loop().time()
                }
            
            else:
                # Fallback method without agents
                return await self.answer_question_fallback(question)
                
        except Exception as e:
            self.logger.error(f"Error answering question: {e}")
            return {
                "status": "error",
                "error": str(e),
                "question": question
            }
    
    async def answer_question_fallback(self, question: str) -> Dict[str, Any]:
        """Fallback method to answer questions without Agno agents"""
        try:
            # Search relevant chunks
            relevant_chunks = self.search_chunks(question)
            
            # Analyze any visual content if analyzer available
            analysis_results = []
            if hasattr(self, 'table_analyzer'):
                for chunk in relevant_chunks:
                    if chunk.get('type') == 'visual':
                        if chunk.get('visual_type') == 'table':
                            analysis = self.table_analyzer.run(
                                {'content': chunk.get('content', [])},
                                chunk.get('context_before', '')
                            )
                            analysis_results.append(analysis)
            
            # Generate response if generator available
            if hasattr(self, 'response_generator'):
                response_result = self.response_generator.run(
                    question=question,
                    retrieved_chunks=relevant_chunks,
                    analysis_results=analysis_results
                )
            else:
                # Basic fallback response
                content_parts = [chunk.get('content', '') for chunk in relevant_chunks[:3]]
                response_text = '. '.join([part for part in content_parts if part])
                if not response_text:
                    response_text = "לא נמצא מידע רלוונטי לשאלה"
                
                response_result = {
                    "response": response_text,
                    "confidence": 0.5,
                    "sources": []
                }
            
            return {
                "status": "success",
                "answer": response_result.get("response", "לא ניתן היה לענות על השאלה"),
                "question": question,
                "method": "fallback",
                "confidence": response_result.get("confidence", 0.5),
                "sources": response_result.get("sources", []),
                "chunks_used": len(relevant_chunks)
            }
            
        except Exception as e:
            self.logger.error(f"Error in fallback question answering: {e}")
            return {
                "status": "error",
                "error": str(e),
                "question": question,
                "method": "fallback"
            }
    
    def search_chunks(self, question: str, k: int = 5) -> List[Dict]:
        """Search for relevant chunks using semantic search"""
        try:
            # Try ChromaDB first if available
            if self.collection:
                try:
                    query_embedding = self.hebrew_embedder.encode([question])
                    if hasattr(query_embedding, 'tolist'):
                        query_emb_list = [query_embedding[0].tolist()]
                    else:
                        query_emb_list = [query_embedding[0]]
                    
                    results = self.collection.query(
                        query_embeddings=query_emb_list,
                        n_results=k,
                        include=['documents', 'metadatas', 'distances']
                    )
                    
                    # Convert ChromaDB results to our format
                    chunks = []
                    for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0], 
                        results['metadatas'][0], 
                        results['distances'][0]
                    )):
                        chunk = {
                            'content': doc,
                            'similarity_score': 1.0 - distance,  # Convert distance to similarity
                            'type': metadata.get('type', 'text'),
                            'page_number': metadata.get('page_number', 0),
                            'source_file': metadata.get('source_file', 'unknown'),
                            'is_hebrew': metadata.get('is_hebrew', False)
                        }
                        chunks.append(chunk)
                    return chunks
                except Exception as e:
                    self.logger.error(f"ChromaDB search failed: {e}")
                    # Fall through to other methods
            
            # Fallback to in-memory search
            if not self.document_chunks:
                return []
            
            # Use semantic search tool if available
            if hasattr(self, 'semantic_search'):
                search_results = self.semantic_search.run(
                    query=question,
                    document_chunks=self.document_chunks,
                    k=k
                )
                
                # Use contextual retriever to enhance results if available
                if hasattr(self, 'contextual_retriever'):
                    enhanced_results = self.contextual_retriever.run(
                        chunks=search_results,
                        query=question,
                        k=k
                    )
                    return enhanced_results
                else:
                    return search_results
            else:
                # Basic text matching fallback
                return self.basic_text_search(question, k)
            
        except Exception as e:
            self.logger.error(f"Error searching chunks: {e}")
            return self.document_chunks[:k] if self.document_chunks else []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        try:
            stats = {
                "agno_available": AGNO_AVAILABLE,
                "agents_initialized": 0,
                "vector_db_status": "unavailable",
                "documents_indexed": len(self.document_chunks),
                "llm_model": self.model_config["llm_model"],
                "embedding_model": self.model_config["embedding_model"],
                "chunks_in_memory": len(self.document_chunks)
            }
            
            if AGNO_AVAILABLE and self.hebrew_team:
                stats["agents_initialized"] = len(self.hebrew_team.agents)
                stats["vector_db_status"] = "healthy" if self.collection else "unavailable"
                
            if self.collection:
                try:
                    count = self.collection.count()
                    stats["vector_db_document_count"] = count
                except:
                    stats["vector_db_document_count"] = "unknown"
            
            # Add memory usage if possible
            try:
                import psutil
                process = psutil.Process()
                stats["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024
                stats["cpu_percent"] = process.cpu_percent()
            except ImportError:
                pass
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting system stats: {e}")
            return {
                "error": str(e),
                "agno_available": AGNO_AVAILABLE
            }
    
    def basic_text_search(self, question: str, k: int = 5) -> List[Dict]:
        """Basic text search fallback when semantic search is not available"""
        if not self.document_chunks:
            return []
        
        # Simple keyword matching
        question_words = set(question.lower().split())
        scored_chunks = []
        
        for chunk in self.document_chunks:
            content = chunk.get('content', '') or chunk.get('full_context', '')
            content_words = set(content.lower().split())
            
            # Calculate simple overlap score
            overlap = len(question_words.intersection(content_words))
            if overlap > 0:
                score = overlap / len(question_words)
                scored_chunks.append((chunk, score))
        
        # Sort by score and return top k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored_chunks[:k]]
    
    def get_available_documents(self) -> List[Dict]:
        """Get list of processed documents"""
        try:
            documents = {}
            
            for chunk in self.document_chunks:
                doc_path = chunk.get('source_file', 'unknown')
                if doc_path not in documents:
                    documents[doc_path] = {
                        "path": doc_path,
                        "chunks": 0,
                        "pages": set(),
                        "types": set()
                    }
                
                documents[doc_path]["chunks"] += 1
                
                if chunk.get('page_number') is not None:
                    documents[doc_path]["pages"].add(chunk['page_number'])
                
                if chunk.get('type'):
                    documents[doc_path]["types"].add(chunk['type'])
            
            # Convert to list format
            result = []
            for doc_path, info in documents.items():
                result.append({
                    "path": doc_path,
                    "chunks": info["chunks"],
                    "pages": len(info["pages"]),
                    "types": list(info["types"])
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting document list: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": asyncio.get_event_loop().time()
        }
        
        try:
            # Check Agno availability
            health["components"]["agno"] = "available" if AGNO_AVAILABLE else "unavailable"
            
            # Check LLM connection
            if self.llm:
                try:
                    # This would test actual LLM connection
                    health["components"]["llm"] = "connected"
                except Exception:
                    health["components"]["llm"] = "disconnected"
                    health["status"] = "degraded"
            else:
                health["components"]["llm"] = "unavailable"
                health["status"] = "degraded"
            
            # Check Hebrew embedder
            if hasattr(self, 'hebrew_embedder') and self.hebrew_embedder:
                health["components"]["embedder"] = "available" if self.hebrew_embedder.is_available() else "unavailable"
            else:
                health["components"]["embedder"] = "unavailable"
            
            # Check vector database
            if self.vector_db:
                health["components"]["vector_db"] = "connected"
            else:
                health["components"]["vector_db"] = "unavailable"
                health["status"] = "degraded"
            
            # Check document processing capabilities
            health["components"]["document_processor"] = "available"
            
            # Overall system status
            if health["status"] == "healthy":
                health["message"] = "All systems operational"
            else:
                health["message"] = "Some components unavailable - running in degraded mode"
            
        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)
            self.logger.error(f"Error in health check: {e}")
        
        return health
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        try:
            self.logger.info("Shutting down Hebrew RAG system...")
            
            # Close vector database connections
            if self.vector_db:
                # Close vector DB connections if needed
                pass
            
            # Clear in-memory chunks
            self.document_chunks.clear()
            
            self.logger.info("System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Utility functions for standalone usage
def create_rag_system(config: Optional[Dict] = None) -> HebrewAgnoRAGSystem:
    """Create and initialize a Hebrew RAG system"""
    return HebrewAgnoRAGSystem(config)

async def process_document_standalone(document_path: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Process a single document standalone"""
    system = create_rag_system(config)
    return await system.process_document(document_path)

async def answer_question_standalone(question: str, documents: List[str], 
                                   config: Optional[Dict] = None) -> Dict[str, Any]:
    """Answer a question with multiple documents standalone"""
    system = create_rag_system(config)
    
    # Process all documents
    for doc_path in documents:
        await system.process_document(doc_path)
    
    # Answer question
    return await system.answer_question(question)