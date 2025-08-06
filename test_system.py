#!/usr/bin/env python3
"""
Simple test script for Hebrew Agentic RAG System
Tests basic functionality without requiring full Agno setup
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        from hebrew_rag_system import HebrewAgnoRAGSystem
        logger.info("✅ HebrewAgnoRAGSystem imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import HebrewAgnoRAGSystem: {e}")
        return False
    
    try:
        from hebrew_tools import (
            HebrewDocumentProcessor,
            HebrewSemanticSearch,
            HebrewTableAnalyzer,
            HebrewResponseGenerator,
            HebrewEmbedder
        )
        logger.info("✅ Hebrew tools imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import hebrew_tools: {e}")
        return False
    
    try:
        import main
        logger.info("✅ Main FastAPI app imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import main app: {e}")
        return False
    
    return True

def test_hebrew_embedder():
    """Test Hebrew embedder functionality"""
    logger.info("Testing Hebrew embedder...")
    
    try:
        from hebrew_tools import HebrewEmbedder
        
        embedder = HebrewEmbedder()
        
        # Test Hebrew text
        hebrew_text = "זהו טקסט בדיקה בעברית למערכת עיבוד השפה העברית"
        embedding = embedder.encode([hebrew_text])
        
        logger.info(f"✅ Hebrew embedder working. Embedding shape: {embedding.shape}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Hebrew embedder test failed: {e}")
        return False

def test_document_processor():
    """Test document processor with a simple text file"""
    logger.info("Testing document processor...")
    
    try:
        from hebrew_tools import HebrewDocumentProcessor
        
        # Create a simple test file
        test_file = Path("./test_document.txt")
        test_content = """
שלום עולם!
זהו מסמך בדיקה בעברית.

טבלת נתונים דוגמא:
שם | גיל | עיר
יוסי | 25 | תל אביב
שרה | 30 | ירושלים

סיום המסמך.
"""
        
        test_file.write_text(test_content, encoding='utf-8')
        
        processor = HebrewDocumentProcessor(
            ocr_enabled=False,  # Skip OCR for text file
            layout_analysis=False,  # Skip complex layout
            visual_extraction=False
        )
        
        result = processor.run(str(test_file))
        
        # Clean up
        test_file.unlink()
        
        if result.get("status") == "success":
            chunks = result.get("chunks", [])
            logger.info(f"✅ Document processor working. Generated {len(chunks)} chunks")
            return True
        else:
            logger.error(f"❌ Document processing failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Document processor test failed: {e}")
        return False

async def test_rag_system():
    """Test the main RAG system"""
    logger.info("Testing RAG system...")
    
    try:
        from hebrew_rag_system import HebrewAgnoRAGSystem
        
        # Test with minimal configuration
        config = {
            "llm_model": "mistral:7b-instruct",
            "embedding_model": "./models/multilingual-miniLM",
            "ollama_base_url": "http://localhost:11434",
            "chroma_db_path": "./test_chroma_db",
            "max_tokens": 1024,
            "temperature": 0.1
        }
        
        # Create directories
        Path("./test_chroma_db").mkdir(exist_ok=True)
        Path("./models").mkdir(exist_ok=True)
        
        system = HebrewAgnoRAGSystem(config)
        
        # Test health check
        health = await system.health_check()
        logger.info(f"System health: {health.get('status', 'unknown')}")
        
        # Test system stats
        stats = system.get_system_stats()
        logger.info(f"System stats: Agno available: {stats.get('agno_available', False)}")
        
        logger.info("✅ RAG system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG system test failed: {e}")
        return False

def test_hebrew_tools():
    """Test Hebrew-specific tools"""
    logger.info("Testing Hebrew tools...")
    
    try:
        from hebrew_tools import (
            HebrewSemanticSearch, 
            HebrewTableAnalyzer,
            HebrewResponseGenerator
        )
        
        # Test semantic search
        search = HebrewSemanticSearch()
        test_chunks = [{
            'chunk_id': 'test_1',
            'content': 'זהו טקסט עברי לבדיקה',
            'type': 'text',
            'is_hebrew': True
        }]
        
        results = search.run("בדיקה", document_chunks=test_chunks, k=1)
        logger.info(f"✅ Semantic search returned {len(results)} results")
        
        # Test table analyzer
        analyzer = HebrewTableAnalyzer()
        test_table = {
            'content': [
                ['שם', 'גיל', 'עיר'],
                ['יוסי', '25', 'תל אביב'],
                ['שרה', '30', 'ירושלים']
            ]
        }
        
        table_result = analyzer.run(test_table, "טבלת נתונים")
        if table_result.get("status") == "success":
            logger.info("✅ Table analyzer working")
        else:
            logger.error(f"❌ Table analyzer failed: {table_result.get('error')}")
        
        # Test response generator
        generator = HebrewResponseGenerator()
        test_response = generator.run(
            "מה האוכלוסייה?",
            test_chunks,
            []
        )
        
        if test_response.get("status") == "success":
            logger.info("✅ Response generator working")
        else:
            logger.error(f"❌ Response generator failed: {test_response.get('error')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hebrew tools test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting Hebrew Agentic RAG System Tests")
    
    test_results = []
    
    # Run tests
    test_results.append(("Imports", test_imports()))
    test_results.append(("Hebrew Embedder", test_hebrew_embedder()))
    test_results.append(("Document Processor", test_document_processor()))
    test_results.append(("RAG System", await test_rag_system()))
    test_results.append(("Hebrew Tools", test_hebrew_tools()))
    
    # Report results
    logger.info("=" * 50)
    logger.info("Test Results Summary:")
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 50)
    logger.info(f"Tests passed: {passed}/{len(test_results)}")
    
    if passed == len(test_results):
        logger.info("🎉 All tests passed! System is ready to use.")
        return 0
    else:
        logger.info("⚠️ Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)