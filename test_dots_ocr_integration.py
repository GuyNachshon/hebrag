#!/usr/bin/env python3
"""
Test script for dots.ocr integration with Hebrew RAG system.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from hebrew_tools.document_processor import (
        EnhancedHebrewDocumentProcessor, 
        DotsOCRProcessor,
        DOTS_OCR_AVAILABLE
    )
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)
    # Create mock classes for testing
    DOTS_OCR_AVAILABLE = False
    
    class DotsOCRProcessor:
        def __init__(self, *args, **kwargs):
            self.available = False
    
    class EnhancedHebrewDocumentProcessor:
        def __init__(self, *args, **kwargs):
            self.use_dots_ocr = False
            self.ocr_enabled = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dots_ocr_availability():
    """Test if dots.ocr dependencies are available"""
    print("🔍 Testing dots.ocr availability...")
    print(f"DOTS_OCR_AVAILABLE: {DOTS_OCR_AVAILABLE}")
    
    if DOTS_OCR_AVAILABLE:
        print("✅ dots.ocr dependencies are available")
        try:
            processor = DotsOCRProcessor()
            print(f"✅ DotsOCRProcessor initialized successfully")
            print(f"   - Model path: {processor.model_path}")
            print(f"   - Available: {processor.available}")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize DotsOCRProcessor: {e}")
            return False
    else:
        print("⚠️  dots.ocr dependencies not available")
        return False

def test_enhanced_processor():
    """Test the enhanced Hebrew document processor"""
    print("\n🔍 Testing EnhancedHebrewDocumentProcessor...")
    
    try:
        # Initialize processor with dots.ocr enabled
        processor = EnhancedHebrewDocumentProcessor(
            use_dots_ocr=True,
            ocr_enabled=True,
            layout_analysis=True,
            visual_extraction=True
        )
        
        print(f"✅ EnhancedHebrewDocumentProcessor initialized")
        print(f"   - dots.ocr enabled: {processor.use_dots_ocr}")
        print(f"   - PaddleOCR fallback: {processor.ocr_enabled}")
        print(f"   - Layout analysis: {processor.layout_analysis}")
        
        return processor
    except Exception as e:
        print(f"❌ Failed to initialize EnhancedHebrewDocumentProcessor: {e}")
        return None

def test_document_processing(processor, test_file=None):
    """Test document processing with a sample file"""
    if not processor:
        print("❌ No processor available for testing")
        return False
    
    print(f"\n🔍 Testing document processing...")
    
    # Look for test documents
    test_paths = [
        Path("test_documents/sample.pdf"),
        Path("documents/sample.pdf"),
        Path("sample.pdf")
    ]
    
    if test_file:
        test_paths.insert(0, Path(test_file))
    
    test_doc = None
    for path in test_paths:
        if path.exists():
            test_doc = path
            break
    
    if not test_doc:
        print("⚠️  No test document found. Creating a mock test...")
        print("   To test with a real document, place a PDF file at one of these paths:")
        for path in test_paths:
            print(f"     - {path}")
        
        # Test with non-existent file to verify error handling
        result = processor.run("non_existent_file.pdf")
        if result.get("status") == "error":
            print("✅ Error handling works correctly for missing files")
            return True
        else:
            print("❌ Error handling failed")
            return False
    
    print(f"📄 Processing test document: {test_doc}")
    
    try:
        result = processor.run(str(test_doc))
        
        if result.get("status") == "success":
            print("✅ Document processing successful!")
            print(f"   - Processor used: {result.get('processor_used', 'fallback')}")
            print(f"   - Total chunks: {result.get('total_chunks', 0)}")
            print(f"   - Pages processed: {result.get('pages', 'unknown')}")
            
            # Show sample chunks
            chunks = result.get("chunks", [])
            if chunks:
                print(f"\n📋 Sample chunks (showing first 3):")
                for i, chunk in enumerate(chunks[:3]):
                    print(f"   Chunk {i+1}:")
                    print(f"     - ID: {chunk.get('chunk_id', 'N/A')}")
                    print(f"     - Type: {chunk.get('type', 'N/A')}")
                    print(f"     - Hebrew: {chunk.get('is_hebrew', False)}")
                    print(f"     - Content preview: {chunk.get('content', '')[:100]}...")
            
            return True
        else:
            print(f"❌ Document processing failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during document processing: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting dots.ocr integration tests...\n")
    
    # First check if we can import the modules
    if not IMPORT_SUCCESS:
        print(f"❌ Failed to import required modules: {IMPORT_ERROR}")
        print("   This is likely due to missing dependencies.")
        print("   To fix this:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Ensure all packages are available")
        print("   3. Run this test again")
        return
    
    print("✅ Successfully imported integration modules")
    
    # Test 1: Check availability
    dots_available = test_dots_ocr_availability()
    
    # Test 2: Initialize processor
    processor = test_enhanced_processor()
    
    # Test 3: Process document
    processing_success = test_document_processing(processor)
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"   - Module imports: {'✅' if IMPORT_SUCCESS else '❌'}")
    print(f"   - dots.ocr available: {'✅' if dots_available else '❌'}")
    print(f"   - Processor initialization: {'✅' if processor else '❌'}")
    print(f"   - Document processing: {'✅' if processing_success else '❌'}")
    
    if IMPORT_SUCCESS:
        if dots_available and processor and processing_success:
            print(f"\n🎉 All tests passed! Integration is working correctly.")
        elif processor and processing_success:
            print(f"\n✅ Integration working correctly with fallback mode.")
            print("   Note: dots.ocr not available, using PaddleOCR fallback")
        else:
            print(f"\n⚠️  Some tests failed. Check the logs above for details.")
    
    if not dots_available and IMPORT_SUCCESS:
        print("\n   To enable dots.ocr:")
        print("   1. Install dependencies: pip install torch==2.7.0 transformers accelerate")
        print("   2. Ensure PyTorch 2.7.0+ is installed")
        print("   3. Run this test again")

if __name__ == "__main__":
    main()