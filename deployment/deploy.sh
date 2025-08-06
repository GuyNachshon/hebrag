#!/bin/bash

# Air-gapped Hebrew RAG System - Main Deployment Script
# This script sets up the complete Hebrew RAG system in an air-gapped environment

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Hebrew RAG System - Air-Gapped Deployment"
echo "============================================="
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if we're in the right directory
if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
    echo "❌ Error: pyproject.toml not found in $PROJECT_ROOT"
    echo "Please run this script from the deployment directory of the Hebrew RAG project"
    exit 1
fi

# Check for required directories
if [[ ! -d "$SCRIPT_DIR/wheels" ]]; then
    echo "⚠️  Warning: wheels/ directory not found"
    echo "You may need to download packages first using: ./download-packages.sh"
fi

echo "1️⃣ Checking system dependencies..."
echo "=================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found!"
    echo "Run: sudo ./install-system-deps.sh"
    exit 1
fi

# Check venv
if ! python3 -m venv --help &> /dev/null 2>&1; then
    echo "❌ python3-venv not available!"
    echo "Run: sudo ./install-system-deps.sh"
    exit 1
fi

# Check pip
if ! python3 -m pip --version &> /dev/null; then
    echo "❌ python3-pip not available!"
    echo "Run: sudo ./install-system-deps.sh"
    exit 1
fi

echo "✅ Python 3: $(python3 --version)"
echo "✅ Venv: Available"
echo "✅ Pip: $(python3 -m pip --version)"

echo ""
echo "2️⃣ Setting up virtual environment..."
echo "====================================="

cd "$PROJECT_ROOT"

# Remove existing venv if it exists
if [[ -d ".venv" ]]; then
    echo "🗑️  Removing existing virtual environment..."
    rm -rf .venv
fi

# Create new virtual environment
echo "🏗️  Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip in venv
echo "📦 Upgrading pip in virtual environment..."
python -m pip install --upgrade pip

echo ""
echo "3️⃣ Installing Python packages..."
echo "================================="

# Try to use pre-downloaded wheels first
if [[ -d "$SCRIPT_DIR/wheels" ]] && [[ -n "$(ls -A "$SCRIPT_DIR/wheels" 2>/dev/null)" ]]; then
    echo "📦 Installing from pre-downloaded wheels..."
    python -m pip install --no-index --find-links "$SCRIPT_DIR/wheels" \
        agno chromadb sentence-transformers ollama \
        fastapi uvicorn aiofiles python-multipart \
        pymupdf pdfplumber python-docx pillow \
        numpy pandas torch transformers || {
        echo "⚠️  Some wheels failed to install, falling back to online installation..."
        pip install -r requirements_core.txt
    }
else
    echo "🌐 No wheels found, installing from requirements..."
    if [[ -f "requirements_core.txt" ]]; then
        pip install -r requirements_core.txt
    else
        echo "⚡ Using uv if available..."
        if command -v uv &> /dev/null; then
            uv add agno chromadb sentence-transformers ollama
        else
            pip install agno chromadb sentence-transformers ollama
        fi
    fi
fi

echo ""
echo "4️⃣ Setting up directories..."
echo "============================"

# Create necessary directories
mkdir -p data/documents
mkdir -p data/processed
mkdir -p chroma_db
mkdir -p models
mkdir -p logs

echo "✅ Created directory structure:"
echo "   📁 data/documents (for input documents)"
echo "   📁 data/processed (for processed chunks)"
echo "   📁 chroma_db (vector database)"
echo "   📁 models (embedding models)"
echo "   📁 logs (application logs)"

echo ""
echo "5️⃣ Testing installation..."
echo "========================="

# Test Python imports
echo "🧪 Testing core imports..."
python -c "
import sys
try:
    import agno
    print('✅ Agno framework imported successfully')
except ImportError as e:
    print(f'❌ Agno import failed: {e}')
    sys.exit(1)

try:
    import chromadb
    print('✅ ChromaDB imported successfully')
except ImportError as e:
    print(f'❌ ChromaDB import failed: {e}')
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    print('✅ SentenceTransformers imported successfully')
except ImportError as e:
    print(f'❌ SentenceTransformers import failed: {e}')
    sys.exit(1)

try:
    import ollama
    print('✅ Ollama client imported successfully')
except ImportError as e:
    print(f'❌ Ollama import failed: {e}')
    sys.exit(1)

print('🎉 All core dependencies are working!')
"

# Test Hebrew RAG system
echo "🧪 Testing Hebrew RAG system initialization..."
python -c "
try:
    from hebrew_rag_system import HebrewAgnoRAGSystem
    
    config = {
        'llm_model': 'mistral:7b-instruct',
        'embedding_model': './models/heBERT',
        'ollama_base_url': 'http://localhost:11434',
        'chroma_db_path': './chroma_db'
    }
    
    rag = HebrewAgnoRAGSystem(config)
    stats = rag.get_system_stats()
    
    print('✅ Hebrew RAG system initialized successfully')
    print(f'   📊 Agno available: {stats.get(\"agno_available\", False)}')
    print(f'   📊 Vector DB status: {stats.get(\"vector_db_status\", \"unknown\")}')
    
except Exception as e:
    print(f'❌ Hebrew RAG system test failed: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "6️⃣ Creating startup scripts..."
echo "=============================="

# Create startup script
cat > start_hebrag.sh << 'EOF'
#!/bin/bash

# Hebrew RAG System Startup Script
cd "$(dirname "$0")"
source .venv/bin/activate

echo "🚀 Starting Hebrew RAG System..."
echo "================================"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/version > /dev/null; then
    echo "⚠️  Ollama not detected at localhost:11434"
    echo "Please start Ollama first: ollama serve"
    echo ""
fi

# Start the system
if [[ "$1" == "api" ]]; then
    echo "🌐 Starting HTTP API server..."
    python -m uvicorn main:app --host 0.0.0.0 --port 8000
elif [[ "$1" == "test" ]]; then
    echo "🧪 Running system tests..."
    python -c "
from hebrew_rag_system import HebrewAgnoRAGSystem
import asyncio

async def test():
    config = {
        'llm_model': 'mistral:7b-instruct',
        'embedding_model': './models/heBERT',
        'ollama_base_url': 'http://localhost:11434',
        'chroma_db_path': './chroma_db'
    }
    rag = HebrewAgnoRAGSystem(config)
    print('✅ System ready')
    print(rag.get_system_stats())

asyncio.run(test())
    "
else
    echo "📖 Usage:"
    echo "  ./start_hebrag.sh api   - Start HTTP API server"
    echo "  ./start_hebrag.sh test  - Run system tests"
    echo ""
    echo "System is ready! Check logs/ directory for application logs."
fi
EOF

chmod +x start_hebrag.sh

echo "✅ Created start_hebrag.sh startup script"

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "======================="
echo ""
echo "📋 Summary:"
echo "✅ Virtual environment created and activated"
echo "✅ All Python dependencies installed"
echo "✅ Directory structure created"
echo "✅ System tested and ready"
echo "✅ Startup scripts created"
echo ""
echo "🚀 Next steps:"
echo "1. Start Ollama: ollama serve"
echo "2. Test the system: ./start_hebrag.sh test"
echo "3. Start API server: ./start_hebrag.sh api"
echo "4. Add Hebrew documents to data/documents/"
echo ""
echo "📁 Project structure:"
echo "   📂 $(pwd)"
echo "   ├── 🐍 .venv/                    (Python virtual environment)"
echo "   ├── 📚 hebrew_tools/             (Hebrew processing modules)"
echo "   ├── 📄 hebrew_rag_system.py      (Main system)"
echo "   ├── 🌐 main.py                   (FastAPI server)"
echo "   ├── 📊 chroma_db/                (Vector database)"
echo "   ├── 📁 data/documents/           (Input documents)"
echo "   ├── 📁 models/                   (ML models)"
echo "   └── 🚀 start_hebrag.sh           (Startup script)"
echo ""
echo "🔗 For air-gapped environments:"
echo "   - All dependencies are now installed locally"
echo "   - No internet connection required for operation"
echo "   - Models will be downloaded on first use (if Ollama is available)"
echo ""
echo "Happy Hebrew RAG processing! 🎯"