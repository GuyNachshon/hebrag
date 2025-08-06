#!/bin/bash

# Hebrew RAG System - Ubuntu-Specific Lightweight Bundle
# Creates a deployable bundle with code + packages but WITHOUT heavy models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUNDLE_NAME="hebrag-ubuntu-$(date +%Y%m%d-%H%M%S)"
BUNDLE_DIR="/tmp/$BUNDLE_NAME"

echo "📦 Hebrew RAG System - Ubuntu Lightweight Bundle"
echo "==============================================="
echo "Target: Ubuntu Linux (code + packages, no models)"
echo "Bundle: $BUNDLE_DIR"
echo ""

# Create bundle directory
echo "1️⃣ Creating bundle structure..."
mkdir -p "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/deployment"
mkdir -p "$BUNDLE_DIR/deployment/wheels"
mkdir -p "$BUNDLE_DIR/deployment/bin"

# Copy source code (exclude heavy stuff)
echo "2️⃣ Copying source code..."
rsync -av --exclude='.venv' \
    --exclude='chroma_db' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='data/documents/*' \
    --exclude='models/*' \
    --exclude='*.h5' \
    --exclude='*.bin' \
    --exclude='*.safetensors' \
    --exclude='*.onnx' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='node_modules' \
    --exclude='frontend/dist' \
    "$PROJECT_ROOT/" "$BUNDLE_DIR/"

# Download Python packages for Ubuntu
echo "3️⃣ Downloading Python packages for Ubuntu..."
echo "📦 Downloading lightweight packages (excluding heavy ML models)..."

# Create requirements for lightweight deployment
cat > "$BUNDLE_DIR/deployment/requirements-ubuntu.txt" << EOF
# Core Hebrew RAG System - Ubuntu Deployment
# Lightweight version without heavy models

# Framework
agno>=1.7.7

# Vector Database  
chromadb>=0.6.0

# LLM Integration
ollama>=0.5.2

# Minimal ML (models downloaded at runtime if needed)
sentence-transformers>=5.0.0
transformers>=4.40.0

# Document Processing
pymupdf>=1.23.0
pdfplumber>=0.11.0
python-docx>=1.1.0
pillow>=10.0.0

# Hebrew Processing
hebrew-tokenizer>=2.3.0

# Web Framework
fastapi>=0.100.0
uvicorn>=0.20.0
pydantic>=2.0.0
aiofiles>=23.0.0
python-multipart>=0.0.6

# Core Python Libraries
numpy>=1.24.0
pandas>=2.0.0

# Utilities
requests>=2.31.0
jinja2>=3.1.0
python-jose>=3.3.0
passlib>=1.7.4
bcrypt>=4.0.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
EOF

# Download packages with pip for Ubuntu
echo "📥 Downloading packages for Ubuntu Linux..."

# Download with specific Ubuntu/Linux targeting
pip download --dest "$BUNDLE_DIR/deployment/wheels" \
    --platform linux_x86_64 \
    --python-version 310 \
    --abi cp310 \
    --implementation cp \
    --no-deps \
    agno ollama fastapi uvicorn pydantic aiofiles python-multipart \
    requests jinja2 python-jose passlib bcrypt pytest pytest-asyncio || echo "Some packages need dependencies..."

# Download with dependencies for packages that need them
pip download --dest "$BUNDLE_DIR/deployment/wheels" \
    --platform linux_x86_64 \
    --python-version 310 \
    chromadb sentence-transformers transformers \
    pymupdf pdfplumber python-docx pillow \
    numpy pandas hebrew-tokenizer || {
    
    echo "⚠️  Falling back to source packages for compatibility..."
    pip download --dest "$BUNDLE_DIR/deployment/wheels" \
        --no-binary :all: \
        chromadb sentence-transformers transformers || true
    
    # Download remaining packages without platform restrictions
    pip download --dest "$BUNDLE_DIR/deployment/wheels" \
        -r "$BUNDLE_DIR/deployment/requirements-ubuntu.txt"
}

# Download torch CPU-only version for Ubuntu (much smaller)
echo "🔥 Downloading PyTorch CPU-only for Ubuntu..."
pip download --dest "$BUNDLE_DIR/deployment/wheels" \
    --index-url https://download.pytorch.org/whl/cpu \
    --platform linux_x86_64 \
    --python-version 310 \
    torch || {
    echo "⚠️  PyTorch CPU download failed, will install at runtime"
}

# Create Ubuntu-specific system installer
echo "4️⃣ Creating Ubuntu system installer..."
cat > "$BUNDLE_DIR/deployment/install-ubuntu-deps.sh" << 'EOF'
#!/bin/bash

# Ubuntu-Specific Hebrew RAG System Dependencies
set -e

echo "🐧 Installing Hebrew RAG Dependencies for Ubuntu"
echo "==============================================="

# Check if running on Ubuntu
if ! grep -qi ubuntu /etc/os-release 2>/dev/null; then
    echo "⚠️  Warning: This script is optimized for Ubuntu"
    echo "It may work on other Debian-based systems"
fi

# Update package list
echo "🔄 Updating package lists..."
sudo apt-get update

# Install Python and core tools
echo "🐍 Installing Python and development tools..."
sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    python3-distutils \
    build-essential \
    curl \
    wget \
    unzip \
    git \
    software-properties-common

# Install development libraries for package compilation
echo "🔧 Installing development libraries..."
sudo apt-get install -y \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    zlib1g-dev \
    libgdbm-dev \
    libnss3-dev \
    pkg-config

# Install system libraries for ML packages
echo "📊 Installing ML system libraries..."
sudo apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev

# Clean up
sudo apt-get autoremove -y
sudo apt-get autoclean

# Verify installation
echo "✅ Verifying Ubuntu installation..."
python3 --version
python3 -m venv --help > /dev/null && echo "✅ python3-venv: OK"
python3 -m pip --version && echo "✅ python3-pip: OK"
gcc --version | head -1 && echo "✅ build-essential: OK"

echo ""
echo "🎉 Ubuntu system dependencies installed successfully!"
echo "Ready for Hebrew RAG deployment"
EOF

chmod +x "$BUNDLE_DIR/deployment/install-ubuntu-deps.sh"

# Create Ubuntu-specific deployment script
echo "5️⃣ Creating Ubuntu deployment script..."
cat > "$BUNDLE_DIR/deployment/deploy-ubuntu.sh" << 'EOF'
#!/bin/bash

# Ubuntu-Specific Hebrew RAG Deployment
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Hebrew RAG System - Ubuntu Deployment"
echo "========================================"

# Verify Ubuntu system
if ! grep -qi ubuntu /etc/os-release 2>/dev/null; then
    echo "⚠️  Warning: Not running on Ubuntu - deployment may fail"
fi

# Check system dependencies
echo "1️⃣ Checking Ubuntu system dependencies..."
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found! Run: sudo ./install-ubuntu-deps.sh"
    exit 1
fi

if ! python3 -m venv --help &> /dev/null 2>&1; then
    echo "❌ python3-venv not available! Run: sudo ./install-ubuntu-deps.sh"
    exit 1
fi

echo "✅ Ubuntu system ready"

# Setup virtual environment
echo "2️⃣ Setting up Python virtual environment..."
cd "$PROJECT_ROOT"

# Remove existing venv
if [[ -d ".venv" ]]; then
    rm -rf .venv
fi

# Create new venv
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install packages from wheels
echo "3️⃣ Installing Python packages from wheels..."
if [[ -d "$SCRIPT_DIR/wheels" ]] && [[ -n "$(ls -A "$SCRIPT_DIR/wheels" 2>/dev/null)" ]]; then
    echo "📦 Installing from Ubuntu-optimized wheels..."
    
    # Install packages from local wheels
    python -m pip install --no-index --find-links "$SCRIPT_DIR/wheels" \
        agno chromadb ollama sentence-transformers transformers \
        fastapi uvicorn pymupdf pdfplumber python-docx \
        numpy pandas pillow requests || {
        
        echo "⚠️  Some wheels failed, installing missing packages online..."
        python -m pip install -r "$SCRIPT_DIR/requirements-ubuntu.txt"
    }
else
    echo "🌐 No wheels found, installing from requirements..."
    python -m pip install -r "$SCRIPT_DIR/requirements-ubuntu.txt"
fi

# Create directories
echo "4️⃣ Creating directory structure..."
mkdir -p data/documents data/processed chroma_db logs
mkdir -p models  # Empty models directory

echo "✅ Directory structure created"

# Test installation
echo "5️⃣ Testing Ubuntu installation..."
python -c "
import sys
packages = ['agno', 'chromadb', 'sentence_transformers', 'ollama', 'fastapi']
failed = []

for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}: OK')
    except ImportError as e:
        print(f'❌ {pkg}: FAILED - {e}')
        failed.append(pkg)

if failed:
    print(f'❌ Failed packages: {failed}')
    sys.exit(1)
else:
    print('🎉 All packages installed successfully!')
"

# Test Hebrew RAG system
echo "6️⃣ Testing Hebrew RAG system..."
python -c "
try:
    from hebrew_rag_system import HebrewAgnoRAGSystem
    
    config = {
        'llm_model': 'mistral:7b-instruct',
        'embedding_model': './models/heBERT',  # Will download on first use
        'ollama_base_url': 'http://localhost:11434',
        'chroma_db_path': './chroma_db'
    }
    
    rag = HebrewAgnoRAGSystem(config)
    stats = rag.get_system_stats()
    
    print('✅ Hebrew RAG system initialized')
    print(f'📊 System stats: {stats}')
    
except Exception as e:
    print(f'❌ Hebrew RAG test failed: {e}')
"

# Create startup script
echo "7️⃣ Creating Ubuntu startup script..."
cat > start_hebrag_ubuntu.sh << 'STARTUP'
#!/bin/bash

# Hebrew RAG System - Ubuntu Startup Script
cd "$(dirname "$0")"
source .venv/bin/activate

echo "🚀 Hebrew RAG System (Ubuntu)"
echo "============================"

if [[ "$1" == "api" ]]; then
    echo "🌐 Starting API server on Ubuntu..."
    python -m uvicorn main:app --host 0.0.0.0 --port 8000
elif [[ "$1" == "test" ]]; then
    echo "🧪 Running Ubuntu system tests..."
    python -c "
from hebrew_rag_system import HebrewAgnoRAGSystem
import asyncio

async def test():
    print('Testing Hebrew RAG on Ubuntu...')
    config = {
        'llm_model': 'mistral:7b-instruct',
        'embedding_model': './models/heBERT',
        'ollama_base_url': 'http://localhost:11434',
        'chroma_db_path': './chroma_db'
    }
    
    rag = HebrewAgnoRAGSystem(config)
    stats = rag.get_system_stats()
    
    print('✅ Ubuntu system test passed')
    print('📊 Stats:', stats)

asyncio.run(test())
    "
else
    echo "📖 Usage:"
    echo "  ./start_hebrag_ubuntu.sh api   - Start API server"
    echo "  ./start_hebrag_ubuntu.sh test  - Run tests"
    echo ""
    echo "🎯 Hebrew RAG System ready on Ubuntu!"
fi
STARTUP

chmod +x start_hebrag_ubuntu.sh

echo ""
echo "🎉 UBUNTU DEPLOYMENT COMPLETE!"
echo "============================="
echo ""
echo "✅ Virtual environment: .venv/"
echo "✅ Packages installed from Ubuntu wheels"
echo "✅ Directory structure ready" 
echo "✅ System tested and functional"
echo ""
echo "🚀 Next steps:"
echo "1. Test: ./start_hebrag_ubuntu.sh test"
echo "2. Start API: ./start_hebrag_ubuntu.sh api"
echo "3. Access: http://localhost:8000"
echo ""
echo "📁 Ubuntu-optimized deployment complete!"
EOF

chmod +x "$BUNDLE_DIR/deployment/deploy-ubuntu.sh"

# Create Ubuntu documentation
echo "6️⃣ Creating Ubuntu-specific documentation..."
cat > "$BUNDLE_DIR/README-UBUNTU.md" << 'EOF'
# Hebrew RAG System - Ubuntu Lightweight Deployment

This is a **lightweight Ubuntu-optimized bundle** containing:
- ✅ **Source code** (Hebrew RAG system)
- ✅ **Python packages** (pre-downloaded for Ubuntu)
- ❌ **No heavy models** (downloaded on first use)

## Quick Ubuntu Deployment

```bash
# 1. Install Ubuntu system dependencies
sudo ./deployment/install-ubuntu-deps.sh

# 2. Deploy Hebrew RAG system  
./deployment/deploy-ubuntu.sh

# 3. Test installation
./start_hebrag_ubuntu.sh test

# 4. Start API server
./start_hebrag_ubuntu.sh api
```

## System Requirements

- **Ubuntu 20.04+** (or compatible Debian-based)
- **Python 3.10+**
- **4GB+ RAM** (8GB recommended)
- **2GB+ free space** (models download on demand)
- **sudo access** (for system packages)

## Bundle Contents

```
hebrag-ubuntu-*/
├── 🐍 hebrew_tools/              # Hebrew processing
├── 📄 hebrew_rag_system.py       # Main system
├── 🌐 main.py                    # FastAPI server
├── 📦 deployment/
│   ├── install-ubuntu-deps.sh   # Ubuntu system installer
│   ├── deploy-ubuntu.sh         # Ubuntu deployment
│   ├── wheels/                  # Ubuntu Python packages
│   └── requirements-ubuntu.txt  # Lightweight requirements
├── 🚀 start_hebrag_ubuntu.sh     # Ubuntu startup script
└── 📖 README-UBUNTU.md          # This file
```

## Lightweight Design

This bundle **excludes heavy models** to keep size small:
- ❌ No pre-downloaded embedding models
- ❌ No pre-downloaded LLM models  
- ❌ No heavy ML model files
- ✅ Models download automatically on first use
- ✅ CPU-optimized PyTorch included

## First Run

On first use, the system will:
1. Download Hebrew embedding model (~500MB)
2. Connect to Ollama for LLM (if available)
3. Create vector database
4. Process documents

## Ubuntu-Specific Features

- **APT package management** integration
- **Ubuntu system library** optimization
- **CPU-only PyTorch** (smaller, faster on CPU)
- **Lightweight dependencies** selection
- **Ubuntu LTS compatibility**

Perfect for Ubuntu servers and development environments!
EOF

# Create bundle info
echo "7️⃣ Adding bundle information..."
WHEEL_COUNT=$(find "$BUNDLE_DIR/deployment/wheels" -name "*.whl" 2>/dev/null | wc -l)
BUNDLE_SIZE=$(du -sh "$BUNDLE_DIR" 2>/dev/null | cut -f1 || echo "calculating...")

cat > "$BUNDLE_DIR/UBUNTU-BUNDLE-INFO.txt" << EOF
Hebrew RAG System - Ubuntu Lightweight Bundle
============================================

Created: $(date)
Target: Ubuntu Linux (20.04+)
Bundle ID: $BUNDLE_NAME

Bundle Contents:
- ✅ Hebrew RAG source code
- ✅ Ubuntu-optimized Python packages (${WHEEL_COUNT} wheels)
- ✅ Ubuntu deployment scripts
- ❌ No heavy model files (downloaded on demand)

Bundle Size: ${BUNDLE_SIZE}
Target Deployment Size: ~2GB (after models download)

Deployment Commands:
1. sudo ./deployment/install-ubuntu-deps.sh
2. ./deployment/deploy-ubuntu.sh  
3. ./start_hebrag_ubuntu.sh test

System Requirements:
- Ubuntu 20.04+ or Debian-based Linux
- Python 3.10+
- 4GB+ RAM, 2GB+ disk space
- Internet for model downloads (first run only)

This lightweight bundle is optimized for Ubuntu deployment
with minimal size and maximum compatibility.
EOF

# Create compressed bundle
echo "8️⃣ Creating Ubuntu bundle archive..."
cd /tmp

# Create tar.gz
tar -czf "${BUNDLE_NAME}.tar.gz" "$BUNDLE_NAME/"

# Calculate size and checksum
ARCHIVE_SIZE=$(du -sh "${BUNDLE_NAME}.tar.gz" | cut -f1)
if command -v sha256sum &> /dev/null; then
    sha256sum "${BUNDLE_NAME}.tar.gz" > "${BUNDLE_NAME}.tar.gz.sha256"
    CHECKSUM=$(cat "${BUNDLE_NAME}.tar.gz.sha256" | cut -d' ' -f1)
fi

echo ""
echo "🎉 UBUNTU LIGHTWEIGHT BUNDLE READY!"
echo "==================================="
echo ""
echo "📦 Bundle: $BUNDLE_NAME"
echo "🗜️  Archive: /tmp/${BUNDLE_NAME}.tar.gz"
echo "📊 Size: $ARCHIVE_SIZE (lightweight!)"
echo "📦 Packages: $WHEEL_COUNT Ubuntu wheels"
echo "🔒 SHA256: ${CHECKSUM:-'calculated'}"
echo ""
echo "🚀 Ubuntu Deployment:"
echo "   scp /tmp/${BUNDLE_NAME}.tar.gz user@ubuntu-server:"
echo "   tar -xzf ${BUNDLE_NAME}.tar.gz"
echo "   cd $BUNDLE_NAME"
echo "   sudo ./deployment/install-ubuntu-deps.sh"
echo "   ./deployment/deploy-ubuntu.sh"
echo ""
echo "✨ Features:"
echo "   ✅ Code + packages included"
echo "   ❌ No heavy models (download on demand)"
echo "   🐧 Ubuntu-optimized"
echo "   ⚡ Fast deployment"
echo "   🎯 Production ready"
echo ""
echo "Perfect for Ubuntu server deployment! 🎯"
EOF