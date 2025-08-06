#!/bin/bash

# Hebrew Agentic RAG System - Docker-based Air-Gapped Bundle Creator
# This script runs the bundle creation process inside a Linux Docker container
# to avoid macOS compatibility issues

set -e

echo "=========================================="
echo "Hebrew RAG Docker-based Bundle Creator"
echo "=========================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is required but not installed."
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Configuration
BUNDLE_DIR="hebrew_rag_airgapped_bundle"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creating Docker-based bundle creation environment..."

# Create Dockerfile for bundle creation
cat > Dockerfile.bundle-creator << 'EOF'
FROM ubuntu:22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ca-certificates \
    curl \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    nodejs \
    npm \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# setup sudo
RUN apt-get update && apt-get install -y sudo
RUN echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove -y $pkg || true; done

RUN sudo apt-get -y install ca-certificates curl
RUN sudo install -m 0755 -d /etc/apt/keyrings
RUN sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
RUN sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN sudo apt-get update -y

RUN sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel packaging

# Create working directory
WORKDIR /workspace

# Copy bundle creation script
COPY bundle_creation_linux.sh /workspace/
RUN chmod +x /workspace/bundle_creation_linux.sh

CMD ["/workspace/bundle_creation_linux.sh"]
EOF

# Create the Linux bundle creation script
cat > bundle_creation_linux.sh << 'EOF'
#!/bin/bash

set -e

echo "Running bundle creation in Linux environment..."

# Configuration
BUNDLE_DIR="hebrew_rag_airgapped_bundle"
MODELS_DIR="$BUNDLE_DIR/models"
PACKAGES_DIR="$BUNDLE_DIR/packages"
DOCKER_DIR="$BUNDLE_DIR/docker"
SCRIPTS_DIR="$BUNDLE_DIR/scripts"
CONFIG_DIR="$BUNDLE_DIR/config"
DOCS_DIR="$BUNDLE_DIR/documentation"

# Create bundle directory structure
echo "Creating bundle directory structure..."
mkdir -p "$BUNDLE_DIR"/{models,packages,docker,scripts,config,documentation,tests,src,frontend,npm_packages,nodejs}
mkdir -p "$MODELS_DIR"/{ollama,transformers,ocr,embeddings}
mkdir -p "$SCRIPTS_DIR"/{deployment,maintenance,testing}

# Create separate directories for each platform
PLATFORMS=("linux_x86_64" "macosx_10_9_x86_64" "macosx_11_0_arm64" "any")
for platform in "${PLATFORMS[@]}"; do
    mkdir -p "$PACKAGES_DIR/$platform"
done

# Updated packages list with compatible versions
PACKAGES=(
    "agno>=1.0.0"
    "ollama>=0.4.0"
    "openai>=1.97.0"
    "jiter"
    "chromadb>=0.4.0"
    "torch"
    "transformers>=4.30.0"
    "sentence-transformers>=2.3.0"
    "unstructured[local-inference]>=0.10.0"
    "pymupdf>=1.20.0"
    "pdfplumber>=0.9.0"
    "python-docx"
    "pillow>=10.0.0"
    "hebrew-tokenizer>=2.0.0"
    "numpy>=1.24.0"
    "pandas>=2.0.0"
    "fastapi>=0.100.0"
    "uvicorn[standard]>=0.20.0"
    "pydantic>=2.0.0"
    "aiofiles>=23.0.0"
    "python-multipart>=0.0.6"
    "jinja2>=3.0.0"
    "python-jose[cryptography]>=3.0.0"
    "passlib[bcrypt]>=1.7.0"
    "bcrypt>=4.0.0"
    "wheel>=0.40.0"
)

echo "Downloading Python packages for multiple platforms..."

# Download for Linux x86_64 (primary target)
echo "Downloading for Linux x86_64..."
pip3 download --dest "$PACKAGES_DIR/linux_x86_64" \
    --platform linux_x86_64 --only-binary=:all: \
    "${PACKAGES[@]}" || {
    echo "⚠️ Warning: Some Linux packages failed with platform restriction, trying without..."
    pip3 download --dest "$PACKAGES_DIR/linux_x86_64" \
        --prefer-binary "${PACKAGES[@]}"
}

# Download platform-independent packages
echo "Downloading platform-independent packages..."
pip3 download --dest "$PACKAGES_DIR/any" \
    --platform any --only-binary=:all: \
    "${PACKAGES[@]}" || {
    echo "⚠️ Warning: Platform-independent download failed, using standard download..."
    pip3 download --dest "$PACKAGES_DIR/any" \
        --prefer-binary "${PACKAGES[@]}"
}

# Download source packages as fallback
echo "Downloading source packages as fallback..."
pip3 download --dest "$PACKAGES_DIR" --no-binary=:all: \
    "${PACKAGES[@]}" || {
    echo "⚠️ Warning: Some source packages failed to download"
}

# Download Hugging Face models (if not already present)
echo "Downloading Hebrew language models..."
if [ ! -d "$MODELS_DIR/transformers/heBERT" ]; then
    git clone https://huggingface.co/avichr/heBERT "$MODELS_DIR/transformers/heBERT" || echo "Warning: Failed to download heBERT"
fi

if [ ! -d "$MODELS_DIR/transformers/multilingual-miniLM" ]; then
    git clone https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 "$MODELS_DIR/transformers/multilingual-miniLM" || echo "Warning: Failed to download multilingual model"
fi

# Copy frontend if it exists
if [ -d "/workspace/frontend" ]; then
    echo "Copying frontend..."
    cp -r /workspace/frontend/* "$BUNDLE_DIR/frontend/"
    
    # Bundle npm packages in Linux environment
    cd "$BUNDLE_DIR/frontend"
    if [ -f "package.json" ]; then
        echo "Installing npm dependencies in Linux..."
        npm install --prefer-offline || echo "Warning: npm install failed"
        
        if [ -d "node_modules" ]; then
            echo "Bundling node_modules..."
            tar -czf "../npm_packages/node_modules.tar.gz" node_modules/
        fi
    fi
    cd /workspace
fi

# Download and prepare Ollama models
echo "Setting up Ollama models..."
ollama serve &
OLLAMA_PID=$!
sleep 10

# Download essential models
OLLAMA_MODELS=(
    "nomic-embed-text:latest"
    "qwen2.5:7b"
    "llama3.1:8b"
)

mkdir -p "$MODELS_DIR/ollama"
for model in "${OLLAMA_MODELS[@]}"; do
    echo "Downloading Ollama model: $model"
    if ollama pull "$model"; then
        echo "✓ Downloaded $model"
    else
        echo "⚠️ Warning: Failed to download $model"
    fi
done

# Create model export script
cat > "$MODELS_DIR/ollama/export_models.sh" << 'EXPORT_SCRIPT'
#!/bin/bash
echo "Exporting Ollama models for air-gapped deployment..."
# Note: This is a placeholder. Ollama model export requires manual steps
# See documentation for detailed instructions
EXPORT_SCRIPT
chmod +x "$MODELS_DIR/ollama/export_models.sh"

# Stop Ollama
kill $OLLAMA_PID || true

# Copy source code from mount point
if [ -d "/workspace/src" ]; then
    echo "Copying source code..."
    cp -r /workspace/src/* "$BUNDLE_DIR/src/"
fi

if [ -d "/workspace/hebrew_tools" ]; then
    echo "Copying Hebrew tools..."
    cp -r /workspace/hebrew_tools "$BUNDLE_DIR/"
else
    echo "Creating Hebrew tools stub..."
    mkdir -p "$BUNDLE_DIR/hebrew_tools"
    
    # Create __init__.py with stub classes
    cat > "$BUNDLE_DIR/hebrew_tools/__init__.py" << 'HEBREW_TOOLS_EOF'
"""
Hebrew Tools Module - Stub Implementation

This module contains placeholder implementations for Hebrew-specific
document processing tools. These should be implemented based on your
specific requirements.

To implement these tools properly:
1. Install additional Hebrew NLP libraries as needed
2. Implement the actual logic for each class
3. Test with Hebrew documents in your domain
"""

class HebrewDocumentProcessor:
    """Processes Hebrew documents maintaining spatial context"""
    
    def __init__(self):
        print("⚠️  Using stub HebrewDocumentProcessor - implement actual logic")
    
    def process_document(self, document_path: str):
        """Process a Hebrew document and extract contextual chunks"""
        # TODO: Implement actual Hebrew document processing
        return {"status": "stub", "message": "Hebrew document processing not implemented"}

class HebrewSemanticSearch:
    """Performs semantic search on Hebrew content"""
    
    def __init__(self):
        print("⚠️  Using stub HebrewSemanticSearch - implement actual logic")
    
    def search(self, query: str, documents: list):
        """Search for semantically similar Hebrew content"""
        # TODO: Implement actual Hebrew semantic search
        return {"status": "stub", "message": "Hebrew semantic search not implemented"}

class HebrewTableAnalyzer:
    """Analyzes tables and charts in Hebrew documents"""
    
    def __init__(self):
        print("⚠️  Using stub HebrewTableAnalyzer - implement actual logic")
    
    def analyze_table(self, table_data):
        """Analyze Hebrew table data"""
        # TODO: Implement actual Hebrew table analysis
        return {"status": "stub", "message": "Hebrew table analysis not implemented"}

class HebrewResponseGenerator:
    """Generates Hebrew responses from retrieved content"""
    
    def __init__(self):
        print("⚠️  Using stub HebrewResponseGenerator - implement actual logic")
    
    def generate_response(self, query: str, context: str):
        """Generate a Hebrew response"""
        # TODO: Implement actual Hebrew response generation
        return {"status": "stub", "message": "Hebrew response generation not implemented"}

# Export all classes
__all__ = [
    'HebrewDocumentProcessor',
    'HebrewSemanticSearch', 
    'HebrewTableAnalyzer',
    'HebrewResponseGenerator'
]
HEBREW_TOOLS_EOF

    # Create document processor module
    cat > "$BUNDLE_DIR/hebrew_tools/document_processor.py" << 'DOC_PROCESSOR_EOF'
"""
Hebrew Document Processor

TODO: Implement actual Hebrew document processing logic
- Support for RTL text handling
- Hebrew OCR integration
- Contextual chunk extraction
- Table and image processing
"""

from typing import Dict, List, Any

class HebrewDocumentProcessor:
    def __init__(self):
        pass
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process Hebrew PDF document"""
        # TODO: Implement Hebrew PDF processing
        return {"error": "Not implemented"}
    
    def extract_text_with_layout(self, document_path: str) -> List[Dict]:
        """Extract text while preserving layout information"""
        # TODO: Implement layout-aware text extraction
        return []
    
    def process_tables(self, document_path: str) -> List[Dict]:
        """Extract and process Hebrew tables"""
        # TODO: Implement Hebrew table processing
        return []
DOC_PROCESSOR_EOF

    # Create semantic search module
    cat > "$BUNDLE_DIR/hebrew_tools/semantic_search.py" << 'SEMANTIC_SEARCH_EOF'
"""
Hebrew Semantic Search

TODO: Implement Hebrew-aware semantic search
- Hebrew word embeddings
- Morphological analysis
- Synonym handling
- Context-aware retrieval
"""

from typing import List, Dict, Any

class HebrewSemanticSearch:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
    
    def encode_hebrew_text(self, text: str) -> List[float]:
        """Encode Hebrew text into embeddings"""
        # TODO: Implement Hebrew text encoding
        return []
    
    def search_similar(self, query: str, corpus: List[str], top_k: int = 5) -> List[Dict]:
        """Find semantically similar Hebrew texts"""
        # TODO: Implement Hebrew semantic search
        return []
    
    def preprocess_hebrew_query(self, query: str) -> str:
        """Preprocess Hebrew query for better search"""
        # TODO: Implement Hebrew query preprocessing
        return query
SEMANTIC_SEARCH_EOF

    # Create table analyzer module  
    cat > "$BUNDLE_DIR/hebrew_tools/table_analyzer.py" << 'TABLE_ANALYZER_EOF'
"""
Hebrew Table Analyzer

TODO: Implement Hebrew table analysis
- Hebrew header detection
- Numeric data extraction
- RTL table processing
- Chart and graph interpretation
"""

from typing import Dict, List, Any

class HebrewTableAnalyzer:
    def __init__(self):
        pass
    
    def detect_hebrew_tables(self, document_path: str) -> List[Dict]:
        """Detect tables in Hebrew documents"""
        # TODO: Implement Hebrew table detection
        return []
    
    def extract_table_data(self, table_region: Dict) -> Dict[str, Any]:
        """Extract structured data from Hebrew tables"""
        # TODO: Implement Hebrew table data extraction
        return {}
    
    def analyze_numeric_data(self, table_data: Dict) -> Dict[str, Any]:
        """Analyze numeric data in Hebrew context"""
        # TODO: Implement Hebrew numeric analysis
        return {}
TABLE_ANALYZER_EOF

    # Create response generator module
    cat > "$BUNDLE_DIR/hebrew_tools/response_generator.py" << 'RESPONSE_GEN_EOF'
"""
Hebrew Response Generator

TODO: Implement Hebrew response generation
- Natural Hebrew language generation
- Context-aware responses
- Citation formatting
- Grammar and style checking
"""

from typing import Dict, List, Any

class HebrewResponseGenerator:
    def __init__(self, model=None):
        self.model = model
    
    def generate_hebrew_response(self, query: str, context: List[str]) -> str:
        """Generate natural Hebrew response"""
        # TODO: Implement Hebrew response generation
        return "מצטער, מחולל תגובות עברית לא מיושם עדיין"
    
    def format_citations(self, sources: List[Dict]) -> str:
        """Format citations in Hebrew style"""
        # TODO: Implement Hebrew citation formatting
        return ""
    
    def check_hebrew_grammar(self, text: str) -> Dict[str, Any]:
        """Check Hebrew grammar and style"""
        # TODO: Implement Hebrew grammar checking
        return {"status": "not_implemented"}
RESPONSE_GEN_EOF

    # Create embedder module
    cat > "$BUNDLE_DIR/hebrew_tools/embedder.py" << 'EMBEDDER_EOF'
"""
Hebrew Embedder

TODO: Implement Hebrew-specific embedding functionality
- Hebrew BERT models
- Multilingual embeddings
- Domain-specific Hebrew models
"""

from typing import List
import numpy as np

class HebrewEmbedder:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "./models/transformers/heBERT"
        # TODO: Initialize Hebrew embedding model
        
    def embed_text(self, text: str) -> np.ndarray:
        """Embed Hebrew text"""
        # TODO: Implement Hebrew text embedding
        return np.zeros(768)  # Placeholder
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embed multiple Hebrew documents"""
        # TODO: Implement batch Hebrew document embedding
        return np.zeros((len(documents), 768))  # Placeholder
EMBEDDER_EOF

    echo "✓ Created Hebrew tools stub modules"
fi

# Create requirements.txt with updated versions
cat > "$BUNDLE_DIR/requirements.txt" << 'REQ_EOF'
# Core Agno Framework
agno>=1.0.0

# LLM Integration
ollama>=0.4.0
openai>=1.97.0
jiter

# Vector Database
chromadb>=0.4.0

# ML/AI Libraries
torch
transformers>=4.30.0
sentence-transformers>=2.3.0

# Document Processing
unstructured[local-inference]>=0.10.0
pymupdf>=1.20.0
pdfplumber>=0.9.0
python-docx
pillow>=10.0.0

# Hebrew Language Processing
hebrew-tokenizer>=2.0.0

# Scientific Computing
numpy>=1.24.0
pandas>=2.0.0

# Web Framework
fastapi>=0.100.0
uvicorn[standard]>=0.20.0
pydantic>=2.0.0
aiofiles>=23.0.0
python-multipart>=0.0.6

# Utilities
jinja2>=3.0.0
python-jose[cryptography]>=3.0.0
passlib[bcrypt]>=1.7.0
bcrypt>=4.0.0
wheel>=0.40.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
REQ_EOF

# Create deployment scripts (copied from original but with fixes)
echo "Creating deployment scripts..."
mkdir -p "$SCRIPTS_DIR/deployment"

# Main deployment script
cat > "$SCRIPTS_DIR/deploy.sh" << 'DEPLOY_EOF'
#!/bin/bash
set -e
echo "Deploying Hebrew Agentic RAG System (Air-Gapped)"
echo "Phase 1: Environment setup..."
./scripts/deployment/phase1_environment.sh
echo "Phase 2: Installing dependencies..."
./scripts/deployment/phase2_dependencies.sh
echo "Phase 3: Configuring system..."
./scripts/deployment/phase3_configuration.sh
echo "Phase 4: Starting services..."
./scripts/deployment/phase4_startup.sh
echo "Deployment completed successfully!"
echo "Hebrew RAG system available at http://localhost:8000"
DEPLOY_EOF
chmod +x "$SCRIPTS_DIR/deploy.sh"

# Environment setup phase
cat > "$SCRIPTS_DIR/deployment/phase1_environment.sh" << 'PHASE1_EOF'
#!/bin/bash
echo "Setting up air-gapped environment..."
mkdir -p {documents,logs,chroma_db,cache}
chmod 755 documents logs chroma_db cache

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "✓ Docker installed"
fi

echo "Phase 1 completed"
PHASE1_EOF
chmod +x "$SCRIPTS_DIR/deployment/phase1_environment.sh"

# Dependencies installation phase
cat > "$SCRIPTS_DIR/deployment/phase2_dependencies.sh" << 'PHASE2_EOF'
#!/bin/bash
set -e
echo "Installing dependencies from local packages..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages from local directory
echo "Installing from bundled packages..."
pip install --upgrade --no-index --find-links packages/ pip setuptools wheel
pip install --no-index --find-links packages/ --find-links packages/linux_x86_64 --find-links packages/any -r requirements.txt

# Install Ollama
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "Phase 2 completed"
PHASE2_EOF
chmod +x "$SCRIPTS_DIR/deployment/phase2_dependencies.sh"

# Configuration phase
cat > "$SCRIPTS_DIR/deployment/phase3_configuration.sh" << 'PHASE3_EOF'
#!/bin/bash
echo "Configuring system..."
export AGNO_TELEMETRY=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Create .env file
cat > .env << 'ENV_EOF'
AGNO_TELEMETRY=false
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
OLLAMA_HOST=0.0.0.0:11434
ENV_EOF

echo "Phase 3 completed"
PHASE3_EOF
chmod +x "$SCRIPTS_DIR/deployment/phase3_configuration.sh"

# Startup phase
cat > "$SCRIPTS_DIR/deployment/phase4_startup.sh" << 'PHASE4_EOF'
#!/bin/bash
set -e
echo "Starting services..."
source venv/bin/activate
source .env

# Start Ollama
if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "Starting Ollama..."
    nohup ollama serve > logs/ollama.log 2>&1 &
    sleep 10
fi

# Start the application
echo "Starting Hebrew RAG system..."
if [ -f "src/main.py" ]; then
    nohup python src/main.py > logs/app.log 2>&1 &
    echo "System started! Access at http://localhost:8000"
else
    echo "❌ Application files not found"
    exit 1
fi

echo "Phase 4 completed"
PHASE4_EOF
chmod +x "$SCRIPTS_DIR/deployment/phase4_startup.sh"

# Create documentation
mkdir -p "$DOCS_DIR"
cat > "$DOCS_DIR/README.md" << 'DOC_EOF'
# Hebrew Agentic RAG System - Air-Gapped Bundle

This bundle was created in a Linux Docker environment for maximum compatibility.

## Quick Start
1. Extract bundle on target Linux system
2. Run: `./scripts/deploy.sh`
3. Access: http://localhost:8000

## Requirements
- Linux x86_64 system
- 16GB+ RAM
- 50GB+ storage
- Docker (will be installed if missing)

## Manual Model Export
Ollama models require manual export from a connected system.
See deployment guide for details.
DOC_EOF

# Create verification script
cat > "$BUNDLE_DIR/verify_bundle.sh" << 'VERIFY_EOF'
#!/bin/bash
echo "Verifying bundle..."
echo "✓ Bundle created in Linux Docker environment"

# Count packages
total_packages=$(find packages/ -name "*.whl" -o -name "*.tar.gz" 2>/dev/null | wc -l)
linux_packages=$(find packages/linux_x86_64/ -name "*.whl" 2>/dev/null | wc -l)
echo "✓ Total packages: $total_packages"
echo "✓ Linux packages: $linux_packages"

# Check scripts
if [ -x "scripts/deploy.sh" ]; then
    echo "✓ Deployment script ready"
else
    echo "❌ Deployment script missing"
fi

echo "Bundle verification completed!"
VERIFY_EOF
chmod +x "$BUNDLE_DIR/verify_bundle.sh"

echo "✓ Bundle creation completed successfully!"
echo "Bundle location: $BUNDLE_DIR"
echo "Run: ./$BUNDLE_DIR/verify_bundle.sh"
EOF

chmod +x bundle_creation_linux.sh

# Build the Docker image
echo "Building Docker image for bundle creation..."
docker build -f Dockerfile.bundle-creator -t hebrew-rag-bundle-creator .

# Run the bundle creation in Docker container
echo "Running bundle creation in Linux Docker container..."
docker run -it --rm \
    -v "$SCRIPT_DIR":/workspace \
    -v /var/run/docker.sock:/var/run/docker.sock \
    hebrew-rag-bundle-creator

echo "✓ Docker-based bundle creation completed!"

# Clean up
rm -f Dockerfile.bundle-creator bundle_creation_linux.sh

echo ""
echo "Bundle has been created using Linux environment."
echo "This should resolve the macOS compatibility issues."
echo "Next: cd $BUNDLE_DIR && ./verify_bundle.sh"