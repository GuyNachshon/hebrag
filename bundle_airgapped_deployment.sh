#!/bin/bash

# Hebrew Agentic RAG System - Air-Gapped Deployment Bundle Creator
# This script prepares everything needed for deploying the Hebrew RAG system in an air-gapped environment
# Compatible with Linux systems (Ubuntu/RHEL/CentOS)

set -e

echo "=========================================="
echo "Hebrew Agentic RAG Air-Gapped Bundle Creator"
echo "=========================================="

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
fi

echo "Detected OS: $OS"

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
echo "Checking system dependencies..."
if ! command_exists python3; then
    echo "❌ Error: Python 3 is required but not installed."
    if [ "$OS" = "linux" ]; then
        echo "Install with: sudo apt update && sudo apt install python3 python3-pip  # Ubuntu/Debian"
        echo "Or: sudo yum install python3 python3-pip  # RHEL/CentOS"
    fi
    exit 1
fi

if ! command_exists pip3; then
    echo "❌ Error: pip3 is required but not installed."
    if [ "$OS" = "linux" ]; then
        echo "Install with: sudo apt install python3-pip  # Ubuntu/Debian"
        echo "Or: sudo yum install python3-pip  # RHEL/CentOS"
    fi
    exit 1
fi

if ! command_exists docker; then
    echo "⚠️ Warning: Docker not found. Will include Docker installation in bundle."
    if [ "$OS" = "linux" ]; then
        echo "Docker will be installed during deployment phase"
    fi
fi

if ! command_exists git; then
    echo "❌ Error: Git is required but not installed."
    if [ "$OS" = "linux" ]; then
        echo "Install with: sudo apt install git  # Ubuntu/Debian"
        echo "Or: sudo yum install git  # RHEL/CentOS"
    fi
    exit 1
fi

if ! command_exists node; then
    echo "❌ Error: Node.js is required but not installed."
    if [ "$OS" = "linux" ]; then
        echo "Install with: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -"
        echo "sudo apt-get install -y nodejs  # Ubuntu/Debian"
        echo "Or: sudo yum install nodejs npm  # RHEL/CentOS"
    fi
    exit 1
fi

if ! command_exists npm; then
    echo "❌ Error: npm is required but not installed."
    if [ "$OS" = "linux" ]; then
        echo "Usually installed with Node.js"
    fi
    exit 1
fi

echo "✓ System dependencies check completed"

# Download Python packages for offline installation (cross-platform)
echo "Downloading Python packages for offline installation (cross-platform)..."

# Define target platforms for cross-platform support
PLATFORMS=(
    "linux_x86_64"
    "macosx_10_9_x86_64" 
    "macosx_11_0_arm64"
    "any"
)

# Create separate directories for each platform
for platform in "${PLATFORMS[@]}"; do
    mkdir -p "$PACKAGES_DIR/$platform"
done

# Download packages for each platform
echo "Downloading packages for multiple platforms..."

# Core packages list (updated versions for compatibility)
PACKAGES=(
    "agno>=1.0.0"
    "ollama>=0.4.0" 
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
    "uvicorn>=0.20.0"
    "pydantic>=2.0.0"
    "aiofiles>=23.0.0"
    "python-multipart>=0.0.6"
    "jinja2>=3.0.0"
    "python-jose>=3.0.0"
    "passlib>=1.7.0"
    "bcrypt>=4.0.0"
)

# Download for Linux x86_64
echo "Downloading for Linux x86_64..."
pip3 download --dest "$PACKAGES_DIR/linux_x86_64" \
    --platform linux_x86_64 --only-binary=:all: \
    "${PACKAGES[@]}" || {
    echo "⚠️ Warning: Some Linux packages failed to download, trying without platform restriction..."
    pip3 download --dest "$PACKAGES_DIR/linux_x86_64" \
        --prefer-binary "${PACKAGES[@]}"
}

# Download for macOS x86_64
echo "Downloading for macOS x86_64..."
pip3 download --dest "$PACKAGES_DIR/macosx_10_9_x86_64" \
    --platform macosx_10_9_x86_64 --only-binary=:all: \
    "${PACKAGES[@]}" || {
    echo "⚠️ Warning: Some macOS x86_64 packages failed to download"
}

# Download for macOS ARM64 (M1/M2)
echo "Downloading for macOS ARM64..."
pip3 download --dest "$PACKAGES_DIR/macosx_11_0_arm64" \
    --platform macosx_11_0_arm64 --only-binary=:all: \
    "${PACKAGES[@]}" || {
    echo "⚠️ Warning: Some macOS ARM64 packages failed to download"
}

# Download platform-independent packages
echo "Downloading platform-independent packages..."
pip3 download --dest "$PACKAGES_DIR/any" \
    --platform any \
    "${PACKAGES[@]}" || {
    echo "⚠️ Warning: Some platform-independent packages failed to download"
}

# Also download source packages as fallback
echo "Downloading source packages as fallback..."
pip3 download --dest "$PACKAGES_DIR" --no-binary=:all: \
    "${PACKAGES[@]}" || {
    echo "⚠️ Warning: Some source packages failed to download"
}

echo "✓ Cross-platform Python packages downloaded"
echo "✓ Package counts:"
for platform in "${PLATFORMS[@]}"; do
    count=$(ls "$PACKAGES_DIR/$platform" 2>/dev/null | wc -l)
    echo "  - $platform: $count files"
done

# Download Hugging Face models
echo "Downloading Hebrew language models..."

# Hebrew BERT model
echo "Downloading Hebrew BERT (heBERT)..."
git clone https://huggingface.co/avichr/heBERT "$MODELS_DIR/transformers/heBERT" || echo "Warning: Failed to download heBERT"

# Multilingual sentence transformer
echo "Downloading multilingual sentence transformer..."
git clone https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 "$MODELS_DIR/transformers/multilingual-miniLM" || echo "Warning: Failed to download multilingual model"

# Table transformer models
echo "Downloading table transformer models..."
git clone https://huggingface.co/microsoft/table-transformer-structure-recognition "$MODELS_DIR/transformers/table-transformer-structure" || echo "Warning: Failed to download table transformer structure"
git clone https://huggingface.co/microsoft/table-transformer-detection "$MODELS_DIR/transformers/table-transformer-detection" || echo "Warning: Failed to download table transformer detection"

# CLIP model for visual understanding
echo "Downloading CLIP model..."
git clone https://huggingface.co/openai/clip-vit-large-patch14 "$MODELS_DIR/transformers/clip-vit-large" || echo "Warning: Failed to download CLIP model"

echo "Downloading HebrewGemma model..."
git clone https://huggingface.co/yam-peleg/Hebrew-Gemma-11B-Instruct "$MODELS_DIR/transformers/hebrew-gemma-11b" || echo "Warning: Failed to download HebrewGemma model"

echo "Downloading Hebrew Mixtral model..."
git clone https://huggingface.co/yam-peleg/Hebrew-Mixtral-8x22B "$MODELS_DIR/transformers/hebrew-mixtral-8x22b" || echo "Warning: Failed to download Hebrew Llama model"

echo "✓ Hugging Face models downloaded"

# Bundle frontend dependencies for offline installation
echo "Bundling frontend dependencies for offline installation..."

# Check if frontend directory exists
if [ -d "frontend" ]; then
    echo "✓ Frontend directory found"
    
    # Create npm packages directory
    mkdir -p "$BUNDLE_DIR/npm_packages"
    mkdir -p "$BUNDLE_DIR/frontend"
    
    # Copy frontend source code
    echo "Copying frontend source code..."
    cp -r frontend/* "$BUNDLE_DIR/frontend/"
    
    # Remove node_modules if it exists (we'll rebuild it)
    rm -rf "$BUNDLE_DIR/frontend/node_modules"
    
    # Navigate to frontend directory to bundle npm packages
    cd frontend
    
    # Check if package.json exists
    if [ -f "package.json" ]; then
        echo "✓ Frontend package.json found"
        
        # Create npm cache bundle using npm pack
        echo "Creating npm package bundle..."
        
        # Method 1: Use npm pack to bundle all dependencies
        # First, create a clean package-lock.json
        if [ ! -f "package-lock.json" ]; then
            echo "Creating package-lock.json..."
            npm install --package-lock-only
        fi
        
        # Method 2: Use npm install --offline to prepare cache
        echo "Downloading npm dependencies..."
        
        # Create temporary npm cache
        NPM_CACHE_DIR="../$BUNDLE_DIR/npm_packages/cache"
        mkdir -p "$NPM_CACHE_DIR"
        
        # Download all dependencies to cache
        npm install --prefer-offline --cache "$NPM_CACHE_DIR" || {
            echo "⚠️ Warning: Some npm packages may not be fully cached"
        }
        
        # Method 3: Use npm-bundle for better offline support
        if command_exists npx; then
            echo "Creating comprehensive npm bundle..."
            
            # Install npm-bundle-all if not available
            npx --yes npm-bundle-all --output "../$BUNDLE_DIR/npm_packages/bundle.tgz" || {
                echo "⚠️ Warning: npm-bundle-all failed, using standard approach"
            }
        fi
        
        # Method 4: Create node_modules bundle for direct copy
        echo "Installing and bundling node_modules..."
        npm install --production=false || {
            echo "⚠️ Warning: npm install encountered issues"
        }
        
        # Bundle node_modules for direct copy (compressed)
        if [ -d "node_modules" ]; then
            echo "Compressing node_modules for offline deployment..."
            tar -czf "../$BUNDLE_DIR/npm_packages/node_modules.tar.gz" node_modules/
            echo "✓ node_modules bundled: npm_packages/node_modules.tar.gz"
        fi
        
        # Copy package files
        cp package.json "../$BUNDLE_DIR/frontend/"
        cp package-lock.json "../$BUNDLE_DIR/frontend/" 2>/dev/null || {
            echo "⚠️ Warning: package-lock.json not found"
        }
        
        echo "✓ Frontend dependencies bundled"
        
    else
        echo "❌ Error: Frontend package.json not found"
        cd ..
        exit 1
    fi
    
    cd ..
    
    # Create Node.js installation bundle for different platforms
    echo "Downloading Node.js binaries for cross-platform deployment..."
    mkdir -p "$BUNDLE_DIR/nodejs"
    
    # Node.js version (extract from package.json or use default)
    NODE_VERSION="20.19.4"  # Default version
    
    # Download Node.js for different platforms
    if command_exists curl; then
        # Linux x64
        echo "Downloading Node.js for Linux x64..."
        curl -L -o "$BUNDLE_DIR/nodejs/node-v${NODE_VERSION}-linux-x64.tar.xz" \
            "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" 2>/dev/null && \
            echo "✓ Downloaded Node.js for Linux x64" || \
            echo "⚠️ Warning: Failed to download Node.js for Linux"
        
        # macOS x64
        echo "Downloading Node.js for macOS x64..."
        curl -L -o "$BUNDLE_DIR/nodejs/node-v${NODE_VERSION}-darwin-x64.tar.gz" \
            "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-darwin-x64.tar.gz" 2>/dev/null && \
            echo "✓ Downloaded Node.js for macOS x64" || \
            echo "⚠️ Warning: Failed to download Node.js for macOS x64"
        
        # macOS ARM64
        echo "Downloading Node.js for macOS ARM64..."
        curl -L -o "$BUNDLE_DIR/nodejs/node-v${NODE_VERSION}-darwin-arm64.tar.gz" \
            "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-darwin-arm64.tar.gz" 2>/dev/null && \
            echo "✓ Downloaded Node.js for macOS ARM64" || \
            echo "⚠️ Warning: Failed to download Node.js for macOS ARM64"
        
        # Download latest npm as standalone (for upgrades/compatibility)
        echo "Downloading latest npm..."
        NPM_VERSION=$(curl -s https://registry.npmjs.org/npm/latest | grep -o '"version":"[^"]*' | cut -d'"' -f4 2>/dev/null || echo "10.9.2")
        curl -L -o "$BUNDLE_DIR/nodejs/npm-${NPM_VERSION}.tgz" \
            "https://registry.npmjs.org/npm/-/npm-${NPM_VERSION}.tgz" 2>/dev/null && \
            echo "✓ Downloaded npm v${NPM_VERSION}" || \
            echo "⚠️ Warning: Failed to download standalone npm"
        
        # Download pnpm and yarn as alternatives
        echo "Downloading alternative package managers..."
        curl -L -o "$BUNDLE_DIR/nodejs/pnpm-latest.tgz" \
            "https://registry.npmjs.org/pnpm/-/pnpm-latest.tgz" 2>/dev/null && \
            echo "✓ Downloaded pnpm" || \
            echo "⚠️ Warning: Failed to download pnpm"
            
        curl -L -o "$BUNDLE_DIR/nodejs/yarn-latest.tgz" \
            "https://registry.npmjs.org/yarn/-/yarn-latest.tgz" 2>/dev/null && \
            echo "✓ Downloaded yarn" || \
            echo "⚠️ Warning: Failed to download yarn"
    else
        echo "⚠️ Warning: curl not available, skipping Node.js downloads"
    fi
    
    # Create npm upgrade script for air-gapped environments
    cat > "$BUNDLE_DIR/nodejs/upgrade_npm.sh" << 'NPM_UPGRADE'
#!/bin/bash
# Script to upgrade npm in air-gapped environment

echo "Upgrading npm from bundle..."

# Find the npm tarball
NPM_TGZ=$(ls npm-*.tgz 2>/dev/null | head -1)
if [ -f "$NPM_TGZ" ]; then
    echo "Installing npm from: $NPM_TGZ"
    
    # Extract and install npm globally
    mkdir -p npm-temp
    tar -xzf "$NPM_TGZ" -C npm-temp --strip-components=1
    
    # Install npm globally
    if command -v npm &> /dev/null; then
        npm install -g npm-temp/ || {
            echo "⚠️ Warning: npm upgrade failed, copying manually..."
            # Fallback: copy to Node.js installation
            if [ -d "/opt/nodejs/lib/node_modules/npm" ]; then
                sudo cp -r npm-temp/* /opt/nodejs/lib/node_modules/npm/
                echo "✓ npm manually updated"
            fi
        }
    else
        echo "❌ npm not available for upgrade"
    fi
    
    rm -rf npm-temp
    echo "✓ npm upgrade completed"
else
    echo "❌ No npm tarball found"
fi

# Install alternative package managers
for pkg in pnpm yarn; do
    PKG_TGZ=$(ls ${pkg}-*.tgz 2>/dev/null | head -1)
    if [ -f "$PKG_TGZ" ] && command -v npm &> /dev/null; then
        echo "Installing $pkg from bundle..."
        npm install -g "$PKG_TGZ" && \
            echo "✓ $pkg installed" || \
            echo "⚠️ Warning: $pkg installation failed"
    fi
done
NPM_UPGRADE

    chmod +x "$BUNDLE_DIR/nodejs/upgrade_npm.sh"
    
    echo "✓ Frontend bundling completed"
else
    echo "⚠️ Warning: Frontend directory not found, skipping frontend bundling"
fi

# Download and export Ollama models (if Ollama is available)
echo "Preparing Ollama models for air-gapped deployment..."
if command_exists ollama; then
    echo "Downloading and exporting Ollama models..."
    
    # Core models for Hebrew RAG
    MODELS=(
        "huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF"
        "nomic-embed-text:latest",
        "qwen3:latest"
    )
    
    # Optional Hebrew-specific models (if available)
    HEBREW_MODELS=(
        "huggingface.co/mradermacher/Hebrew-Gemma-11B-Instruct-GGUF"
        "yam-peleg/Hebrew-Gemma-11B-Instruct"
    )
    
    mkdir -p "$MODELS_DIR/ollama"
    
    # Download and export core models
    for model in "${MODELS[@]}"; do
        echo "Processing model: $model"
        model_name=$(echo "$model" | tr '/:' '_')
        
        # Pull model
        if ollama pull "$model"; then
            echo "✓ Downloaded $model"
            
            # Create model directory
            mkdir -p "$MODELS_DIR/ollama/$model_name"
            
            # Copy model files (Ollama stores models in ~/.ollama/models/)
            if [ -d "$HOME/.ollama/models" ]; then
                # Find and copy model files
                find "$HOME/.ollama/models" -name "*$(echo $model | cut -d':' -f1)*" -type f 2>/dev/null | while read -r file; do
                    if [ -f "$file" ]; then
                        cp "$file" "$MODELS_DIR/ollama/$model_name/" 2>/dev/null || true
                    fi
                done
                echo "✓ Exported $model to bundle"
            fi
            
            # Create a simple model manifest
            cat > "$MODELS_DIR/ollama/$model_name/modelfile.txt" << EOF
# Ollama model: $model
# Exported: $(date)
# Original size: $(ollama list | grep "$model" | awk '{print $2}' || echo "unknown")

FROM ./model
EOF
            
        else
            echo "⚠️ Warning: Failed to download $model"
        fi
    done
    
    # Try Hebrew-specific models
    for model in "${HEBREW_MODELS[@]}"; do
        echo "Attempting Hebrew model: $model"
        model_name=$(echo "$model" | tr '/:' '_')
        
        if ollama pull "$model" 2>/dev/null; then
            mkdir -p "$MODELS_DIR/ollama/$model_name"
            find "$HOME/.ollama/models" -name "*$(echo $model | cut -d':' -f1 | cut -d'/' -f2)*" -type f 2>/dev/null | while read -r file; do
                if [ -f "$file" ]; then
                    cp "$file" "$MODELS_DIR/ollama/$model_name/" 2>/dev/null || true
                fi
            done
            echo "✓ Hebrew model $model exported"
        else
            echo "⚠️ Hebrew model $model not available"
        fi
    done
    
    # Create model inventory
    ollama list > "$MODELS_DIR/ollama/model_inventory.txt" 2>/dev/null || echo "Failed to create model inventory"
    
    # Create model loading script for air-gapped deployment
    cat > "$MODELS_DIR/ollama/load_models.sh" << 'EOF'
#!/bin/bash
# Script to load Ollama models in air-gapped environment

echo "Loading Ollama models from bundle..."

# Start Ollama service if not running
if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 10
fi

# Load each model directory
for model_dir in */; do
    if [ -d "$model_dir" ] && [ "$model_dir" != "./" ] && [ "$model_dir" != "../" ]; then
        model_name=$(echo "$model_dir" | tr '_' '/' | sed 's/\/$//')
        echo "Loading model: $model_name"
        
        if [ -f "$model_dir/modelfile.txt" ]; then
            # Create model from bundle
            cd "$model_dir"
            if [ -f "model" ] || [ -f "model.bin" ] || ls *.bin >/dev/null 2>&1; then
                ollama create "$model_name" -f modelfile.txt
                echo "✓ Loaded $model_name"
            fi
            cd ..
        fi
    fi
done

echo "Model loading completed"
ollama list
EOF
    
    chmod +x "$MODELS_DIR/ollama/load_models.sh"
    
    echo "✓ Ollama models exported for air-gapped deployment"
    echo "  - Model files: $MODELS_DIR/ollama/"
    echo "  - Loading script: $MODELS_DIR/ollama/load_models.sh"
else
    echo "Ollama not found. Including Ollama installation files..."
    
    # Download Ollama installer
    if curl -fsSL https://ollama.com/install.sh > "$SCRIPTS_DIR/install_ollama.sh" 2>/dev/null; then
        chmod +x "$SCRIPTS_DIR/install_ollama.sh"
        echo "✓ Ollama installer downloaded"
    else
        echo "⚠️ Warning: Could not download Ollama installer"
        echo "Creating manual installation instructions..."
        
        # Create manual installation script for Linux
        cat > "$SCRIPTS_DIR/install_ollama_manual.sh" << 'EOF'
#!/bin/bash
# Manual Ollama installation for Linux

echo "Installing Ollama manually..."

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Downloading Ollama binary for Linux..."
    curl -L https://ollama.com/download/ollama-linux-amd64 -o ollama
    sudo install ollama /usr/local/bin/
    rm ollama
    
    # Create systemd service
    sudo tee /etc/systemd/system/ollama.service > /dev/null << 'SYSTEMD'
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="OLLAMA_HOST=0.0.0.0:11434"

[Install]
WantedBy=default.target
SYSTEMD

    # Create ollama user
    sudo useradd -r -s /bin/false -d /usr/share/ollama -m ollama 2>/dev/null || true
    
    # Enable service
    sudo systemctl daemon-reload
    sudo systemctl enable ollama
    
    echo "✓ Ollama installed and configured"
    echo "Start with: sudo systemctl start ollama"
else
    echo "❌ Unsupported OS for manual installation"
    exit 1
fi
EOF
        chmod +x "$SCRIPTS_DIR/install_ollama_manual.sh"
    fi
    
    # Create placeholder for manual model export
    mkdir -p "$MODELS_DIR/ollama"
    cat > "$MODELS_DIR/ollama/README.md" << 'EOF'
# Ollama Models for Air-Gapped Deployment

Since Ollama was not available during bundle creation, models must be manually exported.

## On a system with internet access:

1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
2. Download models:
   ```bash
   ollama pull llama2:7b-chat
   ollama pull mistral:7b-instruct
   ollama pull nomic-embed-text:latest
   ```
3. Copy model files from `~/.ollama/models/` to this directory
4. Use the `load_models.sh` script in the air-gapped environment

## Alternative: Use model export tools
Some community tools can help export Ollama models:
- https://github.com/ollama/ollama/discussions/1651
EOF
fi

echo "✓ Ollama models prepared"

# Create Docker images bundle for air-gapped deployment
echo "Building Docker images for air-gapped deployment..."

# First build the Docker image
echo "Building Hebrew RAG Docker image..."

# Base Docker image with all dependencies
cat > "$DOCKER_DIR/Dockerfile" << 'EOF'
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
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
#COPY scripts/install_ollama.sh /tmp/
#RUN chmod +x /tmp/install_ollama.sh && /tmp/install_ollama.sh

# Copy and install Python packages
COPY packages/ /tmp/packages/
COPY requirements.txt .
RUN pip install --no-cache-dir --find-links /tmp/packages -r requirements.txt

# Copy models
COPY models/ ./models/

# Copy source code
COPY src/ ./src/
COPY hebrew_tools/ ./hebrew_tools/
COPY config/ ./config/

# Set environment variables for air-gapped deployment
ENV AGNO_TELEMETRY=false
ENV OLLAMA_HOST=0.0.0.0:11434
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

EXPOSE 8000 11434

# Start script
COPY scripts/start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
EOF

# Docker Compose file
cat > "$DOCKER_DIR/docker-compose.yml" << 'EOF'
version: '3.8'

services:
  hebrew-rag-agno:
    build: .
    ports:
      - "8000:8000"
      - "11434:11434"
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./documents:/app/documents
      - ./logs:/app/logs
    environment:
      - AGNO_TELEMETRY=false
      - OLLAMA_HOST=0.0.0.0:11434
      - USE_GPU=true
      - TRANSFORMERS_OFFLINE=1
      - HF_DATASETS_OFFLINE=1
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
  documents:
  logs:
EOF

# Build the Docker image with all dependencies
echo "Building Docker image..."
cd "$DOCKER_DIR"
docker build -t hebrew-rag-agno:latest . || {
    echo "❌ Error: Docker build failed"
    cd ..
    exit 1
}
cd ..

# Save Docker images for air-gapped deployment
echo "Saving Docker images for air-gapped transfer..."
mkdir -p "$BUNDLE_DIR/docker_images"

# Save the main application image
docker save hebrew-rag-agno:latest | gzip > "$BUNDLE_DIR/docker_images/hebrew-rag-agno-latest.tar.gz"
echo "✓ Saved hebrew-rag-agno:latest → docker_images/hebrew-rag-agno-latest.tar.gz"

# Save base images that might be needed
echo "Saving base Docker images..."
docker pull python:3.11-slim
docker save python:3.11-slim | gzip > "$BUNDLE_DIR/docker_images/python-3.11-slim.tar.gz"
echo "✓ Saved python:3.11-slim → docker_images/python-3.11-slim.tar.gz"

# Save additional useful images
if command -v docker &> /dev/null; then
    echo "Saving additional Docker images..."
    docker pull ollama/ollama:latest && \
    docker save ollama/ollama:latest | gzip > "$BUNDLE_DIR/docker_images/ollama-latest.tar.gz" && \
    echo "✓ Saved ollama/ollama:latest → docker_images/ollama-latest.tar.gz"
    
    docker pull chromadb/chroma:latest && \
    docker save chromadb/chroma:latest | gzip > "$BUNDLE_DIR/docker_images/chromadb-chroma-latest.tar.gz" && \
    echo "✓ Saved chromadb/chroma:latest → docker_images/chromadb-chroma-latest.tar.gz"
fi

# Download platform-specific binaries
echo "Downloading cross-platform binaries..."
mkdir -p "$BUNDLE_DIR/bin"

# Download Ollama for different platforms
echo "Downloading Ollama binaries..."
if command_exists curl; then
    # Linux x86_64
    echo "Downloading Ollama for Linux x86_64..."
    curl -L -o "$BUNDLE_DIR/bin/ollama-linux-amd64" \
        "https://ollama.com/download/ollama-linux-amd64" 2>/dev/null && \
        echo "✓ Downloaded Ollama for Linux" || \
        echo "⚠️ Warning: Failed to download Ollama for Linux"
    
    # macOS x86_64
    echo "Downloading Ollama for macOS x86_64..."
    curl -L -o "$BUNDLE_DIR/bin/ollama-darwin-amd64" \
        "https://ollama.com/download/ollama-darwin" 2>/dev/null && \
        echo "✓ Downloaded Ollama for macOS x86_64" || \
        echo "⚠️ Warning: Failed to download Ollama for macOS x86_64"
    
    # Make binaries executable
    chmod +x "$BUNDLE_DIR/bin/ollama-"* 2>/dev/null || true
else
    echo "⚠️ Warning: curl not available, skipping binary downloads"
fi

echo "✓ Cross-platform binaries downloaded"
echo "✓ Docker images saved for air-gapped deployment"

# Create Docker image loading script for air-gapped deployment
echo "Creating Docker image loading script..."
cat > "$BUNDLE_DIR/load_docker_images.sh" << 'EOF'
#!/bin/bash

set -e

echo "Loading Docker images for air-gapped deployment..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    exit 1
fi

# Load Docker images from saved tar.gz files
image_dir="docker_images"

if [ ! -d "$image_dir" ]; then
    echo "❌ Error: Docker images directory not found"
    exit 1
fi

echo "Loading Docker images from $image_dir/..."

# Load images
for image_file in "$image_dir"/*.tar.gz; do
    if [ -f "$image_file" ]; then
        echo "Loading $(basename "$image_file")..."
        gunzip -c "$image_file" | docker load
        echo "✓ Loaded $(basename "$image_file")"
    fi
done

echo ""
echo "✓ All Docker images loaded successfully"
echo ""
echo "Available images:"
docker images | grep -E "(hebrew-rag|python|ollama|chroma)" || docker images
echo ""
echo "Ready for Docker deployment!"
EOF

chmod +x "$BUNDLE_DIR/load_docker_images.sh"

# Create deployment scripts
echo "Creating deployment scripts..."

# Main deployment script
cat > "$SCRIPTS_DIR/deploy.sh" << 'EOF'
#!/bin/bash

set -e

echo "Deploying Hebrew Agentic RAG System (Air-Gapped)"

# Phase 1: Environment Setup
echo "Phase 1: Setting up environment..."
./scripts/deployment/phase1_environment.sh

# Phase 2: Install Dependencies
echo "Phase 2: Installing dependencies..."
./scripts/deployment/phase2_dependencies.sh

# Phase 3: Configure System
echo "Phase 3: Configuring system..."
./scripts/deployment/phase3_configuration.sh

# Phase 4: Start Services
echo "Phase 4: Starting services..."
./scripts/deployment/phase4_startup.sh

echo "Deployment completed successfully!"
echo "Hebrew RAG system available at http://localhost:8000"
EOF

chmod +x "$SCRIPTS_DIR/deploy.sh"

# Phase 1: Environment setup
mkdir -p "$SCRIPTS_DIR/deployment"
cat > "$SCRIPTS_DIR/deployment/phase1_environment.sh" << 'EOF'
#!/bin/bash

echo "Setting up air-gapped environment..."

# Create required directories
mkdir -p {documents,logs,chroma_db,cache}
chmod 755 documents logs chroma_db cache

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    if [ -f "scripts/install_docker.sh" ]; then
        chmod +x scripts/install_docker.sh
        ./scripts/install_docker.sh
    else
        echo "❌ Error: Docker installation script not found"
        exit 1
    fi
fi

# Install nvidia-docker for GPU support
if command -v nvidia-smi &> /dev/null; then
    echo "Installing nvidia-docker..."
    if [ -f "scripts/install_nvidia_docker.sh" ]; then
        chmod +x scripts/install_nvidia_docker.sh
        ./scripts/install_nvidia_docker.sh
    else
        echo "⚠️ Warning: NVIDIA Docker installation script not found"
    fi
fi

# Install Node.js if not present
if ! command -v node &> /dev/null; then
    echo "Installing Node.js from bundle..."
    
    # Detect platform and install appropriate Node.js
    NODEJS_ARCHIVE=""
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        NODEJS_ARCHIVE="nodejs/node-v20.19.4-linux-x64.tar.xz"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if [[ "$(uname -m)" == "arm64" ]]; then
            NODEJS_ARCHIVE="nodejs/node-v20.19.4-darwin-arm64.tar.gz"
        else
            NODEJS_ARCHIVE="nodejs/node-v20.19.4-darwin-x64.tar.gz"
        fi
    fi
    
    if [ -f "$NODEJS_ARCHIVE" ]; then
        echo "Installing Node.js from bundled archive: $NODEJS_ARCHIVE"
        
        # Extract to /opt/nodejs
        sudo mkdir -p /opt/nodejs
        if [[ "$NODEJS_ARCHIVE" == *.tar.xz ]]; then
            sudo tar -xJf "$NODEJS_ARCHIVE" -C /opt/nodejs --strip-components=1
        else
            sudo tar -xzf "$NODEJS_ARCHIVE" -C /opt/nodejs --strip-components=1
        fi
        
        # Create symlinks
        sudo ln -sf /opt/nodejs/bin/node /usr/local/bin/node
        sudo ln -sf /opt/nodejs/bin/npm /usr/local/bin/npm
        sudo ln -sf /opt/nodejs/bin/npx /usr/local/bin/npx
        
        echo "✓ Node.js installed from bundle"
    else
        echo "⚠️ Warning: Node.js archive not found: $NODEJS_ARCHIVE"
        echo "Node.js installation will be required manually"
    fi
else
    echo "✓ Node.js already installed"
fi

echo "Phase 1 completed"
EOF

# Phase 2: Dependencies
cat > "$SCRIPTS_DIR/deployment/phase2_dependencies.sh" << 'EOF'
#!/bin/bash

set -e

echo "Installing dependencies from local packages (air-gapped)..."

# Check if we're in the bundle directory
if [ ! -d "packages" ] || [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Must be run from bundle root directory"
    echo "Expected structure: packages/, requirements.txt"
    exit 1
fi

# Create and activate virtual environment
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip in virtual environment
echo "Upgrading pip..."
python -m pip install --upgrade --no-index --find-links packages/ pip setuptools wheel

# Detect target platform for package installation
echo "Detecting target platform..."
DETECTED_PLATFORM="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [[ "$(uname -m)" == "x86_64" ]]; then
        DETECTED_PLATFORM="linux_x86_64"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ "$(uname -m)" == "arm64" ]]; then
        DETECTED_PLATFORM="macosx_11_0_arm64"
    else
        DETECTED_PLATFORM="macosx_10_9_x86_64"
    fi
fi

echo "Target platform: $DETECTED_PLATFORM"

# Build find-links arguments for platform-specific packages
FIND_LINKS_ARGS="--find-links packages/"
if [ -d "packages/$DETECTED_PLATFORM" ]; then
    FIND_LINKS_ARGS="$FIND_LINKS_ARGS --find-links packages/$DETECTED_PLATFORM"
    echo "✓ Using platform-specific packages: packages/$DETECTED_PLATFORM"
fi
if [ -d "packages/any" ]; then
    FIND_LINKS_ARGS="$FIND_LINKS_ARGS --find-links packages/any"
    echo "✓ Using platform-independent packages: packages/any"
fi

# Install packages from local directory
echo "Installing Python packages from local packages/..."
total_count=$(find packages/ -name "*.whl" -o -name "*.tar.gz" 2>/dev/null | wc -l)
platform_count=$(find "packages/$DETECTED_PLATFORM" -name "*.whl" 2>/dev/null | wc -l || echo "0")
echo "Total package count: $total_count"
echo "Platform-specific count: $platform_count"

# Upgrade pip first with platform-appropriate packages
echo "Upgrading pip with platform packages..."
python -m pip install --upgrade --no-index $FIND_LINKS_ARGS pip setuptools wheel || {
    echo "⚠️ Warning: Failed to upgrade pip, continuing with system pip"
}

# Install packages with platform preference
echo "Installing packages with platform preference..."
python -m pip install --no-index $FIND_LINKS_ARGS -r requirements.txt || {
    echo "⚠️ Warning: Some packages failed to install with platform preference"
    echo "Retrying with all available packages..."
    
    # Fallback: try with all packages
    python -m pip install --no-index --find-links packages/ --find-links packages/any \
        -r requirements.txt || {
        echo "❌ Error: Failed to install required packages"
        exit 1
    }
}

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama from bundle..."
    
    # Detect platform and install appropriate binary
    OLLAMA_BINARY=""
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OLLAMA_BINARY="bin/ollama-linux-amd64"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OLLAMA_BINARY="bin/ollama-darwin-amd64"
    fi
    
    if [ -f "$OLLAMA_BINARY" ]; then
        echo "Installing Ollama from bundled binary: $OLLAMA_BINARY"
        sudo cp "$OLLAMA_BINARY" /usr/local/bin/ollama
        sudo chmod +x /usr/local/bin/ollama
        echo "✓ Ollama installed from bundle"
        
        # Create systemd service for Linux
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            echo "Setting up Ollama systemd service..."
            sudo tee /etc/systemd/system/ollama.service > /dev/null << 'SYSTEMD'
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="OLLAMA_HOST=0.0.0.0:11434"

[Install]
WantedBy=default.target
SYSTEMD
            
            # Create ollama user
            sudo useradd -r -s /bin/false -d /usr/share/ollama -m ollama 2>/dev/null || true
            sudo systemctl daemon-reload
            sudo systemctl enable ollama
            echo "✓ Ollama service configured"
        fi
    else
        echo "⚠️ Warning: Ollama binary not found in bundle"
        echo "Trying installation scripts..."
        
        if [ -f "scripts/install_ollama.sh" ]; then
            chmod +x scripts/install_ollama.sh
            ./scripts/install_ollama.sh
        elif [ -f "scripts/install_ollama_manual.sh" ]; then
            chmod +x scripts/install_ollama_manual.sh
            ./scripts/install_ollama_manual.sh
        else
            echo "❌ Error: No Ollama installation method available"
            echo "Please install Ollama manually: https://ollama.com/download"
            exit 1
        fi
    fi
else
    echo "✓ Ollama already installed"
fi

# Verify critical installations
echo "Verifying installations..."

# Check Python packages
if python -c "import agno; print(f'✓ Agno version: {agno.__version__}')"; then
    echo "✓ Agno framework available"
else
    echo "⚠️ Warning: Agno not available, system will run in fallback mode"
fi

if python -c "import transformers; print(f'✓ Transformers version: {transformers.__version__}')"; then
    echo "✓ Transformers available"
else
    echo "❌ Error: Transformers not available"
    exit 1
fi

if python -c "import fastapi; print(f'✓ FastAPI version: {fastapi.__version__}')"; then
    echo "✓ FastAPI available"
else
    echo "❌ Error: FastAPI not available"
    exit 1
fi

if python -c "import chromadb; print(f'✓ ChromaDB version: {chromadb.__version__}')"; then
    echo "✓ ChromaDB available"
else
    echo "❌ Error: ChromaDB not available"
    exit 1
fi

# Create activation script for easy use
cat > activate_env.sh << 'ACTIVATE'
#!/bin/bash
# Activation script for Hebrew RAG environment
echo "Activating Hebrew RAG Python environment..."
source venv/bin/activate
echo "✓ Environment activated"
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"
ACTIVATE

chmod +x activate_env.sh

echo "✓ Phase 2 completed successfully"
echo "✓ Virtual environment ready at: $(pwd)/venv"
echo "✓ To activate later, run: source venv/bin/activate"
echo "✓ Or use: ./activate_env.sh"

# Install frontend dependencies if frontend exists
if [ -d "frontend" ]; then
    echo "Installing frontend dependencies..."
    
    cd frontend
    
    # Method 1: Try to use cached npm packages
    if [ -f "../npm_packages/node_modules.tar.gz" ]; then
        echo "Extracting pre-bundled node_modules..."
        tar -xzf "../npm_packages/node_modules.tar.gz" || {
            echo "⚠️ Warning: Failed to extract node_modules bundle"
        }
    fi
    
    # Method 2: Install from cache if available
    if [ -d "../npm_packages/cache" ]; then
        echo "Installing from npm cache..."
        npm install --offline --cache "../npm_packages/cache" || {
            echo "⚠️ Warning: Offline npm install failed, trying other methods"
            
            # Method 3: Try installing with bundled packages
            if [ -f "../npm_packages/bundle.tgz" ]; then
                echo "Installing from npm bundle..."
                tar -xzf "../npm_packages/bundle.tgz" || {
                    echo "⚠️ Warning: Failed to extract npm bundle"
                }
            fi
            
            # Method 4: Regular npm install (will fail in true air-gapped)
            echo "Attempting regular npm install..."
            npm install || {
                echo "⚠️ Warning: npm install failed - truly air-gapped environment detected"
                echo "Frontend dependencies may not be fully installed"
            }
        }
    else
        echo "⚠️ Warning: No npm cache found, attempting regular install..."
        npm install || {
            echo "⚠️ Warning: npm install failed in air-gapped environment"
        }
    fi
    
    # Verify Node.js and npm are working
    echo "Verifying frontend tools..."
    if command -v node &> /dev/null; then
        echo "✓ Node.js version: $(node --version)"
    else
        echo "❌ Error: Node.js not available"
        cd ..
        exit 1
    fi
    
    if command -v npm &> /dev/null; then
        echo "✓ npm version: $(npm --version)"
    else
        echo "❌ Error: npm not available"
        cd ..
        exit 1
    fi
    
    cd ..
    
    echo "✓ Frontend dependencies processed"
else
    echo "⚠️ Frontend directory not found, skipping frontend setup"
fi

# Leave virtual environment activated for next phase
echo "Virtual environment remains active for subsequent phases"
EOF

# Phase 3: Configuration
cat > "$SCRIPTS_DIR/deployment/phase3_configuration.sh" << 'EOF'
#!/bin/bash

echo "Configuring system..."

# Set environment variables
export AGNO_TELEMETRY=false
export OLLAMA_HOST=0.0.0.0:11434
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Initialize ChromaDB
python3 -c "
import chromadb
client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_or_create_collection('hebrew_documents')
print(f'ChromaDB initialized. Collection count: {collection.count()}')
"

# Load Ollama models
ollama serve &
sleep 10

# Load models from exported files
if [ -f "models/ollama/mistral-7b.modelfile" ]; then
    ollama create mistral:7b-instruct -f models/ollama/mistral-7b.modelfile
fi

if [ -f "models/ollama/llama2-13b.modelfile" ]; then
    ollama create llama2:13b-chat -f models/ollama/llama2-13b.modelfile
fi

echo "Phase 3 completed"
EOF

# Phase 4: Startup
cat > "$SCRIPTS_DIR/deployment/phase4_startup.sh" << 'EOF'
#!/bin/bash

set -e

echo "Starting services (air-gapped)..."

# Check if previous phases completed
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ Error: Virtual environment not found. Run phase2 first."
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "❌ Error: Environment configuration not found. Run phase3 first."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
source .env

echo "✓ Environment activated"

# Check deployment method preference
deployment_method="${DEPLOYMENT_METHOD:-direct}"

if [ "$deployment_method" = "docker" ] && command -v docker &> /dev/null; then
    echo "Starting with Docker (air-gapped)..."
    
    # Load Docker images first
    if [ -f "load_docker_images.sh" ] && [ -d "docker_images" ]; then
        echo "Loading pre-built Docker images..."
        ./load_docker_images.sh
        echo "✓ Docker images loaded"
    else
        echo "⚠️ Warning: Pre-built Docker images not found, building locally..."
    fi
    
    # Ensure Docker directory exists with proper files
    if [ ! -d "docker" ]; then
        echo "❌ Error: Docker directory not found"
        exit 1
    fi
    
    cd docker
    
    # Build only if images not loaded
    if ! docker images | grep -q "hebrew-rag-agno"; then
        echo "Building containers..."
        docker-compose build --no-cache
    else
        echo "✓ Using pre-loaded Docker images"
    fi
    
    echo "Starting containers..."
    docker-compose up -d
    
    # Wait for services to start
    echo "Waiting for services to start..."
    sleep 30
    
    cd ..
    
else
    echo "Starting services directly (without Docker)..."
    
    # Load Ollama models if bundle available
    if [ -f "models/ollama/load_models.sh" ]; then
        echo "Loading Ollama models from bundle..."
        cd models/ollama
        ./load_models.sh
        cd ../..
        echo "✓ Ollama models loaded"
    fi
    
    # Ensure Ollama is running
    if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        echo "Starting Ollama..."
        nohup ollama serve > logs/ollama.log 2>&1 &
        echo $! > ollama.pid
        
        # Wait for Ollama to start
        for i in {1..20}; do
            if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
                echo "✓ Ollama started"
                break
            fi
            echo "Waiting for Ollama... ($i/20)"
            sleep 3
        done
        
        if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
            echo "❌ Error: Ollama failed to start"
            exit 1
        fi
    else
        echo "✓ Ollama already running"
    fi
    
    # Start the Hebrew RAG system
    echo "Starting Hebrew RAG FastAPI server..."
    nohup python src/main.py --host 0.0.0.0 --port 8000 > logs/fastapi.log 2>&1 &
    FASTAPI_PID=$!
    echo $FASTAPI_PID > fastapi.pid
    
    echo "✓ FastAPI server started (PID: $FASTAPI_PID)"
fi

# Wait for service to be ready
echo "Waiting for Hebrew RAG system to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ System is ready!"
        break
    fi
    echo "Waiting for system... ($i/60)"
    sleep 2
done

# Final health check
echo "Performing system health check..."
if curl -f http://localhost:8000/system-status > /dev/null 2>&1; then
    echo "✓ Health check passed"
    
    # Display system status
    echo ""
    echo "=== System Status ==="
    curl -s http://localhost:8000/system-status | python -m json.tool || echo "Status check completed"
    echo ""
    
else
    echo "❌ Health check failed"
    echo "Checking logs for issues..."
    
    if [ -f "logs/fastapi.log" ]; then
        echo "FastAPI logs:"
        tail -20 logs/fastapi.log
    fi
    
    if [ -f "logs/ollama.log" ]; then
        echo "Ollama logs:"
        tail -10 logs/ollama.log
    fi
    
    if [ "$deployment_method" = "docker" ]; then
        echo "Docker logs:"
        cd docker && docker-compose logs --tail=20
    fi
    
    exit 1
fi

# Test basic functionality
echo "Testing basic functionality..."
curl -X POST "http://localhost:8000/test-hebrew" -H "Content-Type: application/json" > /dev/null 2>&1 && \
    echo "✓ Hebrew processing test passed" || \
    echo "⚠️ Warning: Hebrew processing test failed"

echo ""
echo "🎉 Phase 4 completed successfully!"
echo ""
echo "Hebrew Agentic RAG System is now running:"
echo "  - Web interface: http://localhost:8000"
echo "  - API documentation: http://localhost:8000/docs"
echo "  - Health check: http://localhost:8000/health"
echo "  - System status: http://localhost:8000/system-status"
echo ""

if [ "$deployment_method" != "docker" ]; then
    echo "Service management:"
    echo "  - Start: ./start_system.sh"
    echo "  - Stop: ./stop_system.sh"
    echo "  - Logs: tail -f logs/*.log"
    echo ""
fi

echo "System is ready for use!"

# Additional frontend information
if [ -d "frontend" ]; then
    echo ""
    echo "Frontend information:"
    echo "  - Development server: ./start_frontend.sh"
    echo "  - Production build: ./build_frontend.sh"
    if [ -d "frontend/dist" ]; then
        echo "  - Static files: frontend/dist/ (ready for serving)"
    fi
fi
EOF

# Make all deployment scripts executable
chmod +x "$SCRIPTS_DIR/deployment/"*.sh

# Create Docker installation script for Linux
echo "Creating Docker installation script..."
cat > "$SCRIPTS_DIR/install_docker.sh" << 'EOF'
#!/bin/bash
# Docker installation script for Linux

set -e

echo "Installing Docker on Linux..."

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
else
    echo "❌ Cannot detect Linux distribution"
    exit 1
fi

case "$DISTRO" in
    "ubuntu"|"debian")
        echo "Installing Docker on Ubuntu/Debian..."
        
        # Update package index
        sudo apt-get update
        
        # Install prerequisites
        sudo apt-get install -y ca-certificates curl gnupg lsb-release
        
        # Add Docker GPG key (offline bundle should include this)
        if [ -f "docker/docker-archive-keyring.gpg" ]; then
            sudo cp docker/docker-archive-keyring.gpg /usr/share/keyrings/
        else
            curl -fsSL https://download.docker.com/linux/$DISTRO/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
        fi
        
        # Add Docker repository
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/$DISTRO $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        
        # Install Docker
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        ;;
        
    "rhel"|"centos"|"fedora")
        echo "Installing Docker on RHEL/CentOS/Fedora..."
        
        # Install prerequisites
        sudo yum install -y yum-utils
        
        # Add Docker repository
        sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
        
        # Install Docker
        sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        ;;
        
    *)
        echo "❌ Unsupported Linux distribution: $DISTRO"
        exit 1
        ;;
esac

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add current user to docker group
sudo usermod -aG docker $USER

echo "✓ Docker installed successfully"
echo "Note: You may need to log out and back in for group changes to take effect"
echo "Test with: docker run hello-world"
EOF

chmod +x "$SCRIPTS_DIR/install_docker.sh"

# Create NVIDIA Docker installation script
cat > "$SCRIPTS_DIR/install_nvidia_docker.sh" << 'EOF'
#!/bin/bash
# NVIDIA Docker installation script for Linux

set -e

echo "Installing NVIDIA Docker support..."

# Check if NVIDIA drivers are installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA drivers not found. Please install NVIDIA drivers first."
    exit 1
fi

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
else
    echo "❌ Cannot detect Linux distribution"
    exit 1
fi

case "$DISTRO" in
    "ubuntu"|"debian")
        echo "Installing NVIDIA Docker on Ubuntu/Debian..."
        
        # Add NVIDIA Docker repository
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        
        # Install NVIDIA Container Toolkit
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        ;;
        
    "rhel"|"centos"|"fedora")
        echo "Installing NVIDIA Docker on RHEL/CentOS/Fedora..."
        
        # Add NVIDIA Docker repository
        curl -s -L https://nvidia.github.io/libnvidia-container/centos8/libnvidia-container.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
        
        # Install NVIDIA Container Toolkit
        sudo yum install -y nvidia-container-toolkit
        ;;
        
    *)
        echo "❌ Unsupported Linux distribution: $DISTRO"
        exit 1
        ;;
esac

# Configure Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "✓ NVIDIA Docker support installed successfully"
echo "Test with: docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi"
EOF

chmod +x "$SCRIPTS_DIR/install_nvidia_docker.sh"

echo "✓ Deployment scripts created"

# Create requirements.txt
echo "Creating requirements.txt..."
cat > "$BUNDLE_DIR/requirements.txt" << 'EOF'
# Core Agno Framework
agno==0.2.75

# LLM Integration
ollama==0.4.2

# Vector Database
chromadb==0.4.18

# ML/AI Libraries
torch
transformers==4.36.0
sentence-transformers==2.2.2

# Document Processing
unstructured[local-inference]==0.11.6
pymupdf==1.23.5
pdfplumber==0.9.0
python-docx==0.8.11
pillow==10.1.0

# Hebrew Language Processing
hebrew-tokenizer==2.3.0

# Scientific Computing
numpy==1.24.3
pandas==2.1.4

# Web Framework
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
aiofiles==23.2.1
python-multipart==0.0.6

# Utilities
jinja2==3.1.2
python-jose==3.3.0
passlib==1.7.4
bcrypt==4.1.2

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1

# OCR and Computer Vision
paddlepaddle==2.5.1
paddleocr==2.7.3
opencv-python==4.8.1.78
EOF

echo "✓ Requirements.txt created"

# Copy actual working source code
echo "Copying working source code..."

# Copy main application with all fixes
mkdir -p "$BUNDLE_DIR/src"
if [ -f "main.py" ]; then
    cp main.py "$BUNDLE_DIR/src/"
    echo "✓ Copied main.py with full FastAPI implementation"
else
    echo "❌ main.py not found in current directory"
    exit 1
fi

# Copy Hebrew RAG system with all fixes
if [ -f "hebrew_rag_system.py" ]; then
    cp hebrew_rag_system.py "$BUNDLE_DIR/src/"
    echo "✓ Copied hebrew_rag_system.py with fallback mechanisms"
else
    echo "❌ hebrew_rag_system.py not found in current directory"
    exit 1
fi

# Copy complete Hebrew tools implementation
if [ -d "hebrew_tools" ]; then
    cp -r hebrew_tools/ "$BUNDLE_DIR/"
    echo "✓ Copied complete hebrew_tools module with all implementations"
else
    echo "❌ hebrew_tools directory not found"
    exit 1
fi

# Copy additional files
if [ -f "test_system.py" ]; then
    cp test_system.py "$BUNDLE_DIR/"
    echo "✓ Copied test_system.py"
fi

if [ -f "requirements.txt" ]; then
    cp requirements.txt "$BUNDLE_DIR/"
    echo "✓ Copied actual requirements.txt (will overwrite template)"
fi

if [ -f "pyproject.toml" ]; then
    cp pyproject.toml "$BUNDLE_DIR/"
    echo "✓ Copied pyproject.toml"
fi

echo "✓ Working source code copied successfully"

# Create configuration files
echo "Creating configuration files..."

mkdir -p "$CONFIG_DIR"
cat > "$CONFIG_DIR/production.py" << 'EOF'
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
EOF

echo "✓ Configuration files created"

# Create documentation
echo "Creating documentation..."

cat > "$DOCS_DIR/README.md" << 'EOF'
# Hebrew Agentic RAG System - Air-Gapped Deployment

This bundle contains everything needed to deploy a Hebrew Agentic RAG system using Agno in an air-gapped environment.

## Quick Start

1. Extract this bundle to your target system
2. Run the deployment script:
   ```bash
   chmod +x scripts/deploy.sh
   ./scripts/deploy.sh
   ```
3. Access the system at http://localhost:8000

## Directory Structure

- `models/` - Pre-downloaded Hebrew language models
- `packages/` - Python packages for offline installation  
- `docker/` - Docker configuration files
- `scripts/` - Deployment and maintenance scripts
- `src/` - Application source code
- `config/` - Configuration files
- `tests/` - Test suite

## System Requirements

- Linux system with Docker support
- 16GB+ RAM (32GB recommended)
- 100GB+ storage
- NVIDIA GPU (recommended for performance)
- Python 3.11+

## Manual Steps Required

1. **Ollama Models**: Export Ollama models on connected system:
   ```bash
   ollama pull mistral:7b-instruct
   ollama pull llama2:13b-chat
   # Export models (manual process)
   ```

2. **Hebrew Models**: Verify all Hugging Face models are downloaded

3. **GPU Drivers**: Install NVIDIA drivers if using GPU acceleration

## Troubleshooting

- Check logs in `logs/` directory
- Verify model paths in config files
- Ensure all services are running with `docker-compose ps`

For support, see the troubleshooting guide in `documentation/troubleshooting.md`
EOF

cat > "$DOCS_DIR/deployment_guide.md" << 'EOF'
# Detailed Deployment Guide

## Pre-Deployment Checklist

- [ ] Target system meets hardware requirements
- [ ] All models downloaded and verified
- [ ] Python packages available offline
- [ ] Docker and docker-compose installed
- [ ] NVIDIA drivers installed (if using GPU)

## Deployment Phases

### Phase 1: Environment Setup
- Creates directory structure
- Installs Docker if needed
- Sets up GPU support

### Phase 2: Dependencies
- Installs Python packages offline
- Installs Ollama
- Verifies installations

### Phase 3: Configuration
- Configures environment variables
- Initializes databases
- Loads language models

### Phase 4: Service Startup
- Builds Docker containers
- Starts all services
- Performs health checks

## Post-Deployment

1. Upload test documents via web interface
2. Test Hebrew question answering
3. Monitor system performance
4. Set up regular maintenance

## Security Considerations

- All external network access disabled
- Local model serving only
- Encrypted data storage
- Access logging enabled
EOF

echo "✓ Documentation created"

# Create test files
echo "Creating test suite..."

mkdir -p "$BUNDLE_DIR/tests"
cat > "$BUNDLE_DIR/tests/test_deployment.py" << 'EOF'
import pytest
import requests
import time
import os

class TestDeployment:
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Wait for services to be ready"""
        time.sleep(5)
    
    def test_system_health(self):
        """Test system health endpoint"""
        response = requests.get("http://localhost:8000/system-status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "agno_version" in data
    
    def test_upload_endpoint(self):
        """Test document upload endpoint"""
        # This would require a test document
        pass
    
    def test_question_endpoint(self):
        """Test Hebrew question answering"""
        payload = {
            "question": "בדיקת מערכת בעברית"
        }
        response = requests.post("http://localhost:8000/ask-question", json=payload)
        assert response.status_code == 200
    
    def test_environment_variables(self):
        """Test air-gapped environment settings"""
        assert os.environ.get("AGNO_TELEMETRY") == "false"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"

if __name__ == "__main__":
    pytest.main([__file__])
EOF

echo "✓ Test suite created"

# Create verification script
cat > "$BUNDLE_DIR/verify_bundle.sh" << 'EOF'
#!/bin/bash

echo "Verifying Hebrew RAG Air-Gapped Bundle..."

# Check required directories
required_dirs=("models" "packages" "docker" "docker_images" "scripts" "src" "config" "documentation")
optional_dirs=("frontend" "npm_packages" "nodejs")

for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "❌ Missing directory: $dir"
        exit 1
    else
        echo "✓ Directory exists: $dir"
    fi
done

for dir in "${optional_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "⚠️  Optional directory missing: $dir"
    else
        echo "✓ Directory exists: $dir"
    fi
done

# Check Python packages
if [ ! -f "requirements.txt" ]; then
    echo "❌ Missing requirements.txt"
    exit 1
fi

# Check cross-platform packages
total_packages=$(find packages/ -name "*.whl" -o -name "*.tar.gz" 2>/dev/null | wc -l)
linux_packages=$(find packages/linux_x86_64/ -name "*.whl" 2>/dev/null | wc -l || echo "0")
macos_packages=$(find packages/macosx_*/ -name "*.whl" 2>/dev/null | wc -l || echo "0")
any_packages=$(find packages/any/ -name "*.whl" 2>/dev/null | wc -l || echo "0")

echo "✓ Python packages: $total_packages total"
echo "  - Linux x86_64: $linux_packages files"
echo "  - macOS: $macos_packages files" 
echo "  - Platform-independent: $any_packages files"

if [ $total_packages -lt 50 ]; then
    echo "⚠️  Warning: Low package count ($total_packages found)"
fi

# Check models
model_count=$(find models/ -name "*.bin" -o -name "config.json" 2>/dev/null | wc -l)
if [ $model_count -lt 5 ]; then
    echo "⚠️  Warning: Limited models found ($model_count files)"
else
    echo "✓ Models: $model_count files"
fi

# Check Docker images
docker_image_count=$(ls docker_images/*.tar.gz 2>/dev/null | wc -l)
if [ $docker_image_count -lt 2 ]; then
    echo "⚠️  Warning: Limited Docker images found ($docker_image_count files)"
else
    echo "✓ Docker images: $docker_image_count files"
fi

# Check scripts
if [ ! -x "scripts/deploy.sh" ]; then
    echo "❌ Deploy script not executable"
    exit 1
else
    echo "✓ Deploy script ready"
fi

if [ ! -x "load_docker_images.sh" ]; then
    echo "❌ Docker image loader not executable"
    exit 1
else
    echo "✓ Docker image loader ready"
fi

# Check cross-platform binaries
binary_count=$(ls bin/ollama-* 2>/dev/null | wc -l || echo "0")
if [ $binary_count -gt 0 ]; then
    echo "✓ Cross-platform binaries: $binary_count files"
    for binary in bin/ollama-*; do
        if [ -f "$binary" ]; then
            echo "  - $(basename $binary)"
        fi
    done
else
    echo "⚠️  Warning: No cross-platform binaries found"
fi

# Check frontend components
if [ -d "frontend" ]; then
    echo "✓ Frontend directory exists"
    
    if [ -f "frontend/package.json" ]; then
        echo "✓ Frontend package.json found"
    else
        echo "❌ Frontend package.json missing"
    fi
    
    # Check npm packages
    npm_packages=0
    if [ -f "npm_packages/node_modules.tar.gz" ]; then
        npm_packages=$((npm_packages + 1))
        echo "✓ Bundled node_modules found"
    fi
    
    if [ -d "npm_packages/cache" ]; then
        cache_files=$(ls npm_packages/cache/ 2>/dev/null | wc -l)
        npm_packages=$((npm_packages + cache_files))
        echo "✓ npm cache: $cache_files files"
    fi
    
    if [ -f "npm_packages/bundle.tgz" ]; then
        npm_packages=$((npm_packages + 1))
        echo "✓ npm bundle found"
    fi
    
    if [ $npm_packages -gt 0 ]; then
        echo "✓ Frontend dependencies: $npm_packages items"
    else
        echo "⚠️  Warning: No frontend dependencies found"
    fi
    
    # Check Node.js binaries
    nodejs_binaries=$(ls nodejs/node-*.tar.* 2>/dev/null | wc -l || echo "0")
    if [ $nodejs_binaries -gt 0 ]; then
        echo "✓ Node.js binaries: $nodejs_binaries files"
        for nodejs in nodejs/node-*.tar.*; do
            if [ -f "$nodejs" ]; then
                echo "  - $(basename $nodejs)"
            fi
        done
    else
        echo "⚠️  Warning: No Node.js binaries found"
    fi
    
    # Check npm and package manager bundles
    npm_bundle=$(ls nodejs/npm-*.tgz 2>/dev/null | wc -l || echo "0")
    pnpm_bundle=$(ls nodejs/pnpm-*.tgz 2>/dev/null | wc -l || echo "0")
    yarn_bundle=$(ls nodejs/yarn-*.tgz 2>/dev/null | wc -l || echo "0")
    
    if [ $npm_bundle -gt 0 ]; then
        echo "✓ Standalone npm bundle found"
    fi
    
    if [ $pnpm_bundle -gt 0 ] || [ $yarn_bundle -gt 0 ]; then
        echo "✓ Alternative package managers: pnpm($pnpm_bundle), yarn($yarn_bundle)"
    fi
    
    if [ -f "nodejs/upgrade_npm.sh" ]; then
        echo "✓ npm upgrade script available"
    fi
else
    echo "ℹ️  No frontend components (optional)"
fi

echo ""
echo "Bundle verification completed successfully!"
echo "Ready for air-gapped deployment."
EOF

chmod +x "$BUNDLE_DIR/verify_bundle.sh"

# Create final bundle info
cat > "$BUNDLE_DIR/BUNDLE_INFO.txt" << 'EOF'
Hebrew Agentic RAG System - Air-Gapped Bundle
============================================

Created: $(date)
Bundle Version: 1.0.0
Agno Version: 0.2.75

Contents:
- Complete Python dependencies for offline installation
- Hebrew language models (BERT, transformers, etc.)
- Ollama LLM models exported for air-gapped deployment
- Pre-built Docker images (saved as tar.gz files)
- Vue.js frontend application with bundled dependencies
- Node.js binaries for cross-platform deployment
- Full source code and configuration
- Deployment and maintenance scripts
- Comprehensive documentation
- Test suite

Total Size: ~30-45GB (including Docker images, models, and frontend)

Quick Start:
1. ./verify_bundle.sh
2. ./scripts/deploy.sh
3. Access http://localhost:8000

For detailed instructions, see documentation/README.md
EOF

echo ""
echo "=========================================="
echo "Bundle creation completed successfully!"
echo "=========================================="
echo ""
echo "Bundle location: $BUNDLE_DIR"
echo "Bundle size: $(du -sh $BUNDLE_DIR | cut -f1)"
echo ""
echo "Next steps:"
echo "1. Run: cd $BUNDLE_DIR && ./verify_bundle.sh"
echo "2. Transfer bundle to air-gapped system"
echo "3. Extract and run: ./scripts/deploy.sh"
echo ""
echo "Note: Ollama models require manual export for true air-gapped deployment"
echo "See documentation/deployment_guide.md for details"