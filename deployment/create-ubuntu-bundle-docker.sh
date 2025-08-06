#!/bin/bash

# Create Ubuntu Bundle using Docker for True Ubuntu Compatibility
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUNDLE_NAME="hebrag-ubuntu-$(date +%Y%m%d-%H%M%S)"

echo "🐧 Hebrew RAG System - Ubuntu Bundle via Docker"
echo "=============================================="
echo "Creating truly Ubuntu-compatible bundle using Docker"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found! Please install Docker to create Ubuntu-compatible bundles"
    echo ""
    echo "Alternatives:"
    echo "1. Run ./prepare-ubuntu-bundle.sh on an Ubuntu system"
    echo "2. Install Docker and re-run this script"
    echo "3. Use the generic bundle (may have compatibility issues)"
    exit 1
fi

# Create temporary directory for Docker context
TEMP_DIR="/tmp/ubuntu-bundle-build"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Create Dockerfile for Ubuntu package building
echo "🐳 Creating Ubuntu Docker environment..."
cat > "$TEMP_DIR/Dockerfile" << 'EOF'
FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    wget \
    git \
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
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /build

# Copy requirements
COPY requirements-ubuntu.txt .

# Create script to download packages
COPY download-packages.sh .
RUN chmod +x download-packages.sh

# Run the download
RUN ./download-packages.sh
EOF

# Create requirements for Docker build
cat > "$TEMP_DIR/requirements-ubuntu.txt" << 'REQS'
# Ubuntu-compatible Hebrew RAG requirements
agno>=1.7.7
chromadb>=0.6.0
ollama>=0.5.2
sentence-transformers>=5.0.0
transformers>=4.40.0
pymupdf>=1.23.0
pdfplumber>=0.11.0
python-docx>=1.1.0
pillow>=10.0.0
hebrew-tokenizer>=2.3.0
fastapi>=0.100.0
uvicorn>=0.20.0
pydantic>=2.0.0
aiofiles>=23.0.0
python-multipart>=0.0.6
numpy>=1.24.0
pandas>=2.0.0
requests>=2.31.0
jinja2>=3.1.0
python-jose>=3.3.0
passlib>=1.7.4
bcrypt>=4.0.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
REQS

# Create download script for Docker
cat > "$TEMP_DIR/download-packages.sh" << 'DOWNLOAD'
#!/bin/bash
set -e

echo "📦 Downloading Ubuntu packages inside Docker..."

# Create wheels directory
mkdir -p /build/wheels

# Upgrade pip
python3 -m pip install --upgrade pip

# Download packages as wheels
python3 -m pip download --dest /build/wheels -r requirements-ubuntu.txt

# Download PyTorch CPU version
echo "🔥 Downloading PyTorch CPU for Ubuntu..."
python3 -m pip download --dest /build/wheels \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

echo "✅ Ubuntu package download complete"
ls -la /build/wheels/ | wc -l
echo "packages downloaded"
DOWNLOAD

# Build Docker image and extract packages
echo "🔨 Building Ubuntu environment in Docker..."
cd "$TEMP_DIR"
docker build -t ubuntu-hebrag-builder .

echo "📤 Extracting Ubuntu packages from Docker..."
CONTAINER_ID=$(docker create ubuntu-hebrag-builder)
docker cp "$CONTAINER_ID:/build/wheels" "$TEMP_DIR/"
docker rm "$CONTAINER_ID"

# Create the actual bundle
echo "📦 Creating Ubuntu bundle..."
BUNDLE_DIR="/tmp/$BUNDLE_NAME"
mkdir -p "$BUNDLE_DIR"

# Copy source code
echo "📋 Copying source code..."
rsync -av --exclude='.venv' --exclude='chroma_db' --exclude='__pycache__' \
    --exclude='*.pyc' --exclude='.git' --exclude='data/documents/*' \
    --exclude='models/*' --exclude='node_modules' \
    "$PROJECT_ROOT/" "$BUNDLE_DIR/"

# Copy Ubuntu wheels
echo "📦 Adding Ubuntu-built packages..."
cp -r "$TEMP_DIR/wheels" "$BUNDLE_DIR/deployment/"

# Copy Ubuntu requirements
cp "$TEMP_DIR/requirements-ubuntu.txt" "$BUNDLE_DIR/deployment/"

# Create bundle info
WHEEL_COUNT=$(ls -1 "$BUNDLE_DIR/deployment/wheels/" | wc -l)
echo "📊 Ubuntu Bundle Statistics:"
echo "   📦 Packages: $WHEEL_COUNT wheels"
echo "   💾 Size: $(du -sh "$BUNDLE_DIR" | cut -f1)"

# Create the bundle archive
echo "🗜️ Creating Ubuntu bundle archive..."
cd /tmp
tar -czf "${BUNDLE_NAME}.tar.gz" "$BUNDLE_NAME/"

# Cleanup
rm -rf "$TEMP_DIR"
docker rmi ubuntu-hebrag-builder || true

echo ""
echo "🎉 UBUNTU BUNDLE CREATED WITH DOCKER!"
echo "====================================="
echo ""
echo "📦 Bundle: ${BUNDLE_NAME}.tar.gz"
echo "📁 Location: /tmp/${BUNDLE_NAME}.tar.gz"  
echo "📊 Size: $(du -sh "/tmp/${BUNDLE_NAME}.tar.gz" | cut -f1)"
echo "📦 Ubuntu packages: $WHEEL_COUNT"
echo ""
echo "🚀 Deploy on Ubuntu:"
echo "   scp /tmp/${BUNDLE_NAME}.tar.gz user@ubuntu-server:"
echo "   tar -xzf ${BUNDLE_NAME}.tar.gz"
echo "   cd $BUNDLE_NAME"
echo "   sudo ./deployment/install-ubuntu-deps.sh"
echo "   ./deployment/deploy-ubuntu.sh"
echo ""
echo "✅ This bundle was built ON Ubuntu (via Docker)"
echo "✅ Packages are truly Ubuntu-compatible"
echo "✅ Ready for air-gapped Ubuntu deployment"