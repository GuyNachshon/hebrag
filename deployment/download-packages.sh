#!/bin/bash

# Air-gapped Hebrew RAG System - Package Downloader
# Downloads all required packages as wheels for offline installation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELS_DIR="$SCRIPT_DIR/wheels"

echo "📦 Hebrew RAG System - Package Downloader for Air-Gap"
echo "====================================================="

# Create wheels directory
mkdir -p "$WHEELS_DIR"

echo "🔄 Downloading packages to: $WHEELS_DIR"
echo ""

# Core packages to download
PACKAGES=(
    "agno>=1.7.7"
    "chromadb>=1.0.15"
    "sentence-transformers>=5.0.0"
    "ollama>=0.5.2"
    "fastapi"
    "uvicorn"
    "pydantic"
    "aiofiles"
    "python-multipart"
    "pymupdf"
    "pdfplumber"
    "python-docx"
    "pillow"
    "numpy"
    "pandas"
    "torch"
    "transformers"
    "jinja2"
    "python-jose"
    "passlib"
    "bcrypt"
    "pytest"
    "pytest-asyncio"
    "requests"
    "hebrew-tokenizer"
)

echo "📋 Packages to download:"
for package in "${PACKAGES[@]}"; do
    echo "   📦 $package"
done
echo ""

# Check if pip is available
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ pip not found! Please install pip first."
    exit 1
fi

# Use pip3 if available, otherwise pip
PIP_CMD="pip3"
if ! command -v pip3 &> /dev/null; then
    PIP_CMD="pip"
fi

echo "🔧 Using $PIP_CMD for downloads"

# Download packages
echo "⬇️  Downloading packages and dependencies..."
$PIP_CMD download --dest "$WHEELS_DIR" "${PACKAGES[@]}"

echo ""
echo "📊 Download summary:"
WHEEL_COUNT=$(find "$WHEELS_DIR" -name "*.whl" | wc -l)
TAR_COUNT=$(find "$WHEELS_DIR" -name "*.tar.gz" | wc -l)
TOTAL_SIZE=$(du -sh "$WHEELS_DIR" | cut -f1)

echo "   📦 Wheels: $WHEEL_COUNT files"
echo "   📦 Source packages: $TAR_COUNT files" 
echo "   💾 Total size: $TOTAL_SIZE"

echo ""
echo "✅ Package download complete!"
echo ""
echo "📂 Downloaded packages are in: $WHEELS_DIR"
echo "🚀 Ready for air-gapped deployment"
echo ""
echo "Next steps for air-gapped environment:"
echo "1. Copy the entire project directory to target system"
echo "2. Run: sudo ./install-system-deps.sh"
echo "3. Run: ./deploy.sh"
echo ""
echo "The deployment will use these pre-downloaded packages automatically."