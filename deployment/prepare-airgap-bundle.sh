#!/bin/bash

# Hebrew RAG System - Air-Gap Bundle Preparation Script
# Prepares a complete bundle for air-gapped deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUNDLE_NAME="hebrag-airgap-bundle-$(date +%Y%m%d-%H%M%S)"
BUNDLE_DIR="$PROJECT_ROOT/../$BUNDLE_NAME"

echo "🎁 Hebrew RAG System - Air-Gap Bundle Preparation"
echo "================================================="
echo "Project root: $PROJECT_ROOT"
echo "Bundle directory: $BUNDLE_DIR"
echo ""

# Create bundle directory
echo "1️⃣ Creating bundle directory..."
mkdir -p "$BUNDLE_DIR"

# Copy source code
echo "2️⃣ Copying source code..."
rsync -av --exclude='.venv' --exclude='chroma_db' --exclude='__pycache__' \
    --exclude='*.pyc' --exclude='.git' --exclude='data/documents/*' \
    "$PROJECT_ROOT/" "$BUNDLE_DIR/"

# Download Python packages
echo "3️⃣ Downloading Python packages..."
cd "$SCRIPT_DIR"
./download-packages.sh

# Copy wheels to bundle
if [[ -d "wheels" ]]; then
    echo "📦 Copying pre-downloaded packages..."
    cp -r wheels "$BUNDLE_DIR/deployment/"
fi

# Download uv binary for faster package management
echo "4️⃣ Downloading uv package manager..."
UV_DIR="$BUNDLE_DIR/deployment/bin"
mkdir -p "$UV_DIR"

# Try to download uv for multiple architectures
echo "⬇️  Downloading uv for different architectures..."

# Linux x86_64
if curl -L -f -o "$UV_DIR/uv-linux-x86_64" \
    "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu" 2>/dev/null; then
    echo "✅ Downloaded uv for Linux x86_64"
    chmod +x "$UV_DIR/uv-linux-x86_64"
fi

# macOS x86_64
if curl -L -f -o "$UV_DIR/uv-macos-x86_64" \
    "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-apple-darwin" 2>/dev/null; then
    echo "✅ Downloaded uv for macOS x86_64"
    chmod +x "$UV_DIR/uv-macos-x86_64"
fi

# macOS ARM64
if curl -L -f -o "$UV_DIR/uv-macos-arm64" \
    "https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin" 2>/dev/null; then
    echo "✅ Downloaded uv for macOS ARM64"
    chmod +x "$UV_DIR/uv-macos-arm64"
fi

# Create platform detection script
cat > "$UV_DIR/install-uv.sh" << 'EOF'
#!/bin/bash
# Auto-install uv for current platform

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux-x86_64"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ "$(uname -m)" == "arm64" ]]; then
        PLATFORM="macos-arm64"
    else
        PLATFORM="macos-x86_64"
    fi
else
    echo "❌ Unsupported platform: $OSTYPE"
    exit 1
fi

UV_BINARY="$SCRIPT_DIR/uv-$PLATFORM"

if [[ -f "$UV_BINARY" ]]; then
    echo "📦 Installing uv for $PLATFORM..."
    mkdir -p "$HOME/.local/bin"
    cp "$UV_BINARY" "$HOME/.local/bin/uv"
    chmod +x "$HOME/.local/bin/uv"
    
    # Add to PATH if not already there
    if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc 2>/dev/null || true
        echo "✅ uv installed to $HOME/.local/bin/uv"
        echo "⚠️  Please restart your shell or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
    else
        echo "✅ uv installed and PATH updated"
    fi
else
    echo "❌ No uv binary available for $PLATFORM"
    echo "Will fall back to pip during deployment"
fi
EOF

chmod +x "$UV_DIR/install-uv.sh"

# Create comprehensive deployment documentation
echo "5️⃣ Creating deployment documentation..."
cat > "$BUNDLE_DIR/DEPLOY-INSTRUCTIONS.txt" << EOF
Hebrew RAG System - Air-Gapped Deployment Bundle
================================================

This bundle contains everything needed to deploy the Hebrew RAG system
in an air-gapped environment (no internet connection required).

QUICK START:
-----------
1. Transfer this entire directory to your air-gapped system
2. cd ${BUNDLE_NAME}/deployment
3. sudo ./install-system-deps.sh
4. ./deploy.sh
5. ./start_hebrag.sh test

CONTENTS:
--------
📁 ${BUNDLE_NAME}/
├── 🐍 hebrew_tools/           Hebrew processing modules
├── 📄 hebrew_rag_system.py    Main system
├── 🌐 main.py                 FastAPI server
├── 📦 deployment/
│   ├── 🔧 install-system-deps.sh    System packages installer
│   ├── 🚀 deploy.sh                 Main deployment script  
│   ├── 📦 wheels/                   Pre-downloaded Python packages
│   ├── ⚡ bin/                      uv package manager binaries
│   └── 📖 README-AIRGAP.md          Detailed instructions
├── 📋 pyproject.toml          Project configuration
└── 📋 requirements_core.txt   Python requirements

SYSTEM REQUIREMENTS:
-------------------
- Linux (Ubuntu 20.04+, CentOS 8+) or macOS
- Python 3.10+
- 8GB+ RAM (16GB recommended)
- 10GB+ free disk space
- sudo access for system package installation

DEPLOYMENT STEPS:
----------------
1. System Dependencies:
   sudo ./deployment/install-system-deps.sh

2. Deploy Application:
   ./deployment/deploy.sh

3. Test Installation:
   ./start_hebrag.sh test

4. Start API Server:
   ./start_hebrag.sh api

The system will be available at: http://localhost:8000

For detailed instructions, see: deployment/README-AIRGAP.md

Generated on: $(date)
Bundle version: $(date +%Y%m%d-%H%M%S)
EOF

# Create version info
echo "6️⃣ Adding version information..."
cat > "$BUNDLE_DIR/VERSION.txt" << EOF
Hebrew RAG System - Air-Gap Bundle
==================================

Bundle Creation Date: $(date)
Bundle Version: $(date +%Y%m%d-%H%M%S)
Python Version: $(python3 --version)
System: $(uname -a)

Included Components:
- Hebrew RAG System (main application)
- Agno Framework (agentic coordination)
- ChromaDB (vector database)
- SentenceTransformers (embeddings)
- FastAPI (web server)
- Hebrew language processing tools

Pre-downloaded Packages:
$(ls -1 "$SCRIPT_DIR/wheels/" 2>/dev/null | wc -l) Python packages
$(du -sh "$SCRIPT_DIR/wheels/" 2>/dev/null | cut -f1) total size

This bundle is ready for air-gapped deployment.
No internet connection required after deployment.
EOF

# Create final bundle archive
echo "7️⃣ Creating deployment archive..."
cd "$(dirname "$BUNDLE_DIR")"

# Create tar.gz archive
tar -czf "${BUNDLE_NAME}.tar.gz" "$BUNDLE_NAME/"

# Create zip archive (for Windows compatibility)
if command -v zip &> /dev/null; then
    zip -r "${BUNDLE_NAME}.zip" "$BUNDLE_NAME/" > /dev/null
fi

# Calculate checksums
if command -v sha256sum &> /dev/null; then
    sha256sum "${BUNDLE_NAME}.tar.gz" > "${BUNDLE_NAME}.tar.gz.sha256"
    if [[ -f "${BUNDLE_NAME}.zip" ]]; then
        sha256sum "${BUNDLE_NAME}.zip" > "${BUNDLE_NAME}.zip.sha256"
    fi
elif command -v shasum &> /dev/null; then
    shasum -a 256 "${BUNDLE_NAME}.tar.gz" > "${BUNDLE_NAME}.tar.gz.sha256"
    if [[ -f "${BUNDLE_NAME}.zip" ]]; then
        shasum -a 256 "${BUNDLE_NAME}.zip" > "${BUNDLE_NAME}.zip.sha256"
    fi
fi

echo ""
echo "🎉 AIR-GAP BUNDLE CREATION COMPLETE!"
echo "====================================="
echo ""
echo "📦 Bundle created: $BUNDLE_NAME"
echo "📁 Directory: $BUNDLE_DIR"
echo "🗜️  Archive: ${BUNDLE_NAME}.tar.gz"
if [[ -f "${BUNDLE_NAME}.zip" ]]; then
    echo "🗜️  Archive: ${BUNDLE_NAME}.zip"
fi
echo ""
echo "📊 Bundle Statistics:"
echo "   📂 Size: $(du -sh "$BUNDLE_DIR" | cut -f1)"
if [[ -f "${BUNDLE_NAME}.tar.gz" ]]; then
    echo "   🗜️  Compressed: $(du -sh "${BUNDLE_NAME}.tar.gz" | cut -f1)"
fi
echo "   📦 Python packages: $(ls -1 "$BUNDLE_DIR/deployment/wheels/" 2>/dev/null | wc -l)"
echo "   ⚡ Package managers: $(ls -1 "$BUNDLE_DIR/deployment/bin/" 2>/dev/null | grep -c uv- || echo 0) uv binaries"
echo ""
echo "🚀 Ready for Air-Gapped Deployment!"
echo ""
echo "Transfer files to air-gapped system:"
if [[ -f "${BUNDLE_NAME}.tar.gz" ]]; then
    echo "   scp ${BUNDLE_NAME}.tar.gz user@airgap-server:"
    echo "   tar -xzf ${BUNDLE_NAME}.tar.gz"
fi
if [[ -f "${BUNDLE_NAME}.zip" ]]; then
    echo "   (or use ${BUNDLE_NAME}.zip for Windows)"
fi
echo ""
echo "Then on air-gapped system:"
echo "   cd $BUNDLE_NAME/deployment"
echo "   sudo ./install-system-deps.sh"
echo "   ./deploy.sh"
echo ""
echo "Happy air-gapped Hebrew RAG processing! 🎯"