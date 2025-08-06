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
