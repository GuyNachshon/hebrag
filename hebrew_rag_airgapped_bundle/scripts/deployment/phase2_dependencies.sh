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
