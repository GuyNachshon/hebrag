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
