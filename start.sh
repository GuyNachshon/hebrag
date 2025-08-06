#!/bin/bash

# Hebrew Agentic RAG System - Container Start Script

set -e

echo "Starting Hebrew Agentic RAG System..."

# Configure environment
export AGNO_TELEMETRY=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/app

# Start Ollama in background (if installed)
if command -v ollama &> /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    OLLAMA_PID=$!
    
    # Wait for Ollama to be ready
    echo "Waiting for Ollama to be ready..."
    max_attempts=30
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
            echo "Ollama is ready"
            break
        fi
        echo "Waiting for Ollama... (attempt $((attempt+1))/$max_attempts)"
        sleep 2
        attempt=$((attempt+1))
    done
    
    if [ $attempt -eq $max_attempts ]; then
        echo "Warning: Ollama did not start within expected time"
    fi
    
    # Load models if model files exist
    if [ -f "./models/mistral-7b.modelfile" ]; then
        echo "Loading Mistral 7B model..."
        ollama create mistral:7b-instruct -f ./models/mistral-7b.modelfile || echo "Failed to load Mistral model"
    fi
    
    if [ -f "./models/llama2-13b.modelfile" ]; then
        echo "Loading Llama2 13B model..."
        ollama create llama2:13b-chat -f ./models/llama2-13b.modelfile || echo "Failed to load Llama2 model"
    fi
    
    # List available models
    echo "Available Ollama models:"
    ollama list || echo "No models available"
else
    echo "Ollama not found - running without local LLM"
fi

# Initialize database directories
echo "Initializing database directories..."
mkdir -p chroma_db documents logs cache

# Test Python imports
echo "Testing Python imports..."
python -c "
try:
    import agno
    print(f'✓ Agno version: {agno.__version__}')
except ImportError as e:
    print(f'⚠ Agno not available: {e}')

try:
    import chromadb
    print('✓ ChromaDB available')
except ImportError as e:
    print(f'⚠ ChromaDB not available: {e}')

try:
    import transformers
    print(f'✓ Transformers version: {transformers.__version__}')
except ImportError as e:
    print(f'⚠ Transformers not available: {e}')

try:
    from hebrew_tools import HebrewDocumentProcessor
    print('✓ Hebrew tools available')
except ImportError as e:
    print(f'⚠ Hebrew tools not available: {e}')
"

# Test ChromaDB initialization
echo "Testing ChromaDB initialization..."
python -c "
try:
    import chromadb
    client = chromadb.PersistentClient(path='./chroma_db')
    collection = client.get_or_create_collection('hebrew_documents')
    print(f'✓ ChromaDB initialized. Collection count: {collection.count()}')
except Exception as e:
    print(f'⚠ ChromaDB initialization failed: {e}')
"

# Validate configuration
echo "Validating configuration..."
python -c "
try:
    from config import Config
    if hasattr(Config, 'validate_config'):
        if Config.validate_config():
            print('✓ Configuration validated successfully')
        else:
            print('⚠ Configuration validation failed')
    else:
        print('✓ Configuration loaded')
except Exception as e:
    print(f'⚠ Configuration error: {e}')
"

# Set up logging
echo "Setting up logging..."
python -c "
try:
    from config import Config
    if hasattr(Config, 'setup_logging'):
        Config.setup_logging()
        print('✓ Logging configured')
    else:
        import logging
        logging.basicConfig(level=logging.INFO)
        print('✓ Basic logging configured')
except Exception as e:
    print(f'⚠ Logging setup failed: {e}')
"

# Start the FastAPI application
echo "Starting FastAPI application..."

# Determine the number of workers based on CPU cores
if [ -z "$API_WORKERS" ]; then
    CPU_CORES=$(nproc)
    API_WORKERS=$((CPU_CORES > 4 ? 4 : CPU_CORES))
    echo "Auto-detected $CPU_CORES CPU cores, using $API_WORKERS workers"
fi

# Function to handle shutdown
shutdown_handler() {
    echo "Received shutdown signal..."
    
    # Kill Ollama if it's running
    if [ ! -z "$OLLAMA_PID" ]; then
        echo "Stopping Ollama..."
        kill $OLLAMA_PID 2>/dev/null || true
    fi
    
    # Kill uvicorn if it's running
    if [ ! -z "$UVICORN_PID" ]; then
        echo "Stopping FastAPI application..."
        kill $UVICORN_PID 2>/dev/null || true
    fi
    
    echo "Shutdown complete"
    exit 0
}

# Set up signal handlers
trap shutdown_handler SIGTERM SIGINT

# Start FastAPI with uvicorn
echo "Starting FastAPI with $API_WORKERS workers..."

if [ "$API_WORKERS" -eq 1 ]; then
    # Single worker mode
    uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --log-level info \
        --access-log \
        --no-use-colors &
    UVICORN_PID=$!
else
    # Multi-worker mode
    uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers $API_WORKERS \
        --worker-class uvicorn.workers.UvicornWorker \
        --log-level info \
        --access-log \
        --no-use-colors &
    UVICORN_PID=$!
fi

echo "FastAPI application started with PID $UVICORN_PID"
echo "System is ready! Access the API at http://localhost:8000"
echo "API documentation available at http://localhost:8000/docs"

# Wait for the application to exit
wait $UVICORN_PID