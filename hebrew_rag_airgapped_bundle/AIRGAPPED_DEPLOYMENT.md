# Air-Gapped Deployment Guide

## Overview

This Hebrew Agentic RAG system is designed for **completely air-gapped environments** - no internet access required after initial setup.

## 📦 What's Included

- ✅ **All Python packages** downloaded as wheels (no PyPI access needed)
- ✅ **Hebrew language models** (heBERT, multilingual transformers)
- ✅ **Ollama LLM models** exported for air-gapped deployment
- ✅ **Pre-built Docker images** saved as tar.gz files
- ✅ **Complete source code** with fallback mechanisms
- ✅ **Virtual environment setup** scripts
- ✅ **Docker containerization** fully air-gapped

## 🚀 Quick Start (Air-Gapped)

### Prerequisites
- Linux system with Python 3.11+
- 16GB+ RAM (32GB recommended)
- 100GB+ storage space

### Step 1: Verify Bundle
```bash
./verify_bundle.sh
```

### Step 2: Run Deployment
```bash
./scripts/deploy.sh
```

This runs all phases automatically:
- **Phase 1**: Environment setup
- **Phase 2**: Virtual environment + offline package installation
- **Phase 3**: System configuration + Ollama setup
- **Phase 4**: Service startup

### Step 3: Access System
- Web Interface: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## 🔧 Manual Phase-by-Phase Deployment

If automatic deployment fails, run phases manually:

### Phase 1: Environment Setup
```bash
./scripts/deployment/phase1_environment.sh
```

### Phase 2: Install Dependencies (Air-Gapped)
```bash
./scripts/deployment/phase2_dependencies.sh
```

**What this does:**
- Creates Python virtual environment (`venv/`)
- Installs all packages from `packages/` directory (no internet needed)
- Verifies critical dependencies
- Creates activation script

### Phase 3: Configure System
```bash
./scripts/deployment/phase3_configuration.sh
```

**What this does:**
- Activates virtual environment
- Sets air-gapped environment variables
- Initializes ChromaDB
- Starts Ollama service
- Loads available models
- Creates start/stop scripts

### Phase 4: Start Services
```bash
./scripts/deployment/phase4_startup.sh
```

**Options:**
- Direct deployment (default)
- Docker deployment (`DEPLOYMENT_METHOD=docker`)

## 🛠️ Daily Operations

### Start System
```bash
./start_system.sh
```

### Stop System
```bash
./stop_system.sh
```

### Check Status
```bash
curl http://localhost:8000/system-status
```

### View Logs
```bash
tail -f logs/*.log
```

### Activate Virtual Environment
```bash
source venv/bin/activate
# OR
./activate_env.sh
```

## 📂 Directory Structure

```
hebrew_rag_airgapped_bundle/
├── packages/           # Python packages (.whl files)
├── models/            # Language models
│   ├── transformers/  # Hugging Face models
│   └── ollama/        # Ollama model files + load_models.sh
├── docker_images/     # Pre-built Docker images (.tar.gz files)
├── src/              # Application source code
├── hebrew_tools/     # Hebrew processing modules
├── venv/             # Python virtual environment
├── scripts/          # Deployment scripts
├── docker/           # Docker configuration
├── logs/             # System logs
├── chroma_db/        # Vector database
├── documents/        # Uploaded documents
├── load_docker_images.sh  # Docker image loader
└── verify_bundle.sh   # Bundle verification
```

## 🔐 Air-Gapped Security Features

- ✅ **No external network calls**
- ✅ **All dependencies bundled**
- ✅ **Local model serving only**
- ✅ **Telemetry disabled**
- ✅ **Offline transformers mode**

## ⚠️ Important Notes

### Docker Deployment
For Docker-based deployment, first load the pre-built images:
```bash
# Load Docker images (automatic in phase 4)
./load_docker_images.sh

# Set Docker deployment mode
export DEPLOYMENT_METHOD=docker

# Deploy with Docker
./scripts/deploy.sh
```

### Ollama Models
Ollama models are automatically exported during bundle creation. If manual export is needed:

**On Connected System:**
```bash
ollama pull mistral:7b-instruct
ollama pull llama2:13b-chat
# Models are automatically copied to bundle during creation
```

**In Air-Gapped System:**
```bash
# Models are loaded automatically via models/ollama/load_models.sh
cd models/ollama
./load_models.sh
```

### Model Fallbacks
The system has multiple fallback levels:
1. **Full Agno + Ollama** (best performance)
2. **Agno without LLM** (good performance)
3. **Fallback mode** (basic functionality)

### Virtual Environment
All Python dependencies are installed in a local virtual environment:
- Prevents conflicts with system Python
- Uses only bundled packages
- No PyPI access required

## 🧪 Testing

### Basic System Test
```bash
python test_system.py
```

### API Tests
```bash
# Hebrew processing test
curl -X POST http://localhost:8000/test-hebrew

# Upload document test
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload-document

# Ask question test
curl -X POST -H "Content-Type: application/json" \
     -d '{"question":"מה הנתונים בטבלה?"}' \
     http://localhost:8000/ask-question
```

## 🔧 Troubleshooting

### Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf venv/
./scripts/deployment/phase2_dependencies.sh
```

### Package Installation Issues
```bash
# Check package count
ls packages/*.whl packages/*.tar.gz | wc -l

# Manual package installation
source venv/bin/activate
pip install --no-index --find-links packages/ package_name
```

### Ollama Issues
```bash
# Check Ollama status
curl http://localhost:11434/api/version

# Restart Ollama
./stop_system.sh
./start_system.sh

# View Ollama logs
tail -f logs/ollama.log
```

### System Won't Start
```bash
# Check all logs
tail -f logs/*.log

# Verify environment
source venv/bin/activate
python -c "import sys; print(sys.executable)"
python -c "import agno, transformers, fastapi"
```

## 📊 Performance Expectations

- **Startup time**: 2-5 minutes (first time)
- **Document processing**: 30-60 seconds per PDF
- **Question answering**: 10-30 seconds
- **Memory usage**: 8-16GB depending on models

## 🎯 Success Criteria

✅ System health check passes  
✅ Hebrew text processing works  
✅ Document upload succeeds  
✅ Question answering responds in Hebrew  
✅ No external network calls  
✅ All services running locally  

The system is designed to be **completely self-contained** and work reliably in air-gapped environments!