# Hebrew RAG System - Air-Gapped Deployment Guide

This guide provides complete instructions for deploying the Hebrew RAG system in air-gapped environments where internet access is not available.

## 📋 Pre-Deployment Checklist

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+, RHEL 8+) or macOS
- **Python**: 3.10 or higher
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 10GB+ free space
- **CPU**: x86_64 architecture

### Required System Packages
The following system packages must be available (installed via package manager):
- `python3` (>= 3.10)
- `python3-venv`
- `python3-pip` 
- `python3-dev`
- `build-essential` (gcc, make, etc.)
- Development libraries for package compilation

## 🚀 Quick Start

### Step 1: Prepare Bundle (On Internet-Connected System)
```bash
# Clone/download the Hebrew RAG system
git clone <repository-url>
cd hebrag

# Download all packages for offline installation
cd deployment
./download-packages.sh

# The system is now ready for transfer to air-gapped environment
```

### Step 2: Transfer to Air-Gapped System
Copy the entire `hebrag` directory to your air-gapped target system.

### Step 3: Deploy on Air-Gapped System
```bash
cd hebrag/deployment

# Install system dependencies (requires sudo)
sudo ./install-system-deps.sh

# Deploy the Hebrew RAG system
./deploy.sh

# Test the installation
./start_hebrag.sh test
```

## 📁 Bundle Structure

```
hebrag/
├── deployment/
│   ├── install-system-deps.sh    # System package installer
│   ├── deploy.sh                 # Main deployment script
│   ├── download-packages.sh      # Package downloader (pre-deployment)
│   ├── wheels/                   # Pre-downloaded Python packages
│   └── README-AIRGAP.md         # This file
├── hebrew_tools/                 # Hebrew processing modules
├── hebrew_rag_system.py          # Main system
├── main.py                       # FastAPI server
├── pyproject.toml               # Project configuration
└── requirements_core.txt        # Core requirements
```

## 🔧 Detailed Deployment Steps

### Phase 1: System Dependencies
```bash
sudo ./install-system-deps.sh
```

This script installs:
- **Python 3.10+** with venv and pip
- **Development tools** (gcc, make, build-essential)
- **System libraries** needed for package compilation
- **uv package manager** (if curl available)

**Supported Package Managers**: apt, yum, dnf, zypper

### Phase 2: Python Environment
```bash
./deploy.sh
```

This script:
1. ✅ Verifies system dependencies
2. 🏗️ Creates Python virtual environment
3. 📦 Installs packages from pre-downloaded wheels
4. 📁 Sets up directory structure
5. 🧪 Tests installation
6. 🚀 Creates startup scripts

### Phase 3: System Validation
```bash
./start_hebrag.sh test
```

Validates:
- ✅ All Python imports working
- ✅ Hebrew RAG system initialization
- ✅ Core components functional
- ✅ Ready for document processing

## 🌐 Running the System

### Start API Server
```bash
# Start the Hebrew RAG API server
./start_hebrag.sh api

# Server will be available at: http://localhost:8000
```

### Test Installation
```bash
# Run comprehensive system tests
./start_hebrag.sh test
```

### Manual Python Usage
```bash
# Activate virtual environment
source .venv/bin/activate

# Use the system programmatically
python -c "
from hebrew_rag_system import HebrewAgnoRAGSystem

config = {
    'llm_model': 'mistral:7b-instruct',
    'embedding_model': './models/heBERT',
    'ollama_base_url': 'http://localhost:11434',
    'chroma_db_path': './chroma_db'
}

rag = HebrewAgnoRAGSystem(config)
print('System ready:', rag.get_system_stats())
"
```

## 🔍 Directory Structure After Deployment

```
hebrag/
├── .venv/                       # Python virtual environment
├── chroma_db/                   # Vector database storage
├── data/
│   ├── documents/               # Input documents (place Hebrew docs here)
│   └── processed/               # Processed document chunks
├── models/                      # ML models (embedding models)
├── logs/                        # Application logs
├── hebrew_tools/                # Hebrew processing modules
├── hebrew_rag_system.py         # Main system
├── main.py                      # FastAPI server
└── start_hebrag.sh             # System startup script
```

## 🎯 Usage Examples

### Process Hebrew Documents
```bash
# Place Hebrew documents in data/documents/
cp your-hebrew-doc.pdf data/documents/

# Process via API
curl -X POST http://localhost:8000/upload \
  -F "file=@data/documents/your-hebrew-doc.pdf"
```

### Ask Questions
```bash
# Ask questions about processed documents
curl -X POST http://localhost:8000/question \
  -H "Content-Type: application/json" \
  -d '{"question": "מה המידע הכלכלי במסמך?"}'
```

## 🛠️ Troubleshooting

### Common Issues

**1. Python 3.10+ Not Available**
```bash
# Check Python version
python3 --version

# If too old, you may need to compile Python from source
# Or use a different system with Python 3.10+
```

**2. System Packages Missing**
```bash
# Re-run system dependencies installer
sudo ./install-system-deps.sh

# Manually install missing packages
sudo apt-get install python3-venv python3-pip python3-dev
```

**3. Package Installation Failures**
```bash
# Check if wheels directory exists and has packages
ls -la deployment/wheels/

# Re-download packages (on internet-connected system)
./download-packages.sh

# Manual package installation
source .venv/bin/activate
pip install --no-index --find-links deployment/wheels/ agno
```

**4. Ollama Not Available**
```bash
# The system works without Ollama for document processing
# LLM features will be limited without Ollama

# Check Ollama status
curl http://localhost:11434/api/version
```

### Logging and Diagnostics
```bash
# Check system status
./start_hebrag.sh test

# View logs
tail -f logs/hebrag.log

# Python diagnostics
source .venv/bin/activate
python -c "
from hebrew_rag_system import HebrewAgnoRAGSystem
import logging
logging.basicConfig(level=logging.DEBUG)
rag = HebrewAgnoRAGSystem({})
print(rag.get_system_stats())
"
```

## 🔒 Security Considerations

### Air-Gapped Environment Benefits
- ✅ **No internet dependency** after deployment
- ✅ **Local processing only** - no data leaves system
- ✅ **Complete offline operation**
- ✅ **Self-contained environment**

### Security Best Practices
- 🔐 Run with minimal privileges (non-root user)
- 🔐 Isolate in container or VM if possible
- 🔐 Regular security updates of system packages
- 🔐 Monitor logs for unusual activity

## 🎉 Success Indicators

After successful deployment, you should see:
- ✅ `./start_hebrag.sh test` passes all checks
- ✅ HTTP API responds at `http://localhost:8000`
- ✅ Hebrew text processing works
- ✅ Document upload and processing functional
- ✅ Vector database operations successful

## 📞 Support

### Self-Diagnostics
1. Run `./start_hebrag.sh test`
2. Check logs in `logs/` directory
3. Verify Python environment: `source .venv/bin/activate && python --version`
4. Test individual components manually

### Common Solutions
- **Performance issues**: Increase system RAM, check CPU usage
- **Storage issues**: Clean up `chroma_db/` and `logs/` periodically
- **Memory errors**: Reduce batch sizes, process smaller documents
- **Package conflicts**: Recreate virtual environment

The Hebrew RAG system is designed for reliable operation in air-gapped environments with comprehensive offline capabilities.