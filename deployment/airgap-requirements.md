# Air-Gapped Deployment Requirements

## System Prerequisites for Air-Gapped Environment

### Python Base Requirements
- **python3** (>= 3.10)
- **python3-venv** - Virtual environment creation
- **python3-pip** - Package installer
- **python3-dev** - Development headers for compiled packages
- **build-essential** - Compilation tools for native extensions

### System Libraries (for package compilation)
- **libssl-dev** - SSL/TLS support
- **libffi-dev** - Foreign function interface
- **libbz2-dev** - Compression support
- **libreadline-dev** - Readline support
- **libsqlite3-dev** - SQLite database support
- **libncurses5-dev** - Terminal UI support
- **libncursesw5-dev** - Wide character terminal support
- **xz-utils** - Compression utilities
- **tk-dev** - GUI toolkit
- **libxml2-dev** - XML parsing
- **libxmlsec1-dev** - XML security
- **libffi-dev** - Foreign function interface
- **liblzma-dev** - LZMA compression

### Additional Runtime Dependencies
- **curl** or **wget** - For downloading models (if needed)
- **unzip** - For extracting archives
- **git** - Version control (optional, for development)

## Bundle Contents Required

### 1. Python Environment Tools
```bash
# These should be pre-installed on target system:
apt-get install python3 python3-venv python3-pip python3-dev build-essential
apt-get install libssl-dev libffi-dev libbz2-dev libreadline-dev libsqlite3-dev
apt-get install libncurses5-dev libncursesw5-dev xz-utils tk-dev
apt-get install libxml2-dev libxmlsec1-dev liblzma-dev
```

### 2. Python Package Manager
- **uv** binary (for fast package management)
- **pip** (fallback package installer)

### 3. Pre-downloaded Python Packages
All packages from `requirements_core.txt` downloaded as wheels:
- agno>=1.7.7
- chromadb>=1.0.15
- sentence-transformers>=5.0.0
- ollama>=0.5.2
- torch (CPU version for air-gap)
- transformers
- numpy
- pandas
- fastapi
- uvicorn
- All dependencies

### 4. Model Files
- Hebrew embedding model (heBERT or multilingual)
- Ollama model files
- Tokenizer files

## Deployment Package Structure
```
hebrag-airgap-bundle/
├── bin/
│   ├── uv                          # Package manager binary
│   └── install-system-deps.sh      # System dependencies installer
├── wheels/                         # Pre-downloaded Python packages
│   ├── agno-1.7.7-py3-none-any.whl
│   ├── chromadb-1.0.15-*.whl
│   ├── torch-*-cpu.whl
│   └── ... (all dependencies)
├── models/
│   ├── heBERT/                     # Hebrew embedding model
│   └── ollama/                     # Ollama model files
├── src/
│   └── hebrag/                     # Source code
├── deploy.sh                       # Main deployment script
├── requirements-system.txt         # System package requirements
└── README-AIRGAP.md               # Deployment instructions
```