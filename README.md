# Hebrew Agentic RAG System

A comprehensive Hebrew-language Retrieval-Augmented Generation (RAG) system built with the Agno framework, designed for air-gapped environments with full offline capabilities.

## 🌟 Features

- **Hebrew Language Support**: Native Hebrew document processing with RTL text handling
- **Multimodal Processing**: Extract and analyze text, tables, and visual elements from documents
- **Agentic Architecture**: Specialized agents for document processing, retrieval, analysis, and response generation
- **Air-Gapped Deployment**: Complete offline operation with bundled dependencies and models
- **FastAPI Web Interface**: RESTful API with interactive documentation
- **Vector Database**: ChromaDB integration for semantic search
- **Local LLM Integration**: Ollama support for private AI inference

## 📋 Requirements

- **System**: Linux x86_64, macOS, or Windows with Docker
- **Memory**: 16GB+ RAM recommended
- **Storage**: 50GB+ free space
- **Docker**: Required for bundle creation and deployment

## 🚀 Quick Start

### 1. Create Air-Gapped Bundle

```bash
# Clone the repository
git clone <repository-url>
cd hebrag

# Create the air-gapped bundle (requires Docker)
./create_bundle_docker.sh
```

This creates a complete `hebrew_rag_airgapped_bundle/` directory with all dependencies.

### 2. Deploy the Bundle

```bash
cd hebrew_rag_airgapped_bundle

# Automated deployment (all phases)
./scripts/deploy.sh

# Or run phases individually:
./scripts/deployment/phase1_environment.sh  # Environment setup
./scripts/deployment/phase2_dependencies.sh # Install dependencies  
./scripts/deployment/phase3_configuration.sh # Configure system
./scripts/deployment/phase4_startup.sh      # Start services
```

### 3. Access the System

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Frontend**: http://localhost:8000/ (if included)

## 📦 Bundle Contents

The air-gapped bundle includes:

```
hebrew_rag_airgapped_bundle/
├── packages/           # Python packages for all platforms
├── models/            # Pre-trained models
│   ├── transformers/  # Hebrew BERT and multilingual models
│   └── ollama/        # Local LLM models
├── src/               # Application source code
├── hebrew_tools/      # Hebrew processing modules
├── frontend/          # Web interface
├── scripts/           # Deployment and maintenance scripts
├── config/            # Configuration files
└── documentation/     # Setup and usage guides
```

## 🔧 Configuration

### Environment Variables

```bash
# Air-gapped mode (recommended)
export AGNO_TELEMETRY=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Ollama configuration
export OLLAMA_HOST=0.0.0.0:11434
```

### Model Configuration

Edit `src/hebrew_rag_system.py` to configure models:

```python
self.model_config = {
    "llm_model": "mistral:7b-instruct",  # Change to your preferred model
    "embedding_model": "./models/transformers/heBERT",
    "ollama_base_url": "http://localhost:11434"
}
```

## 🛠️ Development

### Dependencies

The system requires these key packages (automatically included in bundle):

```
# Core framework
agno>=1.0.0
ollama>=0.4.0
openai>=1.0.0

# Document processing
unstructured[local-inference]>=0.10.0
pymupdf>=1.20.0
hebrew-tokenizer>=2.0.0

# Web framework
fastapi>=0.100.0
uvicorn[standard]>=0.20.0

# Vector database
chromadb>=0.4.0
```

### Hebrew Tools Implementation

The system includes stub implementations for Hebrew-specific processing. To implement actual functionality:

1. **Document Processor** (`hebrew_tools/document_processor.py`):
   - Hebrew OCR integration
   - RTL text layout handling
   - Table and image processing

2. **Semantic Search** (`hebrew_tools/semantic_search.py`):
   - Hebrew word embeddings
   - Morphological analysis
   - Context-aware retrieval

3. **Table Analyzer** (`hebrew_tools/table_analyzer.py`):
   - Hebrew header detection
   - Numeric data extraction
   - Chart interpretation

4. **Response Generator** (`hebrew_tools/response_generator.py`):
   - Natural Hebrew language generation
   - Citation formatting
   - Grammar checking

### Custom Models

To use custom Hebrew models:

1. Place models in `models/transformers/`
2. Update model paths in configuration
3. Ensure compatibility with transformers library

## 🐳 Docker Deployment

### Using Docker Compose

```yaml
version: '3.8'
services:
  hebrew-rag:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./documents:/app/documents
      - ./chroma_db:/app/chroma_db
    environment:
      - AGNO_TELEMETRY=false
      - TRANSFORMERS_OFFLINE=1
```

### Build Custom Image

```bash
cd hebrew_rag_airgapped_bundle
docker build -t hebrew-rag .
docker run -p 8000:8000 hebrew-rag
```

## 📊 Usage Examples

### Document Upload and Processing

```python
import requests

# Upload Hebrew document
files = {'file': open('hebrew_document.pdf', 'rb')}
response = requests.post('http://localhost:8000/upload', files=files)

# Ask question about document
question = {"question": "מה העיקרים המרכזיים במסמך?"}
response = requests.post('http://localhost:8000/ask', json=question)
print(response.json())
```

### Using the Python API

```python
from hebrew_rag_system import HebrewAgnoRAGSystem

# Initialize system
rag = HebrewAgnoRAGSystem()

# Process document
await rag.process_document('path/to/hebrew/document.pdf')

# Ask question
result = await rag.answer_question("מה הנושא הראשי?")
print(result['answer'])
```

## 🔍 Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   # Reinstall packages
   pip install -r requirements.txt
   ```

2. **Ollama Connection Issues**:
   ```bash
   # Check Ollama status
   ollama serve
   curl http://localhost:11434/api/version
   ```

3. **Hebrew Text Display**:
   - Ensure UTF-8 encoding
   - Install Hebrew fonts if needed
   - Check browser/terminal Hebrew support

### Performance Optimization

1. **Memory Usage**: Adjust batch sizes in configuration
2. **GPU Support**: Enable CUDA for transformers if available
3. **Model Size**: Use smaller models for resource-constrained environments

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Update documentation and bundle script
5. Submit pull request

### Development Setup

```bash
# Development installation
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ hebrew_tools/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check `documentation/` in the bundle
- **Issues**: Create GitHub issue with detailed description
- **Hebrew Support**: Include Hebrew text samples when reporting issues

## 🔄 Version History

- **v1.0.0**: Initial release with basic Hebrew RAG functionality
- **v1.1.0**: Added air-gapped bundle creation
- **v1.2.0**: Enhanced Hebrew tools and Docker support

## 🎯 Roadmap

- [ ] Advanced Hebrew morphological analysis
- [ ] Multi-document conversation context
- [ ] Enhanced table processing
- [ ] Hebrew OCR integration
- [ ] Performance benchmarking tools
- [ ] Multi-language support expansion

---

For detailed deployment instructions and troubleshooting guides, see the `documentation/` directory in your bundle.