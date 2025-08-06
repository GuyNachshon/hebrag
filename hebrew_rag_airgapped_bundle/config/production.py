import os
from pathlib import Path

class ProductionConfig:
    # Air-gapped settings
    AGNO_TELEMETRY = False
    DISABLE_EXTERNAL_CALLS = True
    
    # Model paths
    HEBREW_MODELS_PATH = Path("./models")
    OLLAMA_MODEL = "mistral:7b-instruct"
    HEBREW_EMBEDDING_MODEL = "heBERT"
    
    # Database settings
    CHROMA_DB_PATH = Path("./chroma_db")
    VECTOR_DB_COLLECTION = "hebrew_documents"
    
    # Performance settings
    MAX_CONCURRENT_AGENTS = 10
    AGENT_TIMEOUT = 300  # 5 minutes
    MAX_TOKENS_PER_RESPONSE = 2048
    
    # Hebrew processing settings
    HEBREW_OCR_ENABLED = True
    LAYOUT_ANALYSIS_ENABLED = True
    VISUAL_PROCESSING_ENABLED = True
    
    # Security settings
    LOG_LEVEL = "INFO"
    MAX_FILE_SIZE_MB = 100
    ALLOWED_FILE_TYPES = [".pdf", ".docx", ".doc", ".txt"]
    
    # Monitoring settings
    ENABLE_PERFORMANCE_MONITORING = True
    METRICS_RETENTION_DAYS = 30
