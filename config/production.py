# Production Configuration for Hebrew Agentic RAG System
import os
from pathlib import Path
from typing import Dict, Any, List
import logging

class ProductionConfig:
    """Production configuration for air-gapped Hebrew RAG deployment"""
    
    # Air-gapped settings
    AGNO_TELEMETRY = False
    DISABLE_EXTERNAL_CALLS = True
    TRANSFORMERS_OFFLINE = True
    HF_DATASETS_OFFLINE = True
    
    # Model paths (adjust for your deployment)
    HEBREW_MODELS_PATH = Path("./models")
    OLLAMA_MODEL = "mistral:7b-instruct"
    OLLAMA_FALLBACK_MODEL = "llama2:13b-chat"
    HEBREW_EMBEDDING_MODEL = "heBERT"
    
    # Database settings
    CHROMA_DB_PATH = Path("./chroma_db")
    VECTOR_DB_COLLECTION = "hebrew_documents"
    BACKUP_DB_PATH = Path("./backups/chroma_db")
    
    # Performance settings
    MAX_CONCURRENT_AGENTS = 10
    AGENT_TIMEOUT = 300  # 5 minutes for complex Hebrew queries
    MAX_TOKENS_PER_RESPONSE = 2048
    LLM_TEMPERATURE = 0.1  # Low temperature for factual responses
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    MAX_CHUNKS_PER_QUERY = 10
    
    # Hebrew processing settings
    HEBREW_OCR_ENABLED = True
    LAYOUT_ANALYSIS_ENABLED = True
    VISUAL_PROCESSING_ENABLED = True
    RTL_TEXT_SUPPORT = True
    MIXED_LANGUAGE_SUPPORT = True
    
    # Document processing
    MAX_FILE_SIZE_MB = 100
    ALLOWED_FILE_TYPES = [".pdf", ".docx", ".doc", ".txt"]
    SUPPORTED_LANGUAGES = ["he", "en"]  # Hebrew and English
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_WORKERS = 4
    CORS_ORIGINS = ["*"]  # Configure appropriately for production
    REQUEST_TIMEOUT = 300  # 5 minutes
    MAX_REQUEST_SIZE = 100 * 1024 * 1024  # 100MB
    
    # Security settings
    LOG_LEVEL = "INFO"
    LOG_FILE = "./logs/hebrew_rag.log"
    LOG_MAX_SIZE = 100 * 1024 * 1024  # 100MB
    LOG_BACKUP_COUNT = 5
    ENABLE_REQUEST_LOGGING = True
    MASK_SENSITIVE_DATA = True
    
    # Monitoring settings
    ENABLE_PERFORMANCE_MONITORING = True
    METRICS_RETENTION_DAYS = 30
    HEALTH_CHECK_INTERVAL = 60  # seconds
    MEMORY_ALERT_THRESHOLD = 85  # percent
    CPU_ALERT_THRESHOLD = 80  # percent
    
    # Cache settings
    ENABLE_RESPONSE_CACHE = True
    CACHE_TTL_SECONDS = 3600  # 1 hour
    MAX_CACHE_SIZE_MB = 512
    
    # Ollama settings
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_TIMEOUT = 120  # seconds
    OLLAMA_MAX_RETRIES = 3
    OLLAMA_RETRY_DELAY = 5  # seconds
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        config = {}
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                value = getattr(cls, attr)
                if isinstance(value, Path):
                    value = str(value)
                config[attr.lower()] = value
        return config
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate production configuration"""
        errors = []
        
        # Check required paths
        required_paths = [
            cls.HEBREW_MODELS_PATH,
        ]
        
        for path in required_paths:
            if not path.exists():
                errors.append(f"Required path does not exist: {path}")
        
        # Create necessary directories
        directories_to_create = [
            cls.CHROMA_DB_PATH,
            cls.BACKUP_DB_PATH,
            Path("./documents"),
            Path("./logs"),
            Path("./cache")
        ]
        
        for directory in directories_to_create:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {directory}: {e}")
        
        # Check file permissions
        try:
            test_file = Path("./logs/config_test.tmp")
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            errors.append(f"Cannot write to logs directory: {e}")
        
        # Verify Ollama connectivity (if required)
        if cls.OLLAMA_BASE_URL:
            try:
                import requests
                response = requests.get(
                    f"{cls.OLLAMA_BASE_URL}/api/version", 
                    timeout=5
                )
                if response.status_code != 200:
                    errors.append("Ollama is not accessible")
            except Exception as e:
                errors.append(f"Failed to connect to Ollama: {e}")
        
        # Check available disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage("./")
            free_gb = free // (1024**3)
            if free_gb < 10:  # Less than 10GB free
                errors.append(f"Low disk space: {free_gb}GB free")
        except Exception as e:
            errors.append(f"Cannot check disk space: {e}")
        
        if errors:
            for error in errors:
                logging.error(error)
            return False
        
        return True
    
    @classmethod
    def setup_logging(cls):
        """Setup production logging configuration"""
        from logging.handlers import RotatingFileHandler
        
        # Ensure logs directory exists
        log_dir = Path(cls.LOG_FILE).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, cls.LOG_LEVEL))
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            cls.LOG_FILE,
            maxBytes=cls.LOG_MAX_SIZE,
            backupCount=cls.LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(logging.WARNING)
        root_logger.addHandler(console_handler)
        
        # Suppress some noisy loggers
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        logging.info("Production logging configured")
    
    @classmethod
    def get_environment_overrides(cls) -> Dict[str, Any]:
        """Get configuration overrides from environment variables"""
        overrides = {}
        
        # Map environment variables to config attributes
        env_mappings = {
            "HEBREW_RAG_OLLAMA_MODEL": "ollama_model",
            "HEBREW_RAG_LLM_TEMPERATURE": "llm_temperature",
            "HEBREW_RAG_MAX_TOKENS": "max_tokens_per_response",
            "HEBREW_RAG_API_PORT": "api_port",
            "HEBREW_RAG_LOG_LEVEL": "log_level",
            "HEBREW_RAG_OLLAMA_URL": "ollama_base_url",
            "HEBREW_RAG_DB_PATH": "chroma_db_path"
        }
        
        for env_var, config_attr in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                # Type conversion based on attribute
                if config_attr in ["api_port", "max_tokens_per_response"]:
                    try:
                        value = int(value)
                    except ValueError:
                        continue
                elif config_attr in ["llm_temperature"]:
                    try:
                        value = float(value)
                    except ValueError:
                        continue
                elif config_attr in ["chroma_db_path"]:
                    value = Path(value)
                
                overrides[config_attr] = value
        
        return overrides
    
    @classmethod
    def create_runtime_config(cls) -> Dict[str, Any]:
        """Create runtime configuration with environment overrides"""
        config = cls.get_config_dict()
        overrides = cls.get_environment_overrides()
        config.update(overrides)
        return config


class DevelopmentConfig(ProductionConfig):
    """Development configuration with relaxed settings"""
    
    # Development overrides
    LOG_LEVEL = "DEBUG"
    AGNO_TELEMETRY = True  # Allow telemetry in development
    DISABLE_EXTERNAL_CALLS = False
    TRANSFORMERS_OFFLINE = False
    HF_DATASETS_OFFLINE = False
    
    # More permissive settings
    MAX_FILE_SIZE_MB = 50
    AGENT_TIMEOUT = 120  # 2 minutes
    CORS_ORIGINS = ["http://localhost:3000", "http://localhost:8080"]
    
    # Development paths
    HEBREW_MODELS_PATH = Path("./dev_models")
    CHROMA_DB_PATH = Path("./dev_chroma_db")


class TestConfig(ProductionConfig):
    """Test configuration for unit tests"""
    
    # Test overrides
    LOG_LEVEL = "DEBUG"
    CHROMA_DB_PATH = Path("./test_chroma_db")
    HEBREW_MODELS_PATH = Path("./test_models")
    ENABLE_PERFORMANCE_MONITORING = False
    ENABLE_RESPONSE_CACHE = False
    
    # Minimal settings for testing
    MAX_TOKENS_PER_RESPONSE = 512
    AGENT_TIMEOUT = 30
    MAX_CHUNKS_PER_QUERY = 3


def get_config(environment: str = "production") -> ProductionConfig:
    """Get configuration based on environment"""
    configs = {
        "production": ProductionConfig,
        "development": DevelopmentConfig,
        "test": TestConfig
    }
    
    config_class = configs.get(environment, ProductionConfig)
    return config_class


def load_config_from_file(config_file: Path) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file"""
    try:
        import json
        
        if config_file.suffix.lower() == '.json':
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif config_file.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files")
        else:
            raise ValueError(f"Unsupported config file format: {config_file.suffix}")
            
    except Exception as e:
        logging.error(f"Failed to load config from {config_file}: {e}")
        return {}


# Export main configuration
Config = get_config(os.environ.get("HEBREW_RAG_ENV", "production"))