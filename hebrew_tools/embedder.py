# Hebrew Embedder for vector search
from typing import List, Optional, Union, Any
import logging
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available. Embedding functionality will be limited.")

class HebrewEmbedder:
    """
    Hebrew text embedder using local models for air-gapped deployment.
    """
    
    def __init__(self, model_path: str = "./models/heBERT"):
        self.model_path = Path(model_path)
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.setup_model()
    
    def setup_model(self):
        """Initialize the Hebrew embedding model"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Try multiple Hebrew/multilingual models in order of preference
                model_options = [
                    str(self.model_path),  # Local Hebrew model if exists
                    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                    'sentence-transformers/all-MiniLM-L6-v2',  # Small general model
                ]
                
                for model_name in model_options:
                    try:
                        if model_name == str(self.model_path) and not self.model_path.exists():
                            continue
                            
                        self.model = SentenceTransformer(model_name)
                        self.logger.info(f"Successfully loaded embedding model: {model_name}")
                        
                        # Test the model with Hebrew text
                        test_embedding = self.model.encode(['זהו טקסט בדיקה'])
                        self.logger.info(f"Model test successful, embedding dim: {test_embedding.shape}")
                        break
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load model {model_name}: {e}")
                        continue
                
                if self.model is None:
                    self.logger.error("All embedding models failed to load")
            else:
                self.logger.error("SentenceTransformers not available")
                self.model = None
                
        except Exception as e:
            self.logger.error(f"Error setting up Hebrew embedder: {e}")
            self.model = None
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode texts into embeddings
        
        Args:
            texts: Single text or list of texts to encode
            **kwargs: Additional arguments for the encoder
            
        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            # Return dummy embeddings if model not available
            if isinstance(texts, str):
                texts = [texts]
            return np.random.rand(len(texts), 384).astype(np.float32)
        
        try:
            # Normalize Hebrew text before encoding
            if isinstance(texts, str):
                texts = self.normalize_hebrew_text(texts)
            else:
                texts = [self.normalize_hebrew_text(text) for text in texts]
            
            # Encode using the model
            embeddings = self.model.encode(texts, **kwargs)
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error encoding texts: {e}")
            # Return dummy embeddings as fallback
            text_count = 1 if isinstance(texts, str) else len(texts)
            return np.random.rand(text_count, 384).astype(np.float32)
    
    def normalize_hebrew_text(self, text: str) -> str:
        """Normalize Hebrew text for better embedding quality"""
        if not text:
            return text
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove directional marks
        text = text.replace('\u200e', '')  # LTR mark
        text = text.replace('\u200f', '')  # RTL mark
        text = text.replace('\u202a', '')  # LTR embedding
        text = text.replace('\u202b', '')  # RTL embedding
        text = text.replace('\u202c', '')  # Pop directional formatting
        text = text.replace('\u202d', '')  # LTR override
        text = text.replace('\u202e', '')  # RTL override
        
        return text.strip()
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between embeddings"""
        try:
            # Normalize embeddings
            embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
            
            # Calculate cosine similarity
            similarity_matrix = np.dot(embeddings1_norm, embeddings2_norm.T)
            return similarity_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return np.zeros((len(embeddings1), len(embeddings2)))
    
    def is_available(self) -> bool:
        """Check if the embedder is available and working"""
        return self.model is not None