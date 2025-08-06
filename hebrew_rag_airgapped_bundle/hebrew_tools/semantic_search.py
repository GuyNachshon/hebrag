# Hebrew Semantic Search Tools
from agno.tools import Tool
from typing import List, Dict, Optional, Any, Union
import logging
import numpy as np
from .embedder import HebrewEmbedder

class HebrewSemanticSearch(Tool):
    """
    Hebrew semantic search tool with visual content awareness.
    """
    
    def __init__(self, boost_visual_content: bool = True, context_window: int = 500):
        super().__init__(
            name="hebrew_semantic_search",
            description="Search Hebrew documents with semantic understanding and visual content awareness"
        )
        self.boost_visual_content = boost_visual_content
        self.context_window = context_window
        self.hebrew_embedder = self.setup_hebrew_embedder()
        self.logger = logging.getLogger(__name__)
        
        # Hebrew visual keywords for query analysis
        self.visual_keywords_hebrew = [
            'טבלה', 'תרשים', 'גרף', 'נתונים', 'מספרים',
            'תמונה', 'איור', 'דיאגרמה', 'סטטיסטיקה',
            'השוואה', 'מגמה', 'אחוזים', 'ביצועים',
            'תוצאות', 'מידע', 'רשימה', 'פירוט',
            'ניתוח', 'סיכום', 'דוח', 'מסמך'
        ]
    
    def setup_hebrew_embedder(self) -> HebrewEmbedder:
        """Initialize Hebrew embedding model"""
        return HebrewEmbedder()
    
    def run(self, query: str, context: Optional[str] = None, k: int = 5, 
            document_chunks: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Search Hebrew documents with contextual understanding
        
        Args:
            query: Hebrew search query
            context: Optional context for query understanding
            k: Number of results to return
            document_chunks: List of document chunks to search in
            
        Returns:
            List of relevant document chunks with similarity scores
        """
        try:
            if not document_chunks:
                return []
            
            # Process Hebrew query
            processed_query = self.process_hebrew_query(query, context)
            
            # Generate query embedding
            query_embedding = self.hebrew_embedder.encode([processed_query])[0]
            
            # Determine if query needs visual content
            needs_visual = self.query_needs_visual_content(query)
            
            # Search with appropriate strategy
            if needs_visual and self.boost_visual_content:
                results = self.search_with_visual_boost(
                    query_embedding, document_chunks, k * 2
                )
            else:
                results = self.search_semantic(
                    query_embedding, document_chunks, k
                )
            
            # Re-rank and filter results
            final_results = self.rerank_results(results, query)[:k]
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in Hebrew semantic search: {e}")
            return [{
                "status": "error",
                "error": str(e),
                "query": query
            }]
    
    def process_hebrew_query(self, query: str, context: Optional[str] = None) -> str:
        """Process and enhance Hebrew query"""
        processed_query = query.strip()
        
        # Add context if provided
        if context:
            processed_query = f"{context} {processed_query}"
        
        # Normalize Hebrew text
        processed_query = self.normalize_hebrew_text(processed_query)
        
        return processed_query
    
    def normalize_hebrew_text(self, text: str) -> str:
        """Normalize Hebrew text for better search"""
        if not text:
            return text
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove directional marks (same as embedder)
        text = text.replace('\u200e', '')  # LTR mark
        text = text.replace('\u200f', '')  # RTL mark
        text = text.replace('\u202a', '')  # LTR embedding
        text = text.replace('\u202b', '')  # RTL embedding
        text = text.replace('\u202c', '')  # Pop directional formatting
        text = text.replace('\u202d', '')  # LTR override
        text = text.replace('\u202e', '')  # RTL override
        
        return text.strip()
    
    def query_needs_visual_content(self, query: str) -> bool:
        """Detect if Hebrew query requires visual data"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.visual_keywords_hebrew)
    
    def search_semantic(self, query_embedding: np.ndarray, 
                       document_chunks: List[Dict], k: int) -> List[Dict]:
        """Perform semantic search on document chunks"""
        results = []
        
        try:
            # Extract text content from chunks
            chunk_texts = []
            for chunk in document_chunks:
                if chunk.get('type') == 'text':
                    chunk_texts.append(chunk.get('content', ''))
                elif chunk.get('type') == 'visual':
                    # Use full context for visual chunks
                    chunk_texts.append(chunk.get('full_context', ''))
                else:
                    chunk_texts.append(str(chunk.get('content', '')))
            
            if not chunk_texts:
                return results
            
            # Generate embeddings for chunks
            chunk_embeddings = self.hebrew_embedder.encode(chunk_texts)
            
            # Calculate similarities
            similarities = self.calculate_similarities(
                query_embedding.reshape(1, -1), chunk_embeddings
            )[0]
            
            # Create results with scores
            for i, (chunk, similarity) in enumerate(zip(document_chunks, similarities)):
                result = chunk.copy()
                result['similarity_score'] = float(similarity)
                result['search_rank'] = i
                results.append(result)
            
            # Sort by similarity score
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return results[:k]
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return results
    
    def search_with_visual_boost(self, query_embedding: np.ndarray,
                                document_chunks: List[Dict], k: int) -> List[Dict]:
        """Search with boost for visual content when query needs it"""
        results = self.search_semantic(query_embedding, document_chunks, k)
        
        # Boost visual content scores
        for result in results:
            if result.get('type') == 'visual' or result.get('nearby_visuals'):
                result['similarity_score'] *= 1.2  # 20% boost for visual content
                
            # Extra boost for chunks with Hebrew descriptions of visuals
            if result.get('hebrew_description'):
                result['similarity_score'] *= 1.1  # Additional 10% boost
        
        # Re-sort after boosting
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results[:k]
    
    def calculate_similarities(self, query_embeddings: np.ndarray, 
                             chunk_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between query and chunk embeddings"""
        try:
            return self.hebrew_embedder.similarity(query_embeddings, chunk_embeddings)
        except Exception as e:
            self.logger.error(f"Error calculating similarities: {e}")
            # Return zero similarities as fallback
            return np.zeros((len(query_embeddings), len(chunk_embeddings)))
    
    def rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Re-rank results based on additional criteria"""
        # Apply Hebrew-specific re-ranking
        for result in results:
            bonus_score = 0
            content = result.get('content', '') or result.get('full_context', '')
            
            # Boost for exact Hebrew word matches
            query_words = query.split()
            for word in query_words:
                if word in content:
                    bonus_score += 0.1
            
            # Boost for Hebrew content
            if result.get('is_hebrew', False):
                bonus_score += 0.05
            
            # Apply bonus
            result['similarity_score'] = result.get('similarity_score', 0) + bonus_score
        
        # Final sort
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results


class ContextualRetriever(Tool):
    """
    Contextual retriever that preserves multimodal relationships.
    """
    
    def __init__(self, preserve_relationships: bool = True, multimodal_aware: bool = True):
        super().__init__(
            name="contextual_retriever",
            description="Retrieve content while preserving text-visual relationships"
        )
        self.preserve_relationships = preserve_relationships
        self.multimodal_aware = multimodal_aware
        self.logger = logging.getLogger(__name__)
    
    def run(self, chunks: List[Dict], query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve contextual chunks with relationship preservation
        
        Args:
            chunks: List of search result chunks
            query: Original query for context
            k: Number of final results to return
            
        Returns:
            List of enhanced chunks with context
        """
        try:
            enhanced_chunks = []
            
            for chunk in chunks[:k]:
                enhanced_chunk = self.enhance_chunk_context(chunk, chunks)
                enhanced_chunks.append(enhanced_chunk)
            
            return enhanced_chunks
            
        except Exception as e:
            self.logger.error(f"Error in contextual retrieval: {e}")
            return chunks[:k]
    
    def enhance_chunk_context(self, chunk: Dict, all_chunks: List[Dict]) -> Dict:
        """Enhance a chunk with additional context"""
        enhanced = chunk.copy()
        
        try:
            # Add nearby context if preserving relationships
            if self.preserve_relationships:
                related_chunks = self.find_related_chunks(chunk, all_chunks)
                enhanced['related_content'] = related_chunks
            
            # Add multimodal context
            if self.multimodal_aware:
                multimodal_context = self.get_multimodal_context(chunk, all_chunks)
                enhanced['multimodal_context'] = multimodal_context
            
            # Build comprehensive context
            enhanced['comprehensive_context'] = self.build_comprehensive_context(enhanced)
            
        except Exception as e:
            self.logger.error(f"Error enhancing chunk context: {e}")
        
        return enhanced
    
    def find_related_chunks(self, chunk: Dict, all_chunks: List[Dict]) -> List[Dict]:
        """Find chunks related to the given chunk"""
        related = []
        
        chunk_page = chunk.get('page_number')
        chunk_bbox = chunk.get('bbox')
        
        for other_chunk in all_chunks:
            if other_chunk == chunk:
                continue
            
            # Same page chunks
            if other_chunk.get('page_number') == chunk_page:
                # Check spatial proximity if bbox available
                if chunk_bbox and other_chunk.get('bbox'):
                    if self.are_spatially_related(chunk_bbox, other_chunk['bbox']):
                        related.append({
                            'content': other_chunk.get('content', ''),
                            'type': other_chunk.get('type', 'text'),
                            'relationship': 'spatial'
                        })
                else:
                    # Add nearby chunks from same page
                    related.append({
                        'content': other_chunk.get('content', ''),
                        'type': other_chunk.get('type', 'text'),
                        'relationship': 'same_page'
                    })
        
        return related[:3]  # Limit to 3 related chunks
    
    def get_multimodal_context(self, chunk: Dict, all_chunks: List[Dict]) -> Dict:
        """Get multimodal context for a chunk"""
        context = {
            'has_visuals': False,
            'visual_descriptions': [],
            'text_around_visuals': []
        }
        
        # Check if chunk has nearby visuals
        nearby_visuals = chunk.get('nearby_visuals', [])
        if nearby_visuals:
            context['has_visuals'] = True
            
        # If this is a visual chunk, get its descriptions
        if chunk.get('type') == 'visual':
            context['has_visuals'] = True
            if chunk.get('hebrew_description'):
                context['visual_descriptions'].append(chunk['hebrew_description'])
        
        # Find text around visual elements
        for other_chunk in all_chunks:
            if (other_chunk.get('type') == 'visual' and 
                other_chunk.get('page_number') == chunk.get('page_number')):
                
                text_context = other_chunk.get('full_context', '')
                if text_context:
                    context['text_around_visuals'].append(text_context)
        
        return context
    
    def are_spatially_related(self, bbox1: tuple, bbox2: tuple, threshold: float = 100.0) -> bool:
        """Check if two bounding boxes are spatially related"""
        if not (bbox1 and bbox2):
            return False
        
        # Calculate center points
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        # Calculate distance
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        
        return distance < threshold
    
    def build_comprehensive_context(self, enhanced_chunk: Dict) -> str:
        """Build comprehensive context string"""
        context_parts = []
        
        # Main content
        main_content = enhanced_chunk.get('content', '') or enhanced_chunk.get('full_context', '')
        if main_content:
            context_parts.append(main_content)
        
        # Add related content
        related_content = enhanced_chunk.get('related_content', [])
        for related in related_content:
            if related.get('content'):
                context_parts.append(f"[{related.get('relationship', 'related')}] {related['content']}")
        
        # Add visual descriptions
        multimodal = enhanced_chunk.get('multimodal_context', {})
        for desc in multimodal.get('visual_descriptions', []):
            context_parts.append(f"[תיאור ויזואלי] {desc}")
        
        return ' '.join(context_parts)