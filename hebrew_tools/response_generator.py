# Hebrew Response Generation Tools
try:
    from agno.tools import tool
    AGNO_TOOLS_AVAILABLE = True
except ImportError:
    AGNO_TOOLS_AVAILABLE = False
    # Define a simple fallback decorator
    def tool(description: str = ""):
        def decorator(func):
            func.description = description
            return func
        return decorator

from typing import List, Dict, Optional, Any, Union
import logging
import re
from datetime import datetime

class HebrewResponseGenerator:
    """
    Generate comprehensive Hebrew responses with proper citations.
    """
    
    def __init__(self, citation_style: str = "academic", preserve_context: bool = True):
        self.name = "hebrew_response_generator"
        self.description = "Generate comprehensive Hebrew responses with proper citations"
        self.citation_style = citation_style
        self.preserve_context = preserve_context
        self.logger = logging.getLogger(__name__)
    
    def run(self, question: str, retrieved_chunks: List[Dict], 
            analysis_results: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Generate Hebrew response from retrieved information
        
        Args:
            question: Original Hebrew question
            retrieved_chunks: Relevant document chunks
            analysis_results: Visual analysis results (optional)
            
        Returns:
            Dict containing generated response and metadata
        """
        try:
            # Combine textual and visual information
            combined_context = self.combine_multimodal_context(
                retrieved_chunks, analysis_results or []
            )
            
            # Generate structured Hebrew response
            response = self.generate_hebrew_response(
                question, combined_context
            )
            
            # Add citations
            cited_response = self.add_citations(response, retrieved_chunks)
            
            # Validate Hebrew grammar and coherence
            validation_result = self.validate_response(cited_response)
            
            return {
                "status": "success",
                "response": cited_response,
                "confidence": self.calculate_confidence(combined_context),
                "sources": self.extract_source_metadata(retrieved_chunks),
                "validation": validation_result,
                "word_count": len(cited_response.split())
            }
            
        except Exception as e:
            self.logger.error(f"Error generating Hebrew response: {e}")
            return {
                "status": "error",
                "error": str(e),
                "question": question
            }
    
    def combine_multimodal_context(self, retrieved_chunks: List[Dict], 
                                 analysis_results: List[Dict]) -> Dict[str, Any]:
        """Combine textual and visual information into unified context"""
        context = {
            "text_content": [],
            "visual_content": [],
            "tables": [],
            "charts": [],
            "combined_narrative": ""
        }
        
        try:
            # Process retrieved chunks
            for chunk in retrieved_chunks:
                chunk_type = chunk.get('type', 'text')
                content = chunk.get('content', '') or chunk.get('full_context', '')
                
                if chunk_type == 'text':
                    context["text_content"].append({
                        "content": content,
                        "source": chunk.get('chunk_id', 'unknown'),
                        "page": chunk.get('page_number'),
                        "is_hebrew": chunk.get('is_hebrew', False)
                    })
                elif chunk_type == 'visual':
                    visual_info = {
                        "type": chunk.get('visual_type', 'unknown'),
                        "description": chunk.get('hebrew_description', ''),
                        "context": content,
                        "source": chunk.get('chunk_id', 'unknown')
                    }
                    context["visual_content"].append(visual_info)
                    
                    # Categorize visuals
                    if 'table' in chunk.get('visual_type', '').lower():
                        context["tables"].append(visual_info)
                    elif any(word in chunk.get('visual_type', '').lower() 
                           for word in ['chart', 'graph', 'גרף', 'תרשים']):
                        context["charts"].append(visual_info)
            
            # Process analysis results
            for result in analysis_results:
                if result.get('status') == 'success':
                    if 'table_structure' in result:
                        context["tables"].append({
                            "description": result.get('hebrew_description', ''),
                            "insights": result.get('insights', []),
                            "type": "table_analysis"
                        })
                    elif 'chart_type' in result:
                        context["charts"].append({
                            "description": result.get('hebrew_description', ''),
                            "trends": result.get('trends', []),
                            "type": "chart_analysis"
                        })
            
            # Build combined narrative
            context["combined_narrative"] = self.build_narrative(context)
            
        except Exception as e:
            self.logger.error(f"Error combining context: {e}")
        
        return context
    
    def build_narrative(self, context: Dict) -> str:
        """Build a coherent narrative from all context sources"""
        narrative_parts = []
        
        # Add text content
        hebrew_texts = [item['content'] for item in context["text_content"] 
                       if item.get('is_hebrew', False)]
        if hebrew_texts:
            narrative_parts.extend(hebrew_texts[:3])  # Limit to most relevant
        
        # Add non-Hebrew content
        other_texts = [item['content'] for item in context["text_content"] 
                      if not item.get('is_hebrew', False)]
        if other_texts:
            narrative_parts.extend(other_texts[:2])
        
        # Add visual descriptions
        for visual in context["visual_content"]:
            if visual.get('description'):
                narrative_parts.append(f"[ויזואלי] {visual['description']}")
        
        # Add table insights
        for table in context["tables"]:
            if table.get('insights'):
                narrative_parts.extend(table['insights'][:2])
        
        # Add chart information
        for chart in context["charts"]:
            if chart.get('description'):
                narrative_parts.append(f"[תרשים] {chart['description']}")
        
        return " ".join(narrative_parts)
    
    def generate_hebrew_response(self, question: str, context: Dict) -> str:
        """Generate natural Hebrew response using context"""
        try:
            # Analyze question type
            question_type = self.analyze_question_type(question)
            
            # Build response based on question type and available context
            if question_type == "data_query" and (context["tables"] or context["charts"]):
                response = self.generate_data_response(question, context)
            elif question_type == "summary_query":
                response = self.generate_summary_response(question, context)
            elif question_type == "comparison_query":
                response = self.generate_comparison_response(question, context)
            else:
                response = self.generate_general_response(question, context)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating Hebrew response: {e}")
            return self.generate_fallback_response(question, context)
    
    def analyze_question_type(self, question: str) -> str:
        """Analyze the type of Hebrew question"""
        question_lower = question.lower()
        
        # Data-related questions
        data_keywords = ['כמה', 'מה הנתונים', 'איך משתנה', 'מה המספרים', 'באיזה אחוז']
        if any(keyword in question_lower for keyword in data_keywords):
            return "data_query"
        
        # Summary questions
        summary_keywords = ['מה המסקנות', 'סכם', 'תאר', 'הסבר', 'מה העיקר']
        if any(keyword in question_lower for keyword in summary_keywords):
            return "summary_query"
        
        # Comparison questions
        comparison_keywords = ['השווה', 'מה ההבדל', 'איך שונה', 'בהשוואה']
        if any(keyword in question_lower for keyword in comparison_keywords):
            return "comparison_query"
        
        return "general_query"
    
    def generate_data_response(self, question: str, context: Dict) -> str:
        """Generate response focused on data and numbers"""
        response_parts = []
        
        # Start with direct answer if possible
        if context["tables"]:
            table_info = context["tables"][0]
            if table_info.get('insights'):
                response_parts.append("על פי הנתונים בטבלה:")
                response_parts.extend(table_info['insights'][:3])
        
        if context["charts"]:
            chart_info = context["charts"][0]
            if chart_info.get('description'):
                response_parts.append(f"התרשים מציג: {chart_info['description']}")
        
        # Add supporting text context
        if context["text_content"]:
            relevant_text = [item['content'] for item in context["text_content"]
                           if any(word in item['content'].lower() 
                                 for word in ['נתונים', 'מספרים', 'אחוז', 'סכום'])]
            if relevant_text:
                response_parts.extend(relevant_text[:2])
        
        return ". ".join(response_parts) if response_parts else self.generate_fallback_response(question, context)
    
    def generate_summary_response(self, question: str, context: Dict) -> str:
        """Generate summary response"""
        response_parts = []
        
        # Collect key points from all sources
        key_points = []
        
        # From text content
        for item in context["text_content"]:
            if item.get('is_hebrew') and len(item['content']) > 20:
                key_points.append(item['content'])
        
        # From visual descriptions
        for visual in context["visual_content"]:
            if visual.get('description'):
                key_points.append(visual['description'])
        
        if key_points:
            response_parts.append("לסיכום העיקרים:")
            response_parts.extend(key_points[:4])  # Top 4 points
        else:
            response_parts.append("על פי המידע הזמין:")
            response_parts.append(context.get("combined_narrative", "")[:200])
        
        return ". ".join(response_parts)
    
    def generate_comparison_response(self, question: str, context: Dict) -> str:
        """Generate comparison response"""
        response_parts = []
        
        # Look for comparative data
        if context["tables"]:
            response_parts.append("על פי השוואת הנתונים:")
            table_insights = context["tables"][0].get('insights', [])
            if table_insights:
                response_parts.extend(table_insights[:3])
        
        # Add text-based comparisons
        comparative_texts = [item['content'] for item in context["text_content"]
                           if any(word in item['content'].lower() 
                                 for word in ['לעומת', 'בהשוואה', 'יותר', 'פחות', 'שונה'])]
        if comparative_texts:
            response_parts.extend(comparative_texts[:2])
        
        return ". ".join(response_parts) if response_parts else self.generate_fallback_response(question, context)
    
    def generate_general_response(self, question: str, context: Dict) -> str:
        """Generate general response"""
        response_parts = []
        
        # Use combined narrative as base
        narrative = context.get("combined_narrative", "")
        if narrative:
            # Split into sentences and use most relevant
            sentences = [s.strip() for s in narrative.split('.') if s.strip()]
            hebrew_sentences = [s for s in sentences if self.contains_hebrew(s)]
            
            if hebrew_sentences:
                response_parts.extend(hebrew_sentences[:3])
            else:
                response_parts.extend(sentences[:3])
        
        return ". ".join(response_parts) if response_parts else self.generate_fallback_response(question, context)
    
    def generate_fallback_response(self, question: str, context: Dict) -> str:
        """Generate fallback response when other methods fail"""
        if context.get("combined_narrative"):
            return f"על פי המידע הזמין: {context['combined_narrative'][:300]}..."
        else:
            return "מצטער, לא מצאתי מידע מספיק כדי לענות על השאלה באופן מלא."
    
    def contains_hebrew(self, text: str) -> bool:
        """Check if text contains Hebrew characters"""
        if not text:
            return False
        hebrew_chars = set(range(0x0590, 0x05FF))
        return any(ord(char) in hebrew_chars for char in text)
    
    def add_citations(self, response: str, retrieved_chunks: List[Dict]) -> str:
        """Add citations to the response"""
        if not retrieved_chunks or self.citation_style == "none":
            return response
        
        try:
            citations = []
            for i, chunk in enumerate(retrieved_chunks[:3]):  # Max 3 citations
                source = chunk.get('source_file', chunk.get('chunk_id', f'מקור {i+1}'))
                page = chunk.get('page_number')
                
                if self.citation_style == "academic":
                    if page:
                        citation = f"({source}, עמ' {page})"
                    else:
                        citation = f"({source})"
                else:  # simple style
                    citation = f"[מקור {i+1}]"
                
                citations.append(citation)
            
            if citations:
                response += f"\n\nמקורות: {', '.join(citations)}"
            
        except Exception as e:
            self.logger.error(f"Error adding citations: {e}")
        
        return response
    
    def calculate_confidence(self, context: Dict) -> float:
        """Calculate confidence score for the response"""
        score = 0.0
        
        try:
            # Base score from text content
            text_count = len(context.get("text_content", []))
            score += min(text_count * 0.2, 0.6)  # Max 0.6 from text
            
            # Boost from Hebrew content
            hebrew_texts = [item for item in context.get("text_content", [])
                          if item.get('is_hebrew', False)]
            if hebrew_texts:
                score += 0.15
            
            # Boost from visual content
            if context.get("visual_content"):
                score += 0.1
            
            # Boost from structured data
            if context.get("tables") or context.get("charts"):
                score += 0.15
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5  # Default moderate confidence
    
    def extract_source_metadata(self, retrieved_chunks: List[Dict]) -> List[Dict]:
        """Extract metadata about sources used"""
        sources = []
        
        try:
            for chunk in retrieved_chunks:
                source_info = {
                    "id": chunk.get('chunk_id', 'unknown'),
                    "type": chunk.get('type', 'text'),
                    "page": chunk.get('page_number'),
                    "confidence": chunk.get('similarity_score', 0.0),
                    "file": chunk.get('source_file', 'unknown')
                }
                sources.append(source_info)
                
        except Exception as e:
            self.logger.error(f"Error extracting source metadata: {e}")
        
        return sources
    
    def validate_response(self, response: str) -> Dict[str, Any]:
        """Validate Hebrew grammar and coherence"""
        validation = {
            "has_hebrew": False,
            "sentence_count": 0,
            "avg_sentence_length": 0,
            "coherence_score": 0.0,
            "issues": []
        }
        
        try:
            # Check for Hebrew content
            validation["has_hebrew"] = self.contains_hebrew(response)
            
            # Count sentences
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            validation["sentence_count"] = len(sentences)
            
            # Average sentence length
            if sentences:
                total_words = sum(len(sentence.split()) for sentence in sentences)
                validation["avg_sentence_length"] = total_words / len(sentences)
            
            # Basic coherence check
            if validation["sentence_count"] > 0:
                if validation["avg_sentence_length"] > 5:
                    validation["coherence_score"] += 0.3
                if validation["has_hebrew"]:
                    validation["coherence_score"] += 0.4
                if not any(word in response.lower() for word in ['error', 'שגיאה']):
                    validation["coherence_score"] += 0.3
            
            # Check for issues
            if not validation["has_hebrew"] and 'עברית' in response.lower():
                validation["issues"].append("Response should contain Hebrew text")
            
            if validation["sentence_count"] == 0:
                validation["issues"].append("No complete sentences found")
                
        except Exception as e:
            self.logger.error(f"Error validating response: {e}")
            validation["issues"].append(f"Validation error: {e}")
        
        return validation


class AnswerValidator:
    """
    Validate Hebrew answers for grammar and citation accuracy.
    """
    
    def __init__(self, check_hebrew_grammar: bool = True, verify_citations: bool = True):
        self.name = "answer_validator"
        self.description = "Validate Hebrew answers for grammar and citation accuracy"
        self.check_hebrew_grammar = check_hebrew_grammar
        self.verify_citations = verify_citations
        self.logger = logging.getLogger(__name__)
    
    def run(self, answer: str, sources: List[Dict], question: str = "") -> Dict[str, Any]:
        """
        Validate Hebrew answer quality
        
        Args:
            answer: Generated Hebrew answer
            sources: Source documents used
            question: Original question (optional)
            
        Returns:
            Dict containing validation results
        """
        try:
            validation_results = {
                "overall_score": 0.0,
                "grammar_check": {},
                "citation_check": {},
                "content_quality": {},
                "recommendations": []
            }
            
            # Grammar validation
            if self.check_hebrew_grammar:
                validation_results["grammar_check"] = self.validate_hebrew_grammar(answer)
            
            # Citation validation
            if self.verify_citations:
                validation_results["citation_check"] = self.validate_citations(answer, sources)
            
            # Content quality check
            validation_results["content_quality"] = self.validate_content_quality(answer, question)
            
            # Calculate overall score
            validation_results["overall_score"] = self.calculate_overall_score(validation_results)
            
            # Generate recommendations
            validation_results["recommendations"] = self.generate_recommendations(validation_results)
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating answer: {e}")
            return {
                "overall_score": 0.0,
                "error": str(e),
                "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer
            }
    
    def validate_hebrew_grammar(self, answer: str) -> Dict[str, Any]:
        """Validate Hebrew grammar and language quality"""
        grammar_check = {
            "has_hebrew": False,
            "text_direction": "unknown",
            "sentence_structure": "good",
            "language_mixing": False,
            "score": 0.0
        }
        
        try:
            # Check for Hebrew characters
            hebrew_chars = set(range(0x0590, 0x05FF))
            has_hebrew = any(ord(char) in hebrew_chars for char in answer)
            grammar_check["has_hebrew"] = has_hebrew
            
            if has_hebrew:
                grammar_check["score"] += 0.4
                
                # Check for proper sentence structure
                sentences = [s.strip() for s in answer.split('.') if s.strip()]
                if sentences:
                    avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
                    if 5 <= avg_length <= 25:  # Reasonable sentence length
                        grammar_check["sentence_structure"] = "good"
                        grammar_check["score"] += 0.3
                    elif avg_length < 5:
                        grammar_check["sentence_structure"] = "too_short"
                        grammar_check["score"] += 0.1
                    else:
                        grammar_check["sentence_structure"] = "too_long"
                        grammar_check["score"] += 0.2
                
                # Check for language mixing (Hebrew + English in unnatural way)
                words = answer.split()
                hebrew_words = [w for w in words if any(ord(c) in hebrew_chars for c in w)]
                english_words = [w for w in words if w.isascii() and w.isalpha()]
                
                if hebrew_words and english_words:
                    mixing_ratio = len(english_words) / len(words)
                    if mixing_ratio > 0.3:  # More than 30% English
                        grammar_check["language_mixing"] = True
                        grammar_check["score"] -= 0.1
                
                # Detect text direction
                if len(hebrew_words) > len(english_words):
                    grammar_check["text_direction"] = "rtl"
                    grammar_check["score"] += 0.2
                else:
                    grammar_check["text_direction"] = "mixed"
                    grammar_check["score"] += 0.1
            
        except Exception as e:
            self.logger.error(f"Error in grammar validation: {e}")
        
        return grammar_check
    
    def validate_citations(self, answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """Validate citation accuracy and format"""
        citation_check = {
            "has_citations": False,
            "citation_format": "none",
            "source_accuracy": True,
            "citation_count": 0,
            "score": 0.0
        }
        
        try:
            # Look for citation patterns
            citation_patterns = [
                r'\([^)]+\)',  # (source)
                r'\[[^\]]+\]',  # [source]
                r'מקור[ות]?:',   # Hebrew "sources:"
            ]
            
            citations_found = []
            for pattern in citation_patterns:
                matches = re.findall(pattern, answer)
                citations_found.extend(matches)
            
            if citations_found:
                citation_check["has_citations"] = True
                citation_check["citation_count"] = len(citations_found)
                citation_check["score"] += 0.5
                
                # Determine citation format
                if any('(' in cite for cite in citations_found):
                    citation_check["citation_format"] = "academic"
                elif any('[' in cite for cite in citations_found):
                    citation_check["citation_format"] = "bracketed"
                elif any('מקור' in cite for cite in citations_found):
                    citation_check["citation_format"] = "hebrew"
                
                citation_check["score"] += 0.3
                
                # Check if number of citations is reasonable
                source_count = len(sources)
                cite_count = citation_check["citation_count"]
                if cite_count <= source_count:
                    citation_check["source_accuracy"] = True
                    citation_check["score"] += 0.2
                else:
                    citation_check["source_accuracy"] = False
            
        except Exception as e:
            self.logger.error(f"Error in citation validation: {e}")
        
        return citation_check
    
    def validate_content_quality(self, answer: str, question: str = "") -> Dict[str, Any]:
        """Validate content quality and relevance"""
        content_check = {
            "length_appropriate": False,
            "relevance_score": 0.0,
            "completeness": 0.0,
            "clarity": 0.0,
            "score": 0.0
        }
        
        try:
            # Check length appropriateness
            word_count = len(answer.split())
            if 20 <= word_count <= 300:  # Reasonable length for most answers
                content_check["length_appropriate"] = True
                content_check["score"] += 0.2
            
            # Assess relevance if question provided
            if question:
                question_words = set(question.lower().split())
                answer_words = set(answer.lower().split())
                
                # Remove stop words (basic Hebrew stop words)
                stop_words = {'של', 'את', 'על', 'אל', 'מן', 'ב', 'ל', 'כ', 'ה', 'ו', 'ש'}
                question_words = question_words - stop_words
                answer_words = answer_words - stop_words
                
                if question_words and answer_words:
                    overlap = len(question_words.intersection(answer_words))
                    relevance = overlap / len(question_words)
                    content_check["relevance_score"] = relevance
                    content_check["score"] += relevance * 0.3
            
            # Assess completeness (has proper structure)
            sentences = [s.strip() for s in answer.split('.') if s.strip()]
            if sentences:
                if len(sentences) >= 2:  # Multi-sentence answer
                    content_check["completeness"] = 0.8
                    content_check["score"] += 0.2
                else:
                    content_check["completeness"] = 0.5
                    content_check["score"] += 0.1
            
            # Assess clarity (no obvious errors or confusion)
            error_indicators = ['שגיאה', 'error', 'לא מצא', 'לא זמין']
            has_errors = any(indicator in answer.lower() for indicator in error_indicators)
            
            if not has_errors:
                content_check["clarity"] = 0.9
                content_check["score"] += 0.3
            else:
                content_check["clarity"] = 0.3
            
        except Exception as e:
            self.logger.error(f"Error in content validation: {e}")
        
        return content_check
    
    def calculate_overall_score(self, validation_results: Dict) -> float:
        """Calculate overall validation score"""
        try:
            scores = []
            
            # Grammar score (weight: 0.3)
            grammar_score = validation_results.get("grammar_check", {}).get("score", 0.0)
            scores.append(grammar_score * 0.3)
            
            # Citation score (weight: 0.3)
            citation_score = validation_results.get("citation_check", {}).get("score", 0.0)
            scores.append(citation_score * 0.3)
            
            # Content score (weight: 0.4)
            content_score = validation_results.get("content_quality", {}).get("score", 0.0)
            scores.append(content_score * 0.4)
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall score: {e}")
            return 0.0
    
    def generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        try:
            # Grammar recommendations
            grammar = validation_results.get("grammar_check", {})
            if not grammar.get("has_hebrew", False):
                recommendations.append("התשובה צריכה לכלול יותר תוכן בעברית")
            
            if grammar.get("sentence_structure") == "too_short":
                recommendations.append("הרחב את המשפטים לתשובה מפורטת יותר")
            elif grammar.get("sentence_structure") == "too_long":
                recommendations.append("פרק משפטים ארוכים לחלקים קצרים יותר")
            
            if grammar.get("language_mixing", False):
                recommendations.append("הפחת עירוב שפות ושמור על עברית עקבית")
            
            # Citation recommendations
            citation = validation_results.get("citation_check", {})
            if not citation.get("has_citations", False):
                recommendations.append("הוסף ציטוטים ומקורות לתשובה")
            
            if not citation.get("source_accuracy", True):
                recommendations.append("בדוק התאמה בין מספר הציטוטים למספר המקורות")
            
            # Content recommendations
            content = validation_results.get("content_quality", {})
            if not content.get("length_appropriate", False):
                recommendations.append("התאם את אורך התשובה - לא קצרה מדי ולא ארוכה מדי")
            
            if content.get("relevance_score", 0.0) < 0.5:
                recommendations.append("הגבר את הרלוונטיות לשאלה המקורית")
            
            if content.get("completeness", 0.0) < 0.7:
                recommendations.append("הוסף פרטים נוספים להשלמת התשובה")
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("בדוק את איכות התשובה באופן כללי")
        
        return recommendations