# Hebrew Table and Visual Analysis Tools
from agno.tools import Tool
from typing import List, Dict, Optional, Any, Union
import logging
import pandas as pd
import re
from collections import defaultdict

class HebrewTableAnalyzer(Tool):
    """
    Analyze tables in Hebrew documents and extract structured data.
    """
    
    def __init__(self, extract_structure: bool = True, generate_descriptions: bool = True):
        super().__init__(
            name="hebrew_table_analyzer",
            description="Analyze tables in Hebrew documents and extract structured data"
        )
        self.extract_structure = extract_structure
        self.generate_descriptions = generate_descriptions
        self.logger = logging.getLogger(__name__)
    
    def run(self, table_data: Dict, context: str = "") -> Dict[str, Any]:
        """
        Analyze table in Hebrew context and extract insights
        
        Args:
            table_data: Table structure and content
            context: Surrounding Hebrew text context
            
        Returns:
            Dict containing table analysis and Hebrew description
        """
        try:
            # Extract table structure
            structure = self.analyze_table_structure(table_data)
            
            # Generate Hebrew description
            hebrew_description = self.generate_hebrew_description(
                table_data, context, structure
            )
            
            # Extract key insights
            insights = self.extract_table_insights(table_data, context)
            
            # Summarize data
            data_summary = self.summarize_data(table_data)
            
            return {
                "status": "success",
                "table_structure": structure,
                "hebrew_description": hebrew_description,
                "insights": insights,
                "data_summary": data_summary,
                "context": context
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing table: {e}")
            return {
                "status": "error",
                "error": str(e),
                "table_data": str(table_data)[:200] + "..."
            }
    
    def analyze_table_structure(self, table_data: Dict) -> Dict[str, Any]:
        """Extract and analyze table structure"""
        structure = {
            "rows": 0,
            "columns": 0,
            "headers": [],
            "data_types": {},
            "has_hebrew": False,
            "has_numbers": False
        }
        
        try:
            content = table_data.get('content', [])
            if not content:
                return structure
            
            # Basic structure
            structure["rows"] = len(content)
            structure["columns"] = max(len(row) for row in content) if content else 0
            
            # Extract headers (first row)
            if content:
                headers = content[0]
                structure["headers"] = [self.clean_cell_text(cell) for cell in headers]
                
                # Check if headers contain Hebrew
                for header in structure["headers"]:
                    if self.contains_hebrew(header):
                        structure["has_hebrew"] = True
                        break
            
            # Analyze data types in each column
            if len(content) > 1:  # Skip header row
                for col_idx in range(structure["columns"]):
                    column_data = []
                    for row_idx in range(1, len(content)):
                        if col_idx < len(content[row_idx]):
                            cell_value = self.clean_cell_text(content[row_idx][col_idx])
                            column_data.append(cell_value)
                    
                    # Determine column type
                    col_name = structure["headers"][col_idx] if col_idx < len(structure["headers"]) else f"Column_{col_idx}"
                    structure["data_types"][col_name] = self.determine_column_type(column_data)
                    
                    # Check for Hebrew and numbers
                    for cell in column_data:
                        if self.contains_hebrew(cell):
                            structure["has_hebrew"] = True
                        if self.contains_numbers(cell):
                            structure["has_numbers"] = True
            
        except Exception as e:
            self.logger.error(f"Error analyzing table structure: {e}")
        
        return structure
    
    def clean_cell_text(self, cell: Any) -> str:
        """Clean and normalize cell text"""
        if cell is None:
            return ""
        
        text = str(cell).strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def contains_hebrew(self, text: str) -> bool:
        """Check if text contains Hebrew characters"""
        if not text:
            return False
        hebrew_chars = set(range(0x0590, 0x05FF))
        return any(ord(char) in hebrew_chars for char in text)
    
    def contains_numbers(self, text: str) -> bool:
        """Check if text contains numbers"""
        return bool(re.search(r'\d', text))
    
    def determine_column_type(self, column_data: List[str]) -> str:
        """Determine the type of data in a column"""
        if not column_data:
            return "empty"
        
        # Count different types
        numeric_count = 0
        hebrew_count = 0
        date_count = 0
        
        for value in column_data:
            if not value:
                continue
                
            # Check for numbers (including Hebrew-formatted numbers)
            if self.is_numeric_value(value):
                numeric_count += 1
            elif self.contains_hebrew(value):
                hebrew_count += 1
            elif self.is_date_value(value):
                date_count += 1
        
        total_non_empty = len([v for v in column_data if v])
        
        if total_non_empty == 0:
            return "empty"
        
        # Determine predominant type
        if numeric_count / total_non_empty > 0.7:
            return "numeric"
        elif hebrew_count / total_non_empty > 0.7:
            return "hebrew_text"
        elif date_count / total_non_empty > 0.7:
            return "date"
        else:
            return "mixed"
    
    def is_numeric_value(self, value: str) -> bool:
        """Check if value is numeric (including Hebrew formatting)"""
        if not value:
            return False
        
        # Remove common formatting
        cleaned = value.replace(',', '').replace('₪', '').replace('%', '').strip()
        
        try:
            float(cleaned)
            return True
        except ValueError:
            return False
    
    def is_date_value(self, value: str) -> bool:
        """Check if value looks like a date"""
        if not value:
            return False
        
        # Common date patterns
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',  # DD/MM/YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',  # DD-MM-YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, value):
                return True
        
        return False
    
    def generate_hebrew_description(self, table_data: Dict, context: str, structure: Dict) -> str:
        """Generate Hebrew description of the table"""
        try:
            parts = []
            
            # Basic structure description
            rows = structure.get("rows", 0)
            cols = structure.get("columns", 0)
            parts.append(f"טבלה המכילה {rows} שורות ו-{cols} עמודות")
            
            # Headers description
            headers = structure.get("headers", [])
            if headers:
                hebrew_headers = [h for h in headers if self.contains_hebrew(h)]
                if hebrew_headers:
                    parts.append(f"כותרות העמודות כוללות: {', '.join(hebrew_headers[:3])}")
            
            # Data types description
            data_types = structure.get("data_types", {})
            numeric_columns = [col for col, dtype in data_types.items() if dtype == "numeric"]
            hebrew_columns = [col for col, dtype in data_types.items() if dtype == "hebrew_text"]
            
            if numeric_columns:
                parts.append(f"עמודות מספריות: {len(numeric_columns)}")
            if hebrew_columns:
                parts.append(f"עמודות טקסט בעברית: {len(hebrew_columns)}")
            
            # Context integration
            if context and self.contains_hebrew(context):
                context_words = context.split()[:10]  # First 10 words
                if any(self.contains_hebrew(word) for word in context_words):
                    parts.append("הטבלה מופיעה בהקשר של " + " ".join(context_words[:5]))
            
            return ". ".join(parts) + "."
            
        except Exception as e:
            self.logger.error(f"Error generating Hebrew description: {e}")
            return "טבלה המכילה מידע"
    
    def extract_table_insights(self, table_data: Dict, context: str) -> List[str]:
        """Extract key insights from the table"""
        insights = []
        
        try:
            content = table_data.get('content', [])
            if len(content) < 2:  # Need at least header + 1 data row
                return insights
            
            structure = self.analyze_table_structure(table_data)
            
            # Numeric insights
            numeric_columns = [col for col, dtype in structure.get("data_types", {}).items() 
                             if dtype == "numeric"]
            
            for col_name in numeric_columns:
                col_idx = structure.get("headers", []).index(col_name) if col_name in structure.get("headers", []) else -1
                if col_idx >= 0:
                    values = []
                    for row in content[1:]:  # Skip header
                        if col_idx < len(row):
                            try:
                                val = float(self.clean_cell_text(row[col_idx]).replace(',', ''))
                                values.append(val)
                            except ValueError:
                                continue
                    
                    if values:
                        insights.append(f"עמודת '{col_name}': טווח מ-{min(values):.1f} עד {max(values):.1f}")
                        if len(values) > 1:
                            avg = sum(values) / len(values)
                            insights.append(f"ממוצע עמודת '{col_name}': {avg:.1f}")
            
            # Row count insight
            data_rows = len(content) - 1  # Exclude header
            insights.append(f"סה'כ {data_rows} רשומות נתונים")
            
        except Exception as e:
            self.logger.error(f"Error extracting insights: {e}")
        
        return insights
    
    def summarize_data(self, table_data: Dict) -> Dict[str, Any]:
        """Create a summary of the table data"""
        summary = {
            "total_cells": 0,
            "non_empty_cells": 0,
            "numeric_cells": 0,
            "hebrew_cells": 0,
            "key_values": []
        }
        
        try:
            content = table_data.get('content', [])
            
            for row in content:
                for cell in row:
                    summary["total_cells"] += 1
                    
                    cell_text = self.clean_cell_text(cell)
                    if cell_text:
                        summary["non_empty_cells"] += 1
                        
                        if self.is_numeric_value(cell_text):
                            summary["numeric_cells"] += 1
                        elif self.contains_hebrew(cell_text):
                            summary["hebrew_cells"] += 1
                            
                        # Collect interesting values
                        if len(cell_text) > 3 and len(summary["key_values"]) < 10:
                            summary["key_values"].append(cell_text)
            
            # Calculate percentages
            if summary["total_cells"] > 0:
                summary["fill_rate"] = summary["non_empty_cells"] / summary["total_cells"]
                summary["numeric_rate"] = summary["numeric_cells"] / summary["total_cells"]
                summary["hebrew_rate"] = summary["hebrew_cells"] / summary["total_cells"]
            
        except Exception as e:
            self.logger.error(f"Error summarizing data: {e}")
        
        return summary


class HebrewChartAnalyzer(Tool):
    """
    Analyze charts and graphs in Hebrew documents.
    """
    
    def __init__(self, identify_trends: bool = True, extract_data_points: bool = True):
        super().__init__(
            name="hebrew_chart_analyzer",
            description="Analyze charts and graphs in Hebrew documents"
        )
        self.identify_trends = identify_trends
        self.extract_data_points = extract_data_points
        self.logger = logging.getLogger(__name__)
    
    def run(self, chart_data: Dict, context: str = "") -> Dict[str, Any]:
        """
        Analyze chart/graph in Hebrew context
        
        Args:
            chart_data: Chart structure and metadata
            context: Surrounding Hebrew text context
            
        Returns:
            Dict containing chart analysis and Hebrew description
        """
        try:
            # Determine chart type
            chart_type = self.identify_chart_type(chart_data, context)
            
            # Generate Hebrew description
            hebrew_description = self.generate_chart_description(chart_data, context, chart_type)
            
            # Extract trends if applicable
            trends = []
            if self.identify_trends:
                trends = self.analyze_trends(chart_data, chart_type)
            
            # Extract data points if applicable
            data_points = []
            if self.extract_data_points:
                data_points = self.extract_key_data_points(chart_data, chart_type)
            
            return {
                "status": "success",
                "chart_type": chart_type,
                "hebrew_description": hebrew_description,
                "trends": trends,
                "data_points": data_points,
                "context": context
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing chart: {e}")
            return {
                "status": "error",
                "error": str(e),
                "chart_data": str(chart_data)[:200] + "..."
            }
    
    def identify_chart_type(self, chart_data: Dict, context: str) -> str:
        """Identify the type of chart based on data and context"""
        # This would typically analyze the chart image or metadata
        # For now, we'll use context clues
        
        context_lower = context.lower()
        
        if any(word in context_lower for word in ['גרף', 'תרשים', 'chart', 'graph']):
            if any(word in context_lower for word in ['עוגה', 'pie']):
                return "pie_chart"
            elif any(word in context_lower for word in ['עמודות', 'bar', 'column']):
                return "bar_chart"
            elif any(word in context_lower for word in ['קו', 'line', 'מגמה', 'trend']):
                return "line_chart"
            elif any(word in context_lower for word in ['פיזור', 'scatter']):
                return "scatter_plot"
            else:
                return "chart"
        else:
            return "unknown_visual"
    
    def generate_chart_description(self, chart_data: Dict, context: str, chart_type: str) -> str:
        """Generate Hebrew description of the chart"""
        descriptions = {
            "pie_chart": "תרשים עוגה המציג התפלגות נתונים",
            "bar_chart": "תרשים עמודות להשוואת ערכים",
            "line_chart": "גרף קווי המציג מגמות לאורך זמן",
            "scatter_plot": "גרף פיזור המציג קשר בין משתנים",
            "chart": "תרשים המציג נתונים בצורה ויזואלית",
            "unknown_visual": "אלמנט ויזואלי המכיל מידע"
        }
        
        base_description = descriptions.get(chart_type, descriptions["unknown_visual"])
        
        # Add context if available
        if context and self.contains_hebrew(context):
            context_words = context.split()[:8]
            hebrew_context = " ".join([word for word in context_words if self.contains_hebrew(word)][:3])
            if hebrew_context:
                base_description += f" בנושא {hebrew_context}"
        
        return base_description
    
    def contains_hebrew(self, text: str) -> bool:
        """Check if text contains Hebrew characters"""
        if not text:
            return False
        hebrew_chars = set(range(0x0590, 0x05FF))
        return any(ord(char) in hebrew_chars for char in text)
    
    def analyze_trends(self, chart_data: Dict, chart_type: str) -> List[str]:
        """Analyze trends in the chart data"""
        trends = []
        
        # This would typically analyze actual chart data
        # For now, we'll return generic trend descriptions based on type
        
        if chart_type == "line_chart":
            trends.append("מגמת שינוי לאורך זמן")
        elif chart_type == "bar_chart":
            trends.append("השוואה בין קטגוריות שונות")
        elif chart_type == "pie_chart":
            trends.append("התפלגות יחסית של נתונים")
        
        return trends
    
    def extract_key_data_points(self, chart_data: Dict, chart_type: str) -> List[Dict]:
        """Extract key data points from the chart"""
        data_points = []
        
        # This would typically extract actual data from the chart
        # For now, we'll return placeholder structure
        
        if chart_type in ["bar_chart", "line_chart"]:
            data_points.append({
                "type": "value",
                "description": "נקודת נתונים מרכזית",
                "context": "ערך חשוב בתרשים"
            })
        
        return data_points


class VisualContextualizer(Tool):
    """
    Link visual elements to their text context and generate Hebrew descriptions.
    """
    
    def __init__(self, link_to_text: bool = True, hebrew_descriptions: bool = True):
        super().__init__(
            name="visual_contextualizer",
            description="Link visual elements to text context and generate Hebrew descriptions"
        )
        self.link_to_text = link_to_text
        self.hebrew_descriptions = hebrew_descriptions
        self.logger = logging.getLogger(__name__)
    
    def run(self, visual_element: Dict, surrounding_text: List[str]) -> Dict[str, Any]:
        """
        Contextualize a visual element with surrounding text
        
        Args:
            visual_element: Visual element data (table, chart, image, etc.)
            surrounding_text: List of surrounding text snippets
            
        Returns:
            Dict containing contextualized information
        """
        try:
            # Find relevant text context
            relevant_context = self.find_relevant_context(visual_element, surrounding_text)
            
            # Generate Hebrew description
            hebrew_description = ""
            if self.hebrew_descriptions:
                hebrew_description = self.generate_contextual_description(
                    visual_element, relevant_context
                )
            
            # Link to text references
            text_links = []
            if self.link_to_text:
                text_links = self.find_text_references(visual_element, surrounding_text)
            
            return {
                "status": "success",
                "visual_type": visual_element.get("type", "unknown"),
                "hebrew_description": hebrew_description,
                "relevant_context": relevant_context,
                "text_references": text_links,
                "context_strength": self.calculate_context_strength(relevant_context)
            }
            
        except Exception as e:
            self.logger.error(f"Error contextualizing visual element: {e}")
            return {
                "status": "error",
                "error": str(e),
                "visual_element": str(visual_element)[:200] + "..."
            }
    
    def find_relevant_context(self, visual_element: Dict, surrounding_text: List[str]) -> List[str]:
        """Find text snippets most relevant to the visual element"""
        relevant = []
        
        visual_type = visual_element.get("type", "")
        
        # Keywords to look for based on visual type
        keywords = {
            "table": ["טבלה", "נתונים", "מידע", "רשימה", "פירוט"],
            "chart": ["תרשים", "גרף", "מגמה", "השוואה", "ביצועים"],
            "image": ["תמונה", "איור", "דיאגרמה", "צילום"],
            "graph": ["גרף", "נתונים", "סטטיסטיקה", "מדידה"]
        }
        
        search_keywords = keywords.get(visual_type, keywords.get("table", []))
        
        # Score each text snippet
        scored_texts = []
        for text in surrounding_text:
            if not text:
                continue
                
            score = 0
            text_lower = text.lower()
            
            # Score based on keyword presence
            for keyword in search_keywords:
                if keyword in text_lower:
                    score += 2
            
            # Score based on numbers (relevant for data visuals)
            if re.search(r'\d', text):
                score += 1
            
            # Score based on Hebrew content
            if self.contains_hebrew(text):
                score += 1
            
            if score > 0:
                scored_texts.append((text, score))
        
        # Sort by score and return top results
        scored_texts.sort(key=lambda x: x[1], reverse=True)
        relevant = [text for text, score in scored_texts[:3]]
        
        return relevant
    
    def contains_hebrew(self, text: str) -> bool:
        """Check if text contains Hebrew characters"""
        if not text:
            return False
        hebrew_chars = set(range(0x0590, 0x05FF))
        return any(ord(char) in hebrew_chars for char in text)
    
    def generate_contextual_description(self, visual_element: Dict, context: List[str]) -> str:
        """Generate Hebrew description incorporating context"""
        visual_type = visual_element.get("type", "unknown")
        
        # Base descriptions
        base_descriptions = {
            "table": "טבלה",
            "chart": "תרשים",
            "graph": "גרף", 
            "image": "תמונה",
            "diagram": "דיאגרמה"
        }
        
        description = base_descriptions.get(visual_type, "אלמנט ויזואלי")
        
        # Add context information
        if context:
            # Extract key terms from context
            key_terms = []
            for text in context:
                words = text.split()
                hebrew_words = [word for word in words if self.contains_hebrew(word)]
                key_terms.extend(hebrew_words[:3])
            
            if key_terms:
                unique_terms = list(dict.fromkeys(key_terms))[:3]  # Remove duplicates, keep order
                description += f" הקשורה ל{', '.join(unique_terms)}"
        
        return description
    
    def find_text_references(self, visual_element: Dict, surrounding_text: List[str]) -> List[Dict]:
        """Find specific text references to the visual element"""
        references = []
        
        # Look for explicit references
        reference_patterns = [
            r'ראה טבלה',
            r'לפי הטבלה',
            r'בתרשים',
            r'כפי שמוצג',
            r'כמתואר',
            r'להלן',
            r'לעיל'
        ]
        
        for i, text in enumerate(surrounding_text):
            if not text:
                continue
                
            for pattern in reference_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    references.append({
                        "text": text,
                        "position": i,
                        "reference_type": "explicit",
                        "pattern": pattern
                    })
        
        return references
    
    def calculate_context_strength(self, context: List[str]) -> float:
        """Calculate how strong the context connection is"""
        if not context:
            return 0.0
        
        strength = 0.0
        
        for text in context:
            if self.contains_hebrew(text):
                strength += 0.3
            if re.search(r'\d', text):
                strength += 0.2
            if len(text.split()) > 5:  # Substantial text
                strength += 0.2
        
        return min(strength, 1.0)  # Cap at 1.0