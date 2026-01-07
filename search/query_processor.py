"""
Query Processor for Deep Research Agent
Handles query rewriting, expansion, and optimization.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import re


@dataclass
class ProcessedQuery:
    """A processed and enhanced query"""
    original: str
    expanded: str
    variants: List[str]
    keywords: List[str]
    filters: Dict[str, Any]
    search_hints: Dict[str, Any]


class QueryProcessor:
    """
    Processes and enhances search queries for better results.
    Implements query expansion, synonym injection, and multi-perspective generation.
    """
    
    def __init__(self):
        # Common synonyms for query expansion
        self._synonyms = {
            "best": ["top", "leading", "optimal", "premier"],
            "compare": ["versus", "vs", "comparison", "difference"],
            "how": ["method", "technique", "approach", "way"],
            "why": ["reason", "cause", "explanation"],
            "what": ["definition", "meaning", "explanation"],
            "problem": ["issue", "challenge", "difficulty"],
            "solution": ["fix", "resolution", "answer"],
            "improve": ["enhance", "optimize", "boost"],
            "create": ["build", "develop", "make", "generate"],
            "use": ["utilize", "employ", "apply"],
        }
        
        # Domain-specific keywords
        self._domain_keywords = {
            "technology": ["software", "hardware", "system", "platform", "tool"],
            "finance": ["market", "investment", "trading", "stock", "crypto"],
            "science": ["research", "study", "experiment", "theory", "hypothesis"],
            "health": ["medical", "treatment", "disease", "wellness", "therapy"],
            "business": ["company", "startup", "enterprise", "industry", "market"],
        }
        
        # Operators for boolean queries
        self._boolean_ops = {"AND", "OR", "NOT", "-", "+", '"'}
    
    def process(self, query: str) -> ProcessedQuery:
        """Process a query and generate enhanced versions"""
        # Clean and normalize
        cleaned = self._clean_query(query)
        
        # Extract keywords
        keywords = self._extract_keywords(cleaned)
        
        # Detect filters (dates, sites, etc.)
        filters = self._detect_filters(query)
        
        # Generate expanded query
        expanded = self._expand_query(cleaned, keywords)
        
        # Generate variants
        variants = self._generate_variants(cleaned, keywords)
        
        # Generate search hints
        hints = self._generate_hints(cleaned, keywords)
        
        return ProcessedQuery(
            original=query,
            expanded=expanded,
            variants=variants,
            keywords=keywords,
            filters=filters,
            search_hints=hints
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize a query"""
        # Remove extra whitespace
        cleaned = " ".join(query.split())
        
        # Remove common filler words at start
        fillers = ["please", "can you", "i want to", "i need to", "help me"]
        for filler in fillers:
            if cleaned.lower().startswith(filler):
                cleaned = cleaned[len(filler):].strip()
        
        return cleaned
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove stopwords
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can",
            "about", "above", "after", "again", "all", "also", "and",
            "any", "as", "at", "because", "before", "below", "between",
            "both", "but", "by", "for", "from", "here", "how", "if",
            "in", "into", "it", "its", "just", "like", "more", "most",
            "no", "not", "now", "of", "on", "only", "or", "other", "our",
            "out", "over", "own", "same", "so", "some", "such", "than",
            "that", "their", "them", "then", "there", "these", "they",
            "this", "those", "through", "to", "too", "under", "up", "very",
            "we", "what", "when", "where", "which", "while", "who", "why",
            "with", "you", "your"
        }
        
        # Tokenize
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter stopwords and short words
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords
    
    def _detect_filters(self, query: str) -> Dict[str, Any]:
        """Detect search filters from query"""
        filters = {}
        
        # Date filters
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            filters["year"] = int(year_match.group(1))
        
        # Recent filter
        if any(word in query.lower() for word in ["recent", "latest", "new", "current"]):
            filters["recency"] = "recent"
        
        # Site filter
        site_match = re.search(r'site:(\S+)', query)
        if site_match:
            filters["site"] = site_match.group(1)
        
        # File type filter
        type_match = re.search(r'filetype:(\w+)', query)
        if type_match:
            filters["filetype"] = type_match.group(1)
        
        # Language detection (simple)
        if any(word in query.lower() for word in ["english", "en"]):
            filters["language"] = "en"
        
        return filters
    
    def _expand_query(self, query: str, keywords: List[str]) -> str:
        """Expand query with synonyms and related terms"""
        expanded_parts = [query]
        
        # Add synonyms for key terms
        for keyword in keywords[:3]:  # Limit to top 3 keywords
            if keyword in self._synonyms:
                synonyms = self._synonyms[keyword][:2]  # Add up to 2 synonyms
                expanded_parts.extend(synonyms)
        
        return " ".join(expanded_parts)
    
    def _generate_variants(self, query: str, keywords: List[str]) -> List[str]:
        """Generate query variants for broader coverage"""
        variants = []
        
        # Question form
        if not query.endswith("?"):
            variants.append(f"what is {query}?")
            variants.append(f"how does {query} work?")
        
        # Comparison form
        if len(keywords) >= 2:
            variants.append(f"{keywords[0]} vs {keywords[1]}")
        
        # Tutorial/guide form
        variants.append(f"{query} tutorial")
        variants.append(f"{query} guide")
        variants.append(f"{query} explained")
        
        # Academic form
        variants.append(f"{query} research paper")
        variants.append(f"{query} study")
        
        # Recent form
        variants.append(f"{query} 2025")
        variants.append(f"latest {query}")
        
        return variants[:5]  # Limit variants
    
    def _generate_hints(self, query: str, keywords: List[str]) -> Dict[str, Any]:
        """Generate hints for search strategy"""
        hints = {
            "suggested_depth": "normal",
            "suggested_sources": ["web"],
            "is_factual": False,
            "is_comparative": False,
            "is_tutorial": False,
            "is_academic": False,
        }
        
        query_lower = query.lower()
        
        # Detect query intent
        if any(word in query_lower for word in ["what is", "define", "meaning", "definition"]):
            hints["is_factual"] = True
            hints["suggested_depth"] = "shallow"
        
        if any(word in query_lower for word in ["vs", "versus", "compare", "difference", "better"]):
            hints["is_comparative"] = True
            hints["suggested_sources"].append("web")
        
        if any(word in query_lower for word in ["how to", "tutorial", "guide", "step by step"]):
            hints["is_tutorial"] = True
            hints["suggested_depth"] = "deep"
        
        if any(word in query_lower for word in ["research", "study", "paper", "scientific", "academic"]):
            hints["is_academic"] = True
            hints["suggested_sources"].append("academic")
        
        # Technical queries might benefit from local search
        if any(word in query_lower for word in ["code", "programming", "function", "api", "library"]):
            hints["suggested_sources"].append("local")
        
        return hints
    
    def generate_boolean_query(self, must_have: List[str], 
                                should_have: List[str] = None,
                                must_not: List[str] = None) -> str:
        """Generate a boolean search query"""
        parts = []
        
        # Must have terms
        for term in must_have:
            if " " in term:
                parts.append(f'"{term}"')
            else:
                parts.append(f"+{term}")
        
        # Should have terms
        if should_have:
            for term in should_have:
                parts.append(term)
        
        # Must not have terms
        if must_not:
            for term in must_not:
                parts.append(f"-{term}")
        
        return " ".join(parts)
    
    def split_compound_query(self, query: str) -> List[str]:
        """Split a compound query into simpler queries"""
        # Split on common conjunctions
        parts = re.split(r'\s+and\s+|\s+also\s+|\s*,\s*', query, flags=re.IGNORECASE)
        
        # Clean each part
        return [p.strip() for p in parts if p.strip()]
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract named entities from query"""
        entities = {
            "organizations": [],
            "technologies": [],
            "concepts": [],
            "people": [],
        }
        
        # Simple pattern matching for common entity types
        # (In production, use NER from spaCy or similar)
        
        # Organizations (capitalized multi-word)
        org_pattern = r'\b([A-Z][a-zA-Z]+ (?:Inc|Corp|Ltd|LLC|Company|Co)\.?)\b'
        entities["organizations"] = re.findall(org_pattern, query)
        
        # Technologies (common tech terms)
        tech_terms = ["python", "javascript", "react", "node", "api", "machine learning",
                      "ai", "neural network", "blockchain", "cloud", "kubernetes", "docker"]
        for term in tech_terms:
            if term.lower() in query.lower():
                entities["technologies"].append(term)
        
        return entities
