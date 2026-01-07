"""
Citation Memory for Deep Research Agent
Tracks all sources and manages citation formatting.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from urllib.parse import urlparse

from ..core.base import Source


class CitationMemory:
    """
    Manages citations and source tracking throughout research.
    Provides deduplication, reliability scoring, and formatted citations.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self._sources: Dict[str, Source] = {}
        self._url_to_id: Dict[str, str] = {}
        self._citations_used: Dict[str, int] = {}  # Track citation usage count
        self._reliability_cache: Dict[str, float] = {}
        
        if storage_path:
            self._load()
    
    def _load(self) -> None:
        """Load citations from disk"""
        if self.storage_path and self.storage_path.exists():
            citations_file = self.storage_path / "citations.json"
            if citations_file.exists():
                try:
                    with open(citations_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for source_data in data.get("sources", []):
                            source = Source(
                                id=source_data["id"],
                                url=source_data.get("url"),
                                title=source_data.get("title", ""),
                                author=source_data.get("author"),
                                published_date=datetime.fromisoformat(source_data["published_date"]) if source_data.get("published_date") else None,
                                accessed_date=datetime.fromisoformat(source_data["accessed_date"]) if source_data.get("accessed_date") else datetime.now(),
                                source_type=source_data.get("source_type", "web"),
                                reliability_score=source_data.get("reliability_score", 0.5),
                                content_snippet=source_data.get("content_snippet", ""),
                                metadata=source_data.get("metadata", {})
                            )
                            self._sources[source.id] = source
                            if source.url:
                                self._url_to_id[source.url] = source.id
                        self._citations_used = data.get("usage_counts", {})
                except Exception as e:
                    print(f"Warning: Could not load citations: {e}")
    
    def _save(self) -> None:
        """Save citations to disk"""
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        citations_file = self.storage_path / "citations.json"
        
        data = {
            "sources": [
                {
                    "id": s.id,
                    "url": s.url,
                    "title": s.title,
                    "author": s.author,
                    "published_date": s.published_date.isoformat() if s.published_date else None,
                    "accessed_date": s.accessed_date.isoformat(),
                    "source_type": s.source_type,
                    "reliability_score": s.reliability_score,
                    "content_snippet": s.content_snippet,
                    "metadata": s.metadata
                }
                for s in self._sources.values()
            ],
            "usage_counts": self._citations_used
        }
        
        with open(citations_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def add_source(self, source: Source) -> str:
        """Add a source to the citation memory"""
        # Check for duplicate by URL
        if source.url and source.url in self._url_to_id:
            existing_id = self._url_to_id[source.url]
            existing = self._sources[existing_id]
            # Update with new info if available
            if source.title and not existing.title:
                existing.title = source.title
            if source.author and not existing.author:
                existing.author = source.author
            return existing_id
        
        # Add new source
        self._sources[source.id] = source
        if source.url:
            self._url_to_id[source.url] = source.id
        
        self._save()
        return source.id
    
    def get_source(self, source_id: str) -> Optional[Source]:
        """Get a source by ID"""
        return self._sources.get(source_id)
    
    def get_by_url(self, url: str) -> Optional[Source]:
        """Get a source by URL"""
        source_id = self._url_to_id.get(url)
        if source_id:
            return self._sources.get(source_id)
        return None
    
    def cite(self, source_id: str) -> None:
        """Record that a source was cited"""
        if source_id in self._sources:
            self._citations_used[source_id] = self._citations_used.get(source_id, 0) + 1
            self._save()
    
    def get_citation(self, source_id: str, style: str = "apa") -> str:
        """Get formatted citation for a source"""
        source = self._sources.get(source_id)
        if not source:
            return f"[Unknown source: {source_id}]"
        
        self.cite(source_id)
        return source.to_citation(style)
    
    def get_citations_list(self, source_ids: List[str], style: str = "apa") -> List[str]:
        """Get list of formatted citations"""
        return [self.get_citation(sid, style) for sid in source_ids if sid in self._sources]
    
    def format_inline_citation(self, source_id: str, style: str = "numeric") -> str:
        """Format an inline citation"""
        source = self._sources.get(source_id)
        if not source:
            return "[?]"
        
        if style == "numeric":
            # Assign numbers based on first use
            if source_id not in self._citations_used:
                self._citations_used[source_id] = len(self._citations_used) + 1
            return f"[{self._citations_used[source_id]}]"
        elif style == "author_year":
            author = source.author or "Unknown"
            year = source.published_date.year if source.published_date else "n.d."
            return f"({author}, {year})"
        else:
            return f"[{source_id[:8]}]"
    
    def estimate_reliability(self, source: Source) -> float:
        """Estimate reliability score for a source"""
        if source.url in self._reliability_cache:
            return self._reliability_cache[source.url]
        
        score = 0.5  # Default
        
        if source.url:
            domain = urlparse(source.url).netloc.lower()
            
            # High reliability domains
            high_reliability = [
                "gov", "edu", "nature.com", "science.org", "pubmed",
                "ieee.org", "acm.org", "springer.com", "wiley.com",
                "reuters.com", "apnews.com", "bbc.com", "nytimes.com"
            ]
            
            # Medium reliability
            medium_reliability = [
                "wikipedia.org", "britannica.com", "techcrunch.com",
                "wired.com", "arstechnica.com", "medium.com"
            ]
            
            # Low reliability
            low_reliability = [
                "reddit.com", "quora.com", "yahoo.com", "blog"
            ]
            
            for pattern in high_reliability:
                if pattern in domain:
                    score = 0.85
                    break
            else:
                for pattern in medium_reliability:
                    if pattern in domain:
                        score = 0.65
                        break
                else:
                    for pattern in low_reliability:
                        if pattern in domain:
                            score = 0.35
                            break
        
        # Boost for academic sources
        if source.source_type == "academic":
            score = min(1.0, score + 0.2)
        
        # Boost for recent sources
        if source.published_date:
            age_days = (datetime.now() - source.published_date).days
            if age_days < 365:
                score = min(1.0, score + 0.1)
            elif age_days > 1825:  # 5 years
                score = max(0.2, score - 0.1)
        
        self._reliability_cache[source.url] = score
        source.reliability_score = score
        return score
    
    def get_most_reliable(self, n: int = 10) -> List[Source]:
        """Get the most reliable sources"""
        sources = list(self._sources.values())
        sources.sort(key=lambda s: -s.reliability_score)
        return sources[:n]
    
    def get_most_cited(self, n: int = 10) -> List[Source]:
        """Get the most frequently cited sources"""
        sorted_ids = sorted(
            self._citations_used.keys(),
            key=lambda x: -self._citations_used[x]
        )
        return [self._sources[sid] for sid in sorted_ids[:n] if sid in self._sources]
    
    def generate_bibliography(self, style: str = "apa") -> str:
        """Generate a complete bibliography"""
        cited_sources = [
            (self._citations_used.get(sid, 0), sid, source)
            for sid, source in self._sources.items()
        ]
        cited_sources.sort(key=lambda x: -x[0])  # Sort by citation count
        
        lines = ["## References", ""]
        for count, sid, source in cited_sources:
            if count > 0:
                lines.append(f"- {source.to_citation(style)}")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get citation statistics"""
        sources = list(self._sources.values())
        
        if not sources:
            return {"total_sources": 0}
        
        return {
            "total_sources": len(sources),
            "cited_sources": len([s for s in self._citations_used.values() if s > 0]),
            "total_citations": sum(self._citations_used.values()),
            "avg_reliability": sum(s.reliability_score for s in sources) / len(sources),
            "source_types": {},
            "domains": {}
        }
    
    def clear(self) -> None:
        """Clear all citations"""
        self._sources.clear()
        self._url_to_id.clear()
        self._citations_used.clear()
        self._reliability_cache.clear()
        self._save()
