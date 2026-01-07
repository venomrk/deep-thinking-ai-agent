"""
Deep Search Engine for Deep Research Agent
Multi-source search orchestration with query enhancement.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import json
import re

from ..core.base import Source, BaseModule, SearchType
from ..config.settings import get_settings, SearchProvider


@dataclass
class SearchResult:
    """A single search result"""
    url: str
    title: str
    snippet: str
    source_type: str = "web"
    relevance_score: float = 0.5
    published_date: Optional[datetime] = None
    author: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_source(self) -> Source:
        """Convert to Source object"""
        return Source(
            url=self.url,
            title=self.title,
            content_snippet=self.snippet,
            source_type=self.source_type,
            reliability_score=self.relevance_score,
            published_date=self.published_date,
            author=self.author,
            metadata=self.metadata
        )


class BaseSearchProvider(ABC):
    """Abstract base class for search providers"""
    
    def __init__(self, name: str):
        self.name = name
        self._initialized = False
    
    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Execute a search query"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available"""
        pass


class DuckDuckGoSearch(BaseSearchProvider):
    """DuckDuckGo search provider (no API key required)"""
    
    def __init__(self):
        super().__init__("duckduckgo")
    
    def is_available(self) -> bool:
        try:
            from duckduckgo_search import DDGS
            return True
        except ImportError:
            return False
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search using DuckDuckGo"""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num_results):
                    results.append(SearchResult(
                        url=r.get('href', ''),
                        title=r.get('title', ''),
                        snippet=r.get('body', ''),
                        source_type='web',
                        relevance_score=0.5
                    ))
            return results
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []


class WebScraperSearch(BaseSearchProvider):
    """Fallback web scraper using httpx and BeautifulSoup"""
    
    def __init__(self):
        super().__init__("scraper")
    
    def is_available(self) -> bool:
        try:
            import httpx
            return True
        except ImportError:
            return False
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search by scraping Google (fallback)"""
        try:
            import httpx
            from urllib.parse import quote_plus
            
            # Use lite.duckduckgo.com for simple scraping
            url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30)
                
                if response.status_code != 200:
                    return []
                
                # Parse simple results from lite page
                content = response.text
                results = []
                
                # Extract links and snippets from lite HTML
                import re
                links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', content)
                
                for url, title in links[:num_results]:
                    if url.startswith('http') and not 'duckduckgo' in url:
                        results.append(SearchResult(
                            url=url,
                            title=title.strip(),
                            snippet="",
                            source_type='web',
                            relevance_score=0.4
                        ))
                
                return results
        except Exception as e:
            print(f"Web scraper search error: {e}")
            return []


class AcademicSearch(BaseSearchProvider):
    """Academic paper search using Semantic Scholar API"""
    
    def __init__(self):
        super().__init__("academic")
        self.base_url = "https://api.semanticscholar.org/graph/v1"
    
    def is_available(self) -> bool:
        return True  # Public API, no key required
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search academic papers"""
        try:
            import httpx
            from urllib.parse import quote_plus
            
            url = f"{self.base_url}/paper/search"
            params = {
                "query": query,
                "limit": num_results,
                "fields": "title,abstract,authors,year,url,citationCount"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=30)
                
                if response.status_code != 200:
                    return []
                
                data = response.json()
                results = []
                
                for paper in data.get("data", []):
                    # Calculate relevance based on citations
                    citations = paper.get("citationCount", 0)
                    relevance = min(1.0, 0.5 + (citations / 1000))
                    
                    authors = paper.get("authors", [])
                    author_str = ", ".join(a.get("name", "") for a in authors[:3])
                    if len(authors) > 3:
                        author_str += " et al."
                    
                    year = paper.get("year")
                    pub_date = datetime(year, 1, 1) if year else None
                    
                    results.append(SearchResult(
                        url=paper.get("url", f"https://api.semanticscholar.org/paper/{paper.get('paperId', '')}"),
                        title=paper.get("title", ""),
                        snippet=paper.get("abstract", "")[:500],
                        source_type="academic",
                        relevance_score=relevance,
                        published_date=pub_date,
                        author=author_str,
                        metadata={
                            "citations": citations,
                            "paper_id": paper.get("paperId")
                        }
                    ))
                
                return results
        except Exception as e:
            print(f"Academic search error: {e}")
            return []


class LocalDocumentSearch(BaseSearchProvider):
    """Search local documents in workspace"""
    
    def __init__(self, search_paths: List[str] = None):
        super().__init__("local")
        self.search_paths = search_paths or ["."]
    
    def is_available(self) -> bool:
        return True
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search local files"""
        from pathlib import Path
        import os
        
        results = []
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        for search_path in self.search_paths:
            path = Path(search_path)
            if not path.exists():
                continue
            
            # Search Python, Markdown, and text files
            extensions = {'.py', '.md', '.txt', '.json', '.yaml', '.yml'}
            
            for file_path in path.rglob('*'):
                if file_path.suffix not in extensions:
                    continue
                if any(x in str(file_path) for x in ['.git', '__pycache__', '.venv', 'node_modules']):
                    continue
                
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')[:10000]
                    content_lower = content.lower()
                    
                    # Simple relevance scoring
                    matches = sum(1 for term in query_terms if term in content_lower)
                    if matches > 0:
                        relevance = matches / len(query_terms)
                        
                        # Extract snippet around first match
                        snippet = ""
                        for term in query_terms:
                            idx = content_lower.find(term)
                            if idx >= 0:
                                start = max(0, idx - 100)
                                end = min(len(content), idx + 200)
                                snippet = content[start:end]
                                break
                        
                        results.append(SearchResult(
                            url=f"file://{file_path.absolute()}",
                            title=file_path.name,
                            snippet=snippet,
                            source_type="local",
                            relevance_score=relevance,
                            metadata={"path": str(file_path.absolute())}
                        ))
                except Exception:
                    continue
        
        # Sort by relevance and limit
        results.sort(key=lambda x: -x.relevance_score)
        return results[:num_results]


class DeepSearchEngine(BaseModule):
    """
    Multi-source search engine with query enhancement and result aggregation.
    """
    
    def __init__(self):
        super().__init__("search_engine")
        self._providers: Dict[str, BaseSearchProvider] = {}
        self._query_history: List[str] = []
        self._result_cache: Dict[str, List[SearchResult]] = {}
        self._settings = get_settings()
    
    async def initialize(self) -> None:
        """Initialize search providers"""
        # Add DuckDuckGo (free, no API key)
        ddg = DuckDuckGoSearch()
        if ddg.is_available():
            self._providers["duckduckgo"] = ddg
        
        # Add web scraper fallback
        scraper = WebScraperSearch()
        if scraper.is_available():
            self._providers["scraper"] = scraper
        
        # Add academic search
        self._providers["academic"] = AcademicSearch()
        
        # Add local search
        self._providers["local"] = LocalDocumentSearch()
        
        self._initialized = True
    
    async def shutdown(self) -> None:
        """Clean shutdown"""
        self._providers.clear()
        self._initialized = False
    
    async def search(self, query: str, search_types: List[SearchType] = None,
                     num_results: int = 10) -> List[SearchResult]:
        """
        Execute a multi-source search.
        
        Args:
            query: The search query
            search_types: Types of search to perform (web, academic, local)
            num_results: Maximum results per source
            
        Returns:
            List of aggregated, deduplicated results
        """
        if not self._initialized:
            await self.initialize()
        
        search_types = search_types or [SearchType.WEB]
        
        # Check cache
        cache_key = f"{query}:{','.join(st.value for st in search_types)}"
        if cache_key in self._result_cache:
            return self._result_cache[cache_key]
        
        # Record query
        self._query_history.append(query)
        
        # Execute searches in parallel
        tasks = []
        provider_names = []
        
        for search_type in search_types:
            if search_type == SearchType.WEB:
                if "duckduckgo" in self._providers:
                    tasks.append(self._providers["duckduckgo"].search(query, num_results))
                    provider_names.append("duckduckgo")
                elif "scraper" in self._providers:
                    tasks.append(self._providers["scraper"].search(query, num_results))
                    provider_names.append("scraper")
            
            elif search_type == SearchType.ACADEMIC:
                if "academic" in self._providers:
                    tasks.append(self._providers["academic"].search(query, num_results))
                    provider_names.append("academic")
            
            elif search_type == SearchType.LOCAL:
                if "local" in self._providers:
                    tasks.append(self._providers["local"].search(query, num_results))
                    provider_names.append("local")
        
        # Execute all searches
        all_results = []
        if tasks:
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            for provider_name, results in zip(provider_names, results_list):
                if isinstance(results, Exception):
                    print(f"Search error from {provider_name}: {results}")
                    continue
                all_results.extend(results)
        
        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Sort by relevance
        unique_results.sort(key=lambda x: -x.relevance_score)
        
        # Cache results
        self._result_cache[cache_key] = unique_results[:num_results * 2]
        
        return unique_results[:num_results * 2]
    
    async def multi_query_search(self, queries: List[str], 
                                  search_types: List[SearchType] = None,
                                  num_results_per_query: int = 5) -> List[SearchResult]:
        """Execute multiple queries and aggregate results"""
        all_results = []
        
        tasks = [self.search(q, search_types, num_results_per_query) for q in queries]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        for query, results in zip(queries, results_list):
            if isinstance(results, Exception):
                print(f"Multi-query search error for '{query}': {results}")
                continue
            all_results.extend(results)
        
        # Deduplicate and sort
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: -x.relevance_score)
        return unique_results
    
    def clear_cache(self) -> None:
        """Clear the result cache"""
        self._result_cache.clear()
    
    def get_query_history(self) -> List[str]:
        """Get recent query history"""
        return self._query_history[-50:]
