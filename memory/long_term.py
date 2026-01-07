"""
Long-Term Memory for Deep Research Agent
Persistent vector store for semantic search over past research.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
import asyncio

from ..core.base import MemoryEntry, Source, Evidence


@dataclass
class VectorEntry:
    """Entry in the vector store"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    importance: float = 0.5


class SimpleVectorStore:
    """
    Simple JSON-based vector store for when ChromaDB is not available.
    Uses cosine similarity for search.
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._index_file = storage_path / "vector_index.json"
        self._entries: Dict[str, VectorEntry] = {}
        self._load()
    
    def _load(self) -> None:
        """Load index from disk"""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for entry_data in data.get("entries", []):
                        entry = VectorEntry(
                            id=entry_data["id"],
                            content=entry_data["content"],
                            embedding=entry_data["embedding"],
                            metadata=entry_data.get("metadata", {}),
                            created_at=datetime.fromisoformat(entry_data["created_at"]),
                            last_accessed=datetime.fromisoformat(entry_data["last_accessed"]),
                            access_count=entry_data.get("access_count", 0),
                            importance=entry_data.get("importance", 0.5)
                        )
                        self._entries[entry.id] = entry
            except Exception as e:
                print(f"Warning: Could not load vector index: {e}")
    
    def _save(self) -> None:
        """Save index to disk"""
        data = {
            "entries": [
                {
                    "id": e.id,
                    "content": e.content,
                    "embedding": e.embedding,
                    "metadata": e.metadata,
                    "created_at": e.created_at.isoformat(),
                    "last_accessed": e.last_accessed.isoformat(),
                    "access_count": e.access_count,
                    "importance": e.importance
                }
                for e in self._entries.values()
            ]
        }
        with open(self._index_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def add(self, id: str, content: str, embedding: List[float], 
            metadata: Dict[str, Any] = None, importance: float = 0.5) -> None:
        """Add an entry to the store"""
        entry = VectorEntry(
            id=id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            importance=importance
        )
        self._entries[id] = entry
        self._save()
    
    def search(self, query_embedding: List[float], top_k: int = 10, 
               min_similarity: float = 0.0) -> List[Tuple[VectorEntry, float]]:
        """Search for similar entries"""
        results = []
        
        for entry in self._entries.values():
            similarity = self._cosine_similarity(query_embedding, entry.embedding)
            if similarity >= min_similarity:
                results.append((entry, similarity))
        
        # Sort by similarity descending
        results.sort(key=lambda x: -x[1])
        
        # Update access times for returned results
        for entry, _ in results[:top_k]:
            entry.last_accessed = datetime.now()
            entry.access_count += 1
        
        self._save()
        return results[:top_k]
    
    def get(self, id: str) -> Optional[VectorEntry]:
        """Get entry by ID"""
        entry = self._entries.get(id)
        if entry:
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self._save()
        return entry
    
    def delete(self, id: str) -> bool:
        """Delete an entry"""
        if id in self._entries:
            del self._entries[id]
            self._save()
            return True
        return False
    
    def clear(self) -> None:
        """Clear all entries"""
        self._entries.clear()
        self._save()
    
    def __len__(self) -> int:
        return len(self._entries)


class LongTermMemory:
    """
    Persistent long-term memory for research knowledge.
    Uses vector embeddings for semantic search.
    """
    
    def __init__(self, storage_path: Path, embedding_model: str = "text-embedding-004",
                 similarity_threshold: float = 0.75):
        self.storage_path = storage_path
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        
        # Initialize vector store
        self._vector_store = SimpleVectorStore(storage_path / "vectors")
        
        # Metadata store
        self._metadata_file = storage_path / "metadata.json"
        self._metadata: Dict[str, Any] = self._load_metadata()
        
        # Embedding cache
        self._embedding_cache: Dict[str, List[float]] = {}
        
        # LLM client for embeddings (will be set during initialization)
        self._llm_client = None
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from disk"""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"total_entries": 0, "last_updated": None}
    
    def _save_metadata(self) -> None:
        """Save metadata to disk"""
        self._metadata["last_updated"] = datetime.now().isoformat()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        with open(self._metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self._metadata, f)
    
    def set_llm_client(self, client) -> None:
        """Set the LLM client for embeddings"""
        self._llm_client = client
    
    def _content_hash(self, content: str) -> str:
        """Generate hash of content for deduplication"""
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text (with caching)"""
        cache_key = self._content_hash(text)
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # Generate embedding using LLM client
        if self._llm_client:
            try:
                result = await self._llm_client.embed_content(
                    model=f"models/{self.embedding_model}",
                    content=text
                )
                embedding = result['embedding']
                self._embedding_cache[cache_key] = embedding
                return embedding
            except Exception as e:
                print(f"Warning: Could not generate embedding: {e}")
        
        # Fallback: simple hash-based pseudo-embedding
        return self._pseudo_embedding(text)
    
    def _pseudo_embedding(self, text: str, dim: int = 768) -> List[float]:
        """Generate a pseudo-embedding based on text hash (fallback)"""
        import random
        hash_val = self._content_hash(text)
        random.seed(hash_val)
        return [random.gauss(0, 1) for _ in range(dim)]
    
    async def store(self, content: str, content_type: str = "research",
                    metadata: Dict[str, Any] = None, importance: float = 0.5) -> str:
        """Store content in long-term memory"""
        # Generate ID
        entry_id = self._content_hash(content)
        
        # Get embedding
        embedding = await self._get_embedding(content)
        
        # Store in vector store
        full_metadata = {
            "content_type": content_type,
            "importance": importance,
            **(metadata or {})
        }
        
        self._vector_store.add(
            id=entry_id,
            content=content,
            embedding=embedding,
            metadata=full_metadata,
            importance=importance
        )
        
        self._metadata["total_entries"] = len(self._vector_store)
        self._save_metadata()
        
        return entry_id
    
    async def store_evidence(self, evidence: Evidence) -> str:
        """Store evidence in long-term memory"""
        return await self.store(
            content=evidence.content,
            content_type=f"evidence_{evidence.evidence_type.value}",
            metadata={
                "source_url": evidence.source.url,
                "source_title": evidence.source.title,
                "relevance_score": evidence.relevance_score
            },
            importance=evidence.relevance_score
        )
    
    async def search(self, query: str, top_k: int = 10,
                     content_type: Optional[str] = None) -> List[Tuple[str, str, float]]:
        """
        Search for relevant memories.
        Returns list of (id, content, similarity_score) tuples.
        """
        query_embedding = await self._get_embedding(query)
        
        results = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more to filter
            min_similarity=self.similarity_threshold
        )
        
        # Filter by content type if specified
        if content_type:
            results = [
                (e, s) for e, s in results
                if e.metadata.get("content_type") == content_type
            ]
        
        return [
            (entry.id, entry.content, score)
            for entry, score in results[:top_k]
        ]
    
    async def search_similar(self, content: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Find memories similar to given content"""
        return await self.search(content, top_k=top_k)
    
    def get(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory entry"""
        entry = self._vector_store.get(entry_id)
        if entry:
            return {
                "id": entry.id,
                "content": entry.content,
                "metadata": entry.metadata,
                "created_at": entry.created_at.isoformat(),
                "access_count": entry.access_count
            }
        return None
    
    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry"""
        result = self._vector_store.delete(entry_id)
        if result:
            self._metadata["total_entries"] = len(self._vector_store)
            self._save_metadata()
        return result
    
    def cleanup_old(self, max_age_days: int = 30, min_access_count: int = 0) -> int:
        """Remove old, unused memories"""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        removed = 0
        
        # Get all entries
        for entry_id, entry in list(self._vector_store._entries.items()):
            if entry.last_accessed < cutoff and entry.access_count <= min_access_count:
                self._vector_store.delete(entry_id)
                removed += 1
        
        self._metadata["total_entries"] = len(self._vector_store)
        self._save_metadata()
        
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        entries = list(self._vector_store._entries.values())
        
        if not entries:
            return {
                "total_entries": 0,
                "last_updated": self._metadata.get("last_updated")
            }
        
        return {
            "total_entries": len(entries),
            "content_types": {},
            "avg_importance": sum(e.importance for e in entries) / len(entries),
            "avg_access_count": sum(e.access_count for e in entries) / len(entries),
            "oldest_entry": min(e.created_at for e in entries).isoformat(),
            "newest_entry": max(e.created_at for e in entries).isoformat(),
            "last_updated": self._metadata.get("last_updated")
        }
    
    def clear(self) -> None:
        """Clear all long-term memory"""
        self._vector_store.clear()
        self._embedding_cache.clear()
        self._metadata = {"total_entries": 0, "last_updated": None}
        self._save_metadata()
