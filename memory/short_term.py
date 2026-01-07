"""
Short-Term Working Memory for Deep Research Agent
Manages the current research session's active context.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import OrderedDict
import heapq
import json

from ..core.base import Evidence, Claim, Source, MemoryEntry


class PriorityQueue:
    """A priority queue for memory entries based on importance"""
    
    def __init__(self, maxsize: int = 100):
        self.maxsize = maxsize
        self._heap: List[tuple] = []
        self._entry_map: Dict[str, MemoryEntry] = {}
    
    def push(self, entry: MemoryEntry) -> None:
        """Add an entry to the queue"""
        if entry.id in self._entry_map:
            self.remove(entry.id)
        
        # Use negative importance for max-heap behavior
        heapq.heappush(self._heap, (-entry.importance, entry.created_at.timestamp(), entry.id))
        self._entry_map[entry.id] = entry
        
        # Evict lowest priority if over capacity
        while len(self._entry_map) > self.maxsize:
            self._evict_lowest()
    
    def _evict_lowest(self) -> Optional[MemoryEntry]:
        """Remove and return lowest priority entry"""
        while self._heap:
            _, _, entry_id = heapq.heappop(self._heap)
            if entry_id in self._entry_map:
                return self._entry_map.pop(entry_id)
        return None
    
    def remove(self, entry_id: str) -> Optional[MemoryEntry]:
        """Remove an entry by ID"""
        return self._entry_map.pop(entry_id, None)
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get an entry by ID"""
        entry = self._entry_map.get(entry_id)
        if entry:
            entry.access()
        return entry
    
    def get_all(self) -> List[MemoryEntry]:
        """Get all entries sorted by importance"""
        entries = list(self._entry_map.values())
        entries.sort(key=lambda e: -e.importance)
        return entries
    
    def __len__(self) -> int:
        return len(self._entry_map)


class ShortTermMemory:
    """
    Working memory for the current research session.
    Maintains recent findings, cross-references, and context.
    """
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._priority_queue = PriorityQueue(maxsize=capacity)
        self._recent_window: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._window_size = 20
        
        # Specialized indices
        self._by_type: Dict[str, List[str]] = {}
        self._by_source: Dict[str, List[str]] = {}
        self._cross_references: Dict[str, List[str]] = {}
        
        # Session context
        self._current_query: str = ""
        self._current_focus: List[str] = []
        self._context_stack: List[Dict[str, Any]] = []
    
    def store(self, content: str, content_type: str = "fact", 
              importance: float = 0.5, metadata: Dict[str, Any] = None) -> MemoryEntry:
        """Store a new piece of information in working memory"""
        entry = MemoryEntry(
            content=content,
            content_type=content_type,
            importance=importance,
            metadata=metadata or {}
        )
        
        self._priority_queue.push(entry)
        
        # Add to recent window
        self._recent_window[entry.id] = entry
        if len(self._recent_window) > self._window_size:
            self._recent_window.popitem(last=False)
        
        # Index by type
        if content_type not in self._by_type:
            self._by_type[content_type] = []
        self._by_type[content_type].append(entry.id)
        
        # Index by source if available
        source_id = (metadata or {}).get("source_id")
        if source_id:
            if source_id not in self._by_source:
                self._by_source[source_id] = []
            self._by_source[source_id].append(entry.id)
        
        return entry
    
    def store_evidence(self, evidence: Evidence) -> MemoryEntry:
        """Store evidence as a memory entry"""
        return self.store(
            content=evidence.content,
            content_type=f"evidence_{evidence.evidence_type.value}",
            importance=evidence.relevance_score,
            metadata={
                "evidence_id": evidence.id,
                "source_id": evidence.source.id,
                "source_url": evidence.source.url,
                "supports_claim": evidence.supports_claim
            }
        )
    
    def store_claim(self, claim: Claim) -> MemoryEntry:
        """Store a claim as a memory entry"""
        return self.store(
            content=claim.text,
            content_type="claim",
            importance=claim.confidence,
            metadata={
                "claim_id": claim.id,
                "verification_status": claim.verification_status.value,
                "confidence_level": claim.confidence_level.value
            }
        )
    
    def recall(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry"""
        return self._priority_queue.get(entry_id)
    
    def recall_recent(self, n: int = 10) -> List[MemoryEntry]:
        """Get the most recent entries"""
        entries = list(self._recent_window.values())
        return entries[-n:]
    
    def recall_by_type(self, content_type: str) -> List[MemoryEntry]:
        """Get all entries of a specific type"""
        entry_ids = self._by_type.get(content_type, [])
        entries = []
        for eid in entry_ids:
            entry = self._priority_queue.get(eid)
            if entry:
                entries.append(entry)
        return entries
    
    def recall_by_source(self, source_id: str) -> List[MemoryEntry]:
        """Get all entries from a specific source"""
        entry_ids = self._by_source.get(source_id, [])
        entries = []
        for eid in entry_ids:
            entry = self._priority_queue.get(eid)
            if entry:
                entries.append(entry)
        return entries
    
    def recall_important(self, n: int = 10, min_importance: float = 0.7) -> List[MemoryEntry]:
        """Get the most important entries"""
        all_entries = self._priority_queue.get_all()
        filtered = [e for e in all_entries if e.importance >= min_importance]
        return filtered[:n]
    
    def add_cross_reference(self, entry_id_a: str, entry_id_b: str) -> None:
        """Create a cross-reference between two entries"""
        if entry_id_a not in self._cross_references:
            self._cross_references[entry_id_a] = []
        if entry_id_b not in self._cross_references:
            self._cross_references[entry_id_b] = []
        
        if entry_id_b not in self._cross_references[entry_id_a]:
            self._cross_references[entry_id_a].append(entry_id_b)
        if entry_id_a not in self._cross_references[entry_id_b]:
            self._cross_references[entry_id_b].append(entry_id_a)
    
    def get_related(self, entry_id: str) -> List[MemoryEntry]:
        """Get entries cross-referenced with the given entry"""
        related_ids = self._cross_references.get(entry_id, [])
        entries = []
        for rid in related_ids:
            entry = self._priority_queue.get(rid)
            if entry:
                entries.append(entry)
        return entries
    
    def set_context(self, query: str, focus: List[str] = None) -> None:
        """Set the current research context"""
        self._current_query = query
        self._current_focus = focus or []
    
    def push_context(self) -> None:
        """Push current context to stack (for sub-tasks)"""
        self._context_stack.append({
            "query": self._current_query,
            "focus": self._current_focus.copy()
        })
    
    def pop_context(self) -> Optional[Dict[str, Any]]:
        """Pop context from stack"""
        if self._context_stack:
            ctx = self._context_stack.pop()
            self._current_query = ctx["query"]
            self._current_focus = ctx["focus"]
            return ctx
        return None
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current working memory state"""
        all_entries = self._priority_queue.get_all()
        
        return {
            "current_query": self._current_query,
            "focus_areas": self._current_focus,
            "total_entries": len(all_entries),
            "entry_types": {k: len(v) for k, v in self._by_type.items()},
            "sources_count": len(self._by_source),
            "cross_references": len(self._cross_references),
            "recent_count": len(self._recent_window),
            "context_depth": len(self._context_stack)
        }
    
    def to_prompt_context(self, max_entries: int = 20) -> str:
        """Format working memory for inclusion in LLM prompt"""
        important = self.recall_important(n=max_entries)
        
        lines = [f"Current Query: {self._current_query}"]
        if self._current_focus:
            lines.append(f"Focus Areas: {', '.join(self._current_focus)}")
        
        lines.append(f"\nWorking Memory ({len(important)} entries):")
        for entry in important:
            lines.append(f"- [{entry.content_type}] {entry.content[:200]}...")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all working memory"""
        self._priority_queue = PriorityQueue(maxsize=self.capacity)
        self._recent_window.clear()
        self._by_type.clear()
        self._by_source.clear()
        self._cross_references.clear()
        self._current_query = ""
        self._current_focus = []
        self._context_stack = []
