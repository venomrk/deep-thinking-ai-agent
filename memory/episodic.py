"""
Episodic Memory System
Stores and retrieves research episodes for learning across sessions.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import hashlib


@dataclass
class Episode:
    """A single research episode"""
    id: str
    query: str
    context: str
    actions_taken: List[Dict[str, Any]]
    outcome: str
    success: bool
    confidence: float
    lessons_learned: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    sources_used: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "query": self.query,
            "context": self.context,
            "actions_taken": self.actions_taken,
            "outcome": self.outcome,
            "success": self.success,
            "confidence": self.confidence,
            "lessons_learned": self.lessons_learned,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "sources_used": self.sources_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Episode":
        return cls(
            id=data["id"],
            query=data["query"],
            context=data.get("context", ""),
            actions_taken=data.get("actions_taken", []),
            outcome=data.get("outcome", ""),
            success=data.get("success", False),
            confidence=data.get("confidence", 0.5),
            lessons_learned=data.get("lessons_learned", []),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            duration_seconds=data.get("duration_seconds", 0.0),
            sources_used=data.get("sources_used", 0)
        )


class EpisodicMemory:
    """
    Episodic memory for learning across research sessions.
    
    Features:
    - Episode recording and retrieval
    - Similar episode lookup
    - Success/failure pattern learning
    - Cross-session knowledge transfer
    """
    
    def __init__(self, storage_path: Path = None, max_episodes: int = 1000):
        self._storage_path = storage_path or Path("./episodic_memory")
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._max_episodes = max_episodes
        self._episodes: Dict[str, Episode] = {}
        self._index: Dict[str, List[str]] = {}  # keyword -> episode IDs
        
        self._load_episodes()
    
    def _load_episodes(self) -> None:
        """Load episodes from storage"""
        episodes_file = self._storage_path / "episodes.json"
        
        if episodes_file.exists():
            try:
                data = json.loads(episodes_file.read_text())
                for ep_data in data.get("episodes", []):
                    episode = Episode.from_dict(ep_data)
                    self._episodes[episode.id] = episode
                    self._index_episode(episode)
            except Exception as e:
                print(f"Error loading episodes: {e}")
    
    def _save_episodes(self) -> None:
        """Save episodes to storage"""
        episodes_file = self._storage_path / "episodes.json"
        
        data = {
            "episodes": [ep.to_dict() for ep in self._episodes.values()]
        }
        
        episodes_file.write_text(json.dumps(data, indent=2))
    
    def _index_episode(self, episode: Episode) -> None:
        """Index episode for fast lookup"""
        # Extract keywords from query
        words = episode.query.lower().split()
        
        for word in words:
            if len(word) > 3:  # Skip short words
                if word not in self._index:
                    self._index[word] = []
                if episode.id not in self._index[word]:
                    self._index[word].append(episode.id)
    
    def _generate_id(self, query: str) -> str:
        """Generate unique episode ID"""
        content = f"{query}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def record_episode(self, query: str, context: str,
                       actions: List[Dict], outcome: str,
                       success: bool, confidence: float,
                       lessons: List[str] = None,
                       duration: float = 0.0,
                       sources: int = 0) -> str:
        """
        Record a new research episode.
        
        Returns:
            Episode ID
        """
        episode = Episode(
            id=self._generate_id(query),
            query=query,
            context=context,
            actions_taken=actions,
            outcome=outcome,
            success=success,
            confidence=confidence,
            lessons_learned=lessons or [],
            duration_seconds=duration,
            sources_used=sources
        )
        
        self._episodes[episode.id] = episode
        self._index_episode(episode)
        
        # Prune if too many episodes
        if len(self._episodes) > self._max_episodes:
            self._prune_old_episodes()
        
        self._save_episodes()
        
        return episode.id
    
    def _prune_old_episodes(self) -> None:
        """Remove oldest episodes"""
        sorted_episodes = sorted(
            self._episodes.values(),
            key=lambda e: e.timestamp
        )
        
        to_remove = len(sorted_episodes) - self._max_episodes
        for episode in sorted_episodes[:to_remove]:
            del self._episodes[episode.id]
    
    def find_similar(self, query: str, top_k: int = 5) -> List[Episode]:
        """
        Find similar past episodes.
        
        Args:
            query: Current query to match
            top_k: Number of results to return
            
        Returns:
            List of similar episodes sorted by relevance
        """
        query_words = set(query.lower().split())
        
        # Score each episode
        scored = []
        
        for episode in self._episodes.values():
            episode_words = set(episode.query.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(query_words & episode_words)
            union = len(query_words | episode_words)
            similarity = intersection / union if union > 0 else 0
            
            # Boost successful episodes
            if episode.success:
                similarity *= 1.2
            
            # Boost recent episodes
            days_old = (datetime.now() - episode.timestamp).days
            recency_factor = 1.0 / (1 + days_old / 30)
            similarity *= (0.5 + 0.5 * recency_factor)
            
            scored.append((similarity, episode))
        
        # Sort by similarity
        scored.sort(key=lambda x: -x[0])
        
        return [ep for _, ep in scored[:top_k]]
    
    def get_lessons_for(self, query: str) -> List[str]:
        """Get lessons learned from similar past episodes"""
        similar = self.find_similar(query, top_k=3)
        
        lessons = []
        for episode in similar:
            for lesson in episode.lessons_learned:
                if lesson not in lessons:
                    lessons.append(lesson)
        
        return lessons[:10]  # Limit lessons
    
    def get_successful_strategies(self, query: str) -> List[Dict]:
        """Get strategies that worked for similar queries"""
        similar = self.find_similar(query, top_k=5)
        
        strategies = []
        for episode in similar:
            if episode.success and episode.confidence > 0.6:
                strategies.append({
                    "query": episode.query,
                    "actions": episode.actions_taken,
                    "confidence": episode.confidence,
                    "sources_used": episode.sources_used
                })
        
        return strategies
    
    def get_failure_patterns(self, query: str) -> List[Dict]:
        """Get failure patterns from similar queries"""
        similar = self.find_similar(query, top_k=5)
        
        failures = []
        for episode in similar:
            if not episode.success:
                failures.append({
                    "query": episode.query,
                    "outcome": episode.outcome,
                    "lessons": episode.lessons_learned
                })
        
        return failures
    
    def update_episode(self, episode_id: str, 
                       lessons: List[str] = None,
                       success: bool = None) -> bool:
        """Update an existing episode"""
        if episode_id not in self._episodes:
            return False
        
        episode = self._episodes[episode_id]
        
        if lessons:
            episode.lessons_learned.extend(lessons)
        
        if success is not None:
            episode.success = success
        
        self._save_episodes()
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get episodic memory statistics"""
        if not self._episodes:
            return {"total_episodes": 0}
        
        successes = sum(1 for e in self._episodes.values() if e.success)
        total_lessons = sum(len(e.lessons_learned) for e in self._episodes.values())
        avg_confidence = sum(e.confidence for e in self._episodes.values()) / len(self._episodes)
        
        return {
            "total_episodes": len(self._episodes),
            "success_rate": successes / len(self._episodes),
            "total_lessons": total_lessons,
            "avg_confidence": avg_confidence,
            "indexed_keywords": len(self._index)
        }
    
    def clear(self) -> None:
        """Clear all episodes"""
        self._episodes.clear()
        self._index.clear()
        self._save_episodes()
