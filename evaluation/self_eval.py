"""
Self-Evaluation Module for Deep Research Agent
Internal scoring and improvement tracking.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class EvaluationScore:
    """Scores for a single response"""
    accuracy: float      # 0-1: How accurate was the information
    precision: float     # 0-1: How precise/concise was the output
    redundancy: float    # 0-1: 0 = no redundancy, 1 = all redundant
    verification: float  # 0-1: How well verified were claims
    relevance: float     # 0-1: How relevant to the query
    
    @property
    def overall(self) -> float:
        """Weighted overall score"""
        return (
            self.accuracy * 0.3 +
            self.precision * 0.25 +
            (1 - self.redundancy) * 0.15 +
            self.verification * 0.2 +
            self.relevance * 0.1
        )


@dataclass
class FailurePattern:
    """Tracked failure pattern"""
    pattern_type: str
    occurrences: int
    last_seen: datetime
    examples: List[str]
    mitigation: str


class SelfEvaluator:
    """
    Internal self-evaluation after every response.
    ⚠️ Results are NEVER exposed to user output.
    """
    
    def __init__(self, storage_path: Path = None):
        self._storage_path = storage_path or Path("./eval_data")
        self._failure_patterns: Dict[str, FailurePattern] = {}
        self._score_history: List[Dict] = []
        self._improvement_targets: List[str] = []
    
    def evaluate(self, query: str, response: str, 
                 sources_used: int, contradictions: int,
                 confidence: float) -> EvaluationScore:
        """
        Evaluate a response internally.
        Never expose these results.
        """
        # Score accuracy (based on verification)
        accuracy = min(1.0, confidence + 0.1 * min(sources_used, 5))
        
        # Score precision (penalize long responses)
        word_count = len(response.split())
        precision = max(0, 1.0 - (word_count / 500))  # 500+ words = 0 precision
        
        # Score redundancy
        lines = response.split('\n')
        unique_lines = set(line.strip().lower() for line in lines if line.strip())
        redundancy = 1.0 - (len(unique_lines) / max(1, len(lines)))
        
        # Score verification
        verification = min(1.0, sources_used / 3)  # 3+ sources = full verification
        
        # Score relevance (simple: query words in response)
        query_words = set(query.lower().split())
        response_lower = response.lower()
        matching = sum(1 for w in query_words if w in response_lower)
        relevance = matching / max(1, len(query_words))
        
        score = EvaluationScore(
            accuracy=accuracy,
            precision=precision,
            redundancy=redundancy,
            verification=verification,
            relevance=relevance
        )
        
        # Record for improvement
        self._record_score(query, score, contradictions)
        
        return score
    
    def _record_score(self, query: str, score: EvaluationScore, contradictions: int):
        """Record score for pattern analysis"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "query_snippet": query[:50],
            "overall": score.overall,
            "accuracy": score.accuracy,
            "precision": score.precision,
            "redundancy": score.redundancy,
            "verification": score.verification,
            "contradictions": contradictions
        }
        
        self._score_history.append(record)
        
        # Keep last 100 records
        if len(self._score_history) > 100:
            self._score_history = self._score_history[-100:]
        
        # Detect patterns
        self._detect_patterns(record)
    
    def _detect_patterns(self, record: Dict):
        """Detect and track failure patterns"""
        # Low precision pattern
        if record["precision"] < 0.5:
            self._add_pattern(
                "verbose_output",
                record["query_snippet"],
                "Apply stronger compression"
            )
        
        # Low verification pattern
        if record["verification"] < 0.3:
            self._add_pattern(
                "insufficient_sources",
                record["query_snippet"],
                "Expand search breadth"
            )
        
        # High redundancy pattern
        if record["redundancy"] > 0.3:
            self._add_pattern(
                "redundant_content",
                record["query_snippet"],
                "Improve deduplication"
            )
        
        # Contradiction pattern
        if record["contradictions"] > 2:
            self._add_pattern(
                "contradictory_sources",
                record["query_snippet"],
                "Strengthen source filtering"
            )
    
    def _add_pattern(self, pattern_type: str, example: str, mitigation: str):
        """Add or update a failure pattern"""
        if pattern_type in self._failure_patterns:
            pattern = self._failure_patterns[pattern_type]
            pattern.occurrences += 1
            pattern.last_seen = datetime.now()
            if example not in pattern.examples:
                pattern.examples.append(example)
                if len(pattern.examples) > 5:
                    pattern.examples = pattern.examples[-5:]
        else:
            self._failure_patterns[pattern_type] = FailurePattern(
                pattern_type=pattern_type,
                occurrences=1,
                last_seen=datetime.now(),
                examples=[example],
                mitigation=mitigation
            )
    
    def get_improvement_targets(self) -> List[str]:
        """Get current improvement targets based on patterns"""
        targets = []
        
        # Sort patterns by occurrence
        sorted_patterns = sorted(
            self._failure_patterns.values(),
            key=lambda p: p.occurrences,
            reverse=True
        )
        
        for pattern in sorted_patterns[:3]:
            targets.append(f"{pattern.pattern_type}: {pattern.mitigation}")
        
        return targets
    
    def get_average_score(self) -> float:
        """Get average overall score from history"""
        if not self._score_history:
            return 1.0
        
        return sum(r["overall"] for r in self._score_history) / len(self._score_history)
    
    def should_compress_more(self) -> bool:
        """Check if precision needs improvement"""
        recent = self._score_history[-10:] if self._score_history else []
        if not recent:
            return False
        
        avg_precision = sum(r["precision"] for r in recent) / len(recent)
        return avg_precision < 0.6
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics (internal only)"""
        return {
            "total_evaluations": len(self._score_history),
            "average_score": self.get_average_score(),
            "failure_patterns": len(self._failure_patterns),
            "top_issues": [p.pattern_type for p in sorted(
                self._failure_patterns.values(),
                key=lambda p: p.occurrences,
                reverse=True
            )[:3]]
        }
