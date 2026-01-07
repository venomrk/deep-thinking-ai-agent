"""
Failure Memory for Deep Research Agent
Tracks failures to prevent repeated mistakes and enable learning.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import json
import hashlib

from ..core.base import FailureRecord


class FailureMemory:
    """
    Tracks failures, errors, and dead-ends to prevent repeated mistakes.
    Implements pattern recognition for common failure modes.
    """
    
    def __init__(self, storage_path: Optional[Path] = None, ttl_hours: int = 24):
        self.storage_path = storage_path
        self.ttl_hours = ttl_hours
        
        self._failures: Dict[str, FailureRecord] = {}
        self._by_type: Dict[str, List[str]] = defaultdict(list)
        self._by_context: Dict[str, List[str]] = defaultdict(list)
        self._pattern_counts: Dict[str, int] = defaultdict(int)
        
        if storage_path:
            self._load()
    
    def _load(self) -> None:
        """Load failures from disk"""
        if self.storage_path and self.storage_path.exists():
            failures_file = self.storage_path / "failures.json"
            if failures_file.exists():
                try:
                    with open(failures_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for failure_data in data.get("failures", []):
                            failure = FailureRecord(
                                id=failure_data["id"],
                                failure_type=failure_data["failure_type"],
                                context=failure_data["context"],
                                action_attempted=failure_data["action_attempted"],
                                error_message=failure_data["error_message"],
                                recovery_action=failure_data.get("recovery_action"),
                                prevented_recurrence=failure_data.get("prevented_recurrence", False),
                                created_at=datetime.fromisoformat(failure_data["created_at"])
                            )
                            self._add_to_indices(failure)
                        self._pattern_counts = defaultdict(int, data.get("pattern_counts", {}))
                except Exception as e:
                    print(f"Warning: Could not load failures: {e}")
    
    def _save(self) -> None:
        """Save failures to disk"""
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        failures_file = self.storage_path / "failures.json"
        
        data = {
            "failures": [
                {
                    "id": f.id,
                    "failure_type": f.failure_type,
                    "context": f.context,
                    "action_attempted": f.action_attempted,
                    "error_message": f.error_message,
                    "recovery_action": f.recovery_action,
                    "prevented_recurrence": f.prevented_recurrence,
                    "created_at": f.created_at.isoformat()
                }
                for f in self._failures.values()
            ],
            "pattern_counts": dict(self._pattern_counts)
        }
        
        with open(failures_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _add_to_indices(self, failure: FailureRecord) -> None:
        """Add failure to lookup indices"""
        self._failures[failure.id] = failure
        self._by_type[failure.failure_type].append(failure.id)
        
        # Extract context key (first word or hash)
        context_key = failure.context.split()[0] if failure.context else "unknown"
        self._by_context[context_key].append(failure.id)
    
    def _generate_pattern_key(self, failure_type: str, action: str) -> str:
        """Generate a pattern key for failure tracking"""
        return hashlib.md5(f"{failure_type}:{action}".encode()).hexdigest()[:16]
    
    def record_failure(self, failure_type: str, context: str, action: str,
                       error_message: str, recovery_action: Optional[str] = None) -> FailureRecord:
        """Record a new failure"""
        failure = FailureRecord(
            failure_type=failure_type,
            context=context,
            action_attempted=action,
            error_message=error_message,
            recovery_action=recovery_action
        )
        
        self._add_to_indices(failure)
        
        # Track pattern
        pattern_key = self._generate_pattern_key(failure_type, action)
        self._pattern_counts[pattern_key] += 1
        
        self._cleanup_old()
        self._save()
        
        return failure
    
    def record_search_failure(self, query: str, error: str) -> FailureRecord:
        """Record a search failure"""
        return self.record_failure(
            failure_type="search_failure",
            context=f"Searching for: {query[:100]}",
            action="web_search",
            error_message=error
        )
    
    def record_reasoning_deadend(self, thought: str, reason: str) -> FailureRecord:
        """Record a reasoning dead-end"""
        return self.record_failure(
            failure_type="reasoning_deadend",
            context=f"Reasoning about: {thought[:100]}",
            action="tree_exploration",
            error_message=reason
        )
    
    def record_tool_failure(self, tool_name: str, params: Dict, error: str) -> FailureRecord:
        """Record a tool execution failure"""
        return self.record_failure(
            failure_type="tool_failure",
            context=f"Tool: {tool_name}",
            action=json.dumps(params)[:200],
            error_message=error
        )
    
    def record_verification_failure(self, claim: str, reason: str) -> FailureRecord:
        """Record a verification failure"""
        return self.record_failure(
            failure_type="verification_failure",
            context=f"Verifying: {claim[:100]}",
            action="fact_check",
            error_message=reason
        )
    
    def should_avoid(self, failure_type: str, action: str, threshold: int = 3) -> bool:
        """Check if an action should be avoided based on failure history"""
        pattern_key = self._generate_pattern_key(failure_type, action)
        return self._pattern_counts.get(pattern_key, 0) >= threshold
    
    def get_failures_by_type(self, failure_type: str) -> List[FailureRecord]:
        """Get all failures of a specific type"""
        return [
            self._failures[fid]
            for fid in self._by_type.get(failure_type, [])
            if fid in self._failures
        ]
    
    def get_recent_failures(self, n: int = 10, failure_type: Optional[str] = None) -> List[FailureRecord]:
        """Get most recent failures"""
        failures = list(self._failures.values())
        
        if failure_type:
            failures = [f for f in failures if f.failure_type == failure_type]
        
        failures.sort(key=lambda f: f.created_at, reverse=True)
        return failures[:n]
    
    def get_recovery_suggestions(self, failure_type: str) -> List[str]:
        """Get recovery suggestions based on past failures"""
        failures = self.get_failures_by_type(failure_type)
        suggestions = []
        
        for failure in failures:
            if failure.recovery_action and failure.prevented_recurrence:
                suggestions.append(failure.recovery_action)
        
        # Deduplicate while preserving order
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique_suggestions.append(s)
        
        return unique_suggestions
    
    def mark_recovery_successful(self, failure_id: str, recovery_action: str) -> None:
        """Mark that a recovery action was successful"""
        if failure_id in self._failures:
            self._failures[failure_id].recovery_action = recovery_action
            self._failures[failure_id].prevented_recurrence = True
            self._save()
    
    def get_common_patterns(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the most common failure patterns"""
        sorted_patterns = sorted(
            self._pattern_counts.items(),
            key=lambda x: -x[1]
        )
        
        patterns = []
        for pattern_key, count in sorted_patterns[:n]:
            # Find example failure
            for failure in self._failures.values():
                if self._generate_pattern_key(failure.failure_type, failure.action_attempted) == pattern_key:
                    patterns.append({
                        "pattern_key": pattern_key,
                        "count": count,
                        "failure_type": failure.failure_type,
                        "example_action": failure.action_attempted,
                        "example_error": failure.error_message
                    })
                    break
        
        return patterns
    
    def _cleanup_old(self) -> None:
        """Remove old failures past TTL"""
        cutoff = datetime.now() - timedelta(hours=self.ttl_hours)
        
        to_remove = [
            fid for fid, failure in self._failures.items()
            if failure.created_at < cutoff and not failure.prevented_recurrence
        ]
        
        for fid in to_remove:
            failure = self._failures.pop(fid, None)
            if failure:
                # Remove from indices
                if failure.failure_type in self._by_type:
                    try:
                        self._by_type[failure.failure_type].remove(fid)
                    except ValueError:
                        pass
    
    def to_prompt_context(self, failure_types: Optional[List[str]] = None) -> str:
        """Format failure memory for inclusion in LLM prompt"""
        lines = ["Recent Failures to Avoid:"]
        
        failures = self.get_recent_failures(n=10)
        if failure_types:
            failures = [f for f in failures if f.failure_type in failure_types]
        
        for failure in failures[:5]:
            lines.append(f"- [{failure.failure_type}] {failure.action_attempted[:50]}: {failure.error_message[:100]}")
        
        patterns = self.get_common_patterns(n=3)
        if patterns:
            lines.append("\nCommon Failure Patterns:")
            for p in patterns:
                lines.append(f"- {p['failure_type']} (occurred {p['count']} times)")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get failure statistics"""
        failures = list(self._failures.values())
        
        if not failures:
            return {"total_failures": 0}
        
        type_counts = defaultdict(int)
        for f in failures:
            type_counts[f.failure_type] += 1
        
        return {
            "total_failures": len(failures),
            "failures_by_type": dict(type_counts),
            "patterns_tracked": len(self._pattern_counts),
            "successful_recoveries": len([f for f in failures if f.prevented_recurrence]),
            "oldest_failure": min(f.created_at for f in failures).isoformat(),
            "newest_failure": max(f.created_at for f in failures).isoformat()
        }
    
    def clear(self) -> None:
        """Clear all failure records"""
        self._failures.clear()
        self._by_type.clear()
        self._by_context.clear()
        self._pattern_counts.clear()
        self._save()
