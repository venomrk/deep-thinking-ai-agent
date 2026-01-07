"""
Reflection & Self-Correction Engine (DeepSeek R1 Style)
Enables the agent to analyze and revise its own reasoning.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio


class ReflectionType(Enum):
    """Types of reflection"""
    SELF_VERIFICATION = "self_verification"
    ERROR_CORRECTION = "error_correction"
    STRATEGY_REVISION = "strategy_revision"
    ASSUMPTION_CHECK = "assumption_check"


@dataclass
class ReflectionStep:
    """A single reflection step"""
    step_id: str
    reflection_type: ReflectionType
    original_thought: str
    critique: str
    revised_thought: Optional[str]
    confidence_before: float
    confidence_after: float
    corrections_made: List[str]


@dataclass
class ReflectionResult:
    """Result of reflection process"""
    original_conclusion: str
    final_conclusion: str
    reflection_steps: List[ReflectionStep]
    total_corrections: int
    confidence_improvement: float
    reflection_depth: int
    approved: bool


class ReflectionEngine:
    """
    DeepSeek R1-style reflection and self-correction.
    
    Features:
    - Self-verification of outputs
    - Iterative error correction
    - Strategy revision when stuck
    - Assumption validation
    - Rejection sampling for best outputs
    """
    
    def __init__(self, llm_client=None, max_iterations: int = 3):
        self._llm_client = llm_client
        self._max_iterations = max_iterations
        self._reflection_history: List[ReflectionResult] = []
        self._correction_patterns: Dict[str, int] = {}
    
    def set_llm_client(self, client) -> None:
        """Set LLM client"""
        self._llm_client = client
    
    async def reflect(self, thought: str, context: Dict[str, Any],
                      evidence: List[Dict] = None) -> ReflectionResult:
        """
        Perform reflection on a thought/conclusion.
        
        Args:
            thought: The thought to reflect on
            context: Context including query and evidence
            evidence: Supporting evidence
            
        Returns:
            ReflectionResult with corrected conclusion
        """
        steps = []
        current_thought = thought
        current_confidence = 0.7
        
        for i in range(self._max_iterations):
            # Step 1: Self-verification
            verification = await self._self_verify(current_thought, evidence or [])
            steps.append(verification)
            
            if not verification.corrections_made:
                # No issues found, check assumptions
                assumption_check = await self._check_assumptions(current_thought, context)
                steps.append(assumption_check)
                
                if not assumption_check.corrections_made:
                    break  # Reflection complete
                
                current_thought = assumption_check.revised_thought or current_thought
                current_confidence = assumption_check.confidence_after
            else:
                current_thought = verification.revised_thought or current_thought
                current_confidence = verification.confidence_after
        
        # Calculate improvement
        original_confidence = 0.7
        confidence_improvement = current_confidence - original_confidence
        
        result = ReflectionResult(
            original_conclusion=thought,
            final_conclusion=current_thought,
            reflection_steps=steps,
            total_corrections=sum(len(s.corrections_made) for s in steps),
            confidence_improvement=confidence_improvement,
            reflection_depth=len(steps),
            approved=current_confidence >= 0.6
        )
        
        self._reflection_history.append(result)
        return result
    
    async def _self_verify(self, thought: str, evidence: List[Dict]) -> ReflectionStep:
        """Verify thought against evidence"""
        corrections = []
        revised = thought
        confidence_before = 0.7
        confidence_after = 0.7
        critique = ""
        
        # Check for unsupported claims
        if evidence:
            evidence_text = " ".join(str(e.get("content", "")) for e in evidence[:5])
            
            # Simple verification: check if key words from thought are in evidence
            thought_words = set(thought.lower().split())
            evidence_words = set(evidence_text.lower().split())
            
            unsupported_words = []
            for word in thought_words:
                if len(word) > 5 and word not in evidence_words:
                    if word not in ["which", "where", "there", "these", "their", "about"]:
                        unsupported_words.append(word)
            
            if len(unsupported_words) > 5:
                corrections.append("Removed unsupported specifics")
                critique = f"Found {len(unsupported_words)} potentially unsupported terms"
                confidence_after = 0.6
        
        # Check for overconfident language
        overconfident = ["definitely", "always", "never", "certainly", "absolutely"]
        for word in overconfident:
            if word in thought.lower():
                revised = thought.replace(word, "likely")
                corrections.append(f"Hedged overconfident term: {word}")
                critique += f" Overconfident language detected: '{word}'."
                confidence_after = max(0.5, confidence_after - 0.05)
        
        return ReflectionStep(
            step_id=f"verify_{datetime.now().timestamp()}",
            reflection_type=ReflectionType.SELF_VERIFICATION,
            original_thought=thought,
            critique=critique.strip() or "No issues found",
            revised_thought=revised if corrections else None,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            corrections_made=corrections
        )
    
    async def _check_assumptions(self, thought: str, context: Dict) -> ReflectionStep:
        """Check and validate assumptions"""
        corrections = []
        revised = thought
        critique = ""
        confidence_before = 0.7
        confidence_after = 0.75
        
        # Check for implicit assumptions
        assumption_indicators = [
            ("assuming", "Explicit assumption found"),
            ("obviously", "Implicit assumption of obviousness"),
            ("clearly", "Implicit assumption of clarity"),
            ("everyone knows", "Assumed universal knowledge"),
            ("it is known", "Appeal to common knowledge")
        ]
        
        for indicator, issue in assumption_indicators:
            if indicator in thought.lower():
                critique += f" {issue}."
                corrections.append(f"Flagged assumption: {indicator}")
        
        # If assumptions found, suggest revision
        if corrections:
            confidence_after = 0.6
            revised = thought + " [Assumptions noted - may require verification]"
        else:
            confidence_after = 0.8
        
        return ReflectionStep(
            step_id=f"assume_{datetime.now().timestamp()}",
            reflection_type=ReflectionType.ASSUMPTION_CHECK,
            original_thought=thought,
            critique=critique.strip() or "No problematic assumptions found",
            revised_thought=revised if corrections else None,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            corrections_made=corrections
        )
    
    async def rejection_sample(self, candidates: List[str], 
                                context: Dict[str, Any]) -> Tuple[str, float]:
        """
        Select best output from multiple candidates (rejection sampling).
        
        Args:
            candidates: List of candidate outputs
            context: Context for evaluation
            
        Returns:
            Best candidate and its score
        """
        if not candidates:
            return "", 0.0
        
        scored = []
        
        for candidate in candidates:
            # Score based on multiple criteria
            score = 0.0
            
            # Length score (prefer concise)
            words = len(candidate.split())
            if words < 50:
                score += 0.3
            elif words < 100:
                score += 0.2
            else:
                score += 0.1
            
            # Confidence language score
            if "uncertain" in candidate.lower() or "may" in candidate.lower():
                score += 0.1  # Appropriate hedging
            
            overconfident = ["definitely", "always", "never", "certainly"]
            if not any(oc in candidate.lower() for oc in overconfident):
                score += 0.2
            
            # Evidence reference score
            if "source" in candidate.lower() or "according" in candidate.lower():
                score += 0.2
            
            # Clarity score (no repetition)
            sentences = candidate.split('.')
            unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
            if len(unique_sentences) == len([s for s in sentences if s.strip()]):
                score += 0.2
            
            scored.append((candidate, score))
        
        # Sort by score
        scored.sort(key=lambda x: -x[1])
        
        return scored[0]
    
    async def strategy_revision(self, failed_strategy: str, 
                                 error: str) -> Dict[str, Any]:
        """
        Revise strategy after failure.
        
        Args:
            failed_strategy: The strategy that failed
            error: Error message or reason
            
        Returns:
            Revised strategy recommendation
        """
        # Track failure pattern
        pattern_key = f"{failed_strategy[:30]}:{error[:30]}"
        self._correction_patterns[pattern_key] = self._correction_patterns.get(pattern_key, 0) + 1
        
        # Suggest alternatives based on failure type
        alternatives = []
        
        if "timeout" in error.lower():
            alternatives.append("Reduce search scope")
            alternatives.append("Use cached results")
        elif "not found" in error.lower():
            alternatives.append("Broaden search query")
            alternatives.append("Try alternative sources")
        elif "contradiction" in error.lower():
            alternatives.append("Prioritize higher-reliability sources")
            alternatives.append("Present multiple perspectives")
        else:
            alternatives.append("Simplify the query")
            alternatives.append("Break into smaller sub-tasks")
        
        return {
            "failed_strategy": failed_strategy,
            "error": error,
            "pattern_count": self._correction_patterns[pattern_key],
            "alternatives": alternatives,
            "recommendation": alternatives[0] if alternatives else "Retry with different approach"
        }
    
    def get_correction_stats(self) -> Dict[str, Any]:
        """Get statistics on corrections made"""
        if not self._reflection_history:
            return {"total_reflections": 0}
        
        total_corrections = sum(r.total_corrections for r in self._reflection_history)
        avg_improvement = sum(r.confidence_improvement for r in self._reflection_history) / len(self._reflection_history)
        approval_rate = sum(1 for r in self._reflection_history if r.approved) / len(self._reflection_history)
        
        return {
            "total_reflections": len(self._reflection_history),
            "total_corrections": total_corrections,
            "avg_confidence_improvement": avg_improvement,
            "approval_rate": approval_rate,
            "top_correction_patterns": sorted(
                self._correction_patterns.items(),
                key=lambda x: -x[1]
            )[:5]
        }
