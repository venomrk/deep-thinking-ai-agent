"""
Fact Checker for Deep Research Agent
Multi-source claim verification system.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..core.base import (
    Claim, Evidence, Source, VerificationStatus, 
    EvidenceType, BaseModule
)
from ..search.engine import DeepSearchEngine, SearchResult
from ..config.prompts import PromptTemplates


@dataclass
class VerificationResult:
    """Result of verifying a claim"""
    claim: Claim
    status: VerificationStatus
    confidence: float
    supporting_evidence: List[Evidence]
    contradicting_evidence: List[Evidence]
    sources_checked: int
    explanation: str
    verified_at: datetime = field(default_factory=datetime.now)


class FactChecker(BaseModule):
    """
    Multi-source fact-checking for claims.
    Features:
    - Cross-source verification
    - Evidence strength scoring
    - Contradiction detection
    - Confidence calculation
    """
    
    def __init__(self, search_engine: DeepSearchEngine = None, llm_client=None):
        super().__init__("fact_checker")
        self._search_engine = search_engine
        self._llm_client = llm_client
        self._verification_cache: Dict[str, VerificationResult] = {}
        self._min_sources = 2
    
    async def initialize(self) -> None:
        """Initialize the fact checker"""
        if not self._search_engine:
            self._search_engine = DeepSearchEngine()
            await self._search_engine.initialize()
        self._initialized = True
    
    async def shutdown(self) -> None:
        """Clean shutdown"""
        if self._search_engine:
            await self._search_engine.shutdown()
        self._initialized = False
    
    def set_llm_client(self, client) -> None:
        """Set LLM client for analysis"""
        self._llm_client = client
    
    async def verify_claim(self, claim: Claim) -> VerificationResult:
        """
        Verify a claim against multiple sources.
        
        Args:
            claim: The claim to verify
            
        Returns:
            VerificationResult with status and evidence
        """
        # Check cache
        cache_key = claim.text[:100]
        if cache_key in self._verification_cache:
            cached = self._verification_cache[cache_key]
            # Return cached if less than 1 hour old
            age = (datetime.now() - cached.verified_at).seconds
            if age < 3600:
                return cached
        
        # Search for evidence
        supporting, contradicting = await self._gather_evidence(claim)
        
        # Calculate confidence
        confidence = self._calculate_confidence(supporting, contradicting)
        
        # Determine status
        status = self._determine_status(confidence, len(supporting), len(contradicting))
        
        # Generate explanation
        explanation = self._generate_explanation(
            claim, status, confidence, supporting, contradicting
        )
        
        # Update claim
        claim.supporting_evidence = supporting
        claim.contradicting_evidence = contradicting
        claim.verification_status = status
        claim.confidence = confidence
        claim.verified_at = datetime.now()
        
        result = VerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            sources_checked=len(supporting) + len(contradicting),
            explanation=explanation
        )
        
        # Cache result
        self._verification_cache[cache_key] = result
        
        return result
    
    async def _gather_evidence(self, claim: Claim) -> Tuple[List[Evidence], List[Evidence]]:
        """Gather supporting and contradicting evidence"""
        from ..core.base import SearchType
        
        # Generate search queries
        queries = self._generate_verification_queries(claim.text)
        
        # Search multiple sources
        all_results = await self._search_engine.multi_query_search(
            queries,
            search_types=[SearchType.WEB, SearchType.ACADEMIC],
            num_results_per_query=5
        )
        
        # Analyze each result for support/contradiction
        supporting = []
        contradicting = []
        
        for result in all_results[:20]:  # Limit to top 20
            evidence, is_supporting = await self._analyze_result(claim, result)
            if evidence:
                if is_supporting:
                    supporting.append(evidence)
                else:
                    contradicting.append(evidence)
        
        return supporting, contradicting
    
    def _generate_verification_queries(self, claim_text: str) -> List[str]:
        """Generate queries to verify a claim"""
        queries = [claim_text]
        
        # Add verification-focused queries
        queries.append(f'"{claim_text}" fact check')
        queries.append(f'"{claim_text}" evidence')
        queries.append(f'Is it true that {claim_text}')
        
        # Add negation query
        queries.append(f'{claim_text} false OR myth OR debunked')
        
        return queries
    
    async def _analyze_result(self, claim: Claim, 
                               result: SearchResult) -> Tuple[Optional[Evidence], bool]:
        """Analyze a search result for claim support"""
        if not result.snippet:
            return None, True
        
        # Simple keyword-based analysis (fallback if no LLM)
        claim_words = set(claim.text.lower().split())
        snippet_lower = result.snippet.lower()
        
        # Check for negation words
        negation_words = {'not', 'false', 'incorrect', 'wrong', 'myth', 'debunked', 'untrue'}
        has_negation = any(neg in snippet_lower for neg in negation_words)
        
        # Calculate relevance
        word_matches = sum(1 for w in claim_words if w in snippet_lower)
        relevance = word_matches / len(claim_words) if claim_words else 0
        
        if relevance < 0.3:
            return None, True  # Not relevant enough
        
        # Create evidence
        source = result.to_source()
        evidence = Evidence(
            source=source,
            content=result.snippet,
            evidence_type=EvidenceType.FACT if result.source_type == "academic" else EvidenceType.PARAPHRASE,
            relevance_score=relevance * result.relevance_score,
            supports_claim=not has_negation,
            context=result.title
        )
        
        return evidence, not has_negation
    
    def _calculate_confidence(self, supporting: List[Evidence], 
                               contradicting: List[Evidence]) -> float:
        """Calculate confidence score based on evidence"""
        if not supporting and not contradicting:
            return 0.5  # No evidence either way
        
        # Weight evidence by relevance and source reliability
        support_score = sum(
            e.relevance_score * e.source.reliability_score 
            for e in supporting
        )
        contradict_score = sum(
            e.relevance_score * e.source.reliability_score 
            for e in contradicting
        )
        
        total = support_score + contradict_score
        if total == 0:
            return 0.5
        
        # Base confidence on ratio
        base_confidence = support_score / total
        
        # Adjust for number of sources (more sources = more confidence in result)
        source_factor = min(1.0, (len(supporting) + len(contradicting)) / 5)
        
        # Move confidence away from 0.5 based on source factor
        if base_confidence > 0.5:
            return 0.5 + (base_confidence - 0.5) * source_factor
        else:
            return 0.5 - (0.5 - base_confidence) * source_factor
    
    def _determine_status(self, confidence: float, 
                          supporting_count: int,
                          contradicting_count: int) -> VerificationStatus:
        """Determine verification status from confidence and evidence counts"""
        if supporting_count == 0 and contradicting_count == 0:
            return VerificationStatus.UNVERIFIED
        
        if confidence >= 0.85 and supporting_count >= self._min_sources:
            return VerificationStatus.VERIFIED
        elif confidence >= 0.65:
            return VerificationStatus.LIKELY_TRUE
        elif confidence <= 0.35:
            return VerificationStatus.LIKELY_FALSE
        elif confidence <= 0.15 and contradicting_count >= self._min_sources:
            return VerificationStatus.FALSE
        else:
            return VerificationStatus.UNCERTAIN
    
    def _generate_explanation(self, claim: Claim, status: VerificationStatus,
                               confidence: float, supporting: List[Evidence],
                               contradicting: List[Evidence]) -> str:
        """Generate explanation for verification result"""
        status_text = {
            VerificationStatus.VERIFIED: "This claim is verified",
            VerificationStatus.LIKELY_TRUE: "This claim is likely true",
            VerificationStatus.UNCERTAIN: "This claim cannot be conclusively verified",
            VerificationStatus.LIKELY_FALSE: "This claim is likely false",
            VerificationStatus.FALSE: "This claim is false",
            VerificationStatus.UNVERIFIED: "No evidence found for this claim"
        }
        
        explanation = [status_text.get(status, "Status unknown")]
        explanation.append(f"(confidence: {confidence:.0%})")
        
        if supporting:
            explanation.append(f"\n\nSupporting evidence ({len(supporting)} sources):")
            for e in supporting[:3]:
                explanation.append(f"- {e.source.title or e.source.url}: {e.content[:100]}...")
        
        if contradicting:
            explanation.append(f"\n\nContradicting evidence ({len(contradicting)} sources):")
            for e in contradicting[:3]:
                explanation.append(f"- {e.source.title or e.source.url}: {e.content[:100]}...")
        
        return "\n".join(explanation)
    
    async def batch_verify(self, claims: List[Claim]) -> List[VerificationResult]:
        """Verify multiple claims in parallel"""
        tasks = [self.verify_claim(claim) for claim in claims]
        return await asyncio.gather(*tasks)
    
    def clear_cache(self) -> None:
        """Clear verification cache"""
        self._verification_cache.clear()
