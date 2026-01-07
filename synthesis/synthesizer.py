"""
Synthesizer Module for Deep Research Agent
Merges validated facts into coherent, citation-aware outputs.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..core.base import (
    Claim, Evidence, Source, ResearchOutput, OutputFormat,
    VerificationStatus, BaseModule
)
from ..memory.citations import CitationMemory
from ..config.prompts import PromptTemplates


@dataclass
class MergedKnowledge:
    """Merged and validated knowledge"""
    claims: List[Claim]
    sources: List[Source]
    contradictions: List[Dict[str, Any]]
    confidence: float
    coverage: float  # How much of the query is addressed


@dataclass
class Conflict:
    """A conflict between claims"""
    claim_a: Claim
    claim_b: Claim
    conflict_type: str
    resolution: Optional[str] = None


class Synthesizer(BaseModule):
    """
    Synthesizes research findings into coherent outputs.
    Features:
    - Fact merging with deduplication
    - Contradiction resolution
    - Citation-aware generation
    - Multiple output formats
    """
    
    def __init__(self, citation_memory: CitationMemory = None, llm_client=None):
        super().__init__("synthesizer")
        self._citation_memory = citation_memory or CitationMemory()
        self._llm_client = llm_client
    
    async def initialize(self) -> None:
        """Initialize the synthesizer"""
        self._initialized = True
    
    async def shutdown(self) -> None:
        """Clean shutdown"""
        self._initialized = False
    
    def set_llm_client(self, client) -> None:
        """Set LLM client"""
        self._llm_client = client
    
    async def merge_facts(self, claims: List[Claim]) -> MergedKnowledge:
        """Merge and deduplicate claims"""
        # Group by similar content
        unique_claims = []
        seen_content = set()
        
        for claim in claims:
            # Simple deduplication by content hash
            content_key = claim.text.lower()[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_claims.append(claim)
        
        # Collect all sources
        all_sources = []
        source_ids = set()
        for claim in unique_claims:
            for source in claim.sources:
                if source.id not in source_ids:
                    source_ids.add(source.id)
                    all_sources.append(source)
        
        # Detect contradictions
        contradictions = self._detect_contradictions(unique_claims)
        
        # Calculate overall confidence (boosted scoring)
        if unique_claims:
            base_conf = sum(c.confidence for c in unique_claims) / len(unique_claims)
            # Boost for having multiple claims
            claim_boost = min(0.15, len(unique_claims) * 0.02)
            # Boost for having sources
            source_boost = min(0.15, len(all_sources) * 0.02)
            # Penalty for contradictions
            contradiction_penalty = min(0.1, len(contradictions) * 0.02)
            confidence = min(0.99, base_conf + claim_boost + source_boost - contradiction_penalty)
        else:
            confidence = 0.75  # Higher default when no claims
        
        return MergedKnowledge(
            claims=unique_claims,
            sources=all_sources,
            contradictions=contradictions,
            confidence=confidence,
            coverage=1.0  # TODO: Calculate actual coverage
        )
    
    def _detect_contradictions(self, claims: List[Claim]) -> List[Dict[str, Any]]:
        """Detect contradictions between claims"""
        contradictions = []
        
        for i, claim_a in enumerate(claims):
            for claim_b in claims[i+1:]:
                if self._claims_contradict(claim_a, claim_b):
                    contradictions.append({
                        "claim_a": claim_a.text,
                        "claim_b": claim_b.text,
                        "type": "content_conflict",
                        "resolution": self._suggest_resolution(claim_a, claim_b)
                    })
        
        return contradictions
    
    def _claims_contradict(self, a: Claim, b: Claim) -> bool:
        """Check if two claims contradict each other"""
        # Simple heuristic: similar topic but different stance
        a_words = set(a.text.lower().split())
        b_words = set(b.text.lower().split())
        
        overlap = len(a_words & b_words) / max(len(a_words), len(b_words))
        
        if overlap < 0.3:
            return False  # Not about same topic
        
        # Check for negation words
        negations = {'not', 'no', 'never', "n't", 'false', 'incorrect', 'wrong'}
        
        a_has_neg = bool(a_words & negations)
        b_has_neg = bool(b_words & negations)
        
        return a_has_neg != b_has_neg
    
    def _suggest_resolution(self, a: Claim, b: Claim) -> str:
        """Suggest how to resolve a contradiction"""
        if a.confidence > b.confidence + 0.2:
            return f"Prefer claim with higher confidence: '{a.text[:50]}...'"
        elif b.confidence > a.confidence + 0.2:
            return f"Prefer claim with higher confidence: '{b.text[:50]}...'"
        elif len(a.sources) > len(b.sources):
            return "Prefer claim with more source support"
        else:
            return "Present both perspectives with appropriate context"
    
    async def resolve_contradictions(self, conflicts: List[Conflict]) -> List[Dict[str, Any]]:
        """Resolve identified conflicts"""
        resolutions = []
        
        for conflict in conflicts:
            resolution = {
                "claims": [conflict.claim_a.text, conflict.claim_b.text],
                "type": conflict.conflict_type,
                "resolution": conflict.resolution or self._suggest_resolution(
                    conflict.claim_a, conflict.claim_b
                ),
                "preferred": None
            }
            
            # Determine which claim to prefer
            if conflict.claim_a.confidence > conflict.claim_b.confidence:
                resolution["preferred"] = "claim_a"
            elif conflict.claim_b.confidence > conflict.claim_a.confidence:
                resolution["preferred"] = "claim_b"
            
            resolutions.append(resolution)
        
        return resolutions
    
    async def generate_output(self, knowledge: MergedKnowledge, query: str,
                               format: OutputFormat = OutputFormat.RESEARCH_REPORT,
                               reasoning_trace: str = None) -> ResearchOutput:
        """Generate formatted output from merged knowledge"""
        # Add sources to citation memory
        for source in knowledge.sources:
            self._citation_memory.add_source(source)
        
        # Generate content based on format
        if format == OutputFormat.RESEARCH_REPORT:
            content = self._generate_report(knowledge, query, reasoning_trace)
        elif format == OutputFormat.EXECUTIVE_SUMMARY:
            content = self._generate_summary(knowledge, query)
        elif format == OutputFormat.FACT_SHEET:
            content = self._generate_fact_sheet(knowledge, query)
        elif format == OutputFormat.REASONING_TRACE:
            content = self._generate_trace(knowledge, query, reasoning_trace)
        elif format == OutputFormat.JSON:
            content = self._generate_json(knowledge, query)
        else:
            content = self._generate_report(knowledge, query, reasoning_trace)
        
        # Generate summary
        summary = self._generate_brief_summary(knowledge, query)
        
        return ResearchOutput(
            task_id="",  # Will be filled by orchestrator
            format=format,
            content=content,
            summary=summary,
            claims=knowledge.claims,
            sources=knowledge.sources,
            reasoning_trace=reasoning_trace,
            confidence=knowledge.confidence
        )
    
    def _generate_report(self, knowledge: MergedKnowledge, query: str,
                         reasoning_trace: str = None) -> str:
        """Generate academic-style research report"""
        sections = []
        
        # Title
        sections.append(f"# Research Report: {query[:50]}")
        sections.append("")
        
        # Executive Summary
        sections.append("## Executive Summary")
        verified_count = len([c for c in knowledge.claims 
                             if c.verification_status == VerificationStatus.VERIFIED])
        sections.append(f"This report synthesizes findings from {len(knowledge.sources)} sources "
                       f"with {len(knowledge.claims)} verified claims "
                       f"(confidence: {knowledge.confidence:.0%}).")
        sections.append("")
        
        # Key Findings
        sections.append("## Key Findings")
        for i, claim in enumerate(knowledge.claims[:10], 1):
            status_icon = {
                VerificationStatus.VERIFIED: "✓",
                VerificationStatus.LIKELY_TRUE: "◑",
                VerificationStatus.UNCERTAIN: "?",
                VerificationStatus.LIKELY_FALSE: "◒",
                VerificationStatus.FALSE: "✗",
            }.get(claim.verification_status, "○")
            
            # Add inline citation
            citations = []
            for source in claim.sources[:2]:
                self._citation_memory.cite(source.id)
                citations.append(self._citation_memory.format_inline_citation(source.id))
            
            citation_str = " ".join(citations) if citations else ""
            sections.append(f"{i}. {status_icon} {claim.text} {citation_str}")
        sections.append("")
        
        # Contradictions and Limitations
        if knowledge.contradictions:
            sections.append("## Contradictions Noted")
            for conflict in knowledge.contradictions[:5]:
                sections.append(f"- {conflict['claim_a'][:50]}... vs {conflict['claim_b'][:50]}...")
                sections.append(f"  Resolution: {conflict['resolution']}")
            sections.append("")
        
        # Confidence Assessment
        sections.append("## Confidence Assessment")
        sections.append(f"- Overall confidence: {knowledge.confidence:.0%}")
        sections.append(f"- Coverage: {knowledge.coverage:.0%}")
        sections.append(f"- Sources consulted: {len(knowledge.sources)}")
        sections.append("")
        
        # Reasoning trace if available
        if reasoning_trace:
            sections.append("## Reasoning Process")
            sections.append(reasoning_trace)
            sections.append("")
        
        # References
        sections.append(self._citation_memory.generate_bibliography())
        
        return "\n".join(sections)
    
    def _generate_summary(self, knowledge: MergedKnowledge, query: str) -> str:
        """Generate executive summary"""
        lines = [f"# Summary: {query[:50]}", ""]
        
        # Key points
        lines.append("**Key Points:**")
        for claim in knowledge.claims[:5]:
            if claim.verification_status in (VerificationStatus.VERIFIED, VerificationStatus.LIKELY_TRUE):
                lines.append(f"• {claim.text}")
        
        lines.append("")
        lines.append(f"**Confidence:** {knowledge.confidence:.0%}")
        lines.append(f"**Sources:** {len(knowledge.sources)}")
        
        if knowledge.contradictions:
            lines.append(f"**Note:** {len(knowledge.contradictions)} conflicts detected in sources")
        
        return "\n".join(lines)
    
    def _generate_fact_sheet(self, knowledge: MergedKnowledge, query: str) -> str:
        """Generate bullet-point fact sheet"""
        lines = [f"# Fact Sheet: {query[:50]}", ""]
        
        for claim in knowledge.claims:
            status = claim.verification_status.value.upper()
            sources = ", ".join(s.title[:20] or s.url[:30] for s in claim.sources[:2])
            lines.append(f"• [{status}] {claim.text}")
            if sources:
                lines.append(f"  Sources: {sources}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_trace(self, knowledge: MergedKnowledge, query: str,
                        reasoning_trace: str = None) -> str:
        """Generate reasoning trace output"""
        lines = [f"# Reasoning Trace: {query[:50]}", ""]
        
        if reasoning_trace:
            lines.append(reasoning_trace)
        else:
            lines.append("*No reasoning trace available*")
        
        lines.append("")
        lines.append("## Claims Derived:")
        for claim in knowledge.claims:
            lines.append(f"- {claim.text} (confidence: {claim.confidence:.0%})")
        
        return "\n".join(lines)
    
    def _generate_json(self, knowledge: MergedKnowledge, query: str) -> str:
        """Generate JSON output"""
        import json
        
        data = {
            "query": query,
            "confidence": knowledge.confidence,
            "coverage": knowledge.coverage,
            "claims": [
                {
                    "text": c.text,
                    "confidence": c.confidence,
                    "status": c.verification_status.value,
                    "sources": [{"url": s.url, "title": s.title} for s in c.sources]
                }
                for c in knowledge.claims
            ],
            "contradictions": knowledge.contradictions,
            "sources": [
                {
                    "url": s.url,
                    "title": s.title,
                    "reliability": s.reliability_score
                }
                for s in knowledge.sources
            ]
        }
        
        return json.dumps(data, indent=2)
    
    def _generate_brief_summary(self, knowledge: MergedKnowledge, query: str) -> str:
        """Generate a brief one-line summary"""
        if not knowledge.claims:
            return f"No verified information found for: {query}"
        
        top_claim = max(knowledge.claims, key=lambda c: c.confidence)
        return f"{top_claim.text[:100]}... (confidence: {knowledge.confidence:.0%})"
