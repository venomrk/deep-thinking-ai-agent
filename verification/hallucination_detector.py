"""
Hallucination Detector for Deep Research Agent
Identifies generated content that may be fabricated.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re

from ..core.base import Source, Evidence


@dataclass
class HallucinationFlag:
    """A potential hallucination in content"""
    text: str
    issue_type: str  # unsourced, fabricated, inconsistent, overconfident
    severity: str  # high, medium, low
    explanation: str
    suggested_fix: str
    start_pos: int
    end_pos: int


@dataclass
class HallucinationReport:
    """Report of hallucination analysis"""
    original_content: str
    flags: List[HallucinationFlag]
    hallucination_risk: float  # 0-1
    clean_content: str
    sources_required: List[str]


class HallucinationDetector:
    """
    Detects potential hallucinations in generated content.
    Checks for:
    - Claims without source backing
    - Fabricated statistics/numbers
    - Inconsistencies with known facts
    - Overconfident assertions
    """
    
    def __init__(self):
        # Patterns that often indicate potential hallucinations
        self._confidence_patterns = [
            (r'\bdefinitely\b', 'overconfident', 'low'),
            (r'\babsolutely\b', 'overconfident', 'low'),
            (r'\balways\b', 'overconfident', 'medium'),
            (r'\bnever\b', 'overconfident', 'medium'),
            (r'\beveryone knows\b', 'overconfident', 'medium'),
            (r'\bit is certain\b', 'overconfident', 'medium'),
            (r'\bwithout a doubt\b', 'overconfident', 'low'),
        ]
        
        # Patterns for statistics that need sourcing
        self._stat_patterns = [
            r'\b\d+(?:\.\d+)?%',  # Percentages
            r'\$\d+(?:,\d{3})*(?:\.\d+)?',  # Dollar amounts
            r'\b\d{4}\b(?=.*(?:study|research|found|showed))',  # Years with research
            r'\b\d+(?:,\d{3})+\b',  # Large numbers
        ]
        
        # Patterns for citations that might be fabricated
        self._citation_patterns = [
            r'according to (?:a |the )?(?:\d{4} )?(?:study|research|report)',
            r'(?:study|research) (?:by|from|at) [A-Z][a-z]+',
            r'\([A-Z][a-z]+,? \d{4}\)',  # (Author, Year) format
            r'et al\.',
        ]
    
    def analyze(self, content: str, sources: List[Source] = None,
                known_facts: List[str] = None) -> HallucinationReport:
        """
        Analyze content for potential hallucinations.
        
        Args:
            content: The generated content to analyze
            sources: Known sources that should back claims
            known_facts: Known facts to check consistency
            
        Returns:
            HallucinationReport with flags and clean content
        """
        sources = sources or []
        known_facts = known_facts or []
        
        flags = []
        
        # Check for overconfident language
        flags.extend(self._check_overconfidence(content))
        
        # Check for unsourced statistics
        flags.extend(self._check_unsourced_stats(content, sources))
        
        # Check for potentially fabricated citations
        flags.extend(self._check_fabricated_citations(content, sources))
        
        # Check for inconsistencies with known facts
        flags.extend(self._check_inconsistencies(content, known_facts))
        
        # Check for unsourced specific claims
        flags.extend(self._check_unsourced_specifics(content, sources))
        
        # Calculate overall risk
        risk = self._calculate_risk(flags, len(content))
        
        # Generate clean content
        clean = self._generate_clean_content(content, flags)
        
        # Identify claims that need sources
        sources_needed = self._identify_source_needs(content, flags)
        
        return HallucinationReport(
            original_content=content,
            flags=flags,
            hallucination_risk=risk,
            clean_content=clean,
            sources_required=sources_needed
        )
    
    def _check_overconfidence(self, content: str) -> List[HallucinationFlag]:
        """Check for overconfident language"""
        flags = []
        content_lower = content.lower()
        
        for pattern, issue_type, severity in self._confidence_patterns:
            for match in re.finditer(pattern, content_lower):
                flags.append(HallucinationFlag(
                    text=match.group(),
                    issue_type=issue_type,
                    severity=severity,
                    explanation=f"Overconfident language: '{match.group()}'",
                    suggested_fix=f"Consider hedging: 'likely', 'often', 'typically'",
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        return flags
    
    def _check_unsourced_stats(self, content: str, 
                                sources: List[Source]) -> List[HallucinationFlag]:
        """Check for statistics that aren't backed by sources"""
        flags = []
        
        # Collect all numbers/stats mentioned in sources
        source_numbers = set()
        for source in sources:
            source_text = source.content_snippet or ""
            for pattern in self._stat_patterns:
                for match in re.finditer(pattern, source_text):
                    source_numbers.add(match.group())
        
        # Find stats in content
        for pattern in self._stat_patterns:
            for match in re.finditer(pattern, content):
                stat = match.group()
                if stat not in source_numbers:
                    # Context around the stat
                    start = max(0, match.start() - 30)
                    end = min(len(content), match.end() + 30)
                    context = content[start:end]
                    
                    flags.append(HallucinationFlag(
                        text=stat,
                        issue_type="unsourced",
                        severity="high",
                        explanation=f"Statistic '{stat}' not found in provided sources",
                        suggested_fix="Verify this statistic or add a citation",
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
        
        return flags
    
    def _check_fabricated_citations(self, content: str,
                                     sources: List[Source]) -> List[HallucinationFlag]:
        """Check for citations that might be fabricated"""
        flags = []
        
        # Get list of actual source titles/authors
        known_sources = set()
        for source in sources:
            if source.title:
                known_sources.add(source.title.lower())
            if source.author:
                known_sources.add(source.author.lower())
        
        for pattern in self._citation_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                citation = match.group()
                
                # Check if this matches any known source
                found = any(
                    known.lower() in citation.lower() or citation.lower() in known.lower()
                    for known in known_sources
                )
                
                if not found and not sources:
                    # Only flag if we have sources to compare against
                    continue
                
                if not found:
                    flags.append(HallucinationFlag(
                        text=citation,
                        issue_type="fabricated",
                        severity="high",
                        explanation=f"Citation '{citation}' not found in known sources",
                        suggested_fix="Verify this citation exists or remove it",
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
        
        return flags
    
    def _check_inconsistencies(self, content: str,
                                known_facts: List[str]) -> List[HallucinationFlag]:
        """Check for inconsistencies with known facts"""
        flags = []
        
        # Simple word-based contradiction detection
        negation_words = {'not', 'never', 'no', "n't", 'false', 'incorrect'}
        
        for fact in known_facts:
            fact_words = set(fact.lower().split())
            
            # Split content into sentences
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                sentence_words = set(sentence_lower.split())
                
                # Check if sentence discusses same topic
                overlap = len(fact_words & sentence_words) / len(fact_words) if fact_words else 0
                
                if overlap > 0.5:
                    # Check for contradiction (negation present)
                    has_negation = bool(negation_words & sentence_words)
                    
                    if has_negation:
                        flags.append(HallucinationFlag(
                            text=sentence.strip(),
                            issue_type="inconsistent",
                            severity="medium",
                            explanation=f"May contradict known fact: '{fact[:50]}...'",
                            suggested_fix="Verify this claim against established facts",
                            start_pos=content.find(sentence),
                            end_pos=content.find(sentence) + len(sentence)
                        ))
        
        return flags
    
    def _check_unsourced_specifics(self, content: str,
                                    sources: List[Source]) -> List[HallucinationFlag]:
        """Check for specific claims that should have sources"""
        flags = []
        
        # Patterns for specific claims
        specific_patterns = [
            r'(?:founded|established|created|invented) in \d{4}',
            r'CEO of [A-Z][a-zA-Z]+',
            r'located in [A-Z][a-zA-Z]+(?:, [A-Z][a-zA-Z]+)?',
            r'worth (?:over |approximately )?\$[\d,.]+(?:M|B|million|billion)?',
        ]
        
        source_content = " ".join(s.content_snippet or "" for s in sources)
        
        for pattern in specific_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                claim = match.group()
                
                # Check if this specific claim is in sources
                if claim.lower() not in source_content.lower():
                    flags.append(HallucinationFlag(
                        text=claim,
                        issue_type="unsourced",
                        severity="medium",
                        explanation=f"Specific claim '{claim}' not verified in sources",
                        suggested_fix="Add a citation or verify this information",
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
        
        return flags
    
    def _calculate_risk(self, flags: List[HallucinationFlag], 
                        content_length: int) -> float:
        """Calculate overall hallucination risk"""
        if not flags:
            return 0.0
        
        # Weight by severity
        severity_weights = {"high": 1.0, "medium": 0.5, "low": 0.2}
        
        total_weight = sum(
            severity_weights.get(f.severity, 0.3) 
            for f in flags
        )
        
        # Normalize by content length (more flags per char = higher risk)
        length_factor = min(1.0, content_length / 500)  # Normalize to ~500 chars
        
        risk = min(1.0, total_weight / (5 * length_factor))
        
        return risk
    
    def _generate_clean_content(self, content: str,
                                 flags: List[HallucinationFlag]) -> str:
        """Generate clean version with hedging added"""
        clean = content
        
        # Track offset from modifications
        offset = 0
        
        # Sort flags by position
        sorted_flags = sorted(flags, key=lambda f: f.start_pos)
        
        for flag in sorted_flags:
            if flag.issue_type == "overconfident":
                # Add hedging
                original = clean[flag.start_pos + offset:flag.end_pos + offset]
                hedged = f"(likely) {original}"
                clean = clean[:flag.start_pos + offset] + hedged + clean[flag.end_pos + offset:]
                offset += len(hedged) - len(original)
            
            elif flag.severity == "high":
                # Add warning marker
                original = clean[flag.start_pos + offset:flag.end_pos + offset]
                marked = f"[UNVERIFIED: {original}]"
                clean = clean[:flag.start_pos + offset] + marked + clean[flag.end_pos + offset:]
                offset += len(marked) - len(original)
        
        return clean
    
    def _identify_source_needs(self, content: str,
                                flags: List[HallucinationFlag]) -> List[str]:
        """Identify claims that need source citations"""
        needs = []
        
        for flag in flags:
            if flag.issue_type in ("unsourced", "fabricated"):
                needs.append(flag.text)
        
        return needs
