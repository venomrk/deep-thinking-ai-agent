"""
Base Classes and Data Structures for Deep Research Agent
Defines core interfaces and data types used throughout the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, TypeVar, Generic
from uuid import uuid4
import json


# ==================== ENUMS ====================

class TaskStatus(Enum):
    """Status of a research task"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SearchType(Enum):
    """Types of search operations"""
    WEB = "web"
    ACADEMIC = "academic"
    LOCAL = "local"
    CALCULATION = "calculation"


class EvidenceType(Enum):
    """Types of evidence"""
    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"
    STATISTIC = "statistic"
    FACT = "fact"
    OPINION = "opinion"
    INFERENCE = "inference"


class ConfidenceLevel(Enum):
    """Confidence levels for claims"""
    DEFINITE = "definite"      # 0.9+
    PROBABLE = "probable"      # 0.7-0.9
    POSSIBLE = "possible"      # 0.5-0.7
    SPECULATIVE = "speculative"  # <0.5


class VerificationStatus(Enum):
    """Status of claim verification"""
    VERIFIED = "verified"
    LIKELY_TRUE = "likely_true"
    UNCERTAIN = "uncertain"
    LIKELY_FALSE = "likely_false"
    FALSE = "false"
    UNVERIFIED = "unverified"


# ==================== BASE CLASSES ====================

class BaseModule(ABC):
    """Abstract base class for all agent modules"""
    
    def __init__(self, name: str):
        self.name = name
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the module"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown of the module"""
        pass
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized


# ==================== DATA STRUCTURES ====================

@dataclass
class Source:
    """Represents an information source"""
    id: str = field(default_factory=lambda: str(uuid4()))
    url: Optional[str] = None
    title: str = ""
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    accessed_date: datetime = field(default_factory=datetime.now)
    source_type: str = "web"  # web, academic, local, generated
    reliability_score: float = 0.5
    content_snippet: str = ""
    full_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_citation(self, style: str = "apa") -> str:
        """Generate citation in specified style"""
        if style == "apa":
            author = self.author or "Unknown"
            year = self.published_date.year if self.published_date else "n.d."
            return f"{author} ({year}). {self.title}. Retrieved from {self.url}"
        elif style == "ieee":
            author = self.author or "Unknown"
            return f'[{self.id[:8]}] {author}, "{self.title}," {self.url}'
        else:
            return f"{self.title} ({self.url})"


@dataclass
class Evidence:
    """A piece of evidence supporting or refuting a claim"""
    id: str = field(default_factory=lambda: str(uuid4()))
    source: Source = field(default_factory=Source)
    content: str = ""
    evidence_type: EvidenceType = EvidenceType.FACT
    relevance_score: float = 0.5
    supports_claim: bool = True
    extracted_at: datetime = field(default_factory=datetime.now)
    context: str = ""
    
    def __str__(self) -> str:
        stance = "supports" if self.supports_claim else "contradicts"
        return f"[{self.evidence_type.value}] {self.content[:100]}... ({stance})"


@dataclass
class Claim:
    """A claim or assertion that can be verified"""
    id: str = field(default_factory=lambda: str(uuid4()))
    text: str = ""
    confidence: float = 0.5
    confidence_level: ConfidenceLevel = ConfidenceLevel.POSSIBLE
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    supporting_evidence: List[Evidence] = field(default_factory=list)
    contradicting_evidence: List[Evidence] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    verified_at: Optional[datetime] = None
    
    def update_confidence(self) -> None:
        """Update confidence based on evidence"""
        if not self.supporting_evidence and not self.contradicting_evidence:
            self.confidence = 0.5
        else:
            support_score = sum(e.relevance_score for e in self.supporting_evidence)
            contradict_score = sum(e.relevance_score for e in self.contradicting_evidence)
            total = support_score + contradict_score
            if total > 0:
                self.confidence = support_score / total
        
        # Update confidence level
        if self.confidence >= 0.9:
            self.confidence_level = ConfidenceLevel.DEFINITE
        elif self.confidence >= 0.7:
            self.confidence_level = ConfidenceLevel.PROBABLE
        elif self.confidence >= 0.5:
            self.confidence_level = ConfidenceLevel.POSSIBLE
        else:
            self.confidence_level = ConfidenceLevel.SPECULATIVE


@dataclass
class SubQuestion:
    """A sub-question derived from the main query"""
    id: str = field(default_factory=lambda: str(uuid4()))
    question: str = ""
    rationale: str = ""
    search_type: SearchType = SearchType.WEB
    importance: int = 5
    depends_on: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    answer: Optional[str] = None
    evidence: List[Evidence] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ResearchTask:
    """A complete research task"""
    id: str = field(default_factory=lambda: str(uuid4()))
    query: str = ""
    intent: str = ""
    complexity: int = 5
    domains: List[str] = field(default_factory=list)
    sub_questions: List[SubQuestion] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    claims: List[Claim] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    final_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== REASONING STRUCTURES ====================

@dataclass
class ReasoningNode:
    """A node in the Tree-of-Thoughts reasoning tree"""
    id: str = field(default_factory=lambda: str(uuid4()))
    thought: str = ""
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    confidence: float = 0.5
    is_terminal: bool = False
    is_pruned: bool = False
    evidence: List[Evidence] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    falsification_criteria: str = ""
    evaluation_scores: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0
    
    @property
    def overall_score(self) -> float:
        if not self.evaluation_scores:
            return self.confidence
        return sum(self.evaluation_scores.values()) / len(self.evaluation_scores)


@dataclass
class ReasoningTree:
    """Complete reasoning tree structure"""
    id: str = field(default_factory=lambda: str(uuid4()))
    problem: str = ""
    root_id: Optional[str] = None
    nodes: Dict[str, ReasoningNode] = field(default_factory=dict)
    best_path: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_node(self, node: ReasoningNode) -> None:
        """Add a node to the tree"""
        self.nodes[node.id] = node
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].children_ids.append(node.id)
    
    def get_node(self, node_id: str) -> Optional[ReasoningNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_path_to_node(self, node_id: str) -> List[ReasoningNode]:
        """Get the path from root to a specific node"""
        path = []
        current_id = node_id
        while current_id and current_id in self.nodes:
            node = self.nodes[current_id]
            path.insert(0, node)
            current_id = node.parent_id
        return path
    
    def get_leaves(self) -> List[ReasoningNode]:
        """Get all leaf nodes"""
        return [n for n in self.nodes.values() if n.is_leaf and not n.is_pruned]
    
    def prune_node(self, node_id: str) -> None:
        """Mark a node and all descendants as pruned"""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.is_pruned = True
        
        for child_id in node.children_ids:
            self.prune_node(child_id)


# ==================== TOOL STRUCTURES ====================

@dataclass
class ToolResult:
    """Result from a tool execution"""
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolIntent:
    """Intent for tool selection"""
    action: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: str = ""
    urgency: int = 5
    fallback_allowed: bool = True


# ==================== MEMORY STRUCTURES ====================

@dataclass
class MemoryEntry:
    """An entry in the memory system"""
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    content_type: str = "text"
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 0.5
    
    def access(self) -> None:
        """Record an access to this memory"""
        self.accessed_at = datetime.now()
        self.access_count += 1


@dataclass
class FailureRecord:
    """Record of a failure for learning"""
    id: str = field(default_factory=lambda: str(uuid4()))
    failure_type: str = ""
    context: str = ""
    action_attempted: str = ""
    error_message: str = ""
    recovery_action: Optional[str] = None
    prevented_recurrence: bool = False
    created_at: datetime = field(default_factory=datetime.now)


# ==================== OUTPUT STRUCTURES ====================

class OutputFormat(Enum):
    """Supported output formats"""
    RESEARCH_REPORT = "research_report"
    EXECUTIVE_SUMMARY = "executive_summary"
    FACT_SHEET = "fact_sheet"
    REASONING_TRACE = "reasoning_trace"
    JSON = "json"


@dataclass
class ResearchOutput:
    """Final output from a research task"""
    task_id: str = ""
    format: OutputFormat = OutputFormat.RESEARCH_REPORT
    content: str = ""
    summary: str = ""
    claims: List[Claim] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    reasoning_trace: Optional[str] = None
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps({
            "task_id": self.task_id,
            "format": self.format.value,
            "content": self.content,
            "summary": self.summary,
            "confidence": self.confidence,
            "claims": [
                {
                    "text": c.text,
                    "confidence": c.confidence,
                    "status": c.verification_status.value,
                    "sources": [s.url for s in c.sources]
                }
                for c in self.claims
            ],
            "sources": [
                {
                    "title": s.title,
                    "url": s.url,
                    "reliability": s.reliability_score
                }
                for s in self.sources
            ],
            "created_at": self.created_at.isoformat()
        }, indent=2)
