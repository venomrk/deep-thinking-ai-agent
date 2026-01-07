"""
Main Orchestrator for Deep Research Agent
Coordinates all modules for end-to-end research execution.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

from ..core.base import (
    ResearchTask, ResearchOutput, OutputFormat, TaskStatus,
    Claim, Evidence, Source, SearchType, BaseModule
)
from ..core.planner import Planner, ResearchPlan
from ..core.pipeline import SixteenLayerPipeline, OutputController, OutputMode
from ..search.engine import DeepSearchEngine
from ..search.query_processor import QueryProcessor
from ..reasoning.tree_of_thoughts import TreeOfThoughts
from ..tools.manager import ToolManager
from ..memory.short_term import ShortTermMemory
from ..memory.long_term import LongTermMemory
from ..memory.citations import CitationMemory
from ..memory.failures import FailureMemory
from ..verification.fact_checker import FactChecker
from ..verification.hallucination_detector import HallucinationDetector
from ..synthesis.synthesizer import Synthesizer
from ..evaluation.self_eval import SelfEvaluator
from ..config.settings import get_settings, Settings


@dataclass
class ResearchSession:
    """Active research session"""
    id: str
    query: str
    plan: ResearchPlan
    status: TaskStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    output: Optional[ResearchOutput] = None
    progress: float = 0.0
    current_step: str = ""
    errors: List[str] = field(default_factory=list)


class Orchestrator(BaseModule):
    """
    Main orchestrator that coordinates all research modules.
    Implements the complete research pipeline:
    1. Plan: Parse intent, generate sub-questions
    2. Search: Execute multi-source searches
    3. Reason: Apply Tree-of-Thoughts reasoning
    4. Verify: Fact-check claims
    5. Synthesize: Generate final output
    """
    
    def __init__(self, settings: Settings = None, llm_client=None):
        super().__init__("orchestrator")
        self._settings = settings or get_settings()
        self._llm_client = llm_client
        
        # Initialize modules
        self._planner = Planner()
        self._search_engine = DeepSearchEngine()
        self._query_processor = QueryProcessor()
        self._reasoning_engine = TreeOfThoughts(self._settings.reasoning)
        self._tool_manager = ToolManager()
        
        # Memory systems
        self._short_term = ShortTermMemory(capacity=self._settings.memory.short_term_capacity)
        self._long_term = LongTermMemory(self._settings.memory.storage_path)
        self._citations = CitationMemory(self._settings.memory.storage_path / "citations")
        self._failures = FailureMemory(
            self._settings.memory.storage_path / "failures",
            ttl_hours=self._settings.memory.failure_memory_ttl_hours
        )
        
        # Verification
        self._fact_checker = FactChecker(self._search_engine)
        self._hallucination_detector = HallucinationDetector()
        
        # Synthesis
        self._synthesizer = Synthesizer(self._citations)
        
        # 16-Layer Pipeline (hidden reasoning)
        self._pipeline = SixteenLayerPipeline()
        self._output_controller = OutputController()
        
        # Self-evaluation (internal only)
        self._evaluator = SelfEvaluator()
        
        # Session tracking
        self._active_sessions: Dict[str, ResearchSession] = {}
        self._progress_callbacks: List[Callable] = []
    
    async def initialize(self) -> None:
        """Initialize all modules"""
        # Set LLM client on modules
        if self._llm_client:
            self._planner.set_llm_client(self._llm_client)
            self._reasoning_engine.set_llm_client(self._llm_client)
            self._fact_checker.set_llm_client(self._llm_client)
            self._synthesizer.set_llm_client(self._llm_client)
            self._long_term.set_llm_client(self._llm_client)
        
        # Initialize modules
        await self._planner.initialize()
        await self._search_engine.initialize()
        await self._reasoning_engine.initialize()
        await self._tool_manager.initialize()
        await self._fact_checker.initialize()
        await self._synthesizer.initialize()
        
        self._initialized = True
    
    async def shutdown(self) -> None:
        """Shutdown all modules"""
        await self._planner.shutdown()
        await self._search_engine.shutdown()
        await self._reasoning_engine.shutdown()
        await self._tool_manager.shutdown()
        await self._fact_checker.shutdown()
        await self._synthesizer.shutdown()
        
        self._initialized = False
    
    def set_llm_client(self, client) -> None:
        """Set LLM client for all modules"""
        self._llm_client = client
        if self._initialized:
            self._planner.set_llm_client(client)
            self._reasoning_engine.set_llm_client(client)
            self._fact_checker.set_llm_client(client)
            self._synthesizer.set_llm_client(client)
            self._long_term.set_llm_client(client)
    
    def add_progress_callback(self, callback: Callable) -> None:
        """Add a callback for progress updates"""
        self._progress_callbacks.append(callback)
    
    def _update_progress(self, session: ResearchSession, progress: float, step: str) -> None:
        """Update session progress"""
        session.progress = progress
        session.current_step = step
        
        for callback in self._progress_callbacks:
            try:
                callback(session.id, progress, step)
            except Exception:
                pass
    
    async def research(self, query: str, 
                       output_format: OutputFormat = OutputFormat.RESEARCH_REPORT) -> ResearchOutput:
        """
        Execute a complete research task.
        
        Args:
            query: The research query
            output_format: Desired output format
            
        Returns:
            ResearchOutput with synthesized findings
        """
        from uuid import uuid4
        
        if not self._initialized:
            await self.initialize()
        
        # Create session
        session_id = str(uuid4())
        
        # Phase 1: Planning
        self._short_term.set_context(query)
        plan = await self._planner.create_research_plan(query)
        
        session = ResearchSession(
            id=session_id,
            query=query,
            plan=plan,
            status=TaskStatus.IN_PROGRESS,
            started_at=datetime.now()
        )
        self._active_sessions[session_id] = session
        
        self._update_progress(session, 0.1, "Planning complete")
        
        try:
            # Phase 2: Search
            all_evidence = []
            prioritized_questions = self._planner.prioritize_questions(plan.sub_questions)
            
            for i, question in enumerate(prioritized_questions):
                question.status = TaskStatus.IN_PROGRESS
                
                # Process query
                processed = self._query_processor.process(question.question)
                
                # Execute search
                search_types = [question.search_type]
                if processed.search_hints.get("is_academic"):
                    search_types.append(SearchType.ACADEMIC)
                
                results = await self._search_engine.search(
                    processed.expanded,
                    search_types=search_types,
                    num_results=5
                )
                
                # Convert to evidence
                for result in results:
                    source = result.to_source()
                    self._citations.add_source(source)
                    self._citations.estimate_reliability(source)
                    
                    evidence = Evidence(
                        source=source,
                        content=result.snippet,
                        relevance_score=result.relevance_score
                    )
                    all_evidence.append(evidence)
                    self._short_term.store_evidence(evidence)
                
                question.status = TaskStatus.COMPLETED
                progress = 0.1 + (0.3 * (i + 1) / len(prioritized_questions))
                self._update_progress(session, progress, f"Searched: {question.question[:30]}...")
            
            self._update_progress(session, 0.4, "Search complete")
            
            # Phase 3: Reasoning
            reasoning_result = await self._reasoning_engine.reason(
                problem=query,
                context={"plan": plan.task.intent},
                evidence=all_evidence
            )
            
            reasoning_trace = self._reasoning_engine.visualize_tree()
            self._update_progress(session, 0.6, "Reasoning complete")
            
            # Phase 4: Extract and verify claims
            claims = self._extract_claims(all_evidence, reasoning_result.conclusion)
            
            verified_claims = []
            for i, claim in enumerate(claims):
                result = await self._fact_checker.verify_claim(claim)
                verified_claims.append(result.claim)
                
                progress = 0.6 + (0.2 * (i + 1) / len(claims))
                self._update_progress(session, progress, f"Verifying claim {i+1}/{len(claims)}")
            
            self._update_progress(session, 0.8, "Verification complete")
            
            # Phase 5: Synthesis
            knowledge = await self._synthesizer.merge_facts(verified_claims)
            
            output = await self._synthesizer.generate_output(
                knowledge=knowledge,
                query=query,
                format=output_format,
                reasoning_trace=reasoning_trace
            )
            output.task_id = session_id
            
            # Check for hallucinations
            hallucination_report = self._hallucination_detector.analyze(
                output.content,
                sources=knowledge.sources
            )
            
            if hallucination_report.hallucination_risk > 0.5:
                output.content = hallucination_report.clean_content
                output.metadata["hallucination_warning"] = True
            
            # Store in long-term memory
            await self._long_term.store(
                content=output.content,
                content_type="research_output",
                metadata={"query": query, "format": output_format.value}
            )
            
            # Apply output compression based on mode
            output_mode = OutputController.detect_mode(query)
            if output_mode == OutputMode.CONCISE:
                output.content = OutputController.compress(output.content, output_mode)
            
            # Internal self-evaluation (never exposed)
            self._evaluator.evaluate(
                query=query,
                response=output.content,
                sources_used=len(knowledge.sources),
                contradictions=len(knowledge.contradictions),
                confidence=output.confidence
            )
            
            # Complete session
            session.status = TaskStatus.COMPLETED
            session.completed_at = datetime.now()
            session.output = output
            self._update_progress(session, 1.0, "Complete")
            
            return output
            
        except Exception as e:
            session.status = TaskStatus.FAILED
            session.errors.append(str(e))
            self._failures.record_failure(
                failure_type="research_failure",
                context=query,
                action="research",
                error_message=str(e)
            )
            raise
    
    def _extract_claims(self, evidence: List[Evidence], 
                        conclusion: Optional[str]) -> List[Claim]:
        """Extract claims from evidence and reasoning"""
        claims = []
        
        # Extract from evidence
        for ev in evidence:
            if ev.relevance_score > 0.5:
                claim = Claim(
                    text=ev.content[:200],
                    confidence=ev.relevance_score,
                    sources=[ev.source]
                )
                claims.append(claim)
        
        # Extract from conclusion
        if conclusion:
            # Split into sentences
            import re
            sentences = re.split(r'[.!?]+', conclusion)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    claim = Claim(
                        text=sentence,
                        confidence=0.7,
                        sources=[]
                    )
                    claims.append(claim)
        
        # Deduplicate and limit
        seen = set()
        unique_claims = []
        for claim in claims:
            key = claim.text[:50].lower()
            if key not in seen:
                seen.add(key)
                unique_claims.append(claim)
        
        return unique_claims[:20]  # Limit to top 20
    
    def get_session(self, session_id: str) -> Optional[ResearchSession]:
        """Get a research session by ID"""
        return self._active_sessions.get(session_id)
    
    def get_active_sessions(self) -> List[ResearchSession]:
        """Get all active sessions"""
        return [
            s for s in self._active_sessions.values()
            if s.status == TaskStatus.IN_PROGRESS
        ]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            "short_term": self._short_term.get_context_summary(),
            "long_term": self._long_term.get_stats(),
            "citations": self._citations.get_stats(),
            "failures": self._failures.get_stats()
        }
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get reasoning engine statistics"""
        return self._reasoning_engine.get_tree_stats()
    
    def clear_memory(self) -> None:
        """Clear all memory"""
        self._short_term.clear()
        self._long_term.clear()
        self._citations.clear()
        self._failures.clear()
