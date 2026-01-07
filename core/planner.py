"""
Planner Module for Deep Research Agent
Decomposes user intent into structured research tasks.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import re

from ..core.base import (
    ResearchTask, SubQuestion, SearchType, TaskStatus, BaseModule
)
from ..config.prompts import PromptTemplates
from ..config.settings import get_settings

# Global flag to skip LLM calls after quota exhaustion (for speed)
_llm_quota_exhausted = False


@dataclass
class Intent:
    """Parsed user intent"""
    primary_intent: str  # research, factual_question, comparison, explanation, analysis
    complexity: int  # 1-10
    domains: List[str]
    constraints: List[str]
    ambiguities: List[str]
    requires_clarification: bool


@dataclass
class SearchStrategy:
    """Strategy for searching"""
    strategy: str  # deep, broad, targeted
    priority_engines: List[str]
    num_queries: int
    query_refinements: List[str]
    rationale: str


@dataclass
class ResearchPlan:
    """Complete research plan"""
    task: ResearchTask
    sub_questions: List[SubQuestion]
    search_strategy: SearchStrategy
    estimated_time: int  # seconds
    estimated_cost: float


class Planner(BaseModule):
    """
    Plans and decomposes research tasks.
    Features:
    - Intent parsing
    - Sub-question generation
    - Search strategy selection
    - Task dependency management
    """
    
    def __init__(self, llm_client=None):
        super().__init__("planner")
        self._llm_client = llm_client
        self._settings = get_settings()
    
    async def initialize(self) -> None:
        """Initialize the planner"""
        self._initialized = True
    
    async def shutdown(self) -> None:
        """Clean shutdown"""
        self._initialized = False
    
    def set_llm_client(self, client) -> None:
        """Set LLM client"""
        self._llm_client = client
    
    async def parse_intent(self, query: str) -> Intent:
        """Parse user query to understand intent"""
        if self._llm_client:
            return await self._parse_intent_with_llm(query)
        return self._parse_intent_heuristic(query)
    
    async def _parse_intent_with_llm(self, query: str) -> Intent:
        """Parse intent using LLM"""
        global _llm_quota_exhausted
        
        # Skip LLM if quota already exhausted (speed optimization)
        if _llm_quota_exhausted:
            return self._parse_intent_heuristic(query)
        
        prompt = PromptTemplates.INTENT_PARSER.substitute(query=query)
        
        try:
            response = await self._llm_client.generate_content_async(prompt)
            return self._parse_intent_response(response.text)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                _llm_quota_exhausted = True
                print(f"LLM quota exhausted - switching to fast fallback mode")
            else:
                print(f"LLM intent parsing failed: {e}")
            return self._parse_intent_heuristic(query)
    
    def _parse_intent_response(self, response: str) -> Intent:
        """Parse LLM response to Intent"""
        try:
            # Extract JSON
            json_match = response.find('{')
            if json_match >= 0:
                json_end = response.rfind('}') + 1
                data = json.loads(response[json_match:json_end])
                
                return Intent(
                    primary_intent=data.get("primary_intent", "research"),
                    complexity=int(data.get("complexity", 5)),
                    domains=data.get("domains", []),
                    constraints=data.get("constraints", []),
                    ambiguities=data.get("ambiguities", []),
                    requires_clarification=data.get("requires_clarification", False)
                )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        
        return Intent(
            primary_intent="research",
            complexity=5,
            domains=[],
            constraints=[],
            ambiguities=[],
            requires_clarification=False
        )
    
    def _parse_intent_heuristic(self, query: str) -> Intent:
        """Parse intent using heuristics (fallback)"""
        query_lower = query.lower()
        
        # Detect intent type
        if any(w in query_lower for w in ["what is", "define", "meaning of"]):
            intent = "factual_question"
        elif any(w in query_lower for w in ["compare", "vs", "versus", "difference"]):
            intent = "comparison"
        elif any(w in query_lower for w in ["how", "why", "explain"]):
            intent = "explanation"
        elif any(w in query_lower for w in ["analyze", "analysis", "evaluate"]):
            intent = "analysis"
        else:
            intent = "research"
        
        # Estimate complexity
        word_count = len(query.split())
        if word_count < 5:
            complexity = 3
        elif word_count < 15:
            complexity = 5
        else:
            complexity = 7
        
        # Detect domains
        domains = []
        domain_keywords = {
            "technology": ["software", "code", "programming", "tech", "ai", "ml"],
            "finance": ["stock", "market", "trading", "investment", "crypto"],
            "science": ["research", "study", "experiment", "data"],
            "health": ["medical", "health", "disease", "treatment"],
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                domains.append(domain)
        
        return Intent(
            primary_intent=intent,
            complexity=complexity,
            domains=domains,
            constraints=[],
            ambiguities=[],
            requires_clarification=False
        )
    
    async def generate_sub_questions(self, query: str, intent: Intent) -> List[SubQuestion]:
        """Generate sub-questions for research"""
        if self._llm_client:
            return await self._generate_sub_questions_llm(query, intent)
        return self._generate_sub_questions_heuristic(query, intent)
    
    async def _generate_sub_questions_llm(self, query: str, intent: Intent) -> List[SubQuestion]:
        """Generate sub-questions using LLM"""
        global _llm_quota_exhausted
        
        # Skip LLM if quota already exhausted (speed optimization)
        if _llm_quota_exhausted:
            return self._generate_sub_questions_heuristic(query, intent)
        
        prompt = PromptTemplates.SUB_QUESTION_GENERATOR.substitute(
            query=query,
            intent=intent.primary_intent,
            complexity=intent.complexity
        )
        
        try:
            response = await self._llm_client.generate_content_async(prompt)
            return self._parse_sub_questions(response.text)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                _llm_quota_exhausted = True
            return self._generate_sub_questions_heuristic(query, intent)
    
    def _parse_sub_questions(self, response: str) -> List[SubQuestion]:
        """Parse LLM response to sub-questions"""
        questions = []
        
        try:
            json_match = response.find('[')
            if json_match >= 0:
                json_end = response.rfind(']') + 1
                data = json.loads(response[json_match:json_end])
                
                for item in data:
                    search_type_str = item.get("search_type", "web")
                    search_type = {
                        "web": SearchType.WEB,
                        "academic": SearchType.ACADEMIC,
                        "local": SearchType.LOCAL,
                        "calculation": SearchType.CALCULATION
                    }.get(search_type_str, SearchType.WEB)
                    
                    questions.append(SubQuestion(
                        id=item.get("id", f"q{len(questions)}"),
                        question=item.get("question", ""),
                        rationale=item.get("rationale", ""),
                        search_type=search_type,
                        importance=int(item.get("importance", 5)),
                        depends_on=item.get("depends_on", [])
                    ))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        
        return questions
    
    def _generate_sub_questions_heuristic(self, query: str, intent: Intent) -> List[SubQuestion]:
        """Generate sub-questions using heuristics (fast mode - single question)"""
        # Single question for speed - covers the main query
        return [SubQuestion(
            id="q0",
            question=query,  # Use original query directly
            rationale="Direct query for fast response",
            search_type=SearchType.WEB,
            importance=9,
            depends_on=[]
        )]
    
    async def decide_search_strategy(self, task: ResearchTask,
                                      current_knowledge: str = "") -> SearchStrategy:
        """Decide optimal search strategy"""
        # Base strategy on complexity
        if task.complexity <= 3:
            strategy = "targeted"
            num_queries = 3
        elif task.complexity <= 6:
            strategy = "broad"
            num_queries = 5
        else:
            strategy = "deep"
            num_queries = 8
        
        # Choose engines
        engines = ["duckduckgo"]
        if any(d in ["science", "technology"] for d in task.domains):
            engines.append("academic")
        
        return SearchStrategy(
            strategy=strategy,
            priority_engines=engines,
            num_queries=num_queries,
            query_refinements=[],
            rationale=f"Using {strategy} strategy based on complexity {task.complexity}"
        )
    
    async def create_research_plan(self, query: str) -> ResearchPlan:
        """Create a complete research plan"""
        # Parse intent
        intent = await self.parse_intent(query)
        
        # Generate sub-questions
        sub_questions = await self.generate_sub_questions(query, intent)
        
        # Create task
        task = ResearchTask(
            query=query,
            intent=intent.primary_intent,
            complexity=intent.complexity,
            domains=intent.domains,
            sub_questions=sub_questions,
            status=TaskStatus.PENDING
        )
        
        # Decide strategy
        strategy = await self.decide_search_strategy(task)
        
        # Estimate time and cost
        estimated_time = intent.complexity * 30  # seconds
        estimated_cost = len(sub_questions) * 0.01  # rough estimate
        
        return ResearchPlan(
            task=task,
            sub_questions=sub_questions,
            search_strategy=strategy,
            estimated_time=estimated_time,
            estimated_cost=estimated_cost
        )
    
    def prioritize_questions(self, questions: List[SubQuestion]) -> List[SubQuestion]:
        """Prioritize sub-questions by importance and dependencies"""
        # Build dependency graph
        completed = set()
        ordered = []
        remaining = list(questions)
        
        while remaining:
            # Find questions with satisfied dependencies
            ready = [
                q for q in remaining
                if all(dep in completed for dep in q.depends_on)
            ]
            
            if not ready:
                # Break cycle by taking highest importance
                ready = [max(remaining, key=lambda q: q.importance)]
            
            # Sort by importance
            ready.sort(key=lambda q: -q.importance)
            
            for q in ready:
                ordered.append(q)
                completed.add(q.id)
                remaining.remove(q)
        
        return ordered
