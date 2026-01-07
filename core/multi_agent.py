"""
Multi-Agent Orchestration System (CrewAI/LangGraph Style)
Enables collaboration between specialized agents.
"""

from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import uuid


class AgentRole(Enum):
    """Predefined agent roles"""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    FACT_CHECKER = "fact_checker"
    PLANNER = "planner"
    EXECUTOR = "executor"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class AgentTask:
    """Task assigned to an agent"""
    id: str
    description: str
    assigned_to: str  # Agent ID
    status: TaskStatus
    priority: int
    dependencies: List[str]
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class AgentMessage:
    """Message between agents"""
    id: str
    from_agent: str
    to_agent: str
    content: Any
    message_type: str  # request, response, broadcast, handoff
    timestamp: datetime = field(default_factory=datetime.now)
    requires_response: bool = False


class BaseAgent(ABC):
    """Base class for specialized agents"""
    
    def __init__(self, agent_id: str, role: AgentRole, name: str):
        self.id = agent_id
        self.role = role
        self.name = name
        self.capabilities: List[str] = []
        self.current_task: Optional[AgentTask] = None
        self._message_queue: List[AgentMessage] = []
    
    @abstractmethod
    async def execute(self, task: AgentTask) -> Dict[str, Any]:
        """Execute assigned task"""
        pass
    
    @abstractmethod
    async def can_handle(self, task_description: str) -> float:
        """Return confidence (0-1) that agent can handle task"""
        pass
    
    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive and optionally respond to a message"""
        self._message_queue.append(message)
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "id": self.id,
            "role": self.role.value,
            "name": self.name,
            "busy": self.current_task is not None,
            "pending_messages": len(self._message_queue)
        }


class ResearcherAgent(BaseAgent):
    """Agent specialized in information gathering"""
    
    def __init__(self, agent_id: str, search_engine=None):
        super().__init__(agent_id, AgentRole.RESEARCHER, "Researcher")
        self.capabilities = ["web_search", "academic_search", "document_analysis"]
        self._search_engine = search_engine
    
    async def execute(self, task: AgentTask) -> Dict[str, Any]:
        self.current_task = task
        
        query = task.input_data.get("query", task.description)
        results = []
        
        # Simulate research (integrate with actual search engine)
        if self._search_engine:
            results = await self._search_engine.search(query)
        else:
            results = [{"content": f"Research findings for: {query}", "source": "simulated"}]
        
        self.current_task = None
        
        return {
            "findings": results,
            "query": query,
            "sources_checked": len(results)
        }
    
    async def can_handle(self, task_description: str) -> float:
        keywords = ["research", "find", "search", "look up", "gather", "investigate"]
        desc_lower = task_description.lower()
        
        matches = sum(1 for k in keywords if k in desc_lower)
        return min(1.0, matches * 0.3)


class AnalystAgent(BaseAgent):
    """Agent specialized in analysis and synthesis"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentRole.ANALYST, "Analyst")
        self.capabilities = ["data_analysis", "pattern_recognition", "synthesis"]
    
    async def execute(self, task: AgentTask) -> Dict[str, Any]:
        self.current_task = task
        
        data = task.input_data.get("data", [])
        analysis = {
            "patterns": [],
            "key_findings": [],
            "confidence": 0.7
        }
        
        # Simple analysis
        if isinstance(data, list):
            analysis["item_count"] = len(data)
            analysis["key_findings"].append(f"Analyzed {len(data)} items")
        
        self.current_task = None
        return analysis
    
    async def can_handle(self, task_description: str) -> float:
        keywords = ["analyze", "analysis", "synthesize", "compare", "evaluate"]
        desc_lower = task_description.lower()
        
        matches = sum(1 for k in keywords if k in desc_lower)
        return min(1.0, matches * 0.3)


class CriticAgent(BaseAgent):
    """Agent specialized in verification and criticism"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, AgentRole.CRITIC, "Critic")
        self.capabilities = ["fact_checking", "error_detection", "quality_review"]
    
    async def execute(self, task: AgentTask) -> Dict[str, Any]:
        self.current_task = task
        
        content = task.input_data.get("content", "")
        
        critique = {
            "issues": [],
            "suggestions": [],
            "quality_score": 0.8,
            "approved": True
        }
        
        # Simple critique
        if len(content) < 50:
            critique["issues"].append("Content too brief")
            critique["quality_score"] -= 0.2
        
        if "definitely" in str(content).lower():
            critique["issues"].append("Overconfident language detected")
            critique["suggestions"].append("Use hedging language")
            critique["quality_score"] -= 0.1
        
        critique["approved"] = critique["quality_score"] >= 0.6
        
        self.current_task = None
        return critique
    
    async def can_handle(self, task_description: str) -> float:
        keywords = ["review", "verify", "check", "validate", "critique", "assess"]
        desc_lower = task_description.lower()
        
        matches = sum(1 for k in keywords if k in desc_lower)
        return min(1.0, matches * 0.3)


class AgentOrchestrator:
    """
    Multi-agent orchestrator (CrewAI/LangGraph style).
    
    Features:
    - Role-based agent assignment
    - Task dependency management
    - Inter-agent communication
    - Workflow execution
    - Cross-validation between agents
    """
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._tasks: Dict[str, AgentTask] = {}
        self._message_history: List[AgentMessage] = []
        self._workflow_results: Dict[str, Any] = {}
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent"""
        self._agents[agent.id] = agent
    
    def create_default_crew(self, search_engine=None) -> None:
        """Create default agent crew"""
        self.register_agent(ResearcherAgent("researcher_1", search_engine))
        self.register_agent(AnalystAgent("analyst_1"))
        self.register_agent(CriticAgent("critic_1"))
    
    async def assign_task(self, description: str, input_data: Dict,
                          priority: int = 5,
                          dependencies: List[str] = None) -> str:
        """
        Assign task to best-suited agent.
        
        Args:
            description: Task description
            input_data: Input data for task
            priority: Task priority (1-10)
            dependencies: IDs of tasks that must complete first
            
        Returns:
            Task ID
        """
        # Find best agent for task
        best_agent = None
        best_confidence = 0.0
        
        for agent in self._agents.values():
            confidence = await agent.can_handle(description)
            if confidence > best_confidence:
                best_confidence = confidence
                best_agent = agent
        
        if not best_agent:
            raise ValueError("No suitable agent found for task")
        
        # Create task
        task_id = str(uuid.uuid4())[:8]
        task = AgentTask(
            id=task_id,
            description=description,
            assigned_to=best_agent.id,
            status=TaskStatus.PENDING,
            priority=priority,
            dependencies=dependencies or [],
            input_data=input_data
        )
        
        self._tasks[task_id] = task
        
        return task_id
    
    async def execute_workflow(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a workflow of tasks.
        
        Args:
            tasks: List of task definitions [{"description": "", "input": {}, "depends_on": []}]
            
        Returns:
            Workflow results
        """
        task_ids = []
        
        # Create all tasks
        for task_def in tasks:
            task_id = await self.assign_task(
                description=task_def.get("description", ""),
                input_data=task_def.get("input", {}),
                priority=task_def.get("priority", 5),
                dependencies=task_def.get("depends_on", [])
            )
            task_ids.append(task_id)
        
        # Execute tasks respecting dependencies
        results = {}
        completed = set()
        
        while len(completed) < len(task_ids):
            # Find tasks ready to execute
            ready_tasks = [
                self._tasks[tid] for tid in task_ids
                if tid not in completed and
                all(dep in completed for dep in self._tasks[tid].dependencies)
            ]
            
            if not ready_tasks:
                # Deadlock or all done
                break
            
            # Execute ready tasks in parallel
            executions = []
            for task in ready_tasks:
                agent = self._agents[task.assigned_to]
                executions.append(self._execute_task(task, agent))
            
            task_results = await asyncio.gather(*executions)
            
            for task, result in zip(ready_tasks, task_results):
                results[task.id] = result
                completed.add(task.id)
                
                # Make output available to dependent tasks
                self._workflow_results[task.id] = result
        
        return {
            "tasks_completed": len(completed),
            "results": results
        }
    
    async def _execute_task(self, task: AgentTask, agent: BaseAgent) -> Dict[str, Any]:
        """Execute a single task"""
        task.status = TaskStatus.IN_PROGRESS
        
        try:
            # Inject results from dependencies
            for dep_id in task.dependencies:
                if dep_id in self._workflow_results:
                    task.input_data[f"from_{dep_id}"] = self._workflow_results[dep_id]
            
            result = await agent.execute(task)
            
            task.status = TaskStatus.COMPLETED
            task.output_data = result
            task.completed_at = datetime.now()
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            return {"error": str(e)}
    
    async def cross_validate(self, content: Any, validators: List[str] = None) -> Dict[str, Any]:
        """
        Cross-validate content using multiple agents.
        Reduces hallucinations through multi-agent verification.
        """
        validator_agents = []
        
        if validators:
            validator_agents = [self._agents[v] for v in validators if v in self._agents]
        else:
            # Use all critic/analyst agents
            validator_agents = [
                a for a in self._agents.values()
                if a.role in (AgentRole.CRITIC, AgentRole.ANALYST)
            ]
        
        if not validator_agents:
            return {"validated": True, "validators": 0}
        
        # Get validations from each agent
        validations = []
        for agent in validator_agents:
            task = AgentTask(
                id=f"validate_{uuid.uuid4().hex[:6]}",
                description="Validate content",
                assigned_to=agent.id,
                status=TaskStatus.PENDING,
                priority=8,
                dependencies=[],
                input_data={"content": content}
            )
            
            result = await agent.execute(task)
            validations.append({
                "agent": agent.id,
                "approved": result.get("approved", False),
                "score": result.get("quality_score", 0.5),
                "issues": result.get("issues", [])
            })
        
        # Aggregate results
        approved_count = sum(1 for v in validations if v["approved"])
        avg_score = sum(v["score"] for v in validations) / len(validations)
        all_issues = [issue for v in validations for issue in v["issues"]]
        
        return {
            "validated": approved_count >= len(validations) / 2,
            "approval_rate": approved_count / len(validations),
            "avg_score": avg_score,
            "validators": len(validations),
            "issues": list(set(all_issues))
        }
    
    async def send_message(self, from_agent: str, to_agent: str,
                           content: Any, message_type: str = "request") -> Optional[AgentMessage]:
        """Send message between agents"""
        message = AgentMessage(
            id=str(uuid.uuid4())[:8],
            from_agent=from_agent,
            to_agent=to_agent,
            content=content,
            message_type=message_type
        )
        
        self._message_history.append(message)
        
        if to_agent in self._agents:
            response = await self._agents[to_agent].receive_message(message)
            return response
        
        return None
    
    def get_crew_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            "agents": [a.get_status() for a in self._agents.values()],
            "active_tasks": sum(1 for t in self._tasks.values() if t.status == TaskStatus.IN_PROGRESS),
            "completed_tasks": sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED),
            "messages_exchanged": len(self._message_history)
        }
