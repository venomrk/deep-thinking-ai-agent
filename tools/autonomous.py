"""
Autonomous Tool-Use System (o3 Style)
Enables autonomous discovery and use of tools.
"""

from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import json


class ToolCategory(Enum):
    """Categories of tools"""
    SEARCH = "search"
    CODE = "code"
    IMAGE = "image"
    CALCULATION = "calculation"
    WEB = "web"
    FILE = "file"
    REASONING = "reasoning"


@dataclass
class ToolSpec:
    """Specification for a tool"""
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Dict[str, Any]]  # param_name -> {type, description, required}
    examples: List[Dict[str, Any]]
    cost_per_call: float = 0.0
    avg_latency_ms: int = 100
    
    def to_schema(self) -> Dict:
        """Convert to JSON schema"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": [k for k, v in self.parameters.items() if v.get("required")]
            }
        }


@dataclass
class ToolCall:
    """Record of a tool call"""
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    success: bool
    latency_ms: int
    timestamp: datetime = field(default_factory=datetime.now)


class AutonomousToolManager:
    """
    Autonomous tool-use system (o3-style).
    
    Features:
    - Dynamic tool discovery
    - Intelligent tool selection based on task
    - Tool chaining for complex tasks
    - Automatic error recovery
    - Cost-aware execution
    """
    
    def __init__(self, max_chain_length: int = 5, budget: float = 1.0):
        self._tools: Dict[str, ToolSpec] = {}
        self._executors: Dict[str, Callable] = {}
        self._call_history: List[ToolCall] = []
        self._max_chain_length = max_chain_length
        self._budget = budget
        self._spent = 0.0
    
    def register_tool(self, spec: ToolSpec, 
                      executor: Callable[..., Awaitable[Any]]) -> None:
        """Register a tool with its executor"""
        self._tools[spec.name] = spec
        self._executors[spec.name] = executor
    
    def register_defaults(self) -> None:
        """Register default tools"""
        # Python code execution tool
        python_spec = ToolSpec(
            name="run_python",
            description="Execute Python code for calculations or data processing",
            category=ToolCategory.CODE,
            parameters={
                "code": {"type": "string", "description": "Python code to execute", "required": True}
            },
            examples=[{"code": "result = 2 + 2", "output": "4"}],
            cost_per_call=0.001
        )
        self.register_tool(python_spec, self._execute_python)
        
        # Web search tool
        search_spec = ToolSpec(
            name="web_search",
            description="Search the web for information",
            category=ToolCategory.SEARCH,
            parameters={
                "query": {"type": "string", "description": "Search query", "required": True},
                "num_results": {"type": "integer", "description": "Number of results", "required": False}
            },
            examples=[{"query": "quantum computing", "output": "list of results"}],
            cost_per_call=0.01
        )
        self.register_tool(search_spec, self._execute_search)
        
        # Calculator tool
        calc_spec = ToolSpec(
            name="calculate",
            description="Perform mathematical calculations",
            category=ToolCategory.CALCULATION,
            parameters={
                "expression": {"type": "string", "description": "Math expression", "required": True}
            },
            examples=[{"expression": "sqrt(16) * 2", "output": "8.0"}],
            cost_per_call=0.0
        )
        self.register_tool(calc_spec, self._execute_calculate)
    
    async def _execute_python(self, code: str) -> Dict:
        """Execute Python code safely"""
        import math
        
        try:
            # Safe execution environment
            safe_globals = {
                "__builtins__": {},
                "math": math,
                "sum": sum,
                "len": len,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "range": range,
                "list": list,
                "dict": dict,
                "str": str,
                "int": int,
                "float": float
            }
            
            local_vars = {}
            exec(code, safe_globals, local_vars)
            
            return {
                "success": True,
                "result": local_vars.get("result", str(local_vars))
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_search(self, query: str, num_results: int = 5) -> Dict:
        """Execute web search"""
        # Placeholder - integrate with actual search engine
        return {
            "success": True,
            "results": [f"Result for: {query}"],
            "count": 1
        }
    
    async def _execute_calculate(self, expression: str) -> Dict:
        """Execute calculation"""
        import math
        
        try:
            allowed = {
                'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
                'tan': math.tan, 'log': math.log, 'exp': math.exp,
                'pi': math.pi, 'e': math.e, 'pow': pow, 'abs': abs
            }
            
            result = eval(expression, {"__builtins__": {}}, allowed)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def select_tool(self, task: str) -> Optional[ToolSpec]:
        """
        Autonomously select best tool for task.
        
        Args:
            task: Task description
            
        Returns:
            Best tool specification or None
        """
        task_lower = task.lower()
        
        # Score each tool
        scored = []
        
        for tool in self._tools.values():
            score = 0.0
            
            # Check description match
            desc_words = set(tool.description.lower().split())
            task_words = set(task_lower.split())
            overlap = len(desc_words & task_words)
            score += overlap * 0.2
            
            # Check category relevance
            category_keywords = {
                ToolCategory.SEARCH: ["search", "find", "look up", "research"],
                ToolCategory.CODE: ["code", "python", "program", "script"],
                ToolCategory.CALCULATION: ["calculate", "math", "compute", "sum"],
                ToolCategory.IMAGE: ["image", "picture", "visual", "photo"],
                ToolCategory.WEB: ["web", "url", "page", "website"]
            }
            
            for keyword in category_keywords.get(tool.category, []):
                if keyword in task_lower:
                    score += 0.3
            
            # Prefer cheaper tools
            score += (1 - min(1, tool.cost_per_call / 0.1)) * 0.1
            
            if score > 0:
                scored.append((score, tool))
        
        if not scored:
            return None
        
        # Return highest scoring tool
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]
    
    async def execute(self, tool_name: str, **params) -> Dict:
        """
        Execute a tool.
        
        Args:
            tool_name: Name of tool to execute
            **params: Tool parameters
            
        Returns:
            Tool execution result
        """
        if tool_name not in self._tools:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}
        
        tool = self._tools[tool_name]
        
        # Check budget
        if self._spent + tool.cost_per_call > self._budget:
            return {"success": False, "error": "Budget exceeded"}
        
        # Validate required parameters
        for param_name, param_spec in tool.parameters.items():
            if param_spec.get("required") and param_name not in params:
                return {"success": False, "error": f"Missing required parameter: {param_name}"}
        
        # Execute
        start_time = datetime.now()
        
        try:
            executor = self._executors[tool_name]
            result = await executor(**params)
            
            latency = int((datetime.now() - start_time).total_seconds() * 1000)
            success = result.get("success", True)
            
            # Record call
            call = ToolCall(
                tool_name=tool_name,
                parameters=params,
                result=result,
                success=success,
                latency_ms=latency
            )
            self._call_history.append(call)
            
            # Update spending
            if success:
                self._spent += tool.cost_per_call
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def chain_execute(self, tasks: List[Dict[str, Any]]) -> List[Dict]:
        """
        Execute a chain of tool calls.
        
        Args:
            tasks: List of {"task": "description"} or {"tool": "name", "params": {...}}
            
        Returns:
            List of results
        """
        results = []
        context = {}
        
        for i, task in enumerate(tasks[:self._max_chain_length]):
            if "tool" in task:
                # Direct tool call
                result = await self.execute(task["tool"], **task.get("params", {}))
            else:
                # Autonomous tool selection
                tool = await self.select_tool(task.get("task", ""))
                if tool:
                    # Inject context from previous results
                    params = task.get("params", {})
                    params["_context"] = context
                    result = await self.execute(tool.name, **params)
                else:
                    result = {"success": False, "error": "No suitable tool found"}
            
            results.append(result)
            
            # Update context for next task
            if result.get("success"):
                context[f"step_{i}"] = result.get("result")
            else:
                # Stop chain on failure
                break
        
        return results
    
    async def auto_recover(self, tool_name: str, params: Dict,
                           max_retries: int = 3) -> Dict:
        """
        Execute with automatic error recovery.
        
        Tries alternative strategies on failure.
        """
        for attempt in range(max_retries):
            result = await self.execute(tool_name, **params)
            
            if result.get("success"):
                return result
            
            # Try recovery strategies
            error = result.get("error", "")
            
            # Strategy 1: Simplify parameters
            if attempt == 0 and "code" in params:
                params["code"] = params["code"][:500]  # Truncate
                continue
            
            # Strategy 2: Try alternative tool
            if attempt == 1:
                alt_tool = await self._find_alternative(tool_name)
                if alt_tool:
                    result = await self.execute(alt_tool.name, **params)
                    if result.get("success"):
                        return result
            
            # Strategy 3: Wait and retry
            await asyncio.sleep(1)
        
        return result
    
    async def _find_alternative(self, tool_name: str) -> Optional[ToolSpec]:
        """Find alternative tool with same category"""
        if tool_name not in self._tools:
            return None
        
        original = self._tools[tool_name]
        
        for tool in self._tools.values():
            if tool.name != tool_name and tool.category == original.category:
                return tool
        
        return None
    
    def get_available_tools(self) -> List[Dict]:
        """Get list of available tools"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category.value,
                "cost": t.cost_per_call
            }
            for t in self._tools.values()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        if not self._call_history:
            return {"total_calls": 0}
        
        success_count = sum(1 for c in self._call_history if c.success)
        avg_latency = sum(c.latency_ms for c in self._call_history) / len(self._call_history)
        
        tool_counts = {}
        for call in self._call_history:
            tool_counts[call.tool_name] = tool_counts.get(call.tool_name, 0) + 1
        
        return {
            "total_calls": len(self._call_history),
            "success_rate": success_count / len(self._call_history),
            "avg_latency_ms": avg_latency,
            "total_spent": self._spent,
            "budget_remaining": self._budget - self._spent,
            "tool_usage": tool_counts
        }
