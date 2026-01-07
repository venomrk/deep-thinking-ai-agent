"""
Tool Manager for Deep Research Agent
Handles tool registration, selection, and execution with error handling.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum
import json

from ..core.base import ToolResult, ToolIntent, BaseModule
from ..memory.failures import FailureMemory


class ToolCategory(Enum):
    """Categories of tools"""
    SEARCH = "search"
    CALCULATION = "calculation"
    WEB = "web"
    FILE = "file"
    CODE = "code"
    ANALYSIS = "analysis"


@dataclass
class ToolStats:
    """Statistics for a tool"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_time: float = 0.0
    total_cost: float = 0.0
    avg_time: float = 0.0
    success_rate: float = 1.0
    
    def record_call(self, success: bool, time: float, cost: float = 0.0):
        self.total_calls += 1
        self.total_time += time
        self.total_cost += cost
        
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        
        self.avg_time = self.total_time / self.total_calls
        self.success_rate = self.successful_calls / self.total_calls


class BaseTool(ABC):
    """Abstract base class for tools"""
    
    def __init__(self, name: str, description: str, category: ToolCategory):
        self.name = name
        self.description = description
        self.category = category
        self.cost_per_call: float = 0.0
        self.timeout: int = 30
        self.retry_count: int = 3
    
    @abstractmethod
    async def execute(self, **params) -> ToolResult:
        """Execute the tool with given parameters"""
        pass
    
    @abstractmethod
    def validate_params(self, **params) -> bool:
        """Validate input parameters"""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool parameters"""
        return {"properties": {}, "required": []}


class WebFetchTool(BaseTool):
    """Tool for fetching web content"""
    
    def __init__(self):
        super().__init__(
            name="web_fetch",
            description="Fetch and extract content from a URL",
            category=ToolCategory.WEB
        )
        self.timeout = 30
    
    def validate_params(self, **params) -> bool:
        return "url" in params and params["url"].startswith("http")
    
    async def execute(self, **params) -> ToolResult:
        import time
        start = time.time()
        
        try:
            import httpx
            
            url = params["url"]
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=self.timeout, follow_redirects=True)
                
                if response.status_code != 200:
                    return ToolResult(
                        success=False,
                        error=f"HTTP {response.status_code}",
                        execution_time=time.time() - start
                    )
                
                # Extract text content
                content = response.text
                
                # Simple HTML to text conversion
                import re
                # Remove scripts and styles
                content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
                content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
                # Remove HTML tags
                content = re.sub(r'<[^>]+>', ' ', content)
                # Clean whitespace
                content = ' '.join(content.split())
                
                return ToolResult(
                    success=True,
                    output=content[:10000],  # Limit content size
                    execution_time=time.time() - start,
                    metadata={"url": url, "length": len(content)}
                )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"}
            },
            "required": ["url"]
        }


class CalculationTool(BaseTool):
    """Tool for mathematical calculations"""
    
    def __init__(self):
        super().__init__(
            name="calculate",
            description="Perform mathematical calculations",
            category=ToolCategory.CALCULATION
        )
    
    def validate_params(self, **params) -> bool:
        return "expression" in params
    
    async def execute(self, **params) -> ToolResult:
        import time
        start = time.time()
        
        try:
            expression = params["expression"]
            
            # Safe evaluation (basic math only)
            import math
            allowed = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow, 'sqrt': math.sqrt,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'log': math.log, 'log10': math.log10, 'exp': math.exp,
                'pi': math.pi, 'e': math.e
            }
            
            result = eval(expression, {"__builtins__": {}}, allowed)
            
            return ToolResult(
                success=True,
                output=result,
                execution_time=time.time() - start,
                metadata={"expression": expression}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            "required": ["expression"]
        }


class FileReadTool(BaseTool):
    """Tool for reading local files"""
    
    def __init__(self, allowed_paths: List[str] = None):
        super().__init__(
            name="file_read",
            description="Read content from a local file",
            category=ToolCategory.FILE
        )
        self.allowed_paths = allowed_paths or ["."]
    
    def validate_params(self, **params) -> bool:
        return "path" in params
    
    def _is_path_allowed(self, path: str) -> bool:
        from pathlib import Path
        target = Path(path).resolve()
        
        for allowed in self.allowed_paths:
            allowed_path = Path(allowed).resolve()
            try:
                target.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        return False
    
    async def execute(self, **params) -> ToolResult:
        import time
        start = time.time()
        
        try:
            from pathlib import Path
            
            path = params["path"]
            
            if not self._is_path_allowed(path):
                return ToolResult(
                    success=False,
                    error="Path not allowed",
                    execution_time=time.time() - start
                )
            
            file_path = Path(path)
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    error="File not found",
                    execution_time=time.time() - start
                )
            
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            return ToolResult(
                success=True,
                output=content[:50000],  # Limit size
                execution_time=time.time() - start,
                metadata={"path": str(file_path.absolute()), "size": len(content)}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "properties": {
                "path": {"type": "string", "description": "Path to file to read"}
            },
            "required": ["path"]
        }


class ToolManager(BaseModule):
    """
    Manages tool registration, selection, and execution.
    Features:
    - Tool registry with categories
    - Cost-aware tool selection
    - Error handling with retries
    - Fallback chains
    - Usage statistics
    """
    
    def __init__(self, failure_memory: FailureMemory = None):
        super().__init__("tool_manager")
        self._registry: Dict[str, BaseTool] = {}
        self._stats: Dict[str, ToolStats] = {}
        self._fallback_chains: Dict[str, List[str]] = {}
        self._failure_memory = failure_memory
    
    async def initialize(self) -> None:
        """Initialize with default tools"""
        # Register default tools
        self.register(WebFetchTool())
        self.register(CalculationTool())
        self.register(FileReadTool())
        
        self._initialized = True
    
    async def shutdown(self) -> None:
        """Clean shutdown"""
        self._registry.clear()
        self._initialized = False
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool"""
        self._registry[tool.name] = tool
        self._stats[tool.name] = ToolStats()
    
    def unregister(self, tool_name: str) -> bool:
        """Unregister a tool"""
        if tool_name in self._registry:
            del self._registry[tool_name]
            del self._stats[tool_name]
            return True
        return False
    
    def set_fallback_chain(self, tool_name: str, fallbacks: List[str]) -> None:
        """Set fallback tools for a primary tool"""
        self._fallback_chains[tool_name] = fallbacks
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self._registry.get(name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Get all tools in a category"""
        return [t for t in self._registry.values() if t.category == category]
    
    async def select_tool(self, intent: ToolIntent) -> Optional[BaseTool]:
        """Select the best tool for an intent"""
        candidates = []
        
        for tool in self._registry.values():
            # Score based on intent match
            score = 0.0
            
            # Category match
            if intent.action.lower() in tool.category.value:
                score += 0.4
            
            # Name match
            if intent.action.lower() in tool.name.lower():
                score += 0.3
            
            # Description match
            if any(word in tool.description.lower() for word in intent.action.lower().split()):
                score += 0.2
            
            # Success rate bonus
            stats = self._stats.get(tool.name)
            if stats and stats.total_calls > 0:
                score += stats.success_rate * 0.1
            
            if score > 0:
                candidates.append((score, tool))
        
        if not candidates:
            return None
        
        # Sort by score
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]
    
    async def execute(self, tool_name: str, **params) -> ToolResult:
        """Execute a tool with parameters and error handling"""
        tool = self._registry.get(tool_name)
        if not tool:
            return ToolResult(success=False, error=f"Tool '{tool_name}' not found")
        
        # Validate parameters
        if not tool.validate_params(**params):
            return ToolResult(success=False, error="Invalid parameters")
        
        # Check if we should avoid this tool
        if self._failure_memory and self._failure_memory.should_avoid(
            "tool_failure", tool_name, threshold=5
        ):
            # Try fallback
            fallbacks = self._fallback_chains.get(tool_name, [])
            for fallback_name in fallbacks:
                if not self._failure_memory.should_avoid("tool_failure", fallback_name):
                    return await self.execute(fallback_name, **params)
            
            return ToolResult(success=False, error="Tool avoided due to repeated failures")
        
        # Execute with retries
        last_error = None
        for attempt in range(tool.retry_count):
            try:
                result = await asyncio.wait_for(
                    tool.execute(**params),
                    timeout=tool.timeout
                )
                
                # Record stats
                self._stats[tool_name].record_call(
                    success=result.success,
                    time=result.execution_time,
                    cost=result.cost
                )
                
                if result.success:
                    return result
                else:
                    last_error = result.error
                    
                    # Record failure
                    if self._failure_memory:
                        self._failure_memory.record_tool_failure(
                            tool_name, params, result.error
                        )
                    
            except asyncio.TimeoutError:
                last_error = "Timeout"
                self._stats[tool_name].record_call(success=False, time=tool.timeout)
            except Exception as e:
                last_error = str(e)
                self._stats[tool_name].record_call(success=False, time=0)
            
            # Wait before retry
            if attempt < tool.retry_count - 1:
                await asyncio.sleep(1 * (attempt + 1))
        
        # Try fallback chain
        fallbacks = self._fallback_chains.get(tool_name, [])
        for fallback_name in fallbacks:
            fallback_result = await self.execute(fallback_name, **params)
            if fallback_result.success:
                return fallback_result
        
        return ToolResult(success=False, error=f"All attempts failed: {last_error}")
    
    def get_tool_stats(self, tool_name: str = None) -> Dict[str, Any]:
        """Get statistics for tools"""
        if tool_name:
            stats = self._stats.get(tool_name)
            if stats:
                return {
                    "name": tool_name,
                    "total_calls": stats.total_calls,
                    "success_rate": stats.success_rate,
                    "avg_time": stats.avg_time,
                    "total_cost": stats.total_cost
                }
            return {}
        
        return {
            name: {
                "total_calls": s.total_calls,
                "success_rate": s.success_rate,
                "avg_time": s.avg_time
            }
            for name, s in self._stats.items()
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category.value,
                "schema": t.get_schema()
            }
            for t in self._registry.values()
        ]
