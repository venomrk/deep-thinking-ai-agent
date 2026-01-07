"""
FastAPI Server for Deep Research Agent
REST API endpoints for research operations.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

# Import agent modules
from ..core.orchestrator import Orchestrator
from ..core.base import OutputFormat


# Pydantic models for API
class ResearchRequest(BaseModel):
    query: str = Field(..., description="Research query")
    format: str = Field("report", description="Output format: report, summary, facts, trace, json")
    async_mode: bool = Field(False, description="Run asynchronously")


class ResearchResponse(BaseModel):
    task_id: str
    status: str
    content: Optional[str] = None
    summary: Optional[str] = None
    confidence: Optional[float] = None
    sources_count: int = 0
    claims_count: int = 0
    created_at: str


class SessionInfo(BaseModel):
    id: str
    query: str
    status: str
    progress: float
    current_step: str
    started_at: str


class StatsResponse(BaseModel):
    reasoning: Dict[str, Any]
    memory: Dict[str, Any]


# Create FastAPI app
app = FastAPI(
    title="Deep Research Agent API",
    description="Advanced AI-powered research and verification system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
_orchestrator: Optional[Orchestrator] = None
_active_tasks: Dict[str, asyncio.Task] = {}


def get_orchestrator() -> Orchestrator:
    """Get or create orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
        
        # Setup LLM
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                llm_client = genai.GenerativeModel("gemini-2.0-flash")
                _orchestrator.set_llm_client(llm_client)
            except Exception:
                pass
    
    return _orchestrator


@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    orchestrator = get_orchestrator()
    await orchestrator.initialize()


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    if _orchestrator:
        await _orchestrator.shutdown()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Deep Research Agent",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Execute a research query.
    
    - **query**: The research question
    - **format**: Output format (report, summary, facts, trace, json)
    - **async_mode**: If true, run in background and return task ID
    """
    orchestrator = get_orchestrator()
    
    # Map format
    format_map = {
        "report": OutputFormat.RESEARCH_REPORT,
        "summary": OutputFormat.EXECUTIVE_SUMMARY,
        "facts": OutputFormat.FACT_SHEET,
        "trace": OutputFormat.REASONING_TRACE,
        "json": OutputFormat.JSON
    }
    fmt = format_map.get(request.format, OutputFormat.RESEARCH_REPORT)
    
    if request.async_mode:
        # Run in background
        from uuid import uuid4
        task_id = str(uuid4())
        
        async def run_task():
            try:
                await orchestrator.research(request.query, fmt)
            except Exception as e:
                print(f"Background task error: {e}")
        
        task = asyncio.create_task(run_task())
        _active_tasks[task_id] = task
        
        return ResearchResponse(
            task_id=task_id,
            status="pending",
            created_at=datetime.now().isoformat(),
            sources_count=0,
            claims_count=0
        )
    
    # Run synchronously
    try:
        result = await orchestrator.research(request.query, fmt)
        
        return ResearchResponse(
            task_id=result.task_id,
            status="completed",
            content=result.content,
            summary=result.summary,
            confidence=result.confidence,
            sources_count=len(result.sources),
            claims_count=len(result.claims),
            created_at=result.created_at.isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/research/{task_id}", response_model=ResearchResponse)
async def get_research(task_id: str):
    """Get research result by task ID"""
    orchestrator = get_orchestrator()
    session = orchestrator.get_session(task_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return ResearchResponse(
        task_id=session.id,
        status=session.status.value,
        content=session.output.content if session.output else None,
        summary=session.output.summary if session.output else None,
        confidence=session.output.confidence if session.output else None,
        sources_count=len(session.output.sources) if session.output else 0,
        claims_count=len(session.output.claims) if session.output else 0,
        created_at=session.started_at.isoformat()
    )


@app.get("/sessions", response_model=List[SessionInfo])
async def list_sessions():
    """List active research sessions"""
    orchestrator = get_orchestrator()
    sessions = orchestrator.get_active_sessions()
    
    return [
        SessionInfo(
            id=s.id,
            query=s.query,
            status=s.status.value,
            progress=s.progress,
            current_step=s.current_step,
            started_at=s.started_at.isoformat()
        )
        for s in sessions
    ]


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    orchestrator = get_orchestrator()
    
    return StatsResponse(
        reasoning=orchestrator.get_reasoning_stats(),
        memory=orchestrator.get_memory_stats()
    )


@app.post("/clear-memory")
async def clear_memory():
    """Clear all memory systems"""
    orchestrator = get_orchestrator()
    orchestrator.clear_memory()
    return {"status": "cleared"}


@app.get("/stream/{task_id}")
async def stream_progress(task_id: str):
    """Stream research progress via SSE"""
    orchestrator = get_orchestrator()
    
    async def generate():
        while True:
            session = orchestrator.get_session(task_id)
            if not session:
                yield f"data: {json.dumps({'error': 'not found'})}\n\n"
                break
            
            data = {
                "progress": session.progress,
                "step": session.current_step,
                "status": session.status.value
            }
            yield f"data: {json.dumps(data)}\n\n"
            
            if session.status.value in ("completed", "failed"):
                break
            
            await asyncio.sleep(1)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
