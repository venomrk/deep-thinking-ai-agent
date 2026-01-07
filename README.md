# 🔬 Deep Thinking AI Agent

**State-of-the-art AI-powered Deep Search & Research System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Author:** Rakshith Kumar  
> **Email:** rakshith098765@gmail.com

A benchmark-driven agentic AI system that thinks deeper, searches broader, and verifies harder than existing research agents.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **16-Layer Hidden Reasoning** | Internal reasoning stack with self-verification |
| **Multi-Agent Orchestration** | Specialized agents (Researcher, Analyst, Critic) collaborate |
| **DeepSeek R1-Style Reflection** | Self-correction and assumption checking |
| **Tree-of-Thoughts Reasoning** | Dynamic depth, probabilistic pruning, backtracking |
| **Multi-Source Verification** | Cross-source fact-checking, hallucination detection |
| **Episodic Memory** | Learns from past research sessions |
| **Autonomous Tool-Use** | o3-style dynamic tool selection and chaining |
| **Precision Output** | Concise by default, detailed when requested |

## 🚀 Quick Start

```bash
# Install
pip install -r deep_research_agent/requirements.txt

# Set API key (optional, for LLM reasoning)
export GEMINI_API_KEY=your_key_here

# Run interactive mode
python deep_research_agent/main.py --interactive

# Or single query
python deep_research_agent/main.py "What is quantum computing?" -f summary
```

## 📁 Architecture

```
deep_research_agent/
├── config/          # Settings & prompts
├── core/            # Orchestrator, Planner, Pipeline, Multi-Agent
├── search/          # Multi-source search engine
├── reasoning/       # Tree-of-Thoughts, Reflection
├── tools/           # Tool manager, Autonomous tool-use
├── memory/          # Short-term, Long-term, Episodic, Citations
├── verification/    # Fact-checker, Hallucination detector
├── synthesis/       # Output generator
├── evaluation/      # Self-evaluation
├── api/             # FastAPI server
└── web/             # Dashboard UI
```

## 🧠 16-Layer Reasoning Pipeline

Internal reasoning (hidden from output):

1. Intent Interpretation → 2. Query Decomposition → 3. Search Breadth → 4. Search Depth → 5. Hypothesis Generation → 6. Hypothesis Pruning → 7. Tool Selection → 8. Retrieval Verification → 9. Cross-Source Comparison → 10. Contradiction Detection → 11. Evidence Scoring → 12. Confidence Scoring → 13. Critic Review → 14. Redundancy Elimination → 15. Precision Compression → 16. Final Veto/Approval

## 🔧 API Usage

```bash
# Start server
python -m deep_research_agent.api.server

# POST /research
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest AI trends?", "format": "summary"}'
```

## 📊 Output Formats

- `report` — Full research report with citations
- `summary` — Executive summary (default: concise)
- `facts` — Bullet-point fact sheet
- `trace` — Reasoning trace visualization
- `json` — Structured JSON output

## 🛡️ Verification System

- **Multi-source fact-checking** — No single-source trust
- **Hallucination detection** — Flags unsourced claims
- **Contradiction detection** — Identifies conflicting sources
- **Confidence scoring** — Transparent uncertainty

## 🤖 Multi-Agent System

```python
from deep_research_agent.core.multi_agent import AgentOrchestrator

orchestrator = AgentOrchestrator()
orchestrator.create_default_crew()

# Execute workflow
results = await orchestrator.execute_workflow([
    {"description": "Research quantum computing"},
    {"description": "Analyze findings", "depends_on": ["task_0"]},
    {"description": "Review and validate", "depends_on": ["task_1"]}
])
```

## 📈 Self-Improvement

- Tracks accuracy, precision, redundancy per response
- Learns from failure patterns
- Episodic memory for cross-session learning

## 📋 Requirements

- Python 3.10+
- `google-generativeai` (optional, for Gemini LLM)
- `fastapi`, `uvicorn` (for API server)
- `httpx`, `duckduckgo-search` (for search)

## 📜 License

MIT License

---

**Built with 16 layers of reasoning** 🧠
