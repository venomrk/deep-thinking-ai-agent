"""
Prompt Templates for Deep Research Agent
All LLM prompts are centralized here for easy modification and versioning.
"""

from typing import Dict, List, Optional
from string import Template


class PromptTemplates:
    """Central repository for all LLM prompt templates"""
    
    # ==================== PLANNER PROMPTS ====================
    
    INTENT_PARSER = Template("""
You are an expert research assistant analyzing a user query to understand their intent.

USER QUERY: $query

Analyze this query and provide:
1. PRIMARY_INTENT: The main goal (research, factual_question, comparison, explanation, analysis)
2. COMPLEXITY: Rate 1-10 how complex this research task is
3. DOMAINS: List relevant knowledge domains
4. CONSTRAINTS: Any time, scope, or format constraints mentioned
5. AMBIGUITIES: Any unclear aspects that need clarification

Output as JSON:
{
    "primary_intent": "...",
    "complexity": N,
    "domains": ["...", "..."],
    "constraints": ["..."],
    "ambiguities": ["..."],
    "requires_clarification": true/false
}
""")

    SUB_QUESTION_GENERATOR = Template("""
You are a research planner breaking down a complex query into sub-questions.

MAIN QUERY: $query
INTENT: $intent
COMPLEXITY: $complexity

Generate a list of sub-questions that, when answered, will fully address the main query.
Order them by dependency (foundational questions first).

For each sub-question specify:
- The question itself
- Why it's needed
- What type of search is best (web, academic, calculation, local)
- Estimated importance (1-10)

Output as JSON array:
[
    {
        "id": "q1",
        "question": "...",
        "rationale": "...",
        "search_type": "web|academic|calculation|local",
        "importance": N,
        "depends_on": ["q0"] or []
    }
]
""")

    SEARCH_STRATEGY = Template("""
Given the research task and available information, decide the optimal search strategy.

TASK: $task
CURRENT_KNOWLEDGE: $current_knowledge
REMAINING_BUDGET: $budget

Decide:
1. Should we go DEEPER (more detailed queries on current topic) or BROADER (explore related areas)?
2. Which search engines to prioritize?
3. How many queries to run?
4. Any query refinements needed?

Output as JSON:
{
    "strategy": "deep|broad|targeted",
    "priority_engines": ["..."],
    "num_queries": N,
    "query_refinements": ["..."],
    "rationale": "..."
}
""")

    # ==================== REASONING PROMPTS ====================
    
    THOUGHT_GENERATOR = Template("""
You are reasoning about a problem using Tree-of-Thoughts methodology.

PROBLEM: $problem
CURRENT_PATH: $current_path
AVAILABLE_EVIDENCE: $evidence

Generate $num_thoughts distinct thoughts/hypotheses that could advance our understanding.
Each thought should:
- Build on the current reasoning path
- Be grounded in available evidence
- Represent a meaningfully different direction

For each thought provide:
- The thought itself
- Key assumptions made
- What evidence supports it
- What would falsify it
- Confidence score (0-1)

Output as JSON array:
[
    {
        "thought": "...",
        "assumptions": ["..."],
        "supporting_evidence": ["..."],
        "falsification_criteria": "...",
        "confidence": 0.X
    }
]
""")

    THOUGHT_EVALUATOR = Template("""
Evaluate this reasoning thought for quality and promise.

THOUGHT: $thought
CONTEXT: $context
EVIDENCE: $evidence

Score on these dimensions (0-1 each):
1. LOGICAL_VALIDITY: Is the reasoning sound?
2. EVIDENCE_SUPPORT: How well does evidence support this?
3. NOVELTY: Does this add new insight?
4. ACTIONABILITY: Can we verify or build on this?
5. RISK: How likely is this a dead end?

Overall recommendation: PURSUE, DEFER, or PRUNE

Output as JSON:
{
    "logical_validity": 0.X,
    "evidence_support": 0.X,
    "novelty": 0.X,
    "actionability": 0.X,
    "risk": 0.X,
    "overall_score": 0.X,
    "recommendation": "PURSUE|DEFER|PRUNE",
    "rationale": "..."
}
""")

    HYPOTHESIS_COMPARATOR = Template("""
Compare these competing hypotheses and determine which is best supported.

HYPOTHESES:
$hypotheses

AVAILABLE EVIDENCE:
$evidence

For each hypothesis:
1. Count supporting evidence
2. Count contradicting evidence
3. Identify gaps in evidence
4. Rate overall plausibility

Determine:
- Which hypothesis is best supported
- What additional evidence would help distinguish them
- Should any be merged or refined?

Output as JSON:
{
    "hypothesis_scores": [
        {"id": "...", "support": N, "contradict": N, "gaps": N, "plausibility": 0.X}
    ],
    "best_hypothesis": "...",
    "needed_evidence": ["..."],
    "merge_suggestions": ["..."]
}
""")

    # ==================== VERIFICATION PROMPTS ====================
    
    FACT_CHECKER = Template("""
Verify this claim against the provided sources.

CLAIM: $claim
SOURCES: $sources

For each source, determine:
1. Does it support, contradict, or not address the claim?
2. How reliable is this source?
3. Is the information current?

Overall verdict:
- VERIFIED: Multiple reliable sources confirm
- LIKELY_TRUE: Some support, no contradiction
- UNCERTAIN: Mixed or insufficient evidence
- LIKELY_FALSE: Evidence contradicts
- FALSE: Clear contradiction from reliable sources

Output as JSON:
{
    "source_analysis": [
        {
            "source_id": "...",
            "stance": "support|contradict|neutral",
            "reliability": 0.X,
            "currency": "current|dated|unknown",
            "key_quote": "..."
        }
    ],
    "verdict": "VERIFIED|LIKELY_TRUE|UNCERTAIN|LIKELY_FALSE|FALSE",
    "confidence": 0.X,
    "explanation": "..."
}
""")

    HALLUCINATION_DETECTOR = Template("""
Analyze this generated content for potential hallucinations.

CONTENT: $content
KNOWN_SOURCES: $sources
KNOWN_FACTS: $facts

Check for:
1. Claims not supported by any source
2. Specific numbers/dates/names that aren't sourced
3. Logical inconsistencies
4. Overconfident statements without evidence
5. Made-up citations or references

Output as JSON:
{
    "potential_hallucinations": [
        {
            "text": "...",
            "issue": "unsourced|fabricated|inconsistent|overconfident",
            "severity": "high|medium|low",
            "suggested_fix": "..."
        }
    ],
    "overall_hallucination_risk": 0.X,
    "clean_content": "..."
}
""")

    CONTRADICTION_DETECTOR = Template("""
Analyze these facts for contradictions.

FACTS:
$facts

Identify:
1. Direct contradictions (A says X, B says not-X)
2. Numerical conflicts (different values for same thing)
3. Temporal conflicts (incompatible timelines)
4. Scope conflicts (claims true in different contexts)

For each conflict suggest resolution:
- Which source is more reliable?
- Is it a genuine contradiction or context-dependent?
- How should we present this in output?

Output as JSON:
{
    "contradictions": [
        {
            "fact_a": "...",
            "fact_b": "...",
            "type": "direct|numerical|temporal|scope",
            "resolution": "...",
            "preferred_fact": "a|b|neither|both_with_context"
        }
    ],
    "consistency_score": 0.X
}
""")

    # ==================== SYNTHESIS PROMPTS ====================
    
    SYNTHESIZER = Template("""
Synthesize these verified facts into a coherent response.

QUERY: $query
VERIFIED_FACTS: $facts
REASONING_TRACE: $reasoning
OUTPUT_FORMAT: $format

Requirements:
1. Only include verified information
2. Cite sources for each claim
3. Acknowledge uncertainty where it exists
4. Resolve contradictions transparently
5. Match the requested output format

Output format options:
- research_report: Academic style with sections
- executive_summary: Brief, action-oriented
- fact_sheet: Bullet points with citations
- reasoning_trace: Step-by-step logic
- json: Machine-readable with citations

Generate the synthesis now.
""")

    UNCERTAINTY_ARTICULATOR = Template("""
Express appropriate uncertainty for these conclusions.

CONCLUSIONS: $conclusions
EVIDENCE_STRENGTH: $evidence_strength
CONTRADICTIONS: $contradictions

For each conclusion:
1. Rate certainty level (definite, probable, possible, speculative)
2. Articulate what we know vs. what we infer
3. State what additional evidence would increase certainty
4. Provide appropriate hedging language

Output as JSON:
{
    "articulated_conclusions": [
        {
            "conclusion": "...",
            "certainty": "definite|probable|possible|speculative",
            "known_facts": ["..."],
            "inferences": ["..."],
            "needed_evidence": ["..."],
            "hedged_statement": "..."
        }
    ]
}
""")

    # ==================== SELF-IMPROVEMENT PROMPTS ====================
    
    SELF_EVALUATOR = Template("""
Evaluate the quality of this research task completion.

ORIGINAL_QUERY: $query
FINAL_OUTPUT: $output
REASONING_TRACE: $trace
SOURCES_USED: $sources
TIME_TAKEN: $time
COST: $cost

Rate on these dimensions (1-10):
1. COMPLETENESS: All aspects of query addressed?
2. ACCURACY: Claims well-supported by evidence?
3. DEPTH: Sufficient analysis and insight?
4. CLARITY: Easy to understand?
5. EFFICIENCY: Good use of resources?

Identify:
- What went well
- What could be improved
- Specific failure points
- Recommendations for future similar tasks

Output as JSON:
{
    "scores": {
        "completeness": N,
        "accuracy": N,
        "depth": N,
        "clarity": N,
        "efficiency": N,
        "overall": N
    },
    "successes": ["..."],
    "improvements": ["..."],
    "failures": ["..."],
    "recommendations": ["..."]
}
""")


def get_prompt(template_name: str, **kwargs) -> str:
    """Get a formatted prompt by name"""
    template = getattr(PromptTemplates, template_name, None)
    if template is None:
        raise ValueError(f"Unknown prompt template: {template_name}")
    return template.substitute(**kwargs)
