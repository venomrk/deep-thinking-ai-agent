"""
Tree-of-Thoughts Reasoning Engine for Deep Research Agent
Implements advanced branching reasoning with dynamic depth and pruning.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import random

from ..core.base import (
    ReasoningNode, ReasoningTree, Evidence, BaseModule,
    ConfidenceLevel
)
from ..config.settings import get_settings, ReasoningConfig
from ..config.prompts import PromptTemplates


class NodeStatus(Enum):
    """Status of a reasoning node"""
    PENDING = "pending"
    EXPLORING = "exploring"
    EVALUATED = "evaluated"
    PRUNED = "pruned"
    TERMINAL = "terminal"


@dataclass
class ThoughtCandidate:
    """A candidate thought to be added to the tree"""
    thought: str
    assumptions: List[str]
    supporting_evidence: List[str]
    falsification_criteria: str
    confidence: float
    parent_id: str


@dataclass
class ExplorationResult:
    """Result of exploring a reasoning path"""
    path: List[ReasoningNode]
    final_confidence: float
    conclusion: Optional[str]
    evidence_used: List[Evidence]
    is_complete: bool


class TreeOfThoughts(BaseModule):
    """
    Advanced Tree-of-Thoughts reasoning implementation.
    Features:
    - Dynamic tree depth based on problem complexity
    - Probabilistic pruning with confidence thresholds
    - Parallel branch exploration
    - Hypothesis comparison and merging
    - Backtracking on low confidence
    """
    
    def __init__(self, config: ReasoningConfig = None, llm_client=None):
        super().__init__("tree_of_thoughts")
        self.config = config or ReasoningConfig()
        self._llm_client = llm_client
        
        self._current_tree: Optional[ReasoningTree] = None
        self._node_status: Dict[str, NodeStatus] = {}
        self._exploration_count = 0
        self._pruned_count = 0
    
    async def initialize(self) -> None:
        """Initialize the reasoning engine"""
        self._initialized = True
    
    async def shutdown(self) -> None:
        """Clean shutdown"""
        self._current_tree = None
        self._initialized = False
    
    def set_llm_client(self, client) -> None:
        """Set the LLM client for thought generation"""
        self._llm_client = client
    
    async def reason(self, problem: str, context: Dict[str, Any] = None,
                     evidence: List[Evidence] = None) -> ExplorationResult:
        """
        Main reasoning loop with tree exploration.
        
        Args:
            problem: The problem or question to reason about
            context: Additional context for reasoning
            evidence: Available evidence to consider
            
        Returns:
            ExplorationResult with the best reasoning path
        """
        # Initialize tree
        self._current_tree = ReasoningTree(problem=problem)
        self._node_status = {}
        self._exploration_count = 0
        self._pruned_count = 0
        
        # Create root node
        root = ReasoningNode(
            thought=f"Initial analysis of: {problem}",
            depth=0,
            confidence=0.5
        )
        self._current_tree.root_id = root.id
        self._current_tree.add_node(root)
        self._node_status[root.id] = NodeStatus.PENDING
        
        # Main exploration loop
        best_path = None
        best_confidence = 0.0
        
        for iteration in range(self.config.max_iterations):
            # Select node to explore
            node_to_explore = await self._select_node()
            
            if not node_to_explore:
                break  # No more nodes to explore
            
            # Check depth limit
            effective_depth = self._calculate_effective_depth(problem)
            if node_to_explore.depth >= effective_depth:
                node_to_explore.is_terminal = True
                self._node_status[node_to_explore.id] = NodeStatus.TERMINAL
                
                # Evaluate terminal node
                path = self._current_tree.get_path_to_node(node_to_explore.id)
                path_confidence = self._calculate_path_confidence(path)
                
                if path_confidence > best_confidence:
                    best_confidence = path_confidence
                    best_path = path
                
                continue
            
            # Generate and evaluate child thoughts
            self._node_status[node_to_explore.id] = NodeStatus.EXPLORING
            
            candidates = await self._generate_thoughts(
                node_to_explore, 
                problem, 
                context or {},
                evidence or []
            )
            
            # Add promising candidates as children
            for candidate in candidates:
                if await self._should_add_candidate(candidate):
                    child = self._create_node_from_candidate(candidate, node_to_explore)
                    self._current_tree.add_node(child)
                    self._node_status[child.id] = NodeStatus.PENDING
            
            self._node_status[node_to_explore.id] = NodeStatus.EVALUATED
            self._exploration_count += 1
            
            # Check termination conditions
            if best_confidence >= self.config.confidence_threshold:
                break
        
        # Find best path if not already found
        if not best_path:
            best_path = self._find_best_path()
            if best_path:
                best_confidence = self._calculate_path_confidence(best_path)
        
        # Generate conclusion
        conclusion = await self._synthesize_conclusion(best_path, problem) if best_path else None
        
        return ExplorationResult(
            path=best_path or [],
            final_confidence=best_confidence,
            conclusion=conclusion,
            evidence_used=evidence or [],
            is_complete=best_confidence >= self.config.confidence_threshold
        )
    
    async def _select_node(self) -> Optional[ReasoningNode]:
        """Select the next node to explore using UCB-like strategy"""
        pending_nodes = [
            self._current_tree.get_node(nid)
            for nid, status in self._node_status.items()
            if status == NodeStatus.PENDING
        ]
        
        if not pending_nodes:
            return None
        
        # Score nodes based on confidence and exploration value
        scored_nodes = []
        for node in pending_nodes:
            if node is None:
                continue
            
            # UCB-like score: exploitation (confidence) + exploration (depth bonus)
            exploitation = node.confidence
            exploration = 1.0 / (node.depth + 1)  # Prefer shallower nodes
            
            # Add randomness for diversity
            noise = random.uniform(0, 0.1)
            
            score = exploitation * 0.7 + exploration * 0.3 + noise
            scored_nodes.append((score, node))
        
        if not scored_nodes:
            return None
        
        # Select highest scoring node
        scored_nodes.sort(key=lambda x: -x[0])
        return scored_nodes[0][1]
    
    def _calculate_effective_depth(self, problem: str) -> int:
        """Calculate effective max depth based on problem complexity"""
        if not self.config.enable_dynamic_depth:
            return self.config.max_depth
        
        # Estimate complexity from problem length and structure
        word_count = len(problem.split())
        
        if word_count < 10:
            return max(2, self.config.max_depth - 2)
        elif word_count < 30:
            return self.config.max_depth
        else:
            return min(self.config.max_depth + 2, 10)
    
    async def _generate_thoughts(self, node: ReasoningNode, problem: str,
                                  context: Dict[str, Any],
                                  evidence: List[Evidence]) -> List[ThoughtCandidate]:
        """Generate candidate thoughts from current node"""
        if not self._llm_client:
            # Fallback: generate simple variants
            return self._generate_simple_thoughts(node, problem)
        
        # Build prompt
        current_path = self._current_tree.get_path_to_node(node.id)
        path_str = " -> ".join([n.thought[:50] for n in current_path])
        evidence_str = "\n".join([f"- {e.content[:100]}" for e in evidence[:5]])
        
        prompt = PromptTemplates.THOUGHT_GENERATOR.substitute(
            problem=problem,
            current_path=path_str,
            evidence=evidence_str or "No evidence yet",
            num_thoughts=self.config.branching_factor
        )
        
        try:
            response = await self._llm_client.generate_content_async(prompt)
            return self._parse_thought_response(response.text, node.id)
        except Exception as e:
            print(f"Thought generation error: {e}")
            return self._generate_simple_thoughts(node, problem)
    
    def _generate_simple_thoughts(self, node: ReasoningNode, 
                                   problem: str) -> List[ThoughtCandidate]:
        """Generate simple thought variants without LLM"""
        base_thought = node.thought
        candidates = []
        
        # Generate variants by different analysis angles
        angles = [
            f"Considering the factual aspects of {problem}",
            f"Analyzing the implications of {problem}",
            f"Exploring alternative perspectives on {problem}",
        ]
        
        for i, angle in enumerate(angles[:self.config.branching_factor]):
            candidates.append(ThoughtCandidate(
                thought=angle,
                assumptions=[f"Assumption based on {base_thought[:30]}"],
                supporting_evidence=[],
                falsification_criteria="Contradicting evidence found",
                confidence=0.4 + random.uniform(0, 0.2),
                parent_id=node.id
            ))
        
        return candidates
    
    def _parse_thought_response(self, response: str, 
                                 parent_id: str) -> List[ThoughtCandidate]:
        """Parse LLM response into thought candidates"""
        candidates = []
        
        try:
            # Try to extract JSON from response
            json_match = response.find('[')
            if json_match >= 0:
                json_end = response.rfind(']') + 1
                json_str = response[json_match:json_end]
                thoughts = json.loads(json_str)
                
                for t in thoughts:
                    candidates.append(ThoughtCandidate(
                        thought=t.get("thought", ""),
                        assumptions=t.get("assumptions", []),
                        supporting_evidence=t.get("supporting_evidence", []),
                        falsification_criteria=t.get("falsification_criteria", ""),
                        confidence=float(t.get("confidence", 0.5)),
                        parent_id=parent_id
                    ))
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: treat response as single thought
            candidates.append(ThoughtCandidate(
                thought=response[:500],
                assumptions=[],
                supporting_evidence=[],
                falsification_criteria="",
                confidence=0.5,
                parent_id=parent_id
            ))
        
        return candidates
    
    async def _should_add_candidate(self, candidate: ThoughtCandidate) -> bool:
        """Determine if a candidate should be added to the tree"""
        # Basic pruning threshold
        if candidate.confidence < self.config.pruning_threshold:
            self._pruned_count += 1
            return False
        
        # Check for duplicate thoughts
        for node in self._current_tree.nodes.values():
            if self._thoughts_similar(node.thought, candidate.thought):
                return False
        
        return True
    
    def _thoughts_similar(self, thought1: str, thought2: str, 
                          threshold: float = 0.8) -> bool:
        """Check if two thoughts are too similar"""
        # Simple word overlap check
        words1 = set(thought1.lower().split())
        words2 = set(thought2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap >= threshold
    
    def _create_node_from_candidate(self, candidate: ThoughtCandidate,
                                     parent: ReasoningNode) -> ReasoningNode:
        """Create a reasoning node from a thought candidate"""
        return ReasoningNode(
            thought=candidate.thought,
            parent_id=parent.id,
            depth=parent.depth + 1,
            confidence=candidate.confidence,
            assumptions=candidate.assumptions,
            falsification_criteria=candidate.falsification_criteria
        )
    
    def _calculate_path_confidence(self, path: List[ReasoningNode]) -> float:
        """Calculate overall confidence for a reasoning path"""
        if not path:
            return 0.0
        
        # Geometric mean of confidences (penalizes low-confidence steps)
        product = 1.0
        for node in path:
            product *= node.confidence
        
        return product ** (1 / len(path))
    
    def _find_best_path(self) -> Optional[List[ReasoningNode]]:
        """Find the best reasoning path in the tree"""
        leaves = self._current_tree.get_leaves()
        
        if not leaves:
            return None
        
        best_path = None
        best_confidence = 0.0
        
        for leaf in leaves:
            path = self._current_tree.get_path_to_node(leaf.id)
            confidence = self._calculate_path_confidence(path)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_path = path
        
        return best_path
    
    async def _synthesize_conclusion(self, path: List[ReasoningNode],
                                      problem: str) -> str:
        """Synthesize a conclusion from the reasoning path"""
        if not path:
            return "Unable to reach a conclusion."
        
        # Build conclusion from path
        steps = []
        for i, node in enumerate(path):
            steps.append(f"{i+1}. {node.thought}")
        
        conclusion = f"""
Based on the analysis of: {problem}

Reasoning steps:
{chr(10).join(steps)}

Final conclusion with confidence {path[-1].confidence:.2f}:
{path[-1].thought}
"""
        return conclusion.strip()
    
    async def backtrack(self, from_node_id: str) -> Optional[ReasoningNode]:
        """Backtrack from a node and try alternative paths"""
        node = self._current_tree.get_node(from_node_id)
        if not node or not node.parent_id:
            return None
        
        # Prune the current branch
        self._current_tree.prune_node(from_node_id)
        self._node_status[from_node_id] = NodeStatus.PRUNED
        self._pruned_count += 1
        
        # Return parent for re-exploration
        parent = self._current_tree.get_node(node.parent_id)
        if parent and self._node_status.get(parent.id) != NodeStatus.PRUNED:
            self._node_status[parent.id] = NodeStatus.PENDING
            return parent
        
        return None
    
    async def compare_hypotheses(self, node_ids: List[str],
                                  evidence: List[Evidence]) -> Dict[str, Any]:
        """Compare multiple hypotheses/branches"""
        results = []
        
        for node_id in node_ids:
            node = self._current_tree.get_node(node_id)
            if not node:
                continue
            
            path = self._current_tree.get_path_to_node(node_id)
            path_confidence = self._calculate_path_confidence(path)
            
            # Count supporting/contradicting evidence
            supporting = 0
            contradicting = 0
            
            for e in evidence:
                if e.supports_claim:
                    supporting += 1
                else:
                    contradicting += 1
            
            results.append({
                "node_id": node_id,
                "thought": node.thought,
                "path_confidence": path_confidence,
                "supporting_evidence": supporting,
                "contradicting_evidence": contradicting,
                "depth": node.depth
            })
        
        # Sort by confidence
        results.sort(key=lambda x: -x["path_confidence"])
        
        return {
            "hypotheses": results,
            "best_hypothesis": results[0]["node_id"] if results else None,
            "confidence_spread": max(r["path_confidence"] for r in results) - min(r["path_confidence"] for r in results) if len(results) > 1 else 0
        }
    
    def get_tree_stats(self) -> Dict[str, Any]:
        """Get statistics about the current reasoning tree"""
        if not self._current_tree:
            return {"status": "no_tree"}
        
        nodes = list(self._current_tree.nodes.values())
        
        return {
            "total_nodes": len(nodes),
            "max_depth": max(n.depth for n in nodes) if nodes else 0,
            "avg_confidence": sum(n.confidence for n in nodes) / len(nodes) if nodes else 0,
            "explorations": self._exploration_count,
            "pruned": self._pruned_count,
            "leaves": len(self._current_tree.get_leaves()),
            "status_counts": {
                status.value: len([n for n, s in self._node_status.items() if s == status])
                for status in NodeStatus
            }
        }
    
    def visualize_tree(self) -> str:
        """Generate a text visualization of the tree"""
        if not self._current_tree or not self._current_tree.root_id:
            return "No tree to visualize"
        
        lines = [f"Problem: {self._current_tree.problem[:50]}...", ""]
        
        def render_node(node_id: str, prefix: str = "", is_last: bool = True):
            node = self._current_tree.get_node(node_id)
            if not node:
                return
            
            connector = "└── " if is_last else "├── "
            status = self._node_status.get(node_id, NodeStatus.PENDING)
            status_icon = {
                NodeStatus.PENDING: "○",
                NodeStatus.EXPLORING: "◎",
                NodeStatus.EVALUATED: "●",
                NodeStatus.PRUNED: "✗",
                NodeStatus.TERMINAL: "◆"
            }.get(status, "?")
            
            lines.append(f"{prefix}{connector}{status_icon} [{node.confidence:.2f}] {node.thought[:40]}...")
            
            child_prefix = prefix + ("    " if is_last else "│   ")
            children = node.children_ids
            for i, child_id in enumerate(children):
                render_node(child_id, child_prefix, i == len(children) - 1)
        
        render_node(self._current_tree.root_id)
        return "\n".join(lines)
