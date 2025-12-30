"""
Bottleneck Detection

Identifies structural bottlenecks and high-risk nodes in the knowledge graph.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import networkx as nx
from loguru import logger

from ..graph.knowledge_graph import CourseKnowledgeGraph


class BottleneckDetector:
    """
    Detects bottleneck nodes and edges in the knowledge graph
    """
    
    def __init__(self, knowledge_graph: CourseKnowledgeGraph):
        """
        Initialize bottleneck detector
        
        Args:
            knowledge_graph: Course knowledge graph
        """
        self.knowledge_graph = knowledge_graph
    
    def detect_node_bottlenecks(
        self,
        centrality_threshold: float = 0.7,
        dropout_threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Detect node-level bottlenecks
        
        Bottlenecks are nodes with:
        - High centrality (many paths go through them)
        - High dropout rate or low completion rate
        
        Args:
            centrality_threshold: Minimum centrality to be considered
            dropout_threshold: Minimum dropout rate to be considered
            
        Returns:
            List of bottleneck node information
        """
        logger.info("Detecting node bottlenecks...")
        
        # Compute centrality
        centrality = self.knowledge_graph.compute_node_centrality()
        
        bottlenecks = []
        
        for node_id, cent in centrality.items():
            node = self.knowledge_graph.get_node(node_id)
            
            if not node:
                continue
            
            # Check if node is a bottleneck
            is_high_centrality = cent > centrality_threshold
            is_high_dropout = (node.dropout_rate or 0) > dropout_threshold
            is_low_completion = (node.completion_rate or 1) < (1 - dropout_threshold)
            
            if is_high_centrality and (is_high_dropout or is_low_completion):
                bottleneck_info = {
                    "node_id": node_id,
                    "name": node.name,
                    "node_type": node.node_type,
                    "centrality": cent,
                    "dropout_rate": node.dropout_rate,
                    "completion_rate": node.completion_rate,
                    "difficulty": node.difficulty,
                    "bottleneck_score": self._compute_bottleneck_score(cent, node),
                    "reason": self._get_bottleneck_reason(cent, node)
                }
                bottlenecks.append(bottleneck_info)
        
        # Sort by bottleneck score
        bottlenecks.sort(key=lambda x: x["bottleneck_score"], reverse=True)
        
        logger.info(f"Found {len(bottlenecks)} node bottlenecks")
        
        return bottlenecks
    
    def detect_edge_bottlenecks(
        self,
        transition_failure_threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Detect edge-level bottlenecks (difficult transitions)
        
        Args:
            transition_failure_threshold: Minimum failure rate for transitions
            
        Returns:
            List of bottleneck edge information
        """
        logger.info("Detecting edge bottlenecks...")
        
        bottleneck_edges = []
        
        for edge in self.knowledge_graph.edges:
            # Check if edge has high transition failure
            transition_failure = 1.0 - (edge.transition_success_rate or 0.5)
            dropout_after = edge.dropout_after_transition or 0.0
            
            if transition_failure > transition_failure_threshold or dropout_after > transition_failure_threshold:
                source_node = self.knowledge_graph.get_node(edge.source)
                target_node = self.knowledge_graph.get_node(edge.target)
                
                edge_info = {
                    "source": edge.source,
                    "target": edge.target,
                    "source_name": source_node.name if source_node else "Unknown",
                    "target_name": target_node.name if target_node else "Unknown",
                    "edge_type": edge.edge_type,
                    "transition_success_rate": edge.transition_success_rate,
                    "transition_failure_rate": transition_failure,
                    "dropout_after_transition": dropout_after,
                    "difficulty_gap": self._compute_difficulty_gap(source_node, target_node),
                    "bottleneck_score": transition_failure + dropout_after
                }
                bottleneck_edges.append(edge_info)
        
        # Sort by bottleneck score
        bottleneck_edges.sort(key=lambda x: x["bottleneck_score"], reverse=True)
        
        logger.info(f"Found {len(bottleneck_edges)} edge bottlenecks")
        
        return bottleneck_edges
    
    def identify_skill_gaps(self) -> List[Dict[str, Any]]:
        """
        Identify skills that act as prerequisites but have low mastery
        
        Returns:
            List of skill gap information
        """
        logger.info("Identifying skill gaps...")
        
        skill_gaps = []
        
        for node_id, node in self.knowledge_graph.nodes.items():
            if node.node_type != "skill":
                continue
            
            # Check if skill is a prerequisite for other nodes
            dependents = list(self.knowledge_graph.graph.successors(node_id))
            
            if not dependents:
                continue
            
            # Check if skill has low completion/high dropout
            is_gap = (
                (node.completion_rate and node.completion_rate < 0.6) or
                (node.dropout_rate and node.dropout_rate > 0.3)
            )
            
            if is_gap:
                # Compute impact (how many modules depend on this skill)
                impact_score = len(dependents)
                
                skill_info = {
                    "skill_id": node_id,
                    "skill_name": node.name,
                    "completion_rate": node.completion_rate,
                    "dropout_rate": node.dropout_rate,
                    "difficulty": node.difficulty,
                    "num_dependents": len(dependents),
                    "dependent_modules": dependents,
                    "impact_score": impact_score,
                    "severity": "high" if impact_score > 3 else "medium" if impact_score > 1 else "low"
                }
                skill_gaps.append(skill_info)
        
        # Sort by impact score
        skill_gaps.sort(key=lambda x: x["impact_score"], reverse=True)
        
        logger.info(f"Found {len(skill_gaps)} skill gaps")
        
        return skill_gaps
    
    def get_dropout_hotspots(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get top-k modules with highest dropout rates
        
        Args:
            top_k: Number of hotspots to return
            
        Returns:
            List of dropout hotspot information
        """
        hotspots = []
        
        for node_id, node in self.knowledge_graph.nodes.items():
            if node.dropout_rate is not None and node.dropout_rate > 0:
                hotspot_info = {
                    "node_id": node_id,
                    "name": node.name,
                    "node_type": node.node_type,
                    "dropout_rate": node.dropout_rate,
                    "completion_rate": node.completion_rate,
                    "difficulty": node.difficulty,
                    "average_score": node.average_score
                }
                hotspots.append(hotspot_info)
        
        # Sort by dropout rate
        hotspots.sort(key=lambda x: x["dropout_rate"], reverse=True)
        
        return hotspots[:top_k]
    
    def analyze_module_transitions(
        self,
        module_id: str
    ) -> Dict[str, Any]:
        """
        Analyze transitions into and out of a specific module
        
        Args:
            module_id: Module to analyze
            
        Returns:
            Analysis of module transitions
        """
        node = self.knowledge_graph.get_node(module_id)
        if not node:
            return {}
        
        # Get predecessors and successors
        predecessors = list(self.knowledge_graph.graph.predecessors(module_id))
        successors = list(self.knowledge_graph.graph.successors(module_id))
        
        # Analyze incoming transitions
        incoming_analysis = []
        for pred in predecessors:
            pred_node = self.knowledge_graph.get_node(pred)
            incoming_analysis.append({
                "from_module": pred,
                "from_name": pred_node.name if pred_node else "Unknown",
                "difficulty_increase": (node.difficulty or 0) - (pred_node.difficulty or 0) if pred_node else 0
            })
        
        # Analyze outgoing transitions
        outgoing_analysis = []
        for succ in successors:
            succ_node = self.knowledge_graph.get_node(succ)
            outgoing_analysis.append({
                "to_module": succ,
                "to_name": succ_node.name if succ_node else "Unknown",
                "difficulty_increase": (succ_node.difficulty or 0) - (node.difficulty or 0) if succ_node else 0
            })
        
        return {
            "module_id": module_id,
            "module_name": node.name,
            "num_incoming": len(predecessors),
            "num_outgoing": len(successors),
            "incoming_transitions": incoming_analysis,
            "outgoing_transitions": outgoing_analysis,
            "avg_incoming_difficulty_increase": np.mean([t["difficulty_increase"] for t in incoming_analysis]) if incoming_analysis else 0,
            "avg_outgoing_difficulty_increase": np.mean([t["difficulty_increase"] for t in outgoing_analysis]) if outgoing_analysis else 0
        }
    
    @staticmethod
    def _compute_bottleneck_score(centrality: float, node) -> float:
        """Compute overall bottleneck score"""
        dropout_component = node.dropout_rate or 0.5
        completion_component = 1.0 - (node.completion_rate or 0.5)
        difficulty_component = (node.difficulty or 5.0) / 10.0
        
        # Weighted combination
        score = (
            0.4 * centrality +
            0.3 * dropout_component +
            0.2 * completion_component +
            0.1 * difficulty_component
        )
        
        return score
    
    @staticmethod
    def _get_bottleneck_reason(centrality: float, node) -> str:
        """Generate human-readable bottleneck reason"""
        reasons = []
        
        if centrality > 0.7:
            reasons.append("high structural importance")
        
        if node.dropout_rate and node.dropout_rate > 0.4:
            reasons.append(f"high dropout rate ({node.dropout_rate:.1%})")
        
        if node.completion_rate and node.completion_rate < 0.6:
            reasons.append(f"low completion rate ({node.completion_rate:.1%})")
        
        if node.difficulty and node.difficulty > 7:
            reasons.append("high difficulty")
        
        return "; ".join(reasons) if reasons else "multiple factors"
    
    @staticmethod
    def _compute_difficulty_gap(source_node, target_node) -> float:
        """Compute difficulty gap between two nodes"""
        if not source_node or not target_node:
            return 0.0
        
        source_diff = source_node.difficulty or 5.0
        target_diff = target_node.difficulty or 5.0
        
        return target_diff - source_diff


