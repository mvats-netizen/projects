"""
Knowledge Graph Builder and Manager

Constructs and manages course knowledge graphs from structured data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import networkx as nx
import pandas as pd
from loguru import logger

from .graph_schema import (
    Node, Edge, GraphMetadata, NodeType, EdgeType,
    ModuleNode, AssessmentNode, SkillNode
)


class CourseKnowledgeGraph:
    """
    Represents a course as a directed knowledge graph
    
    The graph contains:
    - Nodes: modules, assessments, skills
    - Edges: prerequisites, skill relationships, ordering
    """
    
    def __init__(self, metadata: GraphMetadata):
        """
        Initialize empty knowledge graph
        
        Args:
            metadata: Course metadata
        """
        self.metadata = metadata
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        
    def add_node(self, node: Node) -> None:
        """Add a node to the graph"""
        self.nodes[node.node_id] = node
        self.graph.add_node(
            node.node_id,
            node_type=node.node_type,
            name=node.name,
            difficulty=node.difficulty,
            **node.metadata
        )
        logger.debug(f"Added node: {node.node_id} ({node.node_type})")
        
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph"""
        if edge.source not in self.nodes:
            logger.warning(f"Source node {edge.source} not in graph")
            return
        if edge.target not in self.nodes:
            logger.warning(f"Target node {edge.target} not in graph")
            return
            
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source,
            edge.target,
            edge_type=edge.edge_type,
            weight=edge.weight,
            **edge.metadata
        )
        logger.debug(f"Added edge: {edge.source} -> {edge.target} ({edge.edge_type})")
        
    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by ID"""
        return self.nodes.get(node_id)
    
    def get_prerequisites(self, node_id: str) -> List[str]:
        """Get all prerequisite nodes for a given node"""
        return [
            pred for pred in self.graph.predecessors(node_id)
            if self.graph[pred][node_id].get("edge_type") == EdgeType.PREREQUISITE
        ]
    
    def get_successors(self, node_id: str) -> List[str]:
        """Get all successor nodes"""
        return list(self.graph.successors(node_id))
    
    def get_modules_in_order(self) -> List[str]:
        """Get modules in topological order"""
        module_ids = [
            nid for nid, node in self.nodes.items()
            if node.node_type == NodeType.MODULE
        ]
        # Sort by module_order if available, otherwise topological sort
        try:
            return [
                node_id for node_id in nx.topological_sort(self.graph)
                if node_id in module_ids
            ]
        except nx.NetworkXError:
            logger.warning("Graph contains cycles, using module_order fallback")
            return sorted(
                module_ids,
                key=lambda nid: getattr(self.nodes[nid], "module_order", 0)
            )
    
    def get_skills_for_module(self, module_id: str) -> List[str]:
        """Get all skills taught or required by a module"""
        return [
            succ for succ in self.graph.successors(module_id)
            if self.nodes[succ].node_type == NodeType.SKILL
        ]
    
    def compute_node_centrality(self) -> Dict[str, float]:
        """Compute betweenness centrality for each node"""
        return nx.betweenness_centrality(self.graph)
    
    def identify_bottlenecks(self, threshold: float = 0.7) -> List[str]:
        """
        Identify bottleneck nodes with high centrality and low success rate
        
        Args:
            threshold: Centrality threshold for bottleneck detection
            
        Returns:
            List of bottleneck node IDs
        """
        centrality = self.compute_node_centrality()
        bottlenecks = []
        
        for node_id, cent in centrality.items():
            node = self.nodes[node_id]
            if cent > threshold and node.completion_rate and node.completion_rate < 0.6:
                bottlenecks.append(node_id)
                
        return bottlenecks
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Compute various graph statistics"""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_modules": sum(1 for n in self.nodes.values() if n.node_type == NodeType.MODULE),
            "num_assessments": sum(1 for n in self.nodes.values() if n.node_type == NodeType.ASSESSMENT),
            "num_skills": sum(1 for n in self.nodes.values() if n.node_type == NodeType.SKILL),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
            "num_connected_components": nx.number_weakly_connected_components(self.graph),
            "density": nx.density(self.graph),
        }
    
    def export_to_json(self, filepath: Path) -> None:
        """Export graph to JSON format"""
        data = {
            "metadata": self.metadata.dict(),
            "nodes": [node.dict() for node in self.nodes.values()],
            "edges": [edge.dict() for edge in self.edges],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported graph to {filepath}")
        
    def export_to_networkx(self) -> nx.DiGraph:
        """Export as NetworkX graph"""
        return self.graph.copy()


class KnowledgeGraphBuilder:
    """
    Builds knowledge graphs from various data sources
    """
    
    def __init__(self):
        self.graph: Optional[CourseKnowledgeGraph] = None
        
    def build_from_course_data(
        self,
        course_structure_path: str,
        learner_data_path: Optional[str] = None
    ) -> CourseKnowledgeGraph:
        """
        Build knowledge graph from course structure JSON
        
        Args:
            course_structure_path: Path to course structure JSON
            learner_data_path: Optional path to learner trajectory data
            
        Returns:
            Constructed CourseKnowledgeGraph
        """
        logger.info(f"Building knowledge graph from {course_structure_path}")
        
        # Load course structure
        with open(course_structure_path, "r") as f:
            course_data = json.load(f)
        
        # Create metadata
        metadata = GraphMetadata(
            course_id=course_data["course_id"],
            course_name=course_data.get("course_name", "Unknown Course"),
            metadata=course_data.get("metadata", {})
        )
        
        self.graph = CourseKnowledgeGraph(metadata)
        
        # Add modules
        for module_data in course_data.get("modules", []):
            module = ModuleNode(
                node_id=module_data["module_id"],
                name=module_data["name"],
                difficulty=module_data.get("difficulty"),
                module_order=module_data.get("order"),
                estimated_hours=module_data.get("estimated_hours"),
                skills_taught=module_data.get("skills", []),
                prerequisite_skills=module_data.get("prerequisites", []),
                metadata=module_data.get("metadata", {})
            )
            self.graph.add_node(module)
        
        # Add assessments
        for assessment_data in course_data.get("assessments", []):
            assessment = AssessmentNode(
                node_id=assessment_data["assessment_id"],
                name=assessment_data["name"],
                difficulty=assessment_data.get("difficulty"),
                assessment_type=assessment_data.get("type", "quiz"),
                max_score=assessment_data.get("max_score", 100.0),
                passing_score=assessment_data.get("passing_score"),
                skills_assessed=assessment_data.get("skills", []),
                metadata=assessment_data.get("metadata", {})
            )
            self.graph.add_node(assessment)
        
        # Add skills
        for skill_data in course_data.get("skills", []):
            skill = SkillNode(
                node_id=skill_data["skill_id"],
                name=skill_data["name"],
                difficulty=skill_data.get("difficulty"),
                skill_category=skill_data.get("category"),
                is_prerequisite=skill_data.get("is_prerequisite", False),
                metadata=skill_data.get("metadata", {})
            )
            self.graph.add_node(skill)
        
        # Add edges
        for edge_data in course_data.get("edges", []):
            edge = Edge(
                source=edge_data["source"],
                target=edge_data["target"],
                edge_type=EdgeType(edge_data["type"]),
                weight=edge_data.get("weight", 1.0),
                metadata=edge_data.get("metadata", {})
            )
            self.graph.add_edge(edge)
        
        # Enrich with learner data if provided
        if learner_data_path:
            self._enrich_with_learner_data(learner_data_path)
        
        # Update metadata counts
        stats = self.graph.get_graph_statistics()
        self.graph.metadata.num_nodes = stats["num_nodes"]
        self.graph.metadata.num_edges = stats["num_edges"]
        
        logger.info(f"Built graph with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
        return self.graph
    
    def _enrich_with_learner_data(self, learner_data_path: str) -> None:
        """
        Enrich graph nodes and edges with learner statistics
        
        Args:
            learner_data_path: Path to learner trajectory CSV
        """
        logger.info(f"Enriching graph with learner data from {learner_data_path}")
        
        df = pd.read_csv(learner_data_path)
        
        # Compute node-level statistics
        node_stats = df.groupby("module_id").agg({
            "completed": "mean",
            "score": "mean",
            "time_spent": "mean",
            "dropped_out": "mean"
        }).to_dict("index")
        
        for node_id, stats in node_stats.items():
            if node_id in self.graph.nodes:
                node = self.graph.nodes[node_id]
                node.completion_rate = stats.get("completed", None)
                node.average_score = stats.get("score", None)
                node.average_time_spent = stats.get("time_spent", None)
                node.dropout_rate = stats.get("dropped_out", None)
        
        logger.info("Graph enriched with learner statistics")
    
    def build_from_dataframe(
        self,
        modules_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        course_id: str,
        course_name: str
    ) -> CourseKnowledgeGraph:
        """
        Build knowledge graph from pandas DataFrames
        
        Args:
            modules_df: DataFrame with columns [module_id, name, difficulty, ...]
            edges_df: DataFrame with columns [source, target, type, weight, ...]
            course_id: Unique course identifier
            course_name: Human-readable course name
            
        Returns:
            Constructed CourseKnowledgeGraph
        """
        logger.info(f"Building knowledge graph from DataFrames")
        
        metadata = GraphMetadata(
            course_id=course_id,
            course_name=course_name
        )
        
        self.graph = CourseKnowledgeGraph(metadata)
        
        # Add nodes from DataFrame
        for _, row in modules_df.iterrows():
            node = ModuleNode(
                node_id=row["module_id"],
                name=row["name"],
                difficulty=row.get("difficulty"),
                metadata=row.to_dict()
            )
            self.graph.add_node(node)
        
        # Add edges from DataFrame
        for _, row in edges_df.iterrows():
            edge = Edge(
                source=row["source"],
                target=row["target"],
                edge_type=EdgeType(row["type"]),
                weight=row.get("weight", 1.0)
            )
            self.graph.add_edge(edge)
        
        logger.info(f"Built graph from DataFrames")
        return self.graph


