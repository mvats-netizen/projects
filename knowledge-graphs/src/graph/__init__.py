"""
Knowledge Graph Construction Module

Builds and manages course knowledge graphs with modules, skills, and prerequisites.
"""

from .knowledge_graph import KnowledgeGraphBuilder, CourseKnowledgeGraph
from .graph_schema import Node, Edge, GraphMetadata

__all__ = [
    "KnowledgeGraphBuilder",
    "CourseKnowledgeGraph",
    "Node",
    "Edge",
    "GraphMetadata",
]


