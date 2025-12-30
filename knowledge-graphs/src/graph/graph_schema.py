"""
Graph Schema Definitions

Defines data models for nodes, edges, and metadata in the knowledge graph.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph"""
    MODULE = "module"
    ASSESSMENT = "assessment"
    SKILL = "skill"
    CONCEPT = "concept"


class EdgeType(str, Enum):
    """Types of edges in the knowledge graph"""
    PREREQUISITE = "prerequisite"
    TEACHES = "teaches"
    REQUIRES = "requires"
    FOLLOWS = "follows"
    ASSESSES = "assesses"


class Node(BaseModel):
    """
    Represents a node in the knowledge graph
    
    Attributes:
        node_id: Unique identifier
        node_type: Type of node (module, skill, etc.)
        name: Human-readable name
        difficulty: Estimated difficulty (1-10 scale)
        metadata: Additional attributes
    """
    node_id: str
    node_type: NodeType
    name: str
    difficulty: Optional[float] = Field(default=None, ge=1.0, le=10.0)
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Statistical metrics (computed from learner data)
    completion_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    average_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    average_time_spent: Optional[float] = None  # in seconds
    dropout_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    class Config:
        use_enum_values = True


class Edge(BaseModel):
    """
    Represents an edge (relationship) in the knowledge graph
    
    Attributes:
        source: Source node ID
        target: Target node ID
        edge_type: Type of relationship
        weight: Edge weight (e.g., strength of prerequisite)
        metadata: Additional attributes
    """
    source: str
    target: str
    edge_type: EdgeType
    weight: float = Field(default=1.0, ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Transition metrics (computed from learner data)
    transition_success_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    average_transition_time: Optional[float] = None  # days between modules
    dropout_after_transition: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    class Config:
        use_enum_values = True


class GraphMetadata(BaseModel):
    """
    Metadata for the entire knowledge graph
    
    Attributes:
        course_id: Unique course identifier
        course_name: Human-readable course name
        num_nodes: Total nodes in graph
        num_edges: Total edges in graph
        metadata: Additional course-level attributes
    """
    course_id: str
    course_name: str
    version: str = "1.0"
    num_nodes: int = 0
    num_edges: int = 0
    num_learners: Optional[int] = None
    avg_completion_rate: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModuleNode(Node):
    """Specialized node for course modules"""
    node_type: NodeType = NodeType.MODULE
    module_order: Optional[int] = None
    estimated_hours: Optional[float] = None
    skills_taught: List[str] = Field(default_factory=list)
    prerequisite_skills: List[str] = Field(default_factory=list)


class AssessmentNode(Node):
    """Specialized node for assessments/quizzes"""
    node_type: NodeType = NodeType.ASSESSMENT
    assessment_type: str = "quiz"  # quiz, exam, project, peer_review
    max_score: float = 100.0
    passing_score: Optional[float] = None
    num_questions: Optional[int] = None
    skills_assessed: List[str] = Field(default_factory=list)


class SkillNode(Node):
    """Specialized node for skills/competencies"""
    node_type: NodeType = NodeType.SKILL
    skill_category: Optional[str] = None  # programming, math, theory, etc.
    is_prerequisite: bool = False
    mastery_threshold: Optional[float] = None


