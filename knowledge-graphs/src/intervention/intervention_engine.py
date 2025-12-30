"""
Intervention Engine

Recommends and generates personalized interventions for at-risk learners.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
from loguru import logger

from ..graph.knowledge_graph import CourseKnowledgeGraph
from ..trajectories.trajectory_builder import LearnerTrajectory


class InterventionType(str, Enum):
    """Types of interventions"""
    REMEDIAL_CONTENT = "remedial_content"
    ALTERNATIVE_PATH = "alternative_path"
    TARGETED_NUDGE = "targeted_nudge"
    PEER_SUPPORT = "peer_support"
    INSTRUCTOR_ATTENTION = "instructor_attention"
    PACING_ADJUSTMENT = "pacing_adjustment"
    SKILL_REFRESHER = "skill_refresher"
    MOTIVATIONAL_SUPPORT = "motivational_support"


@dataclass
class Intervention:
    """
    Represents an intervention recommendation
    
    Attributes:
        intervention_type: Type of intervention
        priority: Priority level (1-5, 5 being highest)
        expected_impact: Expected reduction in dropout risk
        description: Human-readable description
        action_items: Specific actions to take
        target_module: Module where intervention should occur
        timing: When to deliver intervention
        resources: Required resources or content
    """
    intervention_type: InterventionType
    priority: int
    expected_impact: float
    description: str
    action_items: List[str]
    target_module: Optional[str] = None
    timing: str = "immediate"
    resources: List[str] = None
    cost_estimate: str = "low"
    
    def __post_init__(self):
        if self.resources is None:
            self.resources = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "type": self.intervention_type.value,
            "priority": self.priority,
            "expected_impact": self.expected_impact,
            "description": self.description,
            "action_items": self.action_items,
            "target_module": self.target_module,
            "timing": self.timing,
            "resources": self.resources,
            "cost_estimate": self.cost_estimate
        }


class InterventionEngine:
    """
    Generates personalized intervention recommendations
    """
    
    def __init__(self, knowledge_graph: CourseKnowledgeGraph):
        """
        Initialize intervention engine
        
        Args:
            knowledge_graph: Course knowledge graph
        """
        self.knowledge_graph = knowledge_graph
    
    def recommend_interventions(
        self,
        learner_id: str,
        trajectory: LearnerTrajectory,
        risk_score: float,
        risk_factors: List[Dict[str, Any]],
        max_interventions: int = 3
    ) -> List[Intervention]:
        """
        Recommend interventions for a learner
        
        Args:
            learner_id: Learner identifier
            trajectory: Learner trajectory
            risk_score: Current risk score
            risk_factors: Contributing risk factors
            max_interventions: Maximum number of interventions to recommend
            
        Returns:
            List of intervention recommendations
        """
        logger.info(f"Generating interventions for learner {learner_id} (risk: {risk_score:.2f})")
        
        interventions = []
        
        # Generate interventions based on risk factors
        for factor in risk_factors:
            factor_type = factor["factor"]
            severity = factor["severity"]
            
            if factor_type == "Low Performance":
                interventions.append(self._create_remedial_content_intervention(
                    trajectory, factor, severity
                ))
            
            elif factor_type == "Inactivity":
                interventions.append(self._create_engagement_nudge_intervention(
                    trajectory, factor, severity
                ))
            
            elif factor_type == "High Module Difficulty":
                interventions.append(self._create_difficulty_adjustment_intervention(
                    trajectory, factor, severity
                ))
            
            elif factor_type == "Missing Prerequisites":
                interventions.append(self._create_prerequisite_review_intervention(
                    trajectory, factor, severity
                ))
            
            elif factor_type == "Multiple Attempts":
                interventions.append(self._create_targeted_support_intervention(
                    trajectory, factor, severity
                ))
            
            elif factor_type == "High-Risk Module":
                interventions.append(self._create_instructor_attention_intervention(
                    trajectory, factor, severity
                ))
            
            elif factor_type == "Declining Performance":
                interventions.append(self._create_motivational_intervention(
                    trajectory, factor, severity
                ))
        
        # General interventions if risk is high but no specific factors identified
        if risk_score > 0.7 and not interventions:
            interventions.append(self._create_general_support_intervention(trajectory))
        
        # Prioritize and filter
        interventions = self._prioritize_interventions(interventions, risk_score)
        
        return interventions[:max_interventions]
    
    def recommend_course_redesign(
        self,
        bottleneck_nodes: List[Dict[str, Any]],
        bottleneck_edges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Recommend course redesign opportunities
        
        Args:
            bottleneck_nodes: List of bottleneck nodes
            bottleneck_edges: List of bottleneck edges
            
        Returns:
            List of redesign recommendations
        """
        logger.info("Generating course redesign recommendations")
        
        recommendations = []
        
        # Node-based recommendations
        for node in bottleneck_nodes[:5]:  # Top 5
            if node["dropout_rate"] > 0.5:
                recommendations.append({
                    "type": "module_simplification",
                    "target": node["node_id"],
                    "target_name": node["name"],
                    "priority": "high",
                    "issue": f"High dropout rate ({node['dropout_rate']:.1%})",
                    "recommendation": "Break down module into smaller, more manageable units",
                    "expected_impact": "20-30% reduction in dropout"
                })
            
            if node.get("centrality", 0) > 0.8 and node["dropout_rate"] > 0.3:
                recommendations.append({
                    "type": "bottleneck_mitigation",
                    "target": node["node_id"],
                    "target_name": node["name"],
                    "priority": "critical",
                    "issue": "Critical bottleneck with high structural importance",
                    "recommendation": "Add alternative learning paths or prerequisites",
                    "expected_impact": "30-40% reduction in dropout"
                })
        
        # Edge-based recommendations
        for edge in bottleneck_edges[:5]:  # Top 5
            if edge.get("difficulty_gap", 0) > 3:
                recommendations.append({
                    "type": "transition_smoothing",
                    "target": f"{edge['source']} → {edge['target']}",
                    "target_name": f"{edge['source_name']} → {edge['target_name']}",
                    "priority": "high",
                    "issue": f"Large difficulty gap ({edge['difficulty_gap']:.1f} points)",
                    "recommendation": "Insert intermediate module or provide transition materials",
                    "expected_impact": "15-25% reduction in transition dropout"
                })
        
        return recommendations
    
    def generate_intervention_report(
        self,
        interventions_by_learner: Dict[str, List[Intervention]]
    ) -> Dict[str, Any]:
        """
        Generate summary report of interventions
        
        Args:
            interventions_by_learner: Dictionary mapping learner_id to interventions
            
        Returns:
            Summary report
        """
        # Count intervention types
        type_counts = {}
        total_expected_impact = 0.0
        priority_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for interventions in interventions_by_learner.values():
            for intervention in interventions:
                itype = intervention.intervention_type.value
                type_counts[itype] = type_counts.get(itype, 0) + 1
                total_expected_impact += intervention.expected_impact
                priority_counts[intervention.priority] += 1
        
        return {
            "total_learners": len(interventions_by_learner),
            "total_interventions": sum(len(v) for v in interventions_by_learner.values()),
            "intervention_type_distribution": type_counts,
            "priority_distribution": priority_counts,
            "avg_expected_impact": total_expected_impact / max(sum(len(v) for v in interventions_by_learner.values()), 1),
            "high_priority_count": priority_counts[4] + priority_counts[5]
        }
    
    def _create_remedial_content_intervention(
        self,
        trajectory: LearnerTrajectory,
        factor: Dict[str, Any],
        severity: str
    ) -> Intervention:
        """Create remedial content intervention"""
        current_module = trajectory.steps[-1].module_id if trajectory.steps else None
        
        return Intervention(
            intervention_type=InterventionType.REMEDIAL_CONTENT,
            priority=5 if severity == "high" else 3,
            expected_impact=0.25,
            description="Provide additional practice materials and examples",
            action_items=[
                "Identify struggling topics from low-scoring assessments",
                "Assign targeted practice exercises",
                "Provide worked examples and explanations",
                "Schedule follow-up assessment"
            ],
            target_module=current_module,
            timing="immediate",
            resources=["Practice problem sets", "Tutorial videos", "Worked examples"],
            cost_estimate="medium"
        )
    
    def _create_engagement_nudge_intervention(
        self,
        trajectory: LearnerTrajectory,
        factor: Dict[str, Any],
        severity: str
    ) -> Intervention:
        """Create engagement nudge intervention"""
        days_inactive = factor.get("value", 0)
        
        return Intervention(
            intervention_type=InterventionType.TARGETED_NUDGE,
            priority=4 if days_inactive > 14 else 3,
            expected_impact=0.20,
            description=f"Re-engage learner after {days_inactive:.0f} days of inactivity",
            action_items=[
                "Send personalized email reminder",
                "Highlight progress made so far",
                "Suggest next steps",
                "Offer support resources"
            ],
            timing="immediate",
            resources=["Email template", "Progress dashboard link"],
            cost_estimate="low"
        )
    
    def _create_difficulty_adjustment_intervention(
        self,
        trajectory: LearnerTrajectory,
        factor: Dict[str, Any],
        severity: str
    ) -> Intervention:
        """Create difficulty adjustment intervention"""
        current_module = trajectory.steps[-1].module_id if trajectory.steps else None
        
        return Intervention(
            intervention_type=InterventionType.REMEDIAL_CONTENT,
            priority=4,
            expected_impact=0.22,
            description="Provide scaffolding for difficult module",
            action_items=[
                "Offer simplified introduction to concepts",
                "Provide additional worked examples",
                "Break down complex topics into steps",
                "Enable optional practice mode"
            ],
            target_module=current_module,
            timing="before module start",
            resources=["Introductory materials", "Simplified examples", "Practice exercises"],
            cost_estimate="medium"
        )
    
    def _create_prerequisite_review_intervention(
        self,
        trajectory: LearnerTrajectory,
        factor: Dict[str, Any],
        severity: str
    ) -> Intervention:
        """Create prerequisite review intervention"""
        current_module = trajectory.steps[-1].module_id if trajectory.steps else None
        
        # Get prerequisite modules
        prerequisites = self.knowledge_graph.get_prerequisites(current_module) if current_module else []
        
        return Intervention(
            intervention_type=InterventionType.SKILL_REFRESHER,
            priority=5,
            expected_impact=0.30,
            description="Review prerequisite skills before continuing",
            action_items=[
                "Identify specific prerequisite gaps",
                "Recommend review modules",
                "Provide prerequisite skill refresher",
                "Allow optional self-assessment"
            ],
            target_module=current_module,
            timing="immediate",
            resources=[f"Review: {p}" for p in prerequisites[:3]],
            cost_estimate="low"
        )
    
    def _create_targeted_support_intervention(
        self,
        trajectory: LearnerTrajectory,
        factor: Dict[str, Any],
        severity: str
    ) -> Intervention:
        """Create targeted support intervention"""
        return Intervention(
            intervention_type=InterventionType.PEER_SUPPORT,
            priority=3,
            expected_impact=0.18,
            description="Connect learner with peer support",
            action_items=[
                "Identify study group or forum",
                "Match with peer mentor",
                "Suggest Q&A sessions",
                "Enable collaborative learning"
            ],
            timing="within 1 week",
            resources=["Study group invitations", "Peer mentor contact"],
            cost_estimate="low"
        )
    
    def _create_instructor_attention_intervention(
        self,
        trajectory: LearnerTrajectory,
        factor: Dict[str, Any],
        severity: str
    ) -> Intervention:
        """Create instructor attention intervention"""
        current_module = trajectory.steps[-1].module_id if trajectory.steps else None
        
        return Intervention(
            intervention_type=InterventionType.INSTRUCTOR_ATTENTION,
            priority=5,
            expected_impact=0.15,
            description="Flag for instructor review and potential course redesign",
            action_items=[
                "Notify instructor of high-risk module",
                "Analyze module-specific pain points",
                "Consider module redesign or clarification",
                "Provide targeted office hours"
            ],
            target_module=current_module,
            timing="ongoing",
            resources=["Instructor dashboard", "Module analytics report"],
            cost_estimate="medium"
        )
    
    def _create_motivational_intervention(
        self,
        trajectory: LearnerTrajectory,
        factor: Dict[str, Any],
        severity: str
    ) -> Intervention:
        """Create motivational intervention"""
        return Intervention(
            intervention_type=InterventionType.MOTIVATIONAL_SUPPORT,
            priority=4,
            expected_impact=0.20,
            description="Provide motivational support and progress encouragement",
            action_items=[
                "Send encouraging progress update",
                "Highlight achievements so far",
                "Share success stories from similar learners",
                "Offer one-on-one coaching session"
            ],
            timing="immediate",
            resources=["Progress visualization", "Success stories", "Coaching session"],
            cost_estimate="low"
        )
    
    def _create_general_support_intervention(
        self,
        trajectory: LearnerTrajectory
    ) -> Intervention:
        """Create general support intervention for high risk without specific factors"""
        return Intervention(
            intervention_type=InterventionType.INSTRUCTOR_ATTENTION,
            priority=4,
            expected_impact=0.15,
            description="General support for high-risk learner",
            action_items=[
                "Reach out with personalized check-in",
                "Assess individual challenges",
                "Offer flexible pacing options",
                "Provide access to all support resources"
            ],
            timing="immediate",
            resources=["Support resources directory", "Flexible pacing guide"],
            cost_estimate="medium"
        )
    
    def _prioritize_interventions(
        self,
        interventions: List[Intervention],
        risk_score: float
    ) -> List[Intervention]:
        """Prioritize interventions by expected impact and priority"""
        # Boost priority for very high risk learners
        if risk_score > 0.8:
            for intervention in interventions:
                intervention.priority = min(intervention.priority + 1, 5)
        
        # Sort by priority (descending) and expected impact (descending)
        interventions.sort(key=lambda x: (x.priority, x.expected_impact), reverse=True)
        
        return interventions


