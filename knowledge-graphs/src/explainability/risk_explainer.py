"""
Risk Explainer

Explains why a learner has high dropout risk using SHAP and feature importance.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from loguru import logger

from ..trajectories.trajectory_builder import LearnerTrajectory
from ..trajectories.trajectory_features import TrajectoryFeatureExtractor
from ..graph.knowledge_graph import CourseKnowledgeGraph


class RiskExplainer:
    """
    Explains dropout risk predictions
    """
    
    def __init__(
        self,
        knowledge_graph: CourseKnowledgeGraph,
        feature_extractor: TrajectoryFeatureExtractor
    ):
        """
        Initialize risk explainer
        
        Args:
            knowledge_graph: Course knowledge graph
            feature_extractor: Trajectory feature extractor
        """
        self.knowledge_graph = knowledge_graph
        self.feature_extractor = feature_extractor
    
    def explain_risk(
        self,
        trajectory: LearnerTrajectory,
        risk_score: float,
        step_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Explain why a learner has high dropout risk
        
        Args:
            trajectory: Learner trajectory
            risk_score: Predicted risk score
            step_idx: Optional step index to explain (default: last step)
            
        Returns:
            Explanation dictionary with contributing factors
        """
        if step_idx is None:
            step_idx = len(trajectory.steps) - 1
        
        # Extract features
        features = self.feature_extractor.extract_step_features(trajectory, step_idx)
        
        # Identify contributing factors
        factors = self._identify_risk_factors(features, trajectory, step_idx)
        
        # Generate explanation
        explanation = {
            "learner_id": trajectory.learner_id,
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "current_module": trajectory.steps[step_idx].module_id,
            "step_position": step_idx,
            "contributing_factors": factors,
            "recommendations": self._generate_recommendations(factors),
            "summary": self._generate_summary(factors, risk_score)
        }
        
        return explanation
    
    def explain_batch(
        self,
        trajectories: Dict[str, LearnerTrajectory],
        risk_scores: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Generate explanations for multiple learners
        
        Args:
            trajectories: Dictionary of learner trajectories
            risk_scores: Dictionary of risk scores per learner
            
        Returns:
            DataFrame with explanations
        """
        explanations = []
        
        for learner_id, trajectory in trajectories.items():
            risk_score = risk_scores.get(learner_id, 0.5)
            
            if not trajectory.steps:
                continue
            
            explanation = self.explain_risk(trajectory, risk_score)
            explanations.append(explanation)
        
        return pd.DataFrame(explanations)
    
    def _identify_risk_factors(
        self,
        features: Dict[str, float],
        trajectory: LearnerTrajectory,
        step_idx: int
    ) -> List[Dict[str, Any]]:
        """Identify specific factors contributing to risk"""
        factors = []
        
        # Performance-related factors
        if features.get("avg_previous_score", 100) < 60:
            factors.append({
                "factor": "Low Performance",
                "severity": "high",
                "value": features["avg_previous_score"],
                "description": f"Average score of {features['avg_previous_score']:.1f}% is below passing threshold"
            })
        
        # Engagement-related factors
        if features.get("time_since_last_activity", 0) > 7:
            factors.append({
                "factor": "Inactivity",
                "severity": "high",
                "value": features["time_since_last_activity"],
                "description": f"{features['time_since_last_activity']:.1f} days since last activity"
            })
        
        # Difficulty-related factors
        if features.get("module_difficulty", 0) > 7:
            factors.append({
                "factor": "High Module Difficulty",
                "severity": "medium",
                "value": features["module_difficulty"],
                "description": f"Current module has difficulty {features['module_difficulty']:.1f}/10"
            })
        
        # Prerequisite factors
        if features.get("prerequisites_completed", 1) < 0.5:
            factors.append({
                "factor": "Missing Prerequisites",
                "severity": "high",
                "value": features["prerequisites_completed"],
                "description": f"Only {features['prerequisites_completed']:.0%} of prerequisites completed"
            })
        
        # Struggle indicators
        if features.get("avg_attempts", 1) > 2:
            factors.append({
                "factor": "Multiple Attempts",
                "severity": "medium",
                "value": features["avg_attempts"],
                "description": f"Average of {features['avg_attempts']:.1f} attempts per module"
            })
        
        # Module-specific risk
        module_dropout_rate = features.get("module_dropout_rate", 0)
        if module_dropout_rate > 0.4:
            factors.append({
                "factor": "High-Risk Module",
                "severity": "high",
                "value": module_dropout_rate,
                "description": f"This module has a {module_dropout_rate:.0%} dropout rate"
            })
        
        # Declining performance trend
        if features.get("score_trend", 0) < -5:
            factors.append({
                "factor": "Declining Performance",
                "severity": "high",
                "value": features["score_trend"],
                "description": "Scores are declining over time"
            })
        
        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        factors.sort(key=lambda x: severity_order[x["severity"]])
        
        return factors
    
    def _generate_recommendations(self, factors: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on risk factors"""
        recommendations = []
        
        factor_types = {f["factor"] for f in factors}
        
        if "Low Performance" in factor_types:
            recommendations.append("Provide additional practice materials or tutoring support")
        
        if "Inactivity" in factor_types:
            recommendations.append("Send engagement nudge or reminder email")
        
        if "High Module Difficulty" in factor_types:
            recommendations.append("Offer supplementary content to ease difficulty transition")
        
        if "Missing Prerequisites" in factor_types:
            recommendations.append("Recommend reviewing prerequisite modules before continuing")
        
        if "Multiple Attempts" in factor_types:
            recommendations.append("Provide targeted remedial content on struggling topics")
        
        if "High-Risk Module" in factor_types:
            recommendations.append("Flag for instructor attention - module may need redesign")
        
        if "Declining Performance" in factor_types:
            recommendations.append("Schedule check-in or provide motivational support")
        
        return recommendations
    
    def _generate_summary(self, factors: List[Dict[str, Any]], risk_score: float) -> str:
        """Generate human-readable summary"""
        if not factors:
            return f"Risk score of {risk_score:.1%} - no major risk factors identified."
        
        high_severity_count = sum(1 for f in factors if f["severity"] == "high")
        
        if high_severity_count == 0:
            return f"Moderate risk ({risk_score:.1%}) with {len(factors)} contributing factors."
        elif high_severity_count == 1:
            main_factor = factors[0]["factor"]
            return f"High risk ({risk_score:.1%}) primarily due to: {main_factor}."
        else:
            return f"High risk ({risk_score:.1%}) with {high_severity_count} major contributing factors."
    
    @staticmethod
    def _get_risk_level(risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score >= 0.7:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"


