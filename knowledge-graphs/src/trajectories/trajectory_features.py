"""
Trajectory Feature Extraction

Extracts features from learner trajectories for ML models.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from loguru import logger

from .trajectory_builder import LearnerTrajectory, TrajectoryStep
from ..graph.knowledge_graph import CourseKnowledgeGraph


class TrajectoryFeatureExtractor:
    """
    Extracts features from learner trajectories for ML models
    """
    
    def __init__(self, knowledge_graph: CourseKnowledgeGraph):
        """
        Initialize feature extractor
        
        Args:
            knowledge_graph: Course knowledge graph for structural features
        """
        self.knowledge_graph = knowledge_graph
    
    def extract_step_features(
        self,
        trajectory: LearnerTrajectory,
        step_idx: int
    ) -> Dict[str, float]:
        """
        Extract features for a specific step in a trajectory
        
        Args:
            trajectory: Learner trajectory
            step_idx: Index of step to extract features for
            
        Returns:
            Dictionary of features
        """
        if step_idx >= len(trajectory.steps):
            raise ValueError(f"Step index {step_idx} out of range")
        
        step = trajectory.steps[step_idx]
        features = {}
        
        # Current step features
        features["current_score"] = step.score if step.score is not None else 0.0
        features["current_time_spent"] = step.time_spent if step.time_spent is not None else 0.0
        features["current_attempts"] = float(step.attempts)
        features["current_completed"] = float(step.completed)
        
        # Historical features (looking back)
        if step_idx > 0:
            prev_steps = trajectory.steps[:step_idx]
            
            # Performance trends
            scores = [s.score for s in prev_steps if s.score is not None]
            features["avg_previous_score"] = np.mean(scores) if scores else 0.0
            features["min_previous_score"] = np.min(scores) if scores else 0.0
            features["max_previous_score"] = np.max(scores) if scores else 0.0
            features["score_trend"] = self._compute_trend(scores) if len(scores) > 1 else 0.0
            
            # Time patterns
            times = [s.time_spent for s in prev_steps if s.time_spent is not None]
            features["avg_previous_time"] = np.mean(times) if times else 0.0
            features["total_time_spent"] = sum(times) if times else 0.0
            
            # Completion rate
            completions = [float(s.completed) for s in prev_steps]
            features["historical_completion_rate"] = np.mean(completions) if completions else 0.0
            
            # Attempts
            attempts = [s.attempts for s in prev_steps]
            features["avg_attempts"] = np.mean(attempts) if attempts else 1.0
            features["max_attempts"] = np.max(attempts) if attempts else 1
            
            # Time since last activity
            time_delta = (step.timestamp - prev_steps[-1].timestamp).total_seconds()
            features["time_since_last_activity"] = time_delta / 86400.0  # days
            
            # Inactivity indicators
            features["long_gap_indicator"] = float(time_delta > 7 * 86400)  # > 7 days
        else:
            # First step - no history
            for key in [
                "avg_previous_score", "min_previous_score", "max_previous_score",
                "score_trend", "avg_previous_time", "total_time_spent",
                "historical_completion_rate", "avg_attempts", "max_attempts",
                "time_since_last_activity", "long_gap_indicator"
            ]:
                features[key] = 0.0
        
        # Graph-based features
        node = self.knowledge_graph.get_node(step.module_id)
        if node:
            features["module_difficulty"] = node.difficulty if node.difficulty else 5.0
            features["module_avg_completion_rate"] = node.completion_rate if node.completion_rate else 0.5
            features["module_avg_score"] = node.average_score if node.average_score else 50.0
            features["module_dropout_rate"] = node.dropout_rate if node.dropout_rate else 0.5
            
            # Structural features
            predecessors = self.knowledge_graph.get_prerequisites(step.module_id)
            successors = self.knowledge_graph.get_successors(step.module_id)
            features["num_prerequisites"] = float(len(predecessors))
            features["num_successors"] = float(len(successors))
            
            # Check if learner visited prerequisites
            visited = set(trajectory.get_visited_modules()[:step_idx])
            features["prerequisites_completed"] = sum(1 for p in predecessors if p in visited) / max(len(predecessors), 1)
        else:
            # Module not in graph
            for key in [
                "module_difficulty", "module_avg_completion_rate", "module_avg_score",
                "module_dropout_rate", "num_prerequisites", "num_successors",
                "prerequisites_completed"
            ]:
                features[key] = 0.0
        
        # Trajectory position features
        features["step_position"] = float(step_idx)
        features["relative_position"] = step_idx / max(len(trajectory.steps), 1)
        features["steps_remaining"] = float(len(trajectory.steps) - step_idx - 1)
        
        return features
    
    def extract_trajectory_features(
        self,
        trajectory: LearnerTrajectory
    ) -> Dict[str, float]:
        """
        Extract aggregate features for entire trajectory
        
        Args:
            trajectory: Learner trajectory
            
        Returns:
            Dictionary of aggregate features
        """
        features = {}
        
        # Basic trajectory stats
        features["trajectory_length"] = float(len(trajectory.steps))
        features["total_time_spent"] = trajectory.total_time_spent
        features["final_completion_rate"] = trajectory.final_completion_rate
        
        # Performance aggregates
        avg_score = trajectory.get_average_score()
        features["avg_score"] = avg_score if avg_score is not None else 0.0
        
        scores = [s.score for s in trajectory.steps if s.score is not None]
        if scores:
            features["score_std"] = np.std(scores)
            features["score_range"] = np.max(scores) - np.min(scores)
        else:
            features["score_std"] = 0.0
            features["score_range"] = 0.0
        
        # Time patterns
        completion_times = trajectory.get_completion_times()
        if completion_times:
            features["avg_time_between_modules"] = np.mean(completion_times)
            features["max_time_between_modules"] = np.max(completion_times)
            features["engagement_consistency"] = 1.0 / (np.std(completion_times) + 1.0)
        else:
            features["avg_time_between_modules"] = 0.0
            features["max_time_between_modules"] = 0.0
            features["engagement_consistency"] = 1.0
        
        # Struggle indicators
        features["total_attempts"] = sum(s.attempts for s in trajectory.steps)
        features["avg_attempts_per_module"] = features["total_attempts"] / max(len(trajectory.steps), 1)
        features["max_attempts_single_module"] = max((s.attempts for s in trajectory.steps), default=1)
        
        # Completion patterns
        features["num_completed"] = sum(1 for s in trajectory.steps if s.completed)
        features["completion_rate"] = features["num_completed"] / max(len(trajectory.steps), 1)
        
        return features
    
    def extract_features_for_dataset(
        self,
        trajectories: Dict[str, LearnerTrajectory],
        include_step_features: bool = True
    ) -> pd.DataFrame:
        """
        Extract features for multiple trajectories
        
        Args:
            trajectories: Dictionary of learner trajectories
            include_step_features: Whether to include per-step features
            
        Returns:
            DataFrame with features and labels
        """
        logger.info(f"Extracting features for {len(trajectories)} trajectories")
        
        all_features = []
        
        for learner_id, trajectory in trajectories.items():
            if include_step_features:
                # Extract features for each step
                for step_idx in range(len(trajectory.steps)):
                    step_features = self.extract_step_features(trajectory, step_idx)
                    step = trajectory.steps[step_idx]
                    
                    # Add identifiers and labels
                    step_features["learner_id"] = learner_id
                    step_features["module_id"] = step.module_id
                    step_features["step_idx"] = step_idx
                    step_features["dropped_out"] = float(step.dropped_out)
                    step_features["label"] = float(trajectory.dropped_out)  # Overall dropout
                    
                    all_features.append(step_features)
            else:
                # Extract trajectory-level features only
                traj_features = self.extract_trajectory_features(trajectory)
                traj_features["learner_id"] = learner_id
                traj_features["dropped_out"] = float(trajectory.dropped_out)
                all_features.append(traj_features)
        
        df = pd.DataFrame(all_features)
        logger.info(f"Extracted {len(df)} feature rows with {len(df.columns)} features")
        
        return df
    
    @staticmethod
    def _compute_trend(values: List[float]) -> float:
        """Compute linear trend (slope) of a sequence of values"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]  # slope


