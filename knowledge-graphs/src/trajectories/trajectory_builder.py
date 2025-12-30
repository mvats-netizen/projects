"""
Learner Trajectory Builder

Constructs learner trajectories from event logs and interaction data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger

from ..graph.knowledge_graph import CourseKnowledgeGraph


@dataclass
class TrajectoryStep:
    """
    Represents a single step in a learner's trajectory
    
    Attributes:
        module_id: Current module/node ID
        timestamp: When this step occurred
        score: Performance score (if applicable)
        time_spent: Time spent in seconds
        attempts: Number of attempts
        completed: Whether completed successfully
        dropped_out: Whether learner dropped out after this step
        features: Additional features
    """
    module_id: str
    timestamp: datetime
    score: Optional[float] = None
    time_spent: Optional[float] = None  # seconds
    attempts: int = 1
    completed: bool = False
    dropped_out: bool = False
    features: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "module_id": self.module_id,
            "timestamp": self.timestamp.isoformat(),
            "score": self.score,
            "time_spent": self.time_spent,
            "attempts": self.attempts,
            "completed": self.completed,
            "dropped_out": self.dropped_out,
            **self.features
        }


@dataclass
class LearnerTrajectory:
    """
    Represents a learner's complete trajectory through a course
    
    Attributes:
        learner_id: Unique learner identifier
        course_id: Course identifier
        steps: Sequence of trajectory steps
        dropped_out: Whether learner dropped out
        final_completion_rate: Percentage of course completed
        metadata: Additional learner metadata
    """
    learner_id: str
    course_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    dropped_out: bool = False
    final_completion_rate: float = 0.0
    total_time_spent: float = 0.0  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: TrajectoryStep) -> None:
        """Add a step to the trajectory"""
        self.steps.append(step)
        if step.time_spent:
            self.total_time_spent += step.time_spent
    
    def get_visited_modules(self) -> List[str]:
        """Get list of all visited module IDs"""
        return [step.module_id for step in self.steps]
    
    def get_trajectory_length(self) -> int:
        """Get number of steps in trajectory"""
        return len(self.steps)
    
    def get_average_score(self) -> Optional[float]:
        """Get average score across all steps"""
        scores = [s.score for s in self.steps if s.score is not None]
        return np.mean(scores) if scores else None
    
    def get_completion_times(self) -> List[float]:
        """Get time between consecutive steps (in days)"""
        if len(self.steps) < 2:
            return []
        
        times = []
        for i in range(1, len(self.steps)):
            delta = (self.steps[i].timestamp - self.steps[i-1].timestamp).total_seconds()
            times.append(delta / 86400.0)  # Convert to days
        return times
    
    def get_dropout_point(self) -> Optional[str]:
        """Get module ID where dropout occurred"""
        for step in self.steps:
            if step.dropped_out:
                return step.module_id
        return None
    
    def to_sequence(self) -> List[str]:
        """Convert trajectory to sequence of module IDs"""
        return self.get_visited_modules()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trajectory to pandas DataFrame"""
        return pd.DataFrame([step.to_dict() for step in self.steps])


class TrajectoryBuilder:
    """
    Builds learner trajectories from raw interaction data
    """
    
    def __init__(self, knowledge_graph: Optional[CourseKnowledgeGraph] = None):
        """
        Initialize trajectory builder
        
        Args:
            knowledge_graph: Optional knowledge graph for validation
        """
        self.knowledge_graph = knowledge_graph
        self.trajectories: Dict[str, LearnerTrajectory] = {}
    
    def build_from_csv(
        self,
        csv_path: str,
        learner_id_col: str = "learner_id",
        module_id_col: str = "module_id",
        timestamp_col: str = "timestamp",
        score_col: Optional[str] = "score",
        time_spent_col: Optional[str] = "time_spent",
        completed_col: Optional[str] = "completed",
        dropped_out_col: Optional[str] = "dropped_out"
    ) -> Dict[str, LearnerTrajectory]:
        """
        Build trajectories from CSV file
        
        Args:
            csv_path: Path to CSV file
            learner_id_col: Column name for learner ID
            module_id_col: Column name for module ID
            timestamp_col: Column name for timestamp
            score_col: Column name for score (optional)
            time_spent_col: Column name for time spent (optional)
            completed_col: Column name for completion status (optional)
            dropped_out_col: Column name for dropout status (optional)
            
        Returns:
            Dictionary mapping learner_id to LearnerTrajectory
        """
        logger.info(f"Building trajectories from {csv_path}")
        
        df = pd.read_csv(csv_path)
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Sort by learner and timestamp
        df = df.sort_values([learner_id_col, timestamp_col])
        
        # Group by learner
        for learner_id, learner_df in df.groupby(learner_id_col):
            course_id = learner_df["course_id"].iloc[0] if "course_id" in learner_df else "unknown"
            trajectory = LearnerTrajectory(
                learner_id=str(learner_id),
                course_id=course_id
            )
            
            # Build steps
            for _, row in learner_df.iterrows():
                step = TrajectoryStep(
                    module_id=row[module_id_col],
                    timestamp=row[timestamp_col],
                    score=row[score_col] if score_col and score_col in row else None,
                    time_spent=row[time_spent_col] if time_spent_col and time_spent_col in row else None,
                    attempts=row.get("attempts", 1),
                    completed=row[completed_col] if completed_col and completed_col in row else False,
                    dropped_out=row[dropped_out_col] if dropped_out_col and dropped_out_col in row else False
                )
                trajectory.add_step(step)
            
            # Set trajectory-level attributes
            trajectory.dropped_out = any(s.dropped_out for s in trajectory.steps)
            completed_steps = sum(1 for s in trajectory.steps if s.completed)
            trajectory.final_completion_rate = completed_steps / len(trajectory.steps) if trajectory.steps else 0.0
            
            self.trajectories[str(learner_id)] = trajectory
        
        logger.info(f"Built {len(self.trajectories)} trajectories")
        return self.trajectories
    
    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        learner_id_col: str = "learner_id",
        module_id_col: str = "module_id",
        timestamp_col: str = "timestamp"
    ) -> Dict[str, LearnerTrajectory]:
        """
        Build trajectories from pandas DataFrame
        
        Args:
            df: DataFrame with learner interaction data
            learner_id_col: Column name for learner ID
            module_id_col: Column name for module ID
            timestamp_col: Column name for timestamp
            
        Returns:
            Dictionary mapping learner_id to LearnerTrajectory
        """
        # Similar to build_from_csv but operates on DataFrame directly
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values([learner_id_col, timestamp_col])
        
        for learner_id, learner_df in df.groupby(learner_id_col):
            trajectory = LearnerTrajectory(
                learner_id=str(learner_id),
                course_id=learner_df.get("course_id", "unknown").iloc[0]
            )
            
            for _, row in learner_df.iterrows():
                step = TrajectoryStep(
                    module_id=row[module_id_col],
                    timestamp=row[timestamp_col],
                    score=row.get("score"),
                    time_spent=row.get("time_spent"),
                    completed=row.get("completed", False),
                    dropped_out=row.get("dropped_out", False)
                )
                trajectory.add_step(step)
            
            trajectory.dropped_out = any(s.dropped_out for s in trajectory.steps)
            self.trajectories[str(learner_id)] = trajectory
        
        return self.trajectories
    
    def get_trajectory(self, learner_id: str) -> Optional[LearnerTrajectory]:
        """Get trajectory for a specific learner"""
        return self.trajectories.get(learner_id)
    
    def filter_trajectories(
        self,
        min_length: int = 1,
        only_dropouts: bool = False,
        only_completions: bool = False
    ) -> Dict[str, LearnerTrajectory]:
        """
        Filter trajectories based on criteria
        
        Args:
            min_length: Minimum trajectory length
            only_dropouts: Only include dropout trajectories
            only_completions: Only include completion trajectories
            
        Returns:
            Filtered dictionary of trajectories
        """
        filtered = {}
        
        for learner_id, traj in self.trajectories.items():
            if len(traj.steps) < min_length:
                continue
            
            if only_dropouts and not traj.dropped_out:
                continue
            
            if only_completions and traj.dropped_out:
                continue
            
            filtered[learner_id] = traj
        
        logger.info(f"Filtered to {len(filtered)} trajectories")
        return filtered
    
    def get_trajectory_statistics(self) -> Dict[str, Any]:
        """Compute statistics across all trajectories"""
        if not self.trajectories:
            return {}
        
        lengths = [len(t.steps) for t in self.trajectories.values()]
        dropout_rate = sum(1 for t in self.trajectories.values() if t.dropped_out) / len(self.trajectories)
        avg_scores = [t.get_average_score() for t in self.trajectories.values()]
        avg_scores = [s for s in avg_scores if s is not None]
        
        return {
            "num_trajectories": len(self.trajectories),
            "avg_trajectory_length": np.mean(lengths),
            "median_trajectory_length": np.median(lengths),
            "min_trajectory_length": np.min(lengths),
            "max_trajectory_length": np.max(lengths),
            "dropout_rate": dropout_rate,
            "avg_score": np.mean(avg_scores) if avg_scores else None,
            "unique_learners": len(set(self.trajectories.keys())),
        }
    
    def export_to_csv(self, output_path: str) -> None:
        """Export all trajectories to CSV"""
        all_steps = []
        
        for learner_id, trajectory in self.trajectories.items():
            for step in trajectory.steps:
                step_dict = step.to_dict()
                step_dict["learner_id"] = learner_id
                step_dict["course_id"] = trajectory.course_id
                all_steps.append(step_dict)
        
        df = pd.DataFrame(all_steps)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported trajectories to {output_path}")


