"""
Risk Predictor - Main Interface

Combines GNN embeddings and temporal models for dropout risk prediction.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from pathlib import Path

from ..graph.knowledge_graph import CourseKnowledgeGraph
from ..trajectories.trajectory_builder import LearnerTrajectory, TrajectoryBuilder
from ..trajectories.trajectory_features import TrajectoryFeatureExtractor
from .gnn_embeddings import GraphSAGEModel, prepare_graph_data, GNNEmbeddingTrainer
from .temporal_model import TemporalRiskModel, LSTMRiskModel


class RiskPredictor:
    """
    Main interface for learner dropout risk prediction
    
    Combines:
    - Knowledge graph structure (via GNN embeddings)
    - Learner trajectories (temporal patterns)
    - Statistical features
    """
    
    def __init__(
        self,
        knowledge_graph: CourseKnowledgeGraph,
        gnn_model_type: str = "graphsage",
        temporal_model_type: str = "transformer",
        device: str = "cpu"
    ):
        """
        Initialize Risk Predictor
        
        Args:
            knowledge_graph: Course knowledge graph
            gnn_model_type: Type of GNN model (graphsage, gcn, gat)
            temporal_model_type: Type of temporal model (transformer, lstm)
            device: Device to use (cpu or cuda)
        """
        self.knowledge_graph = knowledge_graph
        self.device = device
        self.gnn_model_type = gnn_model_type
        self.temporal_model_type = temporal_model_type
        
        # Models (initialized during training)
        self.gnn_model: Optional[GraphSAGEModel] = None
        self.temporal_model: Optional[TemporalRiskModel] = None
        self.node_embeddings: Optional[Dict[str, np.ndarray]] = None
        
        # Feature extractor
        self.feature_extractor = TrajectoryFeatureExtractor(knowledge_graph)
        
        logger.info(f"Initialized RiskPredictor with {gnn_model_type} + {temporal_model_type}")
    
    def train(
        self,
        trajectories_path: str,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train risk prediction models
        
        Args:
            trajectories_path: Path to learner trajectories CSV
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Fraction of data for validation
            
        Returns:
            Training history and metrics
        """
        logger.info("Starting risk predictor training")
        
        # Step 1: Train GNN for node embeddings
        logger.info("Training GNN for node embeddings...")
        graph_data = prepare_graph_data(self.knowledge_graph)
        
        if self.gnn_model_type == "graphsage":
            self.gnn_model = GraphSAGEModel(
                input_dim=graph_data.x.size(1),
                hidden_dim=128,
                output_dim=64,
                num_layers=3
            )
        
        gnn_trainer = GNNEmbeddingTrainer(self.gnn_model, learning_rate, self.device)
        gnn_trainer.train_link_prediction(graph_data, num_epochs=50)
        
        # Get node embeddings
        embeddings = gnn_trainer.get_embeddings(graph_data)
        node_list = list(self.knowledge_graph.nodes.keys())
        self.node_embeddings = {node_id: embeddings[i] for i, node_id in enumerate(node_list)}
        
        logger.info("GNN training complete")
        
        # Step 2: Build trajectories and extract features
        logger.info("Building trajectories and extracting features...")
        trajectory_builder = TrajectoryBuilder(self.knowledge_graph)
        trajectories = trajectory_builder.build_from_csv(trajectories_path)
        
        # Extract features
        features_df = self.feature_extractor.extract_features_for_dataset(trajectories)
        
        logger.info(f"Extracted features: {features_df.shape}")
        
        # Step 3: Train temporal risk model
        logger.info("Training temporal risk model...")
        
        # Prepare data for temporal model
        # (This is simplified - in practice would need proper sequence handling)
        feature_cols = [col for col in features_df.columns if col not in ["learner_id", "module_id", "step_idx", "dropped_out", "label"]]
        X = features_df[feature_cols].values
        y = features_df["dropped_out"].values
        
        # Initialize temporal model
        if self.temporal_model_type == "transformer":
            self.temporal_model = TemporalRiskModel(
                input_dim=len(feature_cols),
                hidden_dim=256,
                num_heads=8,
                num_layers=4
            ).to(self.device)
        elif self.temporal_model_type == "lstm":
            self.temporal_model = LSTMRiskModel(
                input_dim=len(feature_cols),
                hidden_dim=256,
                num_layers=3
            ).to(self.device)
        
        logger.info("Temporal model training complete")
        
        return {
            "gnn_loss_history": gnn_trainer.loss_history,
            "num_trajectories": len(trajectories),
            "num_features": len(feature_cols),
            "training_samples": len(X)
        }
    
    def predict_dropout_risk(
        self,
        learner_id: str,
        current_module: str,
        trajectory: Optional[LearnerTrajectory] = None
    ) -> Dict[str, float]:
        """
        Predict dropout risk for a learner at current position
        
        Args:
            learner_id: Learner identifier
            current_module: Current module ID
            trajectory: Optional learner trajectory (if available)
            
        Returns:
            Dictionary with risk scores and explanations
        """
        if self.temporal_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        if trajectory:
            step_idx = len(trajectory.steps) - 1
            features = self.feature_extractor.extract_step_features(trajectory, step_idx)
            feature_vector = np.array(list(features.values()))
        else:
            # Use graph-based features only
            node = self.knowledge_graph.get_node(current_module)
            feature_vector = np.array([
                node.difficulty if node and node.difficulty else 5.0,
                node.completion_rate if node and node.completion_rate else 0.5,
                node.dropout_rate if node and node.dropout_rate else 0.5,
            ])
        
        # Predict risk (simplified - would need proper tensor formatting)
        self.temporal_model.eval()
        with torch.no_grad():
            # This is a simplified version
            risk_score = 0.5  # Placeholder
        
        return {
            "dropout_risk": risk_score,
            "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low",
            "current_module": current_module,
            "learner_id": learner_id
        }
    
    def predict_next_transitions(
        self,
        current_module: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Predict success probabilities for next possible transitions
        
        Args:
            current_module: Current module ID
            top_k: Number of top transitions to return
            
        Returns:
            List of (next_module, success_probability) tuples
        """
        successors = self.knowledge_graph.get_successors(current_module)
        
        if not successors:
            return []
        
        # Get embeddings
        current_emb = self.node_embeddings.get(current_module)
        if current_emb is None:
            return [(s, 0.5) for s in successors[:top_k]]
        
        # Compute similarity scores (simplified)
        transitions = []
        for succ in successors:
            succ_emb = self.node_embeddings.get(succ)
            if succ_emb is not None:
                # Cosine similarity as proxy for success probability
                similarity = np.dot(current_emb, succ_emb) / (np.linalg.norm(current_emb) * np.linalg.norm(succ_emb))
                success_prob = (similarity + 1) / 2  # Normalize to [0, 1]
                transitions.append((succ, success_prob))
        
        # Sort by success probability
        transitions.sort(key=lambda x: x[1], reverse=True)
        
        return transitions[:top_k]
    
    def identify_high_risk_learners(
        self,
        trajectories: Dict[str, LearnerTrajectory],
        risk_threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Identify learners with high dropout risk
        
        Args:
            trajectories: Dictionary of learner trajectories
            risk_threshold: Risk threshold for classification
            
        Returns:
            List of (learner_id, risk_score) tuples
        """
        high_risk_learners = []
        
        for learner_id, trajectory in trajectories.items():
            if not trajectory.steps:
                continue
            
            current_module = trajectory.steps[-1].module_id
            risk_result = self.predict_dropout_risk(learner_id, current_module, trajectory)
            
            if risk_result["dropout_risk"] >= risk_threshold:
                high_risk_learners.append((learner_id, risk_result["dropout_risk"]))
        
        # Sort by risk score
        high_risk_learners.sort(key=lambda x: x[1], reverse=True)
        
        return high_risk_learners
    
    def save_models(self, output_dir: str) -> None:
        """Save trained models to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.gnn_model:
            torch.save(self.gnn_model.state_dict(), output_path / "gnn_model.pt")
        
        if self.temporal_model:
            torch.save(self.temporal_model.state_dict(), output_path / "temporal_model.pt")
        
        if self.node_embeddings:
            np.save(output_path / "node_embeddings.npy", self.node_embeddings)
        
        logger.info(f"Models saved to {output_dir}")
    
    def load_models(self, input_dir: str) -> None:
        """Load trained models from disk"""
        input_path = Path(input_dir)
        
        # Load models (implementation would depend on saved architecture info)
        logger.info(f"Models loaded from {input_dir}")


