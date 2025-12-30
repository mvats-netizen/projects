"""
Graph Neural Network Models for Node Embeddings

Implements GraphSAGE, GCN, and GAT for learning node representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch_geometric.data import Data
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger

from ..graph.knowledge_graph import CourseKnowledgeGraph


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE model for learning node embeddings
    
    Aggregates information from neighbor nodes to learn representations.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        aggregator: str = "mean"
    ):
        """
        Initialize GraphSAGE model
        
        Args:
            input_dim: Input feature dimensionality
            hidden_dim: Hidden layer dimensionality
            output_dim: Output embedding dimensionality
            num_layers: Number of GraphSAGE layers
            dropout: Dropout rate
            aggregator: Aggregation function (mean, max, lstm)
        """
        super(GraphSAGEModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggregator))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, output_dim, aggr=aggregator))
        
        logger.info(f"Initialized GraphSAGE with {num_layers} layers")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer (no activation)
        x = self.convs[-1](x, edge_index)
        
        return x
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
        """
        Get node embeddings as numpy array
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Node embeddings as numpy array
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(x, edge_index)
        return embeddings.cpu().numpy()


class GCNModel(nn.Module):
    """
    Graph Convolutional Network (GCN) for learning node embeddings
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        """
        Initialize GCN model
        
        Args:
            input_dim: Input feature dimensionality
            hidden_dim: Hidden layer dimensionality
            output_dim: Output embedding dimensionality
            num_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super(GCNModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        logger.info(f"Initialized GCN with {num_layers} layers")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x


class GATModel(nn.Module):
    """
    Graph Attention Network (GAT) for learning node embeddings
    
    Uses attention mechanism to weight neighbor contributions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        """
        Initialize GAT model
        
        Args:
            input_dim: Input feature dimensionality
            hidden_dim: Hidden layer dimensionality
            output_dim: Output embedding dimensionality
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(GATModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
        
        # Output layer (single head)
        self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False, dropout=dropout))
        
        logger.info(f"Initialized GAT with {num_layers} layers and {num_heads} heads")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x


class GNNEmbeddingTrainer:
    """
    Trainer for GNN embedding models
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: str = "cpu"
    ):
        """
        Initialize trainer
        
        Args:
            model: GNN model to train
            learning_rate: Learning rate for optimizer
            device: Device to train on (cpu or cuda)
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_history = []
    
    def train_link_prediction(
        self,
        data: Data,
        num_epochs: int = 100,
        negative_sampling_ratio: float = 1.0
    ) -> List[float]:
        """
        Train model using link prediction task
        
        Args:
            data: PyTorch Geometric Data object
            num_epochs: Number of training epochs
            negative_sampling_ratio: Ratio of negative to positive samples
            
        Returns:
            List of training losses
        """
        self.model.train()
        data = data.to(self.device)
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.model(data.x, data.edge_index)
            
            # Link prediction loss (dot product similarity)
            pos_edge_index = data.edge_index
            neg_edge_index = self._negative_sampling(data, negative_sampling_ratio)
            
            # Positive edges (should have high similarity)
            pos_src = embeddings[pos_edge_index[0]]
            pos_dst = embeddings[pos_edge_index[1]]
            pos_scores = (pos_src * pos_dst).sum(dim=1)
            pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-15).mean()
            
            # Negative edges (should have low similarity)
            neg_src = embeddings[neg_edge_index[0]]
            neg_dst = embeddings[neg_edge_index[1]]
            neg_scores = (neg_src * neg_dst).sum(dim=1)
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-15).mean()
            
            loss = pos_loss + neg_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            self.loss_history.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        
        return self.loss_history
    
    def _negative_sampling(self, data: Data, ratio: float) -> torch.Tensor:
        """Generate negative edge samples"""
        num_nodes = data.x.size(0)
        num_neg_samples = int(data.edge_index.size(1) * ratio)
        
        # Randomly sample node pairs
        neg_edges = torch.randint(0, num_nodes, (2, num_neg_samples), device=self.device)
        
        return neg_edges
    
    def get_embeddings(self, data: Data) -> np.ndarray:
        """Get trained embeddings"""
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(data.x, data.edge_index)
        
        return embeddings.cpu().numpy()


def prepare_graph_data(
    knowledge_graph: CourseKnowledgeGraph,
    feature_dim: int = 32
) -> Data:
    """
    Prepare PyTorch Geometric Data object from knowledge graph
    
    Args:
        knowledge_graph: Course knowledge graph
        feature_dim: Dimension of initial node features
        
    Returns:
        PyTorch Geometric Data object
    """
    # Get node list
    node_list = list(knowledge_graph.nodes.keys())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    
    # Create initial node features (one-hot + metadata)
    num_nodes = len(node_list)
    node_features = []
    
    for node_id in node_list:
        node = knowledge_graph.nodes[node_id]
        
        # Simple feature vector: [difficulty, completion_rate, avg_score, dropout_rate]
        features = [
            node.difficulty if node.difficulty else 5.0,
            node.completion_rate if node.completion_rate else 0.5,
            node.average_score if node.average_score else 50.0,
            node.dropout_rate if node.dropout_rate else 0.5,
        ]
        
        # Pad or truncate to feature_dim
        while len(features) < feature_dim:
            features.append(0.0)
        features = features[:feature_dim]
        
        node_features.append(features)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Create edge index
    edge_list = []
    for edge in knowledge_graph.edges:
        src_idx = node_to_idx[edge.source]
        dst_idx = node_to_idx[edge.target]
        edge_list.append([src_idx, dst_idx])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    data = Data(x=x, edge_index=edge_index)
    
    logger.info(f"Prepared graph data: {num_nodes} nodes, {len(edge_list)} edges")
    
    return data


