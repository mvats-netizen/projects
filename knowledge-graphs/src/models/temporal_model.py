"""
Temporal Risk Prediction Model

Transformer and RNN-based models for predicting dropout risk over time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from loguru import logger


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalRiskModel(nn.Module):
    """
    Transformer-based temporal model for dropout risk prediction
    
    Predicts dropout probability at each step in the learner trajectory.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.2,
        max_seq_length: int = 100,
        output_dim: int = 1  # Dropout probability
    ):
        """
        Initialize Temporal Risk Model
        
        Args:
            input_dim: Input feature dimensionality
            hidden_dim: Hidden layer dimensionality
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            output_dim: Output dimensionality (1 for binary classification)
        """
        super(TemporalRiskModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        logger.info(f"Initialized Temporal Risk Model with {num_layers} transformer layers")
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features [batch_size, seq_length, input_dim]
            mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Risk predictions [batch_size, seq_length, output_dim]
        """
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        if mask is not None:
            # Convert mask to attention mask format
            mask = mask.bool()
            mask = ~mask  # Invert: True = masked position
        
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Output projection
        output = self.output_projection(x)
        
        return output
    
    def predict_risk(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict dropout risk (probability)
        
        Args:
            x: Input features
            mask: Attention mask
            
        Returns:
            Dropout probabilities [batch_size, seq_length]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, mask)
            probs = torch.sigmoid(logits).squeeze(-1)
        return probs


class LSTMRiskModel(nn.Module):
    """
    LSTM-based temporal model for dropout risk prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        output_dim: int = 1
    ):
        """
        Initialize LSTM Risk Model
        
        Args:
            input_dim: Input feature dimensionality
            hidden_dim: Hidden state dimensionality
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_dim: Output dimensionality
        """
        super(LSTMRiskModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        logger.info(f"Initialized LSTM Risk Model with {num_layers} layers")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features [batch_size, seq_length, input_dim]
            
        Returns:
            Risk predictions [batch_size, seq_length, output_dim]
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Output projection
        output = self.output_projection(lstm_out)
        
        return output


class EdgeTransitionPredictor(nn.Module):
    """
    Predicts success probability for transitions between nodes (edges)
    """
    
    def __init__(
        self,
        node_embedding_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        """
        Initialize Edge Transition Predictor
        
        Args:
            node_embedding_dim: Dimensionality of node embeddings
            hidden_dim: Hidden layer dimensionality
            dropout: Dropout rate
        """
        super(EdgeTransitionPredictor, self).__init__()
        
        # Edge feature extraction
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        logger.info("Initialized Edge Transition Predictor")
    
    def forward(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict transition success probability
        
        Args:
            source_embeddings: Source node embeddings [batch_size, embedding_dim]
            target_embeddings: Target node embeddings [batch_size, embedding_dim]
            
        Returns:
            Transition success probabilities [batch_size, 1]
        """
        # Concatenate source and target embeddings
        edge_features = torch.cat([source_embeddings, target_embeddings], dim=-1)
        
        # Predict transition success
        logits = self.edge_mlp(edge_features)
        probs = torch.sigmoid(logits)
        
        return probs


class TemporalRiskTrainer:
    """
    Trainer for temporal risk models
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: str = "cpu",
        use_class_weights: bool = True
    ):
        """
        Initialize trainer
        
        Args:
            model: Temporal risk model
            learning_rate: Learning rate
            device: Device to train on
            use_class_weights: Whether to use class weights for imbalanced data
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.use_class_weights = use_class_weights
        self.loss_history = []
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        class_weight: Optional[torch.Tensor] = None
    ) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            class_weight: Optional class weights for loss
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move to device
            x = batch["features"].to(self.device)
            y = batch["labels"].to(self.device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(x, mask).squeeze(-1)
            
            # Compute loss
            if self.use_class_weights and class_weight is not None:
                criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
            else:
                criterion = nn.BCEWithLogitsLoss()
            
            loss = criterion(logits, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.loss_history.append(avg_loss)
        
        return avg_loss
    
    def evaluate(
        self,
        eval_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float, float]:
        """
        Evaluate model
        
        Args:
            eval_loader: Evaluation data loader
            
        Returns:
            Tuple of (loss, accuracy, auc)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        num_batches = 0
        
        criterion = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for batch in eval_loader:
                x = batch["features"].to(self.device)
                y = batch["labels"].to(self.device)
                mask = batch.get("mask", None)
                if mask is not None:
                    mask = mask.to(self.device)
                
                logits = self.model(x, mask).squeeze(-1)
                loss = criterion(logits, y)
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(y.cpu().numpy().flatten())
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Compute accuracy
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
        
        return avg_loss, accuracy, 0.0  # AUC computation would require sklearn


