"""
Evaluation Metrics

Metrics for evaluating risk prediction models.
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc
)
from loguru import logger


class RiskMetrics:
    """
    Compute evaluation metrics for risk prediction
    """
    
    @staticmethod
    def compute_classification_metrics(
        y_true: List[float],
        y_pred: List[float],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute classification metrics
        
        Args:
            y_true: True labels (0/1)
            y_pred: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        y_pred_binary = [1 if p >= threshold else 0 for p in y_pred]
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred_binary),
            "precision": precision_score(y_true, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true, y_pred_binary, zero_division=0),
            "f1_score": f1_score(y_true, y_pred_binary, zero_division=0),
        }
        
        # Add AUC if possible
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
            
            # Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            metrics["pr_auc"] = auc(recall, precision)
        except:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
        
        return metrics
    
    @staticmethod
    def compute_confusion_matrix(
        y_true: List[float],
        y_pred: List[float],
        threshold: float = 0.5
    ) -> Dict[str, int]:
        """
        Compute confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary with TP, TN, FP, FN
        """
        y_pred_binary = [1 if p >= threshold else 0 for p in y_pred]
        
        tp = sum(1 for t, p in zip(y_true, y_pred_binary) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred_binary) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred_binary) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred_binary) if t == 1 and p == 0)
        
        return {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn
        }
    
    @staticmethod
    def compute_calibration_metrics(
        y_true: List[float],
        y_pred: List[float],
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Compute calibration metrics (how well predicted probabilities match reality)
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            Calibration information
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bins) - 1
        
        bin_info = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            
            bin_true = np.array(y_true)[mask]
            bin_pred = np.array(y_pred)[mask]
            
            bin_info.append({
                "bin_idx": i,
                "bin_range": (bins[i], bins[i+1]),
                "count": mask.sum(),
                "mean_predicted": bin_pred.mean(),
                "mean_actual": bin_true.mean(),
                "calibration_error": abs(bin_pred.mean() - bin_true.mean())
            })
        
        # Expected Calibration Error (ECE)
        ece = np.mean([b["calibration_error"] * b["count"] for b in bin_info]) / len(y_true)
        
        return {
            "bin_info": bin_info,
            "expected_calibration_error": ece
        }
    
    @staticmethod
    def compute_intervention_impact(
        baseline_dropout_rate: float,
        post_intervention_dropout_rate: float
    ) -> Dict[str, float]:
        """
        Compute intervention impact metrics
        
        Args:
            baseline_dropout_rate: Dropout rate before intervention
            post_intervention_dropout_rate: Dropout rate after intervention
            
        Returns:
            Impact metrics
        """
        absolute_reduction = baseline_dropout_rate - post_intervention_dropout_rate
        relative_reduction = absolute_reduction / baseline_dropout_rate if baseline_dropout_rate > 0 else 0
        
        return {
            "baseline_dropout_rate": baseline_dropout_rate,
            "post_intervention_dropout_rate": post_intervention_dropout_rate,
            "absolute_reduction": absolute_reduction,
            "relative_reduction": relative_reduction,
            "improvement_percentage": relative_reduction * 100
        }


