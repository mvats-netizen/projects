"""
Risk Visualization

Visualizes risk distributions, predictions, and trends.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from loguru import logger


class RiskVisualizer:
    """
    Visualizes risk predictions and distributions
    """
    
    def plot_risk_distribution(
        self,
        risk_scores: Dict[str, float],
        output_file: Optional[str] = None,
        show: bool = True
    ) -> go.Figure:
        """
        Plot distribution of risk scores across learners
        
        Args:
            risk_scores: Dictionary mapping learner_id to risk score
            output_file: Optional output HTML file
            show: Whether to display the figure
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating risk distribution plot")
        
        scores = list(risk_scores.values())
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=20,
            marker=dict(
                color=scores,
                colorscale='RdYlGn_r',
                showscale=True
            ),
            name='Risk Distribution'
        ))
        
        fig.update_layout(
            title="Learner Dropout Risk Distribution",
            xaxis_title="Risk Score",
            yaxis_title="Number of Learners",
            height=500
        )
        
        # Add threshold lines
        fig.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="High Risk")
        fig.add_vline(x=0.4, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
        
        if output_file:
            fig.write_html(output_file)
        
        if show:
            fig.show()
        
        return fig
    
    def plot_risk_over_time(
        self,
        learner_id: str,
        risk_trajectory: List[float],
        timestamps: Optional[List[str]] = None,
        output_file: Optional[str] = None,
        show: bool = True
    ) -> go.Figure:
        """
        Plot risk score evolution over time for a learner
        
        Args:
            learner_id: Learner identifier
            risk_trajectory: List of risk scores over time
            timestamps: Optional timestamps
            output_file: Optional output HTML file
            show: Whether to display the figure
            
        Returns:
            Plotly figure object
        """
        logger.info(f"Creating risk trajectory plot for {learner_id}")
        
        if timestamps is None:
            timestamps = list(range(len(risk_trajectory)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=risk_trajectory,
            mode='lines+markers',
            name='Risk Score',
            line=dict(width=3),
            marker=dict(size=8)
        ))
        
        # Add risk threshold bands
        fig.add_hrect(y0=0.7, y1=1.0, fillcolor="red", opacity=0.1, line_width=0, annotation_text="High Risk", annotation_position="right")
        fig.add_hrect(y0=0.4, y1=0.7, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="Medium Risk", annotation_position="right")
        fig.add_hrect(y0=0.0, y1=0.4, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Low Risk", annotation_position="right")
        
        fig.update_layout(
            title=f"Risk Trajectory for Learner {learner_id}",
            xaxis_title="Step / Time",
            yaxis_title="Dropout Risk Score",
            yaxis_range=[0, 1],
            height=500
        )
        
        if output_file:
            fig.write_html(output_file)
        
        if show:
            fig.show()
        
        return fig
    
    def plot_module_risk_heatmap(
        self,
        module_risk_data: Dict[str, Dict[str, float]],
        output_file: Optional[str] = None,
        show: bool = True
    ) -> go.Figure:
        """
        Plot heatmap of risk by module and learner segment
        
        Args:
            module_risk_data: Nested dict {module_id: {segment: risk}}
            output_file: Optional output HTML file
            show: Whether to display the figure
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating module risk heatmap")
        
        # Convert to DataFrame
        df = pd.DataFrame(module_risk_data).T
        
        fig = go.Figure(data=go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            colorscale='RdYlGn_r',
            text=df.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Risk Score")
        ))
        
        fig.update_layout(
            title="Dropout Risk by Module and Segment",
            xaxis_title="Learner Segment",
            yaxis_title="Module",
            height=600
        )
        
        if output_file:
            fig.write_html(output_file)
        
        if show:
            fig.show()
        
        return fig
    
    def plot_feature_importance(
        self,
        features: List[str],
        importance_scores: List[float],
        output_file: Optional[str] = None,
        show: bool = True
    ) -> go.Figure:
        """
        Plot feature importance for risk prediction
        
        Args:
            features: List of feature names
            importance_scores: Importance scores
            output_file: Optional output HTML file
            show: Whether to display the figure
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating feature importance plot")
        
        # Sort by importance
        sorted_pairs = sorted(zip(features, importance_scores), key=lambda x: x[1], reverse=True)
        features_sorted, scores_sorted = zip(*sorted_pairs[:20])  # Top 20
        
        fig = go.Figure(go.Bar(
            x=scores_sorted,
            y=features_sorted,
            orientation='h',
            marker=dict(color=scores_sorted, colorscale='Blues')
        ))
        
        fig.update_layout(
            title="Top 20 Features for Risk Prediction",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=600
        )
        
        if output_file:
            fig.write_html(output_file)
        
        if show:
            fig.show()
        
        return fig
    
    def create_dashboard(
        self,
        risk_scores: Dict[str, float],
        bottleneck_nodes: List[Dict[str, Any]],
        interventions_summary: Dict[str, Any],
        output_file: str = "dashboard.html"
    ) -> None:
        """
        Create comprehensive dashboard with multiple visualizations
        
        Args:
            risk_scores: Learner risk scores
            bottleneck_nodes: Bottleneck information
            interventions_summary: Intervention summary statistics
            output_file: Output HTML file
        """
        logger.info("Creating comprehensive dashboard")
        
        from plotly.subplots import make_subplots
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Risk Distribution",
                "Top Bottleneck Modules",
                "Intervention Types",
                "Risk Level Breakdown"
            ),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "pie"}]]
        )
        
        # 1. Risk distribution
        scores = list(risk_scores.values())
        fig.add_trace(
            go.Histogram(x=scores, nbinsx=20, name="Risk Distribution"),
            row=1, col=1
        )
        
        # 2. Top bottlenecks
        if bottleneck_nodes:
            bottleneck_names = [b["name"][:20] for b in bottleneck_nodes[:10]]
            bottleneck_scores = [b["bottleneck_score"] for b in bottleneck_nodes[:10]]
            fig.add_trace(
                go.Bar(x=bottleneck_scores, y=bottleneck_names, orientation='h'),
                row=1, col=2
            )
        
        # 3. Intervention types
        if interventions_summary:
            types = list(interventions_summary.get("intervention_type_distribution", {}).keys())
            counts = list(interventions_summary.get("intervention_type_distribution", {}).values())
            fig.add_trace(
                go.Pie(labels=types, values=counts, name="Interventions"),
                row=2, col=1
            )
        
        # 4. Risk level breakdown
        high_risk = sum(1 for s in scores if s >= 0.7)
        medium_risk = sum(1 for s in scores if 0.4 <= s < 0.7)
        low_risk = sum(1 for s in scores if s < 0.4)
        
        fig.add_trace(
            go.Pie(
                labels=["High Risk", "Medium Risk", "Low Risk"],
                values=[high_risk, medium_risk, low_risk],
                marker=dict(colors=['red', 'orange', 'green'])
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="Learner Risk Analysis Dashboard")
        
        fig.write_html(output_file)
        logger.info(f"Dashboard saved to {output_file}")


