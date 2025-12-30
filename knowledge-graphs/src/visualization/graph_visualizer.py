"""
Graph Visualization

Visualizes knowledge graphs with risk hotspots and bottlenecks.
"""

from typing import Dict, List, Optional, Any
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from loguru import logger

from ..graph.knowledge_graph import CourseKnowledgeGraph


class GraphVisualizer:
    """
    Visualizes knowledge graphs with interactive plots
    """
    
    def __init__(self, knowledge_graph: CourseKnowledgeGraph):
        """
        Initialize graph visualizer
        
        Args:
            knowledge_graph: Course knowledge graph
        """
        self.knowledge_graph = knowledge_graph
        self.graph = knowledge_graph.graph
    
    def visualize_graph(
        self,
        layout: str = "hierarchical",
        node_color_by: str = "dropout_rate",
        node_size_by: str = "completion_rate",
        output_file: Optional[str] = None,
        show: bool = True
    ) -> go.Figure:
        """
        Create interactive visualization of knowledge graph
        
        Args:
            layout: Layout algorithm (hierarchical, spring, circular)
            node_color_by: Node attribute to use for coloring
            node_size_by: Node attribute to use for sizing
            output_file: Optional output HTML file
            show: Whether to display the figure
            
        Returns:
            Plotly figure object
        """
        logger.info(f"Creating graph visualization with {layout} layout")
        
        # Compute layout positions
        if layout == "hierarchical":
            pos = self._hierarchical_layout()
        elif layout == "spring":
            pos = nx.spring_layout(self.graph, k=2, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        # Extract node and edge data
        edge_traces = self._create_edge_traces(pos)
        node_trace = self._create_node_trace(pos, node_color_by, node_size_by)
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title="Course Knowledge Graph",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800,
                plot_bgcolor='white'
            )
        )
        
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Saved visualization to {output_file}")
        
        if show:
            fig.show()
        
        return fig
    
    def visualize_risk_hotspots(
        self,
        output_file: Optional[str] = None,
        show: bool = True
    ) -> go.Figure:
        """
        Visualize risk hotspots in the graph
        
        Args:
            output_file: Optional output HTML file
            show: Whether to display the figure
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating risk hotspots visualization")
        
        pos = self._hierarchical_layout()
        
        # Create edge traces
        edge_traces = self._create_edge_traces(pos, color='lightgray')
        
        # Create node trace with risk coloring
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        node_hover = []
        
        for node_id, (x, y) in pos.items():
            node = self.knowledge_graph.get_node(node_id)
            if not node:
                continue
            
            node_x.append(x)
            node_y.append(y)
            
            # Color by dropout rate
            dropout_rate = node.dropout_rate if node.dropout_rate else 0.3
            node_colors.append(dropout_rate)
            
            # Size by centrality
            centrality = nx.betweenness_centrality(self.graph).get(node_id, 0.1)
            node_sizes.append(centrality * 100 + 10)
            
            # Text labels
            node_text.append(node.name[:15])
            
            # Hover info
            hover_info = (
                f"<b>{node.name}</b><br>"
                f"Type: {node.node_type}<br>"
                f"Dropout Rate: {dropout_rate:.1%}<br>"
                f"Completion Rate: {node.completion_rate:.1%if node.completion_rate else 'N/A'}<br>"
                f"Difficulty: {node.difficulty if node.difficulty else 'N/A'}"
            )
            node_hover.append(hover_info)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Dropout Rate"),
                line=dict(width=2, color='white')
            ),
            text=node_text,
            textposition="top center",
            hovertext=node_hover,
            hoverinfo='text'
        )
        
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title="Risk Hotspots in Course",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800,
                plot_bgcolor='white'
            )
        )
        
        if output_file:
            fig.write_html(output_file)
        
        if show:
            fig.show()
        
        return fig
    
    def visualize_learner_trajectory(
        self,
        module_sequence: List[str],
        output_file: Optional[str] = None,
        show: bool = True
    ) -> go.Figure:
        """
        Visualize a learner's trajectory through the graph
        
        Args:
            module_sequence: Sequence of module IDs visited
            output_file: Optional output HTML file
            show: Whether to display the figure
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating trajectory visualization")
        
        pos = self._hierarchical_layout()
        
        # Create all edges (faded)
        edge_traces = self._create_edge_traces(pos, color='lightgray', width=1)
        
        # Highlight trajectory path
        traj_x = []
        traj_y = []
        
        for i in range(len(module_sequence) - 1):
            if module_sequence[i] in pos and module_sequence[i+1] in pos:
                x0, y0 = pos[module_sequence[i]]
                x1, y1 = pos[module_sequence[i+1]]
                traj_x.extend([x0, x1, None])
                traj_y.extend([y0, y1, None])
        
        trajectory_trace = go.Scatter(
            x=traj_x,
            y=traj_y,
            mode='lines',
            line=dict(width=4, color='blue'),
            name='Learner Path'
        )
        
        # Create node trace
        node_trace = self._create_simple_node_trace(pos)
        
        fig = go.Figure(
            data=edge_traces + [trajectory_trace, node_trace],
            layout=go.Layout(
                title="Learner Trajectory",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800
            )
        )
        
        if output_file:
            fig.write_html(output_file)
        
        if show:
            fig.show()
        
        return fig
    
    def _hierarchical_layout(self) -> Dict[str, tuple]:
        """Compute hierarchical layout for DAG"""
        try:
            # Try topological sort for DAG
            layers = list(nx.topological_generations(self.graph))
            pos = {}
            
            for layer_idx, layer in enumerate(layers):
                layer_nodes = list(layer)
                for node_idx, node_id in enumerate(layer_nodes):
                    x = node_idx - len(layer_nodes) / 2
                    y = -layer_idx
                    pos[node_id] = (x, y)
            
            return pos
        except:
            # Fallback to spring layout if not a DAG
            return nx.spring_layout(self.graph)
    
    def _create_edge_traces(
        self,
        pos: Dict[str, tuple],
        color: str = 'gray',
        width: float = 1.5
    ) -> List[go.Scatter]:
        """Create edge traces for plotly"""
        edge_traces = []
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=width, color=color),
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
        
        return edge_traces
    
    def _create_node_trace(
        self,
        pos: Dict[str, tuple],
        color_by: str,
        size_by: str
    ) -> go.Scatter:
        """Create node trace with coloring and sizing"""
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        
        for node_id, (x, y) in pos.items():
            node = self.knowledge_graph.get_node(node_id)
            if not node:
                continue
            
            node_x.append(x)
            node_y.append(y)
            
            # Get color value
            color_val = getattr(node, color_by, 0.5)
            node_colors.append(color_val if color_val else 0.5)
            
            # Get size value
            size_val = getattr(node, size_by, 0.5)
            node_sizes.append((size_val if size_val else 0.5) * 30 + 10)
            
            node_text.append(node.name[:15])
        
        return go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                line=dict(width=2, color='white')
            ),
            text=node_text,
            textposition="top center",
            hoverinfo='text'
        )
    
    def _create_simple_node_trace(self, pos: Dict[str, tuple]) -> go.Scatter:
        """Create simple node trace"""
        node_x = [pos[node][0] for node in pos]
        node_y = [pos[node][1] for node in pos]
        
        return go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            marker=dict(size=10, color='lightblue', line=dict(width=2, color='darkblue')),
            hoverinfo='none'
        )


