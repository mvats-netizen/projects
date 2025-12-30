# Quick Start Guide

Get started with the Graph-Native Learner Risk & Intervention Engine in minutes.

## Installation

```bash
# Navigate to the project directory
cd knowledge-graphs

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Generate Sample Data

Before running the engine, generate sample learner trajectory data:

```bash
python examples/generate_sample_data.py
```

This will create `data/sample_learner_trajectories.csv` with synthetic learner data.

## Basic Usage

### 1. Build Knowledge Graph

```python
from src.graph.knowledge_graph import KnowledgeGraphBuilder

# Build graph from course structure
builder = KnowledgeGraphBuilder()
knowledge_graph = builder.build_from_course_data(
    "data/sample_course_structure.json",
    learner_data_path="data/sample_learner_trajectories.csv"
)

# View statistics
stats = knowledge_graph.get_graph_statistics()
print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
```

### 2. Detect Bottlenecks

```python
from src.explainability.bottleneck_detector import BottleneckDetector

detector = BottleneckDetector(knowledge_graph)

# Find bottleneck modules
bottlenecks = detector.detect_node_bottlenecks(
    centrality_threshold=0.5,
    dropout_threshold=0.3
)

# Display top bottlenecks
for bottleneck in bottlenecks[:5]:
    print(f"{bottleneck['name']}: {bottleneck['bottleneck_score']:.2f}")
```

### 3. Build Learner Trajectories

```python
from src.trajectories.trajectory_builder import TrajectoryBuilder

trajectory_builder = TrajectoryBuilder(knowledge_graph)
trajectories = trajectory_builder.build_from_csv(
    "data/sample_learner_trajectories.csv"
)

print(f"Built {len(trajectories)} learner trajectories")
```

### 4. Train Risk Prediction Model

```python
from src.models.risk_predictor import RiskPredictor

predictor = RiskPredictor(
    knowledge_graph,
    gnn_model_type="graphsage",
    temporal_model_type="transformer"
)

# Train models
history = predictor.train(
    trajectories_path="data/sample_learner_trajectories.csv",
    num_epochs=50,
    batch_size=32
)
```

### 5. Predict Dropout Risk

```python
# Predict risk for a specific learner
risk_result = predictor.predict_dropout_risk(
    learner_id="L001",
    current_module="module_3",
    trajectory=trajectories["L001"]
)

print(f"Risk Score: {risk_result['dropout_risk']:.1%}")
print(f"Risk Level: {risk_result['risk_level']}")
```

### 6. Generate Interventions

```python
from src.intervention.intervention_engine import InterventionEngine
from src.explainability.risk_explainer import RiskExplainer

# Explain risk
explainer = RiskExplainer(knowledge_graph, predictor.feature_extractor)
explanation = explainer.explain_risk(
    trajectory=trajectories["L001"],
    risk_score=risk_result['dropout_risk']
)

# Generate interventions
intervention_engine = InterventionEngine(knowledge_graph)
interventions = intervention_engine.recommend_interventions(
    learner_id="L001",
    trajectory=trajectories["L001"],
    risk_score=risk_result['dropout_risk'],
    risk_factors=explanation['contributing_factors'],
    max_interventions=3
)

# Display recommendations
for intervention in interventions:
    print(f"\n{intervention.intervention_type.value}:")
    print(f"  Priority: {intervention.priority}/5")
    print(f"  Expected Impact: {intervention.expected_impact:.1%}")
    print(f"  {intervention.description}")
```

### 7. Visualize Results

```python
from src.visualization.graph_visualizer import GraphVisualizer
from src.visualization.risk_visualizer import RiskVisualizer

# Visualize knowledge graph with risk hotspots
graph_viz = GraphVisualizer(knowledge_graph)
graph_viz.visualize_risk_hotspots(
    output_file="outputs/risk_hotspots.html"
)

# Visualize risk distribution
risk_viz = RiskVisualizer()
risk_scores = {
    learner_id: predictor.predict_dropout_risk(learner_id, "module_3")['dropout_risk']
    for learner_id in list(trajectories.keys())[:100]
}

risk_viz.plot_risk_distribution(
    risk_scores,
    output_file="outputs/risk_distribution.html"
)
```

## Run Complete Example

```bash
# Run basic usage example
python examples/basic_usage.py

# This will:
# 1. Build the knowledge graph
# 2. Initialize all components
# 3. Show you what's possible
```

## Explore with Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/01_explore_knowledge_graph.ipynb
```

## What's Next?

- **Customize**: Modify `config/model_config.yaml` for your use case
- **Real Data**: Replace sample data with actual course and learner data
- **Production**: Deploy the risk predictor as an API service
- **Research**: Experiment with different GNN architectures and temporal models

## Key Configuration

Edit `config/model_config.yaml` to customize:

- GNN architecture (GraphSAGE, GCN, GAT)
- Temporal model (Transformer, LSTM, GRU)
- Risk thresholds
- Intervention strategies
- Training hyperparameters

## Troubleshooting

**Issue**: Module import errors
```bash
# Make sure you're in the right directory
cd knowledge-graphs
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Issue**: Missing dependencies
```bash
pip install -r requirements.txt
```

**Issue**: CUDA/GPU errors (PyTorch Geometric)
```bash
# Install CPU-only versions
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Support

- Check the main `README.md` for detailed documentation
- See `examples/` directory for more code samples
- Review `tests/` for usage patterns

## Performance Tips

1. **Start Small**: Test with a subset of learners first
2. **Use GPU**: If available, models train much faster
3. **Batch Processing**: Process predictions in batches
4. **Cache Embeddings**: Save and reuse node embeddings

---

Ready to predict and prevent learner dropout! 🚀


