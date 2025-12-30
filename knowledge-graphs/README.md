# Graph-Native Learner Risk & Intervention Engine

A knowledge graph-based machine learning system to predict learner dropout risk, identify structural bottlenecks in courses, and recommend personalized interventions.

## 🎯 Overview

This engine uses course knowledge graphs (modules, quizzes, skills, prerequisites) and learner trajectories to:
- **Predict** who is likely to drop out and when
- **Identify** why dropout risk increases (skill difficulty, transition complexity)
- **Recommend** targeted interventions for learners
- **Suggest** course redesign opportunities for instructors

## 📊 Architecture

```
Knowledge Graph Construction → Learner Trajectories → GNN Embeddings 
    ↓                              ↓                      ↓
Node/Edge Metadata          Temporal Features      Graph Representations
    ↓                              ↓                      ↓
    └──────────────────────────────┴──────────────────────┘
                                   ↓
                      Temporal Risk Model (Transformer/RNN)
                                   ↓
                    ┌──────────────┴──────────────┐
                    ↓                              ↓
          Explainability Layer          Intervention Engine
          (Bottlenecks, Risks)         (Recommendations)
```

## 🚀 Quick Start

### Installation

```bash
cd knowledge-graphs
pip install -r requirements.txt
```

### Basic Usage

```python
from src.graph.knowledge_graph import KnowledgeGraphBuilder
from src.models.risk_predictor import RiskPredictor

# Build knowledge graph
builder = KnowledgeGraphBuilder()
graph = builder.build_from_course_data("data/course_structure.json")

# Train risk model
predictor = RiskPredictor(graph)
predictor.train("data/learner_trajectories.csv")

# Predict risk for a learner
risk_score = predictor.predict_dropout_risk(learner_id, current_module)
```

## 📁 Project Structure

```
knowledge-graphs/
├── src/
│   ├── graph/              # Knowledge graph construction
│   ├── trajectories/       # Learner trajectory modeling
│   ├── models/             # ML models (GNN, risk prediction)
│   ├── explainability/     # Explainability & bottleneck detection
│   ├── intervention/       # Intervention recommendation engine
│   ├── visualization/      # Graph visualization utilities
│   └── utils/              # Common utilities
├── data/                   # Sample data and schemas
├── notebooks/              # Jupyter notebooks for exploration
├── tests/                  # Unit tests
├── config/                 # Configuration files
└── requirements.txt        # Python dependencies
```

## 🛠️ Core Components

### 1. Knowledge Graph Construction (`src/graph/`)
- Extract modules, assessments, skills from course data
- Build directed graph with prerequisite relationships
- Add metadata: difficulty, completion rates, average scores

### 2. Learner Trajectory Modeling (`src/trajectories/`)
- Track learner paths through the graph
- Capture performance, time spent, retries, inactivity
- Generate temporal sequences for ML

### 3. GNN Node Embeddings (`src/models/gnn_embeddings.py`)
- GraphSAGE, GCN, or GAT architectures
- Learn skill embeddings, module embeddings, transition embeddings
- Capture structural relationships and difficulty

### 4. Temporal Risk Model (`src/models/risk_predictor.py`)
- Transformer or RNN-based temporal model
- Predicts dropout probability at each node
- Risk-per-edge for each possible transition

### 5. Explainability Layer (`src/explainability/`)
- Identifies highest-risk nodes and transitions
- Detects skill-level bottlenecks
- Common failure pattern analysis
- SHAP-based feature importance

### 6. Intervention Engine (`src/intervention/`)
- Recommends remedial content insertion
- Flags course redesign opportunities
- Generates targeted nudges at high-risk moments

## 📊 Data Schema

### Course Structure
```json
{
  "course_id": "ml-fundamentals-101",
  "modules": [
    {
      "module_id": "module_1",
      "name": "Introduction to ML",
      "skills": ["python", "statistics"],
      "difficulty": 2.5,
      "prerequisites": []
    }
  ],
  "assessments": [...],
  "skill_graph": {...}
}
```

### Learner Trajectories
```csv
learner_id,module_id,timestamp,score,time_spent,attempts,completed,dropped_out
L001,module_1,2024-01-01,85,3600,1,true,false
L001,module_2,2024-01-02,72,4200,2,true,false
```

## 🧪 Phase Implementation (6 Weeks)

### Week 1-2: Graph Construction & Data Exploration
- [x] Build course knowledge graph structure
- [x] Map learner trajectories
- [ ] Data validation and quality checks

### Week 3-4: GNN Node Embeddings
- [ ] Train GraphSAGE/GCN model
- [ ] Validate embeddings (clustering, similarity)
- [ ] Visualize learned representations

### Week 5-6: Risk Model + Explanations
- [ ] Train temporal dropout prediction model
- [ ] Generate risk curves per module
- [ ] Explain high-risk nodes
- [ ] Build intervention recommendations

## 📈 Expected Outputs

### For Product & Enterprise
- Module-level dropout analysis
- Bottleneck skill identification
- Intervention recommendations with expected impact

### For Instructors
- Visual graph with hotspot highlighting
- Evidence-backed redesign suggestions
- Comparative analysis across cohorts

### For Platform
- Real-time risk scoring API
- Batch prediction pipeline
- A/B testing framework for interventions

## 🔬 Technologies

- **Graph Processing**: NetworkX, PyTorch Geometric
- **Deep Learning**: PyTorch, Transformers
- **Explainability**: SHAP, Captum
- **Visualization**: Plotly, NetworkX, Graphviz
- **Data**: Pandas, NumPy
- **Orchestration**: MLflow (optional)

## 📝 Development

### Running Tests
```bash
pytest tests/
```

### Training Models
```bash
python -m src.train --config config/training_config.yaml
```

### Generating Reports
```bash
python -m src.analyze --course-id ml-101 --output reports/
```

## 🎓 Research Extensions

- Adaptive learning pathways
- GNKT (Graph Neural Knowledge Tracing)
- Skill mastery modeling
- Organization-level skill gap simulation
- AI-augmented course redesign

## 📄 License

Internal R&D project.

## 👥 Contributors

Muskan Vats - AI Specialist


