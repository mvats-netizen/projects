"""
Basic Usage Example

Demonstrates basic workflow for the Graph-Native Learner Risk & Intervention Engine.
"""

from src.graph.knowledge_graph import KnowledgeGraphBuilder
from src.trajectories.trajectory_builder import TrajectoryBuilder
from src.models.risk_predictor import RiskPredictor
from src.explainability.bottleneck_detector import BottleneckDetector
from src.explainability.risk_explainer import RiskExplainer
from src.intervention.intervention_engine import InterventionEngine
from src.visualization.graph_visualizer import GraphVisualizer
from src.visualization.risk_visualizer import RiskVisualizer


def main():
    """Main workflow example"""
    
    print("=" * 60)
    print("Graph-Native Learner Risk & Intervention Engine")
    print("=" * 60)
    
    # Step 1: Build Knowledge Graph
    print("\n1. Building Knowledge Graph...")
    builder = KnowledgeGraphBuilder()
    knowledge_graph = builder.build_from_course_data(
        "data/sample_course_structure.json"
    )
    
    stats = knowledge_graph.get_graph_statistics()
    print(f"   - Nodes: {stats['num_nodes']}")
    print(f"   - Edges: {stats['num_edges']}")
    print(f"   - Modules: {stats['num_modules']}")
    print(f"   - Skills: {stats['num_skills']}")
    
    # Step 2: Build Learner Trajectories (requires trajectory data)
    print("\n2. Building Learner Trajectories...")
    # Uncomment when you have trajectory data:
    # trajectory_builder = TrajectoryBuilder(knowledge_graph)
    # trajectories = trajectory_builder.build_from_csv("data/sample_learner_trajectories.csv")
    # print(f"   - Built {len(trajectories)} trajectories")
    print("   - Skipped (requires trajectory data)")
    
    # Step 3: Detect Bottlenecks
    print("\n3. Detecting Bottlenecks...")
    bottleneck_detector = BottleneckDetector(knowledge_graph)
    
    # Note: These require learner data enrichment
    # bottlenecks = bottleneck_detector.detect_node_bottlenecks()
    # print(f"   - Found {len(bottlenecks)} bottleneck nodes")
    print("   - Initialized detector")
    
    # Step 4: Visualize Graph
    print("\n4. Creating Visualizations...")
    visualizer = GraphVisualizer(knowledge_graph)
    
    # Create hierarchical layout visualization
    # Uncomment to generate:
    # fig = visualizer.visualize_graph(
    #     layout="hierarchical",
    #     output_file="outputs/knowledge_graph.html",
    #     show=False
    # )
    print("   - Graph visualizer ready")
    
    # Step 5: Initialize Risk Predictor
    print("\n5. Initializing Risk Predictor...")
    predictor = RiskPredictor(
        knowledge_graph,
        gnn_model_type="graphsage",
        temporal_model_type="transformer"
    )
    print("   - Risk predictor initialized")
    
    # Step 6: Train Models (requires trajectory data)
    print("\n6. Training Models...")
    # Uncomment when you have trajectory data:
    # history = predictor.train(
    #     trajectories_path="data/sample_learner_trajectories.csv",
    #     num_epochs=50
    # )
    # print(f"   - Training complete")
    print("   - Skipped (requires trajectory data)")
    
    # Step 7: Make Predictions (after training)
    print("\n7. Making Predictions...")
    # Example prediction:
    # risk_result = predictor.predict_dropout_risk(
    #     learner_id="L001",
    #     current_module="module_3"
    # )
    # print(f"   - Risk Score: {risk_result['dropout_risk']:.2f}")
    # print(f"   - Risk Level: {risk_result['risk_level']}")
    print("   - Predictor ready (requires trained model)")
    
    # Step 8: Generate Interventions
    print("\n8. Intervention Engine...")
    intervention_engine = InterventionEngine(knowledge_graph)
    print("   - Intervention engine ready")
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Generate or load learner trajectory data")
    print("2. Train the risk prediction models")
    print("3. Generate predictions and interventions")
    print("4. Create visualizations and reports")
    print("\nSee examples/ directory for more detailed workflows.")


if __name__ == "__main__":
    main()


