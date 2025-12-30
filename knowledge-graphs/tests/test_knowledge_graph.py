"""
Tests for Knowledge Graph Module
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.graph.knowledge_graph import KnowledgeGraphBuilder, CourseKnowledgeGraph
from src.graph.graph_schema import GraphMetadata, ModuleNode, NodeType


class TestKnowledgeGraph:
    """Tests for knowledge graph construction and operations"""
    
    def test_create_empty_graph(self):
        """Test creating an empty knowledge graph"""
        metadata = GraphMetadata(course_id="test-001", course_name="Test Course")
        kg = CourseKnowledgeGraph(metadata)
        
        assert kg.metadata.course_id == "test-001"
        assert len(kg.nodes) == 0
        assert len(kg.edges) == 0
    
    def test_add_node(self):
        """Test adding nodes to graph"""
        metadata = GraphMetadata(course_id="test-001", course_name="Test Course")
        kg = CourseKnowledgeGraph(metadata)
        
        node = ModuleNode(
            node_id="module_1",
            name="Introduction",
            difficulty=3.0
        )
        kg.add_node(node)
        
        assert len(kg.nodes) == 1
        assert "module_1" in kg.nodes
        assert kg.get_node("module_1").name == "Introduction"
    
    def test_get_modules_in_order(self):
        """Test topological ordering of modules"""
        metadata = GraphMetadata(course_id="test-001", course_name="Test Course")
        kg = CourseKnowledgeGraph(metadata)
        
        # Add modules
        for i in range(1, 4):
            node = ModuleNode(
                node_id=f"module_{i}",
                name=f"Module {i}",
                module_order=i
            )
            kg.add_node(node)
        
        modules = kg.get_modules_in_order()
        assert len(modules) == 3
    
    def test_graph_statistics(self):
        """Test graph statistics computation"""
        metadata = GraphMetadata(course_id="test-001", course_name="Test Course")
        kg = CourseKnowledgeGraph(metadata)
        
        # Add a few nodes
        node = ModuleNode(node_id="module_1", name="Module 1")
        kg.add_node(node)
        
        stats = kg.get_graph_statistics()
        assert stats["num_nodes"] == 1
        assert stats["num_modules"] == 1


class TestKnowledgeGraphBuilder:
    """Tests for knowledge graph builder"""
    
    def test_build_from_minimal_json(self):
        """Test building graph from minimal JSON"""
        # Create minimal course structure
        course_data = {
            "course_id": "test-001",
            "course_name": "Test Course",
            "modules": [
                {
                    "module_id": "module_1",
                    "name": "Introduction",
                    "difficulty": 3.0
                }
            ],
            "skills": [],
            "assessments": [],
            "edges": []
        }
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(course_data, f)
            temp_path = f.name
        
        try:
            # Build graph
            builder = KnowledgeGraphBuilder()
            kg = builder.build_from_course_data(temp_path)
            
            assert kg.metadata.course_id == "test-001"
            assert len(kg.nodes) == 1
            assert "module_1" in kg.nodes
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


