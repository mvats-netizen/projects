"""
Shared test fixtures using actual data files.
"""
import json
import os
import pytest
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"


@pytest.fixture(scope="session")
def project_root():
    """Return the project root path."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir():
    """Return the data directory path."""
    return DATA_DIR


@pytest.fixture(scope="session")
def index_dir():
    """Return the index directory path."""
    return INDEX_DIR


@pytest.fixture(scope="session")
def sample_content():
    """Load the actual sample courses content from JSON."""
    content_file = DATA_DIR / "sample_courses_content.json"
    if content_file.exists():
        with open(content_file, 'r') as f:
            return json.load(f)
    return []


@pytest.fixture(scope="session")
def enriched_content():
    """Load the actual enriched content with LLM metadata."""
    enriched_file = DATA_DIR / "sample_courses_enriched.json"
    if enriched_file.exists():
        with open(enriched_file, 'r') as f:
            return json.load(f)
    return []


@pytest.fixture(scope="session")
def coursera_taxonomy():
    """Load the actual Coursera skills taxonomy."""
    taxonomy_file = DATA_DIR / "taxonomy" / "coursera_skills.json"
    if taxonomy_file.exists():
        with open(taxonomy_file, 'r') as f:
            return json.load(f)
    return {"skills": []}


@pytest.fixture(scope="session")
def index_chunks():
    """Load the actual chunks from the FAISS index."""
    chunks_file = INDEX_DIR / "chunks.json"
    if chunks_file.exists():
        with open(chunks_file, 'r') as f:
            return json.load(f)
    return []


@pytest.fixture(scope="session")
def index_config():
    """Load the index configuration."""
    config_file = INDEX_DIR / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    return {}


@pytest.fixture
def sample_transcript():
    """Return a sample transcript text for chunking tests."""
    return """
    In this lecture, we will learn about machine learning fundamentals.
    Machine learning is a subset of artificial intelligence that enables 
    computers to learn from data without being explicitly programmed.
    
    There are three main types of machine learning:
    1. Supervised Learning - where we train models on labeled data
    2. Unsupervised Learning - where models find patterns in unlabeled data  
    3. Reinforcement Learning - where agents learn through trial and error
    
    Today we'll focus on supervised learning algorithms including:
    - Linear Regression for predicting continuous values
    - Logistic Regression for binary classification
    - Decision Trees for interpretable predictions
    - Neural Networks for complex pattern recognition
    
    Let's start by understanding the basic workflow of a supervised learning project.
    First, you need to collect and prepare your data. Data quality is crucial.
    Next, you split your data into training and testing sets.
    Then you train your model on the training data.
    Finally, you evaluate performance on the test set.
    """


@pytest.fixture
def sample_video_item(sample_content):
    """Return a single video item from actual data."""
    for item in sample_content:
        if item.get('content_type') == 'video':
            return item
    return None


@pytest.fixture
def sample_reading_item(sample_content):
    """Return a single reading item from actual data."""
    for item in sample_content:
        if item.get('content_type') == 'reading':
            return item
    return None
