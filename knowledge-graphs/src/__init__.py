"""
Graph-Native Learner Risk & Intervention Engine

A knowledge graph-based ML system for predicting learner dropout and recommending interventions.
"""

__version__ = "0.1.0"
__author__ = "Muskan Vats"

from . import graph
from . import trajectories
from . import models
from . import explainability
from . import intervention
from . import visualization
from . import utils

__all__ = [
    "graph",
    "trajectories",
    "models",
    "explainability",
    "intervention",
    "visualization",
    "utils",
]


