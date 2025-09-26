"""
Active Learning Framework for CyberPuppy

This module provides intelligent sample selection strategies for annotation,
combining uncertainty sampling and diversity sampling to maximize labeling efficiency.
"""

from .base import ActiveLearner

# Enhanced uncertainty sampling strategies
from .uncertainty_enhanced import (
    EntropySampling,
    LeastConfidenceSampling,
    MarginSampling,
    BayesianUncertaintySampling,
    BALD
)

# Enhanced diversity sampling strategies
from .diversity_enhanced import (
    ClusteringSampling,
    CoreSetSampling,
    RepresentativeSampling,
    DiversityClusteringHybrid
)

# Hybrid query strategies
from .query_strategies import (
    HybridQueryStrategy,
    AdaptiveQueryStrategy,
    MultiStrategyEnsemble
)

# Main active learner
from .active_learner import CyberPuppyActiveLearner

# Annotation interface
from .annotator import InteractiveAnnotator, BatchAnnotator

# Active learning loop
from .loop import ActiveLearningLoop, BatchActiveLearningLoop

# Visualization tools
from .visualization import ActiveLearningVisualizer

__all__ = [
    # Base class
    'ActiveLearner',

    # Uncertainty sampling
    'EntropySampling',
    'LeastConfidenceSampling',
    'MarginSampling',
    'BayesianUncertaintySampling',
    'BALD',

    # Diversity sampling
    'ClusteringSampling',
    'CoreSetSampling',
    'RepresentativeSampling',
    'DiversityClusteringHybrid',

    # Query strategies
    'HybridQueryStrategy',
    'AdaptiveQueryStrategy',
    'MultiStrategyEnsemble',

    # Main components
    'CyberPuppyActiveLearner',
    'InteractiveAnnotator',
    'BatchAnnotator',
    'ActiveLearningLoop',
    'BatchActiveLearningLoop',

    # Visualization
    'ActiveLearningVisualizer'
]