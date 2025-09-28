"""
Active Learning Framework for CyberPuppy

This module provides intelligent sample selection strategies for annotation,
combining uncertainty sampling and diversity sampling to maximize labeling efficiency.
"""

# Main active learner
from .active_learner import CyberPuppyActiveLearner
# Annotation interface
from .annotator import BatchAnnotator, InteractiveAnnotator
from .base import ActiveLearner
# Enhanced diversity sampling strategies
from .diversity_enhanced import (ClusteringSampling, CoreSetSampling,
                                 DiversityClusteringHybrid,
                                 RepresentativeSampling)
# Active learning loop
from .loop import ActiveLearningLoop, BatchActiveLearningLoop
# Hybrid query strategies
from .query_strategies import (AdaptiveQueryStrategy, HybridQueryStrategy,
                               MultiStrategyEnsemble)
# Enhanced uncertainty sampling strategies
from .uncertainty_enhanced import (BALD, BayesianUncertaintySampling,
                                   EntropySampling, LeastConfidenceSampling,
                                   MarginSampling)
# Visualization tools
from .visualization import ActiveLearningVisualizer

__all__ = [
    # Base class
    "ActiveLearner",
    # Uncertainty sampling
    "EntropySampling",
    "LeastConfidenceSampling",
    "MarginSampling",
    "BayesianUncertaintySampling",
    "BALD",
    # Diversity sampling
    "ClusteringSampling",
    "CoreSetSampling",
    "RepresentativeSampling",
    "DiversityClusteringHybrid",
    # Query strategies
    "HybridQueryStrategy",
    "AdaptiveQueryStrategy",
    "MultiStrategyEnsemble",
    # Main components
    "CyberPuppyActiveLearner",
    "InteractiveAnnotator",
    "BatchAnnotator",
    "ActiveLearningLoop",
    "BatchActiveLearningLoop",
    # Visualization
    "ActiveLearningVisualizer",
]
