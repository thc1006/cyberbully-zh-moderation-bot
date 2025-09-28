"""
半監督學習模組

包含：
- Pseudo-labeling Pipeline
- Self-training Framework
- Co-training Strategy
- 知識蒸餾機制
"""

from .co_training import CoTrainingConfig, CoTrainingStrategy
from .consistency import ConsistencyConfig, ConsistencyRegularizer
from .memory_optimizer import MemoryOptimizer
from .pseudo_labeling import PseudoLabelConfig, PseudoLabelingPipeline
from .self_training import (SelfTrainingConfig, SelfTrainingFramework,
                            TeacherStudentTrainer)

__all__ = [
    "PseudoLabelingPipeline",
    "PseudoLabelConfig",
    "SelfTrainingFramework",
    "SelfTrainingConfig",
    "TeacherStudentTrainer",
    "CoTrainingStrategy",
    "CoTrainingConfig",
    "ConsistencyRegularizer",
    "ConsistencyConfig",
    "MemoryOptimizer",
]
