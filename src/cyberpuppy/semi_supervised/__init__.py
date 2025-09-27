"""
半監督學習模組

包含：
- Pseudo-labeling Pipeline
- Self-training Framework
- Co-training Strategy
- 知識蒸餾機制
"""

from .pseudo_labeling import PseudoLabelingPipeline, PseudoLabelConfig
from .self_training import SelfTrainingFramework, TeacherStudentTrainer, SelfTrainingConfig
from .co_training import CoTrainingStrategy, CoTrainingConfig
from .consistency import ConsistencyRegularizer, ConsistencyConfig
from .memory_optimizer import MemoryOptimizer

__all__ = [
    'PseudoLabelingPipeline',
    'PseudoLabelConfig',
    'SelfTrainingFramework',
    'SelfTrainingConfig',
    'TeacherStudentTrainer',
    'CoTrainingStrategy',
    'CoTrainingConfig',
    'ConsistencyRegularizer',
    'ConsistencyConfig',
    'MemoryOptimizer'
]