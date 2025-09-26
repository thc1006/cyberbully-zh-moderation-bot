"""
CyberPuppy 訓練模組
提供完整的模型訓練管理系統
"""

from .config import (
    TrainingPipelineConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    OptimizerConfig,
    ResourceConfig,
    ExperimentConfig,
    ConfigManager,
    get_config
)

# Note: Commented out temporarily due to missing dependencies
# from .trainer import CyberPuppyTrainer, create_trainer_from_config

# from .callbacks import (
#     BaseCallback,
#     EarlyStoppingCallback,
#     ModelCheckpointCallback,
#     GPUMemoryCallback,
#     TensorBoardCallback,
#     MetricsLoggerCallback,
#     ProgressCallback,
#     CallbackManager,
#     create_default_callbacks,
#     TrainingState
# )

# from .utils import (
#     AutoBatchSizeFinder,
#     MemoryOptimizer,
#     GradientAccumulator,
#     WarmupScheduler,
#     TrainingMonitor,
#     ExperimentTracker,
#     set_seed,
#     count_parameters,
#     create_optimizer,
#     estimate_training_time,
#     save_predictions
# )

from .checkpoint_manager import (
    CheckpointManager,
    TrainingMetrics,
    CheckpointInfo
)

__all__ = [
    # Config
    'TrainingPipelineConfig',
    'ModelConfig',
    'DataConfig',
    'TrainingConfig',
    'OptimizerConfig',
    'ResourceConfig',
    'ExperimentConfig',
    'ConfigManager',
    'get_config',

    # Trainer (commented out due to missing dependencies)
    # 'CyberPuppyTrainer',
    # 'create_trainer_from_config',

    # Callbacks (commented out due to missing dependencies)
    # 'BaseCallback',
    # 'EarlyStoppingCallback',
    # 'ModelCheckpointCallback',
    # 'GPUMemoryCallback',
    # 'TensorBoardCallback',
    # 'MetricsLoggerCallback',
    # 'ProgressCallback',
    # 'CallbackManager',
    # 'create_default_callbacks',
    # 'TrainingState',

    # Utils (commented out due to missing dependencies)
    # 'AutoBatchSizeFinder',
    # 'MemoryOptimizer',
    # 'GradientAccumulator',
    # 'WarmupScheduler',
    # 'TrainingMonitor',
    # 'ExperimentTracker',
    # 'set_seed',
    # 'count_parameters',
    # 'create_optimizer',
    # 'estimate_training_time',
    # 'save_predictions',

    # Checkpoint Management
    'CheckpointManager',
    'TrainingMetrics',
    'CheckpointInfo'
]