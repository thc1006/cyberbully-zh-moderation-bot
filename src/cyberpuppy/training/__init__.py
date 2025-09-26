"""
CyberPuppy 訓練模組
提供完整的模型訓練管理系統
"""

from .config import (
    TrainingMasterConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    OptimizationConfig,
    CallbackConfig,
    ExperimentConfig,
    create_arg_parser,
    load_config_with_overrides
)

from .trainer import CyberPuppyTrainer, create_trainer_from_config

from .callbacks import (
    BaseCallback,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    GPUMemoryCallback,
    TensorBoardCallback,
    MetricsLoggerCallback,
    ProgressCallback,
    CallbackManager,
    create_default_callbacks,
    TrainingState
)

from .utils import (
    AutoBatchSizeFinder,
    MemoryOptimizer,
    GradientAccumulator,
    WarmupScheduler,
    TrainingMonitor,
    ExperimentTracker,
    set_seed,
    count_parameters,
    create_optimizer,
    estimate_training_time,
    save_predictions
)

__all__ = [
    # Config
    'TrainingMasterConfig',
    'ModelConfig',
    'DataConfig',
    'TrainingConfig',
    'OptimizationConfig',
    'CallbackConfig',
    'ExperimentConfig',
    'create_arg_parser',
    'load_config_with_overrides',

    # Trainer
    'CyberPuppyTrainer',
    'create_trainer_from_config',

    # Callbacks
    'BaseCallback',
    'EarlyStoppingCallback',
    'ModelCheckpointCallback',
    'GPUMemoryCallback',
    'TensorBoardCallback',
    'MetricsLoggerCallback',
    'ProgressCallback',
    'CallbackManager',
    'create_default_callbacks',
    'TrainingState',

    # Utils
    'AutoBatchSizeFinder',
    'MemoryOptimizer',
    'GradientAccumulator',
    'WarmupScheduler',
    'TrainingMonitor',
    'ExperimentTracker',
    'set_seed',
    'count_parameters',
    'create_optimizer',
    'estimate_training_time',
    'save_predictions'
]