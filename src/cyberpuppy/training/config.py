"""
訓練配置管理系統
提供超參數配置、實驗版本控制和環境配置管理
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


@dataclass
class OptimizerConfig:
    """優化器配置"""

    name: str = "adamw"
    lr: float = 2e-5
    weight_decay: float = 0.01
    eps: float = 1e-8
    betas: tuple = (0.9, 0.999)

    # 學習率調度器
    scheduler_type: str = "cosine"  # cosine, linear, polynomial
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1


@dataclass
class DataConfig:
    """數據配置"""

    max_length: int = 512
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 4
    pin_memory: bool = True

    # 動態批次大小
    dynamic_batch_size: bool = True
    min_batch_size: int = 4
    max_batch_size: int = 32

    # 數據增強
    data_augmentation: bool = True
    augment_ratio: float = 0.3


@dataclass
class ModelConfig:
    """模型配置"""

    model_name: str = "hfl/chinese-macbert-base"
    num_labels: int = 3
    dropout: float = 0.1

    # 多任務配置
    use_multitask: bool = True
    task_weights: Dict[str, float] = field(
        default_factory=lambda: {"toxicity": 1.0, "bullying": 1.0, "emotion": 0.5, "role": 0.5}
    )

    # 模型優化
    gradient_checkpointing: bool = True
    use_attention_dropout: bool = True


@dataclass
class TrainingConfig:
    """訓練配置"""

    num_epochs: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50

    # 早停配置
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    # 混合精度訓練
    fp16: bool = True
    fp16_opt_level: str = "O1"

    # 梯度裁剪
    max_grad_norm: float = 1.0

    # 驗證配置
    validation_strategy: str = "steps"  # steps, epoch
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1_macro"
    greater_is_better: bool = True


@dataclass
class ResourceConfig:
    """資源配置（針對 RTX 3050 4GB 優化）"""

    # GPU 配置
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.9
    allow_growth: bool = True

    # 記憶體優化
    dataloader_pin_memory: bool = False  # RTX 3050 記憶體有限
    empty_cache_steps: int = 100

    # 多 GPU 支援
    use_ddp: bool = False  # 檢測多 GPU 時自動啟用
    local_rank: int = -1

    # CPU 後備
    fallback_to_cpu: bool = True
    cpu_threads: int = 4


@dataclass
class ExperimentConfig:
    """實驗配置"""

    name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # 版本控制
    version: str = "1.0.0"
    base_experiment: Optional[str] = None

    # 追蹤配置
    use_tensorboard: bool = True
    use_mlflow: bool = False
    log_model: bool = True

    # 種子設定
    seed: int = 42
    deterministic: bool = True


@dataclass
class TrainingPipelineConfig:
    """完整訓練管道配置"""

    # 各模組配置
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    # 路徑配置
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self):
        """初始化後處理"""
        # 自動生成實驗名稱
        if not self.experiment.name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment.name = f"exp_{timestamp}"

        # 檢測多 GPU
        if torch.cuda.device_count() > 1:
            self.resources.use_ddp = True

        # 記憶體優化（RTX 3050 4GB）
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024**3)

            if memory_gb <= 4.5:  # RTX 3050 優化
                self.data.batch_size = min(self.data.batch_size, 8)
                self.data.gradient_accumulation_steps = max(
                    self.data.gradient_accumulation_steps, 2
                )
                self.resources.dataloader_pin_memory = False
                self.model.gradient_checkpointing = True

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingPipelineConfig":
        """從字典創建配置"""
        # 遞歸創建嵌套配置對象
        optimizer_config = OptimizerConfig(**config_dict.get("optimizer", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        resources_config = ResourceConfig(**config_dict.get("resources", {}))
        experiment_config = ExperimentConfig(**config_dict.get("experiment", {}))

        # 移除已處理的配置
        main_config = {
            k: v
            for k, v in config_dict.items()
            if k not in ["optimizer", "data", "model", "training", "resources", "experiment"]
        }

        return cls(
            optimizer=optimizer_config,
            data=data_config,
            model=model_config,
            training=training_config,
            resources=resources_config,
            experiment=experiment_config,
            **main_config,
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """保存配置到文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TrainingPipelineConfig":
        """從文件加載配置"""
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def get_experiment_hash(self) -> str:
        """獲取實驗配置的哈希值"""
        # 排除不影響實驗結果的配置
        exclude_keys = {"experiment", "log_dir", "checkpoint_dir"}
        config_for_hash = {k: v for k, v in self.to_dict().items() if k not in exclude_keys}

        config_str = json.dumps(config_for_hash, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]

    def validate(self) -> List[str]:
        """驗證配置的有效性"""
        warnings = []

        # 檢查資源配置
        if self.data.batch_size * self.data.gradient_accumulation_steps < 16:
            warnings.append("有效批次大小小於 16，可能影響訓練穩定性")

        # 檢查學習率
        if self.optimizer.lr > 1e-3:
            warnings.append("學習率過高，建議使用 1e-3 以下")

        # 檢查記憶體配置
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024**3)

            if memory_gb <= 4.5 and self.data.batch_size > 8:
                warnings.append("GPU 記憶體不足，建議降低批次大小")

        return warnings


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # 預設配置模板
        self.templates = {
            "default": TrainingPipelineConfig(),
            "fast_dev": self._create_fast_dev_config(),
            "production": self._create_production_config(),
            "memory_efficient": self._create_memory_efficient_config(),
        }

    def _create_fast_dev_config(self) -> TrainingPipelineConfig:
        """快速開發配置"""
        config = TrainingPipelineConfig()
        config.training.num_epochs = 3
        config.training.save_steps = 100
        config.training.eval_steps = 50
        config.data.batch_size = 8
        config.experiment.name = "fast_dev"
        return config

    def _create_production_config(self) -> TrainingPipelineConfig:
        """生產環境配置"""
        config = TrainingPipelineConfig()
        config.training.num_epochs = 20
        config.training.early_stopping_patience = 5
        config.optimizer.lr = 1e-5
        config.data.batch_size = 16
        config.experiment.name = "production"
        config.experiment.use_mlflow = True
        return config

    def _create_memory_efficient_config(self) -> TrainingPipelineConfig:
        """記憶體優化配置（RTX 3050）"""
        config = TrainingPipelineConfig()
        config.data.batch_size = 4
        config.data.gradient_accumulation_steps = 4
        config.model.gradient_checkpointing = True
        config.resources.dataloader_pin_memory = False
        config.resources.empty_cache_steps = 50
        config.training.fp16 = True
        config.experiment.name = "memory_efficient"
        return config

    def get_template(self, template_name: str) -> TrainingPipelineConfig:
        """獲取配置模板"""
        if template_name not in self.templates:
            available = list(self.templates.keys())
            raise ValueError(f"未知模板 {template_name}，可用模板: {available}")
        return self.templates[template_name]

    def save_template(self, name: str, config: TrainingPipelineConfig):
        """保存自定義模板"""
        self.templates[name] = config
        template_path = self.config_dir / f"template_{name}.json"
        config.save(template_path)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """列出所有實驗配置"""
        experiments = []

        for config_file in self.config_dir.glob("exp_*.json"):
            try:
                config = TrainingPipelineConfig.load(config_file)
                experiments.append(
                    {
                        "name": config.experiment.name,
                        "version": config.experiment.version,
                        "description": config.experiment.description,
                        "tags": config.experiment.tags,
                        "file": str(config_file),
                        "hash": config.get_experiment_hash(),
                    }
                )
            except Exception as e:
                print(f"無法加載配置文件 {config_file}: {e}")

        return sorted(experiments, key=lambda x: x["name"])

    def create_experiment(
        self, base_template: str = "default", name: Optional[str] = None, **overrides
    ) -> TrainingPipelineConfig:
        """創建新實驗配置"""
        config = self.get_template(base_template)

        # 應用覆蓋參數
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # 支持嵌套屬性設置，如 optimizer.lr
                parts = key.split(".")
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)

        # 設置實驗名稱
        if name:
            config.experiment.name = name

        # 保存配置
        config_path = self.config_dir / f"exp_{config.experiment.name}.json"
        config.save(config_path)

        return config


# 全局配置實例
config_manager = ConfigManager()


def get_config(template_name: str = "default") -> TrainingPipelineConfig:
    """快速獲取配置"""
    return config_manager.get_template(template_name)
