"""
訓練回調系統
包含早停、檢查點保存、GPU記憶體監控、TensorBoard日誌等
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """訓練狀態"""

    epoch: int = 0
    step: int = 0
    best_metric: float = -float("inf")
    best_epoch: int = 0
    early_stopping_counter: int = 0
    should_stop: bool = False
    metrics_history: List[Dict[str, float]] = None

    def __post_init__(self):
        if self.metrics_history is None:
            self.metrics_history = []


class BaseCallback(ABC):
    """回調基類"""

    @abstractmethod
    def on_train_begin(self, trainer, **kwargs):
        """訓練開始時調用"""
        pass

    @abstractmethod
    def on_train_end(self, trainer, **kwargs):
        """訓練結束時調用"""
        pass

    @abstractmethod
    def on_epoch_begin(self, trainer, **kwargs):
        """每個epoch開始時調用"""
        pass

    @abstractmethod
    def on_epoch_end(self, trainer, **kwargs):
        """每個epoch結束時調用"""
        pass

    @abstractmethod
    def on_step_begin(self, trainer, **kwargs):
        """每個step開始時調用"""
        pass

    @abstractmethod
    def on_step_end(self, trainer, **kwargs):
        """每個step結束時調用"""
        pass


class EarlyStoppingCallback(BaseCallback):
    """早停回調"""

    def __init__(
        self,
        monitor: str = "eval_f1_macro",
        patience: int = 3,
        mode: str = "max",
        min_delta: float = 0.0001,
        restore_best_weights: bool = True,
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_weights = None

    def on_train_begin(self, trainer, **kwargs):
        self.best_score = -float("inf") if self.mode == "max" else float("inf")
        self.counter = 0

    def on_train_end(self, trainer, **kwargs):
        if self.restore_best_weights and self.best_weights is not None:
            trainer.model.load_state_dict(self.best_weights)
            logger.info(f"恢復最佳權重 (score: {self.best_score:.4f})")

    def on_epoch_begin(self, trainer, **kwargs):
        pass

    def on_epoch_end(self, trainer, **kwargs):
        metrics = kwargs.get("metrics", {})
        current_score = metrics.get(self.monitor)

        if current_score is None:
            logger.warning(f"早停監控指標 '{self.monitor}' 未找到")
            return

        improved = False
        if self.mode == "max":
            if current_score > self.best_score + self.min_delta:
                improved = True
        else:
            if current_score < self.best_score - self.min_delta:
                improved = True

        if improved:
            self.best_score = current_score
            self.counter = 0
            trainer.state.best_metric = current_score
            trainer.state.best_epoch = trainer.state.epoch

            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
                }

            logger.info(f"最佳 {self.monitor}: {current_score:.4f}")
        else:
            self.counter += 1
            logger.info(f"早停計數器: {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            trainer.state.should_stop = True
            logger.info(
                f"早停觸發！最佳 {self.monitor}: {self.best_score:.4f} (epoch {trainer.state.best_epoch})"
            )

    def on_step_begin(self, trainer, **kwargs):
        pass

    def on_step_end(self, trainer, **kwargs):
        pass


class ModelCheckpointCallback(BaseCallback):
    """模型檢查點回調"""

    def __init__(
        self,
        dirpath: str,
        filename: str = "checkpoint-{epoch:02d}-{eval_f1_macro:.4f}",
        monitor: str = "eval_f1_macro",
        mode: str = "max",
        save_best_only: bool = True,
        save_top_k: int = 2,
    ):
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_top_k = save_top_k

        self.best_k_models = []
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_train_begin(self, trainer, **kwargs):
        self.best_k_models = []

    def on_train_end(self, trainer, **kwargs):
        pass

    def on_epoch_begin(self, trainer, **kwargs):
        pass

    def on_epoch_end(self, trainer, **kwargs):
        metrics = kwargs.get("metrics", {})
        current_score = metrics.get(self.monitor, 0)

        should_save = not self.save_best_only

        if self.save_best_only:
            if len(self.best_k_models) < self.save_top_k:
                should_save = True
            else:
                worst_score = (
                    min(self.best_k_models, key=lambda x: x[1])[1]
                    if self.mode == "max"
                    else max(self.best_k_models, key=lambda x: x[1])[1]
                )
                if self.mode == "max" and current_score > worst_score:
                    should_save = True
                elif self.mode == "min" and current_score < worst_score:
                    should_save = True

        if should_save:
            # 格式化檔案名
            filename = self.filename.format(
                epoch=trainer.state.epoch,
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            )

            filepath = self.dirpath / f"{filename}.pt"

            # 保存檢查點
            checkpoint = {
                "epoch": trainer.state.epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scheduler_state_dict": (
                    trainer.scheduler.state_dict() if trainer.scheduler else None
                ),
                "metrics": metrics,
                "config": trainer.config.to_dict(),
            }

            torch.save(checkpoint, filepath)
            logger.info(f"檢查點已保存: {filepath}")

            # 更新最佳模型列表
            if self.save_best_only:
                self.best_k_models.append((str(filepath), current_score))
                self.best_k_models.sort(key=lambda x: x[1], reverse=(self.mode == "max"))

                # 刪除多餘的檢查點
                if len(self.best_k_models) > self.save_top_k:
                    to_remove = self.best_k_models[self.save_top_k :]
                    self.best_k_models = self.best_k_models[: self.save_top_k]

                    for filepath, _ in to_remove:
                        Path(filepath).unlink(missing_ok=True)
                        logger.info(f"刪除舊檢查點: {filepath}")

    def on_step_begin(self, trainer, **kwargs):
        pass

    def on_step_end(self, trainer, **kwargs):
        pass


class GPUMemoryCallback(BaseCallback):
    """GPU記憶體監控回調"""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.memory_stats = []

    def on_train_begin(self, trainer, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            logger.info("GPU記憶體監控開始")

    def on_train_end(self, trainer, **kwargs):
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"訓練期間最大GPU記憶體使用: {max_memory:.2f} GB")

    def on_epoch_begin(self, trainer, **kwargs):
        pass

    def on_epoch_end(self, trainer, **kwargs):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"Epoch {trainer.state.epoch} GPU記憶體: 已分配 {allocated:.2f}GB, 已緩存 {cached:.2f}GB"
            )

    def on_step_begin(self, trainer, **kwargs):
        pass

    def on_step_end(self, trainer, **kwargs):
        if trainer.state.step % self.log_interval == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            if allocated > 3.5:  # RTX 3050 4GB 警告閾值
                logger.warning(f"GPU記憶體使用過高: {allocated:.2f}GB/4GB")


class TensorBoardCallback(BaseCallback):
    """TensorBoard記錄回調"""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.writer = None

    def on_train_begin(self, trainer, **kwargs):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        logger.info(f"TensorBoard日誌目錄: {self.log_dir}")

    def on_train_end(self, trainer, **kwargs):
        if self.writer:
            self.writer.close()

    def on_epoch_begin(self, trainer, **kwargs):
        pass

    def on_epoch_end(self, trainer, **kwargs):
        if not self.writer:
            return

        metrics = kwargs.get("metrics", {})
        epoch = trainer.state.epoch

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(name, value, epoch)

        # 記錄學習率
        if trainer.scheduler:
            lr = trainer.scheduler.get_last_lr()[0]
            self.writer.add_scalar("learning_rate", lr, epoch)

    def on_step_begin(self, trainer, **kwargs):
        pass

    def on_step_end(self, trainer, **kwargs):
        pass


class MetricsLoggerCallback(BaseCallback):
    """指標記錄回調"""

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_train_begin(self, trainer, **kwargs):
        # 初始化日誌檔案
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump([], f)

    def on_train_end(self, trainer, **kwargs):
        pass

    def on_epoch_begin(self, trainer, **kwargs):
        pass

    def on_epoch_end(self, trainer, **kwargs):
        metrics = kwargs.get("metrics", {})
        epoch = trainer.state.epoch

        # 讀取現有日誌
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except:
            logs = []

        # 添加新紀錄
        log_entry = {"epoch": epoch, "timestamp": time.time(), **metrics}
        logs.append(log_entry)

        # 保存更新後的日誌
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

    def on_step_begin(self, trainer, **kwargs):
        pass

    def on_step_end(self, trainer, **kwargs):
        pass


class ProgressCallback(BaseCallback):
    """進度顯示回調"""

    def __init__(self, show_memory: bool = True):
        self.show_memory = show_memory
        self.epoch_start_time = None

    def on_train_begin(self, trainer, **kwargs):
        logger.info("訓練開始")

    def on_train_end(self, trainer, **kwargs):
        logger.info("訓練完成")

    def on_epoch_begin(self, trainer, **kwargs):
        self.epoch_start_time = time.time()
        logger.info(f"Epoch {trainer.state.epoch}/{trainer.config.training.num_epochs} 開始")

    def on_epoch_end(self, trainer, **kwargs):
        if self.epoch_start_time:
            duration = time.time() - self.epoch_start_time
            logger.info(f"Epoch {trainer.state.epoch} 完成，耗時: {duration:.2f}秒")

        metrics = kwargs.get("metrics", {})
        if metrics:
            metric_str = " | ".join(
                f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))
            )
            logger.info(f"Epoch {trainer.state.epoch} 指標: {metric_str}")

    def on_step_begin(self, trainer, **kwargs):
        pass

    def on_step_end(self, trainer, **kwargs):
        pass


class CallbackManager:
    """回調管理器"""

    def __init__(self, callbacks: List[BaseCallback] = None):
        self.callbacks = callbacks or []

    def add_callback(self, callback: BaseCallback):
        """添加回調"""
        self.callbacks.append(callback)

    def remove_callback(self, callback_type: type):
        """移除指定類型的回調"""
        self.callbacks = [cb for cb in self.callbacks if not isinstance(cb, callback_type)]

    def on_train_begin(self, trainer, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(trainer, **kwargs)

    def on_train_end(self, trainer, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(trainer, **kwargs)

    def on_epoch_begin(self, trainer, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, **kwargs)

    def on_epoch_end(self, trainer, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, **kwargs)

    def on_step_begin(self, trainer, **kwargs):
        for callback in self.callbacks:
            callback.on_step_begin(trainer, **kwargs)

    def on_step_end(self, trainer, **kwargs):
        for callback in self.callbacks:
            callback.on_step_end(trainer, **kwargs)


def create_default_callbacks(config) -> List[BaseCallback]:
    """建立預設回調列表"""
    callbacks = []

    # 進度回調
    callbacks.append(ProgressCallback())

    # 早停回調
    callbacks.append(
        EarlyStoppingCallback(
            monitor=config.callbacks.early_stopping_metric,
            patience=config.callbacks.early_stopping_patience,
            mode=config.callbacks.early_stopping_mode,
        )
    )

    # 檢查點回調
    exp_dir = config.get_experiment_dir()
    callbacks.append(
        ModelCheckpointCallback(
            dirpath=exp_dir / "checkpoints",
            monitor=config.callbacks.early_stopping_metric,
            save_best_only=config.callbacks.save_best_only,
            save_top_k=config.callbacks.save_top_k,
        )
    )

    # GPU記憶體監控
    if config.callbacks.monitor_gpu_memory and torch.cuda.is_available():
        callbacks.append(GPUMemoryCallback())

    # TensorBoard
    if config.callbacks.tensorboard_log_dir:
        tb_dir = exp_dir / "tensorboard"
        callbacks.append(TensorBoardCallback(tb_dir))

    # 指標記錄
    log_file = exp_dir / "metrics.json"
    callbacks.append(MetricsLoggerCallback(log_file))

    return callbacks
