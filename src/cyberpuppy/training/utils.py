"""
訓練工具模組
包含自動batch size finder、記憶體優化、梯度累積等工具
"""

import gc
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class AutoBatchSizeFinder:
    """自動尋找最佳batch size"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_memory_gb: float = 3.5,  # RTX 3050 安全閾值
        growth_factor: float = 1.5,
        max_trials: int = 10,
    ):
        self.model = model
        self.device = device
        self.max_memory_gb = max_memory_gb
        self.growth_factor = growth_factor
        self.max_trials = max_trials

    def find_batch_size(self, dataloader: DataLoader, start_batch_size: int = 1) -> int:
        """
        自動尋找最佳batch size

        Args:
            dataloader: 資料載入器
            start_batch_size: 起始batch size

        Returns:
            最佳batch size
        """
        logger.info("開始自動尋找最佳batch size...")

        # 獲取一個樣本批次
        sample_batch = next(iter(dataloader))

        # 移動到設備
        if isinstance(sample_batch, (list, tuple)):
            sample_inputs = [x.to(self.device) if torch.is_tensor(x) else x for x in sample_batch]
        elif isinstance(sample_batch, dict):
            sample_inputs = {
                k: v.to(self.device) if torch.is_tensor(v) else v for k, v in sample_batch.items()
            }
        else:
            sample_inputs = sample_batch.to(self.device)

        current_batch_size = start_batch_size
        best_batch_size = start_batch_size

        for _trial in range(self.max_trials):
            try:
                # 清理記憶體
                torch.cuda.empty_cache()
                gc.collect()

                # 調整batch size
                test_inputs = self._scale_batch(sample_inputs, current_batch_size)

                # 測試前向傳播
                self.model.eval()
                with torch.no_grad():
                    _ = self.model(**test_inputs if isinstance(test_inputs, dict) else test_inputs)

                # 檢查記憶體使用
                memory_used = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"Batch size {current_batch_size}: 記憶體使用 {memory_used:.2f}GB")

                if memory_used > self.max_memory_gb:
                    logger.info(f"記憶體超出限制，使用batch size: {best_batch_size}")
                    break

                best_batch_size = current_batch_size
                current_batch_size = int(current_batch_size * self.growth_factor)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.info(f"OOM at batch size {current_batch_size}, 使用: {best_batch_size}")
                    break
                else:
                    raise e

        logger.info(f"找到最佳batch size: {best_batch_size}")
        return best_batch_size

    def _scale_batch(self, inputs, target_batch_size: int):
        """調整批次大小"""
        if isinstance(inputs, dict):
            scaled = {}
            for key, value in inputs.items():
                if torch.is_tensor(value) and value.dim() > 0:
                    # 重複到目標batch size
                    current_batch_size = value.size(0)
                    repeat_times = max(1, target_batch_size // current_batch_size)
                    remainder = target_batch_size % current_batch_size

                    repeated = value.repeat(repeat_times, *([1] * (value.dim() - 1)))
                    if remainder > 0:
                        repeated = torch.cat([repeated, value[:remainder]], dim=0)
                    scaled[key] = repeated
                else:
                    scaled[key] = value
            return scaled
        elif torch.is_tensor(inputs):
            current_batch_size = inputs.size(0)
            repeat_times = max(1, target_batch_size // current_batch_size)
            remainder = target_batch_size % current_batch_size

            repeated = inputs.repeat(repeat_times, *([1] * (inputs.dim() - 1)))
            if remainder > 0:
                repeated = torch.cat([repeated, inputs[:remainder]], dim=0)
            return repeated
        else:
            return inputs


class MemoryOptimizer:
    """記憶體優化工具"""

    @staticmethod
    def enable_gradient_checkpointing(model: nn.Module):
        """啟用梯度檢查點"""
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("梯度檢查點已啟用")
        else:
            logger.warning("模型不支援梯度檢查點")

    @staticmethod
    def setup_mixed_precision() -> torch.cuda.amp.GradScaler:
        """設定混合精度訓練"""
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
            logger.info("混合精度訓練已啟用")
            return scaler
        else:
            logger.warning("CUDA不可用，無法啟用混合精度")
            return None

    @staticmethod
    def optimize_dataloader(
        dataloader: DataLoader,
        pin_memory: bool = True,
        num_workers: int = 2,
        prefetch_factor: int = 2,
    ) -> DataLoader:
        """優化DataLoader設定"""
        # 重新建立優化後的DataLoader
        optimized_loader = DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=dataloader.shuffle if hasattr(dataloader, "shuffle") else False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor,
            drop_last=dataloader.drop_last,
        )

        logger.info(f"DataLoader已優化: workers={num_workers}, pin_memory={pin_memory}")
        return optimized_loader

    @staticmethod
    def clear_memory():
        """清理記憶體"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """獲取記憶體統計"""
        stats = {}

        # CPU記憶體
        cpu_memory = psutil.virtual_memory()
        stats["cpu_memory_used_gb"] = cpu_memory.used / (1024**3)
        stats["cpu_memory_percent"] = cpu_memory.percent

        # GPU記憶體
        if torch.cuda.is_available():
            stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            stats["gpu_memory_max_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)

        return stats


class GradientAccumulator:
    """梯度累積工具"""

    def __init__(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def should_step(self) -> bool:
        """是否應該執行優化步驟"""
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """縮放損失"""
        return loss / self.accumulation_steps

    def reset(self):
        """重置計數器"""
        self.current_step = 0


class WarmupScheduler:
    """學習率預熱排程器"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        scheduler_type: str = "cosine",
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.scheduler_type = scheduler_type
        self.current_step = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        """執行一步學習率更新"""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # 預熱階段
            warmup_factor = self.current_step / self.warmup_steps
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.base_lrs[i] * warmup_factor
        else:
            # 主要排程階段
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )

            if self.scheduler_type == "cosine":
                factor = 0.5 * (1 + math.cos(math.pi * progress))
            elif self.scheduler_type == "linear":
                factor = 1 - progress
            else:
                factor = 1

            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.base_lrs[i] * factor

    def get_last_lr(self):
        """獲取當前學習率"""
        return [group["lr"] for group in self.optimizer.param_groups]


class TrainingMonitor:
    """訓練監控工具"""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_times = []
        self.losses = []
        self.start_time = None

    def start_epoch(self):
        """開始一個epoch"""
        self.start_time = time.time()
        self.step_times = []
        self.losses = []

    def log_step(self, step: int, loss: float, lr: float):
        """記錄一個步驟"""
        step_time = time.time()
        if len(self.step_times) > 0:
            time_per_step = step_time - self.step_times[-1]
        else:
            time_per_step = 0

        self.step_times.append(step_time)
        self.losses.append(loss)

        if step % self.log_interval == 0:
            avg_loss = np.mean(self.losses[-self.log_interval :])
            memory_stats = MemoryOptimizer.get_memory_stats()

            logger.info(
                f"Step {step} | Loss: {loss:.4f} | Avg Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | Time/Step: {time_per_step:.3f}s | "
                f"GPU Mem: {memory_stats.get('gpu_memory_allocated_gb', 0):.2f}GB"
            )

    def end_epoch(self, epoch: int) -> Dict[str, float]:
        """結束一個epoch並返回統計"""
        if self.start_time is None:
            return {}

        epoch_time = time.time() - self.start_time
        avg_loss = np.mean(self.losses) if self.losses else 0
        steps_per_second = len(self.losses) / epoch_time if epoch_time > 0 else 0

        stats = {
            "epoch_time": epoch_time,
            "avg_loss": avg_loss,
            "steps_per_second": steps_per_second,
        }

        logger.info(f"Epoch {epoch} 統計: {stats}")
        return stats


class ExperimentTracker:
    """實驗追蹤工具"""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.experiment_dir / "metrics.json"
        self.config_file = self.experiment_dir / "config.json"

    def log_config(self, config: Dict[str, Any]):
        """記錄配置"""
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """記錄指標"""
        # 讀取現有指標
        if self.metrics_file.exists():
            with open(self.metrics_file, "r", encoding="utf-8") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []

        # 添加新指標
        metric_entry = {"epoch": epoch, "timestamp": time.time(), **metrics}
        all_metrics.append(metric_entry)

        # 保存
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    def save_model(self, model: nn.Module, filename: str = "final_model.pt"):
        """保存模型"""
        model_path = self.experiment_dir / filename
        torch.save(model.state_dict(), model_path)
        logger.info(f"模型已保存到: {model_path}")

    def load_model(self, model: nn.Module, filename: str = "final_model.pt"):
        """載入模型"""
        model_path = self.experiment_dir / filename
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            logger.info(f"模型已從 {model_path} 載入")
        else:
            logger.warning(f"模型檔案不存在: {model_path}")


def set_seed(seed: int):
    """設定隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"隨機種子設定為: {seed}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """計算模型參數數量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def estimate_training_time(
    num_samples: int, batch_size: int, num_epochs: int, seconds_per_batch: float = 0.5
) -> str:
    """估算訓練時間"""
    total_batches = (num_samples // batch_size) * num_epochs
    total_seconds = total_batches * seconds_per_batch

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)

    return f"{hours}小時{minutes}分鐘"


def create_optimizer(
    model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    """建立優化器"""
    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8
        )
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"不支援的優化器: {optimizer_name}")

    logger.info(f"建立優化器: {optimizer_name}, LR: {learning_rate}, Weight Decay: {weight_decay}")
    return optimizer


def save_predictions(
    predictions: Dict[str, Any], output_path: Path, metadata: Dict[str, Any] = None
):
    """保存預測結果"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {"predictions": predictions, "metadata": metadata or {}, "timestamp": time.time()}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"預測結果已保存到: {output_path}")
