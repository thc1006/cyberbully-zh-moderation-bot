"""
核心訓練器實現
支援多 GPU、混合精度訓練、記憶體優化和錯誤恢復
"""

import gc
import logging
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

from .checkpoint_manager import CheckpointManager
from .config import TrainingPipelineConfig
from .monitor import TrainingMonitor


class MemoryOptimizer:
    """記憶體優化器（針對 RTX 3050 4GB）"""

    def __init__(self, config: TrainingPipelineConfig):
        self.config = config
        self.empty_cache_steps = config.resources.empty_cache_steps
        self.step_counter = 0

    def optimize_memory(self, step: Optional[int] = None):
        """優化記憶體使用"""
        self.step_counter += 1

        if step is None:
            step = self.step_counter

        # 定期清理 GPU 快取
        if step % self.empty_cache_steps == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def get_memory_info(self) -> Dict[str, float]:
        """獲取記憶體信息"""
        info = {}

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            info["gpu_allocated"] = torch.cuda.memory_allocated(device) / (1024**3)
            info["gpu_reserved"] = torch.cuda.memory_reserved(device) / (1024**3)
            info["gpu_free"] = (
                torch.cuda.get_device_properties(device).total_memory
                - torch.cuda.memory_reserved(device)
            ) / (1024**3)

        return info


class DynamicBatchSizer:
    """動態批次大小調整器"""

    def __init__(self, config: TrainingPipelineConfig):
        self.config = config
        self.current_batch_size = config.data.batch_size
        self.min_batch_size = config.data.min_batch_size
        self.max_batch_size = config.data.max_batch_size
        self.last_oom_step = -1

    def adjust_batch_size(self, step: int, oom_occurred: bool = False) -> int:
        """調整批次大小"""
        if not self.config.data.dynamic_batch_size:
            return self.current_batch_size

        if oom_occurred:
            # OOM 發生時降低批次大小
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
            self.last_oom_step = step
            logging.warning(f"OOM detected, reducing batch size to {self.current_batch_size}")

        elif step - self.last_oom_step > 1000:
            # 穩定運行一段時間後嘗試增加批次大小
            if self.current_batch_size < self.max_batch_size:
                self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 1)

        return self.current_batch_size


class TrainingTracker:
    """訓練進度追蹤器"""

    def __init__(self, config: TrainingPipelineConfig):
        self.config = config
        self.metrics_history = []
        self.best_metrics = {}
        self.best_model_step = 0

    def update_metrics(self, step: int, metrics: Dict[str, float], phase: str = "train"):
        """更新指標"""
        timestamped_metrics = {"step": step, "phase": phase, "timestamp": time.time(), **metrics}
        self.metrics_history.append(timestamped_metrics)

        # 更新最佳指標
        if phase == "eval":
            metric_name = self.config.training.metric_for_best_model
            if metric_name in metrics:
                is_better = self._is_metric_better(
                    metrics[metric_name], self.best_metrics.get(metric_name, float("-inf"))
                )

                if is_better:
                    self.best_metrics = metrics.copy()
                    self.best_model_step = step

    def _is_metric_better(self, current: float, best: float) -> bool:
        """判斷指標是否更好"""
        if self.config.training.greater_is_better:
            return current > best
        else:
            return current < best

    def should_stop_early(self, patience: int) -> bool:
        """判斷是否應該早停"""
        if not self.config.training.early_stopping:
            return False

        eval_metrics = [m for m in self.metrics_history[-patience:] if m["phase"] == "eval"]

        if len(eval_metrics) < patience:
            return False

        # 檢查是否在耐心期內沒有改善
        metric_name = self.config.training.metric_for_best_model
        recent_best = max(m[metric_name] for m in eval_metrics if metric_name in m)

        threshold = self.config.training.early_stopping_threshold
        improvement = recent_best - self.best_metrics.get(metric_name, 0)

        return improvement < threshold


class MultitaskTrainer:
    """多任務訓練器"""

    def __init__(
        self,
        config: TrainingPipelineConfig,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        device: Optional[torch.device] = None,
    ):

        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # 設備配置
        self.device = device or self._setup_device()
        self.model = self.model.to(self.device)

        # 分佈式訓練
        self.is_distributed = config.resources.use_ddp and torch.cuda.device_count() > 1
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.device])

        # 優化器和調度器
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # 混合精度
        self.use_amp = config.training.fp16 and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()

        # 輔助組件
        self.memory_optimizer = MemoryOptimizer(config)
        self.batch_sizer = DynamicBatchSizer(config)
        self.tracker = TrainingTracker(config)
        self.monitor = TrainingMonitor(config)
        self.checkpoint_manager = CheckpointManager(config)

        # 訓練狀態
        self.global_step = 0
        self.current_epoch = 0
        self.is_training_interrupted = False

        # 設置日誌
        self._setup_logging()

    def _setup_device(self) -> torch.device:
        """設置計算設備"""
        if not self.config.resources.use_gpu or not torch.cuda.is_available():
            if self.config.resources.fallback_to_cpu:
                logging.info("使用 CPU 進行訓練")
                return torch.device("cpu")
            else:
                raise RuntimeError("GPU 不可用且未啟用 CPU 後備")

        # GPU 配置
        if self.config.resources.local_rank >= 0:
            device = torch.device(f"cuda:{self.config.resources.local_rank}")
        else:
            device = torch.device("cuda:0")

        # 設置記憶體配置
        if hasattr(torch.cuda, "set_memory_fraction"):
            torch.cuda.set_memory_fraction(self.config.resources.gpu_memory_fraction, device.index)

        logging.info(f"使用設備: {device}")
        return device

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """設置優化器"""
        opt_config = self.config.optimizer

        # 分組參數（不同學習率）
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": opt_config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if opt_config.name.lower() == "adamw":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=opt_config.lr,
                eps=opt_config.eps,
                betas=opt_config.betas,
            )
        elif opt_config.name.lower() == "adam":
            optimizer = Adam(
                optimizer_grouped_parameters,
                lr=opt_config.lr,
                eps=opt_config.eps,
                betas=opt_config.betas,
            )
        else:
            raise ValueError(f"不支持的優化器: {opt_config.name}")

        return optimizer

    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """設置學習率調度器"""
        opt_config = self.config.optimizer
        training_config = self.config.training

        if opt_config.scheduler_type == "none":
            return None

        # 計算總步數
        num_training_steps = (
            len(self.train_dataloader)
            * training_config.num_epochs
            // self.config.data.gradient_accumulation_steps
        )

        # 熱身步數
        if opt_config.warmup_steps > 0:
            warmup_steps = opt_config.warmup_steps
        else:
            warmup_steps = int(num_training_steps * opt_config.warmup_ratio)

        if opt_config.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=1,
                last_epoch=-1,
            )
        elif opt_config.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                last_epoch=-1,
            )
        else:
            raise ValueError(f"不支持的調度器: {opt_config.scheduler_type}")

        return scheduler

    def _setup_logging(self):
        """設置日誌"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    f"{self.config.log_dir}/training_{self.config.experiment.name}.log"
                ),
            ],
        )

    def train(self) -> Dict[str, Any]:
        """執行訓練"""
        try:
            self.monitor.start_training()

            # 載入檢查點（如果存在）
            start_epoch, start_step = self.checkpoint_manager.load_latest_checkpoint(
                self.model, self.optimizer, self.scheduler
            )

            self.current_epoch = start_epoch
            self.global_step = start_step

            logging.info(f"開始訓練，從 epoch {start_epoch}, step {start_step}")

            for epoch in range(start_epoch, self.config.training.num_epochs):
                if self.is_training_interrupted:
                    break

                self.current_epoch = epoch

                # 訓練一個 epoch
                self._train_epoch()

                # 評估
                if self._should_evaluate():
                    eval_metrics = self._evaluate()
                    self.tracker.update_metrics(self.global_step, eval_metrics, "eval")

                    # 檢查早停
                    if self.tracker.should_stop_early(self.config.training.early_stopping_patience):
                        logging.info(f"Early stopping at epoch {epoch}")
                        break

                # 保存檢查點
                if self._should_save():
                    self.checkpoint_manager.save_checkpoint(
                        epoch,
                        self.global_step,
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        self.tracker.best_metrics,
                    )

            # 載入最佳模型
            if self.config.training.load_best_model_at_end:
                self.checkpoint_manager.load_best_model(self.model)

            final_eval_metrics = self._evaluate()

            self.monitor.end_training()

            return {
                "best_metrics": self.tracker.best_metrics,
                "final_metrics": final_eval_metrics,
                "total_steps": self.global_step,
                "epochs_completed": self.current_epoch + 1,
            }

        except KeyboardInterrupt:
            logging.info("訓練被用戶中斷")
            self.is_training_interrupted = True
            # 保存當前狀態
            self.checkpoint_manager.save_checkpoint(
                self.current_epoch, self.global_step, self.model, self.optimizer, self.scheduler, {}
            )
            raise

        except Exception as e:
            logging.error(f"訓練過程中發生錯誤: {e}")
            # 保存錯誤狀態
            self.checkpoint_manager.save_checkpoint(
                self.current_epoch,
                self.global_step,
                self.model,
                self.optimizer,
                self.scheduler,
                {},
                is_error=True,
            )
            raise

    def _train_epoch(self) -> Dict[str, float]:
        """訓練一個 epoch"""
        self.model.train()
        epoch_metrics = {"loss": 0.0, "count": 0}

        # 分佈式訓練時設置 epoch
        if hasattr(self.train_dataloader.sampler, "set_epoch"):
            self.train_dataloader.sampler.set_epoch(self.current_epoch)

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch}",
            disable=self.config.resources.local_rank > 0,
        )

        for _step, batch in enumerate(progress_bar):
            try:
                # 動態調整批次大小
                current_batch_size = self.batch_sizer.adjust_batch_size(self.global_step)

                # 執行訓練步
                loss = self._training_step(batch)

                # 更新指標
                epoch_metrics["loss"] += loss
                epoch_metrics["count"] += 1

                # 記錄日誌
                if self.global_step % self.config.training.logging_steps == 0:
                    self._log_training_step(loss, current_batch_size)

                # 記憶體優化
                self.memory_optimizer.optimize_memory(self.global_step)

                # 更新進度條
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        "step": self.global_step,
                    }
                )

                self.global_step += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logging.warning(f"OOM at step {self.global_step}: {e}")

                    # 清理記憶體
                    torch.cuda.empty_cache()
                    gc.collect()

                    # 調整批次大小
                    self.batch_sizer.adjust_batch_size(self.global_step, oom_occurred=True)

                    # 跳過這個批次
                    continue
                else:
                    raise

        # 計算平均損失
        avg_loss = epoch_metrics["loss"] / max(epoch_metrics["count"], 1)
        return {"loss": avg_loss}

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """執行單個訓練步"""
        # 移動數據到設備
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

        # 前向傳播
        if self.use_amp:
            with autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
        else:
            outputs = self.model(**batch)
            loss = outputs.loss

        # 梯度累積
        loss = loss / self.config.data.gradient_accumulation_steps

        # 反向傳播
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # 優化器更新
        if (self.global_step + 1) % self.config.data.gradient_accumulation_steps == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)

            # 梯度裁剪
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.max_grad_norm
                )

            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            self.optimizer.zero_grad()

        return loss.item() * self.config.data.gradient_accumulation_steps

    def _evaluate(self) -> Dict[str, float]:
        """評估模型"""
        self.model.eval()
        eval_metrics = {"loss": 0.0, "count": 0}
        all_predictions = {"toxicity": [], "bullying": [], "emotion": [], "role": []}
        all_labels = {"toxicity": [], "bullying": [], "emotion": [], "role": []}

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                if self.use_amp:
                    with autocast():
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)

                eval_metrics["loss"] += outputs.loss.item()
                eval_metrics["count"] += 1

                # 收集預測和標籤
                for task in all_predictions.keys():
                    if hasattr(outputs, f"{task}_logits"):
                        logits = getattr(outputs, f"{task}_logits")
                        preds = torch.argmax(logits, dim=-1).cpu().numpy()
                        labels = batch[f"{task}_labels"].cpu().numpy()

                        all_predictions[task].extend(preds)
                        all_labels[task].extend(labels)

        # 計算指標
        avg_loss = eval_metrics["loss"] / max(eval_metrics["count"], 1)
        eval_metrics = {"eval_loss": avg_loss}

        # 計算各任務的 F1 分數
        for task in all_predictions.keys():
            if all_predictions[task] and all_labels[task]:
                f1_macro = f1_score(all_labels[task], all_predictions[task], average="macro")
                f1_micro = f1_score(all_labels[task], all_predictions[task], average="micro")
                eval_metrics[f"eval_{task}_f1_macro"] = f1_macro
                eval_metrics[f"eval_{task}_f1_micro"] = f1_micro

        # 計算整體 F1（主要任務權重更高）
        if all_predictions["toxicity"] and all_predictions["bullying"]:
            overall_f1 = (
                eval_metrics.get("eval_toxicity_f1_macro", 0) * 0.4
                + eval_metrics.get("eval_bullying_f1_macro", 0) * 0.4
                + eval_metrics.get("eval_emotion_f1_macro", 0) * 0.1
                + eval_metrics.get("eval_role_f1_macro", 0) * 0.1
            )
            eval_metrics["eval_f1_macro"] = overall_f1

        return eval_metrics

    def _should_evaluate(self) -> bool:
        """判斷是否應該評估"""
        if self.config.training.validation_strategy == "steps":
            return self.global_step % self.config.training.eval_steps == 0
        else:  # epoch
            return True

    def _should_save(self) -> bool:
        """判斷是否應該保存檢查點"""
        return self.global_step % self.config.training.save_steps == 0

    def _log_training_step(self, loss: float, batch_size: int):
        """記錄訓練步驟"""
        metrics = {
            "train_loss": loss,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "batch_size": batch_size,
            "epoch": self.current_epoch,
        }

        # 添加記憶體信息
        memory_info = self.memory_optimizer.get_memory_info()
        metrics.update(memory_info)

        # 記錄到監控器
        self.monitor.log_metrics(self.global_step, metrics)

        # 記錄到追蹤器
        self.tracker.update_metrics(self.global_step, metrics, "train")

        if self.global_step % (self.config.training.logging_steps * 10) == 0:
            logging.info(
                f"Step {self.global_step}: loss={loss:.4f}, "
                f"lr={metrics['learning_rate']:.2e}, "
                f"mem={memory_info.get('gpu_allocated', 0):.1f}GB"
            )


def create_trainer(
    config: TrainingPipelineConfig,
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
) -> MultitaskTrainer:
    """創建訓練器的工廠函數"""
    return MultitaskTrainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )
