"""
Self-training Framework 實作

教師-學生模型架構與知識蒸餾機制
"""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


@dataclass
class SelfTrainingConfig:
    """Self-training 配置"""

    teacher_update_frequency: int = 500
    student_teacher_ratio: float = 0.7  # 學生損失 vs 教師損失的比例
    distillation_temperature: float = 4.0
    ema_decay: float = 0.999  # EMA 更新教師模型的衰減率
    consistency_weight: float = 1.0
    confidence_threshold: float = 0.8
    max_epochs: int = 10
    warmup_steps: int = 1000


class TeacherStudentTrainer:
    """教師-學生訓練器"""

    def __init__(self, config: SelfTrainingConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.step_count = 0

    def create_teacher_model(self, student_model: nn.Module) -> nn.Module:
        """創建教師模型（EMA 版本的學生模型）"""
        teacher_model = copy.deepcopy(student_model)

        # 停用 dropout 和 batch norm 的訓練模式
        teacher_model.eval()

        # 凍結教師模型參數
        for param in teacher_model.parameters():
            param.requires_grad = False

        return teacher_model

    def update_teacher_model(self, teacher_model: nn.Module, student_model: nn.Module):
        """使用 EMA 更新教師模型"""
        alpha = self.config.ema_decay

        with torch.no_grad():
            for teacher_param, student_param in zip(
                teacher_model.parameters(), student_model.parameters()
            ):
                teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data

    def knowledge_distillation_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = None
    ) -> torch.Tensor:
        """知識蒸餾損失"""
        if temperature is None:
            temperature = self.config.distillation_temperature

        # 軟化機率分佈
        F.softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # KL 散度損失
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            teacher_probs,
            reduction="batchmean",
        )

        return kl_loss * (temperature**2)

    def consistency_loss(
        self, logits1: torch.Tensor, logits2: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """一致性損失"""
        # MSE 損失
        loss = F.mse_loss(logits1, logits2, reduction="none")

        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss

    def compute_confidence_mask(
        self, teacher_logits: torch.Tensor, threshold: float = None
    ) -> torch.Tensor:
        """計算高信心樣本遮罩"""
        if threshold is None:
            threshold = self.config.confidence_threshold

        teacher_probs = F.softmax(teacher_logits, dim=-1)
        max_probs, _ = torch.max(teacher_probs, dim=-1)

        return max_probs >= threshold

    def train_step(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        labeled_batch: Dict[str, torch.Tensor],
        unlabeled_batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """單步訓練"""
        student_model.train()
        teacher_model.eval()

        total_loss = 0
        losses = {}

        # 有標籤資料的監督損失
        if labeled_batch:
            labeled_inputs = {
                k: v.to(self.device)
                for k, v in labeled_batch.items()
                if k not in ["text", "labels"]
            }
            labeled_labels = labeled_batch["labels"].to(self.device)

            student_outputs = student_model(**labeled_inputs)
            supervised_loss = criterion(student_outputs.logits, labeled_labels)

            total_loss += supervised_loss
            losses["supervised"] = supervised_loss.item()

        # 無標籤資料的自監督損失
        if unlabeled_batch:
            unlabeled_inputs = {
                k: v.to(self.device)
                for k, v in unlabeled_batch.items()
                if k not in ["text", "labels"]
            }

            with torch.no_grad():
                teacher_outputs = teacher_model(**unlabeled_inputs)
                teacher_logits = teacher_outputs.logits

            student_outputs = student_model(**unlabeled_inputs)
            student_logits = student_outputs.logits

            # 計算信心遮罩
            confidence_mask = self.compute_confidence_mask(teacher_logits)

            if confidence_mask.sum() > 0:
                # 知識蒸餾損失
                distillation_loss = self.knowledge_distillation_loss(
                    student_logits[confidence_mask], teacher_logits[confidence_mask]
                )

                # 一致性損失
                consistency_loss = self.consistency_loss(
                    student_logits, teacher_logits, confidence_mask
                )

                unsupervised_loss = (
                    distillation_loss + self.config.consistency_weight * consistency_loss
                )

                total_loss += self.config.student_teacher_ratio * unsupervised_loss
                losses["distillation"] = distillation_loss.item()
                losses["consistency"] = consistency_loss.item()
                losses["confidence_ratio"] = confidence_mask.float().mean().item()

        # 反向傳播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 更新教師模型
        self.step_count += 1
        if self.step_count % self.config.teacher_update_frequency == 0:
            self.update_teacher_model(teacher_model, student_model)

        losses["total"] = total_loss.item()
        return losses


class SelfTrainingFramework:
    """Self-training 框架"""

    def __init__(self, config: SelfTrainingConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.trainer = TeacherStudentTrainer(config, device)

    def create_augmented_data(
        self, batch: Dict[str, torch.Tensor], augmentation_strength: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """資料增強（可選）"""
        # 這裡可以實作文字增強策略
        # 例如：同義詞替換、句子重排等
        # 目前返回原始資料
        return batch

    def train(
        self,
        student_model: nn.Module,
        labeled_dataloader: torch.utils.data.DataLoader,
        unlabeled_dataloader: torch.utils.data.DataLoader,
        validation_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict[str, List[float]]:
        """完整訓練流程"""
        # 創建教師模型
        teacher_model = self.trainer.create_teacher_model(student_model)

        history = {
            "train_losses": [],
            "val_accuracies": [],
            "supervised_losses": [],
            "distillation_losses": [],
            "consistency_losses": [],
            "confidence_ratios": [],
        }

        best_val_accuracy = 0
        patience_counter = 0
        max_patience = 5

        for epoch in range(self.config.max_epochs):
            student_model.train()
            epoch_losses = {
                "total": [],
                "supervised": [],
                "distillation": [],
                "consistency": [],
                "confidence_ratio": [],
            }

            # 創建資料迭代器
            labeled_iter = iter(labeled_dataloader)
            unlabeled_iter = iter(unlabeled_dataloader)

            max_batches = max(len(labeled_dataloader), len(unlabeled_dataloader))

            for batch_idx in range(max_batches):
                # 獲取有標籤批次
                try:
                    labeled_batch = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_dataloader)
                    labeled_batch = next(labeled_iter)

                # 獲取無標籤批次
                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_dataloader)
                    unlabeled_batch = next(unlabeled_iter)

                # 訓練步驟
                losses = self.trainer.train_step(
                    student_model,
                    teacher_model,
                    labeled_batch,
                    unlabeled_batch,
                    optimizer,
                    criterion,
                )

                # 記錄損失
                for key, value in losses.items():
                    if key in epoch_losses:
                        epoch_losses[key].append(value)

                if scheduler:
                    scheduler.step()

                # 定期印出進度
                if batch_idx % 100 == 0:
                    logger.info(
                        f"Epoch {epoch+1}, Batch {batch_idx}: " f"Loss = {losses['total']:.4f}"
                    )

            # 計算平均損失
            avg_losses = {
                key: np.mean(values) if values else 0 for key, values in epoch_losses.items()
            }

            # 驗證
            val_accuracy = self.validate(student_model, validation_dataloader)

            # 記錄歷史
            history["train_losses"].append(avg_losses["total"])
            history["val_accuracies"].append(val_accuracy)
            history["supervised_losses"].append(avg_losses["supervised"])
            history["distillation_losses"].append(avg_losses["distillation"])
            history["consistency_losses"].append(avg_losses["consistency"])
            history["confidence_ratios"].append(avg_losses["confidence_ratio"])

            logger.info(
                f"Epoch {epoch+1}/{self.config.max_epochs}: "
                f"Train Loss = {avg_losses['total']:.4f}, "
                f"Val Acc = {val_accuracy:.4f}, "
                f"Confidence Ratio = {avg_losses['confidence_ratio']:.3f}"
            )

            # 早停檢查
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0

                # 保存最佳模型
                torch.save(
                    {
                        "student_model": student_model.state_dict(),
                        "teacher_model": teacher_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_accuracy": val_accuracy,
                    },
                    "best_self_training_model.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        return history

    def validate(
        self, model: nn.Module, validation_dataloader: torch.utils.data.DataLoader
    ) -> float:
        """驗證模型"""
        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in validation_dataloader:
                inputs = {
                    k: v.to(self.device) for k, v in batch.items() if k not in ["text", "labels"]
                }
                labels = batch["labels"].to(self.device)

                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)

                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples if total_samples > 0 else 0
        return accuracy

    def predict_with_uncertainty(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_forward_passes: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用 Monte Carlo Dropout 預測不確定性"""
        model.train()  # 啟用 dropout

        all_predictions = []

        with torch.no_grad():
            for _ in range(num_forward_passes):
                batch_predictions = []

                for batch in dataloader:
                    inputs = {
                        k: v.to(self.device)
                        for k, v in batch.items()
                        if k not in ["text", "labels"]
                    }

                    outputs = model(**inputs)
                    probabilities = F.softmax(outputs.logits, dim=-1)
                    batch_predictions.append(probabilities.cpu())

                if batch_predictions:
                    epoch_predictions = torch.cat(batch_predictions, dim=0)
                    all_predictions.append(epoch_predictions)

        if all_predictions:
            # 計算平均預測和不確定性
            predictions = torch.stack(all_predictions)  # [num_passes, num_samples, num_classes]
            mean_predictions = predictions.mean(dim=0)
            uncertainty = predictions.std(dim=0).mean(dim=-1)  # 標準差作為不確定性

            return mean_predictions, uncertainty
        else:
            return torch.empty(0), torch.empty(0)
