"""
Pseudo-labeling Pipeline 實作

使用高信心預測作為偽標籤，動態調整信心閾值，支援迭代式訓練
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


@dataclass
class PseudoLabelConfig:
    """Pseudo-labeling 配置"""

    confidence_threshold: float = 0.9
    min_confidence_threshold: float = 0.7
    max_confidence_threshold: float = 0.95
    threshold_decay: float = 0.95
    threshold_increase: float = 1.01
    max_pseudo_samples: int = 10000
    validation_threshold: float = 0.8
    update_frequency: int = 500
    warmup_steps: int = 1000


class ConfidenceTracker:
    """信心分數追蹤器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.confidences = []
        self.predictions = []
        self.true_labels = []

    def add_batch(
        self,
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        true_labels: Optional[torch.Tensor] = None,
    ):
        """添加批次資料"""
        self.confidences.extend(confidences.cpu().numpy())
        self.predictions.extend(predictions.cpu().numpy())

        if true_labels is not None:
            self.true_labels.extend(true_labels.cpu().numpy())

        # 保持視窗大小
        if len(self.confidences) > self.window_size:
            self.confidences = self.confidences[-self.window_size :]
            self.predictions = self.predictions[-self.window_size :]
            if self.true_labels:
                self.true_labels = self.true_labels[-self.window_size :]

    def get_statistics(self) -> Dict[str, float]:
        """獲取統計資料"""
        if not self.confidences:
            return {}

        stats = {
            "mean_confidence": np.mean(self.confidences),
            "std_confidence": np.std(self.confidences),
            "min_confidence": np.min(self.confidences),
            "max_confidence": np.max(self.confidences),
            "high_confidence_ratio": np.mean(np.array(self.confidences) > 0.9),
        }

        if self.true_labels:
            # 計算高信心預測的準確率
            high_conf_mask = np.array(self.confidences) > 0.9
            if high_conf_mask.sum() > 0:
                high_conf_acc = (
                    np.array(self.predictions)[high_conf_mask]
                    == np.array(self.true_labels)[high_conf_mask]
                ).mean()
                stats["high_confidence_accuracy"] = high_conf_acc

        return stats


class PseudoLabelingPipeline:
    """Pseudo-labeling 流水線"""

    def __init__(self, config: PseudoLabelConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.confidence_tracker = ConfidenceTracker()
        self.current_threshold = config.confidence_threshold
        self.step_count = 0
        self.pseudo_labeled_data = []

    def generate_pseudo_labels(
        self,
        model: nn.Module,
        unlabeled_dataloader: torch.utils.data.DataLoader,
        max_samples: Optional[int] = None,
    ) -> Tuple[List[Dict], Dict[str, float]]:
        """生成偽標籤"""
        model.eval()
        pseudo_samples = []
        all_confidences = []

        max_samples = max_samples or self.config.max_pseudo_samples

        with torch.no_grad():
            for _batch_idx, batch in enumerate(unlabeled_dataloader):
                if len(pseudo_samples) >= max_samples:
                    break

                # 前向傳播
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "text"}
                outputs = model(**inputs)

                # 計算信心分數
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                confidences, predicted_labels = torch.max(probabilities, dim=-1)

                # 篩選高信心樣本
                high_conf_mask = confidences >= self.current_threshold

                if high_conf_mask.sum() > 0:
                    high_conf_indices = torch.where(high_conf_mask)[0]

                    for idx in high_conf_indices:
                        if len(pseudo_samples) >= max_samples:
                            break

                        sample = {
                            "text": batch["text"][idx] if "text" in batch else None,
                            "input_ids": batch["input_ids"][idx].cpu(),
                            "attention_mask": batch["attention_mask"][idx].cpu(),
                            "pseudo_label": predicted_labels[idx].cpu().item(),
                            "confidence": confidences[idx].cpu().item(),
                            "original_logits": logits[idx].cpu(),
                        }
                        pseudo_samples.append(sample)

                all_confidences.extend(confidences.cpu().tolist())

                # 更新信心追蹤器
                self.confidence_tracker.add_batch(confidences, predicted_labels)

        # 統計資訊
        stats = {
            "total_unlabeled": len(unlabeled_dataloader.dataset),
            "pseudo_labeled": len(pseudo_samples),
            "pseudo_label_ratio": len(pseudo_samples) / len(unlabeled_dataloader.dataset),
            "current_threshold": self.current_threshold,
            "mean_confidence": np.mean(all_confidences) if all_confidences else 0,
            **self.confidence_tracker.get_statistics(),
        }

        logger.info(
            f"Generated {len(pseudo_samples)} pseudo-labeled samples "
            f"with threshold {self.current_threshold:.3f}"
        )

        return pseudo_samples, stats

    def update_threshold(self, validation_performance: float):
        """動態更新信心閾值"""
        if validation_performance > self.config.validation_threshold:
            # 表現良好，降低閾值以獲得更多偽標籤
            new_threshold = self.current_threshold * self.config.threshold_decay
            new_threshold = max(new_threshold, self.config.min_confidence_threshold)
        else:
            # 表現不佳，提高閾值以獲得更高品質偽標籤
            new_threshold = self.current_threshold * self.config.threshold_increase
            new_threshold = min(new_threshold, self.config.max_confidence_threshold)

        if new_threshold != self.current_threshold:
            logger.info(f"Updating threshold: {self.current_threshold:.3f} -> {new_threshold:.3f}")
            self.current_threshold = new_threshold

    def validate_pseudo_labels(
        self,
        model: nn.Module,
        pseudo_samples: List[Dict],
        validation_dataloader: torch.utils.data.DataLoader,
    ) -> float:
        """驗證偽標籤品質"""
        model.eval()

        # 在驗證集上評估模型表現
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

        # 分析偽標籤的信心分佈
        if pseudo_samples:
            confidences = [sample["confidence"] for sample in pseudo_samples]
            logger.info(
                f"Pseudo-label confidence stats: "
                f"mean={np.mean(confidences):.3f}, "
                f"std={np.std(confidences):.3f}, "
                f"min={np.min(confidences):.3f}"
            )

        return accuracy

    def create_pseudo_dataset(
        self,
        pseudo_samples: List[Dict],
        original_dataset: torch.utils.data.Dataset,
        mixing_ratio: float = 0.5,
    ) -> torch.utils.data.Dataset:
        """創建混合原始和偽標籤的資料集"""
        from torch.utils.data import ConcatDataset, Subset

        # 創建偽標籤資料集
        pseudo_dataset = PseudoLabelDataset(pseudo_samples)

        # 根據混合比例採樣原始資料
        original_size = int(len(pseudo_samples) * (1 - mixing_ratio) / mixing_ratio)
        original_size = min(original_size, len(original_dataset))

        if original_size > 0:
            indices = np.random.choice(len(original_dataset), original_size, replace=False)
            original_subset = Subset(original_dataset, indices)
            mixed_dataset = ConcatDataset([original_subset, pseudo_dataset])
        else:
            mixed_dataset = pseudo_dataset

        logger.info(
            f"Created mixed dataset: {original_size} original + "
            f"{len(pseudo_samples)} pseudo = {len(mixed_dataset)} total"
        )

        return mixed_dataset

    def iterative_training(
        self,
        model: nn.Module,
        labeled_dataloader: torch.utils.data.DataLoader,
        unlabeled_dataloader: torch.utils.data.DataLoader,
        validation_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_iterations: int = 5,
        epochs_per_iteration: int = 3,
    ) -> Dict[str, List[float]]:
        """迭代式偽標籤訓練"""
        history = {
            "pseudo_label_ratios": [],
            "validation_accuracies": [],
            "thresholds": [],
            "total_pseudo_samples": [],
        }

        for iteration in range(num_iterations):
            logger.info(f"Starting iteration {iteration + 1}/{num_iterations}")

            # 生成偽標籤
            pseudo_samples, stats = self.generate_pseudo_labels(model, unlabeled_dataloader)

            if not pseudo_samples:
                logger.warning("No pseudo-samples generated, stopping")
                break

            # 創建混合資料集
            mixed_dataset = self.create_pseudo_dataset(pseudo_samples, labeled_dataloader.dataset)
            mixed_dataloader = torch.utils.data.DataLoader(
                mixed_dataset,
                batch_size=labeled_dataloader.batch_size,
                shuffle=True,
                num_workers=labeled_dataloader.num_workers,
            )

            # 在混合資料集上訓練
            for epoch in range(epochs_per_iteration):
                model.train()
                total_loss = 0

                for batch in mixed_dataloader:
                    optimizer.zero_grad()

                    inputs = {
                        k: v.to(self.device)
                        for k, v in batch.items()
                        if k not in ["text", "labels"]
                    }
                    labels = batch["labels"].to(self.device)

                    outputs = model(**inputs)
                    loss = criterion(outputs.logits, labels)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(mixed_dataloader)
                logger.info(
                    f"Iteration {iteration + 1}, Epoch {epoch + 1}: " f"Loss = {avg_loss:.4f}"
                )

            # 驗證並更新閾值
            val_accuracy = self.validate_pseudo_labels(model, pseudo_samples, validation_dataloader)
            self.update_threshold(val_accuracy)

            # 記錄歷史
            history["pseudo_label_ratios"].append(stats["pseudo_label_ratio"])
            history["validation_accuracies"].append(val_accuracy)
            history["thresholds"].append(self.current_threshold)
            history["total_pseudo_samples"].append(len(pseudo_samples))

            logger.info(
                f"Iteration {iteration + 1} completed: "
                f"Val Acc = {val_accuracy:.4f}, "
                f"Threshold = {self.current_threshold:.3f}"
            )

        return history


class PseudoLabelDataset(torch.utils.data.Dataset):
    """偽標籤資料集"""

    def __init__(self, pseudo_samples: List[Dict]):
        self.samples = pseudo_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": torch.tensor(sample["pseudo_label"], dtype=torch.long),
        }
