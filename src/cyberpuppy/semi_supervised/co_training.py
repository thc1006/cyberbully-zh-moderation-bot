"""
Co-training Strategy 實作

多視角學習與互補模型訓練策略
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


@dataclass
class CoTrainingConfig:
    """Co-training 配置"""

    confidence_threshold: float = 0.8
    max_unlabeled_ratio: float = 0.3
    agreement_threshold: float = 0.9
    disagreement_weight: float = 0.1
    view_disagreement_penalty: float = 0.05
    max_iterations: int = 10
    samples_per_iteration: int = 1000
    validation_patience: int = 3


class MultiViewFeatureExtractor:
    """多視角特徵提取器"""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def extract_text_features(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """提取文字特徵（第一視角）"""
        return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}

    def extract_emotion_features(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """提取情緒特徵（第二視角）"""
        # 假設我們有預先計算的情緒特徵
        # 這可以是情緒詞彙統計、情緒強度等
        if "emotion_features" in batch:
            return {"emotion_features": batch["emotion_features"]}

        # 如果沒有預計算特徵，使用簡單的統計特徵
        text_ids = batch["input_ids"]
        batch_size, seq_len = text_ids.shape

        # 創建簡單的情緒特徵（這裡是示例，實際應用中需要更複雜的特徵）
        emotion_features = torch.randn(batch_size, 64).to(self.device)

        return {"emotion_features": emotion_features}

    def extract_linguistic_features(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """提取語言學特徵（第三視角）"""
        # 可以包括：POS 標籤、依存關係、句法特徵等
        text_ids = batch["input_ids"]
        batch_size, seq_len = text_ids.shape

        # 創建簡單的語言學特徵
        linguistic_features = torch.randn(batch_size, 32).to(self.device)

        return {"linguistic_features": linguistic_features}


class ViewSpecificModel(nn.Module):
    """視角特定模型"""

    def __init__(self, base_model: nn.Module, view_type: str, feature_dim: int = None):
        super().__init__()
        self.base_model = base_model
        self.view_type = view_type
        self.feature_dim = feature_dim

        # 根據視角類型添加特定的處理層
        if view_type == "emotion" and feature_dim:
            self.emotion_projection = nn.Linear(feature_dim, base_model.config.hidden_size)
        elif view_type == "linguistic" and feature_dim:
            self.linguistic_projection = nn.Linear(feature_dim, base_model.config.hidden_size)

    def forward(self, **inputs):
        if self.view_type == "text":
            return self.base_model(**inputs)
        elif self.view_type == "emotion":
            emotion_features = inputs["emotion_features"]
            projected_features = self.emotion_projection(emotion_features)

            # 結合情緒特徵與文字特徵
            if "input_ids" in inputs:
                text_outputs = self.base_model(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
                )
                # 簡單的特徵融合
                combined_features = text_outputs.last_hidden_state.mean(dim=1) + projected_features
                logits = self.base_model.classifier(combined_features)
                return type("ModelOutput", (), {"logits": logits})()
            else:
                logits = self.base_model.classifier(projected_features)
                return type("ModelOutput", (), {"logits": logits})()
        elif self.view_type == "linguistic":
            linguistic_features = inputs["linguistic_features"]
            projected_features = self.linguistic_projection(linguistic_features)
            logits = self.base_model.classifier(projected_features)
            return type("ModelOutput", (), {"logits": logits})()


class CoTrainingStrategy:
    """Co-training 策略實作"""

    def __init__(self, config: CoTrainingConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.feature_extractor = MultiViewFeatureExtractor(device)

    def create_view_models(
        self, base_model_class, model_config, num_views: int = 2
    ) -> List[ViewSpecificModel]:
        """創建多個視角模型"""
        models = []
        view_types = ["text", "emotion", "linguistic"]

        for i in range(min(num_views, len(view_types))):
            base_model = base_model_class(model_config)
            view_model = ViewSpecificModel(
                base_model, view_types[i], feature_dim=64 if view_types[i] == "emotion" else 32
            )
            models.append(view_model.to(self.device))

        return models

    def extract_features_for_views(
        self, batch: Dict[str, torch.Tensor], view_models: List[ViewSpecificModel]
    ) -> List[Dict[str, torch.Tensor]]:
        """為每個視角提取特徵"""
        view_features = []

        for model in view_models:
            if model.view_type == "text":
                features = self.feature_extractor.extract_text_features(batch)
            elif model.view_type == "emotion":
                features = {
                    **self.feature_extractor.extract_text_features(batch),
                    **self.feature_extractor.extract_emotion_features(batch),
                }
            elif model.view_type == "linguistic":
                features = self.feature_extractor.extract_linguistic_features(batch)
            else:
                features = self.feature_extractor.extract_text_features(batch)

            view_features.append(features)

        return view_features

    def compute_agreement(
        self, predictions: List[torch.Tensor], confidences: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """計算模型間的一致性"""
        # 預測一致性
        pred_agreements = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                agreement = (predictions[i] == predictions[j]).float()
                pred_agreements.append(agreement)

        if pred_agreements:
            avg_agreement = torch.stack(pred_agreements).mean(dim=0)
        else:
            avg_agreement = torch.ones(predictions[0].shape[0]).to(self.device)

        # 信心分數一致性
        if len(confidences) >= 2:
            conf_std = torch.stack(confidences).std(dim=0)
            conf_agreement = 1.0 / (1.0 + conf_std)  # 標準差越小，一致性越高
        else:
            conf_agreement = torch.ones_like(avg_agreement)

        return avg_agreement, conf_agreement

    def select_confident_samples(
        self,
        predictions: List[torch.Tensor],
        confidences: List[torch.Tensor],
        agreement_scores: torch.Tensor,
        batch_data: Dict[str, torch.Tensor],
        max_samples: int,
    ) -> Tuple[List[Dict], List[int]]:
        """選擇高信心且一致的樣本"""
        # 計算綜合分數
        avg_confidence = torch.stack(confidences).mean(dim=0)
        composite_scores = avg_confidence * agreement_scores

        # 篩選高分樣本
        high_score_mask = (avg_confidence >= self.config.confidence_threshold) & (
            agreement_scores >= self.config.agreement_threshold
        )

        if high_score_mask.sum() == 0:
            return [], []

        high_score_indices = torch.where(high_score_mask)[0]

        # 排序並選擇 top-k
        selected_indices = high_score_indices[
            torch.argsort(composite_scores[high_score_indices], descending=True)
        ][:max_samples]

        # 構建選中的樣本
        selected_samples = []
        for idx in selected_indices:
            sample = {
                "input_ids": batch_data["input_ids"][idx].cpu(),
                "attention_mask": batch_data["attention_mask"][idx].cpu(),
                "predictions": [pred[idx].cpu().item() for pred in predictions],
                "confidences": [conf[idx].cpu().item() for conf in confidences],
                "agreement_score": agreement_scores[idx].cpu().item(),
                "composite_score": composite_scores[idx].cpu().item(),
            }
            selected_samples.append(sample)

        return selected_samples, selected_indices.cpu().tolist()

    def co_training_step(
        self,
        view_models: List[ViewSpecificModel],
        unlabeled_batch: Dict[str, torch.Tensor],
        optimizers: List[torch.optim.Optimizer],
        criterions: List[nn.Module],
    ) -> Dict[str, float]:
        """Co-training 訓練步驟"""
        # 提取每個視角的特徵
        view_features = self.extract_features_for_views(unlabeled_batch, view_models)

        # 獲取每個模型的預測
        predictions = []
        confidences = []
        logits_list = []

        for i, (model, features) in enumerate(zip(view_models, view_features)):
            model.eval()
            with torch.no_grad():
                outputs = model(**features)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                max_probs, preds = torch.max(probs, dim=-1)

                predictions.append(preds)
                confidences.append(max_probs)
                logits_list.append(logits)

        # 計算一致性
        pred_agreement, conf_agreement = self.compute_agreement(predictions, confidences)

        # 選擇可靠樣本用於訓練
        selected_samples, selected_indices = self.select_confident_samples(
            predictions,
            confidences,
            pred_agreement,
            unlabeled_batch,
            max_samples=self.config.samples_per_iteration,
        )

        if not selected_samples:
            return {"num_selected": 0, "agreement": 0, "avg_confidence": 0}

        # 使用選定樣本訓練每個模型
        total_losses = []

        for i, (model, optimizer, criterion) in enumerate(zip(view_models, optimizers, criterions)):
            model.train()
            optimizer.zero_grad()

            # 為當前模型準備訓練資料
            if selected_indices:
                selected_features = {}
                for key, value in view_features[i].items():
                    selected_features[key] = value[selected_indices]

                # 使用其他模型的預測作為偽標籤
                other_predictions = []
                for j, pred in enumerate(predictions):
                    if j != i:
                        other_predictions.append(pred[selected_indices])

                if other_predictions:
                    # 使用多數投票或平均作為目標標籤
                    if len(other_predictions) == 1:
                        target_labels = other_predictions[0]
                    else:
                        target_labels = torch.mode(torch.stack(other_predictions), dim=0)[0]

                    # 前向傳播
                    outputs = model(**selected_features)
                    loss = criterion(outputs.logits, target_labels.to(self.device))

                    # 添加分歧懲罰
                    disagreement_penalty = 0
                    current_preds = torch.argmax(outputs.logits, dim=-1)
                    for other_pred in other_predictions:
                        disagreement = (current_preds != other_pred.to(self.device)).float().mean()
                        disagreement_penalty += disagreement

                    total_loss = loss + self.config.view_disagreement_penalty * disagreement_penalty

                    total_loss.backward()
                    optimizer.step()

                    total_losses.append(total_loss.item())

        # 統計資訊
        stats = {
            "num_selected": len(selected_samples),
            "agreement": pred_agreement.mean().item(),
            "avg_confidence": torch.stack(confidences).mean().item(),
            "avg_loss": np.mean(total_losses) if total_losses else 0,
        }

        return stats

    def train(
        self,
        view_models: List[ViewSpecificModel],
        labeled_dataloader: torch.utils.data.DataLoader,
        unlabeled_dataloader: torch.utils.data.DataLoader,
        validation_dataloader: torch.utils.data.DataLoader,
        optimizers: List[torch.optim.Optimizer],
        criterions: List[nn.Module],
    ) -> Dict[str, List[float]]:
        """完整的 Co-training 訓練流程"""
        history = {
            "train_accuracies": [],
            "val_accuracies": [],
            "agreements": [],
            "num_selected_samples": [],
            "avg_confidences": [],
        }

        best_val_accuracy = 0
        patience_counter = 0

        for iteration in range(self.config.max_iterations):
            logger.info(f"Starting co-training iteration {iteration + 1}")

            # 在有標籤資料上訓練
            self._train_on_labeled_data(view_models, labeled_dataloader, optimizers, criterions)

            # Co-training 在無標籤資料上
            iteration_stats = {"agreements": [], "num_selected": [], "avg_confidences": []}

            for batch in unlabeled_dataloader:
                stats = self.co_training_step(view_models, batch, optimizers, criterions)

                if stats["num_selected"] > 0:
                    iteration_stats["agreements"].append(stats["agreement"])
                    iteration_stats["num_selected"].append(stats["num_selected"])
                    iteration_stats["avg_confidences"].append(stats["avg_confidence"])

            # 驗證
            val_accuracies = []
            for model in view_models:
                val_acc = self._validate_model(model, validation_dataloader)
                val_accuracies.append(val_acc)

            avg_val_accuracy = np.mean(val_accuracies)

            # 記錄歷史
            history["val_accuracies"].append(avg_val_accuracy)
            history["agreements"].append(
                np.mean(iteration_stats["agreements"]) if iteration_stats["agreements"] else 0
            )
            history["num_selected_samples"].append(np.sum(iteration_stats["num_selected"]))
            history["avg_confidences"].append(
                np.mean(iteration_stats["avg_confidences"])
                if iteration_stats["avg_confidences"]
                else 0
            )

            logger.info(
                f"Iteration {iteration + 1}: "
                f"Val Acc = {avg_val_accuracy:.4f}, "
                f"Agreement = {history['agreements'][-1]:.3f}, "
                f"Selected = {history['num_selected_samples'][-1]}"
            )

            # 早停檢查
            if avg_val_accuracy > best_val_accuracy:
                best_val_accuracy = avg_val_accuracy
                patience_counter = 0

                # 保存最佳模型
                for i, model in enumerate(view_models):
                    torch.save(model.state_dict(), f"best_cotraining_model_view_{i}.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.config.validation_patience:
                    logger.info(f"Early stopping at iteration {iteration + 1}")
                    break

        return history

    def _train_on_labeled_data(
        self,
        view_models: List[ViewSpecificModel],
        labeled_dataloader: torch.utils.data.DataLoader,
        optimizers: List[torch.optim.Optimizer],
        criterions: List[nn.Module],
    ):
        """在有標籤資料上訓練模型"""
        for model in view_models:
            model.train()

        for batch in labeled_dataloader:
            labels = batch["labels"].to(self.device)
            view_features = self.extract_features_for_views(batch, view_models)

            for _i, (model, features, optimizer, criterion) in enumerate(
                zip(view_models, view_features, optimizers, criterions)
            ):
                optimizer.zero_grad()

                outputs = model(**features)
                loss = criterion(outputs.logits, labels)

                loss.backward()
                optimizer.step()

    def _validate_model(
        self, model: ViewSpecificModel, validation_dataloader: torch.utils.data.DataLoader
    ) -> float:
        """驗證單個模型"""
        model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in validation_dataloader:
                labels = batch["labels"].to(self.device)
                view_features = self.extract_features_for_views(batch, [model])[0]

                outputs = model(**view_features)
                predictions = torch.argmax(outputs.logits, dim=-1)

                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        return total_correct / total_samples if total_samples > 0 else 0
