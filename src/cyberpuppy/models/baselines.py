#!/usr/bin/env python3
"""
基線模型
支援多任務分類，包含毒性偵測、情緒分析等
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from ..labeling.label_map import (
    BullyingLevel,
    EmotionType,
    RoleType,
    ToxicityLevel,
    UnifiedLabel,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""

    # 基礎模型
    model_name: str = "hfl/chinese-macbert-base"
    max_length: int = 256

    # 多任務設定
    num_toxicity_classes: int = 3  # none, toxic, severe
    num_bullying_classes: int = 3  # none, harassment, threat
    num_role_classes: int = 4  # none, perpetrator, victim, bystander
    num_emotion_classes: int = 3  # positive, neutral, negative
    use_emotion_regression: bool = False  # 情緒強度回歸

    # 訓練設定
    hidden_dropout: float = 0.1
    classifier_dropout: float = 0.1
    hidden_size: int = 768

    # 損失函數設定
    use_focal_loss: bool = True
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    class_weights: Optional[Dict[str, List[float]]] = None

    # 多任務權重
    task_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                "toxicity": 1.0,
                "bullying": 1.0,
                "role": 0.5,
                "emotion": 0.8,
                "emotion_intensity": 0.6,
            }


class FocalLoss(nn.Module):
    """Focal Loss實現"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, num_classes] 邏輯回歸輸出
            targets: [batch_size] 目標標籤
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class MultiTaskDataset(Dataset):
    """多任務資料集"""

    def __init__(
        self,
        texts: List[str],
        labels: List[UnifiedLabel],
        tokenizer,
        max_length: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 建立標籤映射
        self.toxicity_map = {
            ToxicityLevel.NONE: 0,
            ToxicityLevel.TOXIC: 1,
            ToxicityLevel.SEVERE: 2,
        }
        self.bullying_map = {
            BullyingLevel.NONE: 0,
            BullyingLevel.HARASSMENT: 1,
            BullyingLevel.THREAT: 2,
        }
        self.role_map = {
            RoleType.NONE: 0,
            RoleType.PERPETRATOR: 1,
            RoleType.VICTIM: 2,
            RoleType.BYSTANDER: 3,
        }
        self.emotion_map = {
            EmotionType.POSITIVE: 0,
            EmotionType.NEUTRAL: 1,
            EmotionType.NEGATIVE: 2,
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 分詞
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # 轉換標籤
        toxicity_label = self.toxicity_map.get(label.toxicity, 0)
        bullying_label = self.bullying_map.get(label.bullying, 0)
        role_label = self.role_map.get(label.role, 0)
        emotion_label = self.emotion_map.get(label.emotion, 1)
        emotion_intensity = label.emotion_intensity

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get(
                "token_type_ids", torch.zeros_like(encoding["input_ids"])
            ).squeeze(0),
            "toxicity_label": torch.tensor(toxicity_label, dtype=torch.long),
            "bullying_label": torch.tensor(bullying_label, dtype=torch.long),
            "role_label": torch.tensor(role_label, dtype=torch.long),
            "emotion_label": torch.tensor(emotion_label, dtype=torch.long),
            "emotion_intensity": torch.tensor(emotion_intensity, dtype=torch.float),
        }


class MultiTaskHead(nn.Module):
    """多任務分類頭"""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        hidden_size = config.hidden_size

        # 共享特徵層
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
        )

        # 各任務專用頭
        self.toxicity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(hidden_size // 2, config.num_toxicity_classes),
        )

        self.bullying_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(hidden_size // 2, config.num_bullying_classes),
        )

        self.role_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(hidden_size // 2, config.num_role_classes),
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(hidden_size // 2, config.num_emotion_classes),
        )

        # 情緒強度回歸頭（可選）
        if config.use_emotion_regression:
            self.emotion_intensity_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(config.classifier_dropout),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid(),  # 輸出 0-1，可以乘以最大強度值
            )

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, hidden_size]

        Returns:
            outputs: 各任務的輸出
        """
        # 共享特徵
        shared_features = self.shared_layer(hidden_states)

        # 各任務預測
        outputs = {
            "toxicity": self.toxicity_head(shared_features),
            "bullying": self.bullying_head(shared_features),
            "role": self.role_head(shared_features),
            "emotion": self.emotion_head(shared_features),
        }

        # 情緒強度回歸
        if self.config.use_emotion_regression and hasattr(
            self, "emotion_intensity_head"
        ):
            outputs["emotion_intensity"] = (
                self.emotion_intensity_head(shared_features) * 4.0
            )  # 0-4強度

        return outputs


class BaselineModel(nn.Module):
    """基線模型"""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # 預訓練模型
        self.backbone = AutoModel.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # 設定隱藏層dropout
        if hasattr(self.backbone.config, "hidden_dropout_prob"):
            self.backbone.config.hidden_dropout_prob = config.hidden_dropout

        # 多任務頭
        self.multi_task_head = MultiTaskHead(config)

        # 損失函數
        self._setup_loss_functions()

    def _setup_loss_functions(self):
        """設定損失函數"""
        config = self.config

        if config.use_focal_loss:
            self.toxicity_loss_fn = FocalLoss(config.focal_alpha, config.focal_gamma)
            self.bullying_loss_fn = FocalLoss(config.focal_alpha, config.focal_gamma)
            self.role_loss_fn = FocalLoss(config.focal_alpha, config.focal_gamma)
            self.emotion_loss_fn = FocalLoss(config.focal_alpha, config.focal_gamma)
        else:
            # 標準交叉熵損失
            label_smoothing = config.label_smoothing

            # 類別權重
            toxicity_weights = None
            bullying_weights = None
            role_weights = None
            emotion_weights = None

            if config.class_weights:
                if "toxicity" in config.class_weights:
                    toxicity_weights = torch.tensor(
                        config.class_weights["toxicity"], dtype=torch.float
                    )
                if "bullying" in config.class_weights:
                    bullying_weights = torch.tensor(
                        config.class_weights["bullying"], dtype=torch.float
                    )
                if "role" in config.class_weights:
                    role_weights = torch.tensor(
                        config.class_weights["role"], dtype=torch.float
                    )
                if "emotion" in config.class_weights:
                    emotion_weights = torch.tensor(
                        config.class_weights["emotion"], dtype=torch.float
                    )

            self.toxicity_loss_fn = nn.CrossEntropyLoss(
                weight=toxicity_weights, label_smoothing=label_smoothing
            )
            self.bullying_loss_fn = nn.CrossEntropyLoss(
                weight=bullying_weights, label_smoothing=label_smoothing
            )
            self.role_loss_fn = nn.CrossEntropyLoss(
                weight=role_weights, label_smoothing=label_smoothing
            )
            self.emotion_loss_fn = nn.CrossEntropyLoss(
                weight=emotion_weights, label_smoothing=label_smoothing
            )

        # 情緒強度回歸損失
        if config.use_emotion_regression:
            self.emotion_intensity_loss_fn = nn.MSELoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向傳播

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]

        Returns:
            outputs: 各任務的預測結果
        """
        # Backbone輸出
        backbone_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            backbone_inputs["token_type_ids"] = token_type_ids

        backbone_outputs = self.backbone(**backbone_inputs)

        # 使用[CLS] token的表示
        pooled_output = backbone_outputs.last_hidden_state[
            :, 0, :
        ]  # [batch_size, hidden_size]

        # 多任務預測
        task_outputs = self.multi_task_head(pooled_output)

        return task_outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        task_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        計算多任務損失

        Args:
            outputs: 模型預測輸出
            labels: 真實標籤
            task_mask: 任務掩碼（用於處理缺失標籤）

        Returns:
            losses: 各任務損失
        """
        losses = {}

        # 任務權重
        task_weights = self.config.task_weights

        # 毒性分類損失
        if "toxicity" in outputs and "toxicity_label" in labels:
            mask = (
                task_mask.get("toxicity", torch.ones_like(labels["toxicity_label"]))
                if task_mask
                else None
            )
            if mask is None or mask.sum() > 0:
                toxicity_loss = self.toxicity_loss_fn(
                    outputs["toxicity"], labels["toxicity_label"]
                )
                if mask is not None:
                    toxicity_loss = (toxicity_loss * mask).sum() / mask.sum()
                losses["toxicity"] = toxicity_loss * task_weights.get("toxicity", 1.0)

        # 霸凌分類損失
        if "bullying" in outputs and "bullying_label" in labels:
            mask = (
                task_mask.get("bullying", torch.ones_like(labels["bullying_label"]))
                if task_mask
                else None
            )
            if mask is None or mask.sum() > 0:
                bullying_loss = self.bullying_loss_fn(
                    outputs["bullying"], labels["bullying_label"]
                )
                if mask is not None:
                    bullying_loss = (bullying_loss * mask).sum() / mask.sum()
                losses["bullying"] = bullying_loss * task_weights.get("bullying", 1.0)

        # 角色分類損失
        if "role" in outputs and "role_label" in labels:
            mask = (
                task_mask.get("role", torch.ones_like(labels["role_" "label"]))
                if task_mask
                else None
            )
            if mask is None or mask.sum() > 0:
                role_loss = self.role_loss_fn(outputs["role"], labels["role_label"])
                if mask is not None:
                    role_loss = (role_loss * mask).sum() / mask.sum()
                losses["role"] = role_loss * task_weights.get("role", 1.0)

        # 情緒分類損失
        if "emotion" in outputs and "emotion_label" in labels:
            mask = (
                task_mask.get("emotion", torch.ones_like(labels["emotion_label"]))
                if task_mask
                else None
            )
            if mask is None or mask.sum() > 0:
                emotion_loss = self.emotion_loss_fn(
                    outputs["emotion"], labels["emotion_label"]
                )
                if mask is not None:
                    emotion_loss = (emotion_loss * mask).sum() / mask.sum()
                losses["emotion"] = emotion_loss * task_weights.get("emotion", 1.0)

        # 情緒強度回歸損失
        if (
            "emotion_intensity" in outputs
            and "emotion_intensity" in labels
            and hasattr(self, "emotion_intensity_loss_fn")
        ):
            mask = task_mask.get("emotion_intensity") if task_mask else None
            if mask is None or mask.sum() > 0:
                intensity_loss = self.emotion_intensity_loss_fn(
                    outputs["emotion_intensity"].squeeze(-1),
                    labels["emotion_intensity"],
                )
                if mask is not None:
                    intensity_loss = (intensity_loss * mask).sum() / mask.sum()
                losses["emotion_intensity"] = intensity_loss * task_weights.get(
                    "emotion_intensity", 1.0
                )

        # 總損失
        losses["total"] = sum(losses.values())

        return losses

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[int, float, np.ndarray]]:
        """
        預測

        Returns:
            predictions: 預測結果和概率
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, token_type_ids)

            predictions = {}

            # 分類預測
            for task in ["toxicity", "bullying", "role", "emotion"]:
                if task in outputs:
                    logits = outputs[task]
                    probs = torch.softmax(logits, dim=-1)
                    pred_labels = torch.argmax(logits, dim=-1)

                    predictions[f"{task}_pred"] = pred_labels.cpu().numpy()
                    predictions[f"{task}_probs"] = probs.cpu().numpy()
                    predictions[f"{task}_confidence"] = (
                        probs.max(dim=-1)[0].cpu().numpy()
                    )

            # 情緒強度回歸
            if "emotion_intensity" in outputs:
                predictions["emotion_intensity"] = (
                    outputs["emotion_intensity"].squeeze(-1).cpu().numpy()
                )

        return predictions

    def save_model(self, save_path: str):
        """儲存模型"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 儲存模型狀態
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config,
                "model_class": self.__class__.__name__,
            },
            save_path / "best.ckpt",
        )

        # 儲存tokenizer
        self.tokenizer.save_pretrained(save_path)

        # 儲存配置
        with open(save_path / "model_config.json", "w", encoding="utf-8") as f:
            config_dict = {
                "model_name": self.config.model_name,
                "max_length": self.config.max_length,
                "num_toxicity_classes": self.config.num_toxicity_classes,
                "num_bullying_classes": self.config.num_bullying_classes,
                "num_role_classes": self.config.num_role_classes,
                "num_emotion_classes": self.config.num_emotion_classes,
                "use_emotion_regression": self.config.use_emotion_regression,
                "task_weights": self.config.task_weights,
            }
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load_model(cls, load_path: str):
        """載入模型"""
        load_path = Path(load_path)

        # 載入模型檢查點
        checkpoint = torch.load(
            load_path / "best.ckpt", map_location="cpu", weights_only=False
        )
        config = checkpoint["config"]

        # 建立模型
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(f"Model loaded from {load_path}")
        return model


class ModelEvaluator:
    """模型評估器"""

    def __init__(self, model: BaselineModel, device: torch.device = None):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def evaluate(
        self, dataloader: DataLoader, session_ids: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        評估模型

        Args:
            dataloader: 資料載入器
            session_ids: 會話ID（用於計算會話級F1）

        Returns:
            metrics: 評估指標
        """
        self.model.eval()

        all_predictions = {
            "toxicity": [],
            "bullying": [],
            "role": [],
            "emotion": [],
            "emotion_intensity": [],
        }
        all_labels = {
            "toxicity": [],
            "bullying": [],
            "role": [],
            "emotion": [],
            "emotion_intensity": [],
        }
        all_probs = {"toxicity": [], "bullying": [], "role": [], "emotion": []}

        with torch.no_grad():
            for batch in dataloader:
                # 移動到設備
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                # 前向傳播
                outputs = self.model(input_ids, attention_mask, token_type_ids)

                # 收集預測和標籤
                for task in ["toxicity", "bullying", "role", "emotion"]:
                    if task in outputs:
                        logits = outputs[task]
                        probs = torch.softmax(logits, dim=-1)
                        preds = torch.argmax(logits, dim=-1)

                        all_predictions[task].extend(preds.cpu().numpy())
                        all_probs[task].extend(probs.cpu().numpy())
                        all_labels[task].extend(batch[f"{task}_label"].numpy())

                # 情緒強度回歸
                if "emotion_intensity" in outputs:
                    intensity_preds = outputs["emotion_intensity"].squeeze(-1)
                    all_predictions["emotion_intensity"].extend(
                        intensity_preds.cpu().numpy()
                    )
                    all_labels["emotion_intensity"].extend(batch["emotion_intensity"])

        # 計算指標
        metrics = {}

        # 分類任務指標
        for task in ["toxicity", "bullying", "role", "emotion"]:
            if all_predictions[task] and all_labels[task]:
                y_true = np.array(all_labels[task])
                y_pred = np.array(all_predictions[task])
                y_probs = np.array(all_probs[task])

                # 準確率
                metrics[f"{task}_accuracy"] = accuracy_score(y_true, y_pred)

                # Macro F1
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="macro", zero_division=0
                )
                metrics[f"{task}_macro_precision"] = precision
                metrics[f"{task}_macro_recall"] = recall
                metrics[f"{task}_macro_f1"] = f1

                # AUC（多分類使用one-vs-rest）
                try:
                    if len(np.unique(y_true)) > 2:
                        # 多分類AUC
                        auc_scores = []
                        for class_idx in range(y_probs.shape[1]):
                            if class_idx in y_true:  # 只計算存在的類別
                                binary_labels = (y_true == class_idx).astype(int)
                                if len(np.unique(binary_labels)) > 1:  #
                                    確保有正負樣本
                                    auc = roc_auc_score(
                                        binary_labels, y_probs[:, class_idx]
                                    )
                                    auc_scores.append(auc)

                        if auc_scores:
                            metrics[f"{task}_macro_auc"] = np.mean(auc_scores)
                    else:
                        # 二分類AUC
                        if len(np.unique(y_true)) > 1:
                            auc = roc_auc_score(y_true, y_probs[:, 1])
                            metrics[f"{task}_auc"] = auc
                except Exception as e:
                    logger.warning(f"Could not compute AUC for {task}: {e}")

                # AUCPR（平均精確率）
                try:
                    if len(np.unique(y_true)) > 2:
                        # 多分類AUCPR
                        aucpr_scores = []
                        for class_idx in range(y_probs.shape[1]):
                            if class_idx in y_true:
                                binary_labels = (y_true == class_idx).astype(int)
                                if len(np.unique(binary_labels)) > 1:
                                    aucpr = average_precision_score(
                                        binary_labels, y_probs[:, class_idx]
                                    )
                                    aucpr_scores.append(aucpr)

                        if aucpr_scores:
                            metrics[f"{task}_macro_aucpr"] = np.mean(aucpr_scores)
                    else:
                        # 二分類AUCPR
                        if len(np.unique(y_true)) > 1:
                            aucpr = average_precision_score(y_true, y_probs[:, 1])
                            metrics[f"{task}_aucpr"] = aucpr
                except Exception as e:
                    logger.warning(f"Could not compute AUCPR for {task}: {e}")

        # 情緒強度回歸指標
        if all_predictions["emotion_intensity"] and all_labels["emotion_intensity"]:
            y_true_intensity = np.array(all_labels["emotion_intensity"])
            y_pred_intensity = np.array(all_predictions["emotion_intensity"])

            # MSE和MAE
            mse = np.mean((y_true_intensity - y_pred_intensity) ** 2)
            mae = np.mean(np.abs(y_true_intensity - y_pred_intensity))

            metrics["emotion_intensity_mse"] = mse
            metrics["emotion_intensity_mae"] = mae

            # 相關係數
            correlation = np.corrcoef(y_true_intensity, y_pred_intensity)[0, 1]
            if not np.isnan(correlation):
                metrics["emotion_intensity_correlation"] = correlation

        # 會話級F1（針對SCCD）
        if session_ids and len(session_ids) == len(all_predictions["toxicity"]):
            session_f1 = self._compute_session_level_f1(
                all_predictions["toxicity"], all_labels["toxicity"]
            )
            metrics["session_level_f1"] = session_f1

        return metrics

    def _compute_session_level_f1(
        self, predictions: List[int], labels: List[int], session_ids: List[str]
    ) -> float:
        """計算會話級F1分數"""
        from collections import defaultdict

        # 按會話聚合預測結果
        session_preds = defaultdict(list)
        session_labels = defaultdict(list)

        for pred, label, session_id in zip(predictions, labels, session_ids):
            session_preds[session_id].append(pred)
            session_labels[session_id].append(label)

        # 會話級標籤（如果會話中有任何毒性內容，則會話為毒性）
        session_level_preds = []
        session_level_labels = []

        for session_id in session_preds:
            # 預測：會話中是否有毒性內容
            has_toxic_pred = int(max(session_preds[session_id]) > 0)
            has_toxic_label = int(max(session_labels[session_id]) > 0)

            session_level_preds.append(has_toxic_pred)
            session_level_labels.append(has_toxic_label)

        # 計算會話級F1
        if len(set(session_level_labels)) > 1:
            _, _, f1, _ = precision_recall_fscore_support(
                session_level_labels,
                session_level_preds,
                average="bin" "ary",
                zero_division=0,
            )
            return f1
        else:
            return 0.0

    def generate_classification_report(self, dataloader: DataLoader) -> str:
        """生成分類報告"""
        metrics = self.evaluate(dataloader)

        report_lines = ["Classification Report", "=" * 50]

        for task in ["toxicity", "bullying", "role", "emotion"]:
            if f"{task}_macro_f1" in metrics:
                report_lines.append(f"\n{task.upper()}:")
                report_lines.append(
                    f"  Accuracy: {metrics.get(f'{task}_accuracy', 0):.4f}"
                )
                report_lines.append(
                    f"  Macro F1: {metrics.get(f'{task}_macro_f1',
                    0):.4f}"
                )
                report_lines.append(
                    f"  Macro Precision: {metrics.get(
                        f'{task}_macro_precision',
                        0):.4f}"
                )
                report_lines.append(
                    f"  Macro Recall: {metrics.get(f'{task}_macro_recall',
                    0):.4f}"
                )

                if f"{task}_macro_auc" in metrics:
                    report_lines.append(
                        f"  Macro AUC: \
                        {metrics[f'{task}_macro_auc']:.4f}"
                    )
                if f"{task}_macro_aucpr" in metrics:
                    report_lines.append(
                        f"  Macro AUCPR: \
                        {metrics[f'{task}_macro_aucpr']:.4f}"
                    )

        # 情緒強度
        if "emotion_intensity_mse" in metrics:
            report_lines.append("\nEMOTION INTENSITY (Regression):")
            report_lines.append(
                f"  MSE: \
                {metrics['emotion_intensity_mse']:.4f}"
            )
            report_lines.append(
                f"  MAE: \
                {metrics['emotion_intensity_mae']:.4f}"
            )
            if "emotion_intensity_correlation" in metrics:
                report_lines.append(
                    f"  Correlation: \
                        {metrics['emotion_intensity_correlation']:.4f}"
                )

        # 會話級F1
        if "session_level_f1" in metrics:
            report_lines.append("\nSESSION LEVEL:")
            report_lines.append(
                f"  Session F1: \
                {metrics['session_level_f1']:.4f}"
            )

        return "\n".join(report_lines)


def create_model_variants() -> Dict[str, ModelConfig]:
    """建立不同的模型變體配置"""

    variants = {}

    # MacBERT基線
    variants["macbert_base"] = ModelConfig(
        model_name="hfl/chinese-" "macbert-base",
        use_focal_loss=False,
        label_smoothing=0.0,
    )

    # MacBERT + Focal Loss
    variants["macbert_focal"] = ModelConfig(
        model_name="hfl/chinese-" "macbert-base", use_focal_loss=True, focal_gamma=2.0
    )

    # RoBERTa基線
    variants["roberta_base"] = ModelConfig(
        model_name="hfl/chinese-r" "oberta-wwm-ext",
        use_focal_loss=False,
        label_smoothing=0.0,
    )

    # RoBERTa + 高級特徵
    variants["roberta_advanced"] = ModelConfig(
        model_name="hfl/chinese-roberta-wwm-ext",
        use_focal_loss=True,
        focal_gamma=2.0,
        label_smoothing=0.1,
        use_emotion_regression=True,
        class_weights={
            "toxicity": [1.0, 2.0, 3.0],  # 對嚴重毒性給更高權重
            "bullying": [1.0, 2.0, 3.0],
            "emotion": [1.0, 0.8, 1.5],  # 負面情緒權重較高
        },
    )

    # 單任務毒性偵測
    variants["toxicity_only"] = ModelConfig(
        model_name="hfl/chinese-macbert-base",
        task_weights={"toxicity": 1.0, "emotion": 1.0, "bullying": 1.0, "role": 1.0},
    )

    return variants


def main():
    """示例使用"""
    # 建立配置
    config = ModelConfig(
        model_name="hfl/chinese-" "macbert-base",
        use_focal_loss=True,
        use_emotion_regression=True,
    )

    # 建立模型
    model = BaselineModel(config)

    print(f"Model created with backbone: {config.model_name}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # 測試前向傳播
    batch_size = 2
    seq_len = 128

    dummy_input = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "token_type_ids": torch.zeros(batch_size, seq_len),
    }

    outputs = model(**dummy_input)
    print("\nModel outputs:")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape}")


if __name__ == "__main__":
    main()
