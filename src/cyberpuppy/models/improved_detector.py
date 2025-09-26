#!/usr/bin/env python3
"""
改進的霸凌偵測模型架構
針對F1分數從0.55提升至0.75+而設計

主要改進:
1. 高級損失函數 (Focal Loss, Label Smoothing, Class-Balanced Loss)
2. 增強注意力機制 (Multi-head Cross-attention, Self-attention)
3. 動態任務權重學習
4. 對抗訓練和不確定性估計
5. 殘差連接和層正規化
6. 知識蒸餾和模型集成
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ImprovedModelConfig:
    """改進模型配置"""

    # 基礎配置
    model_name: str = "hfl/chinese-macbert-base"
    hidden_size: int = 768
    max_length: int = 512

    # 任務配置
    num_toxicity_classes: int = 3
    num_bullying_classes: int = 3
    num_role_classes: int = 4
    num_emotion_classes: int = 3

    # 注意力配置
    num_attention_heads: int = 12
    attention_dropout: float = 0.1
    use_cross_attention: bool = True
    use_self_attention: bool = True

    # 損失函數配置
    use_focal_loss: bool = True
    focal_alpha: List[float] = None  # 每個類別的權重
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    use_class_balanced_loss: bool = True

    # 正規化配置
    dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-12
    use_gradient_clipping: bool = True
    max_grad_norm: float = 1.0

    # 對抗訓練配置
    use_adversarial_training: bool = True
    adversarial_epsilon: float = 0.01
    adversarial_alpha: float = 0.3

    # 動態權重配置
    use_dynamic_task_weighting: bool = True
    task_weight_lr: float = 0.01
    temperature_scaling: bool = True

    # 不確定性估計
    use_uncertainty_estimation: bool = True
    monte_carlo_dropout: bool = True
    mc_samples: int = 10

    # 知識蒸餾
    use_knowledge_distillation: bool = False
    teacher_temperature: float = 4.0
    distillation_alpha: float = 0.7

    def __post_init__(self):
        if self.focal_alpha is None:
            # 預設類別權重 (根據類別不平衡調整)
            self.focal_alpha = [0.25, 0.5, 0.25]  # 對毒性類別的權重分配


class ClassBalancedFocalLoss(nn.Module):
    """類別平衡焦點損失"""

    def __init__(self, alpha: List[float], gamma: float = 2.0, beta: float = 0.9999):
        super().__init__()
        self.alpha = torch.tensor(alpha) if alpha else None
        self.gamma = gamma
        self.beta = beta

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                class_counts: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, num_classes] 預測邏輯值
            targets: [batch_size] 真實標籤
            class_counts: [num_classes] 每個類別的樣本數
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Focal loss權重
        focal_weight = (1 - pt) ** self.gamma

        # Alpha權重 - 確保在正確的設備上
        if self.alpha is not None:
            # 將alpha移到正確設備，將targets移到CPU進行索引
            alpha_t = self.alpha[targets.cpu()].to(inputs.device)
            focal_weight = alpha_t * focal_weight

        # 類別平衡權重
        if class_counts is not None:
            effective_num = 1.0 - torch.pow(self.beta, class_counts)
            cb_weights = (1.0 - self.beta) / effective_num
            cb_weights = cb_weights / cb_weights.sum() * len(cb_weights)
            cb_weight_t = cb_weights[targets].to(inputs.device)
            focal_weight = focal_weight * cb_weight_t

        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()


class DynamicTaskWeighting(nn.Module):
    """動態任務權重學習"""

    def __init__(self, num_tasks: int, init_temp: float = 1.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.temp = init_temp

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        使用不確定性權重動態平衡多任務損失
        Reference: Multi-Task Learning Using Uncertainty to Weigh Losses
        """
        stacked_losses = torch.stack(losses)
        precision = torch.exp(-self.log_vars)
        weighted_losses = precision * stacked_losses + self.log_vars
        return weighted_losses.sum()

    def get_weights(self) -> torch.Tensor:
        """獲取當前任務權重"""
        return F.softmax(-self.log_vars / self.temp, dim=0)


class MultiHeadCrossAttention(nn.Module):
    """多頭交叉注意力機制"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch_size, query_len, hidden_size]
            key: [batch_size, key_len, hidden_size]
            value: [batch_size, value_len, hidden_size]
            mask: [batch_size, query_len, key_len]
        """
        batch_size, query_len = query.size(0), query.size(1)
        key_len = key.size(1)

        # 線性變換
        Q = self.query_proj(query)  # [batch_size, query_len, hidden_size]
        K = self.key_proj(key)      # [batch_size, key_len, hidden_size]
        V = self.value_proj(value)  # [batch_size, value_len, hidden_size]

        # 重塑為多頭
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 計算注意力分數
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 應用遮罩
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores.masked_fill_(mask == 0, -1e9)

        # 注意力權重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加權求和
        attn_output = torch.matmul(attn_weights, V)  # [batch_size, num_heads, query_len, head_dim]

        # 合併多頭
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.hidden_size
        )

        # 輸出投影和殘差連接
        output = self.out_proj(attn_output)
        output = self.layer_norm(output + query)

        return output


class EnhancedFeatureExtractor(nn.Module):
    """增強特徵提取器"""

    def __init__(self, config: ImprovedModelConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        # 基礎編碼器
        self.backbone = AutoModel.from_pretrained(config.model_name)

        # 自注意力層
        if config.use_self_attention:
            self.self_attention = MultiHeadCrossAttention(
                hidden_size, config.num_attention_heads, config.attention_dropout
            )

        # 交叉注意力層
        if config.use_cross_attention:
            self.cross_attention = MultiHeadCrossAttention(
                hidden_size, config.num_attention_heads, config.attention_dropout
            )

        # 特徵融合層
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        )

        # 位置感知池化
        self.position_aware_pooling = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                context_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            context_embeddings: [batch_size, context_len, hidden_size] 上下文嵌入
        """
        # 基礎編碼
        backbone_outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = backbone_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 自注意力增強
        if hasattr(self, 'self_attention'):
            enhanced_states = self.self_attention(hidden_states, hidden_states, hidden_states)
        else:
            enhanced_states = hidden_states

        # 交叉注意力融合上下文
        if hasattr(self, 'cross_attention') and context_embeddings is not None:
            context_enhanced = self.cross_attention(
                query=enhanced_states,
                key=context_embeddings,
                value=context_embeddings
            )
        else:
            context_enhanced = enhanced_states

        # 位置感知池化
        query = context_enhanced.mean(dim=1, keepdim=True)  # [batch_size, 1, hidden_size]
        pooled_output, attention_weights = self.position_aware_pooling(
            query, context_enhanced, context_enhanced,
            key_padding_mask=~attention_mask.bool()
        )
        pooled_output = pooled_output.squeeze(1)  # [batch_size, hidden_size]

        # 特徵融合
        final_features = self.feature_fusion(pooled_output)

        return {
            "features": final_features,
            "hidden_states": context_enhanced,
            "attention_weights": attention_weights,
            "pooled_output": pooled_output
        }


class UncertaintyEstimator(nn.Module):
    """不確定性估計器"""

    def __init__(self, hidden_size: int, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout_rate = dropout_rate

        # 分類頭（帶Dropout）
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # 不確定性頭
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor,
                mc_samples: int = 10, training: bool = True) -> Dict[str, torch.Tensor]:
        """
        使用Monte Carlo Dropout估計不確定性
        """
        if not training:
            # 測試時使用MC Dropout
            predictions = []
            uncertainties = []

            for _ in range(mc_samples):
                # 啟用dropout進行隨機採樣
                pred = self.classifier(features)
                uncertainty = self.uncertainty_head(features)

                predictions.append(F.softmax(pred, dim=-1))
                uncertainties.append(uncertainty)

            # 統計結果
            pred_mean = torch.stack(predictions).mean(dim=0)
            pred_var = torch.stack(predictions).var(dim=0)
            uncertainty_mean = torch.stack(uncertainties).mean(dim=0)

            # 總不確定性 = 認知不確定性 + 偶然不確定性
            epistemic_uncertainty = pred_var.sum(dim=-1, keepdim=True)
            aleatoric_uncertainty = uncertainty_mean
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

            return {
                "logits": torch.log(pred_mean + 1e-8),
                "predictions": pred_mean,
                "epistemic_uncertainty": epistemic_uncertainty,
                "aleatoric_uncertainty": aleatoric_uncertainty,
                "total_uncertainty": total_uncertainty
            }
        else:
            # 訓練時正常前向傳播
            logits = self.classifier(features)
            uncertainty = self.uncertainty_head(features)

            return {
                "logits": logits,
                "predictions": F.softmax(logits, dim=-1),
                "aleatoric_uncertainty": uncertainty
            }


class AdversarialTraining(nn.Module):
    """對抗訓練模組"""

    def __init__(self, epsilon: float = 0.01, alpha: float = 0.3):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha

    def fgsm_attack(self, embeddings: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        """Fast Gradient Sign Method攻擊"""
        # 計算梯度
        grad = torch.autograd.grad(
            loss, embeddings, retain_graph=True, create_graph=True
        )[0]

        # 生成對抗樣本
        perturbation = self.epsilon * grad.sign()
        adversarial_embeddings = embeddings + perturbation

        return adversarial_embeddings

    def pgd_attack(self, embeddings: torch.Tensor, loss: torch.Tensor,
                   num_steps: int = 3) -> torch.Tensor:
        """Projected Gradient Descent攻擊"""
        adv_embeddings = embeddings.clone().detach()

        for _ in range(num_steps):
            adv_embeddings.requires_grad_(True)

            # 計算梯度
            grad = torch.autograd.grad(
                loss, adv_embeddings, retain_graph=True, create_graph=True
            )[0]

            # 更新對抗樣本
            adv_embeddings = adv_embeddings + self.alpha * grad.sign()

            # 投影到epsilon球內
            delta = torch.clamp(adv_embeddings - embeddings, -self.epsilon, self.epsilon)
            adv_embeddings = embeddings + delta
            adv_embeddings = adv_embeddings.detach()

        return adv_embeddings


class ImprovedDetector(nn.Module):
    """改進的霸凌偵測模型"""

    def __init__(self, config: ImprovedModelConfig):
        super().__init__()
        self.config = config

        # 分詞器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # 特徵提取器
        self.feature_extractor = EnhancedFeatureExtractor(config)

        # 不確定性估計器（每個任務一個）
        self.toxicity_estimator = UncertaintyEstimator(
            config.hidden_size, config.num_toxicity_classes, config.dropout_rate
        )
        self.bullying_estimator = UncertaintyEstimator(
            config.hidden_size, config.num_bullying_classes, config.dropout_rate
        )
        self.role_estimator = UncertaintyEstimator(
            config.hidden_size, config.num_role_classes, config.dropout_rate
        )
        self.emotion_estimator = UncertaintyEstimator(
            config.hidden_size, config.num_emotion_classes, config.dropout_rate
        )

        # 損失函數
        self._setup_loss_functions()

        # 動態任務權重
        if config.use_dynamic_task_weighting:
            self.task_weighting = DynamicTaskWeighting(num_tasks=4)

        # 對抗訓練
        if config.use_adversarial_training:
            self.adversarial_trainer = AdversarialTraining(
                config.adversarial_epsilon, config.adversarial_alpha
            )

        # 溫度縮放
        if config.temperature_scaling:
            self.temperature = nn.Parameter(torch.ones(4))  # 每個任務一個溫度參數

    def _setup_loss_functions(self):
        """設定損失函數"""
        config = self.config

        # 類別平衡焦點損失
        if config.use_class_balanced_loss and config.use_focal_loss:
            self.toxicity_loss_fn = ClassBalancedFocalLoss(
                config.focal_alpha, config.focal_gamma
            )
            self.bullying_loss_fn = ClassBalancedFocalLoss(
                config.focal_alpha, config.focal_gamma
            )
            self.role_loss_fn = ClassBalancedFocalLoss(
                [0.4, 0.2, 0.2, 0.2], config.focal_gamma  # 角色類別權重
            )
            self.emotion_loss_fn = ClassBalancedFocalLoss(
                [0.3, 0.4, 0.3], config.focal_gamma  # 情緒類別權重
            )
        elif config.use_focal_loss:
            self.toxicity_loss_fn = ClassBalancedFocalLoss(
                config.focal_alpha, config.focal_gamma
            )
            self.bullying_loss_fn = ClassBalancedFocalLoss(
                config.focal_alpha, config.focal_gamma
            )
            self.role_loss_fn = ClassBalancedFocalLoss(
                [0.4, 0.2, 0.2, 0.2], config.focal_gamma
            )
            self.emotion_loss_fn = ClassBalancedFocalLoss(
                [0.3, 0.4, 0.3], config.focal_gamma
            )
        else:
            # 標準損失函數
            self.toxicity_loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
            self.bullying_loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
            self.role_loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
            self.emotion_loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                context_embeddings: Optional[torch.Tensor] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """前向傳播"""
        # 特徵提取
        feature_outputs = self.feature_extractor(
            input_ids, attention_mask, context_embeddings
        )
        features = feature_outputs["features"]

        # 各任務預測（帶不確定性估計）
        toxicity_output = self.toxicity_estimator(
            features, self.config.mc_samples, self.training
        )
        bullying_output = self.bullying_estimator(
            features, self.config.mc_samples, self.training
        )
        role_output = self.role_estimator(
            features, self.config.mc_samples, self.training
        )
        emotion_output = self.emotion_estimator(
            features, self.config.mc_samples, self.training
        )

        # 溫度縮放
        if hasattr(self, 'temperature') and self.config.temperature_scaling:
            toxicity_output["logits"] = toxicity_output["logits"] / self.temperature[0]
            bullying_output["logits"] = bullying_output["logits"] / self.temperature[1]
            role_output["logits"] = role_output["logits"] / self.temperature[2]
            emotion_output["logits"] = emotion_output["logits"] / self.temperature[3]

        outputs = {
            "toxicity": toxicity_output["logits"],
            "bullying": bullying_output["logits"],
            "role": role_output["logits"],
            "emotion": emotion_output["logits"],
            "toxicity_uncertainty": toxicity_output.get("total_uncertainty"),
            "bullying_uncertainty": bullying_output.get("total_uncertainty"),
            "role_uncertainty": role_output.get("total_uncertainty"),
            "emotion_uncertainty": emotion_output.get("total_uncertainty"),
        }

        if return_features:
            outputs.update(feature_outputs)

        return outputs

    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                     labels: Dict[str, torch.Tensor],
                     class_counts: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """計算損失"""
        losses = []
        loss_dict = {}

        # 各任務損失
        if "toxicity" in outputs and "toxicity_label" in labels:
            toxicity_loss = self.toxicity_loss_fn(
                outputs["toxicity"], labels["toxicity_label"],
                class_counts.get("toxicity") if class_counts else None
            )
            losses.append(toxicity_loss)
            loss_dict["toxicity"] = toxicity_loss

        if "bullying" in outputs and "bullying_label" in labels:
            bullying_loss = self.bullying_loss_fn(
                outputs["bullying"], labels["bullying_label"],
                class_counts.get("bullying") if class_counts else None
            )
            losses.append(bullying_loss)
            loss_dict["bullying"] = bullying_loss

        if "role" in outputs and "role_label" in labels:
            role_loss = self.role_loss_fn(
                outputs["role"], labels["role_label"],
                class_counts.get("role") if class_counts else None
            )
            losses.append(role_loss)
            loss_dict["role"] = role_loss

        if "emotion" in outputs and "emotion_label" in labels:
            emotion_loss = self.emotion_loss_fn(
                outputs["emotion"], labels["emotion_label"],
                class_counts.get("emotion") if class_counts else None
            )
            losses.append(emotion_loss)
            loss_dict["emotion"] = emotion_loss

        # 動態權重總損失
        if hasattr(self, 'task_weighting') and self.config.use_dynamic_task_weighting:
            total_loss = self.task_weighting(losses)
            loss_dict["task_weights"] = self.task_weighting.get_weights()
        else:
            total_loss = sum(losses)

        loss_dict["total"] = total_loss
        return loss_dict

    def adversarial_training_step(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                                  labels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """對抗訓練步驟"""
        if not (hasattr(self, 'adversarial_trainer') and self.config.use_adversarial_training):
            return self.compute_loss(self.forward(input_ids, attention_mask), labels)

        # 正常前向傳播
        embeddings = self.feature_extractor.backbone.embeddings(input_ids)
        embeddings.requires_grad_(True)

        # 使用嵌入計算輸出
        backbone_outputs = self.feature_extractor.backbone(inputs_embeds=embeddings, attention_mask=attention_mask)
        feature_outputs = {
            "features": self.feature_extractor.feature_fusion(backbone_outputs.last_hidden_state.mean(dim=1))
        }

        # 正常損失
        normal_outputs = {}
        normal_outputs["toxicity"] = self.toxicity_estimator(feature_outputs["features"], training=True)["logits"]
        normal_outputs["bullying"] = self.bullying_estimator(feature_outputs["features"], training=True)["logits"]
        normal_outputs["role"] = self.role_estimator(feature_outputs["features"], training=True)["logits"]
        normal_outputs["emotion"] = self.emotion_estimator(feature_outputs["features"], training=True)["logits"]

        normal_loss_dict = self.compute_loss(normal_outputs, labels)
        normal_loss = normal_loss_dict["total"]

        # 生成對抗樣本
        adv_embeddings = self.adversarial_trainer.fgsm_attack(embeddings, normal_loss)

        # 對抗樣本前向傳播
        adv_backbone_outputs = self.feature_extractor.backbone(inputs_embeds=adv_embeddings, attention_mask=attention_mask)
        adv_feature_outputs = {
            "features": self.feature_extractor.feature_fusion(adv_backbone_outputs.last_hidden_state.mean(dim=1))
        }

        adv_outputs = {}
        adv_outputs["toxicity"] = self.toxicity_estimator(adv_feature_outputs["features"], training=True)["logits"]
        adv_outputs["bullying"] = self.bullying_estimator(adv_feature_outputs["features"], training=True)["logits"]
        adv_outputs["role"] = self.role_estimator(adv_feature_outputs["features"], training=True)["logits"]
        adv_outputs["emotion"] = self.emotion_estimator(adv_feature_outputs["features"], training=True)["logits"]

        adv_loss_dict = self.compute_loss(adv_outputs, labels)
        adv_loss = adv_loss_dict["total"]

        # 組合損失
        total_loss = normal_loss + self.config.adversarial_alpha * adv_loss

        loss_dict = normal_loss_dict.copy()
        loss_dict["adversarial_loss"] = adv_loss
        loss_dict["total"] = total_loss

        return loss_dict

    def predict(self, text: str, context: Optional[List[str]] = None) -> Dict[str, Union[str, float, torch.Tensor]]:
        """預測單個文本"""
        self.eval()

        # 分詞
        inputs = self.tokenizer(
            text, max_length=self.config.max_length,
            truncation=True, padding="max_length", return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.forward(
                inputs["input_ids"],
                inputs["attention_mask"],
                return_features=False
            )

            # 轉換預測結果
            predictions = {}

            # 各任務預測
            task_maps = {
                "toxicity": {0: "none", 1: "toxic", 2: "severe"},
                "bullying": {0: "none", 1: "harassment", 2: "threat"},
                "role": {0: "none", 1: "perpetrator", 2: "victim", 3: "bystander"},
                "emotion": {0: "pos", 1: "neu", 2: "neg"}
            }

            for task in ["toxicity", "bullying", "role", "emotion"]:
                if task in outputs:
                    logits = outputs[task]
                    probs = F.softmax(logits, dim=-1)
                    pred_idx = torch.argmax(logits, dim=-1).item()
                    confidence = probs.max().item()

                    predictions[f"{task}_prediction"] = task_maps[task][pred_idx]
                    predictions[f"{task}_confidence"] = confidence
                    predictions[f"{task}_probabilities"] = probs.squeeze().tolist()

                    # 不確定性
                    if f"{task}_uncertainty" in outputs and outputs[f"{task}_uncertainty"] is not None:
                        predictions[f"{task}_uncertainty"] = outputs[f"{task}_uncertainty"].item()

        return predictions


def create_improved_config() -> ImprovedModelConfig:
    """創建針對F1優化的配置"""
    return ImprovedModelConfig(
        model_name="hfl/chinese-macbert-base",
        hidden_size=768,
        num_attention_heads=12,

        # 啟用所有改進功能
        use_cross_attention=True,
        use_self_attention=True,
        use_focal_loss=True,
        use_class_balanced_loss=True,
        use_adversarial_training=True,
        use_dynamic_task_weighting=True,
        use_uncertainty_estimation=True,
        temperature_scaling=True,

        # 優化超參數
        focal_gamma=2.0,
        label_smoothing=0.1,
        dropout_rate=0.1,
        adversarial_epsilon=0.01,
        adversarial_alpha=0.3,

        # Monte Carlo設定
        monte_carlo_dropout=True,
        mc_samples=10
    )


if __name__ == "__main__":
    # 示例使用
    config = create_improved_config()
    model = ImprovedDetector(config)

    print(f"改進模型已創建，參數量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可訓練參數: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 測試預測
    sample_text = "你這個垃圾去死"
    prediction = model.predict(sample_text)
    print(f"\n預測結果: {prediction}")