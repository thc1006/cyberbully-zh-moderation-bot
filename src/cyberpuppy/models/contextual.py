#!/usr/bin/env python3
"""
上下文感知模型
支援會話級（SCCD）和事件級（CHNCI）建模
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
from transformers import AutoModel, AutoTokenizer

from ..labeling.label_map import (BullyingLevel, RoleType, ToxicityLevel,
                                  UnifiedLabel)

logger = logging.getLogger(__name__)

# Backwards compatibility alias
ContextualBullyingDetector = None  # Will be defined after ContextualModel class


@dataclass
class ContextualInput:
    """上下文輸入資料結構"""

    # 基本文本
    text: str
    # 會話上下文（SCCD）
    thread_context: Optional[List[str]] = None
    # 事件上下文（CHNCI）
    event_context: Optional[Dict[str, Union[str, List[str]]]] = None
    # 角色資訊
    role_info: Optional[Dict[str, str]] = None
    # 時序資訊
    temporal_info: Optional[Dict[str, Union[int, str]]] = None


@dataclass
class ContextualOutput:
    """上下文模型輸出"""

    # 分類結果
    toxicity_logits: torch.Tensor
    bullying_logits: torch.Tensor
    role_logits: torch.Tensor
    emotion_logits: torch.Tensor

    # 特徵表示
    text_embedding: torch.Tensor
    context_embedding: Optional[torch.Tensor] = None
    event_embedding: Optional[torch.Tensor] = None

    # 對比學習
    contrastive_features: Optional[torch.Tensor] = None

    # 注意力權重
    attention_weights: Optional[Dict[str, torch.Tensor]] = None


class HierarchicalThreadEncoder(nn.Module):
    """階層式會話編碼器（針對SCCD）"""

    def __init__(
        self,
        base_model_name: str = "hfl/chinese-macbert-base",
        hidden_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        max_thread_length: int = 16,
        max_message_length: int = 128,
    ):
        super().__init__()

        self.base_model_name = base_model_name
        self.hidden_size = hidden_size
        self.max_thread_length = max_thread_length
        self.max_message_length = max_message_length

        # 基礎文本編碼器
        self.text_encoder = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # 消息級編碼器
        self.message_encoder = nn.TransformerEncoder(
            TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # 位置編碼
        self.position_embedding = nn.Embedding(max_thread_length, hidden_size)

        # 角色編碼
        self.role_embedding = nn.Embedding(4, hidden_size)  # none, perpetrator, victim, bystander

        # 注意力池化
        self.attention_pooler = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=0.1, batch_first=True
        )

        # 輸出投影
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def encode_message(self, message: str) -> torch.Tensor:
        """編碼單個消息"""
        inputs = self.tokenizer(
            message,
            max_length=self.max_message_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 移到相同設備
        device = next(self.text_encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.text_encoder(**inputs)

        # 使用 [CLS] token 的表示
        return outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]

    def forward(
        self, thread_messages: List[str], roles: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        編碼會話上下文

        Args:
            thread_messages: 會話中的消息列表
            roles: 對應的角色列表

        Returns:
            context_embedding: 會話上下文嵌入 [1, hidden_size]
            attention_weights: 注意力權重 [1, seq_len, seq_len]
        """
        device = next(self.text_encoder.parameters()).device

        # 限制會話長度
        thread_messages = thread_messages[-self.max_thread_length :]
        if roles:
            roles = roles[-self.max_thread_length :]

        # 編碼每個消息
        message_embeddings = []
        for message in thread_messages:
            msg_emb = self.encode_message(message)
            message_embeddings.append(msg_emb)

        if not message_embeddings:
            # 空會話處理
            return torch.zeros(1, self.hidden_size).to(device), None

        # 堆疊消息嵌入 [1, seq_len, hidden_size]
        message_embeddings = torch.cat(message_embeddings, dim=0).unsqueeze(0)
        seq_len = message_embeddings.size(1)

        # 添加位置編碼
        positions = torch.arange(seq_len).to(device)
        pos_embeddings = self.position_embedding(positions).unsqueeze(0)
        message_embeddings += pos_embeddings

        # 添加角色編碼
        if roles:
            role_map = {"none": 0, "perpetrator": 1, "victim": 2, "bystander": 3}
            role_ids = torch.tensor([role_map.get(role.lower(), 0) for role in roles]).to(device)
            role_embeddings = self.role_embedding(role_ids).unsqueeze(0)
            message_embeddings += role_embeddings

        # Transformer 編碼
        encoded = self.message_encoder(message_embeddings)  # [1, seq_len, hidden_size]

        # 注意力池化
        query = encoded.mean(dim=1, keepdim=True)  # [1, 1, hidden_size]
        context_emb, attention_weights = self.attention_pooler(query, encoded, encoded)

        # 輸出投影
        context_emb = self.output_projection(context_emb.squeeze(1))  # [1, hidden_size]

        return context_emb, attention_weights


class EventFeatureExtractor(nn.Module):
    """事件級特徵抽取器（針對CHNCI）"""

    def __init__(
        self,
        base_model_name: str = "hfl/chinese-macbert-base",
        hidden_size: int = 768,
        num_event_types: int = 10,
        num_severity_levels: int = 5,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # 基礎文本編碼器
        self.text_encoder = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # 事件類型嵌入
        self.event_type_embedding = nn.Embedding(num_event_types, hidden_size)

        # 嚴重程度嵌入
        self.severity_embedding = nn.Embedding(num_severity_levels, hidden_size)

        # 參與者角色嵌入
        self.participant_embedding = nn.Embedding(
            4, hidden_size
        )  # none, perpetrator, victim, bystander

        # 時序特徵編碼器
        self.temporal_encoder = nn.Linear(8, hidden_size)  # 時間特徵維度

        # 多頭注意力融合
        self.feature_fusion = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )

        # 特徵整合層
        self.integration_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
        )

    def extract_temporal_features(self, temporal_info: Dict[str, Union[int, str]]) -> torch.Tensor:
        """抽取時序特徵"""
        device = next(self.text_encoder.parameters()).device

        # 預設時序特徵
        temporal_features = torch.zeros(8).to(device)

        if temporal_info:
            # 事件持續時間
            if "duration" in temporal_info:
                temporal_features[0] = float(temporal_info["duration"])

            # 事件頻率
            if "frequency" in temporal_info:
                temporal_features[1] = float(temporal_info["frequency"])

            # 時間間隔特徵
            if "time_intervals" in temporal_info:
                intervals = temporal_info["time_intervals"]
                if isinstance(intervals, list) and intervals:
                    temporal_features[2] = float(np.mean(intervals))
                    temporal_features[3] = float(np.std(intervals) if len(intervals) > 1 else 0)

            # 週期性特徵
            if "periodicity" in temporal_info:
                temporal_features[4] = float(temporal_info["periodicity"])

            # 事件密度
            if "event_density" in temporal_info:
                temporal_features[5] = float(temporal_info["event_density"])

        return self.temporal_encoder(temporal_features.unsqueeze(0))  # [1, hidden_size]

    def forward(
        self,
        text: str,
        event_context: Dict[str, Union[str, List[str]]],
        temporal_info: Optional[Dict[str, Union[int, str]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        抽取事件級特徵

        Args:
            text: 主要文本
            event_context: 事件上下文資訊
            temporal_info: 時序資訊

        Returns:
            event_embedding: 事件嵌入 [1, hidden_size]
            feature_weights: 各特徵的注意力權重
        """
        device = next(self.text_encoder.parameters()).device

        # 編碼主要文本
        inputs = self.tokenizer(
            text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        text_outputs = self.text_encoder(**inputs)
        text_embedding = text_outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]

        # 抽取事件特徵
        features = [text_embedding]
        feature_names = ["text"]

        # 事件類型特徵
        if "event_type" in event_context:
            event_type_map = {
                "cyberbullying": 1,
                "harassment": 2,
                "threat": 3,
                "discrimination": 4,
                "hate_speech": 5,
                "normal": 0,
            }
            event_type_id = event_type_map.get(event_context["event_type"], 0)
            event_type_emb = self.event_type_embedding(torch.tensor([event_type_id]).to(device))
            features.append(event_type_emb)
            feature_names.append("event_type")

        # 嚴重程度特徵
        if "severity" in event_context:
            severity_map = {"none": 0, "low": 1, "moderate": 2, "high": 3, "severe": 4}
            severity_id = severity_map.get(event_context["severity"], 0)
            severity_emb = self.severity_embedding(torch.tensor([severity_id]).to(device))
            features.append(severity_emb)
            feature_names.append("severity")

        # 參與者角色特徵
        if "participants" in event_context:
            participants = event_context["participants"]
            if isinstance(participants, list):
                role_map = {"perpetrator": 1, "victim": 2, "bystander": 3, "none": 0}
                # 使用主要角色或多個角色的平均
                role_ids = [role_map.get(p, 0) for p in participants]
                main_role_id = max(set(role_ids), key=role_ids.count) if role_ids else 0
                participant_emb = self.participant_embedding(
                    torch.tensor([main_role_id]).to(device)
                )
                features.append(participant_emb)
                feature_names.append("participants")

        # 時序特徵
        if temporal_info:
            temporal_emb = self.extract_temporal_features(temporal_info)
            features.append(temporal_emb)
            feature_names.append("temporal")

        # 特徵融合
        if len(features) > 1:
            feature_stack = torch.stack(features, dim=1)  # [1, num_features, hidden_size]

            # 使用注意力融合特徵
            query = feature_stack[:, 0:1, :]  # 使用文本作為查詢
            fused_features, attention_weights = self.feature_fusion(
                query, feature_stack, feature_stack
            )

            # 整合文本和融合特徵
            combined = torch.cat([text_embedding, fused_features.squeeze(1)], dim=-1)
            event_embedding = self.integration_layer(combined)

            # 整理注意力權重
            feature_weights = {
                name: attention_weights[0, 0, i].item() for i, name in enumerate(feature_names)
            }
        else:
            event_embedding = text_embedding
            feature_weights = {"text": 1.0}

        return event_embedding, feature_weights


class ContrastiveLearningModule(nn.Module):
    """對比學習模組"""

    def __init__(
        self,
        embedding_dim: int = 768,
        temperature: float = 0.1,
        projection_dim: int = 128,
    ):
        super().__init__()

        self.temperature = temperature
        self.projection_dim = projection_dim

        # 投影頭
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim * 2, projection_dim),
        )

        # L2 正規化
        self.l2_normalize = nn.functional.normalize

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        對比學習前向傳播

        Args:
            embeddings: 輸入嵌入 [batch_size, embedding_dim]

        Returns:
            projected_embeddings: 投影後的嵌入 [batch_size, projection_dim]
        """
        # 投影
        projected = self.projection_head(embeddings)

        # L2 正規化
        projected = self.l2_normalize(projected, dim=-1)

        return projected

    def compute_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        session_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        計算對比學習損失

        Args:
            embeddings: 投影後的嵌入 [batch_size, projection_dim]
            labels: 標籤 [batch_size]
            session_ids: 會話/事件ID [batch_size]（可選）

        Returns:
            contrastive_loss: 對比損失
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        # 計算相似度矩陣
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # 建立正負樣本掩碼
        if session_ids is not None:
            # 同會話/事件的樣本作為正樣本
            session_mask = (session_ids.unsqueeze(0) == session_ids.unsqueeze(1)).float()
            # 同標籤的樣本也作為正樣本
            label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            positive_mask = torch.max(session_mask, label_mask)
        else:
            # 僅使用標籤作為正樣本
            positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # 移除對角線（自己與自己的相似度）
        positive_mask.fill_diagonal_(0)

        # 負樣本掩碼
        negative_mask = 1.0 - positive_mask
        negative_mask.fill_diagonal_(0)

        # 計算損失
        # 對每個樣本，計算與正樣本的相似度和與所有樣本的相似度
        losses = []
        for i in range(batch_size):
            # 正樣本相似度
            pos_similarities = similarity_matrix[i] * positive_mask[i]
            pos_similarities = pos_similarities[pos_similarities > 0]

            if len(pos_similarities) == 0:
                continue  # 沒有正樣本，跳過

            # 負樣本相似度
            neg_similarities = similarity_matrix[i] * negative_mask[i]
            neg_similarities = neg_similarities[neg_similarities != 0]

            # 計算對比損失（InfoNCE）
            pos_exp = torch.exp(pos_similarities)
            neg_exp = torch.exp(neg_similarities)

            denominator = neg_exp.sum() + pos_exp.sum()
            loss = -torch.log(pos_exp.sum() / denominator)
            losses.append(loss)

        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class ContextualModel(nn.Module):
    """上下文感知的多任務分類模型"""

    def __init__(
        self,
        base_model_name: str = "hfl/chinese-macbert-base",
        hidden_size: int = 768,
        num_toxicity_classes: int = 3,  # none, toxic, severe
        num_bullying_classes: int = 3,  # none, harassment, threat
        num_role_classes: int = 4,  # none, perpetrator, victim, bystander
        num_emotion_classes: int = 3,  # positive, neutral, negative
        use_contrastive: bool = True,
        contrastive_temperature: float = 0.1,
    ):
        super().__init__()

        self.base_model_name = base_model_name
        self.hidden_size = hidden_size
        self.use_contrastive = use_contrastive

        # 基礎文本編碼器
        self.text_encoder = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # 上下文編碼器
        self.thread_encoder = HierarchicalThreadEncoder(
            base_model_name=base_model_name, hidden_size=hidden_size
        )

        # 事件特徵抽取器
        self.event_extractor = EventFeatureExtractor(
            base_model_name=base_model_name, hidden_size=hidden_size
        )

        # 特徵融合層
        self.context_fusion = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=8, dropout=0.1, batch_first=True
        )

        # 分類頭
        self.toxicity_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_toxicity_classes),
        )

        self.bullying_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_bullying_classes),
        )

        self.role_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_role_classes),
        )

        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_emotion_classes),
        )

        # 對比學習模組
        if use_contrastive:
            self.contrastive_module = ContrastiveLearningModule(
                embedding_dim=hidden_size, temperature=contrastive_temperature
            )

    def encode_text(self, text: str) -> torch.Tensor:
        """編碼基礎文本"""
        inputs = self.tokenizer(
            text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        device = next(self.text_encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]

    def forward(
        self, contextual_inputs: List[ContextualInput], return_embeddings: bool = False
    ) -> Union[ContextualOutput, List[ContextualOutput]]:
        """
        前向傳播

        Args:
            contextual_inputs: 上下文輸入列表
            return_embeddings: 是否返回嵌入表示

        Returns:
            outputs: 模型輸出
        """
        batch_outputs = []
        all_embeddings = []

        for ctx_input in contextual_inputs:
            # 編碼基礎文本
            text_emb = self.encode_text(ctx_input.text)

            embeddings_list = [text_emb]
            embedding_names = ["text"]

            # 編碼會話上下文（SCCD）
            context_emb = None
            thread_attention = None
            if ctx_input.thread_context:
                roles = None
                if ctx_input.role_info and "thread_roles" in ctx_input.role_info:
                    roles = ctx_input.role_info["thread_roles"]

                context_emb, thread_attention = self.thread_encoder(ctx_input.thread_context, roles)
                embeddings_list.append(context_emb)
                embedding_names.append("context")

            # 編碼事件特徵（CHNCI）
            event_emb = None
            event_weights = None
            if ctx_input.event_context:
                event_emb, event_weights = self.event_extractor(
                    ctx_input.text, ctx_input.event_context, ctx_input.temporal_info
                )
                embeddings_list.append(event_emb)
                embedding_names.append("event")

            # 特徵融合
            if len(embeddings_list) > 1:
                feature_stack = torch.stack(
                    embeddings_list, dim=1
                )  # [1, num_features, hidden_size]
                query = feature_stack[:, 0:1, :]  # 使用文本作為查詢

                fused_emb, fusion_attention = self.context_fusion(
                    query, feature_stack, feature_stack
                )
                final_embedding = fused_emb.squeeze(1)  # [1, hidden_size]

                # 注意力權重
                attention_weights = {
                    "fusion": fusion_attention,
                    "thread": thread_attention,
                    "event": event_weights,
                }
            else:
                final_embedding = text_emb
                attention_weights = None

            all_embeddings.append(final_embedding)

            # 分類預測
            toxicity_logits = self.toxicity_classifier(final_embedding)
            bullying_logits = self.bullying_classifier(final_embedding)
            role_logits = self.role_classifier(final_embedding)
            emotion_logits = self.emotion_classifier(final_embedding)

            # 對比學習特徵
            contrastive_features = None
            if self.use_contrastive and hasattr(self, "contrastive_module"):
                contrastive_features = self.contrastive_module(final_embedding)

            # 構建輸出
            output = ContextualOutput(
                toxicity_logits=toxicity_logits,
                bullying_logits=bullying_logits,
                role_logits=role_logits,
                emotion_logits=emotion_logits,
                text_embedding=text_emb,
                context_embedding=context_emb,
                event_embedding=event_emb,
                contrastive_features=contrastive_features,
                attention_weights=attention_weights,
            )

            batch_outputs.append(output)

        # 返回結果
        if len(batch_outputs) == 1:
            return batch_outputs[0]

        return batch_outputs

    def compute_loss(
        self,
        outputs: Union[ContextualOutput, List[ContextualOutput]],
        labels: Union[UnifiedLabel, List[UnifiedLabel]],
        session_ids: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        計算損失

        Args:
            outputs: 模型輸出
            labels: 真實標籤
            session_ids: 會話/事件ID（用於對比學習）
            loss_weights: 各任務損失權重

        Returns:
            losses: 各項損失
        """
        if not isinstance(outputs, list):
            outputs = [outputs]
        if not isinstance(labels, list):
            labels = [labels]

        device = next(self.parameters()).device

        # 預設損失權重
        if loss_weights is None:
            loss_weights = {
                "toxicity": 1.0,
                "bullying": 1.0,
                "role": 0.5,
                "emotion": 0.8,
                "contrastive": 0.3,
            }

        # 收集批次數據
        batch_toxicity_logits = []
        batch_bullying_logits = []
        batch_role_logits = []
        batch_emotion_logits = []
        batch_contrastive_features = []

        batch_toxicity_labels = []
        batch_bullying_labels = []
        batch_role_labels = []
        batch_emotion_labels = []

        # 標籤映射
        toxicity_map = {
            ToxicityLevel.NONE: 0,
            ToxicityLevel.TOXIC: 1,
            ToxicityLevel.SEVERE: 2,
        }
        bullying_map = {
            BullyingLevel.NONE: 0,
            BullyingLevel.HARASSMENT: 1,
            BullyingLevel.THREAT: 2,
        }
        role_map = {
            RoleType.NONE: 0,
            RoleType.PERPETRATOR: 1,
            RoleType.VICTIM: 2,
            RoleType.BYSTANDER: 3,
        }
        emotion_map = {"pos": 0, "neu": 1, "neg": 2}

        for output, label in zip(outputs, labels):
            batch_toxicity_logits.append(output.toxicity_logits)
            batch_bullying_logits.append(output.bullying_logits)
            batch_role_logits.append(output.role_logits)
            batch_emotion_logits.append(output.emotion_logits)

            if output.contrastive_features is not None:
                batch_contrastive_features.append(output.contrastive_features)

            # 轉換標籤
            batch_toxicity_labels.append(toxicity_map.get(label.toxicity, 0))
            batch_bullying_labels.append(bullying_map.get(label.bullying, 0))
            batch_role_labels.append(role_map.get(label.role, 0))
            batch_emotion_labels.append(emotion_map.get(label.emotion.value, 1))

        # 堆疊批次張量
        toxicity_logits = torch.cat(batch_toxicity_logits, dim=0)
        bullying_logits = torch.cat(batch_bullying_logits, dim=0)
        role_logits = torch.cat(batch_role_logits, dim=0)
        emotion_logits = torch.cat(batch_emotion_logits, dim=0)

        toxicity_labels = torch.tensor(batch_toxicity_labels, device=device)
        bullying_labels = torch.tensor(batch_bullying_labels, device=device)
        role_labels = torch.tensor(batch_role_labels, device=device)
        emotion_labels = torch.tensor(batch_emotion_labels, device=device)

        # 計算分類損失
        losses = {}

        losses["toxicity"] = (
            F.cross_entropy(toxicity_logits, toxicity_labels) * loss_weights["toxicity"]
        )
        losses["bullying"] = (
            F.cross_entropy(bullying_logits, bullying_labels) * loss_weights["bullying"]
        )
        losses["role"] = F.cross_entropy(role_logits, role_labels) * loss_weights["role"]
        losses["emotion"] = (
            F.cross_entropy(emotion_logits, emotion_labels) * loss_weights["emotion"]
        )

        # 對比學習損失
        if batch_contrastive_features and self.use_contrastive:
            contrastive_features = torch.cat(batch_contrastive_features, dim=0)

            # 使用毒性標籤作為主要對比標籤
            contrastive_loss = self.contrastive_module.compute_contrastive_loss(
                contrastive_features, toxicity_labels, session_ids
            )
            losses["contrastive"] = contrastive_loss * loss_weights["contrastive"]

        # 總損失
        losses["total"] = sum(losses.values())

        return losses

    def predict(self, contextual_input: ContextualInput) -> Dict[str, Union[str, float]]:
        """
        預測單個樣本

        Args:
            contextual_input: 上下文輸入

        Returns:
            predictions: 預測結果
        """
        self.eval()
        with torch.no_grad():
            output = self.forward([contextual_input])

            # 轉換預測結果
            toxicity_pred = torch.argmax(output.toxicity_logits, dim=-1).item()
            bullying_pred = torch.argmax(output.bullying_logits, dim=-1).item()
            role_pred = torch.argmax(output.role_logits, dim=-1).item()
            emotion_pred = torch.argmax(output.emotion_logits, dim=-1).item()

            # 映射回標籤
            toxicity_map = {0: "none", 1: "toxic", 2: "severe"}
            bullying_map = {0: "none", 1: "harassment", 2: "threat"}
            role_map = {0: "none", 1: "perpetrator", 2: "victim", 3: "bystander"}
            emotion_map = {0: "pos", 1: "neu", 2: "neg"}

            # 計算置信度
            toxicity_conf = torch.softmax(output.toxicity_logits, dim=-1).max().item()
            bullying_conf = torch.softmax(output.bullying_logits, dim=-1).max().item()
            role_conf = torch.softmax(output.role_logits, dim=-1).max().item()
            emotion_conf = torch.softmax(output.emotion_logits, dim=-1).max().item()

            return {
                "toxicity": toxicity_map[toxicity_pred],
                "toxicity_confidence": toxicity_conf,
                "bullying": bullying_map[bullying_pred],
                "bullying_confidence": bullying_conf,
                "role": role_map[role_pred],
                "role_confidence": role_conf,
                "emotion": emotion_map[emotion_pred],
                "emotion_confidence": emotion_conf,
            }


def main():
    """示例使用"""
    # 初始化模型
    model = ContextualModel(use_contrastive=True)

    # 創建測試輸入
    ctx_inputs = [
        ContextualInput(
            text="你這個垃圾去死",
            thread_context=["大家好", "你好", "你這個垃圾去死"],
            role_info={"thread_roles": ["none", "none", "perpetrator"]},
            event_context={
                "event_type": "cyberbullying",
                "severity": "high",
                "participants": ["perpetrator", "victim"],
            },
            temporal_info={"duration": 300},
        ),
        ContextualInput(
            text="今天天氣很好",
            thread_context=["今天怎麼樣", "今天天氣很好"],
            event_context={"event_type": "normal", "severity": "none"},
        ),
    ]

    # 前向傳播
    outputs = model(ctx_inputs)
    print(f"處理了 {len(outputs)} 個樣本")

    # 單個預測
    prediction = model.predict(ctx_inputs[0])
    print(f"預測結果: {prediction}")


# Backwards compatibility alias
ContextualBullyingDetector = ContextualModel


if __name__ == "__main__":
    main()
