#!/usr/bin/env python3
"""
測試上下文感知模型
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.cyberpuppy.labeling.label_map import (BullyingLevel, EmotionType,
                                               RoleType, ToxicityLevel,
                                               UnifiedLabel)
from src.cyberpuppy.models.contextual import (ContextualInput, ContextualModel,
                                              ContextualOutput,
                                              ContrastiveLearningModule,
                                              EventFeatureExtractor,
                                              HierarchicalThreadEncoder)


class TestContextualInput:
    """測試上下文輸入資料結構"""

    def test_contextual_input_basic(self):
        """測試基本上下文輸入"""
        ctx_input = ContextualInput(text="測試文本")

        assert ctx_input.text == "測試文本"
        assert ctx_input.thread_context is None
        assert ctx_input.event_context is None
        assert ctx_input.role_info is None
        assert ctx_input.temporal_info is None

    def test_contextual_input_full(self):
        """測試完整上下文輸入"""
        ctx_input = ContextualInput(
            text="測試文本",
            thread_context=["消息1", "消息2"],
            event_context={"event_type": "cyberbullying"},
            role_info={"role": "perpetrator"},
            temporal_info={"duration": 300},
        )

        assert ctx_input.text == "測試文本"
        assert ctx_input.thread_context == ["消息1", "消息2"]
        assert ctx_input.event_context["event_type"] == "cyberbullying"
        assert ctx_input.role_info["role"] == "perpetrator"
        assert ctx_input.temporal_info["duration"] == 300


class TestContextualOutput:
    """測試上下文輸出資料結構"""

    def test_contextual_output_basic(self):
        """測試基本上下文輸出"""
        toxicity_logits = torch.randn(1, 3)
        bullying_logits = torch.randn(1, 3)
        role_logits = torch.randn(1, 4)
        emotion_logits = torch.randn(1, 3)
        text_embedding = torch.randn(1, 768)

        output = ContextualOutput(
            toxicity_logits=toxicity_logits,
            bullying_logits=bullying_logits,
            role_logits=role_logits,
            emotion_logits=emotion_logits,
            text_embedding=text_embedding,
        )

        assert output.toxicity_logits.shape == (1, 3)
        assert output.bullying_logits.shape == (1, 3)
        assert output.role_logits.shape == (1, 4)
        assert output.emotion_logits.shape == (1, 3)
        assert output.text_embedding.shape == (1, 768)


class MockTransformer:
    """模擬 Transformer 模型"""

    class MockOutput:
        def __init__(self):
            self.last_hidden_state = torch.randn(1, 128, 768)

    def from_pretrained(self, model_name):
        return self

    def __call__(self, **inputs):
        return self.MockOutput()

    def parameters(self):
        return [torch.randn(1, requires_grad=True)]


class MockTokenizer:
    """模擬分詞器"""

    def from_pretrained(self, model_name):
        return self

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.randint(1, 1000, (1, 128)),
            "attention_mask": torch.ones(1, 128),
            "token_type_ids": torch.zeros(1, 128),
        }


class TestHierarchicalThreadEncoder:
    """測試階層式會話編碼器"""

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_thread_encoder_init(self, mock_tokenizer, mock_model):
        """測試會話編碼器初始化"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        encoder = HierarchicalThreadEncoder()

        assert encoder.hidden_size == 768
        assert encoder.max_thread_length == 16
        assert encoder.max_message_length == 128

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_encode_message(self, mock_tokenizer, mock_model):
        """測試消息編碼"""
        mock_transformer = MockTransformer()
        mock_model.from_pretrained.return_value = mock_transformer
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        encoder = HierarchicalThreadEncoder()

        # 模擬編碼消息
        with patch.object(encoder, "encode_message") as mock_encode:
            mock_encode.return_value = torch.randn(1, 768)

            result = encoder.encode_message("測試消息")
            assert result.shape == (1, 768)

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_forward_empty_thread(self, mock_tokenizer, mock_model):
        """測試空會話處理"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        encoder = HierarchicalThreadEncoder()

        # 測試空會話
        context_emb, attention = encoder.forward([])

        assert context_emb.shape == (1, 768)
        assert torch.allclose(context_emb, torch.zeros(1, 768))
        assert attention is None

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_forward_with_messages(self, mock_tokenizer, mock_model):
        """測試帶消息的會話處理"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        encoder = HierarchicalThreadEncoder()

        # 模擬 encode_message 方法
        with patch.object(encoder, "encode_message") as mock_encode:
            mock_encode.return_value = torch.randn(1, 768)

            messages = ["消息1", "消息2", "消息3"]
            roles = ["none", "victim", "perpetrator"]

            context_emb, attention = encoder.forward(messages, roles)

            assert context_emb.shape == (1, 768)
            assert mock_encode.call_count == 3


class TestEventFeatureExtractor:
    """測試事件特徵抽取器"""

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_event_extractor_init(self, mock_tokenizer, mock_model):
        """測試事件抽取器初始化"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        extractor = EventFeatureExtractor()

        assert extractor.hidden_size == 768
        assert hasattr(extractor, "event_type_embedding")
        assert hasattr(extractor, "severity_embedding")

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_extract_temporal_features(self, mock_tokenizer, mock_model):
        """測試時序特徵抽取"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        extractor = EventFeatureExtractor()

        # 測試空時序資訊
        temporal_emb = extractor.extract_temporal_features({})
        assert temporal_emb.shape == (1, 768)

        # 測試完整時序資訊
        temporal_info = {
            "duration": 300,
            "frequency": 3,
            "time_intervals": [60, 120, 180],
            "periodicity": 0.5,
            "event_density": 0.8,
        }

        temporal_emb = extractor.extract_temporal_features(temporal_info)
        assert temporal_emb.shape == (1, 768)

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_forward_basic_event(self, mock_tokenizer, mock_model):
        """測試基本事件特徵抽取"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        extractor = EventFeatureExtractor()

        event_context = {
            "event_type": "cyberbullying",
            "severity": "high",
            "participants": ["perpetrator", "victim"],
        }

        temporal_info = {"duration": 300}

        event_emb, feature_weights = extractor.forward("測試"
            "文本", event_context, temporal_info)

        assert event_emb.shape == (1, 768)
        assert isinstance(feature_weights, dict)
        assert "text" in feature_weights


class TestContrastiveLearningModule:
    """測試對比學習模組"""

    def test_contrastive_module_init(self):
        """測試對比學習模組初始化"""
        module = ContrastiveLearningModule()

        assert module.temperature == 0.1
        assert module.projection_dim == 128
        assert hasattr(module, "projection_head")

    def test_forward(self):
        """測試前向傳播"""
        module = ContrastiveLearningModule()

        embeddings = torch.randn(4, 768)
        projected = module.forward(embeddings)

        assert projected.shape == (4, 128)
        # 檢查 L2 正規化
        norms = torch.norm(projected, dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-6)

    def test_compute_contrastive_loss(self):
        """測試對比損失計算"""
        module = ContrastiveLearningModule()

        embeddings = torch.randn(4, 128)
        embeddings = nn.functional.normalize(embeddings, dim=-1)  # L2 正規化

        labels = torch.tensor([0, 0, 1, 1])  # 兩個類別

        loss = module.compute_contrastive_loss(embeddings, labels)

        assert loss.requires_grad
        assert loss.item() >= 0  # 損失應該非負

    def test_compute_contrastive_loss_with_sessions(self):
        """測試帶會話ID的對比損失"""
        module = ContrastiveLearningModule()

        embeddings = torch.randn(4, 128)
        embeddings = nn.functional.normalize(embeddings, dim=-1)

        labels = torch.tensor([0, 1, 0, 1])
        session_ids = torch.tensor([1, 1, 2, 2])  # 兩個會話

        loss = module.compute_contrastive_loss(embeddings, labels, session_ids)

        assert loss.requires_grad
        assert loss.item() >= 0

    def test_no_positive_samples(self):
        """測試無正樣本情況"""
        module = ContrastiveLearningModule()

        embeddings = torch.randn(3, 128)
        embeddings = nn.functional.normalize(embeddings, dim=-1)

        labels = torch.tensor([0, 1, 2])  # 所有樣本標籤不同

        loss = module.compute_contrastive_loss(embeddings, labels)

        # 無正樣本時應該返回零損失
        assert loss.item() == 0.0


class TestContextualModel:
    """測試上下文感知模型"""

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_contextual_model_init(self, mock_tokenizer, mock_model):
        """測試上下文模型初始化"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        model = ContextualModel(use_contrastive=True)

        assert model.hidden_size == 768
        assert model.use_contrastive is True
        assert hasattr(model, "thread_encoder")
        assert hasattr(model, "event_extractor")
        assert hasattr(model, "contrastive_module")

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_encode_text(self, mock_tokenizer, mock_model):
        """測試文本編碼"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        model = ContextualModel()

        text_emb = model.encode_text("測試文本")
        assert text_emb.shape == (1, 768)

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_forward_text_only(self, mock_tokenizer, mock_model):
        """測試僅文本輸入的前向傳播"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        model = ContextualModel()

        ctx_input = ContextualInput(text="測試文本")
        output = model.forward([ctx_input])

        assert isinstance(output, ContextualOutput)
        assert output.toxicity_logits.shape == (1, 3)
        assert output.bullying_logits.shape == (1, 3)
        assert output.role_logits.shape == (1, 4)
        assert output.emotion_logits.shape == (1, 3)

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_forward_with_context(self, mock_tokenizer, mock_model):
        """測試包含上下文的前向傳播"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        model = ContextualModel()

        ctx_input = ContextualInput(
            text="測試文本",
            thread_context=["消息1", "消息2"],
            event_context={"event_type": "cyberbullying", "severity": "high"},
        )

        # 模擬編碼器輸出
        with patch.object(model.thread_encoder, "forward") as mock_thread:
            with patch.object(model.event_extractor, "forward") as mock_event:
                mock_thread.return_value = (torch.randn(1, 768), None)
                mock_event.return_value = (torch.randn(1, 768), {})

                output = model.forward([ctx_input])

                assert isinstance(output, ContextualOutput)
                assert output.context_embedding is not None
                assert output.event_embedding is not None

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_compute_loss(self, mock_tokenizer, mock_model):
        """測試損失計算"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        model = ContextualModel(use_contrastive=True)

        # 創建模擬輸出
        outputs = [
            ContextualOutput(
                toxicity_logits=torch.randn(1, 3),
                bullying_logits=torch.randn(1, 3),
                role_logits=torch.randn(1, 4),
                emotion_logits=torch.randn(1, 3),
                text_embedding=torch.randn(1, 768),
                contrastive_features=torch.randn(1, 128),
            ),
            ContextualOutput(
                toxicity_logits=torch.randn(1, 3),
                bullying_logits=torch.randn(1, 3),
                role_logits=torch.randn(1, 4),
                emotion_logits=torch.randn(1, 3),
                text_embedding=torch.randn(1, 768),
                contrastive_features=torch.randn(1, 128),
            ),
        ]

        labels = [
            UnifiedLabel(
                toxicity=ToxicityLevel.TOXIC,
                bullying=BullyingLevel.HARASSMENT,
                role=RoleType.PERPETRATOR,
                emotion=EmotionType.NEGATIVE,
            ),
            UnifiedLabel(
                toxicity=ToxicityLevel.NONE,
                bullying=BullyingLevel.NONE,
                role=RoleType.NONE,
                emotion=EmotionType.NEUTRAL,
            ),
        ]

        session_ids = torch.tensor([1, 2])

        losses = model.compute_loss(outputs, labels, session_ids)

        assert "toxicity" in losses
        assert "bullying" in losses
        assert "role" in losses
        assert "emotion" in losses
        assert "contrastive" in losses
        assert "total" in losses

        # 檢查損失值
        for key, loss in losses.items():
            assert loss.requires_grad
            assert loss.item() >= 0

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_predict(self, mock_tokenizer, mock_model):
        """測試預測功能"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        model = ContextualModel()

        # 模擬前向傳播輸出
        with patch.object(model, "forward") as mock_forward:
            mock_output = ContextualOutput(
                toxicity_logits=torch.tensor([[2.0, 0.1, 0.1]]),  # toxic 最高
                bullying_logits=torch.tensor(
                    [[0.1,
                    2.0,
                    0.1]]
                ),  # harassment 最高
                role_logits=torch.tensor(
                    [[0.1,
                    2.0,
                    0.1,
                    0.1]]
                ),  # perpetrator 最高
                emotion_logits=torch.tensor([[0.1, 0.1, 2.0]]),  # neg 最高
                text_embedding=torch.randn(1, 768),
            )
            mock_forward.return_value = mock_output

            ctx_input = ContextualInput(text="測試文本")
            prediction = model.predict(ctx_input)

            assert prediction["toxicity"] == "toxic"
            assert prediction["bullying"] == "harassment"
            assert prediction["role"] == "perpetrator"
            assert prediction["emotion"] == "neg"

            # 檢查置信度
            assert "toxicity_confidence" in prediction
            assert "bullying_confidence" in prediction
            assert "role_confidence" in prediction
            assert "emotion_confidence" in prediction

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_batch_processing(self, mock_tokenizer, mock_model):
        """測試批次處理"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        model = ContextualModel()

        ctx_inputs = [
            ContextualInput(text="文本1"),
            ContextualInput(text="文本2"),
            ContextualInput(text="文本3"),
        ]

        outputs = model.forward(ctx_inputs)

        assert isinstance(outputs, list)
        assert len(outputs) == 3

        for output in outputs:
            assert isinstance(output, ContextualOutput)
            assert output.toxicity_logits.shape == (1, 3)


class TestIntegration:
    """整合測試"""

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_end_to_end_sccd_processing(self, mock_tokenizer, mock_model):
        """端到端SCCD會話處理測試"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        model = ContextualModel(use_contrastive=True)

        # SCCD 會話輸入
        ctx_input = ContextualInput(
            text="你這個垃圾去死",
            thread_context=["大家好", "你好嗎", "你這個垃圾去死"],
            role_info={"thread_roles": ["none", "victim", "perpetrator"]},
        )

        # 模擬編碼器輸出
        with patch.object(model.thread_encoder, "forward") as mock_thread:
            mock_thread.return_value = (
                torch.randn(1,
                768),
                torch.randn(1,
                1,
                3)
            )

            output = model.forward([ctx_input])

            assert isinstance(output, ContextualOutput)
            assert output.context_embedding is not None
            assert output.attention_weights is not None

            # 預測
            prediction = model.predict(ctx_input)
            assert "toxicity" in prediction
            assert "bullying" in prediction

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_end_to_end_chnci_processing(self, mock_tokenizer, mock_model):
        """端到端CHNCI事件處理測試"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        model = ContextualModel(use_contrastive=True)

        # CHNCI 事件輸入
        ctx_input = ContextualInput(
            text="威脅性言論",
            event_context={
                "event_type": "cyberbullying",
                "severity": "high",
                "participants": ["perpetrator", "victim"],
            },
            temporal_info={"dura"
                "tion": 300, 
        )

        # 模擬事件抽取器輸出
        with patch.object(model.event_extractor, "forward") as mock_event:
            mock_event.return_value = (
                torch.randn(1, 768),
                {"te"
                    "xt": 0.4, 
            )

            output = model.forward([ctx_input])

            assert isinstance(output, ContextualOutput)
            assert output.event_embedding is not None
            assert output.attention_weights is not None

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_contrastive_learning_training(self, mock_tokenizer, mock_model):
        """對比學習訓練測試"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        model = ContextualModel(use_contrastive=True)

        # 創建同一會話的不同樣本
        ctx_inputs = [
            ContextualInput(text="你很煩人", thread_context=["對話1", "你很煩人"]),
            ContextualInput(text="滾開", thread_context=["對話1", "滾開"]),
            ContextualInput(text="今天天氣不錯", thread_context=["對話2", "今天天氣不錯"]),
        ]

        # 模擬前向傳播
        with patch.object(model.thread_encoder, "forward") as mock_thread:
            mock_thread.return_value = (torch.randn(1, 768), None)

            outputs = model.forward(ctx_inputs)

            # 創建標籤
            labels = [
                UnifiedLabel(toxicity=ToxicityLevel.TOXIC),
                UnifiedLabel(toxicity=ToxicityLevel.TOXIC),
                UnifiedLabel(toxicity=ToxicityLevel.NONE),
            ]

            session_ids = torch.tensor([1, 1, 2])  # 前兩個同會話

            # 計算損失
            losses = model.compute_loss(outputs, labels, session_ids)

            assert "contrastive" in losses
            assert losses["contrastive"].requires_grad
            assert losses["total"].item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
