#!/usr/bin/env python3
"""
改進霸凌偵測模型的單元測試
Tests for improved bullying detection model
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import Dict, Any

# Test target modules
from cyberpuppy.models.improved_detector import (
    ImprovedModelConfig,
    ImprovedBullyingDetector,
    FocalLoss,
    DynamicTaskWeighting,
)


class TestImprovedModelConfig:
    """測試改進模型配置類"""

    def test_config_default_values(self):
        """測試配置預設值"""
        config = ImprovedModelConfig()

        assert config.model_name == "hfl/chinese-macbert-base"
        assert config.hidden_size == 768
        assert config.max_length == 512
        assert config.num_toxicity_classes == 3
        assert config.num_bullying_classes == 3
        assert config.num_role_classes == 4
        assert config.num_emotion_classes == 3

    def test_config_custom_values(self):
        """測試自定義配置值"""
        config = ImprovedModelConfig(
            model_name="custom-model",
            hidden_size=512,
            max_length=256,
            num_toxicity_classes=5
        )

        assert config.model_name == "custom-model"
        assert config.hidden_size == 512
        assert config.max_length == 256
        assert config.num_toxicity_classes == 5

    def test_config_validation(self):
        """測試配置參數驗證"""
        # Test valid parameters
        config = ImprovedModelConfig(attention_dropout=0.1)
        assert 0.0 <= config.attention_dropout <= 1.0

        # Test numerical parameters are reasonable
        assert config.num_attention_heads > 0
        assert config.hidden_size > 0


class TestFocalLoss:
    """測試 Focal Loss 實現"""

    def test_focal_loss_initialization(self):
        """測試 Focal Loss 初始化"""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        assert focal_loss.alpha == 0.25
        assert focal_loss.gamma == 2.0

    def test_focal_loss_forward(self):
        """測試 Focal Loss 前向傳播"""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

        # Create mock inputs
        batch_size, num_classes = 4, 3
        inputs = torch.randn(batch_size, num_classes, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size,))

        # Calculate loss
        loss = focal_loss(inputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative

    def test_focal_loss_gradient(self):
        """測試 Focal Loss 梯度計算"""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

        inputs = torch.randn(2, 3, requires_grad=True)
        targets = torch.tensor([0, 1])

        loss = focal_loss(inputs, targets)
        loss.backward()

        assert inputs.grad is not None
        assert inputs.grad.shape == inputs.shape


class TestDynamicTaskWeighting:
    """測試動態任務權重學習"""

    def test_dynamic_task_weighting_init(self):
        """測試動態任務權重初始化"""
        weighting = DynamicTaskWeighting(num_tasks=4)
        assert weighting.num_tasks == 4
        assert len(weighting.log_vars) == 4

    def test_dynamic_task_weighting_forward(self):
        """測試動態任務權重前向傳播"""
        weighting = DynamicTaskWeighting(num_tasks=3)

        losses = [
            torch.tensor(0.5),
            torch.tensor(0.3),
            torch.tensor(0.7)
        ]

        weighted_loss = weighting(losses)

        assert isinstance(weighted_loss, torch.Tensor)
        assert weighted_loss.item() >= 0


@pytest.fixture
def mock_tokenizer():
    """模擬分詞器"""
    tokenizer = Mock()
    tokenizer.encode_plus.return_value = {
        'input_ids': torch.tensor([[101, 1234, 5678, 102]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1]]),
        'token_type_ids': torch.tensor([[0, 0, 0, 0]])
    }
    tokenizer.vocab_size = 21128
    return tokenizer


@pytest.fixture
def mock_transformer():
    """模擬 Transformer 模型"""
    model = Mock()
    model.config.hidden_size = 768

    # Mock last_hidden_state
    last_hidden_state = torch.randn(1, 4, 768)  # batch, seq_len, hidden

    model.return_value = Mock()
    model.return_value.last_hidden_state = last_hidden_state
    model.return_value.pooler_output = torch.randn(1, 768)

    return model


class TestImprovedBullyingDetector:
    """測試改進霸凌偵測模型"""

    def test_model_initialization(self):
        """測試模型初始化"""
        config = ImprovedModelConfig()

        with patch('cyberpuppy.models.improved_detector.AutoModel.from_pretrained') as mock_model, \
             patch('cyberpuppy.models.improved_detector.AutoTokenizer.from_pretrained') as mock_tokenizer:

            # Mock the transformer model
            mock_model.return_value.config.hidden_size = 768
            mock_tokenizer.return_value.vocab_size = 21128

            model = ImprovedBullyingDetector(config)

            assert model.config == config
            assert hasattr(model, 'transformer')
            assert hasattr(model, 'tokenizer')
            assert hasattr(model, 'toxicity_classifier')
            assert hasattr(model, 'bullying_classifier')

    def test_model_forward_pass(self, mock_tokenizer, mock_transformer):
        """測試模型前向傳播"""
        config = ImprovedModelConfig()

        with patch('cyberpuppy.models.improved_detector.AutoModel.from_pretrained', return_value=mock_transformer), \
             patch('cyberpuppy.models.improved_detector.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            model = ImprovedBullyingDetector(config)

            # Test input
            input_ids = torch.tensor([[101, 1234, 5678, 102]])
            attention_mask = torch.tensor([[1, 1, 1, 1]])

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            assert 'toxicity_logits' in outputs
            assert 'bullying_logits' in outputs
            assert 'emotion_logits' in outputs
            assert 'role_logits' in outputs

    def test_model_predict_method(self, mock_tokenizer, mock_transformer):
        """測試模型預測方法"""
        config = ImprovedModelConfig()

        with patch('cyberpuppy.models.improved_detector.AutoModel.from_pretrained', return_value=mock_transformer), \
             patch('cyberpuppy.models.improved_detector.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            model = ImprovedBullyingDetector(config)
            model.eval()

            text = "這是一個測試句子"

            with torch.no_grad():
                result = model.predict(text)

            assert isinstance(result, dict)
            assert 'toxicity' in result
            assert 'bullying' in result
            assert 'emotion' in result
            assert 'role' in result
            assert 'confidence' in result

    def test_model_training_step(self, mock_tokenizer, mock_transformer):
        """測試模型訓練步驟"""
        config = ImprovedModelConfig()

        with patch('cyberpuppy.models.improved_detector.AutoModel.from_pretrained', return_value=mock_transformer), \
             patch('cyberpuppy.models.improved_detector.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            model = ImprovedBullyingDetector(config)
            model.train()

            # Mock training data
            batch = {
                'input_ids': torch.tensor([[101, 1234, 5678, 102]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1]]),
                'toxicity_labels': torch.tensor([1]),
                'bullying_labels': torch.tensor([0]),
                'emotion_labels': torch.tensor([2]),
                'role_labels': torch.tensor([1]),
            }

            outputs = model(**batch)

            assert 'loss' in outputs
            assert 'toxicity_loss' in outputs
            assert 'bullying_loss' in outputs
            assert isinstance(outputs['loss'], torch.Tensor)

    @pytest.mark.parametrize("text,expected_type", [
        ("你好，今天天氣很好", str),
        ("這個笨蛋真的很煩", str),
        ("", str),
        ("    ", str),
    ])
    def test_model_text_preprocessing(self, text, expected_type, mock_tokenizer, mock_transformer):
        """測試文本前處理"""
        config = ImprovedModelConfig()

        with patch('cyberpuppy.models.improved_detector.AutoModel.from_pretrained', return_value=mock_transformer), \
             patch('cyberpuppy.models.improved_detector.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            model = ImprovedBullyingDetector(config)

            # Test text preprocessing
            processed = model._preprocess_text(text)
            assert isinstance(processed, expected_type)

    def test_model_attention_mechanism(self, mock_tokenizer, mock_transformer):
        """測試注意力機制"""
        config = ImprovedModelConfig(use_cross_attention=True, use_self_attention=True)

        with patch('cyberpuppy.models.improved_detector.AutoModel.from_pretrained', return_value=mock_transformer), \
             patch('cyberpuppy.models.improved_detector.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            model = ImprovedBullyingDetector(config)

            # Check if attention layers are properly initialized
            assert hasattr(model, 'cross_attention')
            assert hasattr(model, 'self_attention')

    def test_model_regularization(self, mock_tokenizer, mock_transformer):
        """測試正規化技術"""
        config = ImprovedModelConfig()

        with patch('cyberpuppy.models.improved_detector.AutoModel.from_pretrained', return_value=mock_transformer), \
             patch('cyberpuppy.models.improved_detector.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            model = ImprovedBullyingDetector(config)

            # Check dropout layers
            assert any(isinstance(module, nn.Dropout) for module in model.modules())

            # Check layer normalization
            assert any(isinstance(module, nn.LayerNorm) for module in model.modules())


class TestModelPerformance:
    """測試模型效能相關功能"""

    def test_model_inference_speed(self, mock_tokenizer, mock_transformer):
        """測試模型推理速度"""
        config = ImprovedModelConfig()

        with patch('cyberpuppy.models.improved_detector.AutoModel.from_pretrained', return_value=mock_transformer), \
             patch('cyberpuppy.models.improved_detector.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            model = ImprovedBullyingDetector(config)
            model.eval()

            import time

            text = "這是一個測試句子用來檢測推理速度"

            start_time = time.time()
            with torch.no_grad():
                result = model.predict(text)
            inference_time = time.time() - start_time

            # Should complete within reasonable time (< 1 second on CPU)
            assert inference_time < 1.0
            assert isinstance(result, dict)

    def test_model_memory_usage(self, mock_tokenizer, mock_transformer):
        """測試模型記憶體使用"""
        config = ImprovedModelConfig()

        with patch('cyberpuppy.models.improved_detector.AutoModel.from_pretrained', return_value=mock_transformer), \
             patch('cyberpuppy.models.improved_detector.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            model = ImprovedBullyingDetector(config)

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            assert total_params > 0
            assert trainable_params > 0
            assert trainable_params <= total_params

    def test_model_batch_processing(self, mock_tokenizer, mock_transformer):
        """測試批次處理"""
        config = ImprovedModelConfig()

        with patch('cyberpuppy.models.improved_detector.AutoModel.from_pretrained', return_value=mock_transformer), \
             patch('cyberpuppy.models.improved_detector.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):

            model = ImprovedBullyingDetector(config)
            model.eval()

            # Mock batch tokenizer output
            mock_tokenizer.encode_plus.return_value = {
                'input_ids': torch.tensor([[101, 1234, 5678, 102], [101, 2345, 6789, 102]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]),
                'token_type_ids': torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
            }

            # Mock transformer output for batch
            mock_transformer.return_value.last_hidden_state = torch.randn(2, 4, 768)
            mock_transformer.return_value.pooler_output = torch.randn(2, 768)

            texts = ["第一個測試句子", "第二個測試句子"]

            with torch.no_grad():
                results = model.predict_batch(texts)

            assert isinstance(results, list)
            assert len(results) == len(texts)
            for result in results:
                assert isinstance(result, dict)
                assert 'toxicity' in result


if __name__ == "__main__":
    # Run basic smoke test
    config = ImprovedModelConfig()
    print(f"測試配置: {config.model_name}")
    print("✅ 改進霸凌偵測模型測試準備完成")