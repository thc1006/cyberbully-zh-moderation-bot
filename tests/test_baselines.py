#!/usr/bin/env python3
"""
測試基線模型
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.cyberpuppy.labeling.label_map import (
    BullyingLevel,
    EmotionType,
    RoleType,
    ToxicityLevel,
    UnifiedLabel,
)
from src.cyberpuppy.models.baselines import (
    BaselineModel,
    FocalLoss,
    ModelConfig,
    ModelEvaluator,
    MultiTaskDataset,
    MultiTaskHead,
    create_model_variants,
)


class TestModelConfig:
    """測試模型配置"""

    def test_model_config_defaults(self):
        """測試預設配置"""
        config = ModelConfig()

        assert config.model_name == "hfl/chinese-macbert-base"
        assert config.max_length == 256
        assert config.num_toxicity_classes == 3
        assert config.num_bullying_classes == 3
        assert config.num_role_classes == 4
        assert config.num_emotion_classes == 3
        assert config.use_emotion_regression is False

    def test_model_config_custom(self):
        """測試自訂配置"""
        config = ModelConfig(
            model_name="hfl/chinese-roberta-wwm-ext",
            max_length=512,
            use_emotion_regression=True,
            use_focal_loss=True,
        )

        assert config.model_name == "hfl/chinese-roberta-wwm-ext"
        assert config.max_length == 512
        assert config.use_emotion_regression is True
        assert config.use_focal_loss is True

    def test_task_weights_post_init(self):
        """測試任務權重自動初始化"""
        config = ModelConfig()

        assert "toxicity" in config.task_weights
        assert "bullying" in config.task_weights
        assert "role" in config.task_weights
        assert "emotion" in config.task_weights
        assert config.task_weights["toxicity"] == 1.0

    def test_custom_task_weights(self):
        """測試自訂任務權重"""
        custom_weights = {"toxicity": 2.0, "emotion": 0.5}
        config = ModelConfig(task_weights=custom_weights)

        assert config.task_weights == custom_weights


class TestFocalLoss:
    """測試Focal Loss"""

    def test_focal_loss_init(self):
        """測試Focal Loss初始化"""
        loss_fn = FocalLoss(alpha=1.0, gamma=2.0)

        assert loss_fn.alpha == 1.0
        assert loss_fn.gamma == 2.0
        assert loss_fn.reduction == "mean"

    def test_focal_loss_forward(self):
        """測試Focal Loss前向傳播"""
        loss_fn = FocalLoss()

        # 模擬邏輯回歸輸出和標籤
        inputs = torch.randn(4, 3, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 1])

        loss = loss_fn(inputs, targets)

        assert loss.requires_grad
        assert loss.item() >= 0
        assert loss.shape == torch.Size([])

    def test_focal_loss_different_reductions(self):
        """測試不同的reduction模式"""
        inputs = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])

        # Mean reduction
        loss_mean = FocalLoss(reduction="mean")(inputs, targets)
        assert loss_mean.shape == torch.Size([])

        # Sum reduction
        loss_sum = FocalLoss(reduction="sum")(inputs, targets)
        assert loss_sum.shape == torch.Size([])

        # None reduction
        loss_none = FocalLoss(reduction="none")(inputs, targets)
        assert loss_none.shape == torch.Size([4])


class TestMultiTaskDataset:
    """測試多任務資料集"""

    def setup_method(self):
        """設置測試環境"""
        # 模擬tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.return_value = {
            "input_ids": torch.randint(1, 1000, (1, 128), dtype=torch.long),
            "attention_mask": torch.ones(1, 128, dtype=torch.long),
            "token_type_ids": torch.zeros(1, 128, dtype=torch.long),
        }

        # 測試資料
        self.texts = ["測試文本1", "測試文本2", "測試文本3"]
        self.labels = [
            UnifiedLabel(
                toxicity=ToxicityLevel.TOXIC,
                bullying=BullyingLevel.HARASSMENT,
                role=RoleType.PERPETRATOR,
                emotion=EmotionType.NEGATIVE,
                emotion_intensity=3,
            ),
            UnifiedLabel(
                toxicity=ToxicityLevel.NONE,
                emotion=EmotionType.POSITIVE,
                emotion_intensity=2,
            ),
            UnifiedLabel(
                toxicity=ToxicityLevel.SEVERE,
                bullying=BullyingLevel.THREAT,
                emotion=EmotionType.NEGATIVE,
                emotion_intensity=4,
            ),
        ]

    def test_dataset_init(self):
        """測試資料集初始化"""
        dataset = MultiTaskDataset(self.texts, self.labels, self.mock_tokenizer)

        assert len(dataset) == 3
        assert dataset.max_length == 256

    def test_dataset_getitem(self):
        """測試資料獲取"""
        dataset = MultiTaskDataset(self.texts, self.labels, self.mock_tokenizer)

        item = dataset[0]

        # 檢查返回的鍵
        expected_keys = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "toxicity_label",
            "bullying_label",
            "role_label",
            "emotion_label",
            "emotion_intensity",
        }
        assert set(item.keys()) == expected_keys

        # 檢查標籤值
        assert item["toxicity_label"].item() == 1  # TOXIC
        assert item["bullying_label"].item() == 1  # HARASSMENT
        assert item["role_label"].item() == 1  # PERPETRATOR
        assert item["emotion_label"].item() == 2  # NEGATIVE
        assert item["emotion_intensity"].item() == 3

    def test_dataset_label_mapping(self):
        """測試標籤映射"""
        dataset = MultiTaskDataset(self.texts, self.labels, self.mock_tokenizer)

        # 測試第二個樣本（預設值）
        item = dataset[1]
        assert item["toxicity_label"].item() == 0  # NONE
        assert item["bullying_label"].item() == 0  # NONE
        assert item["role_label"].item() == 0  # NONE
        assert item["emotion_label"].item() == 0  # POSITIVE


class TestMultiTaskHead:
    """測試多任務頭"""

    def test_multi_task_head_init(self):
        """測試多任務頭初始化"""
        config = ModelConfig()
        head = MultiTaskHead(config)

        assert hasattr(head, "shared_layer")
        assert hasattr(head, "toxicity_head")
        assert hasattr(head, "bullying_head")
        assert hasattr(head, "role_head")
        assert hasattr(head, "emotion_head")

    def test_multi_task_head_init_with_regression(self):
        """測試包含回歸的多任務頭"""
        config = ModelConfig(use_emotion_regression=True)
        head = MultiTaskHead(config)

        assert hasattr(head, "emotion_intensity_head")

    def test_multi_task_head_forward(self):
        """測試多任務頭前向傳播"""
        config = ModelConfig()
        head = MultiTaskHead(config)

        hidden_states = torch.randn(2, 768)
        outputs = head(hidden_states)

        # 檢查輸出
        assert "toxicity" in outputs
        assert "bullying" in outputs
        assert "role" in outputs
        assert "emotion" in outputs

        # 檢查輸出形狀
        assert outputs["toxicity"].shape == (2, 3)
        assert outputs["bullying"].shape == (2, 3)
        assert outputs["role"].shape == (2, 4)
        assert outputs["emotion"].shape == (2, 3)

    def test_multi_task_head_forward_with_regression(self):
        """測試包含回歸的前向傳播"""
        config = ModelConfig(use_emotion_regression=True)
        head = MultiTaskHead(config)

        hidden_states = torch.randn(2, 768)
        outputs = head(hidden_states)

        assert "emotion_intensity" in outputs
        assert outputs["emotion_intensity"].shape == (2, 1)

        # 檢查情緒強度輸出範圍（應該在0-4之間）
        intensity_values = outputs["emotion_intensity"].detach()
        assert torch.all(intensity_values >= 0)
        assert torch.all(intensity_values <= 4)


class MockTransformer:
    """模擬Transformer模型"""

    class MockOutput:
        def __init__(self):
            self.last_hidden_state = torch.randn(2, 128, 768)

    def from_pretrained(self, model_name):
        return self

    def __call__(self, **inputs):
        return self.MockOutput()

    def parameters(self):
        return [torch.randn(1, requires_grad=True)]

    def state_dict(self):
        return {"mock_param": torch.randn(10)}

    def load_state_dict(self, state_dict):
        pass


class MockTokenizer:
    """模擬tokenizer"""

    def from_pretrained(self, model_name):
        return self

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.randint(1, 1000, (1, kwargs.get("max_length", 128))),
            "attention_mask": torch.ones(1, kwargs.get("max_length", 128)),
            "token_type_ids": torch.zeros(1, kwargs.get("max_length", 128)),
        }

    def save_pretrained(self, path):
        pass


class TestBaselineModel:
    """測試基線模型"""

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_baseline_model_init(self, mock_tokenizer, mock_model):
        """測試基線模型初始化"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        config = ModelConfig()
        model = BaselineModel(config)

        assert model.config == config
        assert hasattr(model, "backbone")
        assert hasattr(model, "multi_task_head")

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_baseline_model_forward(self, mock_tokenizer, mock_model):
        """測試基線模型前向傳播"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        config = ModelConfig()
        model = BaselineModel(config)

        batch_size = 2
        seq_len = 128

        input_ids = torch.randint(1, 1000, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

        outputs = model(input_ids, attention_mask, token_type_ids)

        # 檢查輸出
        assert "toxicity" in outputs
        assert "bullying" in outputs
        assert "role" in outputs
        assert "emotion" in outputs

        # 檢查輸出形狀
        assert outputs["toxicity"].shape == (batch_size, 3)
        assert outputs["bullying"].shape == (batch_size, 3)

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_baseline_model_compute_loss(self, mock_tokenizer, mock_model):
        """測試損失計算"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        config = ModelConfig()
        model = BaselineModel(config)

        batch_size = 2

        # 模擬模型輸出
        outputs = {
            "toxicity": torch.randn(batch_size, 3, requires_grad=True),
            "bullying": torch.randn(batch_size, 3, requires_grad=True),
            "role": torch.randn(batch_size, 4, requires_grad=True),
            "emotion": torch.randn(batch_size, 3, requires_grad=True),
        }

        # 模擬標籤
        labels = {
            "toxicity_label": torch.tensor([0, 1]),
            "bullying_label": torch.tensor([0, 1]),
            "role_label": torch.tensor([0, 1]),
            "emotion_label": torch.tensor([1, 2]),
        }

        losses = model.compute_loss(outputs, labels)

        # 檢查損失
        assert "toxicity" in losses
        assert "bullying" in losses
        assert "role" in losses
        assert "emotion" in losses
        assert "total" in losses

        # 檢查損失值
        for loss_name, loss_value in losses.items():
            assert loss_value.requires_grad
            assert loss_value.item() >= 0

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_baseline_model_predict(self, mock_tokenizer, mock_model):
        """測試預測功能"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer.from_pretrained.return_value = MockTokenizer()

        config = ModelConfig()
        model = BaselineModel(config)

        batch_size = 2
        seq_len = 128

        input_ids = torch.randint(1, 1000, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        predictions = model.predict(input_ids, attention_mask)

        # 檢查預測結果
        assert "toxicity_pred" in predictions
        assert "toxicity_probs" in predictions
        assert "toxicity_confidence" in predictions

        # 檢查預測形狀
        assert predictions["toxicity_pred"].shape == (batch_size,)
        assert predictions["toxicity_probs"].shape == (batch_size, 3)
        assert predictions["toxicity_confidence"].shape == (batch_size,)

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_baseline_model_save_load(self, mock_tokenizer, mock_model):
        """測試模型儲存和載入"""
        mock_transformer = MockTransformer()
        mock_model.from_pretrained.return_value = mock_transformer
        mock_tokenizer_obj = MockTokenizer()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_obj

        config = ModelConfig()
        model = BaselineModel(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            # 儲存模型
            model.save_model(temp_dir)

            # 檢查檔案存在
            save_path = Path(temp_dir)
            assert (save_path / "best.ckpt").exists()
            assert (save_path / "model_config.json").exists()

            # 載入模型
            loaded_model = BaselineModel.load_model(temp_dir)
            assert loaded_model.config.model_name == config.model_name


class TestModelEvaluator:
    """測試模型評估器"""

    def setup_method(self):
        """設置測試環境"""
        with (
            patch("transformers.AutoModel") as mock_model,
            patch("transformers.AutoTokenizer") as mock_tokenizer,
        ):
            mock_model.from_pretrained.return_value = MockTransformer()
            mock_tokenizer.from_pretrained.return_value = MockTokenizer()

            config = ModelConfig()
            self.model = BaselineModel(config)
            self.evaluator = ModelEvaluator(self.model)

    def test_evaluator_init(self):
        """測試評估器初始化"""
        assert self.evaluator.model == self.model
        assert isinstance(self.evaluator.device, torch.device)

    def test_session_level_f1(self):
        """測試會話級F1計算"""
        predictions = [0, 1, 0, 1, 0]  # 預測結果
        labels = [0, 1, 1, 1, 0]  # 真實標籤
        session_ids = ["s1", "s1", "s2", "s2", "s3"]  # 會話ID

        f1 = self.evaluator._compute_session_level_f1(predictions, labels, session_ids)

        assert isinstance(f1, float)
        assert 0 <= f1 <= 1


class TestCreateModelVariants:
    """測試模型變體建立"""

    def test_create_model_variants(self):
        """測試建立模型變體"""
        variants = create_model_variants()

        assert isinstance(variants, dict)
        assert len(variants) > 0

        # 檢查特定變體
        assert "macbert_base" in variants
        assert "roberta_base" in variants
        assert "toxicity_only" in variants

        # 檢查變體內容
        macbert_config = variants["macbert_base"]
        assert isinstance(macbert_config, ModelConfig)
        assert macbert_config.model_name == "hfl/chinese-macbert-base"

        roberta_config = variants["roberta_base"]
        assert roberta_config.model_name == "hfl/chinese-roberta-wwm-ext"

        # 檢查單任務配置
        toxicity_only_config = variants["toxicity_only"]
        assert toxicity_only_config.task_weights["toxicity"] == 1.0
        assert toxicity_only_config.task_weights["bullying"] == 0.0


class TestIntegration:
    """整合測試"""

    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_end_to_end_training_setup(self, mock_tokenizer, mock_model):
        """端到端訓練設置測試"""
        mock_model.from_pretrained.return_value = MockTransformer()
        mock_tokenizer_obj = MockTokenizer()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_obj

        # 建立模型
        config = ModelConfig(use_focal_loss=True, use_emotion_regression=True)
        model = BaselineModel(config)

        # 建立測試資料
        texts = ["測試文本1", "測試文本2", "測試文本3", "測試文本4"]
        labels = [
            UnifiedLabel(toxicity=ToxicityLevel.TOXIC, emotion=EmotionType.NEGATIVE),
            UnifiedLabel(toxicity=ToxicityLevel.NONE, emotion=EmotionType.POSITIVE),
            UnifiedLabel(toxicity=ToxicityLevel.SEVERE, emotion=EmotionType.NEGATIVE),
            UnifiedLabel(toxicity=ToxicityLevel.NONE, emotion=EmotionType.NEUTRAL),
        ]

        # 建立資料集
        dataset = MultiTaskDataset(texts, labels, mock_tokenizer_obj, max_length=128)

        # 建立資料載入器
        from torch.utils.data import DataLoader

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # 測試一個批次
        batch = next(iter(dataloader))

        # 前向傳播
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        outputs = model(input_ids, attention_mask, token_type_ids)

        # 計算損失
        labels_dict = {
            "toxicity_label": batch["toxicity_label"],
            "bullying_label": batch["bullying_label"],
            "role_label": batch["role_label"],
            "emotion_label": batch["emotion_label"],
            "emotion_intensity": batch["emotion_intensity"],
        }

        losses = model.compute_loss(outputs, labels_dict)

        # 驗證
        assert "total" in losses
        assert losses["total"].requires_grad
        assert losses["total"].item() >= 0

        # 預測
        predictions = model.predict(input_ids, attention_mask, token_type_ids)
        assert "toxicity_pred" in predictions
        assert predictions["toxicity_pred"].shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
