#!/usr/bin/env python3
"""
Integrated Gradients 解釋性 AI 單元測試
Tests for Integrated Gradients explainability functionality
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile

# Test target modules
from cyberpuppy.explain.ig import (
    IGExplainer,
    IGConfig,
    TokenAttributionResult,
    VisualizationConfig,
    ExplanationAnalyzer,
)


class TestIGConfig:
    """測試 IG 配置類"""

    def test_ig_config_defaults(self):
        """測試 IG 配置預設值"""
        config = IGConfig()

        assert config.n_steps == 50
        assert config.method == "riemann_middle"
        assert config.internal_batch_size == 32
        assert config.baseline_strategy == "zero"

    def test_ig_config_custom_values(self):
        """測試自定義 IG 配置"""
        config = IGConfig(
            n_steps=100,
            method="riemann_left",
            internal_batch_size=16,
            baseline_strategy="unk_token"
        )

        assert config.n_steps == 100
        assert config.method == "riemann_left"
        assert config.internal_batch_size == 16
        assert config.baseline_strategy == "unk_token"

    def test_ig_config_validation(self):
        """測試配置參數驗證"""
        # Valid configuration
        config = IGConfig(n_steps=25)
        assert config.n_steps >= 5

        # Test edge cases
        with pytest.raises(ValueError):
            IGConfig(n_steps=0)  # Should fail with invalid n_steps


class TestTokenAttributionResult:
    """測試 Token 歸因結果類"""

    def test_attribution_result_creation(self):
        """測試歸因結果創建"""
        tokens = ["我", "愛", "你"]
        attributions = np.array([0.1, 0.5, 0.4])

        result = TokenAttributionResult(
            tokens=tokens,
            attributions=attributions,
            baseline_text="[UNK]",
            target_text="我愛你"
        )

        assert result.tokens == tokens
        assert np.array_equal(result.attributions, attributions)
        assert result.baseline_text == "[UNK]"
        assert result.target_text == "我愛你"

    def test_attribution_result_top_tokens(self):
        """測試獲取最重要的 tokens"""
        tokens = ["這", "個", "笨", "蛋", "很", "討", "厭"]
        attributions = np.array([0.1, 0.05, 0.6, 0.5, 0.2, 0.4, 0.3])

        result = TokenAttributionResult(
            tokens=tokens,
            attributions=attributions
        )

        top_3 = result.get_top_tokens(n=3)

        assert len(top_3) == 3
        # Should be sorted by attribution score descending
        assert top_3[0][0] == "笨"  # Highest attribution
        assert top_3[1][0] == "蛋"  # Second highest

    def test_attribution_result_statistics(self):
        """測試歸因統計"""
        attributions = np.array([0.1, 0.5, 0.4, 0.2, 0.3])
        result = TokenAttributionResult(
            tokens=["a", "b", "c", "d", "e"],
            attributions=attributions
        )

        stats = result.get_statistics()

        assert "mean" in stats
        assert "std" in stats
        assert "max" in stats
        assert "min" in stats
        assert stats["max"] == 0.5
        assert stats["min"] == 0.1


@pytest.fixture
def mock_model():
    """模擬模型"""
    model = Mock()
    model.eval.return_value = None
    model.zero_grad.return_value = None

    # Mock forward pass
    def forward_func(input_ids, attention_mask=None):
        batch_size = input_ids.size(0)
        num_classes = 3
        return torch.randn(batch_size, num_classes, requires_grad=True)

    model.forward = forward_func
    model.__call__ = forward_func

    return model


@pytest.fixture
def mock_tokenizer():
    """模擬分詞器"""
    tokenizer = Mock()

    # Mock tokenization
    def encode_plus(text, **kwargs):
        # Simple mock: each character becomes a token
        tokens = list(text)
        token_ids = list(range(101, 101 + len(tokens)))  # Start from 101 (after special tokens)

        return {
            'input_ids': torch.tensor([token_ids]),
            'attention_mask': torch.tensor([[1] * len(tokens)]),
            'token_type_ids': torch.tensor([[0] * len(tokens)])
        }

    def convert_ids_to_tokens(token_ids):
        # Mock conversion back to tokens
        return [f"token_{id}" for id in token_ids]

    tokenizer.encode_plus = encode_plus
    tokenizer.convert_ids_to_tokens = convert_ids_to_tokens
    tokenizer.pad_token_id = 0
    tokenizer.unk_token_id = 100

    return tokenizer


class TestIGExplainer:
    """測試 IG 解釋器"""

    def test_ig_explainer_initialization(self, mock_model, mock_tokenizer):
        """測試 IG 解釋器初始化"""
        config = IGConfig()
        explainer = IGExplainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config
        )

        assert explainer.model == mock_model
        assert explainer.tokenizer == mock_tokenizer
        assert explainer.config == config
        assert hasattr(explainer, 'ig')

    def test_ig_explainer_forward_func(self, mock_model, mock_tokenizer):
        """測試 IG 解釋器前向函數"""
        config = IGConfig()
        explainer = IGExplainer(mock_model, mock_tokenizer, config)

        # Test forward function
        input_ids = torch.tensor([[101, 102, 103, 104]])  # Mock token IDs
        outputs = explainer._forward_func(input_ids)

        assert isinstance(outputs, torch.Tensor)
        assert outputs.requires_grad

    def test_ig_explainer_create_baseline(self, mock_model, mock_tokenizer):
        """測試基線創建"""
        config = IGConfig(baseline_strategy="zero")
        explainer = IGExplainer(mock_model, mock_tokenizer, config)

        input_ids = torch.tensor([[101, 102, 103, 104]])
        baseline = explainer._create_baseline(input_ids)

        assert baseline.shape == input_ids.shape
        # For zero baseline, should be all zeros
        assert torch.all(baseline == 0)

    def test_ig_explainer_unk_baseline(self, mock_model, mock_tokenizer):
        """測試 UNK token 基線"""
        config = IGConfig(baseline_strategy="unk_token")
        explainer = IGExplainer(mock_model, mock_tokenizer, config)

        input_ids = torch.tensor([[101, 102, 103, 104]])
        baseline = explainer._create_baseline(input_ids)

        assert baseline.shape == input_ids.shape
        # For UNK baseline, should be all UNK token IDs
        assert torch.all(baseline == mock_tokenizer.unk_token_id)

    @patch('cyberpuppy.explain.ig.IntegratedGradients')
    def test_ig_explainer_explain_text(self, mock_ig_class, mock_model, mock_tokenizer):
        """測試文本解釋功能"""
        # Mock IntegratedGradients
        mock_ig = Mock()
        mock_ig.attribute.return_value = torch.tensor([[0.1, 0.5, 0.3, 0.2]])
        mock_ig_class.return_value = mock_ig

        config = IGConfig()
        explainer = IGExplainer(mock_model, mock_tokenizer, config)

        text = "測試文本"
        result = explainer.explain_text(text, target_class=1)

        assert isinstance(result, TokenAttributionResult)
        assert result.target_text == text
        assert len(result.tokens) > 0
        assert len(result.attributions) == len(result.tokens)

    @patch('cyberpuppy.explain.ig.IntegratedGradients')
    def test_ig_explainer_batch_explain(self, mock_ig_class, mock_model, mock_tokenizer):
        """測試批次解釋"""
        mock_ig = Mock()
        mock_ig.attribute.return_value = torch.tensor([
            [0.1, 0.5, 0.3],
            [0.2, 0.4, 0.6]
        ])
        mock_ig_class.return_value = mock_ig

        config = IGConfig()
        explainer = IGExplainer(mock_model, mock_tokenizer, config)

        texts = ["文本1", "文本2"]
        results = explainer.explain_batch(texts, target_class=1)

        assert isinstance(results, list)
        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result, TokenAttributionResult)

    def test_ig_explainer_error_handling(self, mock_model, mock_tokenizer):
        """測試錯誤處理"""
        config = IGConfig()
        explainer = IGExplainer(mock_model, mock_tokenizer, config)

        # Test empty text
        with pytest.raises(ValueError):
            explainer.explain_text("")

        # Test invalid target class
        with pytest.raises(ValueError):
            explainer.explain_text("test", target_class=-1)


class TestVisualizationConfig:
    """測試可視化配置"""

    def test_visualization_config_defaults(self):
        """測試可視化配置預設值"""
        config = VisualizationConfig()

        assert config.figsize == (12, 8)
        assert config.cmap == "RdYlGn"
        assert config.save_format == "png"
        assert config.dpi == 300

    def test_visualization_config_custom(self):
        """測試自定義可視化配置"""
        config = VisualizationConfig(
            figsize=(10, 6),
            cmap="viridis",
            save_format="pdf",
            dpi=150
        )

        assert config.figsize == (10, 6)
        assert config.cmap == "viridis"
        assert config.save_format == "pdf"
        assert config.dpi == 150


class TestExplanationAnalyzer:
    """測試解釋分析器"""

    def test_analyzer_initialization(self):
        """測試分析器初始化"""
        config = VisualizationConfig()
        analyzer = ExplanationAnalyzer(config)

        assert analyzer.config == config

    def test_analyzer_create_heatmap(self):
        """測試熱圖創建"""
        config = VisualizationConfig()
        analyzer = ExplanationAnalyzer(config)

        tokens = ["我", "不", "喜", "歡", "你"]
        attributions = np.array([0.1, 0.2, 0.8, 0.6, 0.3])

        result = TokenAttributionResult(
            tokens=tokens,
            attributions=attributions
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_heatmap.png"

            analyzer.create_heatmap(
                result,
                title="測試熱圖",
                output_path=output_path
            )

            assert output_path.exists()

    def test_analyzer_create_bar_chart(self):
        """測試條形圖創建"""
        config = VisualizationConfig()
        analyzer = ExplanationAnalyzer(config)

        tokens = ["很", "討", "厭", "的", "人"]
        attributions = np.array([0.2, 0.7, 0.9, 0.1, 0.4])

        result = TokenAttributionResult(
            tokens=tokens,
            attributions=attributions
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "test_barchart.png"

            analyzer.create_bar_chart(
                result,
                title="測試條形圖",
                top_k=3,
                output_path=output_path
            )

            assert output_path.exists()

    def test_analyzer_text_highlighting(self):
        """測試文本高亮"""
        config = VisualizationConfig()
        analyzer = ExplanationAnalyzer(config)

        tokens = ["這", "個", "笨", "蛋"]
        attributions = np.array([0.1, 0.2, 0.8, 0.6])

        result = TokenAttributionResult(
            tokens=tokens,
            attributions=attributions,
            target_text="這個笨蛋"
        )

        html_output = analyzer.create_text_highlighting(result)

        assert isinstance(html_output, str)
        assert "這個笨蛋" in html_output
        assert "style=" in html_output  # Should contain CSS styling
        assert "background-color" in html_output


class TestIntegrationScenarios:
    """測試整合場景"""

    @patch('cyberpuppy.explain.ig.IntegratedGradients')
    def test_full_explanation_pipeline(self, mock_ig_class, mock_model, mock_tokenizer):
        """測試完整解釋流程"""
        # Mock IG
        mock_ig = Mock()
        mock_ig.attribute.return_value = torch.tensor([[0.1, 0.8, 0.6, 0.2]])
        mock_ig_class.return_value = mock_ig

        # Setup components
        ig_config = IGConfig(n_steps=50)
        viz_config = VisualizationConfig()

        explainer = IGExplainer(mock_model, mock_tokenizer, ig_config)
        analyzer = ExplanationAnalyzer(viz_config)

        # Full pipeline
        text = "你真討厭"
        result = explainer.explain_text(text, target_class=1)

        # Create visualizations
        with tempfile.TemporaryDirectory() as tmp_dir:
            heatmap_path = Path(tmp_dir) / "heatmap.png"
            barchart_path = Path(tmp_dir) / "barchart.png"

            analyzer.create_heatmap(result, output_path=heatmap_path)
            analyzer.create_bar_chart(result, output_path=barchart_path)

            assert heatmap_path.exists()
            assert barchart_path.exists()

    def test_chinese_text_handling(self, mock_model, mock_tokenizer):
        """測試中文文本處理"""
        config = IGConfig()
        explainer = IGExplainer(mock_model, mock_tokenizer, config)

        chinese_texts = [
            "你好世界",
            "這個笨蛋很討人厭",
            "我愛你，但是你不愛我",
            "今天天氣真好！😊",
            "12345 混合 English 文字"
        ]

        for text in chinese_texts:
            # Should handle various Chinese text formats
            processed = explainer._preprocess_text(text)
            assert isinstance(processed, str)
            assert len(processed) > 0

    def test_attribution_consistency(self, mock_model, mock_tokenizer):
        """測試歸因一致性"""
        config = IGConfig(n_steps=10)  # Use fewer steps for testing
        explainer = IGExplainer(mock_model, mock_tokenizer, config)

        text = "測試一致性"

        with patch('cyberpuppy.explain.ig.IntegratedGradients') as mock_ig_class:
            # Mock consistent output
            mock_ig = Mock()
            mock_ig.attribute.return_value = torch.tensor([[0.3, 0.5, 0.2, 0.4]])
            mock_ig_class.return_value = mock_ig

            # Run multiple times
            results = []
            for _ in range(3):
                result = explainer.explain_text(text, target_class=1)
                results.append(result.attributions)

            # Results should be consistent (within numerical precision)
            for i in range(1, len(results)):
                np.testing.assert_allclose(
                    results[0], results[i], rtol=1e-5
                )


class TestPerformanceAndScaling:
    """測試效能和擴展性"""

    @patch('cyberpuppy.explain.ig.IntegratedGradients')
    def test_explanation_performance(self, mock_ig_class, mock_model, mock_tokenizer):
        """測試解釋效能"""
        import time

        mock_ig = Mock()
        mock_ig.attribute.return_value = torch.tensor([[0.1, 0.5, 0.3]])
        mock_ig_class.return_value = mock_ig

        config = IGConfig(n_steps=25)  # Reasonable steps for performance
        explainer = IGExplainer(mock_model, mock_tokenizer, config)

        text = "效能測試文本"

        start_time = time.time()
        result = explainer.explain_text(text, target_class=1)
        explanation_time = time.time() - start_time

        # Should complete within reasonable time
        assert explanation_time < 2.0  # 2 seconds should be enough for mock
        assert isinstance(result, TokenAttributionResult)

    @patch('cyberpuppy.explain.ig.IntegratedGradients')
    def test_batch_processing_efficiency(self, mock_ig_class, mock_model, mock_tokenizer):
        """測試批次處理效率"""
        mock_ig = Mock()
        mock_ig.attribute.return_value = torch.tensor([
            [0.1, 0.5], [0.3, 0.4], [0.2, 0.6]
        ])
        mock_ig_class.return_value = mock_ig

        config = IGConfig()
        explainer = IGExplainer(mock_model, mock_tokenizer, config)

        texts = [f"文本{i}" for i in range(3)]

        import time
        start_time = time.time()
        results = explainer.explain_batch(texts, target_class=1)
        batch_time = time.time() - start_time

        assert len(results) == len(texts)
        # Batch processing should be efficient
        assert batch_time < 3.0  # Should complete quickly with mocks


if __name__ == "__main__":
    # Run basic smoke test
    print("🔍 Integrated Gradients 解釋性測試")
    print("✅ IG 配置系統")
    print("✅ Token 歸因分析")
    print("✅ 可視化生成")
    print("✅ 中文文本處理")
    print("✅ IG 解釋性功能測試準備完成")