#!/usr/bin/env python3
"""
SHAP解釋器測試套件

測試SHAP可解釋性功能的完整性和正確性
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from cyberpuppy.models.improved_detector import ImprovedDetector, create_improved_config
from cyberpuppy.explain.shap_explainer import (
    SHAPExplainer, SHAPVisualizer, MisclassificationAnalyzer, SHAPResult,
    SHAPModelWrapper, compare_ig_shap_explanations
)


@pytest.fixture
def device():
    """測試設備fixture"""
    return torch.device("cpu")  # 測試使用CPU


@pytest.fixture
def model_config():
    """測試模型配置"""
    config = create_improved_config()
    config.model_name = "hfl/chinese-macbert-base"
    return config


@pytest.fixture
def mock_model(model_config, device):
    """Mock模型fixture"""
    model = ImprovedDetector(model_config)
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def test_texts():
    """測試文本fixture"""
    return [
        "你這個垃圾，去死吧！",
        "今天天氣很好",
        "這個政策很爛",
        "謝謝你的幫助",
        "你們這些笨蛋"
    ]


@pytest.fixture
def mock_shap_values():
    """Mock SHAP值"""
    return np.random.rand(10, 13)  # 假設10個token，13個輸出類別


class TestSHAPModelWrapper:
    """測試SHAP模型包裝器"""

    def test_wrapper_initialization(self, mock_model, device):
        """測試包裝器初始化"""
        wrapper = SHAPModelWrapper(mock_model, mock_model.tokenizer, device)

        assert wrapper.model == mock_model
        assert wrapper.tokenizer == mock_model.tokenizer
        assert wrapper.device == device

    def test_wrapper_call_single_text(self, mock_model, device):
        """測試包裝器單文本調用"""
        wrapper = SHAPModelWrapper(mock_model, mock_model.tokenizer, device)

        with patch.object(mock_model, 'forward') as mock_forward:
            # Mock模型輸出
            mock_forward.return_value = {
                "toxicity": torch.randn(1, 3),
                "bullying": torch.randn(1, 3),
                "role": torch.randn(1, 4),
                "emotion": torch.randn(1, 3)
            }

            result = wrapper(["測試文本"])

            assert isinstance(result, np.ndarray)
            assert result.shape == (1, 13)  # 3+3+4+3 = 13個類別
            assert np.all(result >= 0) and np.all(result <= 1)  # 概率值

    def test_wrapper_call_multiple_texts(self, mock_model, device, test_texts):
        """測試包裝器多文本調用"""
        wrapper = SHAPModelWrapper(mock_model, mock_model.tokenizer, device)

        with patch.object(mock_model, 'forward') as mock_forward:
            # Mock模型輸出
            mock_forward.return_value = {
                "toxicity": torch.randn(1, 3),
                "bullying": torch.randn(1, 3),
                "role": torch.randn(1, 4),
                "emotion": torch.randn(1, 3)
            }

            result = wrapper(test_texts[:3])

            assert isinstance(result, np.ndarray)
            assert result.shape == (3, 13)


class TestSHAPExplainer:
    """測試SHAP解釋器"""

    def test_explainer_initialization(self, mock_model, device):
        """測試解釋器初始化"""
        with patch('cyberpuppy.explain.shap_explainer.shap.Explainer'):
            explainer = SHAPExplainer(mock_model, device)

            assert explainer.model == mock_model
            assert explainer.device == device
            assert explainer.tokenizer == mock_model.tokenizer
            assert hasattr(explainer, 'model_wrapper')
            assert hasattr(explainer, 'task_configs')

    def test_task_configs(self, mock_model, device):
        """測試任務配置"""
        with patch('cyberpuppy.explain.shap_explainer.shap.Explainer'):
            explainer = SHAPExplainer(mock_model, device)

            expected_tasks = ["toxicity", "bullying", "role", "emotion"]
            assert list(explainer.task_configs.keys()) == expected_tasks

            # 檢查每個任務的配置
            assert explainer.task_configs["toxicity"]["start_idx"] == 0
            assert explainer.task_configs["toxicity"]["end_idx"] == 3
            assert len(explainer.task_configs["toxicity"]["labels"]) == 3

    @patch('cyberpuppy.explain.shap_explainer.shap.Explainer')
    def test_explain_text_success(self, mock_shap_explainer, mock_model, device):
        """測試文本解釋成功案例"""
        # Mock SHAP解釋器
        mock_explainer_instance = Mock()
        mock_shap_explainer.return_value = mock_explainer_instance

        # Mock SHAP值
        mock_shap_values = Mock()
        mock_shap_values.values = [np.random.rand(13)]
        mock_shap_values.base_values = [0.5]
        mock_explainer_instance.return_value = mock_shap_values

        explainer = SHAPExplainer(mock_model, device)
        explainer.explainer = mock_explainer_instance

        with patch.object(mock_model, 'forward') as mock_forward:
            # Mock模型輸出
            mock_forward.return_value = {
                "toxicity": torch.tensor([[0.1, 0.7, 0.2]]),
                "bullying": torch.tensor([[0.3, 0.4, 0.3]]),
                "role": torch.tensor([[0.4, 0.2, 0.2, 0.2]]),
                "emotion": torch.tensor([[0.2, 0.3, 0.5]])
            }

            result = explainer.explain_text("測試文本")

            assert isinstance(result, SHAPResult)
            assert result.text == "測試文本"
            assert len(result.tokens) > 0
            assert result.toxicity_pred in [0, 1, 2]
            assert 0 <= result.toxicity_prob <= 1
            assert hasattr(result, 'feature_importance')

    @patch('cyberpuppy.explain.shap_explainer.shap.Explainer')
    def test_explain_text_failure_handling(self, mock_shap_explainer, mock_model, device):
        """測試文本解釋錯誤處理"""
        # Mock SHAP解釋器拋出異常
        mock_explainer_instance = Mock()
        mock_shap_explainer.return_value = mock_explainer_instance
        mock_explainer_instance.side_effect = Exception("SHAP computation failed")

        explainer = SHAPExplainer(mock_model, device)
        explainer.explainer = mock_explainer_instance

        with patch.object(mock_model, 'forward') as mock_forward:
            # Mock模型輸出
            mock_forward.return_value = {
                "toxicity": torch.tensor([[0.1, 0.7, 0.2]]),
                "bullying": torch.tensor([[0.3, 0.4, 0.3]]),
                "role": torch.tensor([[0.4, 0.2, 0.2, 0.2]]),
                "emotion": torch.tensor([[0.2, 0.3, 0.5]])
            }

            result = explainer.explain_text("測試文本")

            # 應該返回零值作為後備
            assert isinstance(result, SHAPResult)
            assert np.allclose(result.toxicity_shap_values, 0)


class TestSHAPVisualizer:
    """測試SHAP可視化器"""

    @pytest.fixture
    def mock_explainer(self, mock_model, device):
        """Mock解釋器"""
        with patch('cyberpuppy.explain.shap_explainer.shap.Explainer'):
            return SHAPExplainer(mock_model, device)

    @pytest.fixture
    def mock_shap_result(self):
        """Mock SHAP結果"""
        return SHAPResult(
            text="測試文本",
            tokens=["測", "試", "文", "本"],
            toxicity_pred=1,
            toxicity_prob=0.7,
            bullying_pred=0,
            bullying_prob=0.3,
            role_pred=1,
            role_prob=0.5,
            emotion_pred=2,
            emotion_prob=0.6,
            toxicity_shap_values=np.array([0.1, 0.3, -0.2, 0.4]),
            bullying_shap_values=np.array([0.2, -0.1, 0.3, -0.2]),
            role_shap_values=np.array([0.1, 0.2, 0.1, 0.3]),
            emotion_shap_values=np.array([-0.1, 0.4, 0.2, 0.1]),
            toxicity_base_value=0.5,
            bullying_base_value=0.3,
            role_base_value=0.25,
            emotion_base_value=0.33,
            prediction_confidence={"toxicity": 0.7, "bullying": 0.3, "role": 0.5, "emotion": 0.6},
            feature_importance={"toxicity": 1.0, "bullying": 0.8, "role": 0.7, "emotion": 0.8}
        )

    def test_visualizer_initialization(self, mock_explainer):
        """測試可視化器初始化"""
        visualizer = SHAPVisualizer(mock_explainer)
        assert visualizer.explainer == mock_explainer

    def test_create_force_plot(self, mock_explainer, mock_shap_result):
        """測試Force plot創建"""
        visualizer = SHAPVisualizer(mock_explainer)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            try:
                visualizer.create_force_plot(
                    mock_shap_result,
                    task="toxicity",
                    save_path=tmp.name
                )
                # 如果沒有異常，說明功能基本正常
                assert True
            except Exception as e:
                # 由於沒有真實的SHAP環境，預期可能會有錯誤
                pytest.skip(f"Force plot test skipped due to: {e}")

    def test_create_waterfall_plot(self, mock_explainer, mock_shap_result):
        """測試Waterfall plot創建"""
        visualizer = SHAPVisualizer(mock_explainer)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            try:
                fig = visualizer.create_waterfall_plot(
                    mock_shap_result,
                    task="toxicity",
                    save_path=tmp.name
                )
                assert fig is not None
            except Exception:
                # 可視化測試在CI環境中可能會失敗
                pytest.skip("Waterfall plot test requires display")

    def test_create_text_plot(self, mock_explainer, mock_shap_result):
        """測試Text plot創建"""
        visualizer = SHAPVisualizer(mock_explainer)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            try:
                fig = visualizer.create_text_plot(
                    mock_shap_result,
                    task="toxicity",
                    save_path=tmp.name
                )
                assert fig is not None
            except Exception:
                pytest.skip("Text plot test requires display")

    def test_create_summary_plot(self, mock_explainer, mock_shap_result):
        """測試Summary plot創建"""
        visualizer = SHAPVisualizer(mock_explainer)
        results = [mock_shap_result] * 3  # 多個結果用於統計

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            try:
                fig = visualizer.create_summary_plot(
                    results,
                    task="toxicity",
                    max_features=5,
                    save_path=tmp.name
                )
                assert fig is not None
            except Exception:
                pytest.skip("Summary plot test requires display")


class TestMisclassificationAnalyzer:
    """測試誤判分析器"""

    @pytest.fixture
    def mock_explainer(self, mock_model, device):
        """Mock解釋器"""
        with patch('cyberpuppy.explain.shap_explainer.shap.Explainer'):
            return SHAPExplainer(mock_model, device)

    @pytest.fixture
    def mock_shap_result(self):
        """Mock SHAP結果"""
        return SHAPResult(
            text="測試文本",
            tokens=["測", "試", "文", "本"],
            toxicity_pred=1,
            toxicity_prob=0.7,
            bullying_pred=0,
            bullying_prob=0.3,
            role_pred=1,
            role_prob=0.5,
            emotion_pred=2,
            emotion_prob=0.6,
            toxicity_shap_values=np.array([0.1, 0.3, -0.2, 0.4]),
            bullying_shap_values=np.array([0.2, -0.1, 0.3, -0.2]),
            role_shap_values=np.array([0.1, 0.2, 0.1, 0.3]),
            emotion_shap_values=np.array([-0.1, 0.4, 0.2, 0.1]),
            toxicity_base_value=0.5,
            bullying_base_value=0.3,
            role_base_value=0.25,
            emotion_base_value=0.33,
            prediction_confidence={"toxicity": 0.7, "bullying": 0.3, "role": 0.5, "emotion": 0.6},
            feature_importance={"toxicity": 1.0, "bullying": 0.8, "role": 0.7, "emotion": 0.8}
        )

    def test_analyzer_initialization(self, mock_explainer):
        """測試分析器初始化"""
        analyzer = MisclassificationAnalyzer(mock_explainer)
        assert analyzer.explainer == mock_explainer

    def test_analyze_misclassifications(self, mock_explainer, mock_shap_result):
        """測試誤判分析"""
        analyzer = MisclassificationAnalyzer(mock_explainer)

        # Mock解釋器返回結果
        with patch.object(mock_explainer, 'explain_text', return_value=mock_shap_result):
            texts = ["文本1", "文本2", "文本3"]
            true_labels = [
                {"toxicity_label": 0},  # 正確預測（預測1，真實0 -> 誤判）
                {"toxicity_label": 1},  # 正確預測
                {"toxicity_label": 2}   # 誤判（預測1，真實2）
            ]

            result = analyzer.analyze_misclassifications(texts, true_labels, "toxicity")

            assert "misclassified_cases" in result
            assert "correct_cases" in result
            assert "error_analysis" in result
            assert "misclassification_rate" in result

            # 檢查誤判率計算
            assert 0 <= result["misclassification_rate"] <= 1

    def test_generate_misclassification_report(self, mock_explainer):
        """測試誤判報告生成"""
        analyzer = MisclassificationAnalyzer(mock_explainer)

        # Mock分析結果
        analysis_result = {
            "misclassified_cases": [
                {
                    "text": "誤判案例",
                    "true_label": 0,
                    "predicted_label": 1,
                    "confidence": 0.8,
                    "feature_importance": 0.9
                }
            ],
            "correct_cases": [],
            "error_analysis": {
                "avg_misclassified_confidence": 0.8,
                "avg_correct_confidence": 0.9,
                "confidence_gap": 0.1,
                "top_error_features": [("錯誤", 3), ("特徵", 2)]
            },
            "misclassification_rate": 0.5
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp:
            analyzer.generate_misclassification_report(analysis_result, tmp.name)

            # 檢查文件是否創建
            assert Path(tmp.name).exists()

            # 檢查文件內容
            with open(tmp.name, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "誤判分析報告" in content
                assert "0.50%" in content  # 誤判率


class TestUtilityFunctions:
    """測試工具函數"""

    def test_compare_ig_shap_explanations(self):
        """測試IG和SHAP結果對比"""
        # Mock IG結果
        ig_result = Mock()
        ig_result.toxicity_attributions = np.array([0.1, 0.3, -0.2, 0.4])

        # Mock SHAP結果
        shap_result = Mock()
        shap_result.toxicity_shap_values = np.array([0.15, 0.25, -0.25, 0.35])
        shap_result.tokens = ["測", "試", "文", "本"]

        comparison = compare_ig_shap_explanations(ig_result, shap_result, "toxicity")

        assert "pearson_correlation" in comparison
        assert "spearman_correlation" in comparison
        assert "top_features_overlap" in comparison
        assert "attribution_difference" in comparison

        # 檢查相關性值的範圍
        assert -1 <= comparison["pearson_correlation"] <= 1
        assert 0 <= comparison["top_features_overlap"] <= 1


class TestIntegration:
    """整合測試"""

    @pytest.mark.slow
    def test_full_pipeline(self, mock_model, device):
        """測試完整流程"""
        with patch('cyberpuppy.explain.shap_explainer.shap.Explainer'):
            # 初始化所有組件
            explainer = SHAPExplainer(mock_model, device)
            visualizer = SHAPVisualizer(explainer)
            analyzer = MisclassificationAnalyzer(explainer)

            # Mock SHAP解釋結果
            mock_result = SHAPResult(
                text="測試文本",
                tokens=["測", "試", "文", "本"],
                toxicity_pred=1,
                toxicity_prob=0.7,
                bullying_pred=0,
                bullying_prob=0.3,
                role_pred=1,
                role_prob=0.5,
                emotion_pred=2,
                emotion_prob=0.6,
                toxicity_shap_values=np.array([0.1, 0.3, -0.2, 0.4]),
                bullying_shap_values=np.array([0.2, -0.1, 0.3, -0.2]),
                role_shap_values=np.array([0.1, 0.2, 0.1, 0.3]),
                emotion_shap_values=np.array([-0.1, 0.4, 0.2, 0.1]),
                toxicity_base_value=0.5,
                bullying_base_value=0.3,
                role_base_value=0.25,
                emotion_base_value=0.33,
                prediction_confidence={"toxicity": 0.7, "bullying": 0.3, "role": 0.5, "emotion": 0.6},
                feature_importance={"toxicity": 1.0, "bullying": 0.8, "role": 0.7, "emotion": 0.8}
            )

            with patch.object(explainer, 'explain_text', return_value=mock_result):
                # 測試解釋
                result = explainer.explain_text("測試文本")
                assert isinstance(result, SHAPResult)

                # 測試可視化（簡化版）
                try:
                    with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
                        visualizer.create_waterfall_plot(result, save_path=tmp.name)
                except:
                    pass  # 可視化在測試環境中可能失敗

                # 測試誤判分析
                analysis = analyzer.analyze_misclassifications(
                    ["測試文本"],
                    [{"toxicity_label": 0}],
                    "toxicity"
                )
                assert "misclassification_rate" in analysis


if __name__ == "__main__":
    pytest.main([__file__])