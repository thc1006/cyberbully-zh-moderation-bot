"""
評估模組測試套件
測試霸凌偵測系統的各種評估功能
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
import torch

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cyberpuppy.eval import (
    MetricsCalculator,
    ErrorAnalyzer,
    ResultVisualizer,
    ReportGenerator
)
from src.cyberpuppy.eval.error_analysis import ErrorCase, ErrorPattern
from src.cyberpuppy.eval.robustness import RobustnessTestResult

class TestMetricsCalculator(unittest.TestCase):
    """測試指標計算器"""

    def setUp(self):
        self.calculator = MetricsCalculator()

        # 測試數據
        self.y_true = ['none', 'toxic', 'severe', 'none', 'toxic', 'severe', 'none', 'toxic']
        self.y_pred = ['none', 'toxic', 'none', 'none', 'severe', 'severe', 'toxic', 'toxic']

    def test_calculate_basic_metrics(self):
        """測試基礎指標計算"""

        metrics = self.calculator.calculate_basic_metrics(self.y_true, self.y_pred)

        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)

        # 檢查值的範圍
        for metric_name, value in metrics.items():
            self.assertTrue(0 <= value <= 1, f"{metric_name} 應該在 0-1 之間，實際值: {value}")

    def test_calculate_per_class_metrics(self):
        """測試每類別指標計算"""

        per_class_metrics = self.calculator.calculate_per_class_metrics(self.y_true, self.y_pred)

        # 檢查所有類別都有指標
        expected_classes = ['none', 'toxic', 'severe']
        for class_name in expected_classes:
            self.assertIn(class_name, per_class_metrics)

            class_metrics = per_class_metrics[class_name]
            self.assertIn('precision', class_metrics)
            self.assertIn('recall', class_metrics)
            self.assertIn('f1-score', class_metrics)
            self.assertIn('support', class_metrics)

    def test_confusion_matrix(self):
        """測試混淆矩陣計算"""

        cm = self.calculator.calculate_confusion_matrix(self.y_true, self.y_pred)

        # 檢查矩陣形狀
        self.assertEqual(cm.shape, (3, 3))  # 3個類別

        # 檢查矩陣元素都是非負整數
        self.assertTrue(np.all(cm >= 0))
        self.assertTrue(cm.dtype == int or cm.dtype == np.int64)

    def test_macro_metrics(self):
        """測試宏平均指標"""

        macro_metrics = self.calculator.calculate_macro_metrics(self.y_true, self.y_pred)

        self.assertIn('macro_precision', macro_metrics)
        self.assertIn('macro_recall', macro_metrics)
        self.assertIn('macro_f1', macro_metrics)

    def test_weighted_metrics(self):
        """測試加權平均指標"""

        weighted_metrics = self.calculator.calculate_weighted_metrics(self.y_true, self.y_pred)

        self.assertIn('weighted_precision', weighted_metrics)
        self.assertIn('weighted_recall', weighted_metrics)
        self.assertIn('weighted_f1', weighted_metrics)

    def test_empty_input(self):
        """測試空輸入處理"""

        with self.assertRaises(ValueError):
            self.calculator.calculate_basic_metrics([], [])

    def test_mismatched_length(self):
        """測試長度不匹配的輸入"""

        with self.assertRaises(ValueError):
            self.calculator.calculate_basic_metrics(['none', 'toxic'], ['none'])

class TestErrorAnalyzer(unittest.TestCase):
    """測試錯誤分析器"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = ErrorAnalyzer(output_dir=self.temp_dir)

        # 測試數據
        self.true_labels = ['none', 'toxic', 'severe', 'none', 'toxic']
        self.predicted_labels = ['toxic', 'toxic', 'none', 'none', 'severe']
        self.texts = [
            "這是正常的對話",
            "你真的很笨",
            "我要殺了你",
            "今天天氣很好",
            "滾開，垃圾"
        ]
        self.confidences = [0.8, 0.9, 0.6, 0.7, 0.85]

    def tearDown(self):
        # 清理臨時目錄
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_analyze_errors(self):
        """測試錯誤分析功能"""

        results = self.analyzer.analyze_errors(
            true_labels=self.true_labels,
            predicted_labels=self.predicted_labels,
            texts=self.texts,
            confidences=self.confidences
        )

        # 檢查基本結構
        self.assertIn('timestamp', results)
        self.assertIn('total_errors', results)
        self.assertIn('error_cases', results)
        self.assertIn('error_patterns', results)
        self.assertIn('improvement_suggestions', results)

    def test_create_error_cases(self):
        """測試錯誤案例創建"""

        error_cases = self.analyzer._create_error_cases(
            self.true_labels, self.predicted_labels, self.texts, self.confidences
        )

        # 檢查錯誤案例數量（應該有3個錯誤）
        expected_errors = sum(1 for true, pred in zip(self.true_labels, self.predicted_labels) if true != pred)
        self.assertEqual(len(error_cases), expected_errors)

        # 檢查錯誤案例結構
        for case in error_cases:
            self.assertIsInstance(case, ErrorCase)
            self.assertTrue(hasattr(case, 'text'))
            self.assertTrue(hasattr(case, 'true_label'))
            self.assertTrue(hasattr(case, 'predicted_label'))
            self.assertTrue(hasattr(case, 'error_type'))

    def test_keyword_analysis(self):
        """測試關鍵詞分析"""

        text = "你這個笨蛋，真的很蠢"
        keywords = self.analyzer._find_keywords_in_text(text)

        self.assertIn('笨蛋', keywords)
        self.assertIn('蠢', keywords)

    def test_difficulty_assessment(self):
        """測試困難度評估"""

        # 高信心錯誤應該是困難的
        difficulty = self.analyzer._assess_difficulty_level(
            "正常文本", "none", "toxic", 0.9
        )
        self.assertEqual(difficulty, "hard")

        # 低信心錯誤應該是容易的
        difficulty = self.analyzer._assess_difficulty_level(
            "正常文本", "none", "toxic", 0.3
        )
        self.assertEqual(difficulty, "easy")

class TestResultVisualizer(unittest.TestCase):
    """測試結果視覺化器"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = ResultVisualizer(output_dir=self.temp_dir)

        # 測試數據
        self.y_true = ['none', 'toxic', 'severe', 'none', 'toxic', 'severe']
        self.y_pred = ['none', 'toxic', 'none', 'none', 'severe', 'severe']
        self.confidences = [0.8, 0.9, 0.6, 0.7, 0.85, 0.95]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_confusion_matrix(self, mock_close, mock_savefig):
        """測試混淆矩陣繪製"""

        result_path = self.visualizer.plot_confusion_matrix(
            y_true=self.y_true,
            y_pred=self.y_pred,
            class_names=['none', 'toxic', 'severe']
        )

        # 檢查是否調用了保存函數
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

        # 檢查返回路徑
        self.assertTrue(result_path.endswith('.png'))

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_confidence_distribution(self, mock_close, mock_savefig):
        """測試信心分數分布繪製"""

        result_path = self.visualizer.plot_confidence_distribution(
            confidence_scores=self.confidences,
            predictions=self.y_pred,
            true_labels=self.y_true
        )

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        self.assertTrue(result_path.endswith('.png'))

    def test_metrics_comparison_data_preparation(self):
        """測試指標對比數據準備"""

        metrics_dict = {
            'none': {'precision': 0.8, 'recall': 0.9, 'f1-score': 0.85, 'support': 100},
            'toxic': {'precision': 0.7, 'recall': 0.8, 'f1-score': 0.75, 'support': 80},
            'severe': {'precision': 0.6, 'recall': 0.7, 'f1-score': 0.65, 'support': 60}
        }

        # 這裡我們主要測試數據結構的正確性
        for class_name, metrics in metrics_dict.items():
            self.assertIn('precision', metrics)
            self.assertIn('recall', metrics)
            self.assertIn('f1-score', metrics)
            self.assertIn('support', metrics)

class TestReportGenerator(unittest.TestCase):
    """測試報告生成器"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ReportGenerator(output_dir=self.temp_dir)

        # 模擬評估結果
        self.evaluation_results = {
            'metrics': {
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.82,
                'f1': 0.81
            },
            'per_class_metrics': {
                'none': {'precision': 0.9, 'recall': 0.85, 'f1-score': 0.87, 'support': 100},
                'toxic': {'precision': 0.8, 'recall': 0.75, 'f1-score': 0.77, 'support': 80},
                'severe': {'precision': 0.7, 'recall': 0.8, 'f1-score': 0.75, 'support': 60}
            },
            'confusion_matrix': [[80, 15, 5], [10, 60, 10], [5, 7, 48]],
            'class_names': ['none', 'toxic', 'severe']
        }

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_json_report(self):
        """測試 JSON 報告生成"""

        report_data = {
            'evaluation_results': self.evaluation_results,
            'metadata': {
                'title': 'Test Report',
                'author': 'Test Author',
                'generation_time': '2024-01-01T00:00:00'
            }
        }

        json_path = self.generator._generate_json_report(report_data, None)

        # 檢查文件是否創建
        self.assertTrue(os.path.exists(json_path))

        # 檢查文件內容
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        self.assertIn('evaluation_results', loaded_data)
        self.assertIn('metadata', loaded_data)

    def test_html_report_generation(self):
        """測試 HTML 報告生成"""

        report_data = {
            'evaluation_results': self.evaluation_results,
            'metadata': {
                'title': 'Test Report',
                'author': 'Test Author',
                'company': 'Test Company',
                'generation_time': '2024-01-01T00:00:00',
                'version': '1.0.0'
            }
        }

        from src.cyberpuppy.eval.reports import ReportConfig
        config = ReportConfig()

        html_path = self.generator._generate_html_report(report_data, config)

        # 檢查文件是否創建
        self.assertTrue(os.path.exists(html_path))

        # 檢查 HTML 內容
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        self.assertIn('<!DOCTYPE html>', html_content)
        self.assertIn(config.title, html_content)

    def test_serialization(self):
        """測試數據序列化"""

        # 測試包含 numpy 數組的數據
        data_with_numpy = {
            'array': np.array([1, 2, 3]),
            'float': np.float64(3.14),
            'int': np.int32(42)
        }

        serialized = self.generator._make_serializable(data_with_numpy)

        # 檢查 numpy 類型是否正確轉換
        self.assertIsInstance(serialized['array'], list)
        self.assertIsInstance(serialized['float'], (int, float))
        self.assertIsInstance(serialized['int'], (int, float))

class TestRobustnessEvaluation(unittest.TestCase):
    """測試穩健性評估功能"""

    def setUp(self):
        # 創建模擬的模型和標記器
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()

        # 模擬預測輸出
        self.mock_model.return_value.logits = torch.tensor([[0.1, 0.8, 0.1]])

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.cyberpuppy.eval.robustness.torch')
    def test_robustness_test_result_creation(self, mock_torch):
        """測試穩健性測試結果創建"""

        result = RobustnessTestResult(
            test_name='character_substitution',
            original_text='測試文本',
            modified_text='測試文夲',  # 替換了一個字符
            original_prediction='none',
            modified_prediction='toxic',
            original_confidence=0.8,
            modified_confidence=0.6,
            confidence_drop=0.2,
            prediction_changed=True,
            attack_success=True,
            modification_type='字符替換'
        )

        self.assertEqual(result.test_name, 'character_substitution')
        self.assertTrue(result.attack_success)
        self.assertEqual(result.confidence_drop, 0.2)

class TestIntegration(unittest.TestCase):
    """整合測試"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_evaluation_workflow(self):
        """測試端到端評估流程"""

        # 準備測試數據
        true_labels = ['none', 'toxic', 'severe', 'none', 'toxic']
        predicted_labels = ['none', 'toxic', 'none', 'toxic', 'severe']
        texts = [
            "正常對話",
            "你很笨",
            "我要殺了你",
            "天氣不錯",
            "滾開"
        ]
        confidences = [0.9, 0.85, 0.7, 0.6, 0.8]

        # 1. 指標計算
        calculator = MetricsCalculator()
        metrics = calculator.calculate_all_metrics(true_labels, predicted_labels)

        self.assertIn('accuracy', metrics)
        self.assertIn('f1', metrics)

        # 2. 錯誤分析
        error_analyzer = ErrorAnalyzer(self.temp_dir)
        error_results = error_analyzer.analyze_errors(
            true_labels, predicted_labels, texts, confidences
        )

        self.assertIn('total_errors', error_results)
        self.assertIn('error_cases', error_results)

        # 3. 報告生成
        report_generator = ReportGenerator(self.temp_dir)

        evaluation_results = {
            'metrics': metrics,
            'confusion_matrix': calculator.calculate_confusion_matrix(true_labels, predicted_labels).tolist()
        }

        report_files = report_generator.generate_comprehensive_report(
            evaluation_results=evaluation_results,
            error_analysis=error_results,
            formats=['json']
        )

        self.assertIn('json', report_files)
        self.assertTrue(os.path.exists(report_files['json']))

    def test_data_consistency(self):
        """測試數據一致性"""

        # 測試標籤一致性
        labels = ['none', 'toxic', 'severe']

        # 所有組件都應該支持相同的標籤集
        calculator = MetricsCalculator()

        # 創建一致的測試數據
        y_true = ['none'] * 10 + ['toxic'] * 10 + ['severe'] * 10
        y_pred = ['none'] * 8 + ['toxic'] * 2 + ['toxic'] * 8 + ['severe'] * 2 + ['severe'] * 8 + ['none'] * 2

        # 計算指標
        metrics = calculator.calculate_per_class_metrics(y_true, y_pred)

        # 檢查所有標籤都有對應的指標
        for label in labels:
            self.assertIn(label, metrics)

class TestPerformance(unittest.TestCase):
    """性能測試"""

    def test_large_dataset_handling(self):
        """測試大數據集處理"""

        # 創建大數據集
        n_samples = 1000
        y_true = np.random.choice(['none', 'toxic', 'severe'], n_samples)
        y_pred = np.random.choice(['none', 'toxic', 'severe'], n_samples)

        calculator = MetricsCalculator()

        import time
        start_time = time.time()

        metrics = calculator.calculate_all_metrics(y_true.tolist(), y_pred.tolist())

        end_time = time.time()
        processing_time = end_time - start_time

        # 檢查處理時間合理（應該在幾秒內）
        self.assertLess(processing_time, 5.0, f"處理 {n_samples} 樣本用時過長: {processing_time:.2f}s")

        # 檢查結果正確性
        self.assertIn('accuracy', metrics)
        self.assertTrue(0 <= metrics['accuracy'] <= 1)

    def test_memory_usage(self):
        """測試內存使用"""

        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 創建並處理多個數據集
        calculator = MetricsCalculator()

        for _ in range(10):
            n_samples = 1000
            y_true = np.random.choice(['none', 'toxic', 'severe'], n_samples)
            y_pred = np.random.choice(['none', 'toxic', 'severe'], n_samples)

            calculator.calculate_all_metrics(y_true.tolist(), y_pred.tolist())

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # 內存增長應該合理（小於100MB）
        self.assertLess(memory_increase, 100,
                       f"內存使用增長過多: {memory_increase:.2f}MB")

if __name__ == '__main__':
    # 設定測試運行器
    unittest.main(verbosity=2, buffer=True)