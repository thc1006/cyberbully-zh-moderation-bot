#!/usr/bin/env python3
"""
CyberPuppy 綜合評估腳本
提供霸凌偵測系統的全面評估功能
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cyberpuppy.eval import (
    MetricsCalculator,
    ModelEvaluator,
    ErrorAnalyzer,
    ExplainabilityAnalyzer,
    RobustnessTestSuite,
    ResultVisualizer,
    ReportGenerator
)

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """綜合評估器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config.get('output_dir', 'evaluation_results')
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化各種評估器
        self.metrics_calculator = MetricsCalculator()
        self.error_analyzer = ErrorAnalyzer(self.output_dir)
        self.visualizer = ResultVisualizer(self.output_dir)
        self.report_generator = ReportGenerator(self.output_dir)

        # 模型和標記器將在運行時載入
        self.model = None
        self.tokenizer = None
        self.explainability_analyzer = None
        self.robustness_tester = None

    def load_model(self, model_path: str, tokenizer_path: str = None):
        """載入模型和標記器"""

        logger.info(f"載入模型: {model_path}")

        try:
            # 這裡需要根據實際的模型載入邏輯進行調整
            if tokenizer_path is None:
                tokenizer_path = model_path

            # 示例載入邏輯（需要根據實際情況調整）
            from transformers import AutoModel, AutoTokenizer

            self.model = AutoModel.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            # 初始化需要模型的評估器
            self.explainability_analyzer = ExplainabilityAnalyzer(
                self.model, self.tokenizer, self.output_dir
            )
            self.robustness_tester = RobustnessTestSuite(
                self.model, self.tokenizer, self.output_dir
            )

            logger.info("模型載入成功")

        except Exception as e:
            logger.error(f"模型載入失敗: {str(e)}")
            raise

    def load_test_data(self, data_path: str) -> Dict[str, List]:
        """載入測試數據"""

        logger.info(f"載入測試數據: {data_path}")

        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                texts = df['text'].tolist()
                labels = df['label'].tolist()
            elif data_path.endswith('.json'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                texts = data['texts']
                labels = data['labels']
            else:
                raise ValueError(f"不支援的文件格式: {data_path}")

            logger.info(f"載入 {len(texts)} 個測試樣本")

            return {
                'texts': texts,
                'labels': labels
            }

        except Exception as e:
            logger.error(f"數據載入失敗: {str(e)}")
            raise

    def get_model_predictions(self, texts: List[str]) -> Dict[str, List]:
        """獲取模型預測結果"""

        logger.info("獲取模型預測結果...")

        predictions = []
        confidences = []

        # 這裡需要根據實際的模型推理邏輯進行調整
        for text in texts:
            try:
                # 示例預測邏輯（需要根據實際情況調整）
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=512
                )

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # 假設 outputs.logits 是 (batch_size, num_classes)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    predicted_class = torch.argmax(probs, dim=-1).item()
                    confidence = probs.max().item()

                # 轉換為標籤
                label_mapping = {0: 'none', 1: 'toxic', 2: 'severe'}
                prediction = label_mapping.get(predicted_class, 'none')

                predictions.append(prediction)
                confidences.append(confidence)

            except Exception as e:
                logger.error(f"預測文本時發生錯誤: {str(e)}")
                predictions.append('none')  # 默認預測
                confidences.append(0.0)

        logger.info(f"完成 {len(predictions)} 個樣本的預測")

        return {
            'predictions': predictions,
            'confidences': confidences
        }

    def run_basic_evaluation(self,
                           test_data: Dict[str, List],
                           predictions: Dict[str, List]) -> Dict[str, Any]:
        """運行基礎評估"""

        logger.info("開始基礎評估...")

        true_labels = test_data['labels']
        predicted_labels = predictions['predictions']
        confidence_scores = predictions['confidences']

        # 計算基礎指標
        basic_metrics = self.metrics_calculator.calculate_all_metrics(
            true_labels, predicted_labels
        )

        # 計算每類別指標
        per_class_metrics = self.metrics_calculator.calculate_per_class_metrics(
            true_labels, predicted_labels
        )

        # 計算混淆矩陣
        confusion_matrix = self.metrics_calculator.calculate_confusion_matrix(
            true_labels, predicted_labels
        )

        evaluation_results = {
            'metrics': basic_metrics,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': confusion_matrix.tolist(),
            'class_names': ['none', 'toxic', 'severe'],
            'confidence_scores': confidence_scores,
            'total_samples': len(true_labels),
            'evaluation_time': datetime.now().isoformat()
        }

        logger.info("基礎評估完成")

        return evaluation_results

    def run_error_analysis(self,
                          test_data: Dict[str, List],
                          predictions: Dict[str, List]) -> Dict[str, Any]:
        """運行錯誤分析"""

        logger.info("開始錯誤分析...")

        true_labels = test_data['labels']
        predicted_labels = predictions['predictions']
        texts = test_data['texts']
        confidences = predictions['confidences']

        # 執行錯誤分析
        error_analysis_results = self.error_analyzer.analyze_errors(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
            texts=texts,
            confidences=confidences
        )

        logger.info("錯誤分析完成")

        return error_analysis_results

    def run_explainability_analysis(self,
                                   test_data: Dict[str, List],
                                   max_samples: int = 50) -> List[Dict[str, Any]]:
        """運行可解釋性分析"""

        if self.explainability_analyzer is None:
            logger.warning("可解釋性分析器未初始化，跳過該步驟")
            return []

        logger.info("開始可解釋性分析...")

        texts = test_data['texts'][:max_samples]  # 限制分析數量

        # 運行批量解釋
        explanations_by_method = self.explainability_analyzer.explain_batch(
            texts=texts,
            methods=['integrated_gradients', 'attention'],
            max_examples=max_samples
        )

        # 轉換為可序列化格式
        all_explanations = []
        for method, explanations in explanations_by_method.items():
            for explanation in explanations:
                explanation_dict = {
                    'text': explanation.text,
                    'prediction': explanation.prediction,
                    'confidence': explanation.confidence,
                    'method': explanation.method,
                    'token_attributions': explanation.token_attributions
                }
                all_explanations.append(explanation_dict)

        logger.info(f"可解釋性分析完成，共分析 {len(all_explanations)} 個案例")

        return all_explanations

    def run_robustness_tests(self,
                           test_data: Dict[str, List],
                           max_samples: int = 100) -> Dict[str, Any]:
        """運行穩健性測試"""

        if self.robustness_tester is None:
            logger.warning("穩健性測試器未初始化，跳過該步驟")
            return {}

        logger.info("開始穩健性測試...")

        texts = test_data['texts'][:max_samples]  # 限制測試數量
        labels = test_data['labels'][:max_samples]

        # 運行綜合穩健性測試
        robustness_results = self.robustness_tester.run_comprehensive_test(
            texts=texts,
            labels=labels,
            test_types=[
                'character_substitution',
                'typo_injection',
                'synonym_replacement',
                'punctuation_variation',
                'space_insertion'
            ]
        )

        logger.info("穩健性測試完成")

        return robustness_results

    def generate_visualizations(self,
                              evaluation_results: Dict[str, Any],
                              error_analysis: Dict[str, Any],
                              robustness_results: Dict[str, Any],
                              test_data: Dict[str, List],
                              predictions: Dict[str, List]) -> Dict[str, str]:
        """生成視覺化圖表"""

        logger.info("生成視覺化圖表...")

        visualization_files = {}

        try:
            # 1. 混淆矩陣
            cm_path = self.visualizer.plot_confusion_matrix(
                y_true=test_data['labels'],
                y_pred=predictions['predictions'],
                class_names=evaluation_results['class_names']
            )
            visualization_files['confusion_matrix'] = cm_path

            # 2. 指標對比
            if 'per_class_metrics' in evaluation_results:
                metrics_path = self.visualizer.plot_metrics_comparison(
                    evaluation_results['per_class_metrics']
                )
                visualization_files['metrics_comparison'] = metrics_path

            # 3. 信心分數分布
            conf_path = self.visualizer.plot_confidence_distribution(
                confidence_scores=predictions['confidences'],
                predictions=predictions['predictions'],
                true_labels=test_data['labels']
            )
            visualization_files['confidence_distribution'] = conf_path

            # 4. 錯誤分析圖表
            if error_analysis:
                error_path = self.visualizer.plot_error_analysis(error_analysis)
                visualization_files['error_analysis'] = error_path

            # 5. 穩健性測試圖表
            if robustness_results:
                robustness_path = self.visualizer.plot_robustness_results(robustness_results)
                visualization_files['robustness_results'] = robustness_path

            # 6. 綜合儀表板
            dashboard_path = self.visualizer.create_comprehensive_dashboard(
                evaluation_results, error_analysis, robustness_results
            )
            visualization_files['dashboard'] = dashboard_path

        except Exception as e:
            logger.error(f"生成視覺化時發生錯誤: {str(e)}")

        logger.info(f"視覺化生成完成，共生成 {len(visualization_files)} 個圖表")

        return visualization_files

    def generate_reports(self,
                        evaluation_results: Dict[str, Any],
                        error_analysis: Dict[str, Any],
                        robustness_results: Dict[str, Any],
                        explanations: List[Dict[str, Any]],
                        formats: List[str] = ['html', 'json']) -> Dict[str, str]:
        """生成評估報告"""

        logger.info("生成評估報告...")

        # 生成綜合報告
        report_files = self.report_generator.generate_comprehensive_report(
            evaluation_results=evaluation_results,
            error_analysis=error_analysis,
            robustness_results=robustness_results,
            explanations=explanations,
            formats=formats
        )

        logger.info(f"報告生成完成，共生成 {len(report_files)} 個報告文件")

        return report_files

    def run_comprehensive_evaluation(self,
                                   model_path: str,
                                   data_path: str,
                                   tokenizer_path: str = None,
                                   max_explainability_samples: int = 50,
                                   max_robustness_samples: int = 100,
                                   report_formats: List[str] = ['html', 'json'],
                                   enable_explainability: bool = True,
                                   enable_robustness: bool = True) -> Dict[str, Any]:
        """
        運行綜合評估流程

        Args:
            model_path: 模型路徑
            data_path: 測試數據路徑
            tokenizer_path: 標記器路徑（可選）
            max_explainability_samples: 可解釋性分析最大樣本數
            max_robustness_samples: 穩健性測試最大樣本數
            report_formats: 報告格式列表
            enable_explainability: 是否啟用可解釋性分析
            enable_robustness: 是否啟用穩健性測試

        Returns:
            綜合評估結果
        """

        logger.info("="*60)
        logger.info("開始運行 CyberPuppy 綜合評估")
        logger.info("="*60)

        start_time = datetime.now()

        try:
            # 1. 載入模型
            self.load_model(model_path, tokenizer_path)

            # 2. 載入測試數據
            test_data = self.load_test_data(data_path)

            # 3. 獲取模型預測
            predictions = self.get_model_predictions(test_data['texts'])

            # 4. 基礎評估
            evaluation_results = self.run_basic_evaluation(test_data, predictions)

            # 5. 錯誤分析
            error_analysis = self.run_error_analysis(test_data, predictions)

            # 6. 可解釋性分析（可選）
            explanations = []
            if enable_explainability:
                explanations = self.run_explainability_analysis(
                    test_data, max_explainability_samples
                )

            # 7. 穩健性測試（可選）
            robustness_results = {}
            if enable_robustness:
                robustness_results = self.run_robustness_tests(
                    test_data, max_robustness_samples
                )

            # 8. 生成視覺化
            visualization_files = self.generate_visualizations(
                evaluation_results, error_analysis, robustness_results,
                test_data, predictions
            )

            # 9. 生成報告
            report_files = self.generate_reports(
                evaluation_results, error_analysis, robustness_results,
                explanations, report_formats
            )

            # 10. 總結結果
            end_time = datetime.now()
            evaluation_duration = (end_time - start_time).total_seconds()

            final_results = {
                'evaluation_summary': {
                    'total_samples': len(test_data['texts']),
                    'overall_accuracy': evaluation_results['metrics'].get('accuracy', 0),
                    'overall_f1': evaluation_results['metrics'].get('f1', 0),
                    'total_errors': error_analysis.get('total_errors', 0),
                    'evaluation_duration_seconds': evaluation_duration,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat()
                },
                'evaluation_results': evaluation_results,
                'error_analysis': error_analysis,
                'robustness_results': robustness_results,
                'explanations': explanations,
                'visualization_files': visualization_files,
                'report_files': report_files,
                'config': self.config
            }

            # 保存最終結果
            final_results_path = os.path.join(
                self.output_dir,
                f"comprehensive_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            with open(final_results_path, 'w', encoding='utf-8') as f:
                json.dump(self._make_serializable(final_results), f, ensure_ascii=False, indent=2)

            logger.info("="*60)
            logger.info("CyberPuppy 綜合評估完成！")
            logger.info(f"總樣本數: {len(test_data['texts'])}")
            logger.info(f"整體準確率: {evaluation_results['metrics'].get('accuracy', 0):.3f}")
            logger.info(f"整體 F1 分數: {evaluation_results['metrics'].get('f1', 0):.3f}")
            logger.info(f"總錯誤數: {error_analysis.get('total_errors', 0)}")
            logger.info(f"評估用時: {evaluation_duration:.1f} 秒")
            logger.info(f"結果保存至: {self.output_dir}")
            logger.info("="*60)

            return final_results

        except Exception as e:
            logger.error(f"綜合評估過程中發生錯誤: {str(e)}")
            raise

    def _make_serializable(self, obj):
        """轉換對象為可序列化格式"""

        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        else:
            return obj

def main():
    """主函數"""

    parser = argparse.ArgumentParser(
        description="CyberPuppy 霸凌偵測系統綜合評估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python scripts/evaluate_comprehensive.py \\
    --model_path ./models/cyberpuppy_model \\
    --data_path ./data/test_dataset.csv \\
    --output_dir ./evaluation_results \\
    --formats html json pdf \\
    --max_explainability_samples 50 \\
    --max_robustness_samples 100

支援的數據格式:
  CSV: 需要包含 'text' 和 'label' 列
  JSON: 需要包含 'texts' 和 'labels' 鍵

支援的報告格式:
  html, json, pdf, excel
        """
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型文件路徑'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='測試數據文件路徑（CSV 或 JSON）'
    )

    parser.add_argument(
        '--tokenizer_path',
        type=str,
        help='標記器路徑（如果與模型路徑不同）'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='輸出目錄（默認: evaluation_results）'
    )

    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['html', 'json', 'pdf', 'excel'],
        default=['html', 'json'],
        help='報告格式（默認: html json）'
    )

    parser.add_argument(
        '--max_explainability_samples',
        type=int,
        default=50,
        help='可解釋性分析最大樣本數（默認: 50）'
    )

    parser.add_argument(
        '--max_robustness_samples',
        type=int,
        default=100,
        help='穩健性測試最大樣本數（默認: 100）'
    )

    parser.add_argument(
        '--disable_explainability',
        action='store_true',
        help='禁用可解釋性分析'
    )

    parser.add_argument(
        '--disable_robustness',
        action='store_true',
        help='禁用穩健性測試'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='配置文件路徑（JSON 格式）'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='詳細輸出'
    )

    args = parser.parse_args()

    # 設定日誌級別
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 載入配置
    config = {
        'output_dir': args.output_dir,
        'model_path': args.model_path,
        'data_path': args.data_path,
        'tokenizer_path': args.tokenizer_path,
        'report_formats': args.formats,
        'max_explainability_samples': args.max_explainability_samples,
        'max_robustness_samples': args.max_robustness_samples,
        'enable_explainability': not args.disable_explainability,
        'enable_robustness': not args.disable_robustness
    }

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
            config.update(file_config)

    # 驗證輸入文件
    if not os.path.exists(args.model_path):
        logger.error(f"模型路徑不存在: {args.model_path}")
        sys.exit(1)

    if not os.path.exists(args.data_path):
        logger.error(f"數據路徑不存在: {args.data_path}")
        sys.exit(1)

    try:
        # 創建評估器並運行
        evaluator = ComprehensiveEvaluator(config)

        results = evaluator.run_comprehensive_evaluation(
            model_path=args.model_path,
            data_path=args.data_path,
            tokenizer_path=args.tokenizer_path,
            max_explainability_samples=args.max_explainability_samples,
            max_robustness_samples=args.max_robustness_samples,
            report_formats=args.formats,
            enable_explainability=not args.disable_explainability,
            enable_robustness=not args.disable_robustness
        )

        print(f"\\n評估完成！結果保存在: {args.output_dir}")
        print(f"整體準確率: {results['evaluation_results']['metrics'].get('accuracy', 0):.3f}")
        print(f"整體 F1 分數: {results['evaluation_results']['metrics'].get('f1', 0):.3f}")

    except KeyboardInterrupt:
        logger.info("評估被用戶中斷")
        sys.exit(0)
    except Exception as e:
        logger.error(f"評估過程中發生錯誤: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()