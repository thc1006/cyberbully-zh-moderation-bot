"""
CyberPuppy 評估模組
霸凌偵測系統的全面評估與驗證工具

提供以下功能:
- 標準指標計算 (Precision, Recall, F1, Accuracy)
- 深度錯誤分析與案例研究
- 可解釋性分析 (SHAP, LIME, Integrated Gradients, Attention)
- 穩健性測試 (對抗攻擊、輸入變化容忍度)
- 結果視覺化 (圖表、儀表板、報告)
- 多格式報告生成 (HTML, PDF, JSON, Excel)
"""

# 核心評估模組
from .metrics import (
    CSVExporter,
    EvaluationReport,
    MetricsCalculator,
    OnlineMonitor,
    PrometheusExporter,
    SessionContext,
    ModelEvaluator
)

# 錯誤分析模組
from .error_analysis import (
    ErrorAnalyzer,
    ErrorCase,
    ErrorPattern,
    FalsePositiveAnalyzer,
    FalseNegativeAnalyzer
)

# 可解釋性分析模組
from .explainability import (
    ExplainabilityAnalyzer,
    ExplanationResult,
    SHAPExplainer,
    LIMEExplainer,
    compare_explanations
)

# 穩健性測試模組
from .robustness import (
    RobustnessTestSuite,
    AdversarialTester,
    RobustnessTestResult,
    AdversarialAttackResult
)

# 視覺化模組
from .visualization import (
    ResultVisualizer,
    ConfusionMatrixPlotter,
    AttentionVisualizer
)

# 報告生成模組
from .reports import (
    ReportGenerator,
    ReportConfig,
    HTMLReportGenerator,
    PDFReportGenerator
)

__all__ = [
    # 核心評估
    "MetricsCalculator",
    "ModelEvaluator",
    "SessionContext",
    "OnlineMonitor",
    "PrometheusExporter",
    "CSVExporter",
    "EvaluationReport",

    # 錯誤分析
    "ErrorAnalyzer",
    "ErrorCase",
    "ErrorPattern",
    "FalsePositiveAnalyzer",
    "FalseNegativeAnalyzer",

    # 可解釋性
    "ExplainabilityAnalyzer",
    "ExplanationResult",
    "SHAPExplainer",
    "LIMEExplainer",
    "compare_explanations",

    # 穩健性測試
    "RobustnessTestSuite",
    "AdversarialTester",
    "RobustnessTestResult",
    "AdversarialAttackResult",

    # 視覺化
    "ResultVisualizer",
    "ConfusionMatrixPlotter",
    "AttentionVisualizer",

    # 報告生成
    "ReportGenerator",
    "ReportConfig",
    "HTMLReportGenerator",
    "PDFReportGenerator"
]

# 版本信息
__version__ = "1.0.0"
__author__ = "CyberPuppy Team"
__description__ = "霸凌偵測系統的全面評估與驗證工具"

# 快捷函數
def quick_evaluate(y_true, y_pred, output_dir="evaluation_results"):
    """
    快速評估函數

    Args:
        y_true: 真實標籤列表
        y_pred: 預測標籤列表
        output_dir: 輸出目錄

    Returns:
        評估結果字典
    """
    from .metrics import MetricsCalculator

    calculator = MetricsCalculator()

    results = {
        'basic_metrics': calculator.calculate_basic_metrics(y_true, y_pred),
        'per_class_metrics': calculator.calculate_per_class_metrics(y_true, y_pred),
        'confusion_matrix': calculator.calculate_confusion_matrix(y_true, y_pred).tolist()
    }

    return results

def comprehensive_evaluate(model, tokenizer, test_data, output_dir="evaluation_results"):
    """
    全面評估函數

    Args:
        model: 評估模型
        tokenizer: 文本標記器
        test_data: 測試數據 (包含 'texts' 和 'labels')
        output_dir: 輸出目錄

    Returns:
        全面評估結果
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 初始化各種評估器
    metrics_calc = MetricsCalculator()
    error_analyzer = ErrorAnalyzer(output_dir)
    explainer = ExplainabilityAnalyzer(model, tokenizer, output_dir)
    robustness_tester = RobustnessTestSuite(model, tokenizer, output_dir)
    visualizer = ResultVisualizer(output_dir)
    report_gen = ReportGenerator(output_dir)

    # TODO: 實現模型預測邏輯
    # predictions = get_model_predictions(model, tokenizer, test_data['texts'])

    # 暫時返回結構
    return {
        'message': '全面評估功能需要實現模型預測邏輯',
        'available_evaluators': {
            'metrics_calculator': type(metrics_calc).__name__,
            'error_analyzer': type(error_analyzer).__name__,
            'explainability_analyzer': type(explainer).__name__,
            'robustness_tester': type(robustness_tester).__name__,
            'visualizer': type(visualizer).__name__,
            'report_generator': type(report_gen).__name__
        }
    }
