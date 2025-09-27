"""
可解釋性分析模組
提供霸凌偵測模型的可解釋性分析工具
"""

import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
from dataclasses import dataclass
import json
import os
from datetime import datetime
from collections import defaultdict

# 第三方可解釋性庫
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP 未安裝，SHAP 功能將不可用")

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME 未安裝，LIME 功能將不可用")

try:
    from captum.attr import (
        IntegratedGradients,
        LayerIntegratedGradients,
        TokenReferenceBase,
        visualization
    )
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    logging.warning("Captum 未安裝，Integrated Gradients 功能將不可用")

logger = logging.getLogger(__name__)

@dataclass
class ExplanationResult:
    """解釋結果數據結構"""
    text: str
    prediction: str
    confidence: float
    method: str  # 'shap', 'lime', 'integrated_gradients', 'attention'
    token_attributions: List[Tuple[str, float]]  # (token, attribution_score)
    global_explanation: Optional[Dict[str, Any]] = None
    visualization_path: Optional[str] = None

class ExplainabilityAnalyzer:
    """可解釋性分析主類"""

    def __init__(self, model, tokenizer, output_dir: str = "explainability_results"):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 設定設備
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # 初始化解釋器
        self._init_explainers()

    def _init_explainers(self):
        """初始化各種解釋器"""

        # Integrated Gradients
        if CAPTUM_AVAILABLE:
            self.integrated_gradients = IntegratedGradients(self.model)
            self.token_reference = TokenReferenceBase(reference_token_idx=self.tokenizer.pad_token_id)

        # SHAP
        if SHAP_AVAILABLE:
            self.shap_explainer = None  # 延遲初始化

        # LIME
        if LIME_AVAILABLE:
            self.lime_explainer = LimeTextExplainer(
                class_names=['none', 'toxic', 'severe'],
                mode='classification'
            )

    def explain_prediction(self,
                          text: str,
                          methods: List[str] = ['integrated_gradients', 'attention'],
                          visualize: bool = True) -> List[ExplanationResult]:
        """
        解釋單個預測結果

        Args:
            text: 待解釋的文本
            methods: 使用的解釋方法列表
            visualize: 是否生成視覺化

        Returns:
            解釋結果列表
        """

        logger.info(f"開始解釋預測: {text[:50]}...")

        results = []

        # 獲取預測結果
        prediction, confidence = self._get_prediction(text)

        for method in methods:
            try:
                if method == 'integrated_gradients' and CAPTUM_AVAILABLE:
                    result = self._explain_with_integrated_gradients(text, prediction, confidence)
                elif method == 'shap' and SHAP_AVAILABLE:
                    result = self._explain_with_shap(text, prediction, confidence)
                elif method == 'lime' and LIME_AVAILABLE:
                    result = self._explain_with_lime(text, prediction, confidence)
                elif method == 'attention':
                    result = self._explain_with_attention(text, prediction, confidence)
                else:
                    logger.warning(f"不支援的解釋方法: {method}")
                    continue

                if result and visualize:
                    self._create_visualization(result)

                results.append(result)

            except Exception as e:
                logger.error(f"使用 {method} 解釋時發生錯誤: {str(e)}")

        return results

    def explain_batch(self,
                     texts: List[str],
                     methods: List[str] = ['integrated_gradients'],
                     max_examples: int = 50) -> Dict[str, List[ExplanationResult]]:
        """
        批量解釋預測結果

        Args:
            texts: 文本列表
            methods: 解釋方法列表
            max_examples: 最大解釋數量

        Returns:
            按方法分組的解釋結果
        """

        logger.info(f"開始批量解釋 {len(texts)} 個樣本...")

        results_by_method = {method: [] for method in methods}

        # 限制處理數量
        texts_to_process = texts[:max_examples]

        for i, text in enumerate(texts_to_process):
            if i % 10 == 0:
                logger.info(f"處理進度: {i + 1}/{len(texts_to_process)}")

            try:
                explanations = self.explain_prediction(text, methods, visualize=False)
                for explanation in explanations:
                    results_by_method[explanation.method].append(explanation)

            except Exception as e:
                logger.error(f"解釋第 {i + 1} 個樣本時發生錯誤: {str(e)}")

        return results_by_method

    def _get_prediction(self, text: str) -> Tuple[str, float]:
        """獲取模型預測結果"""

        # 編碼文本
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )

        # 移動到設備
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 獲取預測
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions.max().item()

        # 轉換為標籤
        label_mapping = {0: 'none', 1: 'toxic', 2: 'severe'}
        prediction = label_mapping.get(predicted_class, 'unknown')

        return prediction, confidence

    def _explain_with_integrated_gradients(self,
                                         text: str,
                                         prediction: str,
                                         confidence: float) -> ExplanationResult:
        """使用 Integrated Gradients 進行解釋"""

        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum 未安裝")

        # 編碼文本
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # 生成基準線（全部填充為 PAD token）
        baseline_ids = self.token_reference.generate_reference(
            input_ids.shape[-1], device=self.device
        ).unsqueeze(0)

        # 計算 Integrated Gradients
        def forward_func(input_ids):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits

        attributions = self.integrated_gradients.attribute(
            input_ids,
            baseline_ids,
            target=None,
            n_steps=50,
            return_convergence_delta=False
        )

        # 處理歸因分數
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().numpy()

        # 獲取 tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu())

        # 組合 token 和歸因分數
        token_attributions = []
        for token, score in zip(tokens, attributions):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                token_attributions.append((token, float(score)))

        return ExplanationResult(
            text=text,
            prediction=prediction,
            confidence=confidence,
            method='integrated_gradients',
            token_attributions=token_attributions
        )

    def _explain_with_shap(self,
                          text: str,
                          prediction: str,
                          confidence: float) -> ExplanationResult:
        """使用 SHAP 進行解釋"""

        if not SHAP_AVAILABLE:
            raise ImportError("SHAP 未安裝")

        # 定義模型包裝函數
        def model_wrapper(texts):
            results = []
            for text in texts:
                pred, conf = self._get_prediction(text)
                # 返回所有類別的概率
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

                results.append(probs)

            return np.array(results)

        # 初始化 SHAP 解釋器（如果尚未初始化）
        if self.shap_explainer is None:
            # 創建背景數據集
            background_texts = [
                "這是正常的對話",
                "今天天氣很好",
                "謝謝你的幫助",
                ""  # 空文本作為基準
            ]
            self.shap_explainer = shap.Explainer(model_wrapper, background_texts)

        # 計算 SHAP 值
        shap_values = self.shap_explainer([text])

        # 處理結果
        tokens = text.split()  # 簡單的分詞
        if len(tokens) == 0:
            tokens = [text]

        # 獲取預測類別的 SHAP 值
        pred_class_idx = {'none': 0, 'toxic': 1, 'severe': 2}.get(prediction, 0)

        if len(shap_values.values.shape) > 2:
            attributions = shap_values.values[0, :, pred_class_idx]
        else:
            attributions = shap_values.values[0]

        # 確保長度匹配
        min_length = min(len(tokens), len(attributions))
        token_attributions = [
            (tokens[i], float(attributions[i]))
            for i in range(min_length)
        ]

        return ExplanationResult(
            text=text,
            prediction=prediction,
            confidence=confidence,
            method='shap',
            token_attributions=token_attributions
        )

    def _explain_with_lime(self,
                          text: str,
                          prediction: str,
                          confidence: float) -> ExplanationResult:
        """使用 LIME 進行解釋"""

        if not LIME_AVAILABLE:
            raise ImportError("LIME 未安裝")

        # 定義預測函數
        def predict_fn(texts):
            results = []
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

                results.append(probs)

            return np.array(results)

        # 生成解釋
        explanation = self.lime_explainer.explain_instance(
            text,
            predict_fn,
            num_features=20,
            num_samples=1000
        )

        # 獲取特徵重要性
        feature_importance = explanation.as_list()

        # 轉換為 token attributions 格式
        token_attributions = [
            (feature, score) for feature, score in feature_importance
        ]

        return ExplanationResult(
            text=text,
            prediction=prediction,
            confidence=confidence,
            method='lime',
            token_attributions=token_attributions
        )

    def _explain_with_attention(self,
                               text: str,
                               prediction: str,
                               confidence: float) -> ExplanationResult:
        """使用注意力權重進行解釋"""

        # 編碼文本
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 獲取注意力權重
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions  # (layer, batch, head, seq_len, seq_len)

        # 平均所有層和所有頭的注意力權重
        # 只取最後一層的注意力
        last_layer_attention = attentions[-1]  # (batch, head, seq_len, seq_len)

        # 平均所有頭
        avg_attention = last_layer_attention.mean(dim=1)  # (batch, seq_len, seq_len)

        # 計算每個 token 的總注意力分數（從 CLS token 開始）
        cls_attention = avg_attention[0, 0, :]  # CLS token 對所有 token 的注意力
        attention_scores = cls_attention.cpu().numpy()

        # 獲取 tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().cpu())

        # 組合 token 和注意力分數
        token_attributions = []
        for token, score in zip(tokens, attention_scores):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                token_attributions.append((token, float(score)))

        return ExplanationResult(
            text=text,
            prediction=prediction,
            confidence=confidence,
            method='attention',
            token_attributions=token_attributions
        )

    def _create_visualization(self, result: ExplanationResult):
        """創建可視化圖表"""

        plt.figure(figsize=(12, 6))

        # 提取 tokens 和分數
        tokens = [item[0] for item in result.token_attributions]
        scores = [item[1] for item in result.token_attributions]

        # 限制顯示的 token 數量
        max_tokens = 20
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            scores = scores[:max_tokens]

        # 創建顏色映射
        colors = ['red' if score < 0 else 'green' for score in scores]
        alphas = [min(abs(score) * 2, 1.0) for score in scores]

        # 創建條形圖
        bars = plt.barh(range(len(tokens)), scores, color=colors, alpha=0.7)

        # 設定透明度
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)

        plt.yticks(range(len(tokens)), tokens)
        plt.xlabel('Attribution Score')
        plt.title(f'{result.method.upper()} Explanation\\n'
                 f'Text: {result.text[:50]}...\\n'
                 f'Prediction: {result.prediction} (Confidence: {result.confidence:.3f})')

        plt.tight_layout()

        # 保存圖片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.method}_explanation_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        # 更新結果中的可視化路徑
        result.visualization_path = filepath

        logger.info(f"可視化已保存至: {filepath}")

    def generate_global_explanations(self,
                                   texts: List[str],
                                   labels: List[str],
                                   method: str = 'integrated_gradients') -> Dict[str, Any]:
        """
        生成全局解釋（特徵重要性統計）

        Args:
            texts: 文本列表
            labels: 標籤列表
            method: 解釋方法

        Returns:
            全局解釋結果
        """

        logger.info(f"開始生成全局解釋，方法: {method}")

        # 按類別分組
        label_groups = {'none': [], 'toxic': [], 'severe': []}
        for text, label in zip(texts, labels):
            if label in label_groups:
                label_groups[label].append(text)

        global_explanations = {}

        for label, label_texts in label_groups.items():
            if not label_texts:
                continue

            logger.info(f"處理 {label} 類別，共 {len(label_texts)} 個樣本")

            # 獲取該類別的解釋
            explanations = self.explain_batch(
                label_texts[:50],  # 限制數量以提高效率
                methods=[method],
                max_examples=50
            )[method]

            # 統計詞彙重要性
            token_importance = defaultdict(list)
            for explanation in explanations:
                for token, score in explanation.token_attributions:
                    token_importance[token].append(score)

            # 計算平均重要性
            avg_importance = {
                token: np.mean(scores)
                for token, scores in token_importance.items()
                if len(scores) >= 2  # 至少出現兩次
            }

            # 排序並取前20
            top_tokens = sorted(
                avg_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:20]

            global_explanations[label] = {
                'top_positive_tokens': [(token, score) for token, score in top_tokens if score > 0],
                'top_negative_tokens': [(token, score) for token, score in top_tokens if score < 0],
                'sample_count': len(explanations),
                'method': method
            }

        return global_explanations

    def save_explanations(self, explanations: List[ExplanationResult]):
        """保存解釋結果"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"explanations_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        # 轉換為可序列化格式
        serializable_explanations = []
        for explanation in explanations:
            explanation_dict = {
                'text': explanation.text,
                'prediction': explanation.prediction,
                'confidence': explanation.confidence,
                'method': explanation.method,
                'token_attributions': explanation.token_attributions,
                'visualization_path': explanation.visualization_path
            }
            if explanation.global_explanation:
                explanation_dict['global_explanation'] = explanation.global_explanation

            serializable_explanations.append(explanation_dict)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_explanations, f, ensure_ascii=False, indent=2)

        logger.info(f"解釋結果已保存至: {filepath}")

class SHAPExplainer:
    """SHAP 專門解釋器"""

    def __init__(self, model, tokenizer):
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP 未安裝，請安裝: pip install shap")

        self.model = model
        self.tokenizer = tokenizer

    def explain_with_plots(self, texts: List[str], max_display: int = 10):
        """使用 SHAP 進行解釋並生成圖表"""

        # 實現 SHAP 特定的解釋和可視化
        pass

class LIMEExplainer:
    """LIME 專門解釋器"""

    def __init__(self, model, tokenizer):
        if not LIME_AVAILABLE:
            raise ImportError("LIME 未安裝，請安裝: pip install lime")

        self.model = model
        self.tokenizer = tokenizer

    def explain_with_html(self, text: str, output_path: str):
        """使用 LIME 進行解釋並生成 HTML 報告"""

        # 實現 LIME 特定的解釋和 HTML 生成
        pass

# 工具函數
def compare_explanations(explanations: List[ExplanationResult]) -> Dict[str, Any]:
    """比較不同解釋方法的結果"""

    if len(explanations) < 2:
        return {'error': '需要至少兩個解釋結果進行比較'}

    comparison_results = {
        'methods_compared': [exp.method for exp in explanations],
        'text': explanations[0].text,
        'prediction_consistency': len(set(exp.prediction for exp in explanations)) == 1,
        'confidence_range': (
            min(exp.confidence for exp in explanations),
            max(exp.confidence for exp in explanations)
        )
    }

    # 計算 token 重要性相關性
    if len(explanations) >= 2:
        # 取前兩個解釋進行相關性分析
        exp1, exp2 = explanations[0], explanations[1]

        # 獲取共同的 tokens
        tokens1 = {token for token, _ in exp1.token_attributions}
        tokens2 = {token for token, _ in exp2.token_attributions}
        common_tokens = tokens1.intersection(tokens2)

        if common_tokens:
            scores1 = {token: score for token, score in exp1.token_attributions if token in common_tokens}
            scores2 = {token: score for token, score in exp2.token_attributions if token in common_tokens}

            # 計算相關係數
            s1_values = [scores1[token] for token in common_tokens]
            s2_values = [scores2[token] for token in common_tokens]

            correlation = np.corrcoef(s1_values, s2_values)[0, 1] if len(s1_values) > 1 else 0

            comparison_results['token_correlation'] = {
                'correlation_coefficient': float(correlation),
                'common_tokens_count': len(common_tokens),
                'method1': exp1.method,
                'method2': exp2.method
            }

    return comparison_results