#!/usr/bin/env python3
"""
針對訓練完成的 CyberPuppy 模型的專用評估腳本
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CyberPuppyModelEvaluator:
    """CyberPuppy 模型評估器"""

    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = output_dir
        self.results = {}

        # 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)

        print(f"初始化評估器...")
        print(f"模型路徑: {model_path}")
        print(f"輸出目錄: {output_dir}")

    def load_model(self):
        """載入模型"""
        print("載入模型...")

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            # 載入模型和分詞器
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # 設置評估模式
            self.model.eval()

            # 檢查是否有 GPU
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)

            print(f"模型載入成功，使用設備: {self.device}")
            print(f"模型配置: {self.model.config}")

        except Exception as e:
            print(f"模型載入失敗: {e}")
            raise

    def load_test_data(self, data_path: str) -> Dict[str, List]:
        """載入測試數據"""
        print(f"載入測試數據: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = []
        labels = []

        # 轉換數據格式
        for item in data:
            texts.append(item['text'])

            # 提取 toxicity 標籤
            toxicity_label = item['label'].get('toxicity', 'none')

            # 將標籤轉換為數字 (只有2個類別: none, toxic)
            if toxicity_label == 'none':
                labels.append(0)
            elif toxicity_label == 'toxic':
                labels.append(1)
            else:
                labels.append(0)  # 默認為 none

        print(f"載入 {len(texts)} 個測試樣本")
        print(f"標籤分布: {Counter(labels)}")

        return {
            'texts': texts,
            'labels': labels,
            'raw_data': data
        }

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> Tuple[List[int], List[float], List[List[float]]]:
        """批量預測"""
        print(f"開始批量預測 {len(texts)} 個樣本...")

        all_predictions = []
        all_confidences = []
        all_probabilities = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # 分詞
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )

                # 移到設備
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 預測
                outputs = self.model(**inputs)
                logits = outputs.logits

                # 計算概率
                probs = torch.softmax(logits, dim=-1)

                # 獲取預測結果
                predictions = torch.argmax(probs, dim=-1)
                confidences = torch.max(probs, dim=-1)[0]

                # 轉換為列表
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_confidences.extend(confidences.cpu().numpy().tolist())
                all_probabilities.extend(probs.cpu().numpy().tolist())

                if (i // batch_size + 1) % 10 == 0:
                    print(f"已處理 {i + len(batch_texts)} / {len(texts)} 樣本")

        print("批量預測完成")
        return all_predictions, all_confidences, all_probabilities

    def calculate_metrics(self, y_true: List[int], y_pred: List[int], y_proba: List[List[float]]) -> Dict[str, Any]:
        """計算評估指標"""
        print("計算評估指標...")

        # 基礎指標
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

        # 每類別指標
        per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        # 混淆矩陣
        cm = confusion_matrix(y_true, y_pred)

        # 分類報告 (只有2個類別)
        class_names = ['none', 'toxic']
        clf_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

        # 計算每類別的 AUC（如果有足夠的類別）
        auc_scores = {}
        if len(set(y_true)) > 1 and len(y_proba[0]) > 1:
            try:
                y_proba_array = np.array(y_proba)
                for i, class_name in enumerate(class_names):
                    if i < y_proba_array.shape[1]:
                        # 二分類 AUC
                        y_true_binary = [1 if label == i else 0 for label in y_true]
                        if len(set(y_true_binary)) > 1:  # 確保有兩個類別
                            auc_scores[class_name] = roc_auc_score(y_true_binary, y_proba_array[:, i])
            except Exception as e:
                print(f"AUC 計算失敗: {e}")

        metrics = {
            'accuracy': accuracy,
            'weighted_precision': precision,
            'weighted_recall': recall,
            'weighted_f1': f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_class_metrics': {
                class_names[i]: {
                    'precision': per_class_precision[i] if i < len(per_class_precision) else 0,
                    'recall': per_class_recall[i] if i < len(per_class_recall) else 0,
                    'f1': per_class_f1[i] if i < len(per_class_f1) else 0,
                    'support': int(per_class_support[i]) if i < len(per_class_support) else 0
                }
                for i in range(len(class_names))
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': clf_report,
            'auc_scores': auc_scores,
            'class_names': class_names
        }

        print("指標計算完成")
        return metrics

    def analyze_errors(self, texts: List[str], y_true: List[int], y_pred: List[int],
                      y_proba: List[List[float]]) -> Dict[str, Any]:
        """錯誤分析"""
        print("進行錯誤分析...")

        class_names = ['none', 'toxic']

        # 找出錯誤案例
        errors = []
        correct_predictions = []

        for i, (text, true_label, pred_label, proba) in enumerate(zip(texts, y_true, y_pred, y_proba)):
            confidence = max(proba)

            if true_label != pred_label:
                errors.append({
                    'index': i,
                    'text': text,
                    'true_label': class_names[true_label],
                    'pred_label': class_names[pred_label],
                    'confidence': confidence,
                    'probabilities': proba,
                    'error_type': f"{class_names[true_label]} -> {class_names[pred_label]}"
                })
            else:
                correct_predictions.append({
                    'index': i,
                    'text': text,
                    'label': class_names[true_label],
                    'confidence': confidence
                })

        # 按置信度排序錯誤案例
        errors.sort(key=lambda x: x['confidence'], reverse=True)

        # 分析錯誤模式
        error_patterns = Counter([error['error_type'] for error in errors])

        # 低置信度正確預測
        low_confidence_correct = [pred for pred in correct_predictions if pred['confidence'] < 0.7]
        low_confidence_correct.sort(key=lambda x: x['confidence'])

        # 高置信度錯誤預測
        high_confidence_errors = [error for error in errors if error['confidence'] > 0.7]

        error_analysis = {
            'total_errors': len(errors),
            'total_correct': len(correct_predictions),
            'error_rate': len(errors) / (len(errors) + len(correct_predictions)),
            'error_patterns': dict(error_patterns),
            'top_errors': errors[:20],  # 前20個錯誤案例
            'high_confidence_errors': high_confidence_errors[:10],
            'low_confidence_correct': low_confidence_correct[:10],
            'confidence_distribution': {
                'all_predictions': [max(proba) for proba in y_proba],
                'correct_predictions': [pred['confidence'] for pred in correct_predictions],
                'error_predictions': [error['confidence'] for error in errors]
            }
        }

        print(f"錯誤分析完成: {len(errors)} 個錯誤案例")
        return error_analysis

    def create_visualizations(self, metrics: Dict[str, Any], error_analysis: Dict[str, Any]):
        """創建視覺化圖表"""
        print("創建視覺化圖表...")

        # 設置圖表風格
        plt.style.use('default')

        # 1. 混淆矩陣
        plt.figure(figsize=(8, 6))
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=metrics['class_names'],
                   yticklabels=metrics['class_names'])
        plt.title('混淆矩陣')
        plt.xlabel('預測標籤')
        plt.ylabel('真實標籤')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 每類別指標對比
        plt.figure(figsize=(12, 6))
        classes = metrics['class_names']
        x = np.arange(len(classes))
        width = 0.25

        precision_scores = [metrics['per_class_metrics'][c]['precision'] for c in classes]
        recall_scores = [metrics['per_class_metrics'][c]['recall'] for c in classes]
        f1_scores = [metrics['per_class_metrics'][c]['f1'] for c in classes]

        plt.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        plt.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

        plt.xlabel('類別')
        plt.ylabel('分數')
        plt.title('每類別指標對比')
        plt.xticks(x, classes)
        plt.legend()
        plt.ylim(0, 1.1)

        # 添加數值標籤
        for i, (p, r, f) in enumerate(zip(precision_scores, recall_scores, f1_scores)):
            plt.text(i - width, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=8)
            plt.text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
            plt.text(i + width, f + 0.01, f'{f:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 置信度分布
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        correct_conf = error_analysis['confidence_distribution']['correct_predictions']
        error_conf = error_analysis['confidence_distribution']['error_predictions']

        plt.hist(correct_conf, alpha=0.7, label='正確預測', bins=20, color='green')
        plt.hist(error_conf, alpha=0.7, label='錯誤預測', bins=20, color='red')
        plt.xlabel('置信度')
        plt.ylabel('頻率')
        plt.title('預測置信度分布')
        plt.legend()

        plt.subplot(1, 2, 2)
        error_patterns = error_analysis['error_patterns']
        if error_patterns:
            patterns = list(error_patterns.keys())
            counts = list(error_patterns.values())

            plt.bar(range(len(patterns)), counts)
            plt.xlabel('錯誤類型')
            plt.ylabel('錯誤數量')
            plt.title('錯誤模式分布')
            plt.xticks(range(len(patterns)), patterns, rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("視覺化圖表創建完成")

    def generate_html_report(self, metrics: Dict[str, Any], error_analysis: Dict[str, Any],
                           evaluation_time: str) -> str:
        """生成 HTML 報告"""
        print("生成 HTML 報告...")

        # 檢查關鍵指標是否達標
        bullying_f1 = metrics['per_class_metrics']['toxic']['f1']
        overall_f1 = metrics['weighted_f1']
        accuracy = metrics['accuracy']

        # 達標狀態
        f1_target_met = bullying_f1 >= 0.75
        precision_target_met = metrics['per_class_metrics']['toxic']['precision'] >= 0.70
        recall_target_met = metrics['per_class_metrics']['toxic']['recall'] >= 0.70

        html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CyberPuppy 模型評估報告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .metric-card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .status-badge {{ padding: 5px 10px; border-radius: 20px; color: white; font-weight: bold; }}
        .status-pass {{ background-color: #28a745; }}
        .status-fail {{ background-color: #dc3545; }}
        .status-warning {{ background-color: #ffc107; color: #212529; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .error-sample {{ background: #f8f9fa; border-left: 4px solid #dc3545; padding: 15px; margin: 10px 0; border-radius: 4px; }}
        .improvement-section {{ background: #e3f2fd; border-left: 4px solid #2196f3; padding: 20px; margin: 20px 0; border-radius: 4px; }}
        .chart-container {{ text-align: center; margin: 20px 0; }}
        .chart-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }}
        .summary-stats {{ display: flex; justify-content: space-around; flex-wrap: wrap; margin: 20px 0; }}
        .stat-item {{ text-align: center; padding: 15px; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .stat-label {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 CyberPuppy 霸凌偵測模型評估報告</h1>
        <p>評估時間: {evaluation_time}</p>
        <p>模型路徑: {self.model_path}</p>
    </div>

    <div class="metric-card">
        <h2>📊 評估總結</h2>
        <div class="summary-stats">
            <div class="stat-item">
                <div class="stat-value">{accuracy:.3f}</div>
                <div class="stat-label">整體準確率</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{overall_f1:.3f}</div>
                <div class="stat-label">整體 F1 分數</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{bullying_f1:.3f}</div>
                <div class="stat-label">霸凌偵測 F1</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{error_analysis['total_errors']}</div>
                <div class="stat-label">錯誤案例數</div>
            </div>
        </div>
    </div>

    <div class="metric-card">
        <h2>🎯 關鍵指標達標狀況</h2>
        <div class="metric-grid">
            <div>
                <h3>霸凌偵測 F1 分數</h3>
                <span class="status-badge {'status-pass' if f1_target_met else 'status-fail'}">
                    {bullying_f1:.3f} {'✓ 達標' if f1_target_met else '✗ 未達標'} (目標: ≥0.75)
                </span>
            </div>
            <div>
                <h3>霸凌偵測 Precision</h3>
                <span class="status-badge {'status-pass' if precision_target_met else 'status-fail'}">
                    {metrics['per_class_metrics']['toxic']['precision']:.3f} {'✓ 達標' if precision_target_met else '✗ 未達標'} (目標: ≥0.70)
                </span>
            </div>
            <div>
                <h3>霸凌偵測 Recall</h3>
                <span class="status-badge {'status-pass' if recall_target_met else 'status-fail'}">
                    {metrics['per_class_metrics']['toxic']['recall']:.3f} {'✓ 達標' if recall_target_met else '✗ 未達標'} (目標: ≥0.70)
                </span>
            </div>
        </div>
    </div>

    <div class="metric-card">
        <h2>📈 詳細指標</h2>
        <table>
            <tr>
                <th>類別</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Support</th>
            </tr>
        """

        for class_name in metrics['class_names']:
            class_metrics = metrics['per_class_metrics'][class_name]
            html_content += f"""
            <tr>
                <td>{class_name}</td>
                <td>{class_metrics['precision']:.3f}</td>
                <td>{class_metrics['recall']:.3f}</td>
                <td>{class_metrics['f1']:.3f}</td>
                <td>{class_metrics['support']}</td>
            </tr>
            """

        html_content += f"""
        </table>
    </div>

    <div class="metric-card">
        <h2>🔍 錯誤分析</h2>
        <h3>錯誤模式分布</h3>
        <table>
            <tr><th>錯誤類型</th><th>數量</th><th>百分比</th></tr>
        """

        total_errors = error_analysis['total_errors']
        for pattern, count in error_analysis['error_patterns'].items():
            percentage = (count / total_errors * 100) if total_errors > 0 else 0
            html_content += f"""
            <tr>
                <td>{pattern}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>
            """

        html_content += f"""
        </table>

        <h3>高置信度錯誤案例 (前5個)</h3>
        """

        for i, error in enumerate(error_analysis['high_confidence_errors'][:5]):
            html_content += f"""
            <div class="error-sample">
                <strong>案例 {i+1}</strong> (置信度: {error['confidence']:.3f})<br>
                <strong>文本:</strong> {error['text'][:200]}{'...' if len(error['text']) > 200 else ''}<br>
                <strong>真實標籤:</strong> {error['true_label']} → <strong>預測標籤:</strong> {error['pred_label']}
            </div>
            """

        html_content += """
    </div>

    <div class="chart-container">
        <h2>📊 視覺化分析</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px;">
            <div>
                <h3>混淆矩陣</h3>
                <img src="confusion_matrix.png" alt="混淆矩陣">
            </div>
            <div>
                <h3>每類別指標對比</h3>
                <img src="per_class_metrics.png" alt="每類別指標">
            </div>
            <div style="grid-column: span 2;">
                <h3>錯誤分析</h3>
                <img src="error_analysis.png" alt="錯誤分析">
            </div>
        </div>
    </div>
        """

        # 改進建議
        improvements = []
        if not f1_target_met:
            improvements.append("霸凌偵測 F1 分數未達標準 (0.75)，建議增加訓練數據或調整模型架構")
        if not precision_target_met:
            improvements.append("霸凌偵測 Precision 偏低，可能存在較多誤報，建議提高決策閾值或改進特徵提取")
        if not recall_target_met:
            improvements.append("霸凌偵測 Recall 偏低，可能漏檢較多真實案例，建議增加負樣本數據或調整損失函數")

        if error_analysis['total_errors'] > 0:
            top_error_pattern = max(error_analysis['error_patterns'].items(), key=lambda x: x[1])
            improvements.append(f"主要錯誤模式為 '{top_error_pattern[0]}'，建議針對此類案例進行數據增強")

        if len(error_analysis['high_confidence_errors']) > 5:
            improvements.append("存在較多高置信度錯誤預測，建議檢查模型過度自信問題")

        if improvements:
            html_content += f"""
    <div class="improvement-section">
        <h2>💡 改進建議</h2>
        <ul>
        """
            for improvement in improvements:
                html_content += f"<li>{improvement}</li>"

            html_content += """
        </ul>
    </div>
            """

        html_content += """
    <div class="metric-card">
        <h2>📋 評估總結</h2>
        <p>本次評估完成了對 CyberPuppy 霸凌偵測模型的全面測試，包括基礎指標計算、錯誤分析和視覺化展示。</p>
        <p><strong>建議下一步行動:</strong></p>
        <ul>
            <li>根據錯誤分析結果優化訓練數據</li>
            <li>考慮模型架構調整以提升性能</li>
            <li>實施更嚴格的數據清理流程</li>
            <li>進行更多樣化的測試場景驗證</li>
        </ul>
    </div>

    <footer style="text-align: center; margin-top: 50px; color: #666;">
        <p>CyberPuppy 霸凌偵測系統 - 讓網路環境更安全</p>
    </footer>
</body>
</html>
        """

        # 保存 HTML 報告
        html_path = os.path.join(self.output_dir, 'evaluation_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML 報告已保存: {html_path}")
        return html_path

    def save_results(self, metrics: Dict[str, Any], error_analysis: Dict[str, Any]):
        """保存評估結果"""
        print("保存評估結果...")

        # 保存 JSON 格式結果
        results = {
            'evaluation_time': datetime.now().isoformat(),
            'model_path': self.model_path,
            'metrics': metrics,
            'error_analysis': error_analysis,
            'target_achievement': {
                'bullying_f1_target': 0.75,
                'bullying_f1_actual': metrics['per_class_metrics']['toxic']['f1'],
                'bullying_f1_achieved': metrics['per_class_metrics']['toxic']['f1'] >= 0.75,
                'precision_target': 0.70,
                'precision_actual': metrics['per_class_metrics']['toxic']['precision'],
                'precision_achieved': metrics['per_class_metrics']['toxic']['precision'] >= 0.70,
                'recall_target': 0.70,
                'recall_actual': metrics['per_class_metrics']['toxic']['recall'],
                'recall_achieved': metrics['per_class_metrics']['toxic']['recall'] >= 0.70
            }
        }

        json_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"JSON 結果已保存: {json_path}")

        # 保存簡化的指標總結
        summary = {
            'overall_accuracy': metrics['accuracy'],
            'overall_f1': metrics['weighted_f1'],
            'bullying_detection_f1': metrics['per_class_metrics']['toxic']['f1'],
            'bullying_detection_precision': metrics['per_class_metrics']['toxic']['precision'],
            'bullying_detection_recall': metrics['per_class_metrics']['toxic']['recall'],
            'total_errors': error_analysis['total_errors'],
            'error_rate': error_analysis['error_rate'],
            'targets_achieved': results['target_achievement']
        }

        summary_path = os.path.join(self.output_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return results

    def run_evaluation(self, data_path: str):
        """執行完整評估"""
        print("="*60)
        print("開始 CyberPuppy 模型評估")
        print("="*60)

        start_time = datetime.now()

        try:
            # 1. 載入模型
            self.load_model()

            # 2. 載入測試數據
            test_data = self.load_test_data(data_path)

            # 3. 進行預測
            predictions, confidences, probabilities = self.predict_batch(test_data['texts'])

            # 4. 計算指標
            metrics = self.calculate_metrics(test_data['labels'], predictions, probabilities)

            # 5. 錯誤分析
            error_analysis = self.analyze_errors(
                test_data['texts'], test_data['labels'], predictions, probabilities
            )

            # 6. 創建視覺化
            self.create_visualizations(metrics, error_analysis)

            # 7. 生成報告
            evaluation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            html_report = self.generate_html_report(metrics, error_analysis, evaluation_time)

            # 8. 保存結果
            results = self.save_results(metrics, error_analysis)

            # 9. 輸出總結
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print("="*60)
            print("評估完成！")
            print(f"評估用時: {duration:.1f} 秒")
            print(f"測試樣本數: {len(test_data['texts'])}")
            print(f"整體準確率: {metrics['accuracy']:.3f}")
            print(f"整體 F1 分數: {metrics['weighted_f1']:.3f}")
            print(f"霸凌偵測 F1: {metrics['per_class_metrics']['toxic']['f1']:.3f}")
            print(f"錯誤案例數: {error_analysis['total_errors']}")

            # 目標達成情況
            print("\n🎯 目標達成情況:")
            targets = results['target_achievement']
            print(f"霸凌偵測 F1 ≥ 0.75: {'✓' if targets['bullying_f1_achieved'] else '✗'} ({targets['bullying_f1_actual']:.3f})")
            print(f"Precision ≥ 0.70: {'✓' if targets['precision_achieved'] else '✗'} ({targets['precision_actual']:.3f})")
            print(f"Recall ≥ 0.70: {'✓' if targets['recall_achieved'] else '✗'} ({targets['recall_actual']:.3f})")

            print(f"\n📁 結果保存在: {self.output_dir}")
            print(f"📊 HTML 報告: {html_report}")
            print("="*60)

            return results

        except Exception as e:
            print(f"評估過程中發生錯誤: {e}")
            raise

def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description='CyberPuppy 模型評估工具')
    parser.add_argument('--model_path', type=str, required=True, help='模型路徑')
    parser.add_argument('--data_path', type=str, required=True, help='測試數據路徑')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='輸出目錄')

    args = parser.parse_args()

    # 創建評估器
    evaluator = CyberPuppyModelEvaluator(args.model_path, args.output_dir)

    # 運行評估
    evaluator.run_evaluation(args.data_path)

if __name__ == '__main__':
    main()