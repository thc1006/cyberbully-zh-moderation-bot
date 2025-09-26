"""
結果視覺化模組
提供評估結果的各種視覺化功能
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, classification_report
import io
import base64
import os
from datetime import datetime
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ResultVisualizer:
    """結果視覺化主類"""

    def __init__(self, output_dir: str = "visualization_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 設定顏色主題
        self.color_palette = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'warning': '#C73E1D',
            'info': '#7209B7',
            'light': '#E8F4FD',
            'dark': '#2C3E50'
        }

        # 設定 seaborn 樣式
        sns.set_style("whitegrid")
        sns.set_palette("husl")

    def create_comprehensive_dashboard(self,
                                     evaluation_results: Dict[str, Any],
                                     error_analysis: Dict[str, Any] = None,
                                     robustness_results: Dict[str, Any] = None) -> str:
        """
        創建綜合評估儀表板

        Args:
            evaluation_results: 評估結果
            error_analysis: 錯誤分析結果
            robustness_results: 穩健性測試結果

        Returns:
            HTML 儀表板文件路徑
        """

        logger.info("開始創建綜合評估儀表板...")

        # 創建子圖
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                '整體效能指標', '混淆矩陣', '類別精度對比',
                '信心分數分布', '錯誤類型分析', '穩健性測試結果',
                '特徵重要性', '模型校準', '時間趨勢'
            ],
            specs=[
                [{"type": "bar"}, {"type": "heatmap"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )

        # 1. 整體效能指標
        if 'metrics' in evaluation_results:
            metrics = evaluation_results['metrics']
            metric_names = ['Precision', 'Recall', 'F1-Score']
            metric_values = [
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1', 0)
            ]

            fig.add_trace(
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    name='整體指標',
                    marker_color=self.color_palette['primary']
                ),
                row=1, col=1
            )

        # 2. 混淆矩陣
        if 'confusion_matrix' in evaluation_results:
            cm = evaluation_results['confusion_matrix']
            labels = evaluation_results.get('class_names', ['none', 'toxic', 'severe'])

            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale='Blues',
                    showscale=True
                ),
                row=1, col=2
            )

        # 3. 類別精度對比
        if 'per_class_metrics' in evaluation_results:
            per_class = evaluation_results['per_class_metrics']
            classes = list(per_class.keys())
            precisions = [per_class[cls].get('precision', 0) for cls in classes]
            recalls = [per_class[cls].get('recall', 0) for cls in classes]

            fig.add_trace(
                go.Bar(x=classes, y=precisions, name='Precision',
                      marker_color=self.color_palette['success']),
                row=1, col=3
            )
            fig.add_trace(
                go.Bar(x=classes, y=recalls, name='Recall',
                      marker_color=self.color_palette['warning']),
                row=1, col=3
            )

        # 4. 信心分數分布
        if 'confidence_scores' in evaluation_results:
            confidences = evaluation_results['confidence_scores']
            fig.add_trace(
                go.Histogram(
                    x=confidences,
                    nbinsx=30,
                    name='信心分數分布',
                    marker_color=self.color_palette['info']
                ),
                row=2, col=1
            )

        # 5. 錯誤類型分析
        if error_analysis and 'statistics' in error_analysis:
            error_stats = error_analysis['statistics'].get('error_types', {}))
            error_types = list(error_stats.keys())
            error_counts = list(error_stats.values())

            fig.add_trace(
                go.Pie(
                    labels=error_types,
                    values=error_counts,
                    name="錯誤類型"
                ),
                row=2, col=2
            )

        # 6. 穩健性測試結果
        if robustness_results and 'summary_statistics' in robustness_results:
            rob_stats = robustness_results['summary_statistics']
            attack_types = list(rob_stats.keys())
            success_rates = [rob_stats[attack]['attack_success_rate'] for attack in attack_types]

            fig.add_trace(
                go.Bar(
                    x=attack_types,
                    y=success_rates,
                    name='攻擊成功率',
                    marker_color=self.color_palette['warning']
                ),
                row=2, col=3
            )

        # 更新布局
        fig.update_layout(
            height=1200,
            title_text="CyberPuppy 霸凌偵測系統 - 綜合評估儀表板",
            title_x=0.5,
            title_font_size=20,
            showlegend=True
        )

        # 保存為 HTML
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_dashboard_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)

        fig.write_html(filepath)
        logger.info(f"綜合儀表板已保存至: {filepath}")

        return filepath

    def plot_confusion_matrix(self,
                             y_true: List[str],
                             y_pred: List[str],
                             class_names: List[str] = None,
                             save_path: str = None) -> str:
        """繪製混淆矩陣"""

        if class_names is None:
            class_names = ['none', 'toxic', 'severe']

        # 計算混淆矩陣
        cm = confusion_matrix(y_true, y_pred, labels=class_names)

        # 創建熱力圖
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)

        plt.title('混淆矩陣 (Confusion Matrix)', fontsize=16, fontweight='bold')
        plt.xlabel('預測標籤 (Predicted Label)', fontsize=12)
        plt.ylabel('真實標籤 (True Label)', fontsize=12)

        # 添加準確率信息
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.02, 0.02, f'整體準確率: {accuracy:.3f}', fontsize=10)

        plt.tight_layout()

        # 保存
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"confusion_matrix_{timestamp}.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"混淆矩陣已保存至: {save_path}")
        return save_path

    def plot_metrics_comparison(self,
                              metrics_dict: Dict[str, Dict[str, float]],
                              save_path: str = None) -> str:
        """繪製指標對比圖"""

        # 準備數據
        classes = list(metrics_dict.keys())
        metrics = ['precision', 'recall', 'f1-score', 'support']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('類別效能指標對比', fontsize=16, fontweight='bold')

        colors = [self.color_palette['primary'], self.color_palette['secondary'], self.color_palette['success']]

        for idx, metric in enumerate(metrics[:3]):  # 只顯示前三個指標
            ax = axes[idx // 2, idx % 2]
            values = [metrics_dict[cls].get(metric, 0) for cls in classes]

            bars = ax.bar(classes, values, color=colors[idx % len(colors)], alpha=0.8)
            ax.set_title(f'{metric.upper()}', fontweight='bold')
            ax.set_ylabel('分數')
            ax.set_ylim(0, 1)

            # 添加數值標籤
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')

        # 第四個子圖：支持度（樣本數量）
        ax = axes[1, 1]
        support_values = [metrics_dict[cls].get('support', 0) for cls in classes]
        bars = ax.bar(classes, support_values, color=self.color_palette['info'], alpha=0.8)
        ax.set_title('SUPPORT (樣本數量)', fontweight='bold')
        ax.set_ylabel('樣本數')

        for bar, value in zip(bars, support_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(support_values) * 0.01,
                   f'{int(value)}', ha='center', va='bottom')

        plt.tight_layout()

        # 保存
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"metrics_comparison_{timestamp}.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"指標對比圖已保存至: {save_path}")
        return save_path

    def plot_confidence_distribution(self,
                                   confidence_scores: List[float],
                                   predictions: List[str],
                                   true_labels: List[str],
                                   save_path: str = None) -> str:
        """繪製信心分數分布圖"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('信心分數分布分析', fontsize=16, fontweight='bold')

        # 1. 整體信心分數分布
        ax1 = axes[0, 0]
        ax1.hist(confidence_scores, bins=30, alpha=0.7, color=self.color_palette['primary'])
        ax1.set_title('整體信心分數分布')
        ax1.set_xlabel('信心分數')
        ax1.set_ylabel('頻率')
        ax1.axvline(np.mean(confidence_scores), color='red', linestyle='--',
                   label=f'平均值: {np.mean(confidence_scores):.3f}')
        ax1.legend()

        # 2. 按預測類別分組的信心分數
        ax2 = axes[0, 1]
        for class_name in set(predictions):
            class_confidences = [conf for pred, conf in zip(predictions, confidence_scores)
                               if pred == class_name]
            if class_confidences:
                ax2.hist(class_confidences, alpha=0.6, label=f'{class_name}', bins=20)

        ax2.set_title('按預測類別的信心分數分布')
        ax2.set_xlabel('信心分數')
        ax2.set_ylabel('頻率')
        ax2.legend()

        # 3. 正確vs錯誤預測的信心分數
        ax3 = axes[1, 0]
        correct_confidences = [conf for pred, true, conf in zip(predictions, true_labels, confidence_scores)
                             if pred == true]
        incorrect_confidences = [conf for pred, true, conf in zip(predictions, true_labels, confidence_scores)
                               if pred != true]

        if correct_confidences:
            ax3.hist(correct_confidences, alpha=0.7, label='正確預測', color='green', bins=20)
        if incorrect_confidences:
            ax3.hist(incorrect_confidences, alpha=0.7, label='錯誤預測', color='red', bins=20)

        ax3.set_title('正確vs錯誤預測的信心分數')
        ax3.set_xlabel('信心分數')
        ax3.set_ylabel('頻率')
        ax3.legend()

        # 4. 信心分數 vs 準確率
        ax4 = axes[1, 1]

        # 將信心分數分成十個區間
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        accuracies = []

        for i in range(len(bins) - 1):
            mask = (np.array(confidence_scores) >= bins[i]) & (np.array(confidence_scores) < bins[i + 1])
            if np.sum(mask) > 0:
                bin_predictions = np.array(predictions)[mask]
                bin_true_labels = np.array(true_labels)[mask]
                accuracy = np.mean(bin_predictions == bin_true_labels)
                accuracies.append(accuracy)
            else:
                accuracies.append(0)

        ax4.plot(bin_centers, accuracies, 'o-', color=self.color_palette['secondary'], linewidth=2)
        ax4.plot([0, 1], [0, 1], '--', color='gray', label='完美校準')
        ax4.set_title('模型校準曲線')
        ax4.set_xlabel('信心分數')
        ax4.set_ylabel('準確率')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"confidence_distribution_{timestamp}.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"信心分數分布圖已保存至: {save_path}")
        return save_path

    def plot_error_analysis(self,
                           error_analysis_results: Dict[str, Any],
                           save_path: str = None) -> str:
        """繪製錯誤分析圖表"""

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('錯誤分析報告', fontsize=16, fontweight='bold')

        # 1. 錯誤類型分布
        if 'statistics' in error_analysis_results:
            stats = error_analysis_results['statistics']
            error_types = stats.get('error_types', {})

            ax1 = axes[0, 0]
            if error_types:
                labels = list(error_types.keys())
                sizes = list(error_types.values())
                colors = [self.color_palette['primary'], self.color_palette['secondary'], self.color_palette['warning']]

                wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                  colors=colors[:len(labels)])
                ax1.set_title('錯誤類型分布')

        # 2. 困難程度分布
        if 'statistics' in error_analysis_results:
            difficulty_dist = stats.get('difficulty_distribution', {})

            ax2 = axes[0, 1]
            if difficulty_dist:
                difficulties = list(difficulty_dist.keys())
                counts = list(difficulty_dist.values())

                bars = ax2.bar(difficulties, counts, color=[self.color_palette['success'],
                                                          self.color_palette['info'],
                                                          self.color_palette['warning']])
                ax2.set_title('困難程度分布')
                ax2.set_ylabel('案例數量')

                # 添加數值標籤
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                           f'{count}', ha='center', va='bottom')

        # 3. 信心分數分析
        if 'confidence_analysis' in error_analysis_results:
            conf_analysis = error_analysis_results['confidence_analysis']

            ax3 = axes[0, 2]
            conf_groups = ['high_confidence_errors', 'medium_confidence_errors', 'low_confidence_errors']
            conf_counts = [conf_analysis.get(group, 0) for group in conf_groups]
            conf_labels = ['高信心錯誤', '中信心錯誤', '低信心錯誤']

            bars = ax3.bar(conf_labels, conf_counts,
                         color=[self.color_palette['warning'], self.color_palette['info'], self.color_palette['success']])
            ax3.set_title('按信心分數的錯誤分布')
            ax3.set_ylabel('錯誤數量')
            plt.setp(ax3.get_xticklabels(), rotation=45)

        # 4. 文本長度分析
        if 'text_analysis' in error_analysis_results:
            text_analysis = error_analysis_results['text_analysis']
            length_stats = text_analysis.get('length_statistics', {})

            ax4 = axes[1, 0]
            length_metrics = ['mean_length', 'median_length', 'min_length', 'max_length']
            length_values = [length_stats.get(metric, 0) for metric in length_metrics]
            length_labels = ['平均長度', '中位數長度', '最短長度', '最長長度']

            bars = ax4.bar(length_labels, length_values, color=self.color_palette['primary'])
            ax4.set_title('錯誤案例文本長度統計')
            ax4.set_ylabel('字符數')
            plt.setp(ax4.get_xticklabels(), rotation=45)

        # 5. 關鍵詞統計
        if 'text_analysis' in error_analysis_results:
            keyword_stats = text_analysis.get('keyword_statistics', {})

            ax5 = axes[1, 1]
            keyword_metrics = ['mean_keywords', 'median_keywords', 'max_keywords', 'cases_with_keywords']
            keyword_values = [keyword_stats.get(metric, 0) for metric in keyword_metrics]
            keyword_labels = ['平均關鍵詞', '中位數關鍵詞', '最多關鍵詞', '含關鍵詞案例']

            bars = ax5.bar(keyword_labels, keyword_values, color=self.color_palette['secondary'])
            ax5.set_title('錯誤案例關鍵詞統計')
            ax5.set_ylabel('數量')
            plt.setp(ax5.get_xticklabels(), rotation=45)

        # 6. 改進建議重要性
        if 'improvement_suggestions' in error_analysis_results:
            suggestions = error_analysis_results['improvement_suggestions']

            ax6 = axes[1, 2]
            # 簡化顯示前5個建議
            if suggestions:
                suggestion_text = "改進建議:\n"
                for i, suggestion in enumerate(suggestions[:5], 1):
                    suggestion_text += f"{i}. {suggestion[:30]}...\n"

                ax6.text(0.05, 0.95, suggestion_text, transform=ax6.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=self.color_palette['light']))
                ax6.set_title('改進建議')
                ax6.axis('off')

        plt.tight_layout()

        # 保存
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"error_analysis_{timestamp}.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"錯誤分析圖表已保存至: {save_path}")
        return save_path

    def plot_robustness_results(self,
                              robustness_results: Dict[str, Any],
                              save_path: str = None) -> str:
        """繪製穩健性測試結果"""

        if 'summary_statistics' not in robustness_results:
            logger.warning("穩健性測試結果中缺少摘要統計")
            return ""

        summary_stats = robustness_results['summary_statistics']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('穩健性測試結果', fontsize=16, fontweight='bold')

        # 1. 攻擊成功率對比
        ax1 = axes[0, 0]
        attack_types = list(summary_stats.keys())
        success_rates = [summary_stats[attack]['attack_success_rate'] for attack in attack_types]

        bars = ax1.barh(attack_types, success_rates, color=self.color_palette['warning'])
        ax1.set_title('各攻擊類型成功率')
        ax1.set_xlabel('成功率')
        ax1.set_xlim(0, 1)

        # 添加數值標籤
        for bar, rate in zip(bars, success_rates):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{rate:.3f}', ha='left', va='center')

        # 2. 平均信心下降
        ax2 = axes[0, 1]
        conf_drops = [summary_stats[attack]['average_confidence_drop'] for attack in attack_types]

        bars = ax2.bar(attack_types, conf_drops, color=self.color_palette['info'])
        ax2.set_title('平均信心分數下降')
        ax2.set_ylabel('信心分數下降')
        plt.setp(ax2.get_xticklabels(), rotation=45)

        # 3. 穩健性雷達圖
        ax3 = axes[1, 0]

        # 轉換為穩健性分數（1 - 成功率）
        robustness_scores = [1 - rate for rate in success_rates]

        # 創建雷達圖
        angles = np.linspace(0, 2 * np.pi, len(attack_types), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # 閉合圖形
        robustness_scores_closed = robustness_scores + [robustness_scores[0]]

        ax3 = plt.subplot(223, projection='polar')
        ax3.plot(angles, robustness_scores_closed, 'o-', linewidth=2, color=self.color_palette['primary'])
        ax3.fill(angles, robustness_scores_closed, alpha=0.25, color=self.color_palette['primary'])
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels([attack_type.replace('_', '\n') for attack_type in attack_types], fontsize=8)
        ax3.set_ylim(0, 1)
        ax3.set_title('穩健性雷達圖', y=1.08)

        # 4. 總體穩健性評分
        ax4 = axes[1, 1]

        if 'overall_statistics' in robustness_results:
            overall_stats = robustness_results['overall_statistics']
            robustness_score = overall_stats.get('overall_robustness_score', 0)
            robustness_level = overall_stats.get('robustness_level', 'unknown')

            # 創建儀表盤樣式的圖
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)

            # 背景弧線
            ax4.plot(theta, r, color='lightgray', linewidth=10)

            # 分數弧線
            score_theta = theta[:int(robustness_score * 100)]
            if robustness_score >= 0.8:
                color = 'green'
            elif robustness_score >= 0.6:
                color = 'orange'
            else:
                color = 'red'

            if len(score_theta) > 0:
                ax4.plot(score_theta, r[:len(score_theta)], color=color, linewidth=10)

            # 添加分數文字
            ax4.text(np.pi/2, 0.5, f'{robustness_score:.3f}',
                    ha='center', va='center', fontsize=20, fontweight='bold')
            ax4.text(np.pi/2, 0.2, f'穩健性: {robustness_level.upper()}',
                    ha='center', va='center', fontsize=12)

            ax4.set_ylim(0, 1.2)
            ax4.set_xlim(0, np.pi)
            ax4.axis('off')
            ax4.set_title('總體穩健性評分')

        plt.tight_layout()

        # 保存
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.output_dir, f"robustness_results_{timestamp}.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"穩健性測試圖表已保存至: {save_path}")
        return save_path

    def create_interactive_report(self,
                                evaluation_results: Dict[str, Any],
                                error_analysis: Dict[str, Any] = None,
                                robustness_results: Dict[str, Any] = None,
                                explanations: List[Dict[str, Any]] = None) -> str:
        """創建互動式 HTML 報告"""

        logger.info("創建互動式評估報告...")

        # HTML 模板
        html_template = """
        <!DOCTYPE html>
        <html lang="zh-TW">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>CyberPuppy 霸凌偵測系統 - 評估報告</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Arial', 'Microsoft JhengHei', sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                    padding: 20px;
                    background: linear-gradient(135deg, #2E86AB, #A23B72);
                    color: white;
                    border-radius: 10px;
                }}
                .section {{
                    margin: 30px 0;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #2E86AB;
                }}
                .metric-label {{
                    color: #666;
                    margin-top: 5px;
                }}
                .plot-container {{
                    width: 100%;
                    height: 500px;
                    margin: 20px 0;
                }}
                .tabs {{
                    display: flex;
                    background-color: #f1f1f1;
                    border-radius: 8px 8px 0 0;
                }}
                .tab {{
                    padding: 10px 20px;
                    cursor: pointer;
                    border: none;
                    background-color: transparent;
                    flex: 1;
                    text-align: center;
                }}
                .tab.active {{
                    background-color: #2E86AB;
                    color: white;
                }}
                .tab-content {{
                    display: none;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-top: none;
                    border-radius: 0 0 8px 8px;
                }}
                .tab-content.active {{
                    display: block;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>CyberPuppy 霸凌偵測系統</h1>
                    <h2>評估報告</h2>
                    <p>生成時間: {timestamp}</p>
                </div>

                {content}

                <script>
                    function showTab(tabName) {{
                        // 隱藏所有標籤內容
                        var contents = document.getElementsByClassName('tab-content');
                        for (var i = 0; i < contents.length; i++) {{
                            contents[i].classList.remove('active');
                        }}

                        // 移除所有標籤的 active 類
                        var tabs = document.getElementsByClassName('tab');
                        for (var i = 0; i < tabs.length; i++) {{
                            tabs[i].classList.remove('active');
                        }}

                        // 顯示選中的標籤內容
                        document.getElementById(tabName).classList.add('active');
                        event.target.classList.add('active');
                    }}

                    // 默認顯示第一個標籤
                    document.addEventListener('DOMContentLoaded', function() {{
                        var firstTab = document.querySelector('.tab');
                        if (firstTab) {{
                            firstTab.click();
                        }}
                    }});
                </script>
            </div>
        </body>
        </html>
        """

        # 生成內容
        content_sections = []

        # 1. 總體指標卡片
        if 'metrics' in evaluation_results:
            metrics = evaluation_results['metrics']
            metrics_html = f"""
            <div class="section">
                <h2>總體效能指標</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('precision', 0):.3f}</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('recall', 0):.3f}</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('f1', 0):.3f}</div>
                        <div class="metric-label">F1-Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics.get('accuracy', 0):.3f}</div>
                        <div class="metric-label">Accuracy</div>
                    </div>
                </div>
            </div>
            """
            content_sections.append(metrics_html)

        # 2. 標籤頁內容
        tabs_html = """
        <div class="section">
            <div class="tabs">
                <button class="tab" onclick="showTab('performance')">效能分析</button>
                <button class="tab" onclick="showTab('errors')">錯誤分析</button>
                <button class="tab" onclick="showTab('robustness')">穩健性測試</button>
                <button class="tab" onclick="showTab('explanations')">可解釋性</button>
            </div>

            <div id="performance" class="tab-content">
                <h3>效能分析</h3>
                <div id="performance-plot" class="plot-container"></div>
            </div>

            <div id="errors" class="tab-content">
                <h3>錯誤分析</h3>
                <div id="error-plot" class="plot-container"></div>
            </div>

            <div id="robustness" class="tab-content">
                <h3>穩健性測試</h3>
                <div id="robustness-plot" class="plot-container"></div>
            </div>

            <div id="explanations" class="tab-content">
                <h3>可解釋性分析</h3>
                <div id="explanation-plot" class="plot-container"></div>
            </div>
        </div>
        """

        content_sections.append(tabs_html)

        # 組合所有內容
        full_content = '\n'.join(content_sections)

        # 填入模板
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_content = html_template.format(
            timestamp=timestamp,
            content=full_content
        )

        # 保存 HTML 文件
        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interactive_report_{timestamp_file}.html"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"互動式報告已保存至: {filepath}")
        return filepath

class ConfusionMatrixPlotter:
    """混淆矩陣專門繪製器"""

    @staticmethod
    def plot_normalized_confusion_matrix(y_true, y_pred, class_names, save_path=None):
        """繪製標準化混淆矩陣"""

        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 原始混淆矩陣
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title('原始混淆矩陣')
        ax1.set_xlabel('預測標籤')
        ax1.set_ylabel('真實標籤')

        # 標準化混淆矩陣
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title('標準化混淆矩陣')
        ax2.set_xlabel('預測標籤')
        ax2.set_ylabel('真實標籤')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        return save_path

class AttentionVisualizer:
    """注意力權重視覺化器"""

    @staticmethod
    def plot_attention_heatmap(tokens, attention_weights, save_path=None):
        """繪製注意力權重熱力圖"""

        plt.figure(figsize=(12, 8))

        # 限制顯示的 token 數量
        max_tokens = 20
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            attention_weights = attention_weights[:max_tokens, :max_tokens]

        sns.heatmap(attention_weights,
                   xticklabels=tokens,
                   yticklabels=tokens,
                   cmap='YlOrRd')

        plt.title('注意力權重熱力圖')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        return save_path