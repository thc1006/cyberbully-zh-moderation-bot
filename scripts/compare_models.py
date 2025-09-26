#!/usr/bin/env python3
"""
CyberPuppy Model Comparison Script
=================================

Compares training results across all configurations and generates HTML report.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class ModelComparator:
    """Compare training results and generate comprehensive reports."""

    def __init__(self, session_id: str, project_root: Optional[str] = None):
        self.session_id = session_id
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        self.results = {}
        self.comparison_data = []

    def extract_model_metrics(self, model_dir: Path) -> Dict:
        """Extract metrics from a trained model directory."""
        metrics = {
            'model_path': str(model_dir),
            'config_name': model_dir.name.split('_')[0],
            'training_time': None,
            'f1_macro': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'accuracy': 0.0,
            'toxicity_f1': 0.0,
            'bullying_f1': 0.0,
            'emotion_f1': 0.0,
            'model_size_mb': 0.0,
            'training_loss': [],
            'validation_loss': [],
            'epochs_completed': 0,
            'best_epoch': 0,
            'converged': False,
            'error_analysis': {}
        }

        try:
            # Check for metrics.json
            metrics_file = model_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    saved_metrics = json.load(f)
                metrics.update(saved_metrics)

            # Check for training history
            history_file = model_dir / "training_history.json"
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                metrics['training_loss'] = history.get('train_loss', [])
                metrics['validation_loss'] = history.get('val_loss', [])
                metrics['epochs_completed'] = len(metrics['training_loss'])

                # Find best epoch
                if metrics['validation_loss']:
                    best_epoch = np.argmin(metrics['validation_loss'])
                    metrics['best_epoch'] = best_epoch
                    metrics['converged'] = len(metrics['validation_loss']) > 5  # Basic convergence check

            # Calculate model size
            model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
            if model_files:
                total_size = sum(f.stat().st_size for f in model_files)
                metrics['model_size_mb'] = total_size / (1024 * 1024)

            # Extract training time from logs
            log_file = self.logs_dir / f"training_{self.session_id}.log"
            if log_file.exists():
                training_time = self._extract_training_time(log_file, metrics['config_name'])
                metrics['training_time'] = training_time

        except Exception as e:
            print(f"Warning: Could not extract all metrics for {model_dir}: {e}")

        return metrics

    def _extract_training_time(self, log_file: Path, config_name: str) -> Optional[str]:
        """Extract training time for a specific configuration from logs."""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find training start and end for this config
            start_pattern = f"Training Configuration.*{config_name}"
            end_pattern = f"Completed training {config_name}"

            # Simple extraction - could be improved with regex
            if start_pattern.lower() in content.lower() and end_pattern.lower() in content.lower():
                return "Extracted from logs"  # Placeholder - implement actual time extraction

        except Exception:
            pass

        return None

    def analyze_all_models(self) -> List[Dict]:
        """Analyze all models from the current training session."""
        model_dirs = []

        # Find all model directories from this session
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and self.session_id in model_dir.name:
                model_dirs.append(model_dir)

        if not model_dirs:
            print(f"No model directories found for session {self.session_id}")
            return []

        print(f"Found {len(model_dirs)} models to analyze:")
        for dir_path in model_dirs:
            print(f"  - {dir_path.name}")

        # Extract metrics for each model
        for model_dir in model_dirs:
            config_name = model_dir.name.split('_')[0]
            metrics = self.extract_model_metrics(model_dir)
            self.results[config_name] = metrics
            self.comparison_data.append(metrics)

        return self.comparison_data

    def generate_comparison_plots(self, output_dir: Path) -> List[str]:
        """Generate comparison plots and return list of plot files."""
        plot_files = []

        if not self.comparison_data:
            return plot_files

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. F1 Score Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        configs = [d['config_name'] for d in self.comparison_data]
        f1_scores = [d['f1_macro'] for d in self.comparison_data]

        bars = ax.bar(configs, f1_scores)
        ax.set_title('Model F1 Score Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('F1 Score (Macro)')
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        # Add target line
        ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.7, label='Target (0.75)')
        ax.legend()

        plot_file = output_dir / "f1_comparison.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_file.name)

        # 2. Multi-metric Comparison Radar Chart
        if len(self.comparison_data) > 0:
            metrics_to_plot = ['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

            angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # Complete the circle

            for i, data in enumerate(self.comparison_data):
                values = [data.get(metric, 0) for metric in metrics_to_plot]
                values = np.concatenate((values, [values[0]]))  # Complete the circle

                ax.plot(angles, values, 'o-', linewidth=2, label=data['config_name'])
                ax.fill(angles, values, alpha=0.25)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
            ax.set_ylim(0, 1)
            ax.set_title('Multi-Metric Comparison', fontsize=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

            plot_file = output_dir / "radar_comparison.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(plot_file.name)

        # 3. Training History (if available)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        for data in self.comparison_data:
            if data['training_loss'] and data['validation_loss']:
                epochs = range(1, len(data['training_loss']) + 1)
                ax1.plot(epochs, data['training_loss'], label=f"{data['config_name']} (Train)")
                ax2.plot(epochs, data['validation_loss'], label=f"{data['config_name']} (Val)")

        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plot_file = output_dir / "training_history.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_file.name)

        return plot_files

    def generate_html_report(self, best_config: str = None, best_f1: float = None) -> str:
        """Generate comprehensive HTML comparison report."""
        if not self.comparison_data:
            print("No data available for report generation")
            return ""

        # Create output directory
        output_dir = self.models_dir / f"comparison_{self.session_id}"
        output_dir.mkdir(exist_ok=True)

        # Generate plots
        plot_files = self.generate_comparison_plots(output_dir)

        # Determine best model if not provided
        if best_config is None or best_f1 is None:
            best_model = max(self.comparison_data, key=lambda x: x['f1_macro'])
            best_config = best_model['config_name']
            best_f1 = best_model['f1_macro']

        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CyberPuppy 模型比較報告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 3px solid #3498db;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 2.5em;
        }}
        .subtitle {{
            color: #7f8c8d;
            font-size: 1.2em;
            margin-top: 10px;
        }}
        .summary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .summary h2 {{
            margin-top: 0;
            font-size: 1.8em;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            display: block;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.8;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .comparison-table th {{
            background: #34495e;
            color: white;
            padding: 15px;
            text-align: left;
        }}
        .comparison-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }}
        .comparison-table tr:hover {{
            background-color: #f8f9fa;
        }}
        .best-model {{
            background-color: #d4edda !important;
            font-weight: bold;
        }}
        .plots-section {{
            margin: 30px 0;
        }}
        .plot-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .status-badge {{
            padding: 5px 10px;
            border-radius: 15px;
            color: white;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .status-success {{ background-color: #27ae60; }}
        .status-warning {{ background-color: #f39c12; }}
        .status-error {{ background-color: #e74c3c; }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            color: #7f8c8d;
        }}
        .highlight {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #ff9a9e 0%, #fad0c4 100%);
            transition: width 0.3s ease;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐶 CyberPuppy 模型訓練報告</h1>
            <div class="subtitle">自動化訓練流水線結果 - Session {self.session_id}</div>
            <div class="subtitle">生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>

        <div class="summary">
            <h2>📊 訓練摘要</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <span class="metric-value">{len(self.comparison_data)}</span>
                    <span class="metric-label">訓練配置</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value">{best_config}</span>
                    <span class="metric-label">最佳模型</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value">{best_f1:.3f}</span>
                    <span class="metric-label">最高 F1 分數</span>
                </div>
                <div class="metric-card">
                    <span class="metric-value">{'✅' if best_f1 >= 0.75 else '❌'}</span>
                    <span class="metric-label">達標狀態 (≥0.75)</span>
                </div>
            </div>

            {'<div class="highlight">🎯 <strong>目標達成！</strong> 最佳模型超過 F1 = 0.75 閾值</div>' if best_f1 >= 0.75 else '<div class="highlight">⚠️ <strong>未達目標</strong> 最佳模型未達 F1 = 0.75 閾值，建議調整超參數</div>'}
        </div>

        <h2>📈 模型比較詳情</h2>
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>配置</th>
                    <th>F1 分數</th>
                    <th>精確率</th>
                    <th>召回率</th>
                    <th>準確率</th>
                    <th>模型大小 (MB)</th>
                    <th>訓練輪數</th>
                    <th>收斂狀態</th>
                </tr>
            </thead>
            <tbody>
"""

        # Add table rows for each model
        for data in sorted(self.comparison_data, key=lambda x: x['f1_macro'], reverse=True):
            is_best = data['config_name'] == best_config
            row_class = 'best-model' if is_best else ''

            convergence_status = '✅ 收斂' if data['converged'] else '⚠️ 未收斂'
            status_class = 'status-success' if data['converged'] else 'status-warning'

            html_content += f"""
                <tr class="{row_class}">
                    <td><strong>{data['config_name']}</strong> {'🏆' if is_best else ''}</td>
                    <td>{data['f1_macro']:.4f}</td>
                    <td>{data['precision_macro']:.4f}</td>
                    <td>{data['recall_macro']:.4f}</td>
                    <td>{data['accuracy']:.4f}</td>
                    <td>{data['model_size_mb']:.1f}</td>
                    <td>{data['epochs_completed']}</td>
                    <td><span class="status-badge {status_class}">{convergence_status}</span></td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>

        <div class="plots-section">
            <h2>📊 視覺化比較</h2>
"""

        # Add plot images
        for plot_file in plot_files:
            plot_name = plot_file.replace('.png', '').replace('_', ' ').title()
            html_content += f"""
            <div class="plot-container">
                <h3>{plot_name}</h3>
                <img src="{plot_file}" alt="{plot_name}">
            </div>
"""

        html_content += f"""
        </div>

        <h2>🔧 詳細分析</h2>
        <div class="highlight">
            <h3>模型表現分析：</h3>
            <ul>
"""

        # Add analysis for each model
        for data in self.comparison_data:
            f1_score = data['f1_macro']
            performance_level = "優秀" if f1_score >= 0.8 else "良好" if f1_score >= 0.7 else "需改進" if f1_score >= 0.6 else "差"

            html_content += f"""
                <li><strong>{data['config_name']}</strong>: F1 = {f1_score:.4f} ({performance_level})
                    - 完成 {data['epochs_completed']} 個訓練輪數
                    {'- 已收斂' if data['converged'] else '- 未完全收斂，可能需要更多訓練輪數'}
                </li>
"""

        html_content += f"""
            </ul>

            <h3>建議：</h3>
            <ul>
                {'<li>✅ 最佳模型已達標，可以部署使用</li>' if best_f1 >= 0.75 else '<li>❌ 建議調整學習率、批次大小或模型架構以提升效果</li>'}
                <li>🔄 可考慮使用最佳模型 ({best_config}) 進行進一步的超參數優化</li>
                <li>📊 建議在更大的測試集上驗證模型泛化能力</li>
                <li>🎯 如需更高準確率，可考慮集成學習方法</li>
            </ul>
        </div>

        <div class="footer">
            <p>🤖 由 CyberPuppy 自動化訓練系統生成</p>
            <p>報告檔案位置: models/comparison_report_{self.session_id}.html</p>
        </div>
    </div>
</body>
</html>
"""

        # Save HTML report
        report_file = self.models_dir / f"comparison_report_{self.session_id}.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📊 Comparison report generated: {report_file}")
        return str(report_file)


def main():
    """Main comparison script entry point."""
    parser = argparse.ArgumentParser(description='Compare CyberPuppy model training results')
    parser.add_argument('--session-id', required=True, help='Training session ID')
    parser.add_argument('--best-config', help='Best configuration name')
    parser.add_argument('--best-f1', type=float, help='Best F1 score')
    parser.add_argument('--project-root', help='Project root directory')

    args = parser.parse_args()

    try:
        # Initialize comparator
        comparator = ModelComparator(args.session_id, args.project_root)

        # Analyze all models
        print(f"🔍 Analyzing models for session: {args.session_id}")
        comparison_data = comparator.analyze_all_models()

        if not comparison_data:
            print("❌ No models found for comparison")
            sys.exit(1)

        print(f"✅ Found {len(comparison_data)} models to compare")

        # Generate HTML report
        report_file = comparator.generate_html_report(args.best_config, args.best_f1)

        if report_file:
            print(f"📊 Report generated successfully: {report_file}")

            # Save comparison data as JSON for further analysis
            json_file = comparator.models_dir / f"comparison_data_{args.session_id}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Raw data saved: {json_file}")

        else:
            print("❌ Failed to generate report")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()