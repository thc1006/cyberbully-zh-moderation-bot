#!/usr/bin/env python3
"""
Comprehensive evaluation script for cyberbullying detection model.
Provides detailed metrics, error analysis, and performance visualization.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
    average_precision_score, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.cyberpuppy.config import Config
from src.cyberpuppy.eval.metrics import MetricsCalculator as DetailedMetrics
from src.cyberpuppy.eval.error_analysis import ErrorAnalyzer
from src.cyberpuppy.eval.robustness import RobustnessTestSuite as RobustnessEvaluator
try:
    from src.cyberpuppy.fairness.bias_detection import BiasEvaluator
except ImportError:
    BiasEvaluator = None
from src.cyberpuppy.eval.reports import ReportGenerator

logger = logging.getLogger(__name__)


class BullyingDetectionEvaluator:
    """Comprehensive evaluator for cyberbullying detection models."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize components
        self.metrics_calculator = DetailedMetrics()
        self.error_analyzer = ErrorAnalyzer()
        self.robustness_evaluator = RobustnessEvaluator()
        self.bias_evaluator = BiasEvaluator()
        self.report_generator = ReportGenerator()

        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the trained model and tokenizer."""
        try:
            model_path = self.config.MODEL_SAVE_PATH
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def load_test_data(self, test_file: str) -> Tuple[List[str], List[Dict]]:
        """Load test dataset with texts and multi-label annotations."""
        data = pd.read_json(test_file, lines=True)

        texts = data['text'].tolist()
        labels = []

        for _, row in data.iterrows():
            label_dict = {
                'toxicity': row.get('toxicity', 'none'),
                'bullying': row.get('bullying', 'none'),
                'role': row.get('role', 'none'),
                'emotion': row.get('emotion', 'neu'),
                'emotion_strength': row.get('emotion_strength', 0)
            }
            labels.append(label_dict)

        return texts, labels

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Generate predictions for a batch of texts."""
        predictions = []

        with torch.no_grad():
            for text in texts:
                # Tokenize input
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)

                # Get predictions
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)

                # Extract predictions for each task
                # Note: This assumes a multi-task model architecture
                pred_dict = self._extract_predictions(probs)
                predictions.append(pred_dict)

        return predictions

    def _extract_predictions(self, probs: torch.Tensor) -> Dict:
        """Extract task-specific predictions from model output."""
        # This is a placeholder - adjust based on actual model architecture
        prob_array = probs.cpu().numpy()[0]

        # Assume the model outputs concatenated logits for all tasks
        toxicity_idx = np.argmax(prob_array[:3])  # none, toxic, severe
        bullying_idx = np.argmax(prob_array[3:6])  # none, harassment, threat
        role_idx = np.argmax(prob_array[6:10])  # none, perpetrator, victim, bystander
        emotion_idx = np.argmax(prob_array[10:13])  # pos, neu, neg
        emotion_strength = int(np.round(prob_array[13] * 4))  # 0-4 scale

        toxicity_labels = ['none', 'toxic', 'severe']
        bullying_labels = ['none', 'harassment', 'threat']
        role_labels = ['none', 'perpetrator', 'victim', 'bystander']
        emotion_labels = ['pos', 'neu', 'neg']

        return {
            'toxicity': toxicity_labels[toxicity_idx],
            'bullying': bullying_labels[bullying_idx],
            'role': role_labels[role_idx],
            'emotion': emotion_labels[emotion_idx],
            'emotion_strength': emotion_strength,
            'confidence': {
                'toxicity': float(np.max(prob_array[:3])),
                'bullying': float(np.max(prob_array[3:6])),
                'role': float(np.max(prob_array[6:10])),
                'emotion': float(np.max(prob_array[10:13]))
            }
        }

    def evaluate_comprehensive(
        self,
        texts: List[str],
        true_labels: List[Dict],
        output_dir: str = "evaluation_results"
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation including all metrics and analyses."""

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate predictions
        logger.info("Generating predictions...")
        predictions = self.predict_batch(texts)

        # Calculate basic metrics
        logger.info("Calculating performance metrics...")
        basic_metrics = self.metrics_calculator.calculate_all_metrics(
            true_labels, predictions
        )

        # Error analysis
        logger.info("Performing error analysis...")
        error_analysis = self.error_analyzer.analyze_errors(
            texts, true_labels, predictions
        )

        # Robustness testing
        logger.info("Running robustness tests...")
        robustness_results = self.robustness_evaluator.evaluate_robustness(
            texts, true_labels, self.predict_batch
        )

        # Bias evaluation
        logger.info("Evaluating fairness and bias...")
        bias_results = self.bias_evaluator.evaluate_bias(
            texts, true_labels, predictions
        )

        # Compile results
        evaluation_results = {
            'basic_metrics': basic_metrics,
            'error_analysis': error_analysis,
            'robustness': robustness_results,
            'bias_evaluation': bias_results,
            'metadata': {
                'num_samples': len(texts),
                'model_path': self.config.MODEL_SAVE_PATH,
                'evaluation_date': pd.Timestamp.now().isoformat()
            }
        }

        # Save results
        results_file = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

        # Generate visualizations
        logger.info("Generating visualizations...")
        self._generate_visualizations(evaluation_results, output_dir)

        # Generate comprehensive report
        logger.info("Generating comprehensive report...")
        self.report_generator.generate_report(
            evaluation_results,
            output_path=os.path.join(output_dir, 'evaluation_report.html')
        )

        logger.info(f"Evaluation completed. Results saved to {output_dir}")
        return evaluation_results

    def _generate_visualizations(self, results: Dict, output_dir: str):
        """Generate visualization plots for evaluation results."""

        # Confusion matrices for each task
        self._plot_confusion_matrices(results['basic_metrics'], output_dir)

        # Performance comparison plots
        self._plot_performance_comparison(results['basic_metrics'], output_dir)

        # Error distribution plots
        self._plot_error_distribution(results['error_analysis'], output_dir)

        # Robustness visualization
        self._plot_robustness_results(results['robustness'], output_dir)

        # Bias visualization
        self._plot_bias_results(results['bias_evaluation'], output_dir)

    def _plot_confusion_matrices(self, metrics: Dict, output_dir: str):
        """Plot confusion matrices for each classification task."""

        tasks = ['toxicity', 'bullying', 'role', 'emotion']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i, task in enumerate(tasks):
            if task in metrics and 'confusion_matrix' in metrics[task]:
                cm = np.array(metrics[task]['confusion_matrix'])
                labels = metrics[task]['labels']

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels,
                    ax=axes[i]
                )
                axes[i].set_title(f'{task.capitalize()} Confusion Matrix')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_comparison(self, metrics: Dict, output_dir: str):
        """Plot performance comparison across tasks."""

        tasks = ['toxicity', 'bullying', 'role', 'emotion']
        metric_names = ['precision', 'recall', 'f1']

        # Prepare data for plotting
        plot_data = []
        for task in tasks:
            if task in metrics:
                for metric in metric_names:
                    if f'{metric}_macro' in metrics[task]:
                        plot_data.append({
                            'Task': task.capitalize(),
                            'Metric': metric.capitalize(),
                            'Score': metrics[task][f'{metric}_macro']
                        })

        if plot_data:
            df_plot = pd.DataFrame(plot_data)

            plt.figure(figsize=(12, 6))
            sns.barplot(data=df_plot, x='Task', y='Score', hue='Metric')
            plt.title('Performance Comparison Across Tasks')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.legend(title='Metric')
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'performance_comparison.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_error_distribution(self, error_analysis: Dict, output_dir: str):
        """Plot error distribution analysis."""

        if 'error_types' in error_analysis:
            error_types = error_analysis['error_types']

            plt.figure(figsize=(10, 6))
            categories = list(error_types.keys())
            counts = list(error_types.values())

            plt.bar(categories, counts, alpha=0.7)
            plt.title('Error Type Distribution')
            plt.xlabel('Error Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'error_distribution.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_robustness_results(self, robustness: Dict, output_dir: str):
        """Plot robustness test results."""

        if 'perturbation_results' in robustness:
            results = robustness['perturbation_results']

            plt.figure(figsize=(12, 6))
            perturbation_types = list(results.keys())
            performance_drops = [results[pt]['performance_drop'] for pt in perturbation_types]

            plt.bar(perturbation_types, performance_drops, alpha=0.7, color='coral')
            plt.title('Model Robustness: Performance Drop Under Perturbations')
            plt.xlabel('Perturbation Type')
            plt.ylabel('Performance Drop (%)')
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'robustness_results.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_bias_results(self, bias_evaluation: Dict, output_dir: str):
        """Plot bias evaluation results."""

        if 'fairness_metrics' in bias_evaluation:
            fairness = bias_evaluation['fairness_metrics']

            plt.figure(figsize=(10, 6))
            metrics = list(fairness.keys())
            values = list(fairness.values())

            colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in values]

            plt.bar(metrics, values, alpha=0.7, color=colors)
            plt.title('Fairness Metrics')
            plt.xlabel('Metric')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)

            # Add threshold lines
            plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Good (≥0.8)')
            plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Fair (≥0.6)')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fairness_metrics.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

    def compare_models(
        self,
        baseline_results: Dict,
        current_results: Dict,
        output_dir: str
    ):
        """Compare current model with baseline model."""

        comparison = {}

        # Compare basic metrics
        for task in ['toxicity', 'bullying', 'role', 'emotion']:
            if task in baseline_results['basic_metrics'] and task in current_results['basic_metrics']:
                baseline_f1 = baseline_results['basic_metrics'][task].get('f1_macro', 0)
                current_f1 = current_results['basic_metrics'][task].get('f1_macro', 0)

                comparison[task] = {
                    'baseline_f1': baseline_f1,
                    'current_f1': current_f1,
                    'improvement': current_f1 - baseline_f1,
                    'improvement_pct': ((current_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
                }

        # Save comparison results
        comparison_file = os.path.join(output_dir, 'model_comparison.json')
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        # Generate comparison visualization
        self._plot_model_comparison(comparison, output_dir)

        return comparison

    def _plot_model_comparison(self, comparison: Dict, output_dir: str):
        """Plot model comparison results."""

        tasks = list(comparison.keys())
        baseline_scores = [comparison[task]['baseline_f1'] for task in tasks]
        current_scores = [comparison[task]['current_f1'] for task in tasks]

        x = np.arange(len(tasks))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, baseline_scores, width, label='Baseline', alpha=0.7)
        plt.bar(x + width/2, current_scores, width, label='Current', alpha=0.7)

        plt.xlabel('Tasks')
        plt.ylabel('F1 Score (Macro)')
        plt.title('Model Performance Comparison')
        plt.xticks(x, [task.capitalize() for task in tasks])
        plt.legend()
        plt.ylim(0, 1)

        # Add improvement annotations
        for i, task in enumerate(tasks):
            improvement = comparison[task]['improvement']
            if improvement > 0:
                plt.annotate(f'+{improvement:.3f}',
                           xy=(i, current_scores[i]),
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           color='green',
                           fontweight='bold')
            elif improvement < 0:
                plt.annotate(f'{improvement:.3f}',
                           xy=(i, current_scores[i]),
                           xytext=(0, -15),
                           textcoords='offset points',
                           ha='center',
                           color='red',
                           fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate cyberbullying detection model')
    parser.add_argument('--test_file', required=True, help='Path to test data file')
    parser.add_argument('--config', default='config/model_config.yaml', help='Config file path')
    parser.add_argument('--output_dir', default='evaluation_results', help='Output directory')
    parser.add_argument('--baseline_results', help='Baseline results for comparison')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Load configuration
        config = Config.from_file(args.config)

        # Initialize evaluator
        evaluator = BullyingDetectionEvaluator(config)

        # Load test data
        texts, true_labels = evaluator.load_test_data(args.test_file)

        # Run comprehensive evaluation
        results = evaluator.evaluate_comprehensive(
            texts, true_labels, args.output_dir
        )

        # Compare with baseline if provided
        if args.baseline_results:
            with open(args.baseline_results, 'r', encoding='utf-8') as f:
                baseline_results = json.load(f)

            comparison = evaluator.compare_models(
                baseline_results, results, args.output_dir
            )

            print("\nModel Comparison Summary:")
            for task, metrics in comparison.items():
                improvement = metrics['improvement']
                improvement_pct = metrics['improvement_pct']
                print(f"{task.capitalize()}: {improvement:+.3f} F1 ({improvement_pct:+.1f}%)")

        # Print summary
        print("\nEvaluation Summary:")
        for task in ['toxicity', 'bullying', 'role', 'emotion']:
            if task in results['basic_metrics']:
                f1_macro = results['basic_metrics'][task].get('f1_macro', 0)
                print(f"{task.capitalize()} F1 (macro): {f1_macro:.3f}")

        print(f"\nDetailed results saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    main()