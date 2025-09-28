"""
Visualization tools for active learning results
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set up Chinese font support
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")


class ActiveLearningVisualizer:
    """Visualizer for active learning results and progress"""

    def __init__(self, save_dir: str = "./plots"):
        """
        Initialize visualizer

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_learning_curves(
        self,
        curve_data: Dict[str, List[float]],
        title: str = "Active Learning Progress",
        save_name: Optional[str] = None,
    ) -> str:
        """
        Plot learning curves showing F1 scores vs number of annotations

        Args:
            curve_data: Dictionary with 'annotations' and F1 score lists
            title: Plot title
            save_name: Filename to save plot

        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        annotations = curve_data["annotations"]

        # Plot 1: Overall F1 scores
        ax1.plot(
            annotations, curve_data["f1_macro"], "o-", label="F1 Macro", linewidth=2, markersize=6
        )

        ax1.set_xlabel("Number of Annotations")
        ax1.set_ylabel("F1 Score")
        ax1.set_title(f"{title} - Overall Performance")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Add target line if it exists
        if "target_f1" in curve_data:
            ax1.axhline(
                y=curve_data["target_f1"],
                color="red",
                linestyle="--",
                label=f'Target F1: {curve_data["target_f1"]:.2f}',
            )

        # Plot 2: Per-class F1 scores
        if "f1_toxic" in curve_data and "f1_severe" in curve_data:
            ax2.plot(
                annotations,
                curve_data["f1_toxic"],
                "o-",
                label="F1 Toxic",
                linewidth=2,
                markersize=4,
            )
            ax2.plot(
                annotations,
                curve_data["f1_severe"],
                "o-",
                label="F1 Severe",
                linewidth=2,
                markersize=4,
            )

            ax2.set_xlabel("Number of Annotations")
            ax2.set_ylabel("F1 Score")
            ax2.set_title(f"{title} - Per-Class Performance")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)

        plt.tight_layout()

        # Save plot
        if save_name is None:
            save_name = f"learning_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        plot_path = os.path.join(self.save_dir, save_name)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    def plot_annotation_efficiency(
        self,
        performance_history: List[Dict[str, Any]],
        title: str = "Annotation Efficiency",
        save_name: Optional[str] = None,
    ) -> str:
        """
        Plot annotation efficiency: F1 improvement per annotation

        Args:
            performance_history: List of performance records
            title: Plot title
            save_name: Filename to save plot

        Returns:
            Path to saved plot
        """
        if len(performance_history) < 2:
            raise ValueError("Need at least 2 performance records for efficiency analysis")

        annotations = [record["total_annotations"] for record in performance_history]
        f1_scores = [record["metrics"]["f1_macro"] for record in performance_history]

        # Calculate efficiency: F1 improvement per annotation
        efficiency = []
        for i in range(1, len(f1_scores)):
            f1_diff = f1_scores[i] - f1_scores[i - 1]
            ann_diff = annotations[i] - annotations[i - 1]
            eff = f1_diff / ann_diff if ann_diff > 0 else 0
            efficiency.append(eff)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Cumulative F1 improvement
        cumulative_improvement = [f1 - f1_scores[0] for f1 in f1_scores]
        ax1.plot(annotations, cumulative_improvement, "o-", linewidth=2, markersize=6)
        ax1.set_xlabel("Number of Annotations")
        ax1.set_ylabel("Cumulative F1 Improvement")
        ax1.set_title(f"{title} - Cumulative Improvement")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Marginal efficiency
        ax2.plot(annotations[1:], efficiency, "o-", linewidth=2, markersize=6, color="orange")
        ax2.set_xlabel("Number of Annotations")
        ax2.set_ylabel("F1 Improvement per Annotation")
        ax2.set_title(f"{title} - Marginal Efficiency")
        ax2.grid(True, alpha=0.3)

        # Add trend line
        if len(efficiency) > 2:
            z = np.polyfit(annotations[1:], efficiency, 1)
            p = np.poly1d(z)
            ax2.plot(annotations[1:], p(annotations[1:]), "r--", alpha=0.8, label="Trend")
            ax2.legend()

        plt.tight_layout()

        # Save plot
        if save_name is None:
            save_name = f"annotation_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        plot_path = os.path.join(self.save_dir, save_name)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    def plot_query_strategy_comparison(
        self,
        strategy_results: Dict[str, Dict[str, List[float]]],
        title: str = "Query Strategy Comparison",
        save_name: Optional[str] = None,
    ) -> str:
        """
        Compare different query strategies

        Args:
            strategy_results: Dict mapping strategy names to curve data
            title: Plot title
            save_name: Filename to save plot

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = sns.color_palette("husl", len(strategy_results))

        for i, (strategy_name, curve_data) in enumerate(strategy_results.items()):
            annotations = curve_data["annotations"]
            f1_scores = curve_data["f1_macro"]

            ax.plot(
                annotations,
                f1_scores,
                "o-",
                label=strategy_name,
                linewidth=2,
                markersize=4,
                color=colors[i],
            )

        ax.set_xlabel("Number of Annotations")
        ax.set_ylabel("F1 Score (Macro)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        # Save plot
        if save_name is None:
            save_name = f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        plot_path = os.path.join(self.save_dir, save_name)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    def plot_annotation_distribution(
        self,
        annotations: List[Dict[str, Any]],
        title: str = "Annotation Distribution",
        save_name: Optional[str] = None,
    ) -> str:
        """
        Plot distribution of annotation labels

        Args:
            annotations: List of annotation dictionaries
            title: Plot title
            save_name: Filename to save plot

        Returns:
            Path to saved plot
        """
        if not annotations:
            raise ValueError("No annotations provided")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Toxicity distribution
        toxicity_counts = {}
        for ann in annotations:
            tox = ann.get("toxicity", "unknown")
            toxicity_counts[tox] = toxicity_counts.get(tox, 0) + 1

        ax1.bar(toxicity_counts.keys(), toxicity_counts.values())
        ax1.set_title("Toxicity Distribution")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis="x", rotation=45)

        # Bullying distribution
        bullying_counts = {}
        for ann in annotations:
            bully = ann.get("bullying", "unknown")
            bullying_counts[bully] = bullying_counts.get(bully, 0) + 1

        ax2.bar(bullying_counts.keys(), bullying_counts.values(), color="orange")
        ax2.set_title("Bullying Distribution")
        ax2.set_ylabel("Count")
        ax2.tick_params(axis="x", rotation=45)

        # Emotion distribution
        emotion_counts = {}
        for ann in annotations:
            emotion = ann.get("emotion", "unknown")
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        ax3.bar(emotion_counts.keys(), emotion_counts.values(), color="green")
        ax3.set_title("Emotion Distribution")
        ax3.set_ylabel("Count")
        ax3.tick_params(axis="x", rotation=45)

        # Confidence distribution
        confidences = [ann.get("confidence", 0) for ann in annotations if "confidence" in ann]
        if confidences:
            ax4.hist(confidences, bins=20, alpha=0.7, color="purple")
            ax4.set_title("Confidence Distribution")
            ax4.set_xlabel("Confidence Score")
            ax4.set_ylabel("Count")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        # Save plot
        if save_name is None:
            save_name = f"annotation_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        plot_path = os.path.join(self.save_dir, save_name)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    def plot_uncertainty_vs_diversity(
        self,
        sample_data: Dict[str, List[float]],
        title: str = "Uncertainty vs Diversity",
        save_name: Optional[str] = None,
    ) -> str:
        """
        Plot uncertainty vs diversity scores for selected samples

        Args:
            sample_data: Dict with 'uncertainty' and 'diversity' score lists
            title: Plot title
            save_name: Filename to save plot

        Returns:
            Path to saved plot
        """
        uncertainty = sample_data["uncertainty"]
        diversity = sample_data["diversity"]

        if len(uncertainty) != len(diversity):
            raise ValueError("Uncertainty and diversity lists must have same length")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Scatter plot
        ax1.scatter(uncertainty, diversity, alpha=0.6, s=50)
        ax1.set_xlabel("Uncertainty Score")
        ax1.set_ylabel("Diversity Score")
        ax1.set_title(f"{title} - Scatter Plot")
        ax1.grid(True, alpha=0.3)

        # Add correlation info
        correlation = np.corrcoef(uncertainty, diversity)[0, 1]
        ax1.text(
            0.05,
            0.95,
            f"Correlation: {correlation:.3f}",
            transform=ax1.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )

        # Histogram comparison
        ax2.hist(uncertainty, bins=20, alpha=0.5, label="Uncertainty", density=True)
        ax2.hist(diversity, bins=20, alpha=0.5, label="Diversity", density=True)
        ax2.set_xlabel("Score")
        ax2.set_ylabel("Density")
        ax2.set_title(f"{title} - Score Distributions")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        if save_name is None:
            save_name = f"uncertainty_vs_diversity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        plot_path = os.path.join(self.save_dir, save_name)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    def create_summary_report(
        self, results: Dict[str, Any], save_name: Optional[str] = None
    ) -> str:
        """
        Create a comprehensive summary report with multiple plots

        Args:
            results: Complete active learning results
            save_name: Filename prefix for saved plots

        Returns:
            Path to summary directory
        """
        if save_name is None:
            save_name = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        summary_dir = os.path.join(self.save_dir, save_name)
        os.makedirs(summary_dir, exist_ok=True)

        plots = {}

        # Learning curves
        if "performance_history" in results:
            curve_data = results["performance_history"]
            plots["learning_curves"] = self.plot_learning_curves(
                curve_data,
                title="Active Learning Progress",
                save_name=os.path.join(summary_dir, "learning_curves.png"),
            )

        # Annotation efficiency
        if "performance_history" in results and len(results["performance_history"]) > 1:
            # Convert curve data back to performance history format
            performance_history = []
            for i, ann_count in enumerate(results["performance_history"]["annotations"]):
                record = {
                    "total_annotations": ann_count,
                    "metrics": {"f1_macro": results["performance_history"]["f1_macro"][i]},
                }
                performance_history.append(record)

            plots["efficiency"] = self.plot_annotation_efficiency(
                performance_history,
                title="Annotation Efficiency Analysis",
                save_name=os.path.join(summary_dir, "annotation_efficiency.png"),
            )

        # Create summary text report
        report_path = os.path.join(summary_dir, "summary_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Active Learning Summary Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total iterations: {results.get('total_iterations', 'N/A')}\n")
            f.write(f"Total annotations: {results.get('total_annotations', 'N/A')}\n")
            f.write(f"Total time: {results.get('total_time_seconds', 0):.2f} seconds\n")
            f.write(f"Annotations per hour: {results.get('annotations_per_hour', 0):.1f}\n")

            if "final_performance" in results:
                f.write("\nFinal Performance:\n")
                for metric, value in results["final_performance"].items():
                    f.write(f"  {metric}: {value:.4f}\n")

            if "annotation_statistics" in results:
                f.write("\nAnnotation Statistics:\n")
                stats = results["annotation_statistics"]
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")

            f.write("\nGenerated plots:\n")
            for plot_name, plot_path in plots.items():
                f.write(f"  {plot_name}: {plot_path}\n")

        return summary_dir

    def plot_from_checkpoint(self, checkpoint_path: str) -> str:
        """
        Load results from checkpoint and create plots

        Args:
            checkpoint_path: Path to checkpoint JSON file

        Returns:
            Path to created plots
        """
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)

        # Extract performance history
        performance_history = checkpoint.get("performance_history", [])

        if not performance_history:
            raise ValueError("No performance history found in checkpoint")

        # Convert to curve data format
        curve_data = {
            "annotations": [record["total_annotations"] for record in performance_history],
            "f1_macro": [record["metrics"]["f1_macro"] for record in performance_history],
        }

        # Add per-class scores if available
        if performance_history and "f1_toxic" in performance_history[0]["metrics"]:
            curve_data["f1_toxic"] = [
                record["metrics"]["f1_toxic"] for record in performance_history
            ]
        if performance_history and "f1_severe" in performance_history[0]["metrics"]:
            curve_data["f1_severe"] = [
                record["metrics"]["f1_severe"] for record in performance_history
            ]

        # Create plots
        plots_dir = os.path.join(
            self.save_dir, f"checkpoint_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(plots_dir, exist_ok=True)

        # Learning curves
        self.plot_learning_curves(
            curve_data,
            title="Active Learning Progress (Checkpoint)",
            save_name=os.path.join(plots_dir, "learning_curves.png"),
        )

        # Efficiency plot
        if len(performance_history) > 1:
            self.plot_annotation_efficiency(
                performance_history,
                title="Annotation Efficiency (Checkpoint)",
                save_name=os.path.join(plots_dir, "efficiency.png"),
            )

        return plots_dir
