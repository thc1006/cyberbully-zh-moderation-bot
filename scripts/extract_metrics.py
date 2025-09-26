#!/usr/bin/env python3
"""
CyberPuppy Metrics Extraction
=============================

Extracts training metrics from model directories for comparison and early stopping.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional


def extract_metrics_from_model(model_dir: Path, metric_name: str = "f1_macro") -> Optional[float]:
    """Extract a specific metric from a trained model directory."""

    try:
        # Check for metrics.json first
        metrics_file = model_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)

            if metric_name in metrics:
                return float(metrics[metric_name])

        # Check for eval_results.json (common in HuggingFace models)
        eval_file = model_dir / "eval_results.json"
        if eval_file.exists():
            with open(eval_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            # Try different metric name formats
            possible_names = [
                metric_name,
                f"eval_{metric_name}",
                f"test_{metric_name}",
                f"validation_{metric_name}"
            ]

            for name in possible_names:
                if name in results:
                    return float(results[name])

        # Check for trainer_state.json
        trainer_state_file = model_dir / "trainer_state.json"
        if trainer_state_file.exists():
            with open(trainer_state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            # Extract from log history
            if "log_history" in state:
                best_metric = None
                for entry in state["log_history"]:
                    for name in [metric_name, f"eval_{metric_name}"]:
                        if name in entry:
                            metric_value = float(entry[name])
                            if best_metric is None or metric_value > best_metric:
                                best_metric = metric_value

                if best_metric is not None:
                    return best_metric

        # Check for pytorch_model.bin or safetensors and try to extract from filename
        # This is a fallback for when metrics are embedded in checkpoint names
        checkpoint_files = list(model_dir.glob("checkpoint-*"))
        if checkpoint_files:
            # Sort by checkpoint number
            checkpoint_files.sort(key=lambda x: int(x.name.split('-')[1]) if x.name.split('-')[1].isdigit() else 0)
            latest_checkpoint = checkpoint_files[-1]

            # Check for metrics in the checkpoint directory
            checkpoint_metrics = latest_checkpoint / "eval_results.json"
            if checkpoint_metrics.exists():
                with open(checkpoint_metrics, 'r', encoding='utf-8') as f:
                    results = json.load(f)

                for name in [metric_name, f"eval_{metric_name}"]:
                    if name in results:
                        return float(results[name])

        # If no metrics found, return 0.0 as default
        print(f"Warning: Could not find metric '{metric_name}' in {model_dir}")
        return 0.0

    except Exception as e:
        print(f"Error extracting metrics from {model_dir}: {e}")
        return 0.0


def extract_all_metrics(model_dir: Path) -> Dict[str, float]:
    """Extract all available metrics from a model directory."""

    all_metrics = {}

    try:
        # Standard metric files to check
        metric_files = [
            model_dir / "metrics.json",
            model_dir / "eval_results.json",
            model_dir / "test_results.json",
            model_dir / "validation_results.json"
        ]

        for metric_file in metric_files:
            if metric_file.exists():
                with open(metric_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                    all_metrics.update(metrics)

        # Also check trainer state
        trainer_state_file = model_dir / "trainer_state.json"
        if trainer_state_file.exists():
            with open(trainer_state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            if "log_history" in state:
                # Get the latest evaluation metrics
                latest_eval = {}
                for entry in state["log_history"]:
                    for key, value in entry.items():
                        if key.startswith(("eval_", "test_", "val_")) and isinstance(value, (int, float)):
                            latest_eval[key] = value

                all_metrics.update(latest_eval)

        # Standardize metric names (remove prefixes)
        standardized_metrics = {}
        for key, value in all_metrics.items():
            clean_key = key.replace("eval_", "").replace("test_", "").replace("val_", "")
            standardized_metrics[clean_key] = value

        # Add the original keys as well for compatibility
        standardized_metrics.update(all_metrics)

        return standardized_metrics

    except Exception as e:
        print(f"Error extracting all metrics from {model_dir}: {e}")
        return {}


def save_metrics_summary(model_dir: Path, metrics: Dict[str, float]):
    """Save a standardized metrics summary to the model directory."""

    try:
        summary_file = model_dir / "metrics_summary.json"

        # Create a comprehensive summary
        summary = {
            "extraction_timestamp": str(Path(__file__).stat().st_mtime),
            "model_directory": str(model_dir),
            "metrics": metrics,
            "key_metrics": {
                "f1_macro": metrics.get("f1_macro", 0.0),
                "precision_macro": metrics.get("precision_macro", 0.0),
                "recall_macro": metrics.get("recall_macro", 0.0),
                "accuracy": metrics.get("accuracy", 0.0)
            }
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Metrics summary saved to: {summary_file}")

    except Exception as e:
        print(f"Warning: Could not save metrics summary: {e}")


def main():
    """Main metrics extraction entry point."""
    parser = argparse.ArgumentParser(description='Extract metrics from CyberPuppy model directories')
    parser.add_argument('--model-dir', required=True, help='Model directory path')
    parser.add_argument('--metric', default='f1_macro', help='Metric name to extract')
    parser.add_argument('--all-metrics', action='store_true', help='Extract all available metrics')
    parser.add_argument('--save-summary', action='store_true', help='Save metrics summary to model directory')

    args = parser.parse_args()

    try:
        model_dir = Path(args.model_dir)

        if not model_dir.exists():
            print(f"Error: Model directory does not exist: {model_dir}")
            sys.exit(1)

        if not model_dir.is_dir():
            print(f"Error: Path is not a directory: {model_dir}")
            sys.exit(1)

        if args.all_metrics:
            # Extract all metrics
            metrics = extract_all_metrics(model_dir)

            if metrics:
                print("Extracted metrics:")
                for key, value in sorted(metrics.items()):
                    print(f"  {key}: {value}")

                if args.save_summary:
                    save_metrics_summary(model_dir, metrics)
            else:
                print("No metrics found in model directory")
                sys.exit(1)

        else:
            # Extract specific metric
            metric_value = extract_metrics_from_model(model_dir, args.metric)

            if metric_value is not None:
                # Output just the value for easy parsing by batch scripts
                print(metric_value)
            else:
                print(f"Error: Could not extract metric '{args.metric}' from {model_dir}")
                sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()