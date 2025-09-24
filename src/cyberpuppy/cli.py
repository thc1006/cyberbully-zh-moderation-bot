"""
CyberPuppy Command Line Interface

A comprehensive CLI for Chinese cyberbullying detection and toxicity analysis.
Provides commands for text analysis, model training, evaluation, and export.
"""
import argparse
import json
import csv
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
)
from rich.panel import Panel
from rich.logging import RichHandler

# Import core CyberPuppy components
try:
    from .models.detector import CyberPuppyDetector
except ImportError:
    # Fallback mock detector for CLI testing
    CyberPuppyDetector = None

from .models.trainer import ModelTrainer
from .evaluation.evaluator import ModelEvaluator
from .models.exporter import ModelExporter
from .config import load_config, get_default_config


class MockDetectionResult:
    """Mock detection result for CLI testing."""

    def __init__(self, text: str):
        self.text = text
        self.processing_time = 0.123

        # Mock results based on simple heuristics
        if any(word in text.lower() for word in ['stupid', 'idiot', 'hate', 'kill']):
            self.toxicity = MockPrediction('toxic', 0.85)
            self.bullying = MockPrediction('harassment', 0.80)
            self.emotion = MockEmotionPrediction('neg', 3, 0.78)
            self.role = MockPrediction('perpetrator', 0.72)
        elif any(word in text.lower() for word in ['good', 'great', 'nice', 'love']):
            self.toxicity = MockPrediction('none', 0.92)
            self.bullying = MockPrediction('none', 0.95)
            self.emotion = MockEmotionPrediction('pos', 2, 0.88)
            self.role = MockPrediction('none', 0.90)
        else:
            self.toxicity = MockPrediction('none', 0.75)
            self.bullying = MockPrediction('none', 0.80)
            self.emotion = MockEmotionPrediction('neu', 0, 0.70)
            self.role = MockPrediction('none', 0.85)

        self.explanations = None


class MockPrediction:
    """Mock prediction result."""

    def __init__(self, value: str, confidence: float):
        self.prediction = MockValue(value)
        self.confidence = confidence


class MockEmotionPrediction(MockPrediction):
    """Mock emotion prediction with strength."""

    def __init__(self, value: str, strength: int, confidence: float):
        super().__init__(value, confidence)
        self.strength = strength


class MockValue:
    """Mock enum-like value."""

    def __init__(self, value: str):
        self.value = value


class MockDetector:
    """Mock detector for CLI testing when real detector is not available."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def analyze(self, text: str) -> MockDetectionResult:
        """Analyze text and return mock results."""
        return MockDetectionResult(text)


class CLIError(Exception):
    """CLI-specific error with exit code."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    text: str
    toxicity: str
    bullying: str
    emotion: str
    emotion_strength: int
    role: str
    confidence: float
    explanations: Optional[Dict[str, Any]] = None


class BaseCommand(ABC):
    """Base class for CLI commands."""

    def __init__(self, console: Console):
        self.console = console

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """Execute the command with given arguments."""
        pass

    def _validate_file_exists(
        self,
        filepath: str,
        file_type: str = "File"
    ) -> Path:
        """Validate that a file exists."""
        path = Path(filepath)
        if not path.exists():
            raise CLIError(f"{file_type} not found: {filepath}")
        return path

    def _create_progress_bar(self, description: str = "Proce"
        "ssing") -> Progress:
        """Create a Rich progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )


class AnalyzeCommand(BaseCommand):
    """Command for analyzing text toxicity and emotions."""

    def __init__(self, detector: CyberPuppyDetector, console: Console):
        super().__init__(console)
        self.detector = detector

    def execute(self, args: argparse.Namespace) -> int:
        """Execute text analysis."""
        try:
            if args.text:
                # Single text analysis
                return self._analyze_single_text(args.text, args.format)
            elif args.input:
                # Batch file processing
                return self._analyze_batch_file(
                    args.input,
                    args.output,
                    args.format)
            else:
                # Interactive mode - read from stdin
                return self._analyze_interactive(args.format)

        except CLIError:
            raise
        except Exception as e:
            raise CLIError(f"Analysis failed: {str(e)}")

    def _analyze_single_text(self, text: str, output_format: str) -> int:
        """Analyze a single text input."""
        text_preview = text[:100] + ('...' if len(text) > 100 else '')
        self.console.print(f"[blue]Analyzing text:[/blue] {text_preview}")

        detection_result = self.detector.analyze(text)

        # Convert DetectionResult to dictionary format for CLI compatibility
        result = {
            'text': detection_result.text,
            'toxicity': detection_result.toxicity.prediction.value,
            'bullying': detection_result.bullying.prediction.value,
            'emotion': detection_result.emotion.prediction.value,
            'emotion_strength': detection_result.emotion.strength,
            'role': detection_result.role.prediction.value,
            # Use toxicity confidence as overall confidence score
            'confidence': detection_result.toxicity.confidence,
            'explanations': detection_result.explanations,
            'processing_time': detection_result.processing_time
        }

        analysis_result = AnalysisResult(
            text=text,
            toxicity=result['toxicity'],
            bullying=result['bullying'],
            emotion=result['emotion'],
            emotion_strength=result['emotion_strength'],
            role=result['role'],
            confidence=result['confidence'],
            explanations=result.get('explanations')
        )

        if output_format == 'table':
            self._print_table_result(analysis_result)
        elif output_format == 'json':
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            self._print_simple_result(analysis_result)

        return 0

    def _analyze_batch_file(
        self, input_path: str, output_path: Optional[str], output_format: str
    ) -> int:
        """Analyze texts from a batch file."""
        input_file = self._validate_file_exists(input_path, "Input file")

        # Detect input format
        if input_path.endswith('.csv'):
            texts = self._read_csv_file(input_file)
        elif input_path.endswith('.json'):
            texts = self._read_json_file(input_file)
        elif input_path.endswith('.txt'):
            texts = self._read_text_file(input_file)
        else:
            raise CLIError(f"Unsupported input format: {input_path}")

        results = []
        with self._create_progress_bar("Analyzing texts") as progress:
            task = progress.add_task("Processing", total=len(texts))

            for i, text_data in enumerate(texts):
                text = text_data['text'] if isinstance(
                    text_data,
                    dict) else text_data
                detection_result = self.detector.analyze(text)

                # Convert DetectionResult to dictionary format
                result = {
                    'text': detection_result.text,
                    'toxicity': detection_result.toxicity.prediction.value,
                    'bullying': detection_result.bullying.prediction.value,
                    'emotion': detection_result.emotion.prediction.value,
                    'emotion_strength': detection_result.emotion.strength,
                    'role': detection_result.role.prediction.value,
                    'confidence': detection_result.toxicity.confidence,
                    'processing_time': detection_result.processing_time
                }

                # Add metadata if available
                if isinstance(text_data, dict):
                    result.update(
                        {k: v for k,
                        v in text_data.items() if k != 'text'})
                results.append(result)
                progress.update(task, advance=1)

        # Output results
        if output_path:
            self._save_results(results, output_path, output_format)
            self.console.print(f"[green]*[/green] Results saved to \
                {output_path}")
        else:
            self._display_batch_results(results, output_format)

        self.console.print(f"[green]*[/green] Analyzed {len(results)} texts")
        return 0

    def _analyze_interactive(self, output_format: str) -> int:
        """Analyze text from stdin interactively."""
        self.console.print("[blue]Enter text to analyze (Ctrl"
            "+D or empty line to exit):[/blue]")

        try:
            while True:
                try:
                    text = input("> ")
                    if not text.strip():
                        break

                    detection_result = self.detector.analyze(text)

                    # Convert DetectionResult to dictionary format
                    result = {
                        'text': detection_result.text,
                        'toxicity': detection_result.toxicity.prediction.value,
                        'bullying': detection_result.bullying.prediction.value,
                        'emotion': detection_result.emotion.prediction.value,
                        'emotion_strength': detection_result.emotion.strength,
                        'role': detection_result.role.prediction.value,
                        'confidence': detection_result.toxicity.confidence,
                        'processing_time': detection_result.processing_time
                    }

                    analysis_result = AnalysisResult(
                        text=text,
                        toxicity=result['toxicity'],
                        bullying=result['bullying'],
                        emotion=result['emotion'],
                        emotion_strength=result['emotion_strength'],
                        role=result['role'],
                        confidence=result['confidence']
                    )

                    if output_format == 'json':
                        print(json.dumps(result, indent=2, ensure_ascii=False))
                    else:
                        self._print_simple_result(analysis_result)

                except EOFError:
                    break

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Analysis interrupted[/yellow]")

        return 0

    def _read_csv_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Read texts from CSV file."""
        texts = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'text' in row:
                    texts.append(row)
                else:
                    # Assume first column is text
                    first_col = next(iter(row.values()))
                    texts.append({'text': first_col, **row})
        return texts

    def _read_json_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Read texts from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'texts' in data:
                return data['texts']
            else:
                raise CLIError("JSON file must contain a list"
                    " of texts or {'texts': [...]}")

    def _read_text_file(self, filepath: Path) -> List[str]:
        """Read texts from plain text file (one per line)."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def _save_results(
        self,
        results: List[Dict[str,
        Any]],
        output_path: str,
        format_type: str
    ):
        """Save results to file."""
        if format_type == 'json' or output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif format_type == 'csv' or output_path.endswith('.csv'):
            if results:
                fieldnames = list(results[0].keys())
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)
        else:
            raise CLIError(f"Unsupported output format: {output_path}")

    def _display_batch_results(
        self,
        results: List[Dict[str,
        Any]],
        format_type: str
    ):
        """Display batch results to console."""
        if format_type == 'json':
            print(json.dumps(results, indent=2, ensure_ascii=False))
        elif format_type == 'table':
            table = Table(title="Analysis Results")
            table.add_column("Text", style="cyan", no_wrap=False, max_width=40)
            table.add_column("Toxicity", style="red")
            table.add_column("Bullying", style="yellow")
            table.add_column("Emotion", style="green")
            table.add_column("Confidence", style="blue")

            for result in results:
                text = result['text'][:50] + "..." if len(result['text']) > 50 else result['text']
                table.add_row(
                    text,
                    result.get('toxicity', 'none'),
                    result.get('bullying', 'none'),
                    result.get('emotion', 'neu'),
                    f"{result.get('confidence', 0):.2f}"
                )

            self.console.print(table)

    def _print_table_result(self, result: AnalysisResult):
        """Print analysis result as a formatted table."""
        table = Table(
            title="Analysis Result",
            show_header=True,
            header_style="bold magenta")
        table.add_column("Attribute", style="cyan", width=15)
        table.add_column("Value", style="white")

        # Color code based on severity
        toxicity_color = \
            "red" if result.toxicity in ['toxic', 'severe'] else "green"
        bullying_color = \
            "red" if result.bullying in ['harassment', 'threat'] else "green"
        emotion_color = (
            "green" if result.emotion == 'pos'
            else ("red" if result.emotion == 'neg' else "yellow")
        )

        table.add_row(
            "Text",
            result.text[:100] + "."
                ".." if len(result.text) > 100 else result.text)
        table.add_row(
            "Toxicity",
            f"[{toxicity_color}]{result.toxicity}[/{toxicity_color}]")
        table.add_row(
            "Bullying",
            f"[{bullying_color}]{result.bullying}[/{bullying_color}]")
        table.add_row(
            "Emotion",
            f"[{emotion_color}]{result.emotion}[/{emotion_color}]")
        table.add_row("Emotion Strength", str(result.emotion_strength))
        table.add_row("Role", result.role)
        table.add_row("Confidence", f"{result.confidence:.3f}")

        self.console.print(table)

    def _print_simple_result(self, result: AnalysisResult):
        """Print analysis result in simple format."""
        self.console.print(Panel(
            f"[bold]Analysis Results[/bold]\n\n"
            f"[cyan]Text:[/cyan] {result.text}\n"
            f"[red]Toxicity:[/red] {result.toxicity}\n"
            f"[yellow]Bullying:[/yellow] {result.bullying}\n"
            f"[green]Emotion:[/green] {result.emotion} (strength: \
                {result.emotion_strength})\n"
            f"[blue]Role:[/blue] {result.role}\n"
            f"[white]Confidence:[/white] {result.confidence:.3f}",
            expand=False
        ))


class TrainCommand(BaseCommand):
    """Command for training models."""

    def __init__(self, trainer: ModelTrainer, console: Console):
        super().__init__(console)
        self.trainer = trainer

    def execute(self, args: argparse.Namespace) -> int:
        """Execute model training."""
        try:
            # Load configuration
            config = self._load_training_config(args)

            # Validate dataset
            if args.dataset and not self._validate_dataset(args.dataset):
                raise CLIError(f"Invalid dataset: {args.dataset}")

            # Setup training parameters
            training_params = {
                'dataset': args.dataset,
                'epochs': args.epochs or config.get('epochs', 10),
                'batch_size': args.batch_size or config.get('batch_size', 32),
                'learning_rate': args.learning_rate or config.get(
                    'learning_rate',
                    2e-5),
                'output': args.output or 'model.pt',
                'config': config
            }

            self.console.print(
                f"[blue]Starting training with dataset: \
                    {training_params['dataset']}[/blue]"
            )
            self.console.print(
                f"[cyan]Parameters:[/cyan] \
                    epochs={training_params['epochs']}, "
                f"batch_size={training_params['batch_size']}, "
                f"lr={training_params['learning_rate']}"
            )

            # Create progress tracking
            with self._create_training_progress() as progress:
                task = progress.add_task(
                    f"Training {args.dataset}", total=training_params['epochs']
                )

                def progress_callback(epoch: int, loss: float, f1: float):
                    progress.update(
                        task, advance=1,
                        description=f"Epoch {epoch}: loss={loss:.4f}, \
                            f1={f1:.3f}"
                    )

                training_params['progress_callback'] = progress_callback

                # Execute training
                results = self.trainer.train(**training_params)

            # Display results
            self._display_training_results(results)

            return 0

        except Exception as e:
            raise CLIError(f"Training failed: {str(e)}")

    def _load_training_config(
        self,
        args: argparse.Namespace
    ) -> Dict[str, Any]:
        """Load training configuration from file or defaults."""
        if args.config:
            config_path = self._validate_file_exists(
                args.config,
                "Config file")
            with open(config_path, 'r', encoding='utf-8') as f:
                if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        else:
            return get_default_config()

    def _validate_dataset(self, dataset: str) -> bool:
        """Validate dataset name."""
        valid_datasets = \
            ['COLD', 'ChnSentiCorp', 'DMSC', 'NTUSD', 'SCCD', 'CHNCI']
        return dataset in valid_datasets

    def _create_training_progress(self) -> Progress:
        """Create progress bar for training."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )

    def _display_training_results(self, results: Dict[str, Any]):
        """Display training results."""
        table = Table(
            title="Training Results",
            show_header=True,
            header_style="bold green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Final Loss", f"{results.get('final_loss', 0):.4f}")
        table.add_row("Best F1 Score", f"{results.get('best_f1', 0):.3f}")
        table.add_row(
            "Epochs Completed",
            str(results.get('epochs_completed',
            0)))
        table.add_row("Model Path", results.get('model_path', 'N/A'))

        self.console.print(table)

        if results.get('best_f1', 0) > 0.8:
            self.console.print(
                "[green]* Training completed success"
                    "fully with good performance![/green]"
            )
        else:
            self.console.print(
                "[yellow]! Training completed but pe"
                    "rformance may be suboptimal[/yellow]"
            )


class EvaluateCommand(BaseCommand):
    """Command for evaluating models."""

    def __init__(self, evaluator: ModelEvaluator, console: Console):
        super().__init__(console)
        self.evaluator = evaluator

    def execute(self, args: argparse.Namespace) -> int:
        """Execute model evaluation."""
        try:
            # Validate inputs
            self._validate_file_exists(args.model, "Model file")
            self._validate_file_exists(args.dataset, "Dataset file")

            self.console.print(f"[blue]Evaluating model: {args.model}[/blue]")
            self.console.print(f"[cyan]Dataset: {args.dataset}[/cyan]")

            # Run evaluation
            with self.console.status("[bold green]Running evaluation..."):
                results = self.evaluator.evaluate(args.model, args.dataset)

            # Display results
            self._display_evaluation_results(results, args.metrics)

            # Save results if output specified
            if args.output:
                self._save_evaluation_results(results, args.output)
                self.console.print(f"[green]*[/green] Results saved to \
                    {args.output}")

            return 0

        except Exception as e:
            raise CLIError(f"Evaluation failed: {str(e)}")

    def _display_evaluation_results(
        self,
        results: Dict[str,
        Any],
        requested_metrics: List[str]
    ):
        """Display evaluation results in formatted table."""
        table = Table(
            title="Evaluation Results",
            show_header=True,
            header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        # Core metrics
        if not requested_metrics or 'accuracy' in requested_metrics:
            table.add_row("Accuracy", f"{results.get('accuracy', 0):.4f}")
        if not requested_metrics or 'f1' in requested_metrics:
            table.add_row("F1 Score", f"{results.get('f1', 0):.4f}")
        if not requested_metrics or 'precision' in requested_metrics:
            table.add_row("Precision", f"{results.get('precision', 0):.4f}")
        if not requested_metrics or 'recall' in requested_metrics:
            table.add_row("Recall", f"{results.get('recall', 0):.4f}")

        self.console.print(table)

        # Confusion matrix if available
        if 'confusion_matrix' in results:
            self._display_confusion_matrix(results['confusion_matrix'])

        # Classification report if available
        if 'classification_report' in results:
            self._display_classification_report(results['classification_report'])

    def _display_confusion_matrix(self, confusion_matrix: List[List[int]]):
        """Display confusion matrix."""
        table = Table(title="Confusion Matrix", show_header=True)
        table.add_column("", style="cyan")

        labels = ['none', 'toxic']  # Adjust based on your labels
        for label in labels:
            table.add_column(f"Pred {label}", justify="center")

        for i, (actual_label, row) in enumerate(zip(labels, confusion_matrix)):
            table.add_row(f"Actual {actual_label}", *[str(val) for val in row])

        self.console.print(table)

    def _display_classification_report(self, report: Dict[str, Any]):
        """Display classification report."""
        table = Table(title="Classification Report", show_header=True)
        table.add_column("Class", style="cyan")
        table.add_column("Precision", justify="center")
        table.add_column("Recall", justify="center")
        table.add_column("F1-Score", justify="center")

        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                table.add_row(
                    class_name,
                    f"{metrics.get('precision', 0):.3f}",
                    f"{metrics.get('recall', 0):.3f}",
                    f"{metrics.get('f1-score', 0):.3f}"
                )

        self.console.print(table)

    def _save_evaluation_results(
        self,
        results: Dict[str,
        Any],
        output_path: str
    ):
        """Save evaluation results to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


class ExportCommand(BaseCommand):
    """Command for exporting models."""

    def __init__(self, exporter: ModelExporter, console: Console):
        super().__init__(console)
        self.exporter = exporter

    def execute(self, args: argparse.Namespace) -> int:
        """Execute model export."""
        try:
            # Validate inputs
            self._validate_file_exists(args.model, "Model file")

            supported_formats = ['onnx', 'torchscript', 'huggingface']
            if args.format not in supported_formats:
                raise CLIError(f"Unsupported export format: {args.format}. "
                               f"Supported: {', '.join(supported_formats)}")

            self.console.print(f"[blue]Exporting model: {args.model}[/blue]")
            self.console.print(f"[cyan]Format: {args.format}[/cyan]")
            self.console.print(f"[cyan]Output: {args.output}[/cyan]")

            # Execute export
            with self.console.status(f"[bold green]Exporting to \
                {args.format}..."):
                if args.format == 'onnx':
                    results = self.exporter.export_to_onnx(
                        args.model,
                        args.output)
                elif args.format == 'torchscript':
                    results = self.exporter.export_to_torchscript(
                        args.model,
                        args.output)
                elif args.format == 'huggingface':
                    results = self.exporter.export_to_huggingface(
                        args.model,
                        args.output)

            # Display results
            self._display_export_results(results)

            return 0

        except Exception as e:
            raise CLIError(f"Export failed: {str(e)}")

    def _display_export_results(self, results: Dict[str, Any]):
        """Display export results."""
        if results.get('success'):
            self.console.print("[green]* Export complet"
                "ed successfully![/green]")
            self.console.print(f"[cyan]Output path:[/cyan] \
                {results.get('output_path')}")

            if 'model_size_mb' in results:
                self.console.print(f"[cyan]Model size:[/cyan] \
                    {results['model_size_mb']:.1f} MB")
        else:
            self.console.print("[red]X Export failed![/red]")
            if 'error' in results:
                self.console.print(f"[red]Error:[/red] {results['error']}")


class ConfigCommand(BaseCommand):
    """Command for configuration management."""

    def __init__(self, console: Console):
        super().__init__(console)

    def execute(self, args: argparse.Namespace) -> int:
        """Execute config management."""
        try:
            if args.show:
                return self._show_config()
            elif args.get:
                return self._get_config_value(args.get)
            elif args.set:
                return self._set_config_value(args.set, args.value)
            else:
                self.console.print(
                    "[yellow]No config action specified"
                        ". Use --help for options.[/yellow]"
                )
                return 1

        except Exception as e:
            raise CLIError(f"Config operation failed: {str(e)}")

    def _show_config(self) -> int:
        """Show current configuration."""
        config = load_config()

        self.console.print("[bold]Current Configuration:[/bold]")
        self.console.print(Panel(
            json.dumps(config, indent=2, ensure_ascii=False),
            title="Config",
            expand=False
        ))

        return 0

    def _get_config_value(self, key: str) -> int:
        """Get specific configuration value."""
        config = load_config()

        # Support dotted notation (e.g., model.name)
        keys = key.split('.')
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                self.console.print(f"[red]Configuration key not found: \
                    {key}[/red]")
                return 1

        self.console.print(str(value))
        return 0

    def _set_config_value(self, key: str, value: str) -> int:
        """Set configuration value."""
        # This would need implementation to modify config file
        self.console.print("[yellow]Config modification "
            "not yet implemented[/yellow]")
        return 1


class CyberPuppyCLI:
    """Main CLI application class."""

    def __init__(self):
        self.console = Console()
        self.detector = None
        self.trainer = None
        self.evaluator = None
        self.exporter = None

    def _initialize_components(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CyberPuppy components."""
        try:
            if not self.detector:
                if CyberPuppyDetector is None:
                    # Use mock detector for testing
                    self.detector = MockDetector()
                else:
                    self.detector = \
                        CyberPuppyDetector(config or get_default_config())
            if not self.trainer:
                self.trainer = ModelTrainer(config)
            if not self.evaluator:
                self.evaluator = ModelEvaluator(config)
            if not self.exporter:
                self.exporter = ModelExporter(config)
        except Exception as e:
            raise CLIError(f"Failed to initialize components: {str(e)}")

    def run(self, args: List[str]) -> int:
        """Run the CLI with given arguments."""
        try:
            parser = create_parser()
            parsed_args = parser.parse_args(args)

            # Setup logging
            setup_logging(parsed_args)

            # Load configuration
            config = (
                self._load_config_file(parsed_args.config_file)
                if hasattr(
                    parsed_args,
                    'config_file') and parsed_args.config_file
                else None
            )

            # Initialize components
            self._initialize_components(config)

            # Create and execute command
            command = self._create_command(parsed_args)
            return command.execute(parsed_args)

        except CLIError:
            raise
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Operation inte"
                "rrupted by user[/yellow]")
            return 130
        except Exception as e:
            self.console.print(f"[red]Unexpected error: {str(e)}[/red]")
            return 1

    def _load_config_file(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise CLIError(f"Config file not found: {config_file}")

        with open(config_path, 'r', encoding='utf-8') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)

    def _create_command(self, args: argparse.Namespace) -> BaseCommand:
        """Create appropriate command instance."""
        if args.command == 'analyze':
            return AnalyzeCommand(self.detector, self.console)
        elif args.command == 'train':
            return TrainCommand(self.trainer, self.console)
        elif args.command == 'evaluate':
            return EvaluateCommand(self.evaluator, self.console)
        elif args.command == 'export':
            return ExportCommand(self.exporter, self.console)
        elif args.command == 'config':
            return ConfigCommand(self.console)
        else:
            raise CLIError(f"Unknown command: {args.command}")


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog='cyberpuppy',
        description='CyberPuppy - Chinese Cyberbullying Detection and Toxicity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cyberpuppy analyze "這個人真笨"
  cyberpuppy analyze --input texts.csv --output results.json
  cyberpuppy train --dataset COLD --epochs 10 --output model.pt
  cyberpuppy evaluate --model model.pt --dataset test.csv
  cyberpuppy export --model model.pt --format onnx --output model.onnx

For more information, visit: https://github.com/yourusername/cyberpuppy
        """
    )

    # Global options
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Enable quiet mode')
    parser.add_argument('--config-file', '-c', type=str,
                        help='Path to configuration file (YAML or JSON)')
    parser.add_argument(
        '--version',
        action='version',
        version='CyberPuppy 1.0.0')

    # Subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze text for toxicity and emotions')
    analyze_parser.add_argument('text', nargs='?', help='Text to analyze')
    analyze_parser.add_argument('--input', '-i', type=str,
                                help='Input file (CSV, JSON, or TXT)')
    analyze_parser.add_argument('--output', '-o', type=str,
                                help='Output file for results')
    analyze_parser.add_argument('--format', '-f', choices=['table', 'json',
        'csv'],
                                default='table', help='Output format')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--dataset', '-d', type=str, required=True,
                              choices=['COLD', 'ChnSentiCorp', 'DMSC', 'NTUSD',
                                  'SCCD', 'CHNCI'],
                              help='Dataset to train on')
    train_parser.add_argument('--epochs', '-e', type=int, default=10,
                              help='Number of training epochs')
    train_parser.add_argument('--batch-size', '-b', type=int, default=32,
                              help='Training batch size')
    train_parser.add_argument('--learning-rate', '-lr', type=float,
        default=2e-5,
                              help='Learning rate')
    train_parser.add_argument('--output', '-o', type=str, default='model.pt',
                              help='Output model path')
    train_parser.add_argument('--config', type=str,
                              help='Training configuration file')

    # Evaluate command
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate model performance')
    eval_parser.add_argument('--model', '-m', type=str, required=True,
                             help='Path to model file')
    eval_parser.add_argument('--dataset', '-d', type=str, required=True,
                             help='Path to evaluation dataset')
    eval_parser.add_argument('--output', '-o', type=str,
                             help='Output file for evaluation results')
    eval_parser.add_argument('--metrics', nargs='+',
                             choices=['accuracy', 'precision', 'recall', 'f1'],
                             default=['accuracy', 'f1'],
                             help='Metrics to compute')

    # Export command
    export_parser = subparsers.add_parser(
        'export',
        help='Export model for deployment')
    export_parser.add_argument('--model', '-m', type=str, required=True,
                               help='Path to model file')
    export_parser.add_argument('--format', '-f', type=str, required=True,
                               choices=['onnx', 'torchscript', 'huggingface'],
                               help='Export format')
    export_parser.add_argument('--output', '-o', type=str, required=True,
                               help='Output path')

    # Config command
    config_parser = subparsers.add_parser(
        'config',
        help='Manage configuration')
    config_group = config_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('--show', action='store_true',
                              help='Show current configuration')
    config_group.add_argument('--get', type=str,
                              help='Get configuration value by key')
    config_group.add_argument('--set', type=str,
                              help='Set configuration value')
    config_parser.add_argument('--value', type=str,
                               help='Value to set (used with --set)')

    return parser


def setup_logging(args: argparse.Namespace):
    """Setup logging based on CLI arguments."""
    if hasattr(args, 'quiet') and args.quiet:
        level = logging.ERROR
    elif hasattr(args, 'verbose') and args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )


def format_analysis_result_table(result: Dict[str, Any], text: str) -> Table:
    """Format analysis result as Rich table."""
    table = Table(title="Analysis Result")
    table.add_column("Attribute", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Text", text[:100] + "..." if len(text) > 100 else text)
    table.add_row("Toxicity", result.get('toxicity', 'none'))
    table.add_row("Bullying", result.get('bullying', 'none'))
    table.add_row("Emotion", result.get('emotion', 'neu'))
    table.add_row("Confidence", f"{result.get('confidence', 0):.3f}")

    return table


def format_json_output(result: Dict[str, Any]) -> str:
    """Format result as JSON string."""
    return json.dumps(result, indent=2, ensure_ascii=False)


def create_progress_bar(description: str) -> Progress:
    """Create a Rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn()
    )


def main(args: List[str] = None) -> int:
    """Main entry point for CLI."""
    if args is None:
        args = sys.argv[1:]

    cli = CyberPuppyCLI()

    try:
        return cli.run(args)
    except CLIError as e:
        cli.console.print(f"[red]Error: {str(e)}[/red]")
        return e.exit_code
    except KeyboardInterrupt:
        cli.console.print("\n[yellow]Operation interrupted[/yellow]")
        return 130
    except Exception as e:
        cli.console.print(f"[red]Unexpected error: {str(e)}[/red]")
        return 1


if __name__ == '__main__':
    exit(main())
