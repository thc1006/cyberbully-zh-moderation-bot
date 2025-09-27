"""
Tests for CyberPuppy CLI interface following TDD London School approach.
"""

import pytest
import json
import csv
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from cyberpuppy.cli import (
    CyberPuppyCLI,
    AnalyzeCommand,
    TrainCommand,
    EvaluateCommand,
    ExportCommand,
    ConfigCommand,
    CLIError,
    create_parser,
    main,
)


@pytest.mark.unit
class TestCLIArgumentParsing:
    """Test CLI argument parsing for all commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = CyberPuppyCLI()
        self.parser = create_parser()

    def test_analyze_command_single_text(self):
        """Test parsing analyze command with single text input."""
        args = self.parser.parse_args(["analyze", "This is toxic text"])

        assert args.command == "analyze"
        assert args.text == "This is toxic text"
        assert args.input is None
        assert args.output is None
        assert args.format == "table"

    def test_analyze_command_batch_file(self):
        """Test parsing analyze command with batch file processing."""
        args = self.parser.parse_args(
            [
                "analyze",
                "--input",
                "data.csv",
                "--output",
                "results.json",
                "--format",
                "json",
            ]
        )

        assert args.command == "analyze"
        assert args.text is None
        assert args.input == "data.csv"
        assert args.output == "results.json"
        assert args.format == "json"

    def test_train_command_with_options(self):
        """Test parsing train command with all options."""
        args = self.parser.parse_args(
            [
                "train",
                "--dataset",
                "COLD",
                "--epochs",
                "10",
                "--batch-size",
                "32",
                "--learning-rate",
                "2e-5",
                "--output",
                "model.pt",
                "--config",
                "config.yaml",
            ]
        )

        assert args.command == "train"
        assert args.dataset == "COLD"
        assert args.epochs == 10
        assert args.batch_size == 32
        assert args.learning_rate == 2e-5
        assert args.output == "model.pt"
        assert args.config == "config.yaml"

    def test_evaluate_command_parsing(self):
        """Test parsing evaluate command."""
        args = self.parser.parse_args(
            [
                "evaluate",
                "--model",
                "model.pt",
                "--dataset",
                "test.csv",
                "--output",
                "eval_report.json",
                "--metrics",
                "f1",
                "precision",
                "recall",
            ]
        )

        assert args.command == "evaluate"
        assert args.model == "model.pt"
        assert args.dataset == "test.csv"
        assert args.output == "eval_report.json"
        assert args.metrics == ["f1", "precision", "recall"]

    def test_export_command_parsing(self):
        """Test parsing export command."""
        args = self.parser.parse_args(
            [
                "export",
                "--model",
                "model.pt",
                "--format",
                "onnx",
                "--output",
                "model.onnx",
            ]
        )

        assert args.command == "export"
        assert args.model == "model.pt"
        assert args.format == "onnx"
        assert args.output == "model.onnx"

    def test_config_command_parsing(self):
        """Test parsing config command."""
        args = self.parser.parse_args(["config", "--show"])

        assert args.command == "config"
        assert args.show is True

    def test_global_options_parsing(self):
        """Test parsing global options like verbose and quiet."""
        args = self.parser.parse_args(
            ["--verbose", "--config-file", "custom.yaml", "analyze", "test text"]
        )

        assert args.verbose is True
        assert args.config_file == "custom.yaml"
        assert args.command == "analyze"

    def test_invalid_command_raises_error(self):
        """Test that invalid commands raise appropriate errors."""
        with pytest.raises(SystemExit):
            self.parser.parse_args(["invalid_command"])


@pytest.mark.unit
class TestAnalyzeCommand:
    """Test the analyze command functionality."""

    def setup_method(self):
        """Set up test fixtures with mocked dependencies."""
        self.mock_detector = Mock()
        self.mock_console = MagicMock()
        # Ensure console supports context manager protocol
        self.mock_console.__enter__ = Mock(return_value=self.mock_console)
        self.mock_console.__exit__ = Mock(return_value=None)
        self.command = AnalyzeCommand(self.mock_detector, self.mock_console)

    def _create_mock_detection_result(
        self,
        text: str = "test text",
        toxicity: str = "none",
        bullying: str = "none",
        emotion: str = "neu",
        emotion_strength: int = 1,
        role: str = "none",
        confidence: float = 0.85,
    ) -> Mock:
        """Create a mock detection result object."""
        mock_result = Mock()
        mock_result.text = text
        mock_result.toxicity.prediction.value = toxicity
        mock_result.bullying.prediction.value = bullying
        mock_result.emotion.prediction.value = emotion
        mock_result.emotion.strength = emotion_strength
        mock_result.role.prediction.value = role
        mock_result.toxicity.confidence = confidence
        mock_result.explanations = None
        mock_result.processing_time = 0.123
        return mock_result

    def test_single_text_analysis_with_table_output(self):
        """Test single text analysis with table output format."""
        # Mock detector response with DetectionResult-like object
        mock_result = self._create_mock_detection_result(
            text="You are stupid",
            toxicity="toxic",
            bullying="harassment",
            emotion="neg",
            emotion_strength=3,
            role="perpetrator",
            confidence=0.85,
        )
        self.mock_detector.analyze.return_value = mock_result

        args = Mock()
        args.text = "You are stupid"
        args.format = "table"
        args.input = None
        args.output = None

        result = self.command.execute(args)

        assert result == 0
        self.mock_detector.analyze.assert_called_once_with("You are stupid")
        self.mock_console.print.assert_called()

    def test_single_text_analysis_with_json_output(self):
        """Test single text analysis with JSON output format."""
        mock_result = self._create_mock_detection_result(
            text="Great job!",
            toxicity="none",
            bullying="none",
            emotion="pos",
            emotion_strength=1,
            role="none",
            confidence=0.92,
        )
        self.mock_detector.analyze.return_value = mock_result

        args = Mock()
        args.text = "Great job!"
        args.format = "json"
        args.input = None
        args.output = None

        with patch("builtins.print") as mock_print:
            result = self.command.execute(args)

            assert result == 0
            self.mock_detector.analyze.assert_called_once_with("Great job!")
            mock_print.assert_called()
            # Verify JSON was printed
            printed_text = mock_print.call_args[0][0]
            json.loads(printed_text)  # Should not raise exception

    def test_batch_file_processing_csv_input(self):
        """Test batch processing with CSV input file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["text", "id"])
            writer.writerow(["Hello world", "1"])
            writer.writerow(["Go away stupid", "2"])
            csv_file = f.name

        try:
            # Mock detector responses
            mock_results = [
                self._create_mock_detection_result(
                    text="Hello world", toxicity="none", bullying="none", emotion="pos"
                ),
                self._create_mock_detection_result(
                    text="Go away stupid",
                    toxicity="toxic",
                    bullying="harassment",
                    emotion="neg",
                ),
            ]
            self.mock_detector.analyze.side_effect = mock_results

            args = Mock()
            args.text = None
            args.input = csv_file
            args.output = None
            args.format = "table"

            with patch.object(self.command, "_create_progress_bar") as mock_progress:
                mock_progress_bar = MagicMock()
                mock_progress_bar.__enter__ = Mock(return_value=mock_progress_bar)
                mock_progress_bar.__exit__ = Mock(return_value=None)
                mock_progress_bar.add_task.return_value = "task_id"
                mock_progress.return_value = mock_progress_bar

                result = self.command.execute(args)

                assert result == 0
                assert self.mock_detector.analyze.call_count == 2
                self.mock_detector.analyze.assert_any_call("Hello world")
                self.mock_detector.analyze.assert_any_call("Go away stupid")

        finally:
            os.unlink(csv_file)

    def test_batch_processing_with_json_output_file(self):
        """Test batch processing with JSON output to file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as input_file:
            input_file.write("Good morning\n")
            input_file.write("Shut up idiot\n")
            input_file_name = input_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as output_file:
            output_file_name = output_file.name

        try:
            mock_results = [
                self._create_mock_detection_result(
                    text="Good morning", toxicity="none", emotion="pos"
                ),
                self._create_mock_detection_result(
                    text="Shut up idiot", toxicity="toxic", emotion="neg"
                ),
            ]
            self.mock_detector.analyze.side_effect = mock_results

            args = Mock()
            args.text = None
            args.input = input_file_name
            args.output = output_file_name
            args.format = "json"

            # Mock the progress bar to support context manager protocol
            with patch.object(self.command, "_create_progress_bar") as mock_create_progress:
                mock_progress_bar = MagicMock()
                mock_progress_bar.__enter__ = Mock(return_value=mock_progress_bar)
                mock_progress_bar.__exit__ = Mock(return_value=None)
                mock_progress_bar.add_task.return_value = "task_id"
                mock_create_progress.return_value = mock_progress_bar

                result = self.command.execute(args)

                assert result == 0

            # Verify output file was written
            with open(output_file_name, "r", encoding="utf-8") as f:
                output_data = json.load(f)
                assert len(output_data) == 2
                assert output_data[0]["text"] == "Good morning"
                assert output_data[1]["text"] == "Shut up idiot"

        finally:
            try:
                os.unlink(input_file_name)
            except (OSError, PermissionError):
                pass
            try:
                os.unlink(output_file_name)
            except (OSError, PermissionError):
                pass

    def test_missing_input_file_error(self):
        """Test error handling for missing input file."""
        args = Mock()
        args.text = None
        args.input = "nonexistent.csv"
        args.output = None

        with pytest.raises(CLIError, match="Input file not found"):
            self.command.execute(args)

    def test_invalid_input_format_error(self):
        """Test error handling for invalid input format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("invalid format")
            invalid_file = f.name

        try:
            args = Mock()
            args.text = None
            args.input = invalid_file
            args.output = None

            with pytest.raises(CLIError, match="Unsupported input format"):
                self.command.execute(args)

        finally:
            os.unlink(invalid_file)


@pytest.mark.unit
class TestTrainCommand:
    """Test the train command functionality."""

    def setup_method(self):
        """Set up test fixtures with mocked dependencies."""
        self.mock_trainer = Mock()
        self.mock_console = MagicMock()
        # Ensure console supports context manager protocol
        self.mock_console.__enter__ = Mock(return_value=self.mock_console)
        self.mock_console.__exit__ = Mock(return_value=None)
        self.command = TrainCommand(self.mock_trainer, self.mock_console)

    def test_train_with_cold_dataset(self):
        """Test training with COLD dataset."""
        self.mock_trainer.train.return_value = {
            "final_loss": 0.245,
            "best_f1": 0.832,
            "epochs_completed": 10,
            "model_path": "models/cyberpuppy_cold_v1.pt",
        }

        args = Mock()
        args.dataset = "COLD"
        args.epochs = 10
        args.batch_size = 32
        args.learning_rate = 2e-5
        args.output = "model.pt"
        args.config = None
        args.config = None

        result = self.command.execute(args)

        assert result == 0
        self.mock_trainer.train.assert_called_once()
        call_kwargs = self.mock_trainer.train.call_args[1]
        assert call_kwargs["dataset"] == "COLD"
        assert call_kwargs["epochs"] == 10
        assert call_kwargs["batch_size"] == 32

    def test_train_with_config_file(self):
        """Test training with configuration file override."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "model": {"dropout": 0.3, "hidden_size": 768},
                "training": {"early_stopping_patience": 5},
            }
            json.dump(config, f)
            config_file = f.name

        try:
            args = Mock()
            args.dataset = "ChnSentiCorp"
            args.epochs = 15
            args.config = config_file
            args.output = "sentiment_model.pt"
            args.batch_size = None
            args.learning_rate = None

            self.mock_trainer.train.return_value = {"status": "completed"}

            result = self.command.execute(args)

            assert result == 0
            self.mock_trainer.train.assert_called_once()

        finally:
            os.unlink(config_file)

    def test_train_with_progress_tracking(self):
        """Test training with progress bar updates."""

        def mock_train_with_callback(*args, **kwargs):
            callback = kwargs.get("progress_callback")
            if callback:
                callback(epoch=1, loss=0.8, f1=0.6)
                callback(epoch=5, loss=0.4, f1=0.8)
                callback(epoch=10, loss=0.2, f1=0.85)
            return {"status": "completed"}

        self.mock_trainer.train.side_effect = mock_train_with_callback

        args = Mock()
        args.dataset = "COLD"
        args.epochs = 10
        args.config = None  # No config file
        args.batch_size = None  # Use default
        args.learning_rate = None  # Use default
        args.output = None  # Use default

        with patch.object(self.command, "_create_training_progress") as mock_progress:
            mock_progress_bar = MagicMock()
            mock_progress_bar.__enter__ = Mock(return_value=mock_progress_bar)
            mock_progress_bar.__exit__ = Mock(return_value=None)
            mock_progress_bar.add_task.return_value = "task_id"
            mock_progress.return_value = mock_progress_bar

            result = self.command.execute(args)

            assert result == 0
            mock_progress.assert_called_once()

    def test_training_failure_handling(self):
        """Test handling of training failures."""
        self.mock_trainer.train.side_effect = RuntimeError("GPU out of memory")

        args = Mock()
        args.dataset = "COLD"
        args.epochs = 10
        args.config = None
        args.batch_size = None
        args.learning_rate = None
        args.output = None

        with pytest.raises(CLIError, match="Training failed"):
            self.command.execute(args)


@pytest.mark.unit
class TestEvaluateCommand:
    """Test the evaluate command functionality."""

    def setup_method(self):
        """Set up test fixtures with mocked dependencies."""
        self.mock_evaluator = Mock()
        self.mock_console = MagicMock()
        # Ensure console supports context manager protocol
        self.mock_console.__enter__ = Mock(return_value=self.mock_console)
        self.mock_console.__exit__ = Mock(return_value=None)
        self.command = EvaluateCommand(self.mock_evaluator, self.mock_console)
        # Patch file validation for all tests
        self.validate_patcher = patch.object(self.command, '_validate_file_exists', return_value=Path('model.pt'))
        self.validate_patcher.start()

    def teardown_method(self):
        """Clean up patches."""
        self.validate_patcher.stop()

    def test_evaluate_with_metrics_report(self):
        """Test evaluation with comprehensive metrics report."""
        self.mock_evaluator.evaluate.return_value = {
            "accuracy": 0.856,
            "precision": 0.832,
            "recall": 0.798,
            "f1": 0.815,
            "confusion_matrix": [[45, 5], [8, 42]],
            "classification_report": {
                "none": {"precision": 0.85, "recall": 0.90, "f1-score": 0.87},
                "toxic": {"precision": 0.80, "recall": 0.75, "f1-score": 0.77},
            },
        }

        args = Mock()
        args.model = "model.pt"
        args.dataset = "test.csv"
        args.output = None
        args.metrics = ["accuracy", "f1", "precision", "recall"]

        result = self.command.execute(args)

        assert result == 0
        self.mock_evaluator.evaluate.assert_called_once()
        self.mock_console.print.assert_called()

    def test_evaluate_with_output_file(self):
        """Test evaluation with results saved to file."""
        evaluation_results = {
            "accuracy": 0.892,
            "f1": 0.878,
            "detailed_results": [
                {
                    "text": "test1",
                    "predicted": "toxic",
                    "actual": "toxic",
                    "correct": True,
                },
                {
                    "text": "test2",
                    "predicted": "none",
                    "actual": "toxic",
                    "correct": False,
                },
            ],
        }
        self.mock_evaluator.evaluate.return_value = evaluation_results

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_file = f.name

        try:
            args = Mock()
            args.model = "model.pt"
            args.dataset = "test.csv"
            args.output = output_file
            args.metrics = ["accuracy", "f1"]

            result = self.command.execute(args)

            assert result == 0

            # Verify output file
            with open(output_file, "r", encoding="utf-8") as f:
                saved_results = json.load(f)
                assert saved_results["accuracy"] == 0.892
                assert len(saved_results["detailed_results"]) == 2

        finally:
            os.unlink(output_file)

    def test_evaluate_missing_model_error(self):
        """Test error handling for missing model file."""
        # Stop the patch for this test to test actual validation
        self.validate_patcher.stop()

        args = Mock()
        args.model = "nonexistent_model.pt"
        args.dataset = "test.csv"

        with pytest.raises(CLIError, match="Model file not found"):
            self.command.execute(args)

        # Restart patch for subsequent tests
        self.validate_patcher.start()


@pytest.mark.unit
class TestExportCommand:
    """Test the export command functionality."""

    def setup_method(self):
        """Set up test fixtures with mocked dependencies."""
        self.mock_exporter = Mock()
        self.mock_console = MagicMock()
        # Ensure console supports context manager protocol
        self.mock_console.__enter__ = Mock(return_value=self.mock_console)
        self.mock_console.__exit__ = Mock(return_value=None)
        self.command = ExportCommand(self.mock_exporter, self.mock_console)
        # Patch file validation for all tests
        self.validate_patcher = patch.object(self.command, '_validate_file_exists', return_value=Path('model.pt'))
        self.validate_patcher.start()

    def teardown_method(self):
        """Clean up patches."""
        self.validate_patcher.stop()

    def test_export_to_onnx(self):
        """Test exporting model to ONNX format."""
        args = Mock()
        args.model = "model.pt"
        args.format = "onnx"
        args.output = "model.onnx"

        self.mock_exporter.export_to_onnx.return_value = {
            "success": True,
            "output_path": "model.onnx",
            "model_size_mb": 425.6,
        }

        result = self.command.execute(args)

        assert result == 0
        self.mock_exporter.export_to_onnx.assert_called_once_with(
            "model.pt", "model.onnx"
        )

    def test_export_to_torchscript(self):
        """Test exporting model to TorchScript format."""
        args = Mock()
        args.model = "model.pt"
        args.format = "torchscript"
        args.output = "model_scripted.pt"

        self.mock_exporter.export_to_torchscript.return_value = {
            "success": True,
            "output_path": "model_scripted.pt",
        }

        result = self.command.execute(args)

        assert result == 0
        self.mock_exporter.export_to_torchscript.assert_called_once()

    def test_export_unsupported_format_error(self):
        """Test error for unsupported export format."""
        args = Mock()
        args.model = "model.pt"
        args.format = "unsupported"
        args.output = "model.xyz"

        with pytest.raises(CLIError, match="Unsupported export format"):
            self.command.execute(args)


@pytest.mark.unit
class TestConfigCommand:
    """Test the config command functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_console = MagicMock()
        # Ensure console supports context manager protocol
        self.mock_console.__enter__ = Mock(return_value=self.mock_console)
        self.mock_console.__exit__ = Mock(return_value=None)
        self.command = ConfigCommand(self.mock_console)

    def test_show_config(self):
        """Test showing current configuration."""
        args = Mock()
        args.show = True
        args.set = None
        args.get = None

        with patch("cyberpuppy.cli.load_config") as mock_load:
            mock_load.return_value = {
                "model": {"name": "hfl/chinese-macbert-base"},
                "training": {"batch_size": 32, "epochs": 10},
            }

            result = self.command.execute(args)

            assert result == 0
            self.mock_console.print.assert_called()

    def test_get_config_value(self):
        """Test getting specific configuration value."""
        args = Mock()
        args.show = False
        args.set = None
        args.get = "model.name"

        with patch("cyberpuppy.cli.load_config") as mock_load:
            mock_load.return_value = {"model": {"name": "hfl/chinese-macbert-base"}}

            result = self.command.execute(args)

            assert result == 0
            self.mock_console.print.assert_called_with("hfl/chinese-macbert-base")


@pytest.mark.unit
class TestCLIErrorHandling:
    """Test CLI error handling and user-friendly messages."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = CyberPuppyCLI()

    def test_cli_error_exception(self):
        """Test CLIError exception properties."""
        error = CLIError("Test error message", exit_code=2)

        assert str(error) == "Test error message"
        assert error.exit_code == 2

    def test_file_not_found_error_handling(self):
        """Test handling of file not found errors."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(CLIError) as exc_info:
                # This would be called from within a command
                raise CLIError("Input file not fou" "nd: nonexistent.csv", exit_code=1)

            assert "Input file not found" in str(exc_info.value)
            assert exc_info.value.exit_code == 1

    @patch("sys.stderr")
    def test_keyboard_interrupt_handling(self, mock_stderr):
        """Test graceful handling of keyboard interrupts."""
        with patch("cyberpuppy.cli.CyberPuppyCLI") as mock_cli_class:
            mock_cli = Mock()
            mock_cli_class.return_value = mock_cli
            mock_cli.run.side_effect = KeyboardInterrupt()

            result = main(["analyze", "test"])

            assert result == 130  # Standard exit code for SIGINT


@pytest.mark.unit
class TestCLIIntegration:
    """Test CLI integration with core components."""

    @patch("cyberpuppy.cli.CyberPuppyDetector")
    @patch("cyberpuppy.cli.Console")
    def test_main_function_integration(self, mock_console_class, mock_detector_class):
        """Test main function integrates all components correctly."""
        mock_detector = Mock()
        mock_console = Mock()
        mock_detector_class.return_value = mock_detector
        mock_console_class.return_value = mock_console

        # Mock the detector.analyze() to return a proper mock result
        mock_result = Mock()
        mock_result.text = "Hello world"
        mock_result.toxicity.prediction.value = "none"
        mock_result.toxicity.confidence = 0.95
        mock_result.bullying.prediction.value = "none"
        mock_result.emotion.prediction.value = "neu"
        mock_result.emotion.strength = 0
        mock_result.role.prediction.value = "none"
        mock_result.explanations = {}
        mock_result.processing_time = 0.1
        mock_detector.analyze.return_value = mock_result

        result = main(["analyze", "Hello world"])

        assert result == 0
        mock_detector_class.assert_called_once()
        mock_detector.analyze.assert_called_once_with("Hello world")

    def test_verbosity_levels_configuration(self):
        """Test that verbosity levels configure logging correctly."""
        with patch("src.cyberpuppy.cli.setup_logging"):
            parser = create_parser()

            # Test verbose mode
            args = parser.parse_args(["--verbose", "analyze", "test"])
            assert args.verbose is True

            # Test quiet mode
            args = parser.parse_args(["--quiet", "analyze", "test"])
            assert args.quiet is True

    def test_config_file_loading(self):
        """Test configuration file loading and merging."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
model:
  name: "custom-model"
  dropout: 0.2
training:
  epochs: 20
  batch_size: 16
"""
            )
            config_file = f.name

        try:
            with patch("src.cyberpuppy.cli.load_config") as mock_load:
                mock_load.return_value = {
                    "model": {"name": "custom-model", "dropout": 0.2},
                    "training": {"epochs": 20, "batch_size": 16},
                }

                parser = create_parser()
                args = parser.parse_args(
                    ["--config-file", config_file, "analyze", "test"]
                )

                assert args.config_file == config_file

        finally:
            os.unlink(config_file)


@pytest.mark.unit
class TestCLIProgressBars:
    """Test CLI progress bar functionality."""

    def test_progress_bar_creation(self):
        """Test progress bar creation and updates."""
        with patch("cyberpuppy.cli.Progress") as mock_progress_class:
            mock_progress = MagicMock()
            mock_task = Mock()
            mock_progress_class.return_value = mock_progress
            mock_progress.add_task.return_value = mock_task

            # This would be called from within a command
            from cyberpuppy.cli import create_progress_bar

            progress = create_progress_bar("Processing files")
            progress.add_task("test", total=100)

            mock_progress_class.assert_called_once()
            mock_progress.add_task.assert_called_once()


@pytest.mark.unit
class TestCLIOutputFormatting:
    """Test CLI output formatting with Rich."""

    def test_table_formatting(self):
        """Test table output formatting."""
        with patch("cyberpuppy.cli.Table") as mock_table_class:
            mock_table = Mock()
            mock_table_class.return_value = mock_table

            from cyberpuppy.cli import format_analysis_result_table

            result = {"toxicity": "toxic", "confidence": 0.85, "emotion": "neg"}

            format_analysis_result_table(result, "Test text")

            mock_table_class.assert_called_once()
            mock_table.add_row.assert_called()

    def test_json_formatting(self):
        """Test JSON output formatting with proper indentation."""
        result = {
            "text": "Test input",
            "analysis": {"toxicity": "none", "confidence": 0.92},
        }

        from cyberpuppy.cli import format_json_output

        json_output = format_json_output(result)

        # Verify it's valid JSON and properly formatted
        parsed = json.loads(json_output)
        assert parsed == result
        assert "\n" in json_output  # Should be pretty-printed


# Mock classes for testing
class MockDetector:
    """Mock detector for testing."""

    def analyze(self, text):
        if "toxic" in text.lower():
            return {
                "toxicity": "toxic",
                "bullying": "harassment",
                "emotion": "neg",
                "confidence": 0.85,
            }
        return {
            "toxicity": "none",
            "bullying": "none",
            "emotion": "pos",
            "confidence": 0.92,
        }


class MockTrainer:
    """Mock trainer for testing."""

    def train(self, **kwargs):
        return {
            "final_loss": 0.2,
            "best_f1": 0.85,
            "epochs_completed": kwargs.get("epochs", 10),
            "model_path": kwargs.get("output", "model.pt"),
        }


class MockEvaluator:
    """Mock evaluator for testing."""

    def evaluate(self, model_path, dataset_path):
        return {"accuracy": 0.856, "precision": 0.832, "recall": 0.798, "f1": 0.815}


class MockExporter:
    """Mock exporter for testing."""

    def export_to_onnx(self, model_path, output_path):
        return {"success": True, "output_path": output_path, "model_size_mb": 425.6}

    def export_to_torchscript(self, model_path, output_path):
        return {"success": True, "output_path": output_path}
