"""
Comprehensive test suite for cyberpuppy.eval.metrics module.

This test file targets 70%+ coverage for the metrics evaluation system,
covering:
- MetricResult and SessionContext dataclasses
- MetricsCalculator (classification, probability, session metrics)
- OnlineMonitor (convergence monitoring)
- PrometheusExporter (metrics export)
- CSVExporter (CSV export)
- EvaluationReport (report generation)
"""

import csv
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import directly from metrics module to avoid visualization dependencies
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the metrics module directly without going through __init__
import importlib.util

metrics_path = Path(__file__).parent.parent / "src" / "cyberpuppy" / "eval" / "metrics.py"
spec = importlib.util.spec_from_file_location("cyberpuppy.eval.metrics", metrics_path)
metrics_module = importlib.util.module_from_spec(spec)
sys.modules["cyberpuppy.eval.metrics"] = metrics_module
spec.loader.exec_module(metrics_module)

CSVExporter = metrics_module.CSVExporter
EvaluationReport = metrics_module.EvaluationReport
MetricResult = metrics_module.MetricResult
MetricsCalculator = metrics_module.MetricsCalculator
OnlineMonitor = metrics_module.OnlineMonitor
PrometheusExporter = metrics_module.PrometheusExporter
SessionContext = metrics_module.SessionContext


class TestMetricResult:
    """Test MetricResult dataclass."""

    def test_metric_result_creation(self):
        """Test MetricResult can be created."""
        result = MetricResult(name="f1_score", value=0.85, metadata={"average": "macro"})

        assert result.name == "f1_score"
        assert result.value == 0.85
        assert result.metadata == {"average": "macro"}

    def test_metric_result_to_dict(self):
        """Test MetricResult serialization."""
        result = MetricResult(name="accuracy", value=0.92, metadata={"task": "toxicity"})
        d = result.to_dict()

        assert d["name"] == "accuracy"
        assert d["value"] == 0.92
        assert d["metadata"]["task"] == "toxicity"

    def test_metric_result_default_metadata(self):
        """Test MetricResult with default empty metadata."""
        result = MetricResult(name="test", value=0.5)

        assert result.metadata == {}


class TestSessionContext:
    """Test SessionContext dataclass."""

    def test_session_context_creation(self):
        """Test SessionContext can be created."""
        session = SessionContext(session_id="session_001")

        assert session.session_id == "session_001"
        assert len(session.messages) == 0
        assert session.end_time is None

    def test_add_message(self):
        """Test adding messages to session."""
        session = SessionContext(session_id="session_001")
        msg = {"text": "Hello", "scores": {"toxicity": 0.1}}

        session.add_message(msg)

        assert len(session.messages) == 1
        assert session.messages[0] == msg
        assert session.end_time is not None

    def test_get_duration(self):
        """Test session duration calculation."""
        session = SessionContext(session_id="session_001")
        session.add_message({"text": "Message 1"})

        duration = session.get_duration()

        assert duration >= 0
        assert isinstance(duration, float)

    def test_get_duration_no_end_time(self):
        """Test duration calculation when session hasn't ended."""
        past_time = datetime.now() - timedelta(seconds=5)
        session = SessionContext(session_id="session_001", start_time=past_time)

        duration = session.get_duration()

        assert duration >= 5


class TestMetricsCalculator:
    """Test MetricsCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create MetricsCalculator instance."""
        return MetricsCalculator()

    def test_calculator_initialization(self, calculator):
        """Test MetricsCalculator initializes with label mappings."""
        assert "toxicity" in calculator.label_mapping
        assert "emotion" in calculator.label_mapping
        assert "bullying" in calculator.label_mapping
        assert "role" in calculator.label_mapping

    def test_calculate_classification_metrics_basic(self, calculator):
        """Test basic classification metrics calculation."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 1]

        metrics = calculator.calculate_classification_metrics(y_true, y_pred, "toxicity", "macro")

        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "confusion_matrix" in metrics

    def test_calculate_classification_metrics_accuracy(self, calculator):
        """Test accuracy calculation."""
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 2]

        metrics = calculator.calculate_classification_metrics(y_true, y_pred)

        assert metrics["accuracy"].value == 1.0

    def test_calculate_classification_metrics_with_labels(self, calculator):
        """Test per-class metrics with label mapping."""
        y_true = ["none", "toxic", "severe", "none", "toxic", "severe"]
        y_pred = ["none", "toxic", "severe", "none", "toxic", "toxic"]

        metrics = calculator.calculate_classification_metrics(y_true, y_pred, "toxicity")

        assert "f1_none" in metrics
        assert "f1_toxic" in metrics
        assert "f1_severe" in metrics

    def test_calculate_classification_metrics_different_averages(self, calculator):
        """Test different averaging methods."""
        y_true = [0, 1, 2] * 10
        y_pred = [0, 1, 2] * 10

        metrics_macro = calculator.calculate_classification_metrics(y_true, y_pred, "toxicity", "macro")
        metrics_micro = calculator.calculate_classification_metrics(y_true, y_pred, "toxicity", "micro")

        assert metrics_macro["f1_score"].metadata["average"] == "macro"
        assert metrics_micro["f1_score"].metadata["average"] == "micro"

    def test_calculate_probability_metrics_binary(self, calculator):
        """Test probability metrics for binary classification."""
        y_true = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
        y_prob = np.array([[0.3, 0.7], [0.8, 0.2], [0.2, 0.8], [0.9, 0.1]])

        metrics = calculator.calculate_probability_metrics(y_true, y_prob, "toxicity")

        assert "auc_roc" in metrics or "aucpr" in metrics

    def test_calculate_probability_metrics_multiclass(self, calculator):
        """Test probability metrics for multiclass."""
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
        y_prob = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.6, 0.3, 0.1]])

        metrics = calculator.calculate_probability_metrics(y_true, y_prob, "toxicity")

        assert isinstance(metrics, dict)

    def test_calculate_probability_metrics_error_handling(self, calculator):
        """Test probability metrics handles errors gracefully."""
        y_true = np.array([0, 0])
        y_prob = np.array([[0.5, 0.5], [0.5, 0.5]])

        metrics = calculator.calculate_probability_metrics(y_true, y_prob)

        assert isinstance(metrics, dict)

    def test_calculate_session_metrics_empty(self, calculator):
        """Test session metrics with empty session list."""
        sessions = []

        metrics = calculator.calculate_session_metrics(sessions)

        assert metrics == {}

    def test_calculate_session_metrics_basic(self, calculator):
        """Test basic session metrics."""
        session1 = SessionContext(session_id="s1")
        session1.add_message({"text": "msg1"})
        session1.add_message({"text": "msg2"})

        session2 = SessionContext(session_id="s2")
        session2.add_message({"text": "msg3"})

        metrics = calculator.calculate_session_metrics([session1, session2])

        assert "total_sessions" in metrics
        assert "avg_messages_per_session" in metrics
        assert "avg_session_duration" in metrics

    def test_calculate_session_metrics_escalation(self, calculator):
        """Test session escalation detection."""
        session = SessionContext(session_id="s1")
        session.add_message({"text": "msg1", "scores": {"toxicity": 0.2}})
        session.add_message({"text": "msg2", "scores": {"toxicity": 0.8}})

        metrics = calculator.calculate_session_metrics([session], "toxicity")

        assert "escalation_rate" in metrics
        assert metrics["escalation_rate"].value > 0

    def test_calculate_session_metrics_de_escalation(self, calculator):
        """Test session de-escalation detection."""
        session = SessionContext(session_id="s1")
        session.add_message({"text": "msg1", "scores": {"toxicity": 0.8}})
        session.add_message({"text": "msg2", "scores": {"toxicity": 0.2}})

        metrics = calculator.calculate_session_metrics([session], "toxicity")

        assert "de_escalation_rate" in metrics
        assert metrics["de_escalation_rate"].value > 0

    def test_calculate_session_metrics_intervention(self, calculator):
        """Test intervention success rate calculation."""
        session = SessionContext(session_id="s1")
        session.add_message({"text": "msg1", "scores": {"toxicity": 0.8}, "intervention": True})
        session.add_message({"text": "msg2", "scores": {"toxicity": 0.2}})

        metrics = calculator.calculate_session_metrics([session], "toxicity")

        assert "intervention_success_rate" in metrics
        assert metrics["intervention_success_rate"].value == 1.0


class TestOnlineMonitor:
    """Test OnlineMonitor class."""

    def test_monitor_initialization(self):
        """Test OnlineMonitor initializes correctly."""
        monitor = OnlineMonitor(window_size=50, checkpoint_interval=500)

        assert monitor.window_size == 50
        assert monitor.checkpoint_interval == 500
        assert monitor.total_steps == 0

    def test_update_basic(self):
        """Test basic monitor update."""
        monitor = OnlineMonitor()

        stats = monitor.update(loss=0.5, accuracy=0.8, f1_score=0.75)

        assert stats["step"] == 1
        assert stats["loss"] == 0.5
        assert stats["accuracy"] == 0.8
        assert stats["f1_score"] == 0.75

    def test_update_moving_averages(self):
        """Test moving average calculation."""
        monitor = OnlineMonitor(window_size=3)

        monitor.update(1.0, 0.5, 0.5)
        monitor.update(0.5, 0.7, 0.7)
        stats = monitor.update(0.25, 0.9, 0.9)

        assert stats["loss_avg"] < 1.0
        assert stats["accuracy_avg"] > 0.5

    def test_convergence_detection(self):
        """Test convergence detection."""
        monitor = OnlineMonitor()
        monitor.convergence_patience = 3

        # Use same loss value to trigger no improvement
        monitor.update(1.0, 0.5, 0.5)
        monitor.update(1.0, 0.51, 0.51)
        monitor.update(1.0, 0.52, 0.52)
        stats = monitor.update(1.0, 0.53, 0.53)

        assert stats["converged"]

    def test_no_convergence(self):
        """Test when not converged."""
        monitor = OnlineMonitor()

        monitor.update(1.0, 0.5, 0.5)
        stats = monitor.update(0.5, 0.7, 0.7)

        assert not stats["converged"]

    def test_checkpoint_saving(self):
        """Test history saving at checkpoints."""
        monitor = OnlineMonitor(checkpoint_interval=2)

        monitor.update(1.0, 0.5, 0.5, 0.001)
        monitor.update(0.9, 0.6, 0.6, 0.001)

        assert len(monitor.history["step"]) == 1
        assert len(monitor.history["loss"]) == 1

    def test_get_summary(self):
        """Test summary generation."""
        monitor = OnlineMonitor()

        monitor.update(1.0, 0.5, 0.5)
        monitor.update(0.8, 0.6, 0.6)

        summary = monitor.get_summary()

        assert "total_steps" in summary
        assert "best_loss" in summary
        assert "current_loss_avg" in summary

    def test_get_summary_empty(self):
        """Test summary with no data."""
        monitor = OnlineMonitor()

        summary = monitor.get_summary()

        assert summary == {}

    def test_export_history(self):
        """Test history export."""
        monitor = OnlineMonitor(checkpoint_interval=1)

        monitor.update(1.0, 0.5, 0.5, 0.001)
        monitor.update(0.9, 0.6, 0.6, 0.001)

        history = monitor.export_history()

        assert "step" in history
        assert "loss" in history
        assert len(history["step"]) == 2


class TestPrometheusExporter:
    """Test PrometheusExporter class."""

    def test_prometheus_exporter_initialization(self):
        """Test PrometheusExporter initializes correctly."""
        exporter = PrometheusExporter(job_name="test_job", instance="test:8000")

        assert exporter.job_name == "test_job"
        assert exporter.instance == "test:8000"

    def test_update_metric(self):
        """Test updating a metric."""
        exporter = PrometheusExporter()

        exporter.update_metric("accuracy", 0.85, {"model": "baseline"})

        assert len(exporter.metrics) == 1

    def test_update_metric_without_labels(self):
        """Test updating metric without labels."""
        exporter = PrometheusExporter()

        exporter.update_metric("loss", 0.5)

        assert "loss" in exporter.metrics

    def test_generate_key(self):
        """Test unique key generation."""
        exporter = PrometheusExporter()

        key1 = exporter._generate_key("metric", {"a": "1", "b": "2"})
        key2 = exporter._generate_key("metric", {"b": "2", "a": "1"})

        assert key1 == key2

    def test_generate_key_no_labels(self):
        """Test key generation without labels."""
        exporter = PrometheusExporter()

        key = exporter._generate_key("metric", None)

        assert key == "metric"

    def test_export_format(self):
        """Test Prometheus export format."""
        exporter = PrometheusExporter(job_name="test", instance="localhost:8000")
        exporter.update_metric("f1_score", 0.78, {"task": "toxicity"})

        # Mock the instance_name attribute that export() uses
        exporter.instance_name = exporter.instance

        output = exporter.export()

        assert "# HELP" in output
        assert "# TYPE" in output
        assert "f1_score" in output

    def test_push_to_gateway_success(self):
        """Test successful push to gateway."""
        exporter = PrometheusExporter(job_name="test", instance="localhost:8000")
        exporter.instance_name = exporter.instance
        exporter.update_metric("test_metric", 1.0)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("requests.post", return_value=mock_response):
            result = exporter.push_to_gateway("localhost:9091")

            assert result is True

    def test_push_to_gateway_failure(self):
        """Test failed push to gateway."""
        exporter = PrometheusExporter()
        exporter.update_metric("test_metric", 1.0)

        with patch("requests.post", side_effect=Exception("Network error")):
            result = exporter.push_to_gateway("localhost:9091")

            assert result is False


class TestCSVExporter:
    """Test CSVExporter class."""

    def test_csv_exporter_initialization(self):
        """Test CSVExporter initializes and creates directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CSVExporter(output_dir=tmpdir)

            assert exporter.output_dir == Path(tmpdir)
            assert exporter.output_dir.exists()

    def test_export_metrics(self):
        """Test exporting metrics to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CSVExporter(output_dir=tmpdir)

            metrics = {
                "acc": MetricResult("accuracy", 0.85, {"task": "test"}),
                "f1": MetricResult("f1_score", 0.80, {"task": "test"}),
            }

            filepath = exporter.export_metrics(metrics, "test_metrics.csv")

            assert Path(filepath).exists()

            with open(filepath, "r") as f:
                content = f.read()
                assert "accuracy" in content
                assert "0.85" in content

    def test_export_metrics_auto_filename(self):
        """Test auto-generating filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CSVExporter(output_dir=tmpdir)

            metrics = {"test": MetricResult("test", 1.0)}

            filepath = exporter.export_metrics(metrics)

            assert "metrics_" in filepath
            assert filepath.endswith(".csv")

    def test_export_history(self):
        """Test exporting history to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CSVExporter(output_dir=tmpdir)

            history = {"step": [1, 2, 3], "loss": [1.0, 0.8, 0.6], "accuracy": [0.5, 0.6, 0.7]}

            filepath = exporter.export_history(history, "test_history.csv")

            assert Path(filepath).exists()

            with open(filepath, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 3
                assert rows[0]["step"] == "1"

    def test_export_history_auto_filename(self):
        """Test auto-generating history filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CSVExporter(output_dir=tmpdir)

            history = {"step": [1, 2], "loss": [1.0, 0.8]}

            filepath = exporter.export_history(history)

            assert "history_" in filepath
            assert filepath.endswith(".csv")


class TestEvaluationReport:
    """Test EvaluationReport class."""

    def test_report_initialization(self):
        """Test EvaluationReport initializes correctly."""
        report = EvaluationReport()

        assert isinstance(report.calculator, MetricsCalculator)
        assert len(report.sessions) == 0
        assert len(report.results) == 0

    def test_add_predictions(self):
        """Test adding predictions to report."""
        report = EvaluationReport()

        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 1]

        report.add_predictions(y_true, y_pred, task_name="toxicity")

        assert "toxicity_classification" in report.results

    def test_add_predictions_with_probabilities(self):
        """Test adding predictions with probability scores."""
        report = EvaluationReport()

        y_true = np.array([0, 1, 2])
        y_pred = [0, 1, 2]
        y_prob = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

        report.add_predictions(y_true, y_pred, y_prob, "toxicity")

        assert "toxicity_classification" in report.results
        assert "toxicity_probability" in report.results

    def test_add_session(self):
        """Test adding session to report."""
        report = EvaluationReport()

        session = SessionContext(session_id="s1")
        session.add_message({"text": "test"})

        report.add_session(session)

        assert len(report.sessions) == 1

    def test_generate_report_structure(self):
        """Test report structure."""
        report = EvaluationReport()

        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]

        report.add_predictions(y_true, y_pred)

        full_report = report.generate_report()

        assert "timestamp" in full_report
        assert "metrics" in full_report
        assert "session_metrics" in full_report
        assert "summary" in full_report

    def test_generate_report_with_sessions(self):
        """Test report generation with session data."""
        report = EvaluationReport()

        session = SessionContext(session_id="s1")
        session.add_message({"text": "msg", "scores": {"toxicity": 0.1}})

        report.add_session(session)

        full_report = report.generate_report()

        assert len(full_report["session_metrics"]) > 0

    def test_generate_summary(self):
        """Test summary generation."""
        report = EvaluationReport()

        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]

        report.add_predictions(y_true, y_pred, task_name="toxicity")

        full_report = report.generate_report()

        assert "total_evaluations" in full_report["summary"]
        assert "main_metrics" in full_report["summary"]

    def test_save_report(self):
        """Test saving report to file."""
        report = EvaluationReport()

        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]

        report.add_predictions(y_true, y_pred)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            report.save_report(filepath)

            assert Path(filepath).exists()
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestIntegration:
    """Integration tests for metrics system."""

    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        report = EvaluationReport()

        y_true = [0, 1, 2, 0, 1, 2] * 5
        y_pred = [0, 1, 2, 0, 1, 1] * 5
        y_prob = np.random.rand(30, 3)

        report.add_predictions(y_true, y_pred, y_prob, "toxicity")

        session1 = SessionContext(session_id="s1")
        session1.add_message({"text": "msg1", "scores": {"toxicity": 0.2}})
        session1.add_message({"text": "msg2", "scores": {"toxicity": 0.8}})

        report.add_session(session1)

        full_report = report.generate_report()

        assert len(full_report["metrics"]) > 0
        assert len(full_report["session_metrics"]) > 0

    def test_monitor_to_csv_export(self):
        """Test exporting monitor history to CSV."""
        monitor = OnlineMonitor(checkpoint_interval=1)

        for i in range(5):
            monitor.update(1.0 / (i + 1), 0.5 + i * 0.1, 0.5 + i * 0.1, 0.001)

        history = monitor.export_history()

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CSVExporter(output_dir=tmpdir)
            filepath = exporter.export_history(history)

            assert Path(filepath).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.cyberpuppy.eval.metrics", "--cov-report=term-missing"])