#!/usr/bin/env python3
"""
Comprehensive unit tests for evaluation metrics module
Testing coverage for cyberpuppy.eval.metrics module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile
import shutil

# Import the module under test
from cyberpuppy.eval.metrics import (
    MetricResult,
    SessionContext,
    EvaluationContext,
    MultiTaskMetrics,
    SessionLevelEvaluator,
    RealTimeMonitor,
    ConvergenceTracker,
    PerformanceProfiler,
    MetricAggregator,
    ExportManager,
    compute_classification_metrics,
    compute_session_level_f1,
    evaluate_multilabel_classification,
    create_evaluation_report
)


class TestMetricResult(unittest.TestCase):
    """Test MetricResult dataclass"""

    def test_metric_result_creation(self):
        """Test basic MetricResult creation"""
        result = MetricResult(
            name="accuracy",
            value=0.85,
            metadata={"threshold": 0.5, "samples": 1000}
        )

        self.assertEqual(result.name, "accuracy")
        self.assertEqual(result.value, 0.85)
        self.assertEqual(result.metadata["threshold"], 0.5)
        self.assertEqual(result.metadata["samples"], 1000)

    def test_metric_result_to_dict(self):
        """Test MetricResult to_dict conversion"""
        result = MetricResult(
            name="f1_score",
            value=0.78,
            metadata={"class": "toxic"}
        )

        result_dict = result.to_dict()

        expected_dict = {
            "name": "f1_score",
            "value": 0.78,
            "metadata": {"class": "toxic"}
        }

        self.assertEqual(result_dict, expected_dict)

    def test_metric_result_empty_metadata(self):
        """Test MetricResult with empty metadata"""
        result = MetricResult(name="precision", value=0.92)

        self.assertEqual(result.metadata, {})
        self.assertIsInstance(result.to_dict()["metadata"], dict)

    def test_metric_result_validation(self):
        """Test MetricResult with invalid values"""
        # Test with negative value (which might be valid for some metrics)
        result = MetricResult(name="loss", value=-0.5)
        self.assertEqual(result.value, -0.5)

        # Test with very large value
        result = MetricResult(name="count", value=1e6)
        self.assertEqual(result.value, 1e6)


class TestSessionContext(unittest.TestCase):
    """Test SessionContext dataclass"""

    def test_session_context_creation(self):
        """Test basic SessionContext creation"""
        session = SessionContext(session_id="test_session_001")

        self.assertEqual(session.session_id, "test_session_001")
        self.assertEqual(len(session.messages), 0)
        self.assertIsNotNone(session.start_time)
        self.assertIsNone(session.end_time)

    def test_add_message_to_session(self):
        """Test adding messages to session"""
        session = SessionContext(session_id="test_session_001")

        message1 = {"te"
            "xt": 
        message2 = {"text": "你好", "prediction": 1, "timestamp": datetime.now()}

        session.add_message(message1)
        session.add_message(message2)

        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.messages[0]["text"], "Hello")
        self.assertEqual(session.messages[1]["text"], "你好")
        self.assertIsNotNone(session.end_time)

    def test_session_duration_calculation(self):
        """Test session duration calculation"""
        start_time = datetime.now()
        session = SessionContext(session_id="test_session_001")
        session.start_time = start_time

        # Add a message to set end_time
        session.add_message({"text": "test", "prediction": 0})

        duration = session.end_time - session.start_time
        self.assertIsInstance(duration, timedelta)
        self.assertGreaterEqual(duration.total_seconds(), 0)

    def test_session_with_custom_start_time(self):
        """Test session with custom start time"""
        custom_start = datetime.now() - timedelta(hours=1)
        session = SessionContext(session_id="test_ses"
            "sion_001", start_time=custom_start)

        self.assertEqual(session.start_time, custom_start)


class TestEvaluationContext(unittest.TestCase):
    """Test EvaluationContext class"""

    def test_evaluation_context_creation(self):
        """Test EvaluationContext initialization"""
        context = EvaluationContext(
            dataset_name="test_dataset",
            model_name="test_model",
            config_path="config/test.yaml"
        )

        self.assertEqual(context.dataset_name, "test_dataset")
        self.assertEqual(context.model_name, "test_model")
        self.assertEqual(context.config_path, "config/test.yaml")
        self.assertIsNotNone(context.created_at)

    def test_evaluation_context_with_metadata(self):
        """Test EvaluationContext with metadata"""
        metadata = {"batch_size": 32, "learning_rate": 0.001}
        context = EvaluationContext(
            dataset_name="test_dataset",
            model_name="test_model",
            metadata=metadata
        )

        self.assertEqual(context.metadata["batch_size"], 32)
        self.assertEqual(context.metadata["learning_rate"], 0.001)

    def test_evaluation_context_to_dict(self):
        """Test EvaluationContext serialization"""
        context = EvaluationContext(
            dataset_name="test_dataset",
            model_name="test_model"
        )

        context_dict = context.to_dict()

        self.assertIn("dataset_name", context_dict)
        self.assertIn("model_name", context_dict)
        self.assertIn("created_at", context_dict)
        self.assertIn("metadata", context_dict)


class TestMultiTaskMetrics(unittest.TestCase):
    """Test MultiTaskMetrics class"""

    def setUp(self):
        """Set up test fixtures"""
        self.y_true_toxicity = np.array([0, 1, 1, 0, 1])
        self.y_pred_toxicity = np.array([0, 1, 0, 0, 1])
        self.y_true_emotion = np.array([0, 1, 2, 0, 1])
        self.y_pred_emotion = np.array([0, 1, 1, 0, 2])

    def test_multitask_metrics_creation(self):
        """Test MultiTaskMetrics initialization"""
        metrics = MultiTaskMetrics()

        self.assertIsInstance(metrics.task_metrics, dict)
        self.assertEqual(len(metrics.task_metrics), 0)

    def test_add_task_results(self):
        """Test adding task-specific results"""
        metrics = MultiTaskMetrics()

        # Add toxicity results
        metrics.add_task_results(
            task_name="toxicity",
            y_true=self.y_true_toxicity,
            y_pred=self.y_pred_toxicity
        )

        self.assertIn("toxicity", metrics.task_metrics)
        self.assertIn("accuracy", metrics.task_metrics["toxicity"])

    def test_add_task_results_with_probabilities(self):
        """Test adding task results with probability scores"""
        metrics = MultiTaskMetrics()

        y_proba = np.array(
            [[0.8,
            0.2],
            [0.3,
            0.7],
            [0.9,
            0.1],
            [0.6,
            0.4],
            [0.2,
            0.8]]
        )

        metrics.add_task_results(
            task_name="toxicity",
            y_true=self.y_true_toxicity,
            y_pred=self.y_pred_toxicity,
            y_proba=y_proba
        )

        self.assertIn("toxicity", metrics.task_metrics)
        self.assertIn("auc_roc", metrics.task_metrics["toxicity"])
        self.assertIn("auc_pr", metrics.task_metrics["toxicity"])

    def test_compute_overall_metrics(self):
        """Test overall metrics computation"""
        metrics = MultiTaskMetrics()

        # Add multiple tasks
        metrics.add_task_results(
            task_name="toxicity",
            y_true=self.y_true_toxicity,
            y_pred=self.y_pred_toxicity
        )

        metrics.add_task_results(
            task_name="emotion",
            y_true=self.y_true_emotion,
            y_pred=self.y_pred_emotion
        )

        overall = metrics.compute_overall_metrics()

        self.assertIn("macro_f1", overall)
        self.assertIn("micro_f1", overall)
        self.assertIn("task_count", overall)

    def test_get_summary_report(self):
        """Test summary report generation"""
        metrics = MultiTaskMetrics()

        metrics.add_task_results(
            task_name="toxicity",
            y_true=self.y_true_toxicity,
            y_pred=self.y_pred_toxicity
        )

        summary = metrics.get_summary_report()

        self.assertIn("toxicity", summary)
        self.assertIn("overall", summary)
        self.assertIsInstance(summary["toxicity"]["f1_score"], float)


class TestSessionLevelEvaluator(unittest.TestCase):
    """Test SessionLevelEvaluator class"""

    def setUp(self):
        """Set up test fixtures"""
        self.sample_sessions = [
            SessionContext("session_1"),
            SessionContext("session_2"),
            SessionContext("session_3")
        ]

        # Add messages to sessions
        for i, session in enumerate(self.sample_sessions):
            for j in range(3):
                message = {
                    "text": f"Message {j} in session {i}",
                    "true_label": j % 2,
                    "predicted_label": (j + i) % 2,
                    "confidence": 0.5 + (j * 0.1)
                }
                session.add_message(message)

    def test_session_evaluator_creation(self):
        """Test SessionLevelEvaluator initialization"""
        evaluator = SessionLevelEvaluator()

        self.assertIsInstance(evaluator.sessions, list)
        self.assertEqual(len(evaluator.sessions), 0)

    def test_add_session(self):
        """Test adding sessions to evaluator"""
        evaluator = SessionLevelEvaluator()

        for session in self.sample_sessions:
            evaluator.add_session(session)

        self.assertEqual(len(evaluator.sessions), 3)

    def test_compute_session_level_f1(self):
        """Test session-level F1 computation"""
        evaluator = SessionLevelEvaluator()

        for session in self.sample_sessions:
            evaluator.add_session(session)

        session_f1 = evaluator.compute_session_level_f1()

        self.assertIsInstance(session_f1, float)
        self.assertGreaterEqual(session_f1, 0.0)
        self.assertLessEqual(session_f1, 1.0)

    def test_analyze_session_patterns(self):
        """Test session pattern analysis"""
        evaluator = SessionLevelEvaluator()

        for session in self.sample_sessions:
            evaluator.add_session(session)

        patterns = evaluator.analyze_session_patterns()

        self.assertIn("avg_messages_per_session", patterns)
        self.assertIn("avg_session_duration", patterns)
        self.assertIn("escalation_patterns", patterns)

    def test_compute_escalation_metrics(self):
        """Test escalation metrics computation"""
        evaluator = SessionLevelEvaluator()

        # Create a session with escalation pattern
        escalation_session = SessionContext("escalation_session")
        escalation_messages = [
            {"text": "Start of conversation", "toxicity": "none", "timestamp": 1},
            {"text": "Getting a bit annoyed", "toxicity": "toxic", "timestamp": 2},
            {"text": "This is severe escalation", "toxicity": "severe", "timestamp": 3},
        ]

        for msg in escalation_messages:
            escalation_session.add_message(msg)

        evaluator.add_session(escalation_session)

        escalation_metrics = evaluator.compute_escalation_metrics()

        self.assertIn("escalation_rate", escalation_metrics)
        self.assertIn("peak_intensity", escalation_metrics)


class TestRealTimeMonitor(unittest.TestCase):
    """Test RealTimeMonitor class"""

    def test_real_time_monitor_creation(self):
        """Test RealTimeMonitor initialization"""
        monitor = RealTimeMonitor(window_size=100)

        self.assertEqual(monitor.window_size, 100)
        self.assertIsInstance(monitor.recent_predictions, deque)
        self.assertEqual(len(monitor.recent_predictions), 0)

    def test_add_prediction(self):
        """Test adding predictions to monitor"""
        monitor = RealTimeMonitor(window_size=3)

        predictions = [
            {"text": "test1", "prediction": 0, "confidence": 0.8},
            {"text": "test2", "prediction": 1, "confidence": 0.9},
            {"text": "test3", "prediction": 0, "confidence": 0.7},
            {"te"
                "xt": 
        ]

        for pred in predictions:
            monitor.add_prediction(pred)

        self.assertEqual(len(monitor.recent_predictions), 3)
        self.assertEqual(monitor.recent_predictions[0]["text"], "test2")

    def test_compute_rolling_metrics(self):
        """Test rolling metrics computation"""
        monitor = RealTimeMonitor(window_size=10)

        # Add some predictions
        for i in range(5):
            prediction = {
                "prediction": i % 2,
                "true_label": (i + 1) % 2,
                "confidence": 0.5 + (i * 0.1)
            }
            monitor.add_prediction(prediction)

        rolling_metrics = monitor.compute_rolling_metrics()

        self.assertIn("rolling_accuracy", rolling_metrics)
        self.assertIn("rolling_f1", rolling_metrics)
        self.assertIn("avg_confidence", rolling_metrics)
        self.assertIn("prediction_count", rolling_metrics)

    def test_detect_anomalies(self):
        """Test anomaly detection"""
        monitor = RealTimeMonitor(window_size=10)

        # Add normal predictions
        for i in range(8):
            prediction = {"confidence": 0.8 + (i * 0.01)}
            monitor.add_prediction(prediction)

        # Add anomalous predictions
        monitor.add_prediction({"confidence": 0.1})  # Very low confidence
        monitor.add_prediction({"confidence": 0.05})  # Very low confidence

        anomalies = monitor.detect_anomalies()

        self.assertIn("low_confidence_rate", anomalies)
        self.assertIn("anomalous_predictions", anomalies)

    def test_get_performance_summary(self):
        """Test performance summary generation"""
        monitor = RealTimeMonitor(window_size=5)

        # Add predictions with known outcomes
        predictions = [
            {"prediction": 0, "true_label": 0, "confidence": 0.8},
            {"prediction": 1, "true_label": 1, "confidence": 0.9},
            {"prediction": 0, "true_label": 1, "confidence": 0.6},
            {"prediction": 1, "true_label": 0, "confidence": 0.7}
        ]

        for pred in predictions:
            monitor.add_prediction(pred)

        summary = monitor.get_performance_summary()

        self.assertIn("total_predictions", summary)
        self.assertIn("window_size", summary)
        self.assertIn("metrics", summary)


class TestConvergenceTracker(unittest.TestCase):
    """Test ConvergenceTracker class"""

    def test_convergence_tracker_creation(self):
        """Test ConvergenceTracker initialization"""
        tracker = ConvergenceTracker(patience=5, min_delta=0.001)

        self.assertEqual(tracker.patience, 5)
        self.assertEqual(tracker.min_delta, 0.001)
        self.assertIsInstance(tracker.metric_history, list)

    def test_add_metric_value(self):
        """Test adding metric values"""
        tracker = ConvergenceTracker(patience=3)

        values = [0.5, 0.6, 0.65, 0.63, 0.64, 0.62]

        for value in values:
            tracker.add_metric(value)

        self.assertEqual(len(tracker.metric_history), 6)
        self.assertEqual(tracker.metric_history[-1], 0.62)

    def test_check_convergence(self):
        """Test convergence detection"""
        tracker = ConvergenceTracker(patience=3, min_delta=0.01)

        # Add improving values
        values = [0.5, 0.6, 0.7, 0.8]
        for value in values:
            tracker.add_metric(value)
            is_converged = tracker.check_convergence()
            self.assertFalse(is_converged)

        # Add non-improving values
        non_improving_values = [0.79, 0.78, 0.77]
        for value in non_improving_values:
            tracker.add_metric(value)

        # Should detect convergence after patience is exceeded
        is_converged = tracker.check_convergence()
        self.assertTrue(is_converged)

    def test_early_stopping_recommendation(self):
        """Test early stopping recommendation"""
        tracker = ConvergenceTracker(patience=2, min_delta=0.05)

        # Add values that don't improve significantly
        values = [0.5, 0.51, 0.52, 0.515, 0.513]

        for i, value in enumerate(values):
            tracker.add_metric(value)
            should_stop = tracker.should_early_stop()

            if i >= 4:  # After patience is exceeded
                self.assertTrue(should_stop)
            else:
                self.assertFalse(should_stop)

    def test_get_convergence_report(self):
        """Test convergence report generation"""
        tracker = ConvergenceTracker(patience=3)

        values = [0.1, 0.3, 0.5, 0.6, 0.65, 0.63]
        for value in values:
            tracker.add_metric(value)

        report = tracker.get_convergence_report()

        self.assertIn("total_epochs", report)
        self.assertIn("best_metric", report)
        self.assertIn("current_metric", report)
        self.assertIn("is_converged", report)
        self.assertIn("epochs_since_improvement", report)


class TestPerformanceProfiler(unittest.TestCase):
    """Test PerformanceProfiler class"""

    def test_profiler_creation(self):
        """Test PerformanceProfiler initialization"""
        profiler = PerformanceProfiler()

        self.assertIsInstance(profiler.timing_data, dict)
        self.assertIsInstance(profiler.memory_data, dict)

    @patch('time.time')
    def test_timing_context_manager(self, mock_time):
        """Test timing context manager"""
        mock_time.side_effect = [1000.0, 1005.0]  # 5 second duration

        profiler = PerformanceProfiler()

        with profiler.time_operation("test_operation"):
            pass  # Simulate work

        self.assertIn("test_operation", profiler.timing_data)
        self.assertEqual(profiler.timing_data["test_operation"][-1], 5.0)

    @patch('psutil.Process')
    def test_memory_profiling(self, mock_process_class):
        """Test memory profiling"""
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100 MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process

        profiler = PerformanceProfiler()

        profiler.record_memory_usage("test_operation")

        self.assertIn("test_operation", profiler.memory_data)

    def test_get_performance_report(self):
        """Test performance report generation"""
        profiler = PerformanceProfiler()

        # Add some timing data manually
        profiler.timing_data["operation1"] = [1.0, 1.2, 0.8]
        profiler.timing_data["operation2"] = [2.0, 2.5, 1.8]

        report = profiler.get_performance_report()

        self.assertIn("timing_stats", report)
        self.assertIn("operation1", report["timing_stats"])
        self.assertIn("operation2", report["timing_stats"])


class TestUtilityFunctions(unittest.TestCase):
    """Test standalone utility functions"""

    def test_compute_classification_metrics(self):
        """Test compute_classification_metrics function"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        metrics = compute_classification_metrics(y_true, y_pred)

        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)

        # Check value ranges
        for metric_name, metric_value in metrics.items():
            if metric_name != "confusion_matrix":
                self.assertGreaterEqual(metric_value, 0.0)
                self.assertLessEqual(metric_value, 1.0)

    def test_compute_classification_metrics_with_probabilities(self):
        """Test compute_classification_metrics with probability scores"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array(
            [[0.8,
            0.2],
            [0.3,
            0.7],
            [0.9,
            0.1],
            [0.6,
            0.4],
            [0.2,
            0.8]]
        )

        metrics = compute_classification_metrics(y_true, y_pred, y_proba)

        self.assertIn("auc_roc", metrics)
        self.assertIn("auc_pr", metrics)

    def test_compute_session_level_f1(self):
        """Test compute_session_level_f1 function"""
        session_predictions = [
            {"sessi"
                "on_id": 
            {"sessi"
                "on_id": 
            {"sessi"
                "on_id": 
        ]

        session_f1 = compute_session_level_f1(session_predictions)

        self.assertIsInstance(session_f1, float)
        self.assertGreaterEqual(session_f1, 0.0)
        self.assertLessEqual(session_f1, 1.0)

    def test_evaluate_multilabel_classification(self):
        """Test evaluate_multilabel_classification function"""
        # Multi-label scenario
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
        y_pred = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 1, 1]])

        metrics = evaluate_multilabel_classification(y_true, y_pred)

        self.assertIn("hamming_loss", metrics)
        self.assertIn("subset_accuracy", metrics)
        self.assertIn("micro_f1", metrics)
        self.assertIn("macro_f1", metrics)

    def test_create_evaluation_report(self):
        """Test create_evaluation_report function"""
        evaluation_data = {
            "model_name": "test_model",
            "dataset": "test_dataset",
            "metrics": {
                "accuracy": 0.85,
                "f1_score": 0.78,
                "precision": 0.82,
                "recall": 0.74
            },
            "confusion_matrix": [[80, 10], [15, 95]]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "test_report.json"

            result_path = create_evaluation_report(
                evaluation_data=evaluation_data,
                output_path=report_path
            )

            self.assertEqual(result_path, report_path)
            self.assertTrue(report_path.exists())

            # Verify report content
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)

            self.assertEqual(report_data["model_name"], "test_model")
            self.assertIn("generated_at", report_data)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def test_empty_predictions(self):
        """Test handling of empty prediction arrays"""
        y_true = np.array([])
        y_pred = np.array([])

        # Should handle empty arrays gracefully
        try:
            metrics = compute_classification_metrics(y_true, y_pred)
            # If we get here, it handled empty arrays
            self.assertIsInstance(metrics, dict)
        except ValueError:
            # Expected behavior for empty arrays
            pass

    def test_mismatched_array_lengths(self):
        """Test handling of mismatched array lengths"""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])  # Different length

        with self.assertRaises(ValueError):
            compute_classification_metrics(y_true, y_pred)

    def test_single_class_predictions(self):
        """Test handling of single-class scenarios"""
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1])

        metrics = compute_classification_metrics(y_true, y_pred)

        self.assertEqual(metrics["accuracy"], 1.0)
        # F1 score might be undefined for single class
        self.assertIn("f1_score", metrics)

    def test_convergence_tracker_edge_cases(self):
        """Test ConvergenceTracker edge cases"""
        tracker = ConvergenceTracker(patience=3)

        # Test with single value
        tracker.add_metric(0.5)
        self.assertFalse(tracker.check_convergence())

        # Test with identical values
        identical_values = [0.5] * 5
        for value in identical_values:
            tracker.add_metric(value)

        # Should not converge if values are identical (no improvement)
        is_converged = tracker.check_convergence()
        self.assertTrue(is_converged)  # Should converge due to no improvement


class TestIntegrationScenarios(unittest.TestCase):
    """Test realistic integration scenarios"""

    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow"""
        # Simulate a complete evaluation process
        multitask_metrics = MultiTaskMetrics()

        # Add results for multiple tasks
        tasks = ["toxicity", "emotion", "bullying"]
        for task in tasks:
            y_true = np.random.randint(0, 2, 100)
            y_pred = np.random.randint(0, 2, 100)
            y_proba = np.random.random((100, 2))

            multitask_metrics.add_task_results(task, y_true, y_pred, y_proba)

        # Get overall metrics
        overall_metrics = multitask_metrics.compute_overall_metrics()
        summary = multitask_metrics.get_summary_report()

        self.assertIn("macro_f1", overall_metrics)
        self.assertIn("overall", summary)

        # Test convergence tracking
        tracker = ConvergenceTracker(patience=5)
        for i in range(10):
            # Simulate training progress
            metric_value = 0.5 + (i * 0.02) + np.random.normal(0, 0.01)
            tracker.add_metric(metric_value)

        convergence_report = tracker.get_convergence_report()
        self.assertIn("best_metric", convergence_report)

    def test_session_evaluation_workflow(self):
        """Test session-level evaluation workflow"""
        evaluator = SessionLevelEvaluator()

        # Create multiple sessions with varied patterns
        for session_id in range(5):
            session = SessionContext(f"session_{session_id}")

            # Add messages with different patterns
            message_count = np.random.randint(3, 10)
            for msg_id in range(message_count):
                message = {
                    "text": f"Message {msg_id}",
                    "true_label": np.random.randint(0, 2),
                    "predicted_label": np.random.randint(0, 2),
                    "confidence": np.random.uniform(0.5, 1.0),
                    "toxicity_score": np.random.uniform(0.0, 1.0)
                }
                session.add_message(message)

            evaluator.add_session(session)

        # Compute session-level metrics
        session_f1 = evaluator.compute_session_level_f1()
        patterns = evaluator.analyze_session_patterns()
        escalation_metrics = evaluator.compute_escalation_metrics()

        self.assertIsInstance(session_f1, float)
        self.assertIn("avg_messages_per_session", patterns)
        self.assertIn("escalation_rate", escalation_metrics)

    def test_real_time_monitoring_workflow(self):
        """Test real-time monitoring workflow"""
        monitor = RealTimeMonitor(window_size=50)

        # Simulate real-time predictions
        for i in range(100):
            prediction = {
                "text": f"Text {i}",
                "prediction": np.random.randint(0, 2),
                "true_label": np.random.randint(0, 2),
                "confidence": np.random.uniform(0.3, 1.0),
                "timestamp": datetime.now()
            }
            monitor.add_prediction(prediction)

        # Get rolling metrics
        rolling_metrics = monitor.compute_rolling_metrics()
        anomalies = monitor.detect_anomalies()
        summary = monitor.get_performance_summary()

        self.assertIn("rolling_accuracy", rolling_metrics)
        self.assertIn("low_confidence_rate", anomalies)
        self.assertIn("total_predictions", summary)


if __name__ == '__main__':
    unittest.main()
