"""
Unit tests for CheckpointManager.

Tests cover:
- Basic save/load functionality
- Training history tracking
- Checkpoint cleanup
- Resume detection
- Integrity validation
- Export functionality
"""

import json
import pytest
import tempfile
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.cyberpuppy.training.checkpoint_manager import (
    CheckpointManager,
    TrainingMetrics,
    CheckpointInfo
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_size=10, hidden_size=20, output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def model():
    """Create simple model for testing."""
    return SimpleModel()


@pytest.fixture
def optimizer(model):
    """Create optimizer for testing."""
    return torch.optim.Adam(model.parameters(), lr=0.001)


@pytest.fixture
def sample_metrics():
    """Create sample training metrics."""
    return TrainingMetrics(
        epoch=1,
        train_loss=0.5,
        val_loss=0.4,
        train_metrics={"accuracy": 0.8, "f1": 0.75},
        val_metrics={"accuracy": 0.85, "f1": 0.82},
        learning_rate=0.001,
        timestamp=datetime.now().isoformat(),
        duration_seconds=120.5
    )


@pytest.fixture
def checkpoint_manager(temp_dir):
    """Create checkpoint manager for testing."""
    return CheckpointManager(
        checkpoint_dir=temp_dir / "checkpoints",
        model_name="test_model",
        keep_last_n=3,
        monitor_metric="val_loss",
        mode="min"
    )


class TestCheckpointManager:
    """Test CheckpointManager functionality."""

    def test_initialization(self, temp_dir):
        """Test checkpoint manager initialization."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir / "test_checkpoints",
            model_name="test_model",
            keep_last_n=2,
            monitor_metric="val_accuracy",
            mode="max"
        )

        assert manager.checkpoint_dir == temp_dir / "test_checkpoints"
        assert manager.model_name == "test_model"
        assert manager.keep_last_n == 2
        assert manager.monitor_metric == "val_accuracy"
        assert manager.mode == "max"
        assert manager.checkpoint_dir.exists()

    def test_save_checkpoint(self, checkpoint_manager, model, optimizer, sample_metrics):
        """Test saving a checkpoint."""
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model, optimizer, sample_metrics, is_best=False
        )

        assert checkpoint_path.exists()
        assert "epoch_1" in str(checkpoint_path)

        # Verify checkpoint contents
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == 1
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'metrics' in checkpoint

        # Verify training history was saved
        assert len(checkpoint_manager.training_history) == 1
        assert checkpoint_manager.training_history[0].epoch == 1

    def test_save_best_checkpoint(self, checkpoint_manager, model, optimizer, sample_metrics):
        """Test saving best checkpoint."""
        checkpoint_manager.save_checkpoint(
            model, optimizer, sample_metrics, is_best=True
        )

        assert checkpoint_manager.best_model_path.exists()

    def test_load_checkpoint(self, checkpoint_manager, model, optimizer, sample_metrics):
        """Test loading a checkpoint."""
        # Save a checkpoint first
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model, optimizer, sample_metrics, is_best=False
        )

        # Create new model and optimizer
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.002)

        # Load checkpoint
        epoch, metrics = checkpoint_manager.load_checkpoint(
            new_model, new_optimizer, checkpoint_path
        )

        assert epoch == 1
        assert metrics.epoch == 1
        assert metrics.train_loss == 0.5

        # Verify model state was loaded (weights should be different from initial)
        original_params = list(model.parameters())
        loaded_params = list(new_model.parameters())

        # Check that at least one parameter matches (indicating successful load)
        params_match = any(
            torch.allclose(orig, loaded)
            for orig, loaded in zip(original_params, loaded_params)
        )
        assert params_match

    def test_training_history_persistence(self, checkpoint_manager, model, optimizer):
        """Test that training history persists across manager instances."""
        # Save multiple checkpoints
        for i in range(3):
            metrics = TrainingMetrics(
                epoch=i+1,
                train_loss=0.5 - i*0.1,
                val_loss=0.4 - i*0.05,
                train_metrics={"accuracy": 0.8 + i*0.05},
                val_metrics={"accuracy": 0.85 + i*0.03},
                learning_rate=0.001,
                timestamp=datetime.now().isoformat(),
                duration_seconds=120.0
            )
            checkpoint_manager.save_checkpoint(model, optimizer, metrics)

        # Create new manager instance
        new_manager = CheckpointManager(
            checkpoint_dir=checkpoint_manager.checkpoint_dir,
            model_name="test_model"
        )

        assert len(new_manager.training_history) == 3
        assert new_manager.current_epoch == 3

    def test_checkpoint_cleanup(self, checkpoint_manager, model, optimizer):
        """Test automatic checkpoint cleanup."""
        # Save more checkpoints than keep_last_n
        for i in range(5):
            metrics = TrainingMetrics(
                epoch=i+1,
                train_loss=0.5,
                val_loss=0.4,
                train_metrics={},
                val_metrics={},
                learning_rate=0.001,
                timestamp=datetime.now().isoformat(),
                duration_seconds=120.0
            )
            checkpoint_manager.save_checkpoint(model, optimizer, metrics)

        # Check that only keep_last_n + any best model are kept
        checkpoint_files = list(checkpoint_manager.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        assert len(checkpoint_files) <= checkpoint_manager.keep_last_n

        # Latest checkpoints should still exist
        latest_epochs = sorted([
            int(f.stem.split('_')[-1]) for f in checkpoint_files
        ], reverse=True)
        expected_epochs = [5, 4, 3]  # Last 3 epochs
        assert latest_epochs == expected_epochs

    def test_checkpoint_validation(self, checkpoint_manager, model, optimizer, sample_metrics):
        """Test checkpoint integrity validation."""
        # Save valid checkpoint
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model, optimizer, sample_metrics
        )

        # Valid checkpoint should pass validation
        assert checkpoint_manager._validate_checkpoint(checkpoint_path)

        # Corrupt the checkpoint
        with open(checkpoint_path, 'w') as f:
            f.write("corrupted data")

        # Corrupted checkpoint should fail validation
        assert not checkpoint_manager._validate_checkpoint(checkpoint_path)

    def test_resume_detection(self, checkpoint_manager, model, optimizer, sample_metrics):
        """Test automatic resume detection."""
        # No checkpoints initially
        assert checkpoint_manager.check_for_resume() is None

        # Save a checkpoint
        checkpoint_manager.save_checkpoint(model, optimizer, sample_metrics)

        # Should detect checkpoint for resume
        resume_info = checkpoint_manager.check_for_resume()
        assert resume_info is not None

        checkpoint_path, epoch, metrics = resume_info
        assert epoch == 1
        assert checkpoint_path.exists()

    @patch('builtins.input')
    def test_prompt_for_resume_yes(self, mock_input, checkpoint_manager, model, optimizer, sample_metrics):
        """Test user prompt for resume - yes response."""
        # Save a checkpoint
        checkpoint_manager.save_checkpoint(model, optimizer, sample_metrics)

        # Mock user input - yes
        mock_input.return_value = 'y'

        result = checkpoint_manager.prompt_for_resume()
        assert result is True

    @patch('builtins.input')
    def test_prompt_for_resume_no(self, mock_input, checkpoint_manager, model, optimizer, sample_metrics):
        """Test user prompt for resume - no response."""
        # Save a checkpoint
        checkpoint_manager.save_checkpoint(model, optimizer, sample_metrics)

        # Mock user input - no
        mock_input.return_value = 'n'

        result = checkpoint_manager.prompt_for_resume()
        assert result is False

    def test_export_for_deployment(self, checkpoint_manager, model, optimizer, sample_metrics):
        """Test exporting model for deployment."""
        # Save a checkpoint as best model
        checkpoint_manager.save_checkpoint(model, optimizer, sample_metrics, is_best=True)

        # Export for deployment
        export_path = checkpoint_manager.export_for_deployment(model)

        assert export_path.exists()
        assert "deployment" in str(export_path)

        # Verify export contents
        deployment_data = torch.load(export_path, map_location='cpu')
        assert 'model_state_dict' in deployment_data
        assert 'export_timestamp' in deployment_data
        assert deployment_data['epoch'] == 1

        # Should not contain training-specific data
        assert 'optimizer_state_dict' not in deployment_data

    def test_get_training_summary(self, checkpoint_manager, model, optimizer):
        """Test getting training summary."""
        # Empty summary initially
        summary = checkpoint_manager.get_training_summary()
        assert "No training history" in summary["status"]

        # Add some training history
        for i in range(3):
            metrics = TrainingMetrics(
                epoch=i+1,
                train_loss=0.5 - i*0.1,
                val_loss=0.4 - i*0.05,
                train_metrics={},
                val_metrics={},
                learning_rate=0.001,
                timestamp=datetime.now().isoformat(),
                duration_seconds=120.0
            )
            checkpoint_manager.save_checkpoint(model, optimizer, metrics)

        summary = checkpoint_manager.get_training_summary()
        assert summary["total_epochs"] == 3
        assert summary["current_epoch"] == 3
        assert summary["best_epoch"] == 3  # Best val_loss (mode=min)

    def test_list_checkpoints(self, checkpoint_manager, model, optimizer):
        """Test listing all checkpoints."""
        # Save multiple checkpoints
        metrics_list = []
        for i in range(3):
            metrics = TrainingMetrics(
                epoch=i+1,
                train_loss=0.5 - i*0.1,
                val_loss=0.4 - i*0.05,
                train_metrics={},
                val_metrics={},
                learning_rate=0.001,
                timestamp=datetime.now().isoformat(),
                duration_seconds=120.0
            )
            metrics_list.append(metrics)
            is_best = (i == 2)  # Last one has best val_loss
            checkpoint_manager.save_checkpoint(model, optimizer, metrics, is_best=is_best)

        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 3

        # Check checkpoint info
        for i, checkpoint_info in enumerate(checkpoints):
            assert isinstance(checkpoint_info, CheckpointInfo)
            assert checkpoint_info.epoch == i + 1
            assert checkpoint_info.path.exists()
            # Last checkpoint should be marked as best
            if i == 2:
                assert checkpoint_info.is_best

    def test_checksum_calculation(self, checkpoint_manager, temp_dir):
        """Test checksum calculation."""
        test_file = temp_dir / "test.txt"
        test_content = "Hello, World!"

        with open(test_file, 'w') as f:
            f.write(test_content)

        checksum1 = checkpoint_manager._calculate_checksum(test_file)
        checksum2 = checkpoint_manager._calculate_checksum(test_file)

        # Same file should have same checksum
        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA256 hex length

        # Different content should have different checksum
        with open(test_file, 'w') as f:
            f.write("Different content")

        checksum3 = checkpoint_manager._calculate_checksum(test_file)
        assert checksum1 != checksum3

    def test_best_model_selection_min_mode(self, temp_dir):
        """Test best model selection in min mode."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir / "checkpoints",
            monitor_metric="val_loss",
            mode="min"
        )

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Save checkpoints with different val_loss values
        val_losses = [0.5, 0.3, 0.4, 0.2, 0.6]  # Best should be epoch 4 (val_loss=0.2)

        for i, val_loss in enumerate(val_losses):
            metrics = TrainingMetrics(
                epoch=i+1,
                train_loss=0.5,
                val_loss=val_loss,
                train_metrics={},
                val_metrics={},
                learning_rate=0.001,
                timestamp=datetime.now().isoformat(),
                duration_seconds=120.0
            )
            manager.save_checkpoint(model, optimizer, metrics)

        best_epoch = manager._find_best_epoch()
        assert best_epoch == 4  # Epoch with val_loss=0.2

    def test_best_model_selection_max_mode(self, temp_dir):
        """Test best model selection in max mode."""
        manager = CheckpointManager(
            checkpoint_dir=temp_dir / "checkpoints",
            monitor_metric="val_accuracy",
            mode="max"
        )

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Save checkpoints with different val_accuracy values
        val_accuracies = [0.7, 0.9, 0.8, 0.95, 0.6]  # Best should be epoch 4 (val_accuracy=0.95)

        for i, val_acc in enumerate(val_accuracies):
            metrics = TrainingMetrics(
                epoch=i+1,
                train_loss=0.5,
                val_loss=0.4,
                train_metrics={},
                val_metrics={"val_accuracy": val_acc},
                learning_rate=0.001,
                timestamp=datetime.now().isoformat(),
                duration_seconds=120.0
            )
            manager.save_checkpoint(model, optimizer, metrics)

        best_epoch = manager._find_best_epoch()
        assert best_epoch == 4  # Epoch with val_accuracy=0.95


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""

    def test_training_metrics_creation(self):
        """Test creating TrainingMetrics instance."""
        metrics = TrainingMetrics(
            epoch=5,
            train_loss=0.2,
            val_loss=0.15,
            train_metrics={"accuracy": 0.9},
            val_metrics={"accuracy": 0.92, "f1": 0.89},
            learning_rate=0.0001,
            timestamp="2024-01-01T12:00:00",
            duration_seconds=300.0
        )

        assert metrics.epoch == 5
        assert metrics.train_loss == 0.2
        assert metrics.val_loss == 0.15
        assert metrics.train_metrics["accuracy"] == 0.9
        assert metrics.val_metrics["f1"] == 0.89
        assert metrics.learning_rate == 0.0001
        assert metrics.duration_seconds == 300.0

    def test_training_metrics_serialization(self, sample_metrics):
        """Test TrainingMetrics serialization/deserialization."""
        from dataclasses import asdict

        # Convert to dict
        metrics_dict = asdict(sample_metrics)
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["epoch"] == 1

        # Convert back to TrainingMetrics
        restored_metrics = TrainingMetrics(**metrics_dict)
        assert restored_metrics.epoch == sample_metrics.epoch
        assert restored_metrics.train_loss == sample_metrics.train_loss