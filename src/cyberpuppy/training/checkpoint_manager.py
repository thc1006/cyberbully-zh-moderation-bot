"""
Robust checkpoint and resume system for local training.

This module provides comprehensive checkpoint management including:
- Automatic save/load with integrity validation
- Training history tracking
- Automatic resume detection
- Checkpoint cleanup with retention policy
- Deployment model export
"""

import json
import logging
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Optimizer


logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics for a single epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    learning_rate: float
    timestamp: str
    duration_seconds: float


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    epoch: int
    path: Path
    metrics: TrainingMetrics
    is_best: bool
    file_size: int
    checksum: str
    created_at: str


class CheckpointManager:
    """
    Manages training checkpoints with automatic save/load, cleanup, and resume functionality.

    Features:
    - Save checkpoints every epoch with model, optimizer, and metrics
    - Automatic cleanup keeping best model + last N checkpoints
    - Integrity validation with checksums
    - Training history tracking
    - Interactive resume detection
    - Export to deployment format
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        model_name: str = "model",
        keep_last_n: int = 3,
        monitor_metric: str = "val_loss",
        mode: str = "min"
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            model_name: Name prefix for checkpoint files
            keep_last_n: Number of recent checkpoints to keep
            monitor_metric: Metric to monitor for best model
            mode: 'min' or 'max' for best model selection
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.keep_last_n = keep_last_n
        self.monitor_metric = monitor_metric
        self.mode = mode

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.history_file = self.checkpoint_dir / "training_history.json"
        self.best_model_path = self.checkpoint_dir / "best_model.pt"
        self.optimizer_state_path = self.checkpoint_dir / "optimizer_state.pt"

        # Training state
        self.training_history: List[TrainingMetrics] = []
        self.best_metric_value: Optional[float] = None
        self.current_epoch = 0

        # Load existing history
        self._load_training_history()

        logger.info(f"CheckpointManager initialized at {self.checkpoint_dir}")
        logger.info(f"Monitoring {self.monitor_metric} ({self.mode}), keeping last {self.keep_last_n} checkpoints")

    def _load_training_history(self) -> None:
        """Load training history from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)

                self.training_history = [
                    TrainingMetrics(**metrics) for metrics in history_data
                ]

                if self.training_history:
                    self.current_epoch = self.training_history[-1].epoch
                    # Find best metric value
                    metric_values = [
                        getattr(m, self.monitor_metric, float('inf'))
                        for m in self.training_history
                    ]
                    if self.mode == "min":
                        self.best_metric_value = min(metric_values)
                    else:
                        self.best_metric_value = max(metric_values)

                logger.info(f"Loaded training history: {len(self.training_history)} epochs")

            except Exception as e:
                logger.warning(f"Failed to load training history: {e}")
                self.training_history = []

    def _save_training_history(self) -> None:
        """Save training history to disk."""
        try:
            history_data = [asdict(metrics) for metrics in self.training_history]
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved training history to {self.history_file}")
        except Exception as e:
            logger.error(f"Failed to save training history: {e}")

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""

    def _validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """Validate checkpoint integrity."""
        try:
            # Check if file exists and is not empty
            if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
                return False

            # Try to load the checkpoint (with weights_only=False for backwards compatibility)
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Check required keys
            required_keys = ['epoch', 'model_state_dict', 'metrics']
            if not all(key in checkpoint for key in required_keys):
                logger.warning(f"Checkpoint {checkpoint_path} missing required keys")
                return False

            return True

        except Exception as e:
            logger.error(f"Checkpoint validation failed for {checkpoint_path}: {e}")
            return False

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        metrics: TrainingMetrics,
        is_best: bool = False
    ) -> Path:
        """
        Save a training checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            metrics: Training metrics for this epoch
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint
        """
        try:
            # Update tracking
            self.current_epoch = metrics.epoch
            self.training_history.append(metrics)

            # Create checkpoint data
            checkpoint = {
                'epoch': metrics.epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': asdict(metrics),
                'model_name': self.model_name,
                'timestamp': datetime.now().isoformat(),
                'pytorch_version': torch.__version__
            }

            # Save regular checkpoint
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{metrics.epoch}.pt"
            torch.save(checkpoint, checkpoint_path)

            # Calculate checksum
            checksum = self._calculate_checksum(checkpoint_path)

            # Save optimizer state separately
            torch.save(optimizer.state_dict(), self.optimizer_state_path)

            logger.info(f"Saved checkpoint for epoch {metrics.epoch} to {checkpoint_path}")

            # Save as best model if needed
            if is_best:
                shutil.copy2(checkpoint_path, self.best_model_path)
                logger.info(f"Saved best model (epoch {metrics.epoch})")

            # Save training history
            self._save_training_history()

            # Cleanup old checkpoints
            self._cleanup_checkpoints()

            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        checkpoint_path: Optional[Path] = None
    ) -> Tuple[int, TrainingMetrics]:
        """
        Load a checkpoint and restore model/optimizer state.

        Args:
            model: PyTorch model to load state into
            optimizer: Optimizer to load state into (optional)
            checkpoint_path: Specific checkpoint to load (uses latest if None)

        Returns:
            Tuple of (epoch, metrics)
        """
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint()
            if checkpoint_path is None:
                raise ValueError("No checkpoints found")

        if not self._validate_checkpoint(checkpoint_path):
            raise ValueError(f"Invalid or corrupted checkpoint: {checkpoint_path}")

        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state if provided
            if optimizer is not None:
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                elif self.optimizer_state_path.exists():
                    optimizer_state = torch.load(self.optimizer_state_path, map_location='cpu', weights_only=False)
                    optimizer.load_state_dict(optimizer_state)

            # Extract metrics
            metrics = TrainingMetrics(**checkpoint['metrics'])
            epoch = checkpoint['epoch']

            logger.info(f"Loaded checkpoint from epoch {epoch}")
            return epoch, metrics

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def _get_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint file."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoint_files:
            return None

        # Sort by epoch number
        def extract_epoch(path: Path) -> int:
            try:
                return int(path.stem.split('_')[-1])
            except:
                return 0

        checkpoint_files.sort(key=extract_epoch, reverse=True)
        return checkpoint_files[0]

    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints, keeping best + last N."""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if len(checkpoint_files) <= self.keep_last_n:
                return

            # Sort by epoch number
            def extract_epoch(path: Path) -> int:
                try:
                    return int(path.stem.split('_')[-1])
                except:
                    return 0

            checkpoint_files.sort(key=extract_epoch, reverse=True)

            # Keep the last N checkpoints
            to_keep = set(checkpoint_files[:self.keep_last_n])

            # Always keep the best model checkpoint if it exists
            if self.best_model_path.exists():
                # Find which checkpoint corresponds to best model
                best_epoch = self._find_best_epoch()
                if best_epoch is not None:
                    best_checkpoint = self.checkpoint_dir / f"checkpoint_epoch_{best_epoch}.pt"
                    if best_checkpoint.exists():
                        to_keep.add(best_checkpoint)

            # Remove old checkpoints
            removed_count = 0
            for checkpoint_file in checkpoint_files:
                if checkpoint_file not in to_keep:
                    try:
                        checkpoint_file.unlink()
                        removed_count += 1
                        logger.debug(f"Removed old checkpoint: {checkpoint_file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {checkpoint_file}: {e}")

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old checkpoints")

        except Exception as e:
            logger.error(f"Checkpoint cleanup failed: {e}")

    def _find_best_epoch(self) -> Optional[int]:
        """Find the epoch with the best metric value."""
        if not self.training_history:
            return None

        best_epoch = None
        best_value = None

        for metrics in self.training_history:
            value = getattr(metrics, self.monitor_metric, None)
            if value is None:
                continue

            if best_value is None:
                best_value = value
                best_epoch = metrics.epoch
            elif (self.mode == "min" and value < best_value) or (self.mode == "max" and value > best_value):
                best_value = value
                best_epoch = metrics.epoch

        return best_epoch

    def check_for_resume(self) -> Optional[Tuple[Path, int, TrainingMetrics]]:
        """
        Check if there are checkpoints to resume from.

        Returns:
            None if no checkpoints, or (checkpoint_path, epoch, metrics) tuple
        """
        latest_checkpoint = self._get_latest_checkpoint()
        if latest_checkpoint is None:
            return None

        if not self._validate_checkpoint(latest_checkpoint):
            logger.warning(f"Latest checkpoint {latest_checkpoint} is corrupted")
            return None

        try:
            checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
            epoch = checkpoint['epoch']
            metrics = TrainingMetrics(**checkpoint['metrics'])
            return latest_checkpoint, epoch, metrics
        except Exception as e:
            logger.error(f"Failed to read checkpoint info: {e}")
            return None

    def prompt_for_resume(self) -> bool:
        """
        Interactively ask user if they want to resume training.

        Returns:
            True if user wants to resume, False otherwise
        """
        resume_info = self.check_for_resume()
        if resume_info is None:
            return False

        checkpoint_path, epoch, metrics = resume_info

        print(f"\n{'='*60}")
        print(f"CHECKPOINT FOUND")
        print(f"{'='*60}")
        print(f"Found checkpoint at epoch {epoch}")
        print(f"Path: {checkpoint_path}")
        print(f"Timestamp: {metrics.timestamp}")
        print(f"Train Loss: {metrics.train_loss:.4f}")
        print(f"Val Loss: {metrics.val_loss:.4f}")

        if metrics.val_metrics:
            print("Validation Metrics:")
            for metric, value in metrics.val_metrics.items():
                print(f"  {metric}: {value:.4f}")

        print(f"{'='*60}")

        while True:
            response = input("Resume training from this checkpoint? [Y/n]: ").strip().lower()
            if response in ['', 'y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")

    def export_for_deployment(
        self,
        model: nn.Module,
        export_path: Optional[Path] = None,
        use_best_model: bool = True
    ) -> Path:
        """
        Export model for deployment (without training-specific components).

        Args:
            model: PyTorch model
            export_path: Path to save deployment model
            use_best_model: Whether to use best model or current model

        Returns:
            Path to exported model
        """
        if export_path is None:
            export_path = self.checkpoint_dir / f"{self.model_name}_deployment.pt"

        try:
            if use_best_model and self.best_model_path.exists():
                logger.info("Exporting best model for deployment")
                checkpoint = torch.load(self.best_model_path, map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint['epoch']
                metrics = TrainingMetrics(**checkpoint['metrics'])
            else:
                logger.info("Exporting current model for deployment")
                epoch = self.current_epoch
                metrics = self.training_history[-1] if self.training_history else None

            # Create deployment package
            deployment_data = {
                'model_state_dict': model.state_dict(),
                'model_name': self.model_name,
                'epoch': epoch,
                'export_timestamp': datetime.now().isoformat(),
                'pytorch_version': torch.__version__
            }

            if metrics:
                deployment_data['final_metrics'] = asdict(metrics)

            # Save deployment model
            torch.save(deployment_data, export_path)

            # Calculate and log size
            file_size = export_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Exported deployment model to {export_path} ({file_size:.1f} MB)")

            return export_path

        except Exception as e:
            logger.error(f"Failed to export deployment model: {e}")
            raise

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.training_history:
            return {"status": "No training history found"}

        latest_metrics = self.training_history[-1]
        best_epoch = self._find_best_epoch()

        summary = {
            "total_epochs": len(self.training_history),
            "current_epoch": self.current_epoch,
            "latest_metrics": asdict(latest_metrics),
            "best_epoch": best_epoch,
            "monitor_metric": self.monitor_metric,
            "mode": self.mode,
            "checkpoint_dir": str(self.checkpoint_dir),
            "has_best_model": self.best_model_path.exists()
        }

        if best_epoch is not None:
            best_metrics = next(m for m in self.training_history if m.epoch == best_epoch)
            summary["best_metrics"] = asdict(best_metrics)

        return summary

    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all available checkpoints with their information."""
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_epoch_*.pt"):
            try:
                if not self._validate_checkpoint(checkpoint_file):
                    continue

                checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
                epoch = checkpoint['epoch']
                metrics = TrainingMetrics(**checkpoint['metrics'])

                file_size = checkpoint_file.stat().st_size
                checksum = self._calculate_checksum(checkpoint_file)
                created_at = datetime.fromtimestamp(checkpoint_file.stat().st_mtime).isoformat()

                # Check if this is the best model
                best_epoch = self._find_best_epoch()
                is_best = (best_epoch == epoch)

                checkpoint_info = CheckpointInfo(
                    epoch=epoch,
                    path=checkpoint_file,
                    metrics=metrics,
                    is_best=is_best,
                    file_size=file_size,
                    checksum=checksum,
                    created_at=created_at
                )

                checkpoints.append(checkpoint_info)

            except Exception as e:
                logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")

        # Sort by epoch
        checkpoints.sort(key=lambda x: x.epoch)
        return checkpoints