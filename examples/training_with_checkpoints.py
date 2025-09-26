#!/usr/bin/env python3
"""
Example script demonstrating how to use CheckpointManager for robust training.

This example shows:
1. Setting up checkpoint manager
2. Automatic resume detection
3. Training loop with checkpoints
4. Model export for deployment

Usage:
    python examples/training_with_checkpoints.py --model_name macbert_aggressive
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cyberpuppy.training.checkpoint_manager import CheckpointManager, TrainingMetrics


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleMultiTaskModel(nn.Module):
    """Simple multi-task model for demonstration."""

    def __init__(self, input_size=768, hidden_size=256, num_tasks=3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Task-specific heads
        self.toxicity_head = nn.Linear(hidden_size, 3)  # none, toxic, severe
        self.bullying_head = nn.Linear(hidden_size, 3)  # none, harassment, threat
        self.emotion_head = nn.Linear(hidden_size, 3)   # pos, neu, neg

    def forward(self, x):
        features = self.backbone(x)
        return {
            'toxicity': self.toxicity_head(features),
            'bullying': self.bullying_head(features),
            'emotion': self.emotion_head(features)
        }


def create_dummy_data(num_samples=1000, input_size=768):
    """Create dummy data for demonstration."""
    X = torch.randn(num_samples, input_size)
    y_toxicity = torch.randint(0, 3, (num_samples,))
    y_bullying = torch.randint(0, 3, (num_samples,))
    y_emotion = torch.randint(0, 3, (num_samples,))

    return X, {'toxicity': y_toxicity, 'bullying': y_bullying, 'emotion': y_emotion}


def calculate_metrics(outputs, targets):
    """Calculate training metrics."""
    metrics = {}
    total_correct = 0
    total_samples = 0

    for task in ['toxicity', 'bullying', 'emotion']:
        preds = torch.argmax(outputs[task], dim=1)
        correct = (preds == targets[task]).sum().item()
        accuracy = correct / len(targets[task])
        metrics[f'{task}_accuracy'] = accuracy

        total_correct += correct
        total_samples += len(targets[task])

    # Overall accuracy
    metrics['overall_accuracy'] = total_correct / total_samples
    return metrics


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_outputs = {task: [] for task in ['toxicity', 'bullying', 'emotion']}
    all_targets = {task: [] for task in ['toxicity', 'bullying', 'emotion']}

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        for task in batch_y:
            batch_y[task] = batch_y[task].to(device)

        optimizer.zero_grad()

        outputs = model(batch_x)

        # Calculate multi-task loss
        loss = 0
        for task in ['toxicity', 'bullying', 'emotion']:
            task_loss = criterion(outputs[task], batch_y[task])
            loss += task_loss

            # Store predictions and targets
            all_outputs[task].append(outputs[task].detach().cpu())
            all_targets[task].append(batch_y[task].cpu())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Concatenate all predictions and targets
    for task in ['toxicity', 'bullying', 'emotion']:
        all_outputs[task] = torch.cat(all_outputs[task])
        all_targets[task] = torch.cat(all_targets[task])

    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_outputs, all_targets)

    return avg_loss, metrics


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_outputs = {task: [] for task in ['toxicity', 'bullying', 'emotion']}
    all_targets = {task: [] for task in ['toxicity', 'bullying', 'emotion']}

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            for task in batch_y:
                batch_y[task] = batch_y[task].to(device)

            outputs = model(batch_x)

            # Calculate multi-task loss
            loss = 0
            for task in ['toxicity', 'bullying', 'emotion']:
                task_loss = criterion(outputs[task], batch_y[task])
                loss += task_loss

                # Store predictions and targets
                all_outputs[task].append(outputs[task].cpu())
                all_targets[task].append(batch_y[task].cpu())

            total_loss += loss.item()

    # Concatenate all predictions and targets
    for task in ['toxicity', 'bullying', 'emotion']:
        all_outputs[task] = torch.cat(all_outputs[task])
        all_targets[task] = torch.cat(all_targets[task])

    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(all_outputs, all_targets)

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Training with CheckpointManager example')
    parser.add_argument('--model_name', default='macbert_aggressive', help='Model name for checkpoints')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint_dir', default='models/local_training', help='Checkpoint directory')
    parser.add_argument('--resume', action='store_true', help='Force resume from checkpoint')
    parser.add_argument('--no_prompt', action='store_true', help='Skip resume prompt')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model and optimizer
    model = SimpleMultiTaskModel()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Create dummy data
    logger.info("Creating dummy training data...")
    train_x, train_y = create_dummy_data(2000)
    val_x, val_y = create_dummy_data(500)

    # Create data loaders
    train_dataset = TensorDataset(train_x, train_y['toxicity'], train_y['bullying'], train_y['emotion'])
    val_dataset = TensorDataset(val_x, val_y['toxicity'], val_y['bullying'], val_y['emotion'])

    def collate_fn(batch):
        x = torch.stack([item[0] for item in batch])
        y = {
            'toxicity': torch.stack([item[1] for item in batch]),
            'bullying': torch.stack([item[2] for item in batch]),
            'emotion': torch.stack([item[3] for item in batch])
        }
        return x, y

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Setup checkpoint manager
    checkpoint_dir = Path(args.checkpoint_dir) / args.model_name
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        model_name=args.model_name,
        keep_last_n=3,
        monitor_metric="val_loss",
        mode="min"
    )

    logger.info(f"Checkpoint manager initialized at: {checkpoint_dir}")

    # Check for resume
    start_epoch = 1
    if not args.no_prompt:
        should_resume = checkpoint_manager.prompt_for_resume()
        if should_resume or args.resume:
            try:
                start_epoch, last_metrics = checkpoint_manager.load_checkpoint(model, optimizer)
                start_epoch += 1  # Start from next epoch
                logger.info(f"Resumed from epoch {start_epoch - 1}")
                logger.info(f"Last validation loss: {last_metrics.val_loss:.4f}")
            except Exception as e:
                logger.warning(f"Failed to resume: {e}")
                logger.info("Starting fresh training...")
                start_epoch = 1

    # Training loop
    logger.info(f"Starting training from epoch {start_epoch} to {args.epochs}")
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()

        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info("-" * 50)

        # Training
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)

        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time

        # Log results
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Train Overall Accuracy: {train_metrics['overall_accuracy']:.4f}")
        logger.info(f"Val Overall Accuracy: {val_metrics['overall_accuracy']:.4f}")
        logger.info(f"Epoch Duration: {epoch_duration:.1f}s")

        # Check if this is the best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info("★ New best model!")

        # Create training metrics
        training_metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            learning_rate=optimizer.param_groups[0]['lr'],
            timestamp=datetime.now().isoformat(),
            duration_seconds=epoch_duration
        )

        # Save checkpoint
        try:
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model, optimizer, training_metrics, is_best=is_best
            )
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    # Training completed
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)

    # Print training summary
    summary = checkpoint_manager.get_training_summary()
    logger.info(f"Total epochs: {summary['total_epochs']}")
    logger.info(f"Best epoch: {summary.get('best_epoch', 'N/A')}")
    logger.info(f"Final validation loss: {summary['latest_metrics']['val_loss']:.4f}")

    # Export for deployment
    try:
        export_path = checkpoint_manager.export_for_deployment(model, use_best_model=True)
        logger.info(f"Model exported for deployment: {export_path}")
    except Exception as e:
        logger.error(f"Failed to export model: {e}")

    # List all checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()
    if checkpoints:
        logger.info(f"\nAvailable checkpoints ({len(checkpoints)}):")
        for cp in checkpoints:
            status = " [BEST]" if cp.is_best else ""
            logger.info(f"  Epoch {cp.epoch}: {cp.path.name}{status}")


if __name__ == "__main__":
    main()