#!/usr/bin/env python3
"""
Semi-supervised Training Script for Chinese Cyberbullying Detection.

Combines pseudo-labeling, self-training, and consistency regularization
to leverage unlabeled Chinese text data for improved model performance.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    AdamW, get_linear_schedule_with_warmup
)
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.cyberpuppy.semi_supervised import (
    PseudoLabelingPipeline, PseudoLabelConfig,
    SelfTrainingFramework, SelfTrainingConfig,
    ConsistencyRegularizer, ConsistencyConfig
)
from src.cyberpuppy.models.baselines import MultiTaskModel
from src.cyberpuppy.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Memory optimization utilities for RTX 3050 4GB."""

    @staticmethod
    def get_dynamic_batch_size(dataset_size: int, base_batch_size: int = 8) -> int:
        """Dynamically adjust batch size based on available memory."""
        try:
            # Check available GPU memory
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                free_memory = total_memory - allocated_memory

                # Conservative memory usage (use 70% of free memory)
                memory_per_sample = 50 * 1024 * 1024  # Estimate 50MB per sample
                max_batch_size = int((free_memory * 0.7) / memory_per_sample)

                # Use minimum of calculated and base batch size
                return max(min(max_batch_size, base_batch_size), 1)
            else:
                return base_batch_size
        except Exception as e:
            logger.warning(f"Failed to optimize batch size: {e}")
            return base_batch_size

    @staticmethod
    def clear_cache():
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_datasets(config: Dict[str, Any]) -> tuple:
    """Load labeled, unlabeled, and validation datasets."""
    # This is a placeholder - implement actual data loading logic
    # based on your dataset structure

    logger.info("Loading datasets...")

    # Example data loading (replace with actual implementation)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

    # Load your datasets here
    # labeled_dataset = ...
    # unlabeled_dataset = ...
    # val_dataset = ...

    # For demonstration, create dummy datasets
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size: int, num_classes: int = 3):
            self.size = size
            self.num_classes = num_classes

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(1, 1000, (512,)),
                'attention_mask': torch.ones(512),
                'labels': torch.randint(0, self.num_classes, (1,)).item()
            }

    labeled_dataset = DummyDataset(1000)
    unlabeled_dataset = DummyDataset(5000)
    val_dataset = DummyDataset(500)

    return labeled_dataset, unlabeled_dataset, val_dataset


def create_model(config: Dict[str, Any], device: str) -> nn.Module:
    """Create multi-task model."""
    model_config = AutoConfig.from_pretrained(config['model']['name'])
    model = MultiTaskModel(
        model_name=config['model']['name'],
        num_labels_toxicity=config['model']['num_labels_toxicity'],
        num_labels_emotion=config['model']['num_labels_emotion'],
        dropout_rate=config['model']['dropout_rate']
    )

    return model.to(device)


def create_data_loaders(labeled_dataset, unlabeled_dataset, val_dataset,
                       config: Dict[str, Any]) -> tuple:
    """Create data loaders with dynamic batch sizing."""
    memory_optimizer = MemoryOptimizer()

    # Dynamic batch size based on available memory
    base_batch_size = config['training']['batch_size']
    batch_size = memory_optimizer.get_dynamic_batch_size(
        len(labeled_dataset), base_batch_size
    )

    logger.info(f"Using batch size: {batch_size}")

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return labeled_loader, unlabeled_loader, val_loader


def train_pseudo_labeling(model, labeled_loader, unlabeled_loader, val_loader,
                         config: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Train using pseudo-labeling approach."""
    logger.info("Starting pseudo-labeling training...")

    # Configure pseudo-labeling
    pseudo_config = PseudoLabelConfig(
        confidence_threshold=config['pseudo_labeling']['confidence_threshold'],
        min_confidence_threshold=config['pseudo_labeling']['min_confidence_threshold'],
        max_confidence_threshold=config['pseudo_labeling']['max_confidence_threshold'],
        threshold_decay=config['pseudo_labeling']['threshold_decay'],
        max_pseudo_samples=config['pseudo_labeling']['max_pseudo_samples'],
        validation_metric=config['pseudo_labeling']['validation_metric'],
        early_stopping_patience=config['pseudo_labeling']['early_stopping_patience']
    )

    # Create pipeline
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    pipeline = PseudoLabelingPipeline(model, tokenizer, pseudo_config, device)

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    criterion = nn.CrossEntropyLoss()

    # Run iterative training
    history = pipeline.iterative_training(
        model=model,
        labeled_dataloader=labeled_loader,
        unlabeled_dataloader=unlabeled_loader,
        validation_dataloader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_iterations=config['pseudo_labeling']['num_iterations'],
        epochs_per_iteration=config['pseudo_labeling']['epochs_per_iteration']
    )

    return history


def train_self_training(model, labeled_loader, unlabeled_loader, val_loader,
                       config: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Train using self-training approach."""
    logger.info("Starting self-training...")

    # Configure self-training
    self_training_config = SelfTrainingConfig(
        teacher_update_frequency=config['self_training']['teacher_update_frequency'],
        student_teacher_ratio=config['self_training']['student_teacher_ratio'],
        distillation_temperature=config['self_training']['distillation_temperature'],
        ema_decay=config['self_training']['ema_decay'],
        consistency_weight=config['self_training']['consistency_weight'],
        confidence_threshold=config['self_training']['confidence_threshold'],
        max_epochs=config['self_training']['max_epochs']
    )

    # Create framework
    framework = SelfTrainingFramework(self_training_config, device)

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Setup scheduler
    total_steps = len(labeled_loader) * self_training_config.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss()

    # Run training
    history = framework.train(
        student_model=model,
        labeled_dataloader=labeled_loader,
        unlabeled_dataloader=unlabeled_loader,
        validation_dataloader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler
    )

    return history


def train_consistency_regularization(model, labeled_loader, unlabeled_loader, val_loader,
                                   config: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Train using consistency regularization."""
    logger.info("Starting consistency regularization training...")

    # Configure consistency training
    consistency_config = ConsistencyConfig(
        consistency_weight=config['consistency']['consistency_weight'],
        consistency_ramp_up_epochs=config['consistency']['consistency_ramp_up_epochs'],
        max_consistency_weight=config['consistency']['max_consistency_weight'],
        augmentation_strength=config['consistency']['augmentation_strength'],
        temperature=config['consistency']['temperature'],
        use_confidence_masking=config['consistency']['use_confidence_masking'],
        confidence_threshold=config['consistency']['confidence_threshold']
    )

    # Create regularizer
    regularizer = ConsistencyRegularizer(consistency_config, device)

    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = {
        'train_losses': [],
        'val_accuracies': [],
        'consistency_losses': [],
        'confidence_ratios': []
    }

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 5

    for epoch in range(config['consistency']['max_epochs']):
        regularizer.set_epoch(epoch)
        model.train()

        epoch_losses = []
        epoch_consistency_losses = []
        epoch_confidence_ratios = []

        # Create iterators
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)

        max_batches = max(len(labeled_loader), len(unlabeled_loader))

        for batch_idx in range(max_batches):
            # Get batches
            try:
                labeled_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_batch = next(labeled_iter)

            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            # Mixed training step
            losses = regularizer.mixed_training_step(
                model=model,
                labeled_batch=labeled_batch,
                unlabeled_batch=unlabeled_batch,
                optimizer=optimizer,
                criterion=criterion
            )

            epoch_losses.append(losses['total_loss'])
            epoch_consistency_losses.append(losses['consistency_loss'])
            epoch_confidence_ratios.append(losses['confidence_ratio'])

            # Memory cleanup
            if batch_idx % 50 == 0:
                MemoryOptimizer.clear_cache()

        # Validation
        val_acc = evaluate_model(model, val_loader, device)

        # Record history
        history['train_losses'].append(np.mean(epoch_losses))
        history['val_accuracies'].append(val_acc)
        history['consistency_losses'].append(np.mean(epoch_consistency_losses))
        history['confidence_ratios'].append(np.mean(epoch_confidence_ratios))

        logger.info(f"Epoch {epoch+1}: Train Loss = {np.mean(epoch_losses):.4f}, "
                   f"Val Acc = {val_acc:.4f}, "
                   f"Consistency Loss = {np.mean(epoch_consistency_losses):.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_consistency_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    return history


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    """Evaluate model accuracy."""
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    return total_correct / total_samples if total_samples > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Semi-supervised training for cyberbullying detection')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--method', type=str, choices=['pseudo_labeling', 'self_training', 'consistency'],
                       default='pseudo_labeling', help='Semi-supervised method to use')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for models and logs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')

    args = parser.parse_args()

    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Enable mixed precision training for memory efficiency
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    try:
        # Load datasets
        labeled_dataset, unlabeled_dataset, val_dataset = load_datasets(config)

        # Create data loaders
        labeled_loader, unlabeled_loader, val_loader = create_data_loaders(
            labeled_dataset, unlabeled_dataset, val_dataset, config
        )

        # Create model
        model = create_model(config, device)

        # Log model size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Train based on selected method
        if args.method == 'pseudo_labeling':
            history = train_pseudo_labeling(
                model, labeled_loader, unlabeled_loader, val_loader, config, device
            )
        elif args.method == 'self_training':
            history = train_self_training(
                model, labeled_loader, unlabeled_loader, val_loader, config, device
            )
        elif args.method == 'consistency':
            history = train_consistency_regularization(
                model, labeled_loader, unlabeled_loader, val_loader, config, device
            )

        # Save final model
        model_path = os.path.join(args.output_dir, f'final_{args.method}_model.pt')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

        # Save training history
        history_path = os.path.join(args.output_dir, f'{args.method}_history.yaml')
        with open(history_path, 'w', encoding='utf-8') as f:
            yaml.dump(history, f)
        logger.info(f"Training history saved to {history_path}")

        # Final evaluation
        final_acc = evaluate_model(model, val_loader, device)
        logger.info(f"Final validation accuracy: {final_acc:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        MemoryOptimizer.clear_cache()


if __name__ == '__main__':
    main()