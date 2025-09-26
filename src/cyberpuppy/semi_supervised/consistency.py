"""
Consistency Regularization for Semi-supervised Learning.

Enforces consistency between predictions on augmented versions
of the same input to improve model robustness and generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyConfig:
    """Configuration for consistency regularization."""
    consistency_weight: float = 1.0
    consistency_ramp_up_epochs: int = 5
    max_consistency_weight: float = 10.0
    augmentation_strength: float = 0.1
    temperature: float = 1.0
    ema_decay: float = 0.999
    use_confidence_masking: bool = True
    confidence_threshold: float = 0.8


class TextAugmenter:
    """Text augmentation for consistency training."""

    def __init__(self, augmentation_strength: float = 0.1):
        self.augmentation_strength = augmentation_strength

    def token_dropout(self, input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     dropout_prob: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly dropout tokens (replace with mask token)."""
        mask_token_id = 103  # [MASK] token for BERT-like models

        # Create dropout mask
        dropout_mask = torch.rand_like(input_ids.float()) < dropout_prob
        # Don't dropout special tokens (CLS, SEP, PAD)
        special_tokens = (input_ids == 101) | (input_ids == 102) | (input_ids == 0)
        dropout_mask = dropout_mask & ~special_tokens & attention_mask.bool()

        # Apply dropout
        augmented_input_ids = input_ids.clone()
        augmented_input_ids[dropout_mask] = mask_token_id

        return augmented_input_ids, attention_mask

    def token_shuffle(self, input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     shuffle_prob: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly shuffle adjacent tokens."""
        augmented_input_ids = input_ids.clone()

        for i in range(input_ids.size(0)):  # For each sample in batch
            seq_len = attention_mask[i].sum().item()
            # Skip special tokens at start and end
            start_idx = 1
            end_idx = seq_len - 1

            if end_idx > start_idx + 1:
                for j in range(start_idx, end_idx - 1):
                    if random.random() < shuffle_prob:
                        # Swap with next token
                        augmented_input_ids[i, j], augmented_input_ids[i, j + 1] = \
                            augmented_input_ids[i, j + 1], augmented_input_ids[i, j]

        return augmented_input_ids, attention_mask

    def synonym_replacement(self, input_ids: torch.Tensor,
                           attention_mask: torch.Tensor,
                           replacement_prob: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replace tokens with random tokens (simplified synonym replacement)."""
        vocab_size = 21128  # Chinese BERT vocab size

        augmented_input_ids = input_ids.clone()

        # Create replacement mask
        replacement_mask = torch.rand_like(input_ids.float()) < replacement_prob
        # Don't replace special tokens
        special_tokens = (input_ids == 101) | (input_ids == 102) | (input_ids == 0)
        replacement_mask = replacement_mask & ~special_tokens & attention_mask.bool()

        # Generate random replacements
        random_tokens = torch.randint(
            low=1, high=vocab_size,
            size=input_ids.shape,
            device=input_ids.device
        )

        augmented_input_ids[replacement_mask] = random_tokens[replacement_mask]

        return augmented_input_ids, attention_mask

    def augment_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply random augmentation to a batch."""
        augmented_batch = {}

        # Choose random augmentation
        aug_type = random.choice(['dropout', 'shuffle', 'synonym'])

        if aug_type == 'dropout':
            aug_input_ids, aug_attention_mask = self.token_dropout(
                batch['input_ids'], batch['attention_mask'],
                self.augmentation_strength
            )
        elif aug_type == 'shuffle':
            aug_input_ids, aug_attention_mask = self.token_shuffle(
                batch['input_ids'], batch['attention_mask'],
                self.augmentation_strength
            )
        else:  # synonym
            aug_input_ids, aug_attention_mask = self.synonym_replacement(
                batch['input_ids'], batch['attention_mask'],
                self.augmentation_strength
            )

        augmented_batch['input_ids'] = aug_input_ids
        augmented_batch['attention_mask'] = aug_attention_mask

        # Copy other fields
        for key, value in batch.items():
            if key not in ['input_ids', 'attention_mask']:
                augmented_batch[key] = value

        return augmented_batch


class ConsistencyRegularizer:
    """
    Consistency regularization for semi-supervised learning.

    Enforces that model predictions remain consistent across
    different augmented versions of the same input.
    """

    def __init__(self, config: ConsistencyConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.augmenter = TextAugmenter(config.augmentation_strength)
        self.current_epoch = 0

        # For tracking statistics
        self.consistency_losses = []
        self.confidence_ratios = []

    def get_consistency_weight(self, current_epoch: int) -> float:
        """Get current consistency weight with ramp-up schedule."""
        if current_epoch < self.config.consistency_ramp_up_epochs:
            # Linear ramp-up
            weight = (current_epoch / self.config.consistency_ramp_up_epochs) * \
                    self.config.max_consistency_weight
        else:
            weight = self.config.max_consistency_weight

        return weight

    def compute_consistency_loss(self,
                                original_logits: torch.Tensor,
                                augmented_logits: torch.Tensor,
                                confidence_mask: Optional[torch.Tensor] = None,
                                loss_type: str = "mse") -> torch.Tensor:
        """
        Compute consistency loss between original and augmented predictions.

        Args:
            original_logits: Logits from original input
            augmented_logits: Logits from augmented input
            confidence_mask: Mask for high-confidence predictions
            loss_type: Type of consistency loss ("mse", "kl", "ce")

        Returns:
            Consistency loss
        """
        if loss_type == "mse":
            # MSE between logits
            loss = F.mse_loss(original_logits, augmented_logits, reduction='none')
            loss = loss.mean(dim=-1)  # Average over classes

        elif loss_type == "kl":
            # KL divergence between probability distributions
            original_probs = F.softmax(original_logits / self.config.temperature, dim=-1)
            augmented_log_probs = F.log_softmax(
                augmented_logits / self.config.temperature, dim=-1
            )
            loss = F.kl_div(augmented_log_probs, original_probs, reduction='none')
            loss = loss.sum(dim=-1)  # Sum over classes

        elif loss_type == "ce":
            # Cross-entropy with soft targets
            original_probs = F.softmax(original_logits / self.config.temperature, dim=-1)
            augmented_log_probs = F.log_softmax(augmented_logits, dim=-1)
            loss = -(original_probs * augmented_log_probs).sum(dim=-1)

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Apply confidence masking
        if confidence_mask is not None and self.config.use_confidence_masking:
            loss = loss * confidence_mask
            if confidence_mask.sum() > 0:
                loss = loss.sum() / confidence_mask.sum()
            else:
                loss = torch.tensor(0.0, device=loss.device)
        else:
            loss = loss.mean()

        return loss

    def compute_confidence_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute confidence mask for high-confidence predictions."""
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)
        return max_probs >= self.config.confidence_threshold

    def consistency_training_step(self,
                                 model: nn.Module,
                                 batch: Dict[str, torch.Tensor],
                                 optimizer: torch.optim.Optimizer,
                                 labeled_loss: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Perform one consistency training step.

        Args:
            model: The model to train
            batch: Input batch
            optimizer: Optimizer
            labeled_loss: Optional supervised loss for labeled data

        Returns:
            Dictionary of losses
        """
        model.train()

        # Move batch to device
        original_batch = {k: v.to(self.device) for k, v in batch.items()
                         if k != 'text'}

        # Create augmented version
        augmented_batch = self.augmenter.augment_batch(original_batch)
        augmented_batch = {k: v.to(self.device) for k, v in augmented_batch.items()}

        # Forward pass on original input
        original_outputs = model(**{k: v for k, v in original_batch.items()
                                  if k not in ['labels', 'text']})
        original_logits = original_outputs.logits

        # Forward pass on augmented input
        augmented_outputs = model(**{k: v for k, v in augmented_batch.items()
                                   if k not in ['labels', 'text']})
        augmented_logits = augmented_outputs.logits

        # Compute confidence mask
        confidence_mask = self.compute_confidence_mask(original_logits)

        # Compute consistency loss
        consistency_loss = self.compute_consistency_loss(
            original_logits, augmented_logits, confidence_mask
        )

        # Get current consistency weight
        consistency_weight = self.get_consistency_weight(self.current_epoch)
        weighted_consistency_loss = consistency_weight * consistency_loss

        # Total loss
        total_loss = weighted_consistency_loss
        if labeled_loss is not None:
            total_loss += labeled_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Track statistics
        self.consistency_losses.append(consistency_loss.item())
        self.confidence_ratios.append(confidence_mask.float().mean().item())

        losses = {
            'total_loss': total_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'consistency_weight': consistency_weight,
            'confidence_ratio': confidence_mask.float().mean().item()
        }

        if labeled_loss is not None:
            losses['labeled_loss'] = labeled_loss.item()

        return losses

    def mixed_training_step(self,
                           model: nn.Module,
                           labeled_batch: Optional[Dict[str, torch.Tensor]],
                           unlabeled_batch: Dict[str, torch.Tensor],
                           optimizer: torch.optim.Optimizer,
                           criterion: nn.Module) -> Dict[str, float]:
        """
        Mixed training step with both labeled and unlabeled data.

        Args:
            model: The model to train
            labeled_batch: Labeled data batch (optional)
            unlabeled_batch: Unlabeled data batch
            optimizer: Optimizer
            criterion: Loss criterion for labeled data

        Returns:
            Dictionary of losses
        """
        model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        losses = {}

        # Supervised loss on labeled data
        labeled_loss = None
        if labeled_batch is not None:
            labeled_inputs = {k: v.to(self.device) for k, v in labeled_batch.items()
                            if k not in ['text', 'labels']}
            labels = labeled_batch['labels'].to(self.device)

            labeled_outputs = model(**labeled_inputs)
            labeled_loss = criterion(labeled_outputs.logits, labels)
            total_loss += labeled_loss
            losses['labeled_loss'] = labeled_loss.item()

        # Consistency loss on unlabeled data
        consistency_losses = self.consistency_training_step(
            model, unlabeled_batch, optimizer=None, labeled_loss=None
        )

        consistency_weight = self.get_consistency_weight(self.current_epoch)
        weighted_consistency_loss = consistency_weight * torch.tensor(
            consistency_losses['consistency_loss'], device=self.device
        )
        total_loss += weighted_consistency_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Combine losses
        losses.update({
            'total_loss': total_loss.item(),
            'consistency_loss': consistency_losses['consistency_loss'],
            'consistency_weight': consistency_weight,
            'confidence_ratio': consistency_losses['confidence_ratio']
        })

        return losses

    def evaluate_consistency(self,
                           model: nn.Module,
                           dataloader: torch.utils.data.DataLoader,
                           num_augmentations: int = 5) -> Dict[str, float]:
        """
        Evaluate model consistency across multiple augmentations.

        Args:
            model: Model to evaluate
            dataloader: Data loader
            num_augmentations: Number of augmentations per sample

        Returns:
            Consistency metrics
        """
        model.eval()

        all_original_probs = []
        all_augmented_probs = []
        consistency_scores = []

        with torch.no_grad():
            for batch in dataloader:
                original_batch = {k: v.to(self.device) for k, v in batch.items()
                                if k != 'text'}

                # Original predictions
                original_outputs = model(**{k: v for k, v in original_batch.items()
                                          if k not in ['labels', 'text']})
                original_probs = F.softmax(original_outputs.logits, dim=-1)

                # Multiple augmented predictions
                batch_augmented_probs = []
                for _ in range(num_augmentations):
                    augmented_batch = self.augmenter.augment_batch(original_batch)
                    augmented_outputs = model(**{k: v for k, v in augmented_batch.items()
                                               if k not in ['labels', 'text']})
                    augmented_probs = F.softmax(augmented_outputs.logits, dim=-1)
                    batch_augmented_probs.append(augmented_probs)

                # Compute consistency for this batch
                for i in range(original_probs.size(0)):
                    orig_prob = original_probs[i]
                    aug_probs = [probs[i] for probs in batch_augmented_probs]

                    # Average consistency score
                    consistency = []
                    for aug_prob in aug_probs:
                        # KL divergence as consistency measure
                        kl_div = F.kl_div(
                            torch.log(aug_prob + 1e-8), orig_prob, reduction='sum'
                        )
                        consistency.append((-kl_div).item())  # Negative because lower KL = higher consistency

                    consistency_scores.append(np.mean(consistency))

                all_original_probs.extend(original_probs.cpu().numpy())
                avg_augmented_probs = torch.stack(batch_augmented_probs).mean(dim=0)
                all_augmented_probs.extend(avg_augmented_probs.cpu().numpy())

        # Compute overall metrics
        consistency_score = np.mean(consistency_scores)
        consistency_std = np.std(consistency_scores)

        # Prediction diversity (how much augmentations change predictions)
        original_preds = np.argmax(all_original_probs, axis=-1)
        augmented_preds = np.argmax(all_augmented_probs, axis=-1)
        prediction_stability = (original_preds == augmented_preds).mean()

        metrics = {
            'consistency_score': consistency_score,
            'consistency_std': consistency_std,
            'prediction_stability': prediction_stability,
            'num_samples': len(consistency_scores)
        }

        return metrics

    def set_epoch(self, epoch: int):
        """Set current epoch for consistency weight scheduling."""
        self.current_epoch = epoch

    def get_statistics(self) -> Dict[str, float]:
        """Get training statistics."""
        if not self.consistency_losses:
            return {}

        return {
            'avg_consistency_loss': np.mean(self.consistency_losses[-100:]),  # Last 100 steps
            'avg_confidence_ratio': np.mean(self.confidence_ratios[-100:]),
            'current_consistency_weight': self.get_consistency_weight(self.current_epoch)
        }