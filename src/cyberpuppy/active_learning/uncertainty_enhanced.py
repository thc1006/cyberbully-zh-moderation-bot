"""
Enhanced uncertainty sampling strategies for active learning
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .base import ActiveLearner

logger = logging.getLogger(__name__)


class EntropySampling(ActiveLearner):
    """Entropy-based uncertainty sampling"""

    def select_samples(
        self, unlabeled_data: Dataset, n_samples: int, labeled_data: Optional[Dataset] = None
    ) -> List[int]:
        """
        Select samples with highest prediction entropy

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Not used in this strategy

        Returns:
            List of indices with highest entropy
        """
        _, probabilities = self.get_predictions(unlabeled_data)

        # Calculate entropy: -sum(p * log(p))
        entropies = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)

        # Select samples with highest entropy
        selected_indices = np.argsort(entropies)[-n_samples:].tolist()
        selected_indices.reverse()  # Highest entropy first

        logger.info(f"Selected {len(selected_indices)} samples using entropy sampling")
        logger.debug(f"Entropy range: [{entropies.min():.4f}, {entropies.max():.4f}]")

        return selected_indices


class LeastConfidenceSampling(ActiveLearner):
    """Least confidence uncertainty sampling"""

    def select_samples(
        self, unlabeled_data: Dataset, n_samples: int, labeled_data: Optional[Dataset] = None
    ) -> List[int]:
        """
        Select samples with lowest confidence (lowest max probability)

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Not used in this strategy

        Returns:
            List of indices with lowest confidence
        """
        _, probabilities = self.get_predictions(unlabeled_data)

        # Calculate confidence as max probability
        confidences = np.max(probabilities, axis=1)

        # Select samples with lowest confidence
        selected_indices = np.argsort(confidences)[:n_samples].tolist()

        logger.info(f"Selected {len(selected_indices)} samples using least confidence sampling")
        logger.debug(f"Confidence range: [{confidences.min():.4f}, {confidences.max():.4f}]")

        return selected_indices


class MarginSampling(ActiveLearner):
    """Margin-based uncertainty sampling"""

    def select_samples(
        self, unlabeled_data: Dataset, n_samples: int, labeled_data: Optional[Dataset] = None
    ) -> List[int]:
        """
        Select samples with smallest margin between top two predictions

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Not used in this strategy

        Returns:
            List of indices with smallest margins
        """
        _, probabilities = self.get_predictions(unlabeled_data)

        # Sort probabilities in descending order
        sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]

        # Calculate margin as difference between top two probabilities
        margins = sorted_probs[:, 0] - sorted_probs[:, 1]

        # Select samples with smallest margins
        selected_indices = np.argsort(margins)[:n_samples].tolist()

        logger.info(f"Selected {len(selected_indices)} samples using margin sampling")
        logger.debug(f"Margin range: [{margins.min():.4f}, {margins.max():.4f}]")

        return selected_indices


class BayesianUncertaintySampling(ActiveLearner):
    """MC Dropout-based Bayesian uncertainty sampling"""

    def __init__(self, model, device: str = "cpu", n_dropout_samples: int = 10):
        """
        Initialize Bayesian uncertainty sampler

        Args:
            model: PyTorch model with dropout layers
            device: Device for computation
            n_dropout_samples: Number of forward passes for MC Dropout
        """
        super().__init__(model, device)
        self.n_dropout_samples = n_dropout_samples

    def _enable_dropout(self, model):
        """Enable dropout during inference for MC Dropout"""
        for module in model.modules():
            if module.__class__.__name__.startswith("Dropout"):
                module.train()

    def get_mc_predictions(self, data: Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get MC Dropout predictions with epistemic uncertainty

        Args:
            data: Dataset to predict on

        Returns:
            (predictions, mean_probabilities, uncertainty_scores)
        """
        dataloader = DataLoader(data, batch_size=32, shuffle=False)

        all_predictions = []
        all_probabilities = []

        # Collect predictions from multiple dropout samples
        for _ in range(self.n_dropout_samples):
            batch_predictions = []
            batch_probabilities = []

            self.model.eval()
            self._enable_dropout(self.model)  # Enable dropout for uncertainty

            with torch.no_grad():
                for batch in dataloader:
                    inputs = batch["input_ids"].to(self.device)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)

                    outputs = self.model(inputs, attention_mask=attention_mask)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs

                    probabilities = F.softmax(logits, dim=-1)
                    predictions = torch.argmax(probabilities, dim=-1)

                    batch_predictions.extend(predictions.cpu().numpy())
                    batch_probabilities.extend(probabilities.cpu().numpy())

            all_predictions.append(batch_predictions)
            all_probabilities.append(batch_probabilities)

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)  # (n_samples, n_data)
        all_probabilities = np.array(all_probabilities)  # (n_samples, n_data, n_classes)

        # Calculate mean probabilities and uncertainties
        mean_probabilities = np.mean(all_probabilities, axis=0)

        # Epistemic uncertainty: variance in predictions
        prediction_variance = np.var(all_probabilities, axis=0)
        epistemic_uncertainty = np.sum(prediction_variance, axis=1)

        # Aleatoric uncertainty: mean entropy
        entropies = -np.sum(all_probabilities * np.log(all_probabilities + 1e-10), axis=2)
        aleatoric_uncertainty = np.mean(entropies, axis=0)

        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

        final_predictions = np.argmax(mean_probabilities, axis=1)

        return final_predictions, mean_probabilities, total_uncertainty

    def select_samples(
        self, unlabeled_data: Dataset, n_samples: int, labeled_data: Optional[Dataset] = None
    ) -> List[int]:
        """
        Select samples with highest Bayesian uncertainty

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Not used in this strategy

        Returns:
            List of indices with highest uncertainty
        """
        logger.info(f"Running MC Dropout with {self.n_dropout_samples} samples")

        _, _, uncertainties = self.get_mc_predictions(unlabeled_data)

        # Select samples with highest uncertainty
        selected_indices = np.argsort(uncertainties)[-n_samples:].tolist()
        selected_indices.reverse()  # Highest uncertainty first

        logger.info(f"Selected {len(selected_indices)} samples using Bayesian uncertainty sampling")
        logger.debug(f"Uncertainty range: [{uncertainties.min():.4f}, {uncertainties.max():.4f}]")

        return selected_indices

    def get_predictions(self, data: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Override to use MC Dropout predictions"""
        predictions, probabilities, _ = self.get_mc_predictions(data)
        return predictions, probabilities


class BALD(BayesianUncertaintySampling):
    """Bayesian Active Learning by Disagreement (BALD)"""

    def select_samples(
        self, unlabeled_data: Dataset, n_samples: int, labeled_data: Optional[Dataset] = None
    ) -> List[int]:
        """
        Select samples using BALD: maximizing mutual information

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Not used in this strategy

        Returns:
            List of indices with highest BALD scores
        """
        dataloader = DataLoader(unlabeled_data, batch_size=32, shuffle=False)

        all_probabilities = []

        # Collect predictions from multiple dropout samples
        for _ in range(self.n_dropout_samples):
            batch_probabilities = []

            self.model.eval()
            self._enable_dropout(self.model)

            with torch.no_grad():
                for batch in dataloader:
                    inputs = batch["input_ids"].to(self.device)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)

                    outputs = self.model(inputs, attention_mask=attention_mask)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    probabilities = F.softmax(logits, dim=-1)
                    batch_probabilities.extend(probabilities.cpu().numpy())

            all_probabilities.append(batch_probabilities)

        all_probabilities = np.array(all_probabilities)  # (n_samples, n_data, n_classes)

        # Calculate BALD scores
        # H[y|x,D] - E_theta[H[y|x,theta]]
        mean_probs = np.mean(all_probabilities, axis=0)
        mean_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

        sample_entropies = -np.sum(all_probabilities * np.log(all_probabilities + 1e-10), axis=2)
        expected_entropy = np.mean(sample_entropies, axis=0)

        bald_scores = mean_entropy - expected_entropy

        # Select samples with highest BALD scores
        selected_indices = np.argsort(bald_scores)[-n_samples:].tolist()
        selected_indices.reverse()

        logger.info(f"Selected {len(selected_indices)} samples using BALD")
        logger.debug(f"BALD score range: [{bald_scores.min():.4f}, {bald_scores.max():.4f}]")

        return selected_indices
