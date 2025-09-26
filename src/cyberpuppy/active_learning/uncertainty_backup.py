"""
Uncertainty sampling strategies for active learning
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader

from .base import ActiveLearner


class EntropySampling(ActiveLearner):
    """Entropy-based uncertainty sampling"""

    def select_samples(self,
                      unlabeled_data: Dataset,
                      n_samples: int,
                      labeled_data: Optional[Dataset] = None) -> List[int]:
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

        return selected_indices


class LeastConfidenceSampling(ActiveLearner):
    """Least confidence uncertainty sampling"""

    def select_samples(self,
                      unlabeled_data: Dataset,
                      n_samples: int,
                      labeled_data: Optional[Dataset] = None) -> List[int]:
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

        return selected_indices


class MarginSampling(ActiveLearner):
    """Margin-based uncertainty sampling"""

    def select_samples(self,
                      unlabeled_data: Dataset,
                      n_samples: int,
                      labeled_data: Optional[Dataset] = None) -> List[int]:
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

        return selected_indices


class BayesianUncertaintySampling(ActiveLearner):
    """Bayesian uncertainty using Monte Carlo Dropout"""

    def __init__(self, model, device: str = 'cpu', n_mc_samples: int = 10):
        """
        Initialize Bayesian uncertainty sampler

        Args:
            model: PyTorch model with dropout layers
            device: Device for computation
            n_mc_samples: Number of Monte Carlo samples
        """
        super().__init__(model, device)
        self.n_mc_samples = n_mc_samples

    def get_bayesian_predictions(self, data: Dataset) -> tuple:
        """
        Get predictions with Bayesian uncertainty using MC Dropout

        Args:
            data: Dataset to predict on

        Returns:
            Tuple of (mean_predictions, uncertainties)
        """
        self.model.train()  # Enable dropout
        dataloader = DataLoader(data, batch_size=32, shuffle=False)

        all_predictions = []

        # Multiple forward passes with dropout
        for _ in range(self.n_mc_samples):
            batch_predictions = []

            with torch.no_grad():
                for batch in dataloader:
                    inputs = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    outputs = self.model(input_ids=inputs, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    batch_predictions.append(probs.cpu().numpy())

            all_predictions.append(np.concatenate(batch_predictions))

        # Stack predictions from all MC samples
        predictions_stack = np.stack(all_predictions)  # (n_mc_samples, n_samples, n_classes)

        # Calculate mean and variance
        mean_predictions = np.mean(predictions_stack, axis=0)
        var_predictions = np.var(predictions_stack, axis=0)

        # Calculate uncertainty as mutual information
        # H[y|x] = E[H[y|x,θ]] where expectation is over posterior of θ
        entropy_per_sample = -np.sum(predictions_stack * np.log(predictions_stack + 1e-10), axis=2)
        expected_entropy = np.mean(entropy_per_sample, axis=0)

        # H[E[y|x]]
        entropy_of_expected = -np.sum(mean_predictions * np.log(mean_predictions + 1e-10), axis=1)

        # Mutual information I[y;θ|x] = H[E[y|x]] - E[H[y|x,θ]]
        mutual_information = entropy_of_expected - expected_entropy

        return mean_predictions, mutual_information

    def select_samples(self,
                      unlabeled_data: Dataset,
                      n_samples: int,
                      labeled_data: Optional[Dataset] = None) -> List[int]:
        """
        Select samples with highest Bayesian uncertainty

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Not used in this strategy

        Returns:
            List of indices with highest uncertainty
        """
        _, uncertainties = self.get_bayesian_predictions(unlabeled_data)

        # Select samples with highest uncertainty
        selected_indices = np.argsort(uncertainties)[-n_samples:].tolist()
        selected_indices.reverse()  # Highest uncertainty first

        return selected_indices


class EnsembleUncertaintySampling(ActiveLearner):
    """Uncertainty sampling using model ensemble"""

    def __init__(self, models: List, device: str = 'cpu'):
        """
        Initialize ensemble uncertainty sampler

        Args:
            models: List of PyTorch models
            device: Device for computation
        """
        self.models = models
        self.device = device

    def get_ensemble_predictions(self, data: Dataset) -> tuple:
        """
        Get ensemble predictions and uncertainties

        Args:
            data: Dataset to predict on

        Returns:
            Tuple of (mean_predictions, uncertainties)
        """
        dataloader = DataLoader(data, batch_size=32, shuffle=False)
        all_predictions = []

        for model in self.models:
            model.eval()
            model_predictions = []

            with torch.no_grad():
                for batch in dataloader:
                    inputs = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    outputs = model(input_ids=inputs, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    model_predictions.append(probs.cpu().numpy())

            all_predictions.append(np.concatenate(model_predictions))

        # Stack predictions from all models
        predictions_stack = np.stack(all_predictions)

        # Calculate mean and disagreement
        mean_predictions = np.mean(predictions_stack, axis=0)

        # Calculate variance across models as uncertainty measure
        disagreement = np.var(predictions_stack, axis=0)
        uncertainty = np.mean(disagreement, axis=1)

        return mean_predictions, uncertainty

    def select_samples(self,
                      unlabeled_data: Dataset,
                      n_samples: int,
                      labeled_data: Optional[Dataset] = None) -> List[int]:
        """
        Select samples with highest ensemble disagreement

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Not used in this strategy

        Returns:
            List of indices with highest disagreement
        """
        _, uncertainties = self.get_ensemble_predictions(unlabeled_data)

        # Select samples with highest uncertainty
        selected_indices = np.argsort(uncertainties)[-n_samples:].tolist()
        selected_indices.reverse()

        return selected_indices