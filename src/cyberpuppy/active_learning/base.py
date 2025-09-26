"""
Base classes for active learning framework
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


class ActiveLearner(ABC):
    """Base class for active learning strategies"""

    def __init__(self, model, device: str = 'cpu'):
        """
        Initialize active learner

        Args:
            model: PyTorch model for predictions
            device: Device for computation
        """
        self.model = model
        self.device = device

    @abstractmethod
    def select_samples(self,
                      unlabeled_data: Dataset,
                      n_samples: int,
                      labeled_data: Optional[Dataset] = None) -> List[int]:
        """
        Select most informative samples for annotation

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Already labeled data (for diversity sampling)

        Returns:
            List of indices of selected samples
        """
        pass

    def get_predictions(self, data: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get model predictions and uncertainties

        Args:
            data: Dataset to predict on

        Returns:
            Tuple of (predictions, uncertainties)
        """
        self.model.eval()
        dataloader = DataLoader(data, batch_size=32, shuffle=False)

        all_probs = []
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids=inputs, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)

                all_probs.append(probs.cpu().numpy())
                all_preds.append(torch.argmax(probs, dim=-1).cpu().numpy())

        predictions = np.concatenate(all_preds)
        probabilities = np.concatenate(all_probs)

        return predictions, probabilities


class QueryStrategy(ABC):
    """Base class for query strategies"""

    @abstractmethod
    def query(self,
             unlabeled_data: Dataset,
             n_samples: int,
             model: Any,
             labeled_data: Optional[Dataset] = None) -> List[int]:
        """
        Query samples for annotation

        Args:
            unlabeled_data: Unlabeled dataset
            n_samples: Number of samples to query
            model: Trained model
            labeled_data: Already labeled data

        Returns:
            Indices of selected samples
        """
        pass


class AnnotationInterface(ABC):
    """Base class for annotation interfaces"""

    @abstractmethod
    def annotate_samples(self,
                        samples: List[Any],
                        indices: List[int]) -> List[Dict]:
        """
        Annotate selected samples

        Args:
            samples: List of samples to annotate
            indices: Indices of samples

        Returns:
            List of annotations
        """
        pass


class StoppingCriterion:
    """Stopping criterion for active learning"""

    def __init__(self,
                 target_f1: float = 0.75,
                 patience: int = 3,
                 min_improvement: float = 0.01):
        """
        Initialize stopping criterion

        Args:
            target_f1: Target F1 score to achieve
            patience: Number of iterations without improvement
            min_improvement: Minimum improvement to consider
        """
        self.target_f1 = target_f1
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_f1 = 0.0
        self.no_improvement_count = 0

    def should_stop(self, current_f1: float) -> Tuple[bool, str]:
        """
        Check if active learning should stop

        Args:
            current_f1: Current F1 score

        Returns:
            Tuple of (should_stop, reason)
        """
        # Check if target reached
        if current_f1 >= self.target_f1:
            return True, f"Target F1 score {self.target_f1} reached: {current_f1:.4f}"

        # Check for improvement
        if current_f1 > self.best_f1 + self.min_improvement:
            self.best_f1 = current_f1
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # Check patience
        if self.no_improvement_count >= self.patience:
            return True, f"No improvement for {self.patience} iterations"

        return False, ""


class LearningCurveTracker:
    """Track learning curves during active learning"""

    def __init__(self):
        """Initialize tracker"""
        self.iterations = []
        self.labeled_sizes = []
        self.f1_scores = []
        self.uncertainties = []
        self.diversities = []

    def update(self,
               iteration: int,
               labeled_size: int,
               f1_score: float,
               avg_uncertainty: float = None,
               avg_diversity: float = None):
        """
        Update tracking metrics

        Args:
            iteration: Current iteration
            labeled_size: Size of labeled dataset
            f1_score: Current F1 score
            avg_uncertainty: Average uncertainty of selected samples
            avg_diversity: Average diversity of selected samples
        """
        self.iterations.append(iteration)
        self.labeled_sizes.append(labeled_size)
        self.f1_scores.append(f1_score)
        if avg_uncertainty is not None:
            self.uncertainties.append(avg_uncertainty)
        if avg_diversity is not None:
            self.diversities.append(avg_diversity)

    def get_summary(self) -> Dict:
        """Get summary of learning progress"""
        return {
            'iterations': len(self.iterations),
            'final_f1': self.f1_scores[-1] if self.f1_scores else 0.0,
            'best_f1': max(self.f1_scores) if self.f1_scores else 0.0,
            'total_labeled': self.labeled_sizes[-1] if self.labeled_sizes else 0,
            'improvement_rate': (self.f1_scores[-1] - self.f1_scores[0]) / len(self.f1_scores)
                               if len(self.f1_scores) > 1 else 0.0
        }