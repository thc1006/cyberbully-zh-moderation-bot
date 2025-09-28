"""
Hybrid query strategies combining uncertainty and diversity sampling
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

from .base import QueryStrategy
from .diversity import (ClusteringSampling, CoreSetSampling, DiversityMixin,
                        RepresentativeSampling)
from .uncertainty import (BayesianUncertaintySampling, EntropySampling,
                          LeastConfidenceSampling, MarginSampling)

logger = logging.getLogger(__name__)


class HybridQueryStrategy(QueryStrategy, DiversityMixin):
    """
    Hybrid query strategy combining uncertainty and diversity sampling
    """

    def __init__(
        self,
        uncertainty_strategy: str = "entropy",
        diversity_strategy: str = "clustering",
        uncertainty_weight: float = 0.7,
        diversity_weight: float = 0.3,
        batch_size: int = 10,
        device: str = "cpu",
    ):
        """
        Initialize hybrid query strategy

        Args:
            uncertainty_strategy: Type of uncertainty sampling
            diversity_strategy: Type of diversity sampling
            uncertainty_weight: Weight for uncertainty component
            diversity_weight: Weight for diversity component
            batch_size: Batch size for selection
            device: Device for computation
        """
        self.uncertainty_strategy = uncertainty_strategy
        self.diversity_strategy = diversity_strategy
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.batch_size = batch_size
        self.device = device

        # Validate weights
        if abs(uncertainty_weight + diversity_weight - 1.0) > 1e-6:
            logger.warning("Weights don't sum to 1.0, normalizing...")
            total = uncertainty_weight + diversity_weight
            self.uncertainty_weight = uncertainty_weight / total
            self.diversity_weight = diversity_weight / total

    def _get_uncertainty_sampler(self, model) -> Any:
        """Get uncertainty sampler instance"""
        samplers = {
            "entropy": EntropySampling,
            "least_confidence": LeastConfidenceSampling,
            "margin": MarginSampling,
            "bayesian": BayesianUncertaintySampling,
        }

        sampler_class = samplers.get(self.uncertainty_strategy, EntropySampling)
        return sampler_class(model, self.device)

    def _get_diversity_sampler(self, model) -> Any:
        """Get diversity sampler instance"""
        samplers = {
            "clustering": ClusteringSampling,
            "coreset": CoreSetSampling,
            "representative": RepresentativeSampling,
        }

        sampler_class = samplers.get(self.diversity_strategy, ClusteringSampling)
        return sampler_class(model, self.device)

    def query(
        self,
        unlabeled_data: Dataset,
        n_samples: int,
        model: Any,
        labeled_data: Optional[Dataset] = None,
    ) -> List[int]:
        """
        Query samples using hybrid strategy

        Args:
            unlabeled_data: Unlabeled dataset
            n_samples: Number of samples to query
            model: Trained model
            labeled_data: Already labeled data

        Returns:
            Indices of selected samples
        """
        if n_samples <= 0:
            return []

        # If only one sample needed, use pure uncertainty
        if n_samples == 1:
            uncertainty_sampler = self._get_uncertainty_sampler(model)
            return uncertainty_sampler.select_samples(unlabeled_data, 1, labeled_data)

        # For batch selection, use hybrid approach
        return self._hybrid_batch_selection(unlabeled_data, n_samples, model, labeled_data)

    def _hybrid_batch_selection(
        self,
        unlabeled_data: Dataset,
        n_samples: int,
        model: Any,
        labeled_data: Optional[Dataset] = None,
    ) -> List[int]:
        """
        Perform hybrid batch selection

        Args:
            unlabeled_data: Unlabeled dataset
            n_samples: Number of samples to select
            model: Trained model
            labeled_data: Already labeled data

        Returns:
            Selected sample indices
        """
        # Step 1: Get uncertainty scores for all samples
        uncertainty_sampler = self._get_uncertainty_sampler(model)
        _, probabilities = uncertainty_sampler.get_predictions(unlabeled_data)

        # Calculate uncertainty scores based on strategy
        if self.uncertainty_strategy == "entropy":
            uncertainty_scores = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        elif self.uncertainty_strategy == "least_confidence":
            uncertainty_scores = 1.0 - np.max(probabilities, axis=1)
        elif self.uncertainty_strategy == "margin":
            sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
            uncertainty_scores = 1.0 - (sorted_probs[:, 0] - sorted_probs[:, 1])
        else:  # bayesian
            if hasattr(uncertainty_sampler, "get_bayesian_predictions"):
                _, uncertainty_scores = uncertainty_sampler.get_bayesian_predictions(unlabeled_data)
            else:
                # Fallback to entropy
                uncertainty_scores = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)

        # Normalize uncertainty scores
        uncertainty_scores = (uncertainty_scores - uncertainty_scores.min()) / (
            uncertainty_scores.max() - uncertainty_scores.min() + 1e-10
        )

        # Step 2: Get diversity sampler for feature extraction
        diversity_sampler = self._get_diversity_sampler(model)
        features = diversity_sampler.get_features(unlabeled_data)

        # Step 3: Greedy selection combining uncertainty and diversity
        selected_indices = []
        remaining_indices = list(range(len(unlabeled_data)))

        for _i in range(n_samples):
            if not remaining_indices:
                break

            best_score = -1
            best_idx = None

            for idx in remaining_indices:
                # Uncertainty component
                uncertainty_component = uncertainty_scores[idx] * self.uncertainty_weight

                # Diversity component
                if selected_indices:
                    # Calculate minimum distance to selected samples
                    selected_features = features[selected_indices]
                    current_feature = features[idx : idx + 1]

                    distances = np.sqrt(np.sum((selected_features - current_feature) ** 2, axis=1))
                    min_distance = np.min(distances)
                    # Normalize by average feature norm
                    avg_norm = np.mean(np.linalg.norm(features, axis=1))
                    diversity_component = (
                        min_distance / (avg_norm + 1e-10)
                    ) * self.diversity_weight
                else:
                    # First sample, only uncertainty matters
                    diversity_component = 0.0

                # Combined score
                combined_score = uncertainty_component + diversity_component

                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        return selected_indices


class AdaptiveQueryStrategy(HybridQueryStrategy):
    """
    Adaptive query strategy that adjusts weights based on learning progress
    """

    def __init__(
        self,
        initial_uncertainty_weight: float = 0.8,
        initial_diversity_weight: float = 0.2,
        adaptation_rate: float = 0.1,
        min_uncertainty_weight: float = 0.3,
        max_uncertainty_weight: float = 0.9,
        **kwargs,
    ):
        """
        Initialize adaptive query strategy

        Args:
            initial_uncertainty_weight: Initial weight for uncertainty
            initial_diversity_weight: Initial weight for diversity
            adaptation_rate: Rate of adaptation
            min_uncertainty_weight: Minimum uncertainty weight
            max_uncertainty_weight: Maximum uncertainty weight
        """
        super().__init__(
            uncertainty_weight=initial_uncertainty_weight,
            diversity_weight=initial_diversity_weight,
            **kwargs,
        )
        self.initial_uncertainty_weight = initial_uncertainty_weight
        self.initial_diversity_weight = initial_diversity_weight
        self.adaptation_rate = adaptation_rate
        self.min_uncertainty_weight = min_uncertainty_weight
        self.max_uncertainty_weight = max_uncertainty_weight
        self.performance_history = []

    def adapt_weights(self, current_f1: float):
        """
        Adapt weights based on performance

        Args:
            current_f1: Current F1 score
        """
        self.performance_history.append(current_f1)

        if len(self.performance_history) < 2:
            return  # Need at least 2 points to calculate improvement

        # Calculate improvement rate
        recent_improvement = self.performance_history[-1] - self.performance_history[-2]

        # Adapt weights based on improvement
        if recent_improvement > 0:
            # Good improvement, slightly favor diversity for exploration
            adjustment = self.adaptation_rate * recent_improvement
            new_uncertainty_weight = max(
                self.min_uncertainty_weight, self.uncertainty_weight - adjustment
            )
        else:
            # Poor improvement, favor uncertainty for exploitation
            adjustment = self.adaptation_rate * abs(recent_improvement)
            new_uncertainty_weight = min(
                self.max_uncertainty_weight, self.uncertainty_weight + adjustment
            )

        # Update weights
        self.uncertainty_weight = new_uncertainty_weight
        self.diversity_weight = 1.0 - new_uncertainty_weight

        logger.info(
            f"Adapted weights: uncertainty={self.uncertainty_weight:.3f}, "
            f"diversity={self.diversity_weight:.3f}"
        )

    def query(
        self,
        unlabeled_data: Dataset,
        n_samples: int,
        model: Any,
        labeled_data: Optional[Dataset] = None,
    ) -> List[int]:
        """Query with adaptive weights"""
        return super().query(unlabeled_data, n_samples, model, labeled_data)


class BudgetAwareQueryStrategy(HybridQueryStrategy):
    """
    Budget-aware query strategy that optimizes selection based on annotation budget
    """

    def __init__(
        self,
        total_budget: int,
        budget_stages: List[float] = None,
        stage_strategies: List[Dict] = None,
        **kwargs,
    ):
        """
        Initialize budget-aware strategy

        Args:
            total_budget: Total annotation budget
            budget_stages: Budget milestones (as fractions)
            stage_strategies: Strategy configurations for each stage
        """
        if budget_stages is None:
            budget_stages = [0.3, 0.6, 1.0]
        super().__init__(**kwargs)
        self.total_budget = total_budget
        self.budget_stages = budget_stages
        self.current_budget = 0

        # Default strategies for different stages
        if stage_strategies is None:
            self.stage_strategies = [
                {"uncertainty_weight": 0.9, "diversity_weight": 0.1},  # Early: exploitation
                {"uncertainty_weight": 0.5, "diversity_weight": 0.5},  # Middle: balanced
                {"uncertainty_weight": 0.3, "diversity_weight": 0.7},  # Late: exploration
            ]
        else:
            self.stage_strategies = stage_strategies

    def update_budget(self, used_samples: int):
        """Update current budget usage"""
        self.current_budget += used_samples

    def get_current_stage(self) -> int:
        """Get current budget stage"""
        budget_fraction = self.current_budget / self.total_budget

        for i, stage_threshold in enumerate(self.budget_stages):
            if budget_fraction <= stage_threshold:
                return i

        return len(self.budget_stages) - 1

    def query(
        self,
        unlabeled_data: Dataset,
        n_samples: int,
        model: Any,
        labeled_data: Optional[Dataset] = None,
    ) -> List[int]:
        """Query with budget-aware strategy"""
        # Update strategy based on current stage
        current_stage = self.get_current_stage()
        stage_config = self.stage_strategies[current_stage]

        # Temporarily update weights
        old_uncertainty_weight = self.uncertainty_weight
        old_diversity_weight = self.diversity_weight

        self.uncertainty_weight = stage_config["uncertainty_weight"]
        self.diversity_weight = stage_config["diversity_weight"]

        logger.info(
            f"Budget stage {current_stage + 1}/{len(self.budget_stages)} "
            f"({self.current_budget}/{self.total_budget} samples used)"
        )

        # Perform selection
        selected_indices = super().query(unlabeled_data, n_samples, model, labeled_data)

        # Restore original weights
        self.uncertainty_weight = old_uncertainty_weight
        self.diversity_weight = old_diversity_weight

        return selected_indices
