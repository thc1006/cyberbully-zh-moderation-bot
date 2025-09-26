"""
Hybrid query strategies combining uncertainty and diversity sampling
"""

import numpy as np
from typing import List, Optional, Dict, Any
from torch.utils.data import Dataset
import logging

from .base import ActiveLearner
from .uncertainty_enhanced import (
    EntropySampling, LeastConfidenceSampling, MarginSampling,
    BayesianUncertaintySampling, BALD
)
from .diversity_enhanced import (
    ClusteringSampling, CoreSetSampling, RepresentativeSampling,
    DiversityClusteringHybrid
)

logger = logging.getLogger(__name__)


class HybridQueryStrategy(ActiveLearner):
    """Hybrid strategy combining uncertainty and diversity sampling"""

    def __init__(self, model, device: str = 'cpu',
                 uncertainty_strategy: str = 'entropy',
                 diversity_strategy: str = 'clustering',
                 uncertainty_ratio: float = 0.5,
                 **strategy_kwargs):
        """
        Initialize hybrid query strategy

        Args:
            model: PyTorch model
            device: Device for computation
            uncertainty_strategy: 'entropy', 'confidence', 'margin', 'bayesian', 'bald'
            diversity_strategy: 'clustering', 'coreset', 'representative', 'hybrid'
            uncertainty_ratio: Ratio of samples selected by uncertainty (0-1)
            **strategy_kwargs: Additional arguments for specific strategies
        """
        super().__init__(model, device)
        self.uncertainty_ratio = uncertainty_ratio

        # Initialize uncertainty sampler
        if uncertainty_strategy == 'entropy':
            self.uncertainty_sampler = EntropySampling(model, device)
        elif uncertainty_strategy == 'confidence':
            self.uncertainty_sampler = LeastConfidenceSampling(model, device)
        elif uncertainty_strategy == 'margin':
            self.uncertainty_sampler = MarginSampling(model, device)
        elif uncertainty_strategy == 'bayesian':
            n_dropout = strategy_kwargs.get('n_dropout_samples', 10)
            self.uncertainty_sampler = BayesianUncertaintySampling(
                model, device, n_dropout
            )
        elif uncertainty_strategy == 'bald':
            n_dropout = strategy_kwargs.get('n_dropout_samples', 10)
            self.uncertainty_sampler = BALD(model, device, n_dropout)
        else:
            raise ValueError(f"Unknown uncertainty strategy: {uncertainty_strategy}")

        # Initialize diversity sampler
        if diversity_strategy == 'clustering':
            n_clusters = strategy_kwargs.get('n_clusters', None)
            distance_metric = strategy_kwargs.get('distance_metric', 'euclidean')
            self.diversity_sampler = ClusteringSampling(
                model, device, n_clusters, distance_metric
            )
        elif diversity_strategy == 'coreset':
            distance_metric = strategy_kwargs.get('distance_metric', 'euclidean')
            self.diversity_sampler = CoreSetSampling(model, device, distance_metric)
        elif diversity_strategy == 'representative':
            use_pca = strategy_kwargs.get('use_pca', True)
            pca_components = strategy_kwargs.get('pca_components', 50)
            self.diversity_sampler = RepresentativeSampling(
                model, device, use_pca, pca_components
            )
        elif diversity_strategy == 'hybrid':
            clustering_ratio = strategy_kwargs.get('clustering_ratio', 0.6)
            self.diversity_sampler = DiversityClusteringHybrid(
                model, device, clustering_ratio
            )
        else:
            raise ValueError(f"Unknown diversity strategy: {diversity_strategy}")

        self.uncertainty_strategy_name = uncertainty_strategy
        self.diversity_strategy_name = diversity_strategy

    def select_samples(self,
                      unlabeled_data: Dataset,
                      n_samples: int,
                      labeled_data: Optional[Dataset] = None) -> List[int]:
        """
        Select samples using hybrid uncertainty + diversity approach

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Already labeled data

        Returns:
            List of selected sample indices
        """
        logger.info(f"Hybrid selection: {self.uncertainty_strategy_name} + {self.diversity_strategy_name}")

        # Calculate number of samples for each strategy
        n_uncertainty = int(n_samples * self.uncertainty_ratio)
        n_diversity = n_samples - n_uncertainty

        selected_indices = []

        # Select uncertain samples
        if n_uncertainty > 0:
            uncertainty_indices = self.uncertainty_sampler.select_samples(
                unlabeled_data, n_uncertainty, labeled_data
            )
            selected_indices.extend(uncertainty_indices)
            logger.info(f"Selected {len(uncertainty_indices)} uncertain samples")

        # Select diverse samples (excluding already selected)
        if n_diversity > 0:
            # Create filtered dataset excluding uncertainty selections
            remaining_indices = [i for i in range(len(unlabeled_data))
                               if i not in selected_indices]

            if len(remaining_indices) > 0:
                from torch.utils.data import Subset
                remaining_dataset = Subset(unlabeled_data, remaining_indices)

                diversity_sub_indices = self.diversity_sampler.select_samples(
                    remaining_dataset, min(n_diversity, len(remaining_indices)),
                    labeled_data
                )

                # Map back to original indices
                diversity_indices = [remaining_indices[i] for i in diversity_sub_indices]
                selected_indices.extend(diversity_indices)
                logger.info(f"Selected {len(diversity_indices)} diverse samples")

        logger.info(f"Total selected: {len(selected_indices)} samples")
        return selected_indices


class AdaptiveQueryStrategy(HybridQueryStrategy):
    """Adaptive strategy that adjusts uncertainty/diversity ratio based on performance"""

    def __init__(self, model, device: str = 'cpu',
                 initial_uncertainty_ratio: float = 0.5,
                 adaptation_rate: float = 0.1,
                 **strategy_kwargs):
        """
        Initialize adaptive query strategy

        Args:
            model: PyTorch model
            device: Device for computation
            initial_uncertainty_ratio: Initial ratio for uncertainty sampling
            adaptation_rate: How quickly to adapt the ratio (0-1)
            **strategy_kwargs: Additional arguments for strategies
        """
        super().__init__(model, device, 'entropy', 'clustering',
                        initial_uncertainty_ratio, **strategy_kwargs)
        self.initial_uncertainty_ratio = initial_uncertainty_ratio
        self.adaptation_rate = adaptation_rate
        self.performance_history = []

    def update_performance(self, f1_score: float):
        """
        Update performance history for adaptation

        Args:
            f1_score: Current model F1 score
        """
        self.performance_history.append(f1_score)
        logger.info(f"Updated performance history: F1 = {f1_score:.4f}")

    def adapt_ratio(self):
        """Adapt uncertainty ratio based on recent performance"""
        if len(self.performance_history) < 2:
            return

        # If performance is improving, continue current strategy
        # If stagnating, shift toward diversity
        recent_trend = (self.performance_history[-1] -
                       self.performance_history[-2])

        if recent_trend < 0.01:  # Performance stagnating
            # Increase diversity (decrease uncertainty ratio)
            self.uncertainty_ratio = max(
                0.2, self.uncertainty_ratio - self.adaptation_rate
            )
            logger.info(f"Performance stagnating, increasing diversity. "
                       f"New ratio: {self.uncertainty_ratio:.2f}")
        elif recent_trend > 0.05:  # Good improvement
            # Maintain or slightly increase uncertainty
            self.uncertainty_ratio = min(
                0.8, self.uncertainty_ratio + self.adaptation_rate * 0.5
            )
            logger.info(f"Good improvement, adjusting uncertainty. "
                       f"New ratio: {self.uncertainty_ratio:.2f}")

    def select_samples(self,
                      unlabeled_data: Dataset,
                      n_samples: int,
                      labeled_data: Optional[Dataset] = None) -> List[int]:
        """Select samples with adaptive ratio"""
        self.adapt_ratio()
        return super().select_samples(unlabeled_data, n_samples, labeled_data)


class MultiStrategyEnsemble(ActiveLearner):
    """Ensemble of multiple query strategies with voting"""

    def __init__(self, model, device: str = 'cpu',
                 strategies: Optional[List[Dict[str, Any]]] = None,
                 voting_method: str = 'intersection'):
        """
        Initialize multi-strategy ensemble

        Args:
            model: PyTorch model
            device: Device for computation
            strategies: List of strategy configurations
            voting_method: 'intersection', 'union', 'weighted'
        """
        super().__init__(model, device)
        self.voting_method = voting_method

        if strategies is None:
            # Default ensemble
            strategies = [
                {'uncertainty': 'entropy', 'diversity': 'clustering', 'ratio': 0.5},
                {'uncertainty': 'bald', 'diversity': 'coreset', 'ratio': 0.6},
                {'uncertainty': 'margin', 'diversity': 'representative', 'ratio': 0.4}
            ]

        self.strategies = []
        for config in strategies:
            strategy = HybridQueryStrategy(
                model, device,
                config['uncertainty'], config['diversity'],
                config['ratio']
            )
            self.strategies.append(strategy)

        logger.info(f"Initialized ensemble with {len(self.strategies)} strategies")

    def select_samples(self,
                      unlabeled_data: Dataset,
                      n_samples: int,
                      labeled_data: Optional[Dataset] = None) -> List[int]:
        """
        Select samples using ensemble voting

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Already labeled data

        Returns:
            List of selected sample indices
        """
        all_selections = []

        # Get selections from each strategy
        for i, strategy in enumerate(self.strategies):
            selections = strategy.select_samples(
                unlabeled_data, n_samples, labeled_data
            )
            all_selections.append(set(selections))
            logger.debug(f"Strategy {i} selected {len(selections)} samples")

        # Combine selections based on voting method
        if self.voting_method == 'intersection':
            # Samples selected by multiple strategies
            combined = set.intersection(*all_selections)
            if len(combined) < n_samples:
                # Add most frequently selected samples
                vote_counts = {}
                for selections in all_selections:
                    for idx in selections:
                        vote_counts[idx] = vote_counts.get(idx, 0) + 1

                sorted_by_votes = sorted(vote_counts.items(),
                                       key=lambda x: x[1], reverse=True)
                combined = set([idx for idx, _ in sorted_by_votes[:n_samples]])

        elif self.voting_method == 'union':
            # All samples selected by any strategy
            combined = set.union(*all_selections)
            if len(combined) > n_samples:
                # Prioritize by vote count
                vote_counts = {}
                for selections in all_selections:
                    for idx in selections:
                        vote_counts[idx] = vote_counts.get(idx, 0) + 1

                sorted_by_votes = sorted(vote_counts.items(),
                                       key=lambda x: x[1], reverse=True)
                combined = set([idx for idx, _ in sorted_by_votes[:n_samples]])

        else:  # weighted
            # Weight by strategy performance (if available)
            vote_counts = {}
            for selections in all_selections:
                weight = 1.0  # Could be adjusted based on strategy performance
                for idx in selections:
                    vote_counts[idx] = vote_counts.get(idx, 0) + weight

            sorted_by_votes = sorted(vote_counts.items(),
                                   key=lambda x: x[1], reverse=True)
            combined = set([idx for idx, _ in sorted_by_votes[:n_samples]])

        selected_indices = list(combined)
        logger.info(f"Ensemble selected {len(selected_indices)} samples using {self.voting_method}")

        return selected_indices