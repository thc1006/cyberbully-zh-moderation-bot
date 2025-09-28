"""
Diversity sampling strategies for active learning
"""

import logging
from typing import List, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from torch.utils.data import DataLoader, Dataset

from .base import ActiveLearner

logger = logging.getLogger(__name__)


class ClusteringSampling(ActiveLearner):
    """Clustering-based diversity sampling"""

    def __init__(self, model, device: str = "cpu", n_clusters: Optional[int] = None):
        """
        Initialize clustering sampler

        Args:
            model: PyTorch model for feature extraction
            device: Device for computation
            n_clusters: Number of clusters (auto if None)
        """
        super().__init__(model, device)
        self.n_clusters = n_clusters

    def get_features(self, data: Dataset) -> np.ndarray:
        """
        Extract features from model's hidden layers

        Args:
            data: Dataset to extract features from

        Returns:
            Feature embeddings
        """
        self.model.eval()
        dataloader = DataLoader(data, batch_size=32, shuffle=False)
        all_features = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Get hidden states from the model
                outputs = self.model(
                    input_ids=inputs, attention_mask=attention_mask, output_hidden_states=True
                )

                # Use the [CLS] token embedding from the last hidden layer
                features = outputs.hidden_states[-1][:, 0, :]  # [batch_size, hidden_size]
                all_features.append(features.cpu().numpy())

        return np.concatenate(all_features)

    def select_samples(
        self, unlabeled_data: Dataset, n_samples: int, labeled_data: Optional[Dataset] = None
    ) -> List[int]:
        """
        Select diverse samples using clustering

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Already labeled data for diversity check

        Returns:
            List of diverse sample indices
        """
        # Extract features
        features = self.get_features(unlabeled_data)

        # Determine number of clusters
        n_clusters = self.n_clusters or min(n_samples * 2, len(features))

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)

        # Select samples closest to cluster centers
        selected_indices = []
        cluster_centers = kmeans.cluster_centers_

        for _i in range(min(n_samples, n_clusters)):
            # Find cluster with most samples if we need fewer than n_clusters
            if len(selected_indices) < n_samples:
                cluster_sizes = [(j, np.sum(cluster_labels == j)) for j in range(n_clusters)]
                cluster_sizes.sort(key=lambda x: x[1], reverse=True)

                for cluster_id, _ in cluster_sizes:
                    if cluster_id not in [kmeans.labels_[idx] for idx in selected_indices]:
                        # Find sample closest to this cluster center
                        cluster_mask = cluster_labels == cluster_id
                        cluster_features = features[cluster_mask]
                        cluster_indices = np.where(cluster_mask)[0]

                        distances = euclidean_distances(
                            cluster_features, cluster_centers[cluster_id].reshape(1, -1)
                        ).flatten()

                        closest_idx = cluster_indices[np.argmin(distances)]
                        selected_indices.append(closest_idx)
                        break

        return selected_indices[:n_samples]


class CoreSetSampling(ActiveLearner):
    """CoreSet-based diversity sampling"""

    def __init__(self, model, device: str = "cpu", distance_metric: str = "cosine"):
        """
        Initialize CoreSet sampler

        Args:
            model: PyTorch model for feature extraction
            device: Device for computation
            distance_metric: Distance metric ('cosine' or 'euclidean')
        """
        super().__init__(model, device)
        self.distance_metric = distance_metric

    def get_features(self, data: Dataset) -> np.ndarray:
        """Extract features for CoreSet selection"""
        self.model.eval()
        dataloader = DataLoader(data, batch_size=32, shuffle=False)
        all_features = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=inputs, attention_mask=attention_mask, output_hidden_states=True
                )
                features = outputs.hidden_states[-1][:, 0, :]
                all_features.append(features.cpu().numpy())

        return np.concatenate(all_features)

    def compute_distances(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute distances between feature sets"""
        if self.distance_metric == "cosine":
            return cosine_distances(X, Y)
        else:
            return euclidean_distances(X, Y)

    def select_samples(
        self, unlabeled_data: Dataset, n_samples: int, labeled_data: Optional[Dataset] = None
    ) -> List[int]:
        """
        Select samples using CoreSet greedy selection

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Already labeled data

        Returns:
            List of CoreSet sample indices
        """
        unlabeled_features = self.get_features(unlabeled_data)

        if labeled_data is not None:
            labeled_features = self.get_features(labeled_data)
            # Combine labeled and unlabeled for CoreSet computation
            all_features = np.vstack([labeled_features, unlabeled_features])
            labeled_size = len(labeled_features)
        else:
            all_features = unlabeled_features
            labeled_size = 0

        # Greedy CoreSet selection
        selected_indices = []

        if labeled_size > 0:
            # Start with labeled data as initial CoreSet
            remaining_indices = list(range(labeled_size, len(all_features)))
        else:
            # Start with random sample if no labeled data
            first_idx = np.random.randint(0, len(all_features))
            selected_indices.append(first_idx)
            remaining_indices = [i for i in range(len(all_features)) if i != first_idx]

        # Greedy selection
        for _ in range(n_samples - len(selected_indices)):
            if not remaining_indices:
                break

            best_idx = None
            best_distance = -1

            for idx in remaining_indices:
                # Calculate minimum distance to already selected samples
                if labeled_size > 0 or selected_indices:
                    if labeled_size > 0:
                        selected_for_distance = list(range(labeled_size)) + selected_indices
                    else:
                        selected_for_distance = selected_indices

                    distances = self.compute_distances(
                        all_features[idx : idx + 1], all_features[selected_for_distance]
                    )
                    min_distance = np.min(distances)
                else:
                    min_distance = float("inf")

                if min_distance > best_distance:
                    best_distance = min_distance
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        # Convert to unlabeled indices
        if labeled_size > 0:
            unlabeled_indices = [
                idx - labeled_size for idx in selected_indices if idx >= labeled_size
            ]
        else:
            unlabeled_indices = selected_indices

        return unlabeled_indices[:n_samples]


class RepresentativeSampling(ActiveLearner):
    """Representative sampling based on feature space coverage"""

    def __init__(self, model, device: str = "cpu", coverage_method: str = "pca"):
        """
        Initialize representative sampler

        Args:
            model: PyTorch model for feature extraction
            device: Device for computation
            coverage_method: Method for space coverage ('pca' or 'random')
        """
        super().__init__(model, device)
        self.coverage_method = coverage_method

    def get_features(self, data: Dataset) -> np.ndarray:
        """Extract features for representative selection"""
        self.model.eval()
        dataloader = DataLoader(data, batch_size=32, shuffle=False)
        all_features = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=inputs, attention_mask=attention_mask, output_hidden_states=True
                )
                features = outputs.hidden_states[-1][:, 0, :]
                all_features.append(features.cpu().numpy())

        return np.concatenate(all_features)

    def select_samples(
        self, unlabeled_data: Dataset, n_samples: int, labeled_data: Optional[Dataset] = None
    ) -> List[int]:
        """
        Select representative samples covering feature space

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Not used in this strategy

        Returns:
            List of representative sample indices
        """
        features = self.get_features(unlabeled_data)

        if self.coverage_method == "pca":
            # Use PCA to find principal directions
            pca = PCA(n_components=min(50, features.shape[1]))
            features_pca = pca.fit_transform(features)

            # Divide feature space into grid
            selected_indices = []
            n_dims = min(3, features_pca.shape[1])  # Use top 3 PCA components

            # Create grid points
            grid_size = int(np.ceil(n_samples ** (1 / n_dims)))
            grid_points = []

            for dim in range(n_dims):
                min_val, max_val = features_pca[:, dim].min(), features_pca[:, dim].max()
                grid_points.append(np.linspace(min_val, max_val, grid_size))

            # Find closest sample to each grid point
            import itertools

            grid_combinations = list(itertools.product(*grid_points))[:n_samples]

            for grid_point in grid_combinations:
                # Find closest sample to this grid point
                distances = np.sum((features_pca[:, :n_dims] - np.array(grid_point)) ** 2, axis=1)
                closest_idx = np.argmin(distances)

                if closest_idx not in selected_indices:
                    selected_indices.append(closest_idx)

            # If we don't have enough samples, add random ones
            remaining = [i for i in range(len(features)) if i not in selected_indices]
            while len(selected_indices) < n_samples and remaining:
                idx = np.random.choice(remaining)
                selected_indices.append(idx)
                remaining.remove(idx)

        else:  # random sampling as baseline
            selected_indices = np.random.choice(
                len(features), size=min(n_samples, len(features)), replace=False
            ).tolist()

        return selected_indices[:n_samples]


class DiversityMixin:
    """Mixin class for diversity calculations"""

    @staticmethod
    def calculate_diversity_score(features: np.ndarray, indices: List[int]) -> float:
        """
        Calculate diversity score for selected samples

        Args:
            features: Feature matrix
            indices: Selected sample indices

        Returns:
            Diversity score (higher = more diverse)
        """
        if len(indices) < 2:
            return 0.0

        selected_features = features[indices]
        distances = euclidean_distances(selected_features)

        # Average pairwise distance as diversity measure
        n = len(indices)
        total_distance = np.sum(distances) - np.trace(distances)  # Exclude diagonal
        diversity = total_distance / (n * (n - 1))

        return diversity

    @staticmethod
    def calculate_coverage_score(all_features: np.ndarray, selected_indices: List[int]) -> float:
        """
        Calculate feature space coverage score

        Args:
            all_features: All feature vectors
            selected_indices: Selected sample indices

        Returns:
            Coverage score (higher = better coverage)
        """
        if not selected_indices:
            return 0.0

        selected_features = all_features[selected_indices]

        # Calculate percentage of feature space covered
        # Use convex hull volume approximation
        try:
            from scipy.spatial import ConvexHull

            if len(selected_features) > all_features.shape[1]:  # Need more points than dimensions
                hull = ConvexHull(selected_features)
                coverage = hull.volume
            else:
                # Fallback: use variance as coverage proxy
                coverage = np.mean(np.var(selected_features, axis=0))
        except:
            # Fallback: use variance as coverage proxy
            coverage = np.mean(np.var(selected_features, axis=0))

        return coverage
