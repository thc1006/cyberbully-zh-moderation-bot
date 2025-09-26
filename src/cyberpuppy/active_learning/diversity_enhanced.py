"""
Enhanced diversity sampling strategies for active learning
"""

import numpy as np
import torch
from typing import List, Optional
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
import logging

from .base import ActiveLearner

logger = logging.getLogger(__name__)


class ClusteringSampling(ActiveLearner):
    """K-means clustering-based diversity sampling"""

    def __init__(self, model, device: str = 'cpu', n_clusters: Optional[int] = None,
                 distance_metric: str = 'euclidean'):
        """
        Initialize clustering sampler

        Args:
            model: PyTorch model for feature extraction
            device: Device for computation
            n_clusters: Number of clusters (auto if None)
            distance_metric: Distance metric ('euclidean' or 'cosine')
        """
        super().__init__(model, device)
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric

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
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Get hidden states from the model
                if hasattr(self.model, 'bert') or hasattr(self.model, 'roberta'):
                    outputs = self.model(input_ids=inputs, attention_mask=attention_mask,
                                       output_hidden_states=True)
                    # Use the [CLS] token embedding from the last hidden layer
                    features = outputs.hidden_states[-1][:, 0, :]
                else:
                    # For other models, use the final layer output
                    outputs = self.model(input_ids=inputs, attention_mask=attention_mask)
                    if hasattr(outputs, 'last_hidden_state'):
                        features = outputs.last_hidden_state.mean(dim=1)
                    else:
                        features = outputs.mean(dim=1)

                all_features.append(features.cpu().numpy())

        return np.concatenate(all_features)

    def select_samples(self,
                      unlabeled_data: Dataset,
                      n_samples: int,
                      labeled_data: Optional[Dataset] = None) -> List[int]:
        """
        Select diverse samples using K-means clustering

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Already labeled data for diversity check

        Returns:
            List of diverse sample indices
        """
        # Extract features
        features = self.get_features(unlabeled_data)
        logger.info(f"Extracted features with shape: {features.shape}")

        # Determine number of clusters
        n_clusters = self.n_clusters or min(n_samples * 2, len(features))
        n_clusters = min(n_clusters, len(features))

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)

        # Select representative samples from each cluster
        selected_indices = []
        cluster_centers = kmeans.cluster_centers_

        # Calculate distances based on metric
        if self.distance_metric == 'cosine':
            distances = cosine_distances(features, cluster_centers)
        else:
            distances = euclidean_distances(features, cluster_centers)

        # For each cluster, find the sample closest to the center
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if np.any(cluster_mask):
                cluster_indices = np.where(cluster_mask)[0]
                cluster_distances = distances[cluster_indices, cluster_id]
                closest_idx = cluster_indices[np.argmin(cluster_distances)]
                selected_indices.append(closest_idx)

        # If we need more samples, select second-closest in largest clusters
        while len(selected_indices) < n_samples and len(selected_indices) < len(features):
            # Find largest clusters not yet fully sampled
            cluster_sizes = [(i, np.sum(cluster_labels == i)) for i in range(n_clusters)]
            cluster_sizes.sort(key=lambda x: x[1], reverse=True)

            for cluster_id, size in cluster_sizes:
                if size > 1:  # Only consider clusters with multiple samples
                    cluster_mask = cluster_labels == cluster_id
                    cluster_indices = np.where(cluster_mask)[0]

                    # Remove already selected indices
                    available_indices = [idx for idx in cluster_indices
                                       if idx not in selected_indices]

                    if available_indices:
                        cluster_distances = distances[available_indices, cluster_id]
                        closest_idx = available_indices[np.argmin(cluster_distances)]
                        selected_indices.append(closest_idx)
                        break

            if len(set(selected_indices)) == len(selected_indices):
                break

        # Ensure we have unique indices and limit to n_samples
        selected_indices = list(set(selected_indices))[:n_samples]

        logger.info(f"Selected {len(selected_indices)} diverse samples using K-means clustering")
        logger.debug(f"Cluster distribution: {np.bincount(cluster_labels)}")

        return selected_indices


class CoreSetSampling(ActiveLearner):
    """CoreSet selection for diversity sampling"""

    def __init__(self, model, device: str = 'cpu', distance_metric: str = 'euclidean'):
        """
        Initialize CoreSet sampler

        Args:
            model: PyTorch model for feature extraction
            device: Device for computation
            distance_metric: Distance metric ('euclidean' or 'cosine')
        """
        super().__init__(model, device)
        self.distance_metric = distance_metric

    def get_features(self, data: Dataset) -> np.ndarray:
        """Extract features similar to ClusteringSampling"""
        self.model.eval()
        dataloader = DataLoader(data, batch_size=32, shuffle=False)
        all_features = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                if hasattr(self.model, 'bert') or hasattr(self.model, 'roberta'):
                    outputs = self.model(input_ids=inputs, attention_mask=attention_mask,
                                       output_hidden_states=True)
                    features = outputs.hidden_states[-1][:, 0, :]
                else:
                    outputs = self.model(input_ids=inputs, attention_mask=attention_mask)
                    if hasattr(outputs, 'last_hidden_state'):
                        features = outputs.last_hidden_state.mean(dim=1)
                    else:
                        features = outputs.mean(dim=1)

                all_features.append(features.cpu().numpy())

        return np.concatenate(all_features)

    def greedy_coreset_selection(self, features: np.ndarray, n_samples: int,
                                labeled_features: Optional[np.ndarray] = None) -> List[int]:
        """
        Greedy CoreSet selection algorithm

        Args:
            features: Feature matrix of unlabeled samples
            n_samples: Number of samples to select
            labeled_features: Features of already labeled samples

        Returns:
            List of selected indices
        """
        n_total = len(features)
        if n_samples >= n_total:
            return list(range(n_total))

        selected_indices = []

        # Initialize with labeled features if available
        if labeled_features is not None:
            all_selected_features = [labeled_features]
        else:
            all_selected_features = []

        # Calculate distance matrix
        if self.distance_metric == 'cosine':
            distance_fn = lambda x, y: cosine_distances(x.reshape(1, -1), y.reshape(1, -1))[0, 0]
        else:
            distance_fn = lambda x, y: np.linalg.norm(x - y)

        for _ in range(n_samples):
            max_min_distance = -1
            best_idx = -1

            for i in range(n_total):
                if i in selected_indices:
                    continue

                # Calculate minimum distance to all selected samples
                min_distance = float('inf')

                # Check distance to labeled samples
                for selected_features in all_selected_features:
                    if len(selected_features) > 0:
                        for j in range(len(selected_features)):
                            dist = distance_fn(features[i], selected_features[j])
                            min_distance = min(min_distance, dist)

                # Check distance to already selected unlabeled samples
                for j in selected_indices:
                    dist = distance_fn(features[i], features[j])
                    min_distance = min(min_distance, dist)

                # Update best candidate
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = i

            if best_idx != -1:
                selected_indices.append(best_idx)

        return selected_indices

    def select_samples(self,
                      unlabeled_data: Dataset,
                      n_samples: int,
                      labeled_data: Optional[Dataset] = None) -> List[int]:
        """
        Select diverse samples using CoreSet selection

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Already labeled data for diversity

        Returns:
            List of diverse sample indices
        """
        features = self.get_features(unlabeled_data)
        logger.info(f"Extracted features with shape: {features.shape}")

        labeled_features = None
        if labeled_data is not None:
            labeled_features = self.get_features(labeled_data)
            logger.info(f"Using {len(labeled_features)} labeled samples for CoreSet")

        selected_indices = self.greedy_coreset_selection(
            features, n_samples, labeled_features
        )

        logger.info(f"Selected {len(selected_indices)} diverse samples using CoreSet")

        return selected_indices


class RepresentativeSampling(ActiveLearner):
    """Representative sampling based on feature centrality"""

    def __init__(self, model, device: str = 'cpu', use_pca: bool = True,
                 pca_components: int = 50):
        """
        Initialize representative sampler

        Args:
            model: PyTorch model for feature extraction
            device: Device for computation
            use_pca: Whether to use PCA for dimensionality reduction
            pca_components: Number of PCA components
        """
        super().__init__(model, device)
        self.use_pca = use_pca
        self.pca_components = pca_components

    def get_features(self, data: Dataset) -> np.ndarray:
        """Extract features similar to other diversity methods"""
        self.model.eval()
        dataloader = DataLoader(data, batch_size=32, shuffle=False)
        all_features = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                if hasattr(self.model, 'bert') or hasattr(self.model, 'roberta'):
                    outputs = self.model(input_ids=inputs, attention_mask=attention_mask,
                                       output_hidden_states=True)
                    features = outputs.hidden_states[-1][:, 0, :]
                else:
                    outputs = self.model(input_ids=inputs, attention_mask=attention_mask)
                    if hasattr(outputs, 'last_hidden_state'):
                        features = outputs.last_hidden_state.mean(dim=1)
                    else:
                        features = outputs.mean(dim=1)

                all_features.append(features.cpu().numpy())

        return np.concatenate(all_features)

    def select_samples(self,
                      unlabeled_data: Dataset,
                      n_samples: int,
                      labeled_data: Optional[Dataset] = None) -> List[int]:
        """
        Select representative samples based on centrality

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Not used in this strategy

        Returns:
            List of representative sample indices
        """
        features = self.get_features(unlabeled_data)
        logger.info(f"Extracted features with shape: {features.shape}")

        # Apply PCA if requested
        if self.use_pca and features.shape[1] > self.pca_components:
            pca = PCA(n_components=self.pca_components)
            features = pca.fit_transform(features)
            logger.info(f"Applied PCA, new shape: {features.shape}")

        # Calculate centrality scores
        centroid = np.mean(features, axis=0)
        distances_to_centroid = np.linalg.norm(features - centroid, axis=1)

        # Select samples closest to centroid (most representative)
        selected_indices = np.argsort(distances_to_centroid)[:n_samples].tolist()

        logger.info(f"Selected {len(selected_indices)} representative samples")
        logger.debug(f"Distance to centroid range: [{distances_to_centroid.min():.4f}, {distances_to_centroid.max():.4f}]")

        return selected_indices


class DiversityClusteringHybrid(ActiveLearner):
    """Hybrid approach combining multiple diversity strategies"""

    def __init__(self, model, device: str = 'cpu', clustering_ratio: float = 0.6):
        """
        Initialize hybrid diversity sampler

        Args:
            model: PyTorch model for feature extraction
            device: Device for computation
            clustering_ratio: Ratio of samples to select using clustering
        """
        super().__init__(model, device)
        self.clustering_ratio = clustering_ratio
        self.clustering_sampler = ClusteringSampling(model, device)
        self.coreset_sampler = CoreSetSampling(model, device)

    def select_samples(self,
                      unlabeled_data: Dataset,
                      n_samples: int,
                      labeled_data: Optional[Dataset] = None) -> List[int]:
        """
        Select diverse samples using hybrid approach

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Already labeled data

        Returns:
            List of diverse sample indices
        """
        n_clustering = int(n_samples * self.clustering_ratio)
        n_coreset = n_samples - n_clustering

        # Select samples using clustering
        clustering_indices = self.clustering_sampler.select_samples(
            unlabeled_data, n_clustering, labeled_data
        )

        # Create a subset excluding clustering selections for CoreSet
        remaining_indices = [i for i in range(len(unlabeled_data))
                           if i not in clustering_indices]

        if len(remaining_indices) > 0 and n_coreset > 0:
            # Create subset dataset for CoreSet selection
            from torch.utils.data import Subset
            remaining_dataset = Subset(unlabeled_data, remaining_indices)

            # Select additional samples using CoreSet
            coreset_sub_indices = self.coreset_sampler.select_samples(
                remaining_dataset, min(n_coreset, len(remaining_indices)), labeled_data
            )

            # Map back to original indices
            coreset_indices = [remaining_indices[i] for i in coreset_sub_indices]
        else:
            coreset_indices = []

        # Combine results
        selected_indices = clustering_indices + coreset_indices

        logger.info(f"Selected {len(selected_indices)} samples using hybrid diversity approach")
        logger.info(f"Clustering: {len(clustering_indices)}, CoreSet: {len(coreset_indices)}")

        return selected_indices