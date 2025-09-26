"""
Enhanced Active Learner with comprehensive functionality
"""

import os
import json
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import logging

from .base import ActiveLearner
from .query_strategies import (
    HybridQueryStrategy, AdaptiveQueryStrategy, MultiStrategyEnsemble
)

logger = logging.getLogger(__name__)


class CyberPuppyActiveLearner:
    """
    Enhanced Active Learner for CyberPuppy toxicity detection
    """

    def __init__(self,
                 model,
                 tokenizer,
                 device: str = 'cpu',
                 query_strategy: str = 'hybrid',
                 query_strategy_config: Optional[Dict[str, Any]] = None,
                 save_dir: str = './active_learning_results',
                 target_f1: float = 0.75,
                 max_budget: int = 1000):
        """
        Initialize CyberPuppy Active Learner

        Args:
            model: PyTorch model for toxicity detection
            tokenizer: Tokenizer for text processing
            device: Device for computation
            query_strategy: Type of query strategy to use
            query_strategy_config: Configuration for query strategy
            save_dir: Directory to save results and checkpoints
            target_f1: Target F1 score to stop active learning
            max_budget: Maximum annotation budget
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.target_f1 = target_f1
        self.max_budget = max_budget
        self.save_dir = save_dir

        # Initialize query strategy
        self.query_strategy = self._initialize_query_strategy(
            query_strategy, query_strategy_config or {}
        )

        # Tracking variables
        self.iteration = 0
        self.total_annotations = 0
        self.performance_history = []
        self.annotation_history = []
        self.selected_samples_history = []

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Initialize logging
        self._setup_logging()

        logger.info(f"Initialized CyberPuppy Active Learner with {query_strategy} strategy")

    def _initialize_query_strategy(self, strategy_name: str, config: Dict[str, Any]):
        """Initialize the specified query strategy"""
        if strategy_name == 'hybrid':
            uncertainty = config.get('uncertainty_strategy', 'entropy')
            diversity = config.get('diversity_strategy', 'clustering')
            ratio = config.get('uncertainty_ratio', 0.5)
            return HybridQueryStrategy(
                self.model, self.device, uncertainty, diversity, ratio, **config
            )
        elif strategy_name == 'adaptive':
            initial_ratio = config.get('initial_uncertainty_ratio', 0.5)
            adaptation_rate = config.get('adaptation_rate', 0.1)
            return AdaptiveQueryStrategy(
                self.model, self.device, initial_ratio, adaptation_rate, **config
            )
        elif strategy_name == 'ensemble':
            strategies = config.get('strategies', None)
            voting = config.get('voting_method', 'intersection')
            return MultiStrategyEnsemble(
                self.model, self.device, strategies, voting
            )
        else:
            raise ValueError(f"Unknown query strategy: {strategy_name}")

    def _setup_logging(self):
        """Setup logging for active learning process"""
        log_file = os.path.join(self.save_dir, 'active_learning.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def select_samples_for_annotation(self,
                                    unlabeled_data: Dataset,
                                    n_samples: int,
                                    labeled_data: Optional[Dataset] = None) -> List[int]:
        """
        Select samples for annotation using the configured strategy

        Args:
            unlabeled_data: Dataset of unlabeled samples
            n_samples: Number of samples to select
            labeled_data: Already labeled data

        Returns:
            List of selected sample indices
        """
        logger.info(f"Selecting {n_samples} samples for annotation (iteration {self.iteration})")

        # Check budget
        if self.total_annotations + n_samples > self.max_budget:
            n_samples = self.max_budget - self.total_annotations
            logger.warning(f"Budget constraint: reducing selection to {n_samples} samples")

        if n_samples <= 0:
            logger.warning("Budget exhausted, cannot select more samples")
            return []

        # Select samples using query strategy
        selected_indices = self.query_strategy.select_samples(
            unlabeled_data, n_samples, labeled_data
        )

        # Update tracking
        self.selected_samples_history.append({
            'iteration': self.iteration,
            'indices': selected_indices,
            'timestamp': datetime.now().isoformat()
        })

        logger.info(f"Selected {len(selected_indices)} samples for annotation")
        return selected_indices

    def evaluate_model(self,
                      test_data: Dataset,
                      test_labels: List[int],
                      task: str = 'toxicity') -> Dict[str, float]:
        """
        Evaluate model performance on test data

        Args:
            test_data: Test dataset
            test_labels: True labels for test data
            task: Task name for logging

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {len(test_data)} test samples")

        self.model.eval()
        predictions = []

        with torch.no_grad():
            dataloader = DataLoader(test_data, batch_size=32, shuffle=False)
            for batch in dataloader:
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                outputs = self.model(inputs, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
                predictions.extend(preds.cpu().numpy())

        # Calculate metrics
        f1_macro = f1_score(test_labels, predictions, average='macro')
        f1_micro = f1_score(test_labels, predictions, average='micro')

        # Calculate per-class F1 scores
        f1_per_class = f1_score(test_labels, predictions, average=None)

        metrics = {
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_none': f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
            'f1_toxic': f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
            'f1_severe': f1_per_class[2] if len(f1_per_class) > 2 else 0.0,
            'accuracy': np.mean(np.array(predictions) == np.array(test_labels))
        }

        # Log detailed results
        logger.info(f"Evaluation results - F1 Macro: {f1_macro:.4f}, F1 Micro: {f1_micro:.4f}")
        logger.info(f"Per-class F1: {f1_per_class}")

        # Save classification report
        report = classification_report(
            test_labels, predictions,
            target_names=['none', 'toxic', 'severe'],
            output_dict=True
        )

        # Update performance history
        performance_record = {
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat(),
            'total_annotations': self.total_annotations,
            'metrics': metrics,
            'classification_report': report
        }
        self.performance_history.append(performance_record)

        # Update adaptive strategy if applicable
        if hasattr(self.query_strategy, 'update_performance'):
            self.query_strategy.update_performance(f1_macro)

        return metrics

    def check_stopping_criteria(self, current_f1: float) -> bool:
        """
        Check if active learning should stop

        Args:
            current_f1: Current F1 score

        Returns:
            True if stopping criteria met
        """
        # Check target F1 achieved
        if current_f1 >= self.target_f1:
            logger.info(f"Target F1 score {self.target_f1} achieved: {current_f1:.4f}")
            return True

        # Check budget exhausted
        if self.total_annotations >= self.max_budget:
            logger.info(f"Budget exhausted: {self.total_annotations}/{self.max_budget}")
            return True

        # Check performance plateau (last 3 iterations)
        if len(self.performance_history) >= 3:
            recent_f1s = [record['metrics']['f1_macro']
                         for record in self.performance_history[-3:]]
            improvement = max(recent_f1s) - min(recent_f1s)
            if improvement < 0.005:  # Less than 0.5% improvement
                logger.info(f"Performance plateau detected: {improvement:.4f}")
                return True

        return False

    def save_checkpoint(self, iteration: int):
        """Save active learning checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'total_annotations': self.total_annotations,
            'performance_history': self.performance_history,
            'annotation_history': self.annotation_history,
            'selected_samples_history': self.selected_samples_history,
            'query_strategy_config': {
                'name': type(self.query_strategy).__name__,
                'state': getattr(self.query_strategy, '__dict__', {})
            }
        }

        checkpoint_path = os.path.join(
            self.save_dir, f'checkpoint_iter_{iteration}.pkl'
        )
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        # Also save as JSON for readability
        json_path = os.path.join(
            self.save_dir, f'checkpoint_iter_{iteration}.json'
        )
        json_checkpoint = {k: v for k, v in checkpoint.items()
                          if k != 'query_strategy_config'}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_checkpoint, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved checkpoint for iteration {iteration}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load active learning checkpoint"""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.iteration = checkpoint['iteration']
        self.total_annotations = checkpoint['total_annotations']
        self.performance_history = checkpoint['performance_history']
        self.annotation_history = checkpoint['annotation_history']
        self.selected_samples_history = checkpoint['selected_samples_history']

        logger.info(f"Loaded checkpoint from iteration {self.iteration}")

    def get_learning_curve_data(self) -> Dict[str, List[float]]:
        """Get data for plotting learning curves"""
        annotations = [record['total_annotations']
                      for record in self.performance_history]
        f1_scores = [record['metrics']['f1_macro']
                    for record in self.performance_history]
        f1_toxic = [record['metrics']['f1_toxic']
                   for record in self.performance_history]
        f1_severe = [record['metrics']['f1_severe']
                    for record in self.performance_history]

        return {
            'annotations': annotations,
            'f1_macro': f1_scores,
            'f1_toxic': f1_toxic,
            'f1_severe': f1_severe
        }

    def get_annotation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the annotation process"""
        if not self.annotation_history:
            return {'total_annotations': 0, 'annotations_per_iteration': []}

        annotations_per_iter = [len(record['annotations'])
                               for record in self.annotation_history]

        return {
            'total_annotations': self.total_annotations,
            'iterations': len(self.annotation_history),
            'annotations_per_iteration': annotations_per_iter,
            'avg_annotations_per_iteration': np.mean(annotations_per_iter),
            'max_annotations_per_iteration': max(annotations_per_iter),
            'min_annotations_per_iteration': min(annotations_per_iter)
        }

    def add_annotations(self, annotations: List[Dict[str, Any]]):
        """
        Add new annotations to the history

        Args:
            annotations: List of annotation dictionaries
        """
        annotation_record = {
            'iteration': self.iteration,
            'timestamp': datetime.now().isoformat(),
            'annotations': annotations
        }
        self.annotation_history.append(annotation_record)
        self.total_annotations += len(annotations)

        logger.info(f"Added {len(annotations)} annotations. Total: {self.total_annotations}")

    def next_iteration(self):
        """Move to the next iteration"""
        self.iteration += 1
        logger.info(f"Starting iteration {self.iteration}")

    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary"""
        current_f1 = (self.performance_history[-1]['metrics']['f1_macro']
                     if self.performance_history else 0.0)

        return {
            'iteration': self.iteration,
            'total_annotations': self.total_annotations,
            'budget_remaining': self.max_budget - self.total_annotations,
            'budget_used_pct': (self.total_annotations / self.max_budget) * 100,
            'current_f1_macro': current_f1,
            'target_f1': self.target_f1,
            'target_achieved': current_f1 >= self.target_f1,
            'query_strategy': type(self.query_strategy).__name__,
            'performance_trend': self._get_performance_trend()
        }

    def _get_performance_trend(self) -> str:
        """Get performance trend description"""
        if len(self.performance_history) < 2:
            return 'insufficient_data'

        recent_f1s = [record['metrics']['f1_macro']
                     for record in self.performance_history[-3:]]

        if len(recent_f1s) >= 2:
            trend = recent_f1s[-1] - recent_f1s[0]
            if trend > 0.01:
                return 'improving'
            elif trend < -0.01:
                return 'declining'
            else:
                return 'stable'

        return 'unknown'