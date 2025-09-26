"""
Active Learning Loop Implementation
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import logging

from .active_learner import CyberPuppyActiveLearner
from .annotator import InteractiveAnnotator, BatchAnnotator

logger = logging.getLogger(__name__)


class ActiveLearningLoop:
    """
    Main active learning loop orchestrating the entire process
    """

    def __init__(self,
                 active_learner: CyberPuppyActiveLearner,
                 annotator: InteractiveAnnotator,
                 train_function: Callable,
                 initial_labeled_data: Dataset,
                 unlabeled_pool: Dataset,
                 test_data: Dataset,
                 test_labels: List[int],
                 samples_per_iteration: int = 20,
                 max_iterations: int = 50,
                 evaluation_frequency: int = 1):
        """
        Initialize active learning loop

        Args:
            active_learner: CyberPuppyActiveLearner instance
            annotator: InteractiveAnnotator instance
            train_function: Function to train the model with new data
            initial_labeled_data: Initial labeled training data
            unlabeled_pool: Pool of unlabeled data
            test_data: Test dataset for evaluation
            test_labels: Test labels
            samples_per_iteration: Number of samples to select per iteration
            max_iterations: Maximum number of iterations
            evaluation_frequency: Evaluate model every N iterations
        """
        self.active_learner = active_learner
        self.annotator = annotator
        self.train_function = train_function
        self.initial_labeled_data = initial_labeled_data
        self.unlabeled_pool = unlabeled_pool
        self.test_data = test_data
        self.test_labels = test_labels
        self.samples_per_iteration = samples_per_iteration
        self.max_iterations = max_iterations
        self.evaluation_frequency = evaluation_frequency

        # State tracking
        self.current_labeled_data = initial_labeled_data
        self.remaining_unlabeled_indices = list(range(len(unlabeled_pool)))
        self.selected_indices_history = []
        self.loop_start_time = None

        logger.info(f"Initialized ActiveLearningLoop with {len(initial_labeled_data)} initial samples")

    def run(self, interactive: bool = True, auto_train: bool = True) -> Dict[str, Any]:
        """
        Run the complete active learning loop

        Args:
            interactive: Whether to use interactive annotation
            auto_train: Whether to automatically retrain after each iteration

        Returns:
            Dictionary with loop results and statistics
        """
        logger.info("Starting Active Learning Loop")
        self.loop_start_time = time.time()

        try:
            while not self._should_stop():
                iteration_start_time = time.time()

                logger.info(f"\n{'='*60}")
                logger.info(f"Active Learning Iteration {self.active_learner.iteration + 1}")
                logger.info(f"{'='*60}")

                # Step 1: Select samples for annotation
                selected_indices = self._select_samples()
                if not selected_indices:
                    logger.info("No more samples to select. Stopping loop.")
                    break

                # Step 2: Get samples for annotation
                samples_to_annotate = self._prepare_samples_for_annotation(selected_indices)

                # Step 3: Annotate samples
                if interactive:
                    annotations = self._interactive_annotation(samples_to_annotate, selected_indices)
                else:
                    annotations = self._mock_annotation(samples_to_annotate, selected_indices)

                if not annotations:
                    logger.warning("No annotations collected. Skipping iteration.")
                    continue

                # Step 4: Update labeled dataset
                self._update_labeled_data(annotations, selected_indices)

                # Step 5: Train model with new data
                if auto_train:
                    self._train_model()

                # Step 6: Evaluate model
                if self.active_learner.iteration % self.evaluation_frequency == 0:
                    metrics = self._evaluate_model()
                    logger.info(f"Evaluation metrics: {metrics}")

                # Step 7: Check stopping criteria
                if self._check_stopping_criteria():
                    logger.info("Stopping criteria met. Ending active learning loop.")
                    break

                # Step 8: Save checkpoint
                self._save_checkpoint()

                # Move to next iteration
                self.active_learner.next_iteration()

                iteration_time = time.time() - iteration_start_time
                logger.info(f"Iteration {self.active_learner.iteration} completed in {iteration_time:.2f}s")

        except KeyboardInterrupt:
            logger.info("Active learning loop interrupted by user")
        except Exception as e:
            logger.error(f"Error in active learning loop: {e}")
            raise

        # Final evaluation and summary
        final_results = self._generate_final_results()
        logger.info("Active Learning Loop completed")

        return final_results

    def _select_samples(self) -> List[int]:
        """Select samples for annotation"""
        if not self.remaining_unlabeled_indices:
            return []

        # Create subset of unlabeled data from remaining indices
        from torch.utils.data import Subset
        remaining_unlabeled_data = Subset(self.unlabeled_pool, self.remaining_unlabeled_indices)

        # Select samples using active learner
        selected_sub_indices = self.active_learner.select_samples_for_annotation(
            remaining_unlabeled_data,
            min(self.samples_per_iteration, len(self.remaining_unlabeled_indices)),
            self.current_labeled_data
        )

        # Map back to original indices
        selected_indices = [self.remaining_unlabeled_indices[i] for i in selected_sub_indices]

        # Remove selected indices from remaining pool
        for idx in selected_indices:
            self.remaining_unlabeled_indices.remove(idx)

        self.selected_indices_history.append(selected_indices)

        logger.info(f"Selected {len(selected_indices)} samples for annotation")
        logger.info(f"Remaining unlabeled samples: {len(self.remaining_unlabeled_indices)}")

        return selected_indices

    def _prepare_samples_for_annotation(self, selected_indices: List[int]) -> List[Dict[str, Any]]:
        """Prepare samples for annotation with predictions"""
        samples = []

        for idx in selected_indices:
            # Get the original sample
            sample = self.unlabeled_pool[idx]

            # Add model predictions if available
            try:
                with torch.no_grad():
                    # Get model prediction for this sample
                    predictions = self._get_model_predictions([sample])
                    sample_dict = {
                        'text': sample.get('text', sample.get('input_ids', '')),
                        'metadata': sample.get('metadata', {}),
                        'predictions': predictions[0] if predictions else {}
                    }
            except Exception as e:
                logger.warning(f"Could not get predictions for sample {idx}: {e}")
                sample_dict = {
                    'text': sample.get('text', sample.get('input_ids', '')),
                    'metadata': sample.get('metadata', {}),
                    'predictions': {}
                }

            samples.append(sample_dict)

        return samples

    def _get_model_predictions(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get model predictions for samples"""
        predictions = []

        self.active_learner.model.eval()
        with torch.no_grad():
            for sample in samples:
                try:
                    # This would need to be adapted based on your specific model interface
                    # For now, returning mock predictions
                    pred = {
                        'toxicity': {'label': 'none', 'confidence': 0.5},
                        'bullying': {'label': 'none', 'confidence': 0.5},
                        'emotion': {'label': 'neutral', 'confidence': 0.5}
                    }
                    predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Prediction error: {e}")
                    predictions.append({})

        return predictions

    def _interactive_annotation(self,
                               samples: List[Dict[str, Any]],
                               selected_indices: List[int]) -> List[Dict[str, Any]]:
        """Perform interactive annotation"""
        logger.info("Starting interactive annotation session")

        annotations = self.annotator.annotate_samples(
            samples, selected_indices, show_predictions=True
        )

        # Validate annotations
        validation_results = self.annotator.validate_annotations(annotations)
        if validation_results['issues']:
            logger.warning(f"Annotation validation issues: {len(validation_results['issues'])}")

        return annotations

    def _mock_annotation(self,
                        samples: List[Dict[str, Any]],
                        selected_indices: List[int]) -> List[Dict[str, Any]]:
        """Generate mock annotations for testing"""
        logger.info("Generating mock annotations")

        annotations = []
        for i, (sample, idx) in enumerate(zip(samples, selected_indices)):
            # Generate random but realistic annotations
            annotation = {
                'original_index': idx,
                'text': sample.get('text', ''),
                'timestamp': datetime.now().isoformat(),
                'annotator': 'mock',
                'toxicity': np.random.choice(['none', 'toxic', 'severe'], p=[0.7, 0.25, 0.05]),
                'bullying': np.random.choice(['none', 'harassment', 'threat'], p=[0.8, 0.15, 0.05]),
                'role': np.random.choice(['none', 'perpetrator', 'victim', 'bystander'], p=[0.7, 0.1, 0.1, 0.1]),
                'emotion': np.random.choice(['negative', 'neutral', 'positive'], p=[0.3, 0.5, 0.2]),
                'emotion_strength': np.random.randint(0, 5),
                'confidence': np.random.uniform(0.6, 1.0),
                'comments': ''
            }
            annotations.append(annotation)

        return annotations

    def _update_labeled_data(self, annotations: List[Dict[str, Any]], selected_indices: List[int]):
        """Update the labeled dataset with new annotations"""
        # Record annotations
        self.active_learner.add_annotations(annotations)

        # Convert annotations to dataset format and combine with existing labeled data
        # This is a simplified version - you'd need to implement proper dataset merging
        logger.info(f"Updated labeled dataset with {len(annotations)} new samples")

    def _train_model(self):
        """Train the model with updated labeled data"""
        logger.info("Training model with updated labeled data")

        try:
            # Call the provided training function
            training_results = self.train_function(
                self.current_labeled_data,
                model=self.active_learner.model,
                device=self.active_learner.device
            )
            logger.info(f"Training completed: {training_results}")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model on test data"""
        return self.active_learner.evaluate_model(
            self.test_data, self.test_labels, task='toxicity'
        )

    def _check_stopping_criteria(self) -> bool:
        """Check if active learning should stop"""
        if not self.active_learner.performance_history:
            return False

        current_f1 = self.active_learner.performance_history[-1]['metrics']['f1_macro']
        return self.active_learner.check_stopping_criteria(current_f1)

    def _save_checkpoint(self):
        """Save checkpoint of current state"""
        self.active_learner.save_checkpoint(self.active_learner.iteration)

        # Save loop-specific state
        loop_state = {
            'iteration': self.active_learner.iteration,
            'remaining_unlabeled_indices': self.remaining_unlabeled_indices,
            'selected_indices_history': self.selected_indices_history,
            'loop_start_time': self.loop_start_time
        }

        checkpoint_path = os.path.join(
            self.active_learner.save_dir,
            f'loop_state_iter_{self.active_learner.iteration}.json'
        )

        with open(checkpoint_path, 'w') as f:
            json.dump(loop_state, f, indent=2)

    def _should_stop(self) -> bool:
        """Check if loop should stop"""
        # Maximum iterations reached
        if self.active_learner.iteration >= self.max_iterations:
            logger.info(f"Maximum iterations ({self.max_iterations}) reached")
            return True

        # No more unlabeled data
        if not self.remaining_unlabeled_indices:
            logger.info("No more unlabeled data available")
            return True

        # Budget exhausted
        if self.active_learner.total_annotations >= self.active_learner.max_budget:
            logger.info("Annotation budget exhausted")
            return True

        return False

    def _generate_final_results(self) -> Dict[str, Any]:
        """Generate final results summary"""
        total_time = time.time() - self.loop_start_time if self.loop_start_time else 0

        results = {
            'total_iterations': self.active_learner.iteration,
            'total_annotations': self.active_learner.total_annotations,
            'total_time_seconds': total_time,
            'annotations_per_hour': (self.active_learner.total_annotations / (total_time / 3600))
                                   if total_time > 0 else 0,
            'final_performance': (self.active_learner.performance_history[-1]['metrics']
                                if self.active_learner.performance_history else {}),
            'performance_history': self.active_learner.get_learning_curve_data(),
            'annotation_statistics': self.active_learner.get_annotation_statistics(),
            'status_summary': self.active_learner.get_status_summary(),
            'remaining_unlabeled_samples': len(self.remaining_unlabeled_indices)
        }

        # Save final results
        results_path = os.path.join(
            self.active_learner.save_dir,
            f'final_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Final results saved to {results_path}")

        return results

    def resume_from_checkpoint(self, checkpoint_dir: str, iteration: int):
        """Resume active learning from a checkpoint"""
        # Load active learner checkpoint
        al_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}.pkl')
        self.active_learner.load_checkpoint(al_checkpoint_path)

        # Load loop state
        loop_checkpoint_path = os.path.join(checkpoint_dir, f'loop_state_iter_{iteration}.json')
        with open(loop_checkpoint_path, 'r') as f:
            loop_state = json.load(f)

        self.remaining_unlabeled_indices = loop_state['remaining_unlabeled_indices']
        self.selected_indices_history = loop_state['selected_indices_history']
        self.loop_start_time = loop_state['loop_start_time']

        logger.info(f"Resumed active learning from iteration {iteration}")

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        return {
            'current_iteration': self.active_learner.iteration,
            'total_annotations': self.active_learner.total_annotations,
            'remaining_budget': self.active_learner.max_budget - self.active_learner.total_annotations,
            'remaining_unlabeled': len(self.remaining_unlabeled_indices),
            'current_performance': (self.active_learner.performance_history[-1]['metrics']
                                  if self.active_learner.performance_history else {}),
            'elapsed_time': time.time() - self.loop_start_time if self.loop_start_time else 0
        }


class BatchActiveLearningLoop(ActiveLearningLoop):
    """Active learning loop optimized for batch processing"""

    def __init__(self, *args, batch_size: int = 50, **kwargs):
        """
        Initialize batch active learning loop

        Args:
            batch_size: Size of annotation batches
            *args, **kwargs: Arguments for parent class
        """
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.batch_annotator = BatchAnnotator(self.annotator)

    def _interactive_annotation(self,
                               samples: List[Dict[str, Any]],
                               selected_indices: List[int]) -> List[Dict[str, Any]]:
        """Perform batch interactive annotation"""
        logger.info(f"Starting batch annotation session with {len(samples)} samples")

        # Combine samples with indices for batch processing
        sample_data = []
        for sample, idx in zip(samples, selected_indices):
            sample['original_index'] = idx
            sample_data.append(sample)

        annotations = self.batch_annotator.process_batch(
            sample_data, self.batch_size, save_frequency=3
        )

        return annotations