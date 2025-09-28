"""
Unit tests for active learning framework
"""

import os
# Import active learning modules
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberpuppy.active_learning.active_learner import CyberPuppyActiveLearner
from cyberpuppy.active_learning.annotator import InteractiveAnnotator
from cyberpuppy.active_learning.diversity_enhanced import (
    ClusteringSampling, CoreSetSampling, RepresentativeSampling)
from cyberpuppy.active_learning.loop import ActiveLearningLoop
from cyberpuppy.active_learning.query_strategies import (AdaptiveQueryStrategy,
                                                         HybridQueryStrategy,
                                                         MultiStrategyEnsemble)
from cyberpuppy.active_learning.uncertainty_enhanced import (
    BALD, BayesianUncertaintySampling, EntropySampling,
    LeastConfidenceSampling, MarginSampling)


class MockDataset:
    """Mock dataset for testing"""

    def __init__(self, size=100, features_dim=768):
        self.size = size
        self.features_dim = features_dim
        self.data = []
        for i in range(size):
            self.data.append(
                {
                    "input_ids": torch.randint(0, 1000, (512,)),
                    "attention_mask": torch.ones(512),
                    "text": f"Sample text {i}",
                    "labels": torch.randint(0, 3, (1,)).item(),
                }
            )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class MockModel(torch.nn.Module):
    """Mock model for testing"""

    def __init__(self, num_classes=3, features_dim=768):
        super().__init__()
        self.classifier = torch.nn.Linear(features_dim, num_classes)
        self.features_dim = features_dim

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        batch_size = input_ids.size(0)
        # Simulate hidden states
        hidden_states = torch.randn(batch_size, 512, self.features_dim)

        # Simulate logits
        logits = self.classifier(hidden_states.mean(dim=1))

        if output_hidden_states:
            return type(
                "MockOutput",
                (),
                {"logits": logits, "hidden_states": [hidden_states] * 12},  # Simulate 12 layers
            )()
        else:
            return type("MockOutput", (), {"logits": logits})()


class TestUncertaintySampling(unittest.TestCase):
    """Test uncertainty sampling strategies"""

    def setUp(self):
        self.model = MockModel()
        self.device = "cpu"
        self.dataset = MockDataset(50)

    def test_entropy_sampling(self):
        """Test entropy sampling strategy"""
        sampler = EntropySampling(self.model, self.device)
        selected = sampler.select_samples(self.dataset, n_samples=10)

        self.assertEqual(len(selected), 10)
        self.assertTrue(all(isinstance(idx, int) for idx in selected))
        self.assertTrue(all(0 <= idx < len(self.dataset) for idx in selected))

    def test_least_confidence_sampling(self):
        """Test least confidence sampling strategy"""
        sampler = LeastConfidenceSampling(self.model, self.device)
        selected = sampler.select_samples(self.dataset, n_samples=5)

        self.assertEqual(len(selected), 5)
        self.assertTrue(all(isinstance(idx, int) for idx in selected))

    def test_margin_sampling(self):
        """Test margin sampling strategy"""
        sampler = MarginSampling(self.model, self.device)
        selected = sampler.select_samples(self.dataset, n_samples=8)

        self.assertEqual(len(selected), 8)
        self.assertTrue(all(isinstance(idx, int) for idx in selected))

    def test_bayesian_uncertainty_sampling(self):
        """Test Bayesian uncertainty sampling with MC Dropout"""
        # Add dropout to mock model
        self.model.dropout = torch.nn.Dropout(0.1)

        sampler = BayesianUncertaintySampling(self.model, self.device, n_dropout_samples=3)
        selected = sampler.select_samples(self.dataset, n_samples=6)

        self.assertEqual(len(selected), 6)
        self.assertTrue(all(isinstance(idx, int) for idx in selected))

    def test_bald_sampling(self):
        """Test BALD sampling strategy"""
        self.model.dropout = torch.nn.Dropout(0.1)

        sampler = BALD(self.model, self.device, n_dropout_samples=3)
        selected = sampler.select_samples(self.dataset, n_samples=4)

        self.assertEqual(len(selected), 4)
        self.assertTrue(all(isinstance(idx, int) for idx in selected))


class TestDiversitySampling(unittest.TestCase):
    """Test diversity sampling strategies"""

    def setUp(self):
        self.model = MockModel()
        self.device = "cpu"
        self.dataset = MockDataset(30)

    def test_clustering_sampling(self):
        """Test clustering-based diversity sampling"""
        sampler = ClusteringSampling(self.model, self.device, n_clusters=5)
        selected = sampler.select_samples(self.dataset, n_samples=10)

        self.assertEqual(len(selected), 10)
        self.assertTrue(all(isinstance(idx, int) for idx in selected))
        self.assertEqual(len(set(selected)), len(selected))  # No duplicates

    def test_coreset_sampling(self):
        """Test CoreSet diversity sampling"""
        sampler = CoreSetSampling(self.model, self.device)
        selected = sampler.select_samples(self.dataset, n_samples=8)

        self.assertEqual(len(selected), 8)
        self.assertTrue(all(isinstance(idx, int) for idx in selected))

    def test_representative_sampling(self):
        """Test representative diversity sampling"""
        sampler = RepresentativeSampling(self.model, self.device, use_pca=True)
        selected = sampler.select_samples(self.dataset, n_samples=6)

        self.assertEqual(len(selected), 6)
        self.assertTrue(all(isinstance(idx, int) for idx in selected))


class TestQueryStrategies(unittest.TestCase):
    """Test hybrid and advanced query strategies"""

    def setUp(self):
        self.model = MockModel()
        self.device = "cpu"
        self.dataset = MockDataset(40)

    def test_hybrid_query_strategy(self):
        """Test hybrid uncertainty + diversity strategy"""
        strategy = HybridQueryStrategy(
            self.model,
            self.device,
            uncertainty_strategy="entropy",
            diversity_strategy="clustering",
            uncertainty_ratio=0.6,
        )

        selected = strategy.select_samples(self.dataset, n_samples=12)

        self.assertEqual(len(selected), 12)
        self.assertTrue(all(isinstance(idx, int) for idx in selected))

    def test_adaptive_query_strategy(self):
        """Test adaptive query strategy"""
        strategy = AdaptiveQueryStrategy(
            self.model, self.device, initial_uncertainty_ratio=0.5, adaptation_rate=0.1
        )

        # Test adaptation mechanism
        strategy.update_performance(0.6)
        strategy.update_performance(0.65)  # Improvement
        strategy.update_performance(0.66)  # Small improvement

        selected = strategy.select_samples(self.dataset, n_samples=10)
        self.assertEqual(len(selected), 10)

    def test_multi_strategy_ensemble(self):
        """Test multi-strategy ensemble"""
        strategies = [
            {"uncertainty": "entropy", "diversity": "clustering", "ratio": 0.5},
            {"uncertainty": "margin", "diversity": "coreset", "ratio": 0.6},
        ]

        ensemble = MultiStrategyEnsemble(
            self.model, self.device, strategies=strategies, voting_method="intersection"
        )

        selected = ensemble.select_samples(self.dataset, n_samples=8)
        self.assertEqual(len(selected), 8)


class TestCyberPuppyActiveLearner(unittest.TestCase):
    """Test the main active learner class"""

    def setUp(self):
        self.model = MockModel()
        self.tokenizer = Mock()
        self.device = "cpu"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test active learner initialization"""
        learner = CyberPuppyActiveLearner(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            query_strategy="hybrid",
            save_dir=self.temp_dir,
            target_f1=0.8,
            max_budget=500,
        )

        self.assertEqual(learner.target_f1, 0.8)
        self.assertEqual(learner.max_budget, 500)
        self.assertEqual(learner.iteration, 0)

    def test_sample_selection(self):
        """Test sample selection for annotation"""
        learner = CyberPuppyActiveLearner(
            model=self.model, tokenizer=self.tokenizer, device=self.device, save_dir=self.temp_dir
        )

        dataset = MockDataset(20)
        selected = learner.select_samples_for_annotation(dataset, n_samples=5)

        self.assertEqual(len(selected), 5)
        self.assertTrue(all(isinstance(idx, int) for idx in selected))

    def test_evaluation(self):
        """Test model evaluation"""
        learner = CyberPuppyActiveLearner(
            model=self.model, tokenizer=self.tokenizer, device=self.device, save_dir=self.temp_dir
        )

        dataset = MockDataset(10)
        test_labels = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]

        metrics = learner.evaluate_model(dataset, test_labels)

        self.assertIn("f1_macro", metrics)
        self.assertIn("f1_micro", metrics)
        self.assertIn("accuracy", metrics)
        self.assertTrue(0 <= metrics["f1_macro"] <= 1)

    def test_stopping_criteria(self):
        """Test stopping criteria checking"""
        learner = CyberPuppyActiveLearner(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            save_dir=self.temp_dir,
            target_f1=0.8,
            max_budget=100,
        )

        # Test target achieved
        self.assertTrue(learner.check_stopping_criteria(0.85))

        # Test budget exhausted
        learner.total_annotations = 150
        self.assertTrue(learner.check_stopping_criteria(0.7))

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading"""
        learner = CyberPuppyActiveLearner(
            model=self.model, tokenizer=self.tokenizer, device=self.device, save_dir=self.temp_dir
        )

        # Add some history
        learner.iteration = 5
        learner.total_annotations = 50
        learner.performance_history.append(
            {"iteration": 5, "total_annotations": 50, "metrics": {"f1_macro": 0.75}}
        )

        # Save checkpoint
        learner.save_checkpoint(5)

        # Check files exist
        checkpoint_file = os.path.join(self.temp_dir, "checkpoint_iter_5.pkl")
        json_file = os.path.join(self.temp_dir, "checkpoint_iter_5.json")

        self.assertTrue(os.path.exists(checkpoint_file))
        self.assertTrue(os.path.exists(json_file))

        # Test loading
        new_learner = CyberPuppyActiveLearner(
            model=self.model, tokenizer=self.tokenizer, device=self.device, save_dir=self.temp_dir
        )

        new_learner.load_checkpoint(checkpoint_file)

        self.assertEqual(new_learner.iteration, 5)
        self.assertEqual(new_learner.total_annotations, 50)


class TestInteractiveAnnotator(unittest.TestCase):
    """Test interactive annotation functionality"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test annotator initialization"""
        annotator = InteractiveAnnotator(save_dir=self.temp_dir)
        self.assertEqual(annotator.save_dir, self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))

    def test_annotation_validation(self):
        """Test annotation validation"""
        annotator = InteractiveAnnotator(save_dir=self.temp_dir)

        # Valid annotations
        valid_annotations = [
            {
                "original_index": 0,
                "text": "test text",
                "toxicity": "none",
                "bullying": "none",
                "role": "none",
                "emotion": "neutral",
                "emotion_strength": 2,
                "confidence": 0.8,
            }
        ]

        validation = annotator.validate_annotations(valid_annotations)
        self.assertEqual(validation["valid_samples"], 1)
        self.assertEqual(len(validation["issues"]), 0)

        # Invalid annotations
        invalid_annotations = [
            {
                "original_index": 0,
                "text": "test text",
                "toxicity": "none",
                "bullying": "none",
                "role": "victim",  # Invalid: role but no bullying
                "emotion": "neutral",
                "emotion_strength": 5,  # Invalid: out of range
                "confidence": 1.5,  # Invalid: out of range
            }
        ]

        validation = annotator.validate_annotations(invalid_annotations)
        self.assertEqual(validation["valid_samples"], 0)
        self.assertGreater(len(validation["issues"]), 0)

    def test_annotation_statistics(self):
        """Test annotation statistics calculation"""
        annotator = InteractiveAnnotator(save_dir=self.temp_dir)

        annotations = [
            {
                "toxicity": "none",
                "bullying": "none",
                "emotion": "positive",
                "confidence": 0.8,
                "emotion_strength": 3,
            },
            {
                "toxicity": "toxic",
                "bullying": "harassment",
                "emotion": "negative",
                "confidence": 0.9,
                "emotion_strength": 4,
            },
        ]

        stats = annotator.get_annotation_statistics(annotations)

        self.assertEqual(stats["total_annotations"], 2)
        self.assertIn("toxicity_distribution", stats)
        self.assertIn("bullying_distribution", stats)
        self.assertEqual(stats["toxicity_distribution"]["none"], 1)
        self.assertEqual(stats["toxicity_distribution"]["toxic"], 1)


class TestActiveLearningLoop(unittest.TestCase):
    """Test the main active learning loop"""

    def setUp(self):
        self.model = MockModel()
        self.tokenizer = Mock()
        self.device = "cpu"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_loop_initialization(self):
        """Test loop initialization"""
        active_learner = CyberPuppyActiveLearner(
            model=self.model, tokenizer=self.tokenizer, device=self.device, save_dir=self.temp_dir
        )

        annotator = InteractiveAnnotator(save_dir=self.temp_dir)

        def mock_train_function(labeled_data, model, device):
            return {"loss": 0.5}

        initial_data = MockDataset(10)
        unlabeled_data = MockDataset(30)
        test_data = MockDataset(20)
        test_labels = [0, 1, 2] * 6 + [0, 1]

        loop = ActiveLearningLoop(
            active_learner=active_learner,
            annotator=annotator,
            train_function=mock_train_function,
            initial_labeled_data=initial_data,
            unlabeled_pool=unlabeled_data,
            test_data=test_data,
            test_labels=test_labels,
            samples_per_iteration=5,
            max_iterations=3,
        )

        self.assertEqual(loop.samples_per_iteration, 5)
        self.assertEqual(loop.max_iterations, 3)
        self.assertEqual(len(loop.remaining_unlabeled_indices), 30)

    def test_stopping_conditions(self):
        """Test loop stopping conditions"""
        active_learner = CyberPuppyActiveLearner(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            save_dir=self.temp_dir,
            max_budget=50,
        )

        annotator = InteractiveAnnotator(save_dir=self.temp_dir)

        def mock_train_function(labeled_data, model, device):
            return {"loss": 0.5}

        loop = ActiveLearningLoop(
            active_learner=active_learner,
            annotator=annotator,
            train_function=mock_train_function,
            initial_labeled_data=MockDataset(5),
            unlabeled_pool=MockDataset(10),
            test_data=MockDataset(10),
            test_labels=[0] * 10,
            max_iterations=2,
        )

        # Test max iterations
        loop.active_learner.iteration = 5
        self.assertTrue(loop._should_stop())

        # Test budget exhausted
        loop.active_learner.iteration = 0
        loop.active_learner.total_annotations = 60
        self.assertTrue(loop._should_stop())

        # Test no unlabeled data
        loop.active_learner.total_annotations = 0
        loop.remaining_unlabeled_indices = []
        self.assertTrue(loop._should_stop())


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete active learning workflow"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_workflow_mock(self):
        """Test complete active learning workflow with mocked components"""
        # Setup components
        model = MockModel()
        tokenizer = Mock()

        active_learner = CyberPuppyActiveLearner(
            model=model,
            tokenizer=tokenizer,
            device="cpu",
            query_strategy="hybrid",
            save_dir=self.temp_dir,
            target_f1=0.85,
            max_budget=20,
        )

        annotator = InteractiveAnnotator(save_dir=self.temp_dir)

        def mock_train_function(labeled_data, model, device):
            return {"loss": np.random.uniform(0.3, 0.7)}

        # Create datasets
        initial_data = MockDataset(5)
        unlabeled_data = MockDataset(15)
        test_data = MockDataset(10)
        test_labels = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]

        # Create loop
        loop = ActiveLearningLoop(
            active_learner=active_learner,
            annotator=annotator,
            train_function=mock_train_function,
            initial_labeled_data=initial_data,
            unlabeled_pool=unlabeled_data,
            test_data=test_data,
            test_labels=test_labels,
            samples_per_iteration=3,
            max_iterations=2,
        )

        # Run with mock annotation (non-interactive)
        results = loop.run(interactive=False, auto_train=True)

        # Verify results
        self.assertIn("total_iterations", results)
        self.assertIn("total_annotations", results)
        self.assertIn("final_performance", results)
        self.assertGreater(results["total_annotations"], 0)


if __name__ == "__main__":
    # Create test suite
    test_classes = [
        TestUncertaintySampling,
        TestDiversitySampling,
        TestQueryStrategies,
        TestCyberPuppyActiveLearner,
        TestInteractiveAnnotator,
        TestActiveLearningLoop,
        TestIntegration,
    ]

    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.splitlines()[-1]}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.splitlines()[-1]}")
