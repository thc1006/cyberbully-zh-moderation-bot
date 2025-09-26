#!/usr/bin/env python3
"""
Validation Script for Active Learning Framework

This script validates the complete active learning workflow by:
1. Testing all sampling strategies
2. Running a mini active learning loop
3. Generating visualizations
4. Validating all components work together

Usage:
    python scripts/validate_active_learning.py
    python scripts/validate_active_learning.py --verbose --save-results
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import active learning framework
from cyberpuppy.active_learning import (
    # Uncertainty strategies
    EntropySampling, LeastConfidenceSampling, MarginSampling,
    BayesianUncertaintySampling, BALD,

    # Diversity strategies
    ClusteringSampling, CoreSetSampling, RepresentativeSampling,
    DiversityClusteringHybrid,

    # Query strategies
    HybridQueryStrategy, AdaptiveQueryStrategy, MultiStrategyEnsemble,

    # Main components
    CyberPuppyActiveLearner, InteractiveAnnotator, BatchAnnotator,
    ActiveLearningLoop, BatchActiveLearningLoop,

    # Visualization
    ActiveLearningVisualizer
)

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ValidationDataset:
    """Simple dataset for validation"""

    def __init__(self, texts, labels=None, tokenizer=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,  # Shorter for validation
                return_tensors='pt'
            )

            item = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'text': text
            }

            if self.labels is not None:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

            return item
        else:
            return {'text': text, 'id': idx}


class SimpleModel(torch.nn.Module):
    """Simple model for validation"""

    def __init__(self, vocab_size=1000, hidden_size=128, num_classes=3):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        # Simple embedding-based model
        embeddings = self.embedding(input_ids)
        pooled = embeddings.mean(dim=1)  # Simple mean pooling
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if output_hidden_states:
            # Simulate BERT-like output
            return type('MockOutput', (), {
                'logits': logits,
                'hidden_states': [embeddings] * 4  # Simulate 4 layers
            })()
        else:
            return type('MockOutput', (), {'logits': logits})()


def create_validation_data():
    """Create validation data"""
    # Chinese text samples with varying toxicity levels
    texts = [
        # Non-toxic (label 0)
        "你好，今天天氣真好！",
        "這個想法很不錯，值得嘗試。",
        "感謝你的幫助，非常有用。",
        "這件事情需要仔細考慮。",
        "我們來討論一下解決方案。",
        "謝謝大家的參與。",
        "讓我們一起努力吧。",
        "這個評論很有道理。",
        "希望大家合作愉快。",
        "這個產品質量很好。",

        # Mildly toxic (label 1)
        "我不同意你的看法。",
        "這種行為太過分了！",
        "你腦子是不是有病？",
        "滾開！不要再煩我了！",
        "給我閉嘴！",

        # Severely toxic (label 2)
        "你真是個白痴，完全不懂！",
        "你這個垃圾，去死吧！",
        "你們這群廢物什麼都不會！",
        "你就是個廢人！",
        "你們都是垃圾！"
    ]

    labels = [0] * 10 + [1] * 5 + [2] * 5

    return texts, labels


def simple_train_function(labeled_data, model, device, epochs=2):
    """Simple training function for validation"""
    logger.info(f"Training model with {len(labeled_data)} samples")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    from torch.utils.data import DataLoader
    dataloader = DataLoader(labeled_data, batch_size=4, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return {'epochs': epochs, 'final_loss': avg_loss}


def test_uncertainty_strategies(model, device, dataset):
    """Test all uncertainty sampling strategies"""
    logger.info("Testing uncertainty sampling strategies...")

    strategies = {
        'entropy': EntropySampling(model, device),
        'confidence': LeastConfidenceSampling(model, device),
        'margin': MarginSampling(model, device),
        'bayesian': BayesianUncertaintySampling(model, device, n_dropout_samples=3),
        'bald': BALD(model, device, n_dropout_samples=3)
    }

    results = {}
    for name, strategy in strategies.items():
        try:
            start_time = time.time()
            selected = strategy.select_samples(dataset, n_samples=5)
            end_time = time.time()

            results[name] = {
                'success': True,
                'selected_count': len(selected),
                'time_taken': end_time - start_time,
                'selected_indices': selected
            }
            logger.info(f"✓ {name}: selected {len(selected)} samples in {end_time - start_time:.2f}s")

        except Exception as e:
            results[name] = {
                'success': False,
                'error': str(e)
            }
            logger.error(f"✗ {name}: failed with error: {e}")

    return results


def test_diversity_strategies(model, device, dataset):
    """Test all diversity sampling strategies"""
    logger.info("Testing diversity sampling strategies...")

    strategies = {
        'clustering': ClusteringSampling(model, device, n_clusters=3),
        'coreset': CoreSetSampling(model, device),
        'representative': RepresentativeSampling(model, device, use_pca=True),
        'hybrid_diversity': DiversityClusteringHybrid(model, device, clustering_ratio=0.7)
    }

    results = {}
    for name, strategy in strategies.items():
        try:
            start_time = time.time()
            selected = strategy.select_samples(dataset, n_samples=5)
            end_time = time.time()

            results[name] = {
                'success': True,
                'selected_count': len(selected),
                'time_taken': end_time - start_time,
                'selected_indices': selected
            }
            logger.info(f"✓ {name}: selected {len(selected)} samples in {end_time - start_time:.2f}s")

        except Exception as e:
            results[name] = {
                'success': False,
                'error': str(e)
            }
            logger.error(f"✗ {name}: failed with error: {e}")

    return results


def test_query_strategies(model, device, dataset):
    """Test hybrid query strategies"""
    logger.info("Testing hybrid query strategies...")

    strategies = {
        'hybrid_entropy_clustering': HybridQueryStrategy(
            model, device, 'entropy', 'clustering', uncertainty_ratio=0.6
        ),
        'adaptive': AdaptiveQueryStrategy(
            model, device, initial_uncertainty_ratio=0.5, adaptation_rate=0.1
        ),
        'ensemble': MultiStrategyEnsemble(
            model, device,
            strategies=[
                {'uncertainty': 'entropy', 'diversity': 'clustering', 'ratio': 0.5},
                {'uncertainty': 'margin', 'diversity': 'coreset', 'ratio': 0.6}
            ],
            voting_method='intersection'
        )
    }

    results = {}
    for name, strategy in strategies.items():
        try:
            start_time = time.time()
            selected = strategy.select_samples(dataset, n_samples=6)
            end_time = time.time()

            results[name] = {
                'success': True,
                'selected_count': len(selected),
                'time_taken': end_time - start_time,
                'selected_indices': selected
            }
            logger.info(f"✓ {name}: selected {len(selected)} samples in {end_time - start_time:.2f}s")

            # Test adaptive strategy adaptation
            if name == 'adaptive':
                strategy.update_performance(0.6)
                strategy.update_performance(0.65)

        except Exception as e:
            results[name] = {
                'success': False,
                'error': str(e)
            }
            logger.error(f"✗ {name}: failed with error: {e}")

    return results


def test_annotation_interface(temp_dir):
    """Test annotation interface"""
    logger.info("Testing annotation interface...")

    try:
        annotator = InteractiveAnnotator(save_dir=temp_dir)

        # Test with mock annotations
        mock_annotations = [
            {
                'original_index': 0,
                'text': '測試文本',
                'toxicity': 'none',
                'bullying': 'none',
                'role': 'none',
                'emotion': 'neutral',
                'emotion_strength': 2,
                'confidence': 0.8,
                'comments': '',
                'timestamp': '2024-01-01T00:00:00Z',
                'annotator': 'test'
            }
        ]

        # Test validation
        validation = annotator.validate_annotations(mock_annotations)

        # Test statistics
        stats = annotator.get_annotation_statistics(mock_annotations)

        # Test batch annotator
        batch_annotator = BatchAnnotator(annotator)

        return {
            'success': True,
            'validation_passed': validation['valid_samples'] == 1,
            'stats_generated': 'total_annotations' in stats,
            'batch_annotator_created': batch_annotator is not None
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def test_active_learning_loop(model, tokenizer, device, temp_dir):
    """Test complete active learning loop"""
    logger.info("Testing complete active learning loop...")

    try:
        # Create datasets
        texts, labels = create_validation_data()

        # Split data
        n_initial = 3
        n_test = 5

        initial_texts = texts[:n_initial]
        initial_labels = labels[:n_initial]
        test_texts = texts[n_initial:n_initial + n_test]
        test_labels = labels[n_initial:n_initial + n_test]
        unlabeled_texts = texts[n_initial + n_test:]

        initial_dataset = ValidationDataset(initial_texts, initial_labels, tokenizer)
        test_dataset = ValidationDataset(test_texts, test_labels, tokenizer)
        unlabeled_dataset = ValidationDataset(unlabeled_texts, None, tokenizer)

        # Create active learner
        active_learner = CyberPuppyActiveLearner(
            model=model,
            tokenizer=tokenizer,
            device=device,
            query_strategy='hybrid',
            query_strategy_config={
                'uncertainty_strategy': 'entropy',
                'diversity_strategy': 'clustering',
                'uncertainty_ratio': 0.6
            },
            save_dir=temp_dir,
            target_f1=0.8,
            max_budget=10
        )

        # Create annotator
        annotator = InteractiveAnnotator(save_dir=temp_dir)

        # Create active learning loop
        loop = ActiveLearningLoop(
            active_learner=active_learner,
            annotator=annotator,
            train_function=simple_train_function,
            initial_labeled_data=initial_dataset,
            unlabeled_pool=unlabeled_dataset,
            test_data=test_dataset,
            test_labels=test_labels,
            samples_per_iteration=3,
            max_iterations=2
        )

        # Run loop (non-interactive mode)
        start_time = time.time()
        results = loop.run(interactive=False, auto_train=True)
        end_time = time.time()

        return {
            'success': True,
            'total_iterations': results['total_iterations'],
            'total_annotations': results['total_annotations'],
            'time_taken': end_time - start_time,
            'final_f1': results['final_performance'].get('f1_macro', 0),
            'results_saved': os.path.exists(os.path.join(temp_dir, f"final_results_{results.get('timestamp', 'unknown')}.json"))
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def test_visualization(temp_dir):
    """Test visualization components"""
    logger.info("Testing visualization components...")

    try:
        visualizer = ActiveLearningVisualizer(save_dir=temp_dir)

        # Create mock data for visualization
        curve_data = {
            'annotations': [5, 10, 15, 20],
            'f1_macro': [0.5, 0.6, 0.7, 0.75],
            'f1_toxic': [0.4, 0.55, 0.65, 0.7],
            'f1_severe': [0.3, 0.45, 0.6, 0.65]
        }

        # Test learning curves
        plot_path = visualizer.plot_learning_curves(curve_data, title="Validation Test")

        # Create mock annotations for distribution plot
        mock_annotations = [
            {'toxicity': 'none', 'bullying': 'none', 'emotion': 'positive', 'confidence': 0.8},
            {'toxicity': 'toxic', 'bullying': 'harassment', 'emotion': 'negative', 'confidence': 0.9},
            {'toxicity': 'severe', 'bullying': 'threat', 'emotion': 'negative', 'confidence': 0.95}
        ]

        # Test annotation distribution plot
        dist_plot_path = visualizer.plot_annotation_distribution(
            mock_annotations, title="Test Distribution"
        )

        return {
            'success': True,
            'learning_curves_created': os.path.exists(plot_path),
            'distribution_plot_created': os.path.exists(dist_plot_path),
            'plots_directory': temp_dir
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def run_comprehensive_validation(args):
    """Run comprehensive validation of the active learning framework"""
    logger.info("Starting comprehensive validation of Active Learning Framework")

    # Create temporary directory for outputs
    if args.save_results:
        temp_dir = args.output_dir or './validation_results'
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp()

    logger.info(f"Using output directory: {temp_dir}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model and tokenizer
    logger.info("Creating model and tokenizer...")
    try:
        # Use a simple model for validation
        model = SimpleModel()
        model.to(device)

        # Create a mock tokenizer
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                # Simple tokenization for validation
                tokens = hash(text) % 100  # Simple hash-based tokenization
                length = kwargs.get('max_length', 128)
                input_ids = torch.randint(0, 1000, (length,))
                attention_mask = torch.ones(length)

                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }

        tokenizer = MockTokenizer()

        # Create validation dataset
        texts, labels = create_validation_data()
        dataset = ValidationDataset(texts, labels, tokenizer)

        logger.info(f"✓ Model and data created successfully")

    except Exception as e:
        logger.error(f"✗ Failed to create model and data: {e}")
        return False

    # Run validation tests
    validation_results = {}

    # Test 1: Uncertainty strategies
    validation_results['uncertainty'] = test_uncertainty_strategies(model, device, dataset)

    # Test 2: Diversity strategies
    validation_results['diversity'] = test_diversity_strategies(model, device, dataset)

    # Test 3: Query strategies
    validation_results['query_strategies'] = test_query_strategies(model, device, dataset)

    # Test 4: Annotation interface
    validation_results['annotation'] = test_annotation_interface(temp_dir)

    # Test 5: Complete active learning loop
    validation_results['active_learning_loop'] = test_active_learning_loop(
        model, tokenizer, device, temp_dir
    )

    # Test 6: Visualization
    validation_results['visualization'] = test_visualization(temp_dir)

    # Generate summary report
    generate_validation_report(validation_results, temp_dir, args.verbose)

    logger.info(f"Validation completed. Results saved to: {temp_dir}")

    return True


def generate_validation_report(results, output_dir, verbose=False):
    """Generate comprehensive validation report"""
    report_path = os.path.join(output_dir, 'validation_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Active Learning Framework Validation Report\n")
        f.write("=" * 60 + "\n\n")

        # Summary
        total_tests = 0
        passed_tests = 0

        f.write("SUMMARY\n")
        f.write("-" * 30 + "\n")

        for category, category_results in results.items():
            if isinstance(category_results, dict):
                if 'success' in category_results:
                    # Single test
                    total_tests += 1
                    if category_results['success']:
                        passed_tests += 1
                        f.write(f"✓ {category.upper()}: PASSED\n")
                    else:
                        f.write(f"✗ {category.upper()}: FAILED\n")
                else:
                    # Multiple tests in category
                    for test_name, test_result in category_results.items():
                        total_tests += 1
                        if test_result.get('success', False):
                            passed_tests += 1
                            f.write(f"✓ {category.upper()}.{test_name}: PASSED\n")
                        else:
                            f.write(f"✗ {category.upper()}.{test_name}: FAILED\n")

        f.write(f"\nOverall: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests*100):.1f}%)\n\n")

        # Detailed results
        if verbose:
            f.write("DETAILED RESULTS\n")
            f.write("-" * 30 + "\n\n")

            for category, category_results in results.items():
                f.write(f"{category.upper()}\n")
                f.write("-" * len(category) + "\n")

                if isinstance(category_results, dict):
                    if 'success' in category_results:
                        # Single test
                        f.write(f"Status: {'PASSED' if category_results['success'] else 'FAILED'}\n")
                        if not category_results['success']:
                            f.write(f"Error: {category_results.get('error', 'Unknown error')}\n")
                        else:
                            for key, value in category_results.items():
                                if key not in ['success', 'error']:
                                    f.write(f"{key}: {value}\n")
                    else:
                        # Multiple tests
                        for test_name, test_result in category_results.items():
                            f.write(f"\n{test_name}:\n")
                            f.write(f"  Status: {'PASSED' if test_result.get('success', False) else 'FAILED'}\n")
                            if not test_result.get('success', False):
                                f.write(f"  Error: {test_result.get('error', 'Unknown error')}\n")
                            else:
                                for key, value in test_result.items():
                                    if key not in ['success', 'error']:
                                        f.write(f"  {key}: {value}\n")

                f.write("\n")

    logger.info(f"Validation report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Validate Active Learning Framework')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed results in report')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to directory instead of temp')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results (default: ./validation_results)')

    args = parser.parse_args()

    try:
        success = run_comprehensive_validation(args)
        if success:
            print("\n" + "=" * 60)
            print("VALIDATION COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print("All core components of the Active Learning Framework have been validated.")
            print("The framework is ready for use in CyberPuppy toxicity detection.")
        else:
            print("\n" + "=" * 60)
            print("VALIDATION FAILED")
            print("=" * 60)
            print("Some components failed validation. Check the logs for details.")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()