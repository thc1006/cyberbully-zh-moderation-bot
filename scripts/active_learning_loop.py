#!/usr/bin/env python3
"""
Active Learning Loop Script for CyberPuppy

This script implements the main active learning workflow:
1. Load initial model and data
2. Select uncertain/diverse samples
3. Annotate samples interactively
4. Retrain model with new annotations
5. Evaluate and repeat until target F1 achieved

Usage:
    python scripts/active_learning_loop.py --config config/active_learning.yaml
    python scripts/active_learning_loop.py --initial-samples 100 --target-f1 0.75
"""

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from cyberpuppy.active_learning.active_learner import CyberPuppyActiveLearner
from cyberpuppy.active_learning.annotator import InteractiveAnnotator
from cyberpuppy.active_learning.loop import ActiveLearningLoop, BatchActiveLearningLoop
from cyberpuppy.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleDataset(Dataset):
    """Simple dataset for active learning examples"""

    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
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
            return {'text': text}


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return {}


def create_default_config() -> dict:
    """Create default configuration"""
    return {
        'model': {
            'name': 'hfl/chinese-roberta-wwm-ext',
            'num_labels': 3,
            'max_length': 512
        },
        'active_learning': {
            'strategy': 'hybrid',
            'uncertainty_strategy': 'entropy',
            'diversity_strategy': 'clustering',
            'uncertainty_ratio': 0.5,
            'samples_per_iteration': 20,
            'max_iterations': 50,
            'target_f1': 0.75,
            'max_budget': 1000
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 2e-5,
            'epochs': 3,
            'warmup_steps': 100
        },
        'data': {
            'save_dir': './active_learning_results',
            'annotation_dir': './annotations'
        }
    }


def load_sample_data():
    """Load or create sample data for demonstration"""
    # Sample Chinese texts for toxicity detection
    sample_texts = [
        "你好，今天天氣真好！",
        "這個想法很不錯，值得嘗試。",
        "你真是個白痴，完全不懂！",
        "我覺得這個方案有問題。",
        "滾開！不要再煩我了！",
        "感謝你的幫助，非常有用。",
        "你這個垃圾，去死吧！",
        "這件事情需要仔細考慮。",
        "我不同意你的看法。",
        "你們這群廢物什麼都不會！",
        "這個產品質量很好。",
        "希望大家合作愉快。",
        "你腦子是不是有病？",
        "我們來討論一下解決方案。",
        "這種行為太過分了！",
        "謝謝大家的參與。",
        "你就是個廢人！",
        "讓我們一起努力吧。",
        "這個評論很有道理。",
        "給我閉嘴！"
    ]

    # Sample labels: 0=none, 1=toxic, 2=severe
    sample_labels = [0, 0, 2, 0, 1, 0, 2, 0, 0, 2, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1]

    return sample_texts, sample_labels


def create_datasets(texts, labels, tokenizer, train_ratio=0.1, test_ratio=0.3):
    """Create train/test/unlabeled datasets"""
    n_total = len(texts)
    n_train = int(n_total * train_ratio)
    n_test = int(n_total * test_ratio)

    # Shuffle indices
    indices = np.random.permutation(n_total)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:n_train + n_test]
    unlabeled_indices = indices[n_train + n_test:]

    # Create datasets
    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_dataset = SimpleDataset(train_texts, train_labels, tokenizer)

    test_texts = [texts[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    test_dataset = SimpleDataset(test_texts, test_labels, tokenizer)

    unlabeled_texts = [texts[i] for i in unlabeled_indices]
    unlabeled_dataset = SimpleDataset(unlabeled_texts, None, tokenizer)

    logger.info(f"Created datasets - Train: {len(train_dataset)}, Test: {len(test_dataset)}, Unlabeled: {len(unlabeled_dataset)}")

    return train_dataset, test_dataset, unlabeled_dataset, test_labels


def simple_train_function(labeled_data, model, device, epochs=2):
    """Simple training function for demonstration"""
    logger.info(f"Training model with {len(labeled_data)} samples for {epochs} epochs")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    dataloader = DataLoader(labeled_data, batch_size=8, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    return {'epochs': epochs, 'final_loss': avg_loss}


def main():
    parser = argparse.ArgumentParser(description='Active Learning Loop for CyberPuppy')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--model-name', type=str, default='hfl/chinese-roberta-wwm-ext',
                       help='Model name or path')
    parser.add_argument('--initial-samples', type=int, default=2,
                       help='Number of initial labeled samples')
    parser.add_argument('--samples-per-iter', type=int, default=5,
                       help='Samples to annotate per iteration')
    parser.add_argument('--max-iterations', type=int, default=10,
                       help='Maximum number of iterations')
    parser.add_argument('--target-f1', type=float, default=0.75,
                       help='Target F1 score to stop learning')
    parser.add_argument('--max-budget', type=int, default=100,
                       help='Maximum annotation budget')
    parser.add_argument('--save-dir', type=str, default='./active_learning_results',
                       help='Directory to save results')
    parser.add_argument('--interactive', action='store_true',
                       help='Use interactive annotation (default: mock)')
    parser.add_argument('--batch-mode', action='store_true',
                       help='Use batch annotation mode')
    parser.add_argument('--resume', type=str,
                       help='Resume from checkpoint directory')
    parser.add_argument('--resume-iteration', type=int,
                       help='Iteration to resume from')

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()

    # Override config with command line arguments
    if args.model_name:
        config['model']['name'] = args.model_name
    if args.save_dir:
        config['data']['save_dir'] = args.save_dir

    config['active_learning'].update({
        'samples_per_iteration': args.samples_per_iter,
        'max_iterations': args.max_iterations,
        'target_f1': args.target_f1,
        'max_budget': args.max_budget
    })

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create save directories
    os.makedirs(config['data']['save_dir'], exist_ok=True)
    os.makedirs(config['data'].get('annotation_dir', './annotations'), exist_ok=True)

    try:
        # Load model and tokenizer
        logger.info(f"Loading model: {config['model']['name']}")
        tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        model = AutoModelForSequenceClassification.from_pretrained(
            config['model']['name'],
            num_labels=config['model']['num_labels']
        )
        model.to(device)

        # Load data (in real scenario, this would load your actual datasets)
        texts, labels = load_sample_data()

        # Create datasets
        train_dataset, test_dataset, unlabeled_dataset, test_labels = create_datasets(
            texts, labels, tokenizer, train_ratio=args.initial_samples/len(texts)
        )

        # Initialize active learner
        active_learner = CyberPuppyActiveLearner(
            model=model,
            tokenizer=tokenizer,
            device=device,
            query_strategy=config['active_learning']['strategy'],
            query_strategy_config={
                'uncertainty_strategy': config['active_learning']['uncertainty_strategy'],
                'diversity_strategy': config['active_learning']['diversity_strategy'],
                'uncertainty_ratio': config['active_learning']['uncertainty_ratio']
            },
            save_dir=config['data']['save_dir'],
            target_f1=config['active_learning']['target_f1'],
            max_budget=config['active_learning']['max_budget']
        )

        # Initialize annotator
        annotator = InteractiveAnnotator(
            save_dir=config['data'].get('annotation_dir', './annotations')
        )

        # Initialize active learning loop
        if args.batch_mode:
            loop = BatchActiveLearningLoop(
                active_learner=active_learner,
                annotator=annotator,
                train_function=simple_train_function,
                initial_labeled_data=train_dataset,
                unlabeled_pool=unlabeled_dataset,
                test_data=test_dataset,
                test_labels=test_labels,
                samples_per_iteration=config['active_learning']['samples_per_iteration'],
                max_iterations=config['active_learning']['max_iterations'],
                batch_size=20
            )
        else:
            loop = ActiveLearningLoop(
                active_learner=active_learner,
                annotator=annotator,
                train_function=simple_train_function,
                initial_labeled_data=train_dataset,
                unlabeled_pool=unlabeled_dataset,
                test_data=test_dataset,
                test_labels=test_labels,
                samples_per_iteration=config['active_learning']['samples_per_iteration'],
                max_iterations=config['active_learning']['max_iterations']
            )

        # Resume from checkpoint if specified
        if args.resume and args.resume_iteration is not None:
            loop.resume_from_checkpoint(args.resume, args.resume_iteration)

        # Run active learning loop
        logger.info("Starting Active Learning Loop")
        results = loop.run(interactive=args.interactive, auto_train=True)

        # Display final results
        print(f"\n{'='*60}")
        print("ACTIVE LEARNING COMPLETED")
        print(f"{'='*60}")
        print(f"Total iterations: {results['total_iterations']}")
        print(f"Total annotations: {results['total_annotations']}")
        print(f"Final F1 score: {results['final_performance'].get('f1_macro', 'N/A'):.4f}")
        print(f"Target achieved: {results['final_performance'].get('f1_macro', 0) >= config['active_learning']['target_f1']}")
        print(f"Time elapsed: {results['total_time_seconds']:.2f}s")
        print(f"Annotations per hour: {results['annotations_per_hour']:.1f}")
        print(f"Results saved to: {config['data']['save_dir']}")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        logger.info("Active learning interrupted by user")
    except Exception as e:
        logger.error(f"Error in active learning: {e}")
        raise


if __name__ == '__main__':
    main()