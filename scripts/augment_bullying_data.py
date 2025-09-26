#!/usr/bin/env python3
"""
Cyberbullying Data Augmentation Script

Augments Chinese cyberbullying detection datasets using multiple strategies:
- Synonym replacement with NTUSD sentiment dictionary
- Back-translation (Chinese ↔ English)
- Contextual perturbation with MacBERT
- Easy Data Augmentation (EDA)

Usage:
    python scripts/augment_bullying_data.py --input data/processed/cold_dataset.csv --output data/processed/cold_augmented.csv
    python scripts/augment_bullying_data.py --config configs/augmentation_heavy.yaml --intensity heavy
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import yaml
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberpuppy.data_augmentation import (
    AugmentationPipeline,
    PipelineConfig,
    AugmentationConfig,
    create_augmentation_pipeline
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('augmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def load_dataset(input_path: str) -> pd.DataFrame:
    """Load dataset from various formats."""
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading dataset from {input_path}")

    if input_path.suffix.lower() == '.csv':
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() in ['.json', '.jsonl']:
        df = pd.read_json(input_path, lines=input_path.suffix == '.jsonl')
    elif input_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    logger.info(f"Loaded {len(df)} samples from {input_path}")
    return df


def validate_dataset(df: pd.DataFrame, text_column: str, label_columns: list) -> None:
    """Validate dataset structure and content."""
    # Check required columns
    missing_cols = [col for col in [text_column] + label_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for empty texts
    empty_texts = df[text_column].isna().sum()
    if empty_texts > 0:
        logger.warning(f"Found {empty_texts} empty texts, will be skipped")

    # Check label distribution
    for col in label_columns:
        if col in df.columns:
            dist = df[col].value_counts()
            logger.info(f"Label distribution for {col}:")
            for label, count in dist.items():
                logger.info(f"  {label}: {count} ({count/len(df)*100:.1f}%)")


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Save augmented dataset to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving augmented dataset to {output_path}")

    if output_path.suffix.lower() == '.csv':
        df.to_csv(output_path, index=False, encoding='utf-8')
    elif output_path.suffix.lower() == '.json':
        df.to_json(output_path, orient='records', force_ascii=False, indent=2)
    elif output_path.suffix.lower() == '.jsonl':
        df.to_json(output_path, orient='records', lines=True, force_ascii=False)
    elif output_path.suffix.lower() in ['.xlsx', '.xls']:
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")

    logger.info(f"Saved {len(df)} samples to {output_path}")


def analyze_augmentation_results(original_df: pd.DataFrame, augmented_df: pd.DataFrame,
                               text_column: str, label_columns: list) -> Dict[str, Any]:
    """Analyze augmentation results and generate report."""
    original_size = len(original_df)
    augmented_size = len(augmented_df)
    new_samples = augmented_size - original_size

    analysis = {
        'original_size': original_size,
        'augmented_size': augmented_size,
        'new_samples': new_samples,
        'augmentation_ratio': new_samples / original_size if original_size > 0 else 0,
        'label_distributions': {}
    }

    # Analyze label distributions
    for col in label_columns:
        if col in original_df.columns and col in augmented_df.columns:
            original_dist = original_df[col].value_counts(normalize=True)
            augmented_dist = augmented_df[col].value_counts(normalize=True)

            analysis['label_distributions'][col] = {
                'original': original_dist.to_dict(),
                'augmented': augmented_dist.to_dict()
            }

    # Text length analysis
    original_lengths = original_df[text_column].str.len()
    augmented_lengths = augmented_df[text_column].str.len()

    analysis['text_length'] = {
        'original_mean': original_lengths.mean(),
        'original_std': original_lengths.std(),
        'augmented_mean': augmented_lengths.mean(),
        'augmented_std': augmented_lengths.std()
    }

    return analysis


def print_analysis_report(analysis: Dict[str, Any], pipeline_stats: Dict[str, Any]) -> None:
    """Print detailed analysis report."""
    print("\n" + "="*60)
    print("AUGMENTATION ANALYSIS REPORT")
    print("="*60)

    print(f"\nDataset Size:")
    print(f"  Original: {analysis['original_size']:,} samples")
    print(f"  Augmented: {analysis['augmented_size']:,} samples")
    print(f"  New samples: {analysis['new_samples']:,}")
    print(f"  Augmentation ratio: {analysis['augmentation_ratio']:.2%}")

    print(f"\nText Length Statistics:")
    print(f"  Original - Mean: {analysis['text_length']['original_mean']:.1f}, Std: {analysis['text_length']['original_std']:.1f}")
    print(f"  Augmented - Mean: {analysis['text_length']['augmented_mean']:.1f}, Std: {analysis['text_length']['augmented_std']:.1f}")

    print(f"\nAugmentation Pipeline Statistics:")
    print(f"  Total processed: {pipeline_stats['total_processed']:,}")
    print(f"  Total augmented: {pipeline_stats['total_augmented']:,}")
    print(f"  Quality filtered: {pipeline_stats['quality_filtered']:,}")
    print(f"  Quality pass rate: {pipeline_stats['quality_pass_rate']:.2%}")

    print(f"\nStrategy Usage:")
    for strategy, percentage in pipeline_stats['strategy_percentages'].items():
        print(f"  {strategy.capitalize()}: {percentage:.1f}%")

    print(f"\nLabel Distribution Changes:")
    for col, distributions in analysis['label_distributions'].items():
        print(f"\n  {col}:")
        for label in set(distributions['original'].keys()) | set(distributions['augmented'].keys()):
            orig_pct = distributions['original'].get(label, 0) * 100
            aug_pct = distributions['augmented'].get(label, 0) * 100
            change = aug_pct - orig_pct
            print(f"    {label}: {orig_pct:.1f}% → {aug_pct:.1f}% ({change:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Augment Chinese cyberbullying detection data")

    # Input/Output
    parser.add_argument('--input', '-i', required=True, help='Input dataset file')
    parser.add_argument('--output', '-o', required=True, help='Output augmented dataset file')
    parser.add_argument('--config', '-c', help='Configuration YAML file')

    # Dataset configuration
    parser.add_argument('--text-column', default='text', help='Name of text column')
    parser.add_argument('--label-columns', nargs='+',
                       default=['toxicity', 'bullying', 'role', 'emotion', 'emotion_strength'],
                       help='Names of label columns')

    # Augmentation configuration
    parser.add_argument('--intensity', choices=['light', 'medium', 'heavy'], default='medium',
                       help='Augmentation intensity preset')
    parser.add_argument('--strategies', nargs='+',
                       choices=['synonym', 'backtranslation', 'contextual', 'eda'],
                       help='Augmentation strategies to use')
    parser.add_argument('--augmentation-ratio', type=float, help='Proportion of data to augment')
    parser.add_argument('--augmentations-per-text', type=int, help='Number of augmentations per text')

    # Processing options
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--no-multiprocessing', action='store_true', help='Disable multiprocessing')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')

    # Quality control
    parser.add_argument('--quality-threshold', type=float, help='Minimum quality threshold')
    parser.add_argument('--preserve-balance', action='store_true', help='Preserve label distribution')

    # Output options
    parser.add_argument('--save-analysis', help='Save analysis report to file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce output verbosity')

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        # Load configuration
        config_dict = load_config(args.config)

        # Load dataset
        df = load_dataset(args.input)

        # Validate dataset
        validate_dataset(df, args.text_column, args.label_columns)

        # Filter out empty texts
        original_size = len(df)
        df = df.dropna(subset=[args.text_column])
        df = df[df[args.text_column].str.strip() != '']
        if len(df) < original_size:
            logger.info(f"Filtered out {original_size - len(df)} empty texts")

        # Create augmentation pipeline
        if args.strategies or any(arg for arg in [args.augmentation_ratio, args.augmentations_per_text] if arg is not None):
            # Custom configuration
            pipeline_config = PipelineConfig(
                augmentation_ratio=args.augmentation_ratio or 0.3,
                augmentations_per_text=args.augmentations_per_text or 2,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                use_multiprocessing=not args.no_multiprocessing,
                random_seed=args.random_seed,
                preserve_label_distribution=args.preserve_balance,
                quality_threshold=args.quality_threshold or 0.3
            )

            if args.strategies:
                pipeline_config.use_synonym = 'synonym' in args.strategies
                pipeline_config.use_backtranslation = 'backtranslation' in args.strategies
                pipeline_config.use_contextual = 'contextual' in args.strategies
                pipeline_config.use_eda = 'eda' in args.strategies

            pipeline = AugmentationPipeline(pipeline_config)
        else:
            # Use preset intensity
            pipeline = create_augmentation_pipeline(args.intensity, args.strategies)
            # Override any specific configs
            if args.batch_size != 32:
                pipeline.config.batch_size = args.batch_size
            if args.num_workers != 4:
                pipeline.config.num_workers = args.num_workers
            if args.no_multiprocessing:
                pipeline.config.use_multiprocessing = False

        logger.info(f"Created augmentation pipeline with intensity: {args.intensity}")
        logger.info(f"Enabled strategies: {list(pipeline.augmenters.keys())}")

        # Perform augmentation
        start_time = time.time()

        augmented_df = pipeline.augment_dataframe(
            df,
            text_column=args.text_column,
            label_columns=args.label_columns,
            verbose=not args.quiet
        )

        end_time = time.time()
        processing_time = end_time - start_time

        logger.info(f"Augmentation completed in {processing_time:.2f} seconds")

        # Save augmented dataset
        save_dataset(augmented_df, args.output)

        # Generate analysis
        pipeline_stats = pipeline.get_statistics()
        analysis = analyze_augmentation_results(df, augmented_df, args.text_column, args.label_columns)

        # Print report
        if not args.quiet:
            print_analysis_report(analysis, pipeline_stats)

        # Save analysis if requested
        if args.save_analysis:
            analysis_data = {
                'analysis': analysis,
                'pipeline_stats': pipeline_stats,
                'processing_time': processing_time,
                'config': {
                    'intensity': args.intensity,
                    'strategies': list(pipeline.augmenters.keys()),
                    'augmentation_ratio': pipeline.config.augmentation_ratio,
                    'augmentations_per_text': pipeline.config.augmentations_per_text
                }
            }

            with open(args.save_analysis, 'w', encoding='utf-8') as f:
                yaml.dump(analysis_data, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"Saved analysis report to {args.save_analysis}")

        logger.info("Augmentation process completed successfully!")

    except Exception as e:
        logger.error(f"Augmentation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()