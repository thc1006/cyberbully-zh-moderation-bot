#!/usr/bin/env python3
"""
Interactive Annotation Interface for CyberPuppy

This script provides a command-line interface for annotating text samples
with toxicity, bullying, role, and emotion labels.

Usage:
    python scripts/annotation_interface.py --input data/samples.json --output annotations/
    python scripts/annotation_interface.py --batch-size 10 --resume progress.json
"""

import argparse
import json
import csv
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cyberpuppy.active_learning.annotator import InteractiveAnnotator, BatchAnnotator


def load_samples_from_json(file_path: str) -> List[Dict[str, Any]]:
    """Load samples from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle different JSON formats
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        if 'samples' in data:
            samples = data['samples']
        elif 'data' in data:
            samples = data['data']
        else:
            # Assume the dict itself is the sample
            samples = [data]
    else:
        raise ValueError("Unsupported JSON format")

    # Ensure each sample has required fields
    processed_samples = []
    for i, sample in enumerate(samples):
        if isinstance(sample, str):
            # If sample is just text string
            processed_sample = {'text': sample, 'id': i}
        elif isinstance(sample, dict):
            # Ensure 'text' field exists
            if 'text' not in sample and 'content' not in sample:
                if 'message' in sample:
                    sample['text'] = sample['message']
                else:
                    raise ValueError(f"Sample {i} missing text field")
            processed_sample = sample
        else:
            raise ValueError(f"Invalid sample format at index {i}")

        processed_samples.append(processed_sample)

    return processed_samples


def load_samples_from_csv(file_path: str, text_column: str = 'text') -> List[Dict[str, Any]]:
    """Load samples from CSV file"""
    samples = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if text_column not in row:
                raise ValueError(f"Text column '{text_column}' not found in CSV")

            sample = {
                'text': row[text_column],
                'id': i,
                'metadata': {k: v for k, v in row.items() if k != text_column}
            }
            samples.append(sample)

    return samples


def load_samples_from_txt(file_path: str) -> List[Dict[str, Any]]:
    """Load samples from text file (one sample per line)"""
    samples = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:  # Skip empty lines
                sample = {'text': line, 'id': i}
                samples.append(sample)

    return samples


def save_samples_to_json(samples: List[Dict[str, Any]], file_path: str):
    """Save samples to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


def create_sample_data(output_file: str, num_samples: int = 20):
    """Create sample data for testing"""
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

    samples = []
    for i in range(min(num_samples, len(sample_texts))):
        sample = {
            'id': i,
            'text': sample_texts[i],
            'metadata': {
                'source': 'sample_data',
                'created_at': '2024-01-01T00:00:00Z'
            }
        }
        samples.append(sample)

    save_samples_to_json(samples, output_file)
    print(f"Created {len(samples)} sample records in {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Interactive Annotation Interface')

    # Input/Output options
    parser.add_argument('--input', type=str, help='Input file (JSON, CSV, or TXT)')
    parser.add_argument('--output', type=str, default='./annotations/',
                       help='Output directory for annotations')
    parser.add_argument('--format', choices=['json', 'csv', 'txt'], default='auto',
                       help='Input file format (auto-detect if not specified)')
    parser.add_argument('--text-column', type=str, default='text',
                       help='Column name for text in CSV files')

    # Annotation options
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Number of samples to annotate in each batch')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Start annotation from this sample index')
    parser.add_argument('--max-samples', type=int,
                       help='Maximum number of samples to annotate')
    parser.add_argument('--show-predictions', action='store_true',
                       help='Show model predictions during annotation')

    # Resume/Progress options
    parser.add_argument('--resume', type=str,
                       help='Resume annotation from progress file')
    parser.add_argument('--save-frequency', type=int, default=5,
                       help='Save progress every N batches')

    # Sample data creation
    parser.add_argument('--create-samples', type=str,
                       help='Create sample data file for testing')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of sample records to create')

    # Statistics and validation
    parser.add_argument('--stats', type=str,
                       help='Show statistics for annotation file')
    parser.add_argument('--validate', type=str,
                       help='Validate annotation file')

    args = parser.parse_args()

    # Create sample data if requested
    if args.create_samples:
        create_sample_data(args.create_samples, args.num_samples)
        return

    # Show statistics if requested
    if args.stats:
        annotator = InteractiveAnnotator()
        annotations = annotator.load_annotations(args.stats)
        stats = annotator.get_annotation_statistics(annotations)

        print(f"\n{'='*50}")
        print("ANNOTATION STATISTICS")
        print(f"{'='*50}")
        print(f"Total annotations: {stats['total_annotations']}")
        print(f"Average confidence: {stats['avg_confidence']:.3f}")
        print(f"Average emotion strength: {stats['avg_emotion_strength']:.2f}")

        print(f"\nToxicity distribution:")
        for label, count in stats['toxicity_distribution'].items():
            print(f"  {label}: {count}")

        print(f"\nBullying distribution:")
        for label, count in stats['bullying_distribution'].items():
            print(f"  {label}: {count}")

        print(f"\nEmotion distribution:")
        for label, count in stats['emotion_distribution'].items():
            print(f"  {label}: {count}")

        return

    # Validate annotations if requested
    if args.validate:
        annotator = InteractiveAnnotator()
        annotations = annotator.load_annotations(args.validate)
        validation = annotator.validate_annotations(annotations)

        print(f"\n{'='*50}")
        print("ANNOTATION VALIDATION")
        print(f"{'='*50}")
        print(f"Total samples: {validation['total_samples']}")
        print(f"Valid samples: {validation['valid_samples']}")
        print(f"Issues found: {len(validation['issues'])}")

        if validation['issues']:
            print(f"\nIssues:")
            for issue in validation['issues'][:10]:  # Show first 10 issues
                print(f"  Sample {issue['sample_index']}: {', '.join(issue['issues'])}")
            if len(validation['issues']) > 10:
                print(f"  ... and {len(validation['issues']) - 10} more issues")

        return

    # Ensure input file is provided for annotation
    if not args.input:
        print("Error: --input is required for annotation")
        print("Use --create-samples to create test data")
        print("Use --stats or --validate to analyze existing annotations")
        return

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return

    # Determine file format
    if args.format == 'auto':
        ext = os.path.splitext(args.input)[1].lower()
        if ext == '.json':
            file_format = 'json'
        elif ext == '.csv':
            file_format = 'csv'
        elif ext == '.txt':
            file_format = 'txt'
        else:
            print(f"Error: Cannot auto-detect format for {ext} files")
            return
    else:
        file_format = args.format

    # Load samples
    try:
        print(f"Loading samples from {args.input} (format: {file_format})")

        if file_format == 'json':
            samples = load_samples_from_json(args.input)
        elif file_format == 'csv':
            samples = load_samples_from_csv(args.input, args.text_column)
        elif file_format == 'txt':
            samples = load_samples_from_txt(args.input)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

        print(f"Loaded {len(samples)} samples")

    except Exception as e:
        print(f"Error loading samples: {e}")
        return

    # Apply start index and max samples
    if args.start_index > 0:
        samples = samples[args.start_index:]
        print(f"Starting from index {args.start_index}, {len(samples)} samples remaining")

    if args.max_samples and len(samples) > args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Limited to {args.max_samples} samples")

    if not samples:
        print("No samples to annotate")
        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize annotator
    annotator = InteractiveAnnotator(save_dir=args.output)

    try:
        if args.resume:
            # Resume from progress file
            print(f"Resuming from {args.resume}")
            batch_annotator = BatchAnnotator(annotator)
            existing_annotations = batch_annotator.resume_from_progress(args.resume)
            print(f"Loaded {len(existing_annotations)} existing annotations")

            # Continue with remaining samples
            # This is simplified - in practice you'd need to track which samples were already annotated
            remaining_samples = samples  # In real implementation, filter out already annotated

            all_annotations = existing_annotations + batch_annotator.process_batch(
                remaining_samples, args.batch_size, args.save_frequency
            )

        elif args.batch_size > 1:
            # Batch annotation mode
            batch_annotator = BatchAnnotator(annotator)
            all_annotations = batch_annotator.process_batch(
                samples, args.batch_size, args.save_frequency
            )

        else:
            # Single annotation mode
            sample_indices = list(range(len(samples)))
            all_annotations = annotator.annotate_samples(
                samples, sample_indices, show_predictions=args.show_predictions
            )

        # Final statistics
        if all_annotations:
            stats = annotator.get_annotation_statistics(all_annotations)
            print(f"\n{'='*50}")
            print("FINAL STATISTICS")
            print(f"{'='*50}")
            print(f"Total annotations completed: {stats['total_annotations']}")
            print(f"Average confidence: {stats['avg_confidence']:.3f}")
            print(f"Annotations saved to: {args.output}")

    except KeyboardInterrupt:
        print(f"\nAnnotation interrupted by user")
    except Exception as e:
        print(f"Error during annotation: {e}")
        raise


if __name__ == '__main__':
    main()