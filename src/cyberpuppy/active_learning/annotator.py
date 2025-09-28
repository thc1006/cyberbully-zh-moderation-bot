"""
Interactive Annotator for Active Learning
"""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class InteractiveAnnotator:
    """
    Command-line interface for interactive annotation
    """

    def __init__(self, save_dir: str = "./annotations"):
        """
        Initialize interactive annotator

        Args:
            save_dir: Directory to save annotations
        """
        self.save_dir = save_dir
        self.annotations = []
        self.session_start = datetime.now()

        # Label mappings for CyberPuppy
        self.toxicity_labels = {
            "0": ("none", "Non-toxic content"),
            "1": ("toxic", "Mildly toxic content"),
            "2": ("severe", "Severely toxic content"),
        }

        self.bullying_labels = {
            "0": ("none", "No bullying behavior"),
            "1": ("harassment", "Harassment behavior"),
            "2": ("threat", "Threatening behavior"),
        }

        self.role_labels = {
            "0": ("none", "No specific role"),
            "1": ("perpetrator", "Initiating harmful behavior"),
            "2": ("victim", "Target of harmful behavior"),
            "3": ("bystander", "Witnessing the situation"),
        }

        self.emotion_labels = {
            "0": ("negative", "Negative emotion"),
            "1": ("neutral", "Neutral emotion"),
            "2": ("positive", "Positive emotion"),
        }

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"Initialized InteractiveAnnotator, saving to {save_dir}")

    def annotate_samples(
        self,
        samples: List[Dict[str, Any]],
        sample_indices: List[int],
        show_predictions: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Annotate a list of samples interactively

        Args:
            samples: List of sample dictionaries with text and metadata
            sample_indices: Original indices of the samples
            show_predictions: Whether to show model predictions

        Returns:
            List of annotation dictionaries
        """
        print(f"\n{'='*60}")
        print("Interactive Annotation Session")
        print(f"Samples to annotate: {len(samples)}")
        print(f"Session started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        annotations = []

        for i, (sample, original_idx) in enumerate(zip(samples, sample_indices)):
            print(f"\n{'-'*40}")
            print(f"Sample {i+1}/{len(samples)} (Original Index: {original_idx})")
            print(f"{'-'*40}")

            # Display the text
            text = sample.get("text", sample.get("content", ""))
            print(f"Text: {text}")

            # Show metadata if available
            if "metadata" in sample:
                print(f"Metadata: {sample['metadata']}")

            # Show model predictions if requested
            if show_predictions and "predictions" in sample:
                self._display_predictions(sample["predictions"])

            # Get annotations for different tasks
            annotation = {
                "original_index": original_idx,
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "annotator": "human_interactive",
            }

            # Get toxicity annotation
            annotation["toxicity"] = self._get_toxicity_annotation()

            # Get bullying annotation
            annotation["bullying"] = self._get_bullying_annotation()

            # Get role annotation (if bullying is present)
            if annotation["bullying"] != "none":
                annotation["role"] = self._get_role_annotation()
            else:
                annotation["role"] = "none"

            # Get emotion annotation
            annotation["emotion"] = self._get_emotion_annotation()

            # Get emotion strength (0-4)
            annotation["emotion_strength"] = self._get_emotion_strength()

            # Ask for confidence rating
            annotation["confidence"] = self._get_confidence_rating()

            # Ask for optional comments
            annotation["comments"] = self._get_comments()

            annotations.append(annotation)

            # Show summary
            self._display_annotation_summary(annotation)

            # Ask if user wants to continue or save and exit
            if i < len(samples) - 1:  # Not the last sample
                choice = input("\nContinue to next sample? [y/n/s(ave and exit)]: ").strip().lower()
                if choice == "s":
                    print("Saving annotations and exiting...")
                    break
                elif choice == "n":
                    print("Annotation session stopped by user")
                    break

        # Save annotations
        self._save_annotations(annotations)

        print(f"\n{'='*60}")
        print("Annotation session completed!")
        print(f"Annotated samples: {len(annotations)}")
        print(f"Saved to: {self.save_dir}")
        print(f"{'='*60}")

        return annotations

    def _display_predictions(self, predictions: Dict[str, Any]):
        """Display model predictions"""
        print("\nModel Predictions:")
        for task, pred in predictions.items():
            if isinstance(pred, dict):
                if "label" in pred and "confidence" in pred:
                    print(f"  {task}: {pred['label']} (confidence: {pred['confidence']:.3f})")
            else:
                print(f"  {task}: {pred}")

    def _get_toxicity_annotation(self) -> str:
        """Get toxicity label from user"""
        print("\nToxicity Level:")
        for key, (label, desc) in self.toxicity_labels.items():
            print(f"  {key}: {label} - {desc}")

        while True:
            choice = input("Enter toxicity level [0/1/2]: ").strip()
            if choice in self.toxicity_labels:
                label, _ = self.toxicity_labels[choice]
                return label
            print("Invalid choice. Please enter 0, 1, or 2.")

    def _get_bullying_annotation(self) -> str:
        """Get bullying label from user"""
        print("\nBullying Behavior:")
        for key, (label, desc) in self.bullying_labels.items():
            print(f"  {key}: {label} - {desc}")

        while True:
            choice = input("Enter bullying type [0/1/2]: ").strip()
            if choice in self.bullying_labels:
                label, _ = self.bullying_labels[choice]
                return label
            print("Invalid choice. Please enter 0, 1, or 2.")

    def _get_role_annotation(self) -> str:
        """Get role label from user"""
        print("\nRole in Interaction:")
        for key, (label, desc) in self.role_labels.items():
            print(f"  {key}: {label} - {desc}")

        while True:
            choice = input("Enter role [0/1/2/3]: ").strip()
            if choice in self.role_labels:
                label, _ = self.role_labels[choice]
                return label
            print("Invalid choice. Please enter 0, 1, 2, or 3.")

    def _get_emotion_annotation(self) -> str:
        """Get emotion label from user"""
        print("\nEmotion:")
        for key, (label, desc) in self.emotion_labels.items():
            print(f"  {key}: {label} - {desc}")

        while True:
            choice = input("Enter emotion [0/1/2]: ").strip()
            if choice in self.emotion_labels:
                label, _ = self.emotion_labels[choice]
                return label
            print("Invalid choice. Please enter 0, 1, or 2.")

    def _get_emotion_strength(self) -> int:
        """Get emotion strength from user"""
        print("\nEmotion Strength (0=very weak, 4=very strong):")

        while True:
            try:
                strength = int(input("Enter strength [0-4]: ").strip())
                if 0 <= strength <= 4:
                    return strength
                print("Please enter a number between 0 and 4.")
            except ValueError:
                print("Please enter a valid number.")

    def _get_confidence_rating(self) -> float:
        """Get confidence rating from user"""
        print("\nHow confident are you in this annotation?")
        print("  1: Very uncertain")
        print("  2: Somewhat uncertain")
        print("  3: Neutral")
        print("  4: Somewhat confident")
        print("  5: Very confident")

        while True:
            try:
                confidence = int(input("Enter confidence [1-5]: ").strip())
                if 1 <= confidence <= 5:
                    return confidence / 5.0  # Convert to 0-1 scale
                print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")

    def _get_comments(self) -> str:
        """Get optional comments from user"""
        comments = input("\nOptional comments (press Enter to skip): ").strip()
        return comments

    def _display_annotation_summary(self, annotation: Dict[str, Any]):
        """Display summary of the annotation"""
        print("\nAnnotation Summary:")
        print(f"  Toxicity: {annotation['toxicity']}")
        print(f"  Bullying: {annotation['bullying']}")
        print(f"  Role: {annotation['role']}")
        print(f"  Emotion: {annotation['emotion']} (strength: {annotation['emotion_strength']})")
        print(f"  Confidence: {annotation['confidence']:.2f}")
        if annotation["comments"]:
            print(f"  Comments: {annotation['comments']}")

    def _save_annotations(self, annotations: List[Dict[str, Any]]):
        """Save annotations to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        json_path = os.path.join(self.save_dir, f"annotations_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        # Save as CSV for easy viewing
        csv_path = os.path.join(self.save_dir, f"annotations_{timestamp}.csv")
        if annotations:
            fieldnames = [
                "original_index",
                "text",
                "toxicity",
                "bullying",
                "role",
                "emotion",
                "emotion_strength",
                "confidence",
                "comments",
                "timestamp",
            ]

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for annotation in annotations:
                    writer.writerow(annotation)

        logger.info(f"Saved {len(annotations)} annotations to {json_path} and {csv_path}")

    def load_annotations(self, file_path: str) -> List[Dict[str, Any]]:
        """Load annotations from file"""
        with open(file_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        logger.info(f"Loaded {len(annotations)} annotations from {file_path}")
        return annotations

    def get_annotation_statistics(self, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about annotations"""
        if not annotations:
            return {}

        stats = {
            "total_annotations": len(annotations),
            "toxicity_distribution": {},
            "bullying_distribution": {},
            "role_distribution": {},
            "emotion_distribution": {},
            "avg_confidence": 0.0,
            "avg_emotion_strength": 0.0,
        }

        # Count distributions
        for annotation in annotations:
            # Toxicity
            tox = annotation.get("toxicity", "unknown")
            stats["toxicity_distribution"][tox] = stats["toxicity_distribution"].get(tox, 0) + 1

            # Bullying
            bully = annotation.get("bullying", "unknown")
            stats["bullying_distribution"][bully] = stats["bullying_distribution"].get(bully, 0) + 1

            # Role
            role = annotation.get("role", "unknown")
            stats["role_distribution"][role] = stats["role_distribution"].get(role, 0) + 1

            # Emotion
            emotion = annotation.get("emotion", "unknown")
            stats["emotion_distribution"][emotion] = (
                stats["emotion_distribution"].get(emotion, 0) + 1
            )

        # Calculate averages
        confidences = [a.get("confidence", 0) for a in annotations]
        stats["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0

        strengths = [a.get("emotion_strength", 0) for a in annotations]
        stats["avg_emotion_strength"] = sum(strengths) / len(strengths) if strengths else 0

        return stats

    def validate_annotations(self, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate annotation consistency and quality"""
        validation_results = {"total_samples": len(annotations), "valid_samples": 0, "issues": []}

        for i, annotation in enumerate(annotations):
            issues = []

            # Check required fields
            required_fields = ["toxicity", "bullying", "role", "emotion", "emotion_strength"]
            for field in required_fields:
                if field not in annotation:
                    issues.append(f"Missing required field: {field}")

            # Check value consistency
            if annotation.get("bullying") == "none" and annotation.get("role") not in [
                "none",
                None,
            ]:
                issues.append("Role specified but no bullying behavior")

            # Check confidence range
            confidence = annotation.get("confidence", 0)
            if not (0 <= confidence <= 1):
                issues.append(f"Confidence out of range: {confidence}")

            # Check emotion strength range
            emotion_strength = annotation.get("emotion_strength", 0)
            if not (0 <= emotion_strength <= 4):
                issues.append(f"Emotion strength out of range: {emotion_strength}")

            if not issues:
                validation_results["valid_samples"] += 1
            else:
                validation_results["issues"].append(
                    {
                        "sample_index": i,
                        "original_index": annotation.get("original_index"),
                        "issues": issues,
                    }
                )

        return validation_results


class BatchAnnotator:
    """Batch processor for large-scale annotation tasks"""

    def __init__(self, annotator: InteractiveAnnotator):
        """
        Initialize batch annotator

        Args:
            annotator: InteractiveAnnotator instance
        """
        self.annotator = annotator

    def process_batch(
        self, samples: List[Dict[str, Any]], batch_size: int = 10, save_frequency: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process samples in batches with periodic saving

        Args:
            samples: All samples to annotate
            batch_size: Number of samples per batch
            save_frequency: Save every N batches

        Returns:
            All annotations collected
        """
        all_annotations = []
        total_batches = (len(samples) + batch_size - 1) // batch_size

        print(f"Processing {len(samples)} samples in {total_batches} batches of {batch_size}")

        for batch_idx in range(0, len(samples), batch_size):
            batch_samples = samples[batch_idx : batch_idx + batch_size]
            batch_indices = list(range(batch_idx, min(batch_idx + batch_size, len(samples))))

            current_batch = (batch_idx // batch_size) + 1
            print(f"\n{'='*50}")
            print(f"Batch {current_batch}/{total_batches}")
            print(f"{'='*50}")

            try:
                batch_annotations = self.annotator.annotate_samples(batch_samples, batch_indices)
                all_annotations.extend(batch_annotations)

                # Periodic save
                if current_batch % save_frequency == 0:
                    self._save_progress(all_annotations, current_batch)

            except KeyboardInterrupt:
                print("\nBatch annotation interrupted. Saving progress...")
                self._save_progress(all_annotations, current_batch)
                break

        return all_annotations

    def _save_progress(self, annotations: List[Dict[str, Any]], batch_num: int):
        """Save progress during batch processing"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        progress_file = os.path.join(
            self.annotator.save_dir, f"progress_batch_{batch_num}_{timestamp}.json"
        )

        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)

        print(f"Progress saved: {len(annotations)} annotations in {progress_file}")

    def resume_from_progress(self, progress_file: str) -> List[Dict[str, Any]]:
        """Resume annotation from a progress file"""
        return self.annotator.load_annotations(progress_file)
