#!/usr/bin/env python3
"""
Checkpoint Path Fixer
修復 checkpoint 檔案中的模組路徑問題

This script fixes module path issues in PyTorch checkpoint files
where models were saved with absolute paths that include 'src' prefix.
"""

import sys
import torch
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_checkpoint_paths(checkpoint_path: Path, output_path: Path = None):
    """
    Fix module paths in checkpoint files

    Args:
        checkpoint_path: Path to the checkpoint file
        output_path: Optional output path (defaults to overwriting original)
    """
    if output_path is None:
        output_path = checkpoint_path

    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    try:
        # Load checkpoint with path fixing
        import pickle
        import io

        class PathFixingUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Fix module path issues
                if module.startswith('src.cyberpuppy'):
                    module = module.replace('src.cyberpuppy', 'cyberpuppy')
                elif module.startswith('src.'):
                    module = module[4:]  # Remove 'src.' prefix
                return super().find_class(module, name)

        with open(checkpoint_path, 'rb') as f:
            unpickler = PathFixingUnpickler(f)
            checkpoint = unpickler.load()

        # Convert to expected format for torch.load compatibility
        if not isinstance(checkpoint, dict):
            checkpoint = {'model_state_dict': checkpoint}

        logger.info("Checkpoint loaded successfully")

        # Check and fix config if it exists
        if "config" in checkpoint:
            config = checkpoint["config"]
            logger.info(f"Config type: {type(config)}")

            # If config is a custom class, convert to dict
            if hasattr(config, '__dict__'):
                config_dict = {
                    'model_name': getattr(config, 'model_name', 'hfl/chinese-macbert-base'),
                    'max_length': getattr(config, 'max_length', 256),
                    'num_toxicity_classes': getattr(config, 'num_toxicity_classes', 3),
                    'num_bullying_classes': getattr(config, 'num_bullying_classes', 3),
                    'num_role_classes': getattr(config, 'num_role_classes', 4),
                    'num_emotion_classes': getattr(config, 'num_emotion_classes', 3),
                    'use_emotion_regression': getattr(config, 'use_emotion_regression', False),
                    'task_weights': getattr(config, 'task_weights', {
                        'toxicity': 1.0, 'bullying': 1.0, 'role': 0.5,
                        'emotion': 0.8, 'emotion_intensity': 0.6
                    }),
                    'hidden_dropout': getattr(config, 'hidden_dropout', 0.1),
                    'classifier_dropout': getattr(config, 'classifier_dropout', 0.1),
                    'hidden_size': getattr(config, 'hidden_size', 768),
                    'use_focal_loss': getattr(config, 'use_focal_loss', True),
                    'focal_alpha': getattr(config, 'focal_alpha', 1.0),
                    'focal_gamma': getattr(config, 'focal_gamma', 2.0),
                    'label_smoothing': getattr(config, 'label_smoothing', 0.1),
                }
                checkpoint['config'] = config_dict
                logger.info("Converted config to dictionary")

        # Save fixed checkpoint
        logger.info(f"Saving fixed checkpoint to: {output_path}")
        torch.save(checkpoint, output_path)

        logger.info("Checkpoint fixed successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to fix checkpoint: {e}")
        return False


def main():
    """Main function to fix all checkpoint files"""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"

    checkpoint_paths = [
        models_dir / "macbert_base_demo" / "best.ckpt",
        models_dir / "toxicity_only_demo" / "best.ckpt",
    ]

    print("=" * 50)
    print("CHECKPOINT PATH FIXER")
    print("=" * 50)

    for ckpt_path in checkpoint_paths:
        if not ckpt_path.exists():
            print(f"[SKIP] Checkpoint not found: {ckpt_path}")
            continue

        print(f"\n[FIX] Processing: {ckpt_path}")

        # Create backup
        backup_path = ckpt_path.with_suffix(".ckpt.backup")
        if not backup_path.exists():
            print(f"[BACKUP] Creating backup: {backup_path}")
            import shutil
            shutil.copy2(ckpt_path, backup_path)

        # Fix checkpoint
        success = fix_checkpoint_paths(ckpt_path)
        status = "[OK]" if success else "[FAIL]"
        print(f"{status} {ckpt_path.parent.name}")

    print("\n" + "=" * 50)
    print("CHECKPOINT FIXING COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()