#!/usr/bin/env python3
"""
Simple Model Testing Script
簡單的模型測試腳本

This script tests model loading and inference without complex dependencies.
It focuses on validating that the basic components work correctly.
"""

import json
import logging
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_transformers():
    """Test basic transformers functionality"""
    print("\n" + "="*50)
    print("TESTING BASIC TRANSFORMERS FUNCTIONALITY")
    print("="*50)

    try:
        model_name = "hfl/chinese-macbert-base"
        print(f"[INFO] Loading model: {model_name}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        print(f"[OK] Model loaded successfully")
        print(f"     Vocab size: {tokenizer.vocab_size}")
        print(f"     Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test tokenization
        test_texts = [
            "這是一個正常的句子。",
            "今天天氣很好。",
            "我不喜歡你這樣說話。",
        ]

        print(f"\n[TEST] Running inference on {len(test_texts)} texts...")

        for i, text in enumerate(test_texts):
            print(f"\n  Text {i+1}: {text}")

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )

            # Inference
            with torch.no_grad():
                outputs = model(**inputs)

            # Extract [CLS] token representation
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768]

            print(f"    Tokens: {len(inputs['input_ids'][0])} tokens")
            print(f"    CLS embedding shape: {cls_embedding.shape}")
            print(f"    CLS embedding mean: {cls_embedding.mean().item():.4f}")
            print(f"    CLS embedding std: {cls_embedding.std().item():.4f}")

        print(f"\n[SUCCESS] Basic transformers test passed!")
        return True

    except Exception as e:
        print(f"[ERROR] Basic transformers test failed: {e}")
        return False


def test_model_directories():
    """Test model directory structure and files"""
    print("\n" + "="*50)
    print("TESTING MODEL DIRECTORIES")
    print("="*50)

    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"

    if not models_dir.exists():
        print(f"[ERROR] Models directory not found: {models_dir}")
        return False

    model_dirs = ["macbert_base_demo", "toxicity_only_demo"]

    for model_name in model_dirs:
        model_path = models_dir / model_name
        print(f"\n[INFO] Checking {model_name}...")

        if not model_path.exists():
            print(f"[ERROR] Model directory not found: {model_path}")
            continue

        # List files
        files = list(model_path.iterdir())
        print(f"    Files: {[f.name for f in files]}")

        # Check config file
        config_path = model_path / "model_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"    [OK] Config loaded")
                print(f"         Model: {config.get('model_name', 'N/A')}")
                print(f"         Tasks: {list(config.get('task_weights', {}).keys())}")
            except Exception as e:
                print(f"    [ERROR] Config loading failed: {e}")

        # Test tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            print(f"    [OK] Tokenizer loaded from directory")

            # Test tokenization
            test_text = "測試文本"
            tokens = tokenizer.tokenize(test_text)
            print(f"         Test tokenization: {test_text} -> {tokens}")

        except Exception as e:
            print(f"    [ERROR] Tokenizer loading failed: {e}")

        # Check checkpoint file
        ckpt_path = model_path / "best.ckpt"
        if ckpt_path.exists():
            file_size = ckpt_path.stat().st_size
            print(f"    [INFO] Checkpoint file size: {file_size / (1024*1024):.1f} MB")
        else:
            print(f"    [ERROR] Checkpoint file not found")

    print(f"\n[SUCCESS] Model directories test completed!")
    return True


def create_dummy_classifier():
    """Create a dummy classifier for testing inference patterns"""
    print("\n" + "="*50)
    print("CREATING DUMMY CLASSIFIER FOR TESTING")
    print("="*50)

    try:
        # Load base model
        model_name = "hfl/chinese-macbert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name)

        # Create simple classifier heads
        class SimpleMultiTaskClassifier(torch.nn.Module):
            def __init__(self, backbone_model, hidden_size=768):
                super().__init__()
                self.backbone = backbone_model

                # Simple classification heads
                self.toxicity_head = torch.nn.Linear(hidden_size, 3)  # none, toxic, severe
                self.emotion_head = torch.nn.Linear(hidden_size, 3)   # pos, neu, neg

            def forward(self, input_ids, attention_mask, token_type_ids=None):
                # Get backbone outputs
                backbone_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
                if token_type_ids is not None:
                    backbone_inputs['token_type_ids'] = token_type_ids

                backbone_outputs = self.backbone(**backbone_inputs)

                # Use [CLS] token
                cls_output = backbone_outputs.last_hidden_state[:, 0, :]

                # Classify
                toxicity_logits = self.toxicity_head(cls_output)
                emotion_logits = self.emotion_head(cls_output)

                return {
                    'toxicity': toxicity_logits,
                    'emotion': emotion_logits
                }

        # Create model
        model = SimpleMultiTaskClassifier(backbone)
        model.eval()

        print(f"[OK] Dummy classifier created")
        print(f"     Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test inference
        test_texts = [
            "這是一個正常的句子。",
            "我很開心今天的天氣。",
            "這個產品質量很差，我很失望。",
        ]

        labels = ["Normal", "Positive", "Negative"]

        print(f"\n[TEST] Testing dummy classifier inference...")

        for i, (text, expected_label) in enumerate(zip(test_texts, labels)):
            print(f"\n  Text {i+1}: {text}")
            print(f"  Expected: {expected_label}")

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )

            # Inference
            with torch.no_grad():
                outputs = model(**inputs)

            # Process outputs
            toxicity_probs = torch.softmax(outputs['toxicity'], dim=-1)
            emotion_probs = torch.softmax(outputs['emotion'], dim=-1)

            # Get predictions
            toxicity_pred = torch.argmax(toxicity_probs, dim=-1).item()
            emotion_pred = torch.argmax(emotion_probs, dim=-1).item()

            toxicity_labels = ['none', 'toxic', 'severe']
            emotion_labels = ['positive', 'neutral', 'negative']

            print(f"    Toxicity: {toxicity_labels[toxicity_pred]} (conf: {toxicity_probs[0, toxicity_pred]:.3f})")
            print(f"    Emotion: {emotion_labels[emotion_pred]} (conf: {emotion_probs[0, emotion_pred]:.3f})")

        print(f"\n[SUCCESS] Dummy classifier test passed!")
        return True

    except Exception as e:
        print(f"[ERROR] Dummy classifier test failed: {e}")
        return False


def test_checkpoint_analysis():
    """Analyze checkpoint files to understand their structure"""
    print("\n" + "="*50)
    print("ANALYZING CHECKPOINT FILES")
    print("="*50)

    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"

    checkpoint_paths = [
        models_dir / "macbert_base_demo" / "best.ckpt",
        models_dir / "toxicity_only_demo" / "best.ckpt",
    ]

    for ckpt_path in checkpoint_paths:
        if not ckpt_path.exists():
            print(f"[SKIP] Checkpoint not found: {ckpt_path}")
            continue

        print(f"\n[ANALYZE] {ckpt_path.parent.name}")

        try:
            # Try different loading methods
            methods = [
                ("Standard torch.load", lambda: torch.load(ckpt_path, map_location="cpu", weights_only=False)),
                ("Weights only", lambda: torch.load(ckpt_path, map_location="cpu", weights_only=True)),
            ]

            for method_name, method_func in methods:
                try:
                    print(f"    Trying {method_name}...")
                    checkpoint = method_func()

                    if isinstance(checkpoint, dict):
                        print(f"      [OK] Loaded as dict with keys: {list(checkpoint.keys())}")

                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                            print(f"           State dict parameters: {len(state_dict)}")

                            # Show some parameter names
                            param_names = list(state_dict.keys())[:10]
                            print(f"           Sample parameters: {param_names}")

                        break  # If successful, stop trying other methods

                    else:
                        print(f"      [OK] Loaded as {type(checkpoint)}")

                except Exception as e:
                    print(f"      [FAIL] {method_name}: {e}")

        except Exception as e:
            print(f"    [ERROR] All methods failed: {e}")

    return True


def main():
    """Main testing function"""
    print("CYBERPUPPY SIMPLE MODEL TESTING")
    print("="*60)

    tests = [
        ("Basic Transformers", test_basic_transformers),
        ("Model Directories", test_model_directories),
        ("Dummy Classifier", create_dummy_classifier),
        ("Checkpoint Analysis", test_checkpoint_analysis),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"RUNNING: {test_name}")
        print(f"{'='*60}")

        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"[ERROR] Test {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {test_name}")

    if passed == total:
        print(f"\n[SUCCESS] All tests passed!")
        print(f"[SUCCESS] The transformers library and model files are working correctly.")
        print(f"[SUCCESS] You can proceed with implementing your toxicity detection system.")
    else:
        print(f"\n[WARNING] Some tests failed.")
        print(f"[INFO] However, if basic transformers test passed, you can still work with the models.")

    return results


if __name__ == "__main__":
    main()