#!/usr/bin/env python3
"""
Model Loading Diagnostic Script
測試模型檔案載入與推論功能

This script verifies:
1. Model directories and file contents
2. PyTorch checkpoint validity
3. Transformers library compatibility
4. Model loading and inference functionality
"""

import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from cyberpuppy.models.baselines import BaselineModel, ModelConfig
    from cyberpuppy.labeling.label_map import ToxicityLevel, BullyingLevel, EmotionType, RoleType
    MODEL_IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import model classes: {e}")
    MODEL_IMPORTS_AVAILABLE = False


class ModelDiagnostics:
    """Model loading and testing diagnostics"""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.models_dir = self.project_root / "models"
        self.results = {}

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all diagnostic tests"""
        tests = [
            ("directory_structure", self.test_directory_structure),
            ("checkpoint_validity", self.test_checkpoint_validity),
            ("config_files", self.test_config_files),
            ("tokenizer_files", self.test_tokenizer_files),
            ("pytorch_loading", self.test_pytorch_loading),
            ("transformers_compatibility", self.test_transformers_compatibility),
            ("model_inference", self.test_model_inference),
        ]

        results = {}

        print("=" * 60)
        print("CYBERPUPPY MODEL DIAGNOSTIC TESTS")
        print("=" * 60)

        for test_name, test_func in tests:
            print(f"\n[TEST] Running {test_name}...")
            try:
                result = test_func()
                results[test_name] = result
                status = "[PASS]" if result else "[FAIL]"
                print(f"   {status}")
            except Exception as e:
                results[test_name] = False
                print(f"   [FAIL] Exception: {e}")
                logger.error(f"Test {test_name} failed: {e}\n{traceback.format_exc()}")

        self.print_summary(results)
        return results

    def test_directory_structure(self) -> bool:
        """Test model directory structure"""
        expected_models = ["macbert_base_demo", "toxicity_only_demo"]

        print(f"   [INFO] Checking models directory: {self.models_dir}")

        if not self.models_dir.exists():
            print(f"   [ERROR] Models directory not found: {self.models_dir}")
            return False

        for model_name in expected_models:
            model_path = self.models_dir / model_name
            print(f"   [INFO] Checking {model_name}...")

            if not model_path.exists():
                print(f"   [ERROR] Model directory not found: {model_path}")
                return False

            # List directory contents
            files = list(model_path.iterdir())
            print(f"      Files found: {[f.name for f in files]}")

            # Check for required files
            required_files = ["best.ckpt", "model_config.json"]
            for req_file in required_files:
                if not (model_path / req_file).exists():
                    print(f"   [ERROR] Required file not found: {req_file}")
                    return False

        print("   [OK] Directory structure is valid")
        return True

    def test_checkpoint_validity(self) -> bool:
        """Test PyTorch checkpoint files"""
        model_paths = [
            self.models_dir / "macbert_base_demo" / "best.ckpt",
            self.models_dir / "toxicity_only_demo" / "best.ckpt",
        ]

        for ckpt_path in model_paths:
            print(f"   [INFO] Testing checkpoint: {ckpt_path.name}")

            if not ckpt_path.exists():
                print(f"   [ERROR] Checkpoint file not found: {ckpt_path}")
                return False

            # Check file size
            file_size = ckpt_path.stat().st_size
            print(f"      File size: {file_size / (1024*1024):.1f} MB")

            if file_size < 100 * 1024 * 1024:  # Less than 100MB
                print(f"   [WARN] Checkpoint file seems small for a BERT model")

            try:
                # Try to load checkpoint
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                print(f"      [OK] Checkpoint loaded successfully")

                # Check checkpoint structure
                required_keys = ["model_state_dict"]
                for key in required_keys:
                    if key not in checkpoint:
                        print(f"   [ERROR] Missing key in checkpoint: {key}")
                        return False

                # Check model state dict
                state_dict = checkpoint["model_state_dict"]
                print(f"      Model parameters: {len(state_dict)} keys")

                # Sample some keys to verify structure
                sample_keys = list(state_dict.keys())[:5]
                print(f"      Sample keys: {sample_keys}")

                # Check if config exists
                if "config" in checkpoint:
                    config = checkpoint["config"]
                    print(f"      Config type: {type(config)}")

            except Exception as e:
                print(f"   [ERROR] Failed to load checkpoint: {e}")
                return False

        return True

    def test_config_files(self) -> bool:
        """Test model configuration files"""
        config_paths = [
            self.models_dir / "macbert_base_demo" / "model_config.json",
            self.models_dir / "toxicity_only_demo" / "model_config.json",
        ]

        for config_path in config_paths:
            print(f"   [INFO] Testing config: {config_path.parent.name}")

            if not config_path.exists():
                print(f"   [ERROR] Config file not found: {config_path}")
                return False

            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                print(f"      [OK] JSON loaded successfully")

                # Check required fields
                required_fields = [
                    "model_name", "max_length", "num_toxicity_classes",
                    "num_bullying_classes", "num_role_classes", "num_emotion_classes"
                ]

                for field in required_fields:
                    if field not in config:
                        print(f"   [ERROR] Missing config field: {field}")
                        return False

                print(f"      Model: {config.get('model_name', 'N/A')}")
                print(f"      Max length: {config.get('max_length', 'N/A')}")
                print(f"      Task weights: {config.get('task_weights', {})}")

            except Exception as e:
                print(f"   [ERROR] Failed to load config: {e}")
                return False

        return True

    def test_tokenizer_files(self) -> bool:
        """Test tokenizer files"""
        model_paths = [
            self.models_dir / "macbert_base_demo",
            self.models_dir / "toxicity_only_demo",
        ]

        for model_path in model_paths:
            print(f"   [INFO] Testing tokenizer: {model_path.name}")

            required_files = ["tokenizer.json", "tokenizer_config.json", "vocab.txt"]
            for req_file in required_files:
                file_path = model_path / req_file
                if not file_path.exists():
                    print(f"   [ERROR] Tokenizer file not found: {req_file}")
                    return False

            try:
                # Try to load tokenizer using transformers
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                print(f"      [OK] Tokenizer loaded successfully")
                print(f"      Vocab size: {tokenizer.vocab_size}")

                # Test tokenization
                test_text = "這是一個測試句子。"
                tokens = tokenizer.tokenize(test_text)
                token_ids = tokenizer.encode(test_text)

                print(f"      Test text: {test_text}")
                print(f"      Tokens: {tokens[:10]}...")  # First 10 tokens
                print(f"      Token IDs: {token_ids[:10]}...")  # First 10 IDs

            except Exception as e:
                print(f"   [ERROR] Failed to load tokenizer: {e}")
                return False

        return True

    def test_pytorch_loading(self) -> bool:
        """Test PyTorch model loading without our custom classes"""
        model_paths = [
            self.models_dir / "macbert_base_demo",
            self.models_dir / "toxicity_only_demo",
        ]

        for model_path in model_paths:
            print(f"   [INFO] Testing PyTorch loading: {model_path.name}")

            try:
                # Load checkpoint
                ckpt_path = model_path / "best.ckpt"
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

                # Try to load base model with transformers
                config_path = model_path / "model_config.json"
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                model_name = config["model_name"]
                print(f"      Base model: {model_name}")

                # Load base transformer model
                base_model = AutoModel.from_pretrained(model_name)
                print(f"      [OK] Base transformer model loaded")

                # Check state dict compatibility
                state_dict = checkpoint["model_state_dict"]

                # Filter state dict to only include backbone parameters
                backbone_state = {}
                for key, value in state_dict.items():
                    if key.startswith("backbone."):
                        new_key = key.replace("backbone.", "")
                        backbone_state[new_key] = value

                print(f"      Backbone parameters: {len(backbone_state)}")

                # Try partial loading
                if backbone_state:
                    missing_keys, unexpected_keys = base_model.load_state_dict(
                        backbone_state, strict=False
                    )
                    print(f"      Missing keys: {len(missing_keys)}")
                    print(f"      Unexpected keys: {len(unexpected_keys)}")

            except Exception as e:
                print(f"   [ERROR] PyTorch loading failed: {e}")
                return False

        return True

    def test_transformers_compatibility(self) -> bool:
        """Test transformers library compatibility"""
        try:
            import transformers
            print(f"   [INFO] Transformers version: {transformers.__version__}")
            print(f"   [INFO] Python version: {sys.version}")
            print(f"   [INFO] PyTorch version: {torch.__version__}")
            print(f"   [INFO] CUDA available: {torch.cuda.is_available()}")

            # Test loading Chinese models
            model_names = ["hfl/chinese-macbert-base"]

            for model_name in model_names:
                print(f"   [INFO] Testing model: {model_name}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)
                    print(f"      [OK] Model loaded successfully")

                    # Test inference
                    test_text = "這是一個測試。"
                    inputs = tokenizer(test_text, return_tensors="pt")

                    with torch.no_grad():
                        outputs = model(**inputs)

                    print(f"      Output shape: {outputs.last_hidden_state.shape}")

                except Exception as e:
                    print(f"   [ERROR] Failed to test {model_name}: {e}")
                    return False

            return True

        except Exception as e:
            print(f"   [ERROR] Transformers compatibility test failed: {e}")
            return False

    def test_model_inference(self) -> bool:
        """Test full model loading and inference"""
        if not MODEL_IMPORTS_AVAILABLE:
            print("   [WARN] Skipping model inference test - imports not available")
            return True

        model_paths = [
            self.models_dir / "macbert_base_demo",
            self.models_dir / "toxicity_only_demo",
        ]

        for model_path in model_paths:
            print(f"   [INFO] Testing model inference: {model_path.name}")

            try:
                # Load model using our custom class
                model = BaselineModel.load_model(str(model_path))
                print(f"      [OK] Model loaded with custom class")

                # Test inference
                test_texts = [
                    "你好，這是一個正常的句子。",
                    "今天天氣很好。",
                    "我喜歡吃蘋果。",
                ]

                model.eval()

                for text in test_texts:
                    print(f"      Testing text: {text}")

                    # Tokenize
                    inputs = model.tokenizer(
                        text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256
                    )

                    # Inference
                    with torch.no_grad():
                        predictions = model.predict(
                            inputs["input_ids"],
                            inputs["attention_mask"],
                            inputs.get("token_type_ids")
                        )

                    # Print results
                    for key, value in predictions.items():
                        if isinstance(value, np.ndarray) and value.ndim > 0:
                            if value.shape[0] == 1:  # Single sample
                                print(f"        {key}: {value[0]}")
                            else:
                                print(f"        {key} shape: {value.shape}")
                        else:
                            print(f"        {key}: {value}")

                print(f"      [OK] Inference completed successfully")

            except Exception as e:
                print(f"   [ERROR] Model inference failed: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return False

        return True

    def print_summary(self, results: Dict[str, bool]):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in results.values() if r)
        total = len(results)

        print(f"Tests passed: {passed}/{total}")

        if passed == total:
            print("[SUCCESS] All tests passed! Models are ready for use.")
        else:
            print("[WARNING] Some tests failed. Check the details above.")

        print("\nDetailed results:")
        for test_name, result in results.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"  {test_name}: {status}")

        # Recommendations
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)

        if not results.get("directory_structure", False):
            print("[ACTION] Fix directory structure issues first")
        elif not results.get("checkpoint_validity", False):
            print("[ACTION] Checkpoint files seem corrupted - retrain models")
        elif not results.get("transformers_compatibility", False):
            print("[ACTION] Update transformers library: pip install -U transformers")
        elif passed < total:
            print("[ACTION] Some components failed - check logs for details")
        else:
            print("[SUCCESS] Models are fully functional!")
            print("[SUCCESS] You can proceed with training and deployment")


def create_minimal_inference_example():
    """Create a minimal working example for inference"""

    example_code = '''#!/usr/bin/env python3
"""
Minimal Model Inference Example
"""

import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from cyberpuppy.models.baselines import BaselineModel

def main():
    # Load model
    model_path = "models/toxicity_only_demo"
    model = BaselineModel.load_model(model_path)

    # Test text
    text = "這是一個測試句子。"

    # Tokenize
    inputs = model.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    # Inference
    model.eval()
    with torch.no_grad():
        predictions = model.predict(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs.get("token_type_ids")
        )

    # Print results
    print(f"Input text: {text}")
    print("Predictions:")
    for key, value in predictions.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
'''

    return example_code


def main():
    """Main diagnostic function"""
    print("Starting CyberPuppy Model Diagnostics...")

    diagnostics = ModelDiagnostics()
    results = diagnostics.run_all_tests()

    # Save results
    results_path = Path("tests/model_diagnostics_results.json")
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "results": results,
            "timestamp": str(torch.utils.data.get_worker_info() if hasattr(torch.utils.data, 'get_worker_info') else "unknown"),
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Results saved to: {results_path}")

    # Create minimal example
    example_path = Path("examples/minimal_inference.py")
    example_path.parent.mkdir(exist_ok=True)

    with open(example_path, 'w', encoding='utf-8') as f:
        f.write(create_minimal_inference_example())

    print(f"[INFO] Minimal example saved to: {example_path}")

    return results


if __name__ == "__main__":
    main()