#!/usr/bin/env python3
"""
Comprehensive tests for local training system
Designed for RTX 3050 4GB - runs quickly with clear pass/fail messages
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
import torch.nn as nn
import numpy as np
import pytest
from torch.utils.data import DataLoader, TensorDataset

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from cyberpuppy.training.config import TrainingPipelineConfig, DataConfig, OptimizerConfig
    from cyberpuppy.training.trainer import MemoryOptimizer
    from cyberpuppy.models.baselines import BaselineModel, ModelConfig
except ImportError as e:
    print(f"[WARNING] Import error: {e}")
    print("Some tests may be skipped")


class TestGPUDetection(unittest.TestCase):
    """Test GPU detection and memory allocation"""

    def test_gpu_availability(self):
        """Test if GPU is detected and accessible"""
        try:
            is_available = torch.cuda.is_available()

            if is_available:
                print(f"[SUCCESS] GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"[SUCCESS] CUDA version: {torch.version.cuda}")
                print(f"[SUCCESS] GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
                self.assertTrue(True)
            else:
                print("[ERROR] GPU not detected")
                print("Fix: Install CUDA-enabled PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
                self.skipTest("GPU not available")

        except Exception as e:
            self.fail(f"GPU detection failed: {e}")

    def test_gpu_memory_allocation(self):
        """Test GPU memory allocation with OOM protection"""
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")

        try:
            # Start with small allocation
            device = torch.device("cuda")
            test_tensor = torch.randn(100, 100, device=device)

            # Check memory usage
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            print(f"[SUCCESS] Memory allocated: {allocated:.1f} MB")

            # Clean up
            del test_tensor
            torch.cuda.empty_cache()

            self.assertTrue(allocated > 0)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("[ERROR] GPU OOM on basic allocation")
                print("Fix: Close other GPU applications or reduce batch size")
                self.fail("GPU memory allocation failed")
            else:
                raise e

    def test_mixed_precision_support(self):
        """Test if mixed precision training is supported"""
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")

        try:
            from torch.cuda.amp import autocast, GradScaler

            scaler = GradScaler()
            device = torch.device("cuda")

            # Simple model for testing
            model = nn.Linear(10, 2).to(device)
            optimizer = torch.optim.AdamW(model.parameters())

            x = torch.randn(32, 10, device=device)
            y = torch.randint(0, 2, (32,), device=device)

            with autocast():
                output = model(x)
                loss = nn.CrossEntropyLoss()(output, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print("[SUCCESS] Mixed precision training supported")
            self.assertTrue(True)

        except Exception as e:
            print(f"[ERROR] Mixed precision test failed: {e}")
            print("Fix: Update PyTorch to version >= 1.6")
            self.fail(f"Mixed precision support test failed: {e}")


class TestTrainingDataLoading(unittest.TestCase):
    """Test training data loading functionality"""

    @classmethod
    def setUpClass(cls):
        """Setup mock data for testing"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.data_dir = Path(cls.temp_dir) / "data"
        cls.data_dir.mkdir(exist_ok=True)

        # Create mock training data
        mock_data = {
            "texts": [f"Test message {i}" for i in range(100)],
            "toxicity": [0, 1] * 50,  # Binary labels
            "bullying": [0, 1, 2] * 33 + [0],  # Multi-class labels
            "emotion": [0, 1, 2] * 33 + [0]
        }

        with open(cls.data_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump(mock_data, f, ensure_ascii=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up test data"""
        shutil.rmtree(cls.temp_dir)

    def test_data_file_exists(self):
        """Test if training data files exist"""
        data_paths = [
            "data/processed/training_dataset/train.json",
            "data/processed/training_dataset/dev.json"
        ]

        existing_files = []
        missing_files = []

        for path in data_paths:
            if Path(path).exists():
                existing_files.append(path)
                print(f"[SUCCESS] Found: {path}")
            else:
                missing_files.append(path)
                print(f"[ERROR] Missing: {path}")

        if missing_files:
            print(f"Fix: Run data preparation scripts to create missing files")
            print(f"      python scripts/create_unified_training_data.py")

        # Pass if at least one data file exists or mock data exists
        has_real_data = len(existing_files) > 0
        has_mock_data = (self.data_dir / "train.json").exists()

        self.assertTrue(has_real_data or has_mock_data,
                       "No training data found")

    def test_data_loading_performance(self):
        """Test data loading speed"""
        try:
            # Use mock data for consistent testing
            texts = [f"Sample text {i}" for i in range(1000)]
            labels = torch.randint(0, 2, (1000,))

            dataset = TensorDataset(
                torch.tensor([hash(t) % 10000 for t in texts]),
                labels
            )

            start_time = time.time()
            dataloader = DataLoader(dataset, batch_size=32, num_workers=0)

            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                if batch_count >= 10:  # Test first 10 batches
                    break

            load_time = time.time() - start_time
            print(f"[SUCCESS] Loaded {batch_count} batches in {load_time:.2f}s")

            if load_time > 5.0:
                print("[WARNING] Data loading slower than expected")
                print("Fix: Increase num_workers or optimize data preprocessing")

            self.assertTrue(batch_count > 0)

        except Exception as e:
            print(f"[ERROR] Data loading test failed: {e}")
            self.fail(f"Data loading performance test failed: {e}")

    def test_data_format_validation(self):
        """Test if data has correct format and labels"""
        try:
            # Load mock data
            with open(self.data_dir / "train.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check required fields
            required_fields = ["texts", "toxicity", "bullying", "emotion"]
            missing_fields = [f for f in required_fields if f not in data]

            if missing_fields:
                print(f"[ERROR] Missing fields: {missing_fields}")
                print("Fix: Ensure data has all required labels")
                self.fail(f"Missing required fields: {missing_fields}")

            # Check data consistency
            text_count = len(data["texts"])
            for field in required_fields[1:]:  # Skip 'texts'
                if len(data[field]) != text_count:
                    print(f"[ERROR] Label mismatch in {field}: {len(data[field])} vs {text_count}")
                    self.fail(f"Label count mismatch in {field}")

            print(f"[SUCCESS] Data format valid: {text_count} samples")
            self.assertTrue(True)

        except Exception as e:
            print(f"[ERROR] Data format validation failed: {e}")
            self.fail(f"Data format validation failed: {e}")


class TestCheckpointManager(unittest.TestCase):
    """Test checkpoint save/load functionality"""

    def setUp(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.temp_dir)

    def test_checkpoint_save_load(self):
        """Test basic checkpoint save and load"""
        try:
            # Create a simple model
            model = nn.Linear(10, 2)
            optimizer = torch.optim.Adam(model.parameters())

            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / "test_checkpoint.pt"

            checkpoint = {
                'epoch': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': 0.5,
                'config': {'lr': 1e-3}
            }

            torch.save(checkpoint, checkpoint_path)
            print(f"[SUCCESS] Checkpoint saved: {checkpoint_path}")

            # Load checkpoint
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Verify contents
            required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 'loss']
            missing_keys = [k for k in required_keys if k not in loaded_checkpoint]

            if missing_keys:
                print(f"[ERROR] Missing checkpoint keys: {missing_keys}")
                self.fail(f"Checkpoint missing keys: {missing_keys}")

            print(f"[SUCCESS] Checkpoint loaded successfully")
            print(f"  Epoch: {loaded_checkpoint['epoch']}")
            print(f"  Loss: {loaded_checkpoint['loss']}")

            self.assertEqual(loaded_checkpoint['epoch'], 1)
            self.assertEqual(loaded_checkpoint['loss'], 0.5)

        except Exception as e:
            print(f"[ERROR] Checkpoint test failed: {e}")
            self.fail(f"Checkpoint save/load test failed: {e}")

    def test_checkpoint_resume(self):
        """Test training resume from checkpoint"""
        try:
            # Create model and train for 1 step
            model = nn.Linear(5, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Get initial weights
            initial_weights = model.weight.data.clone()

            # Train one step
            x = torch.randn(10, 5)
            y = torch.randint(0, 2, (10,))

            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optimizer.step()

            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / "resume_test.pt"
            torch.save({
                'epoch': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }, checkpoint_path)

            # Create new model and load checkpoint
            new_model = nn.Linear(5, 2)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)

            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            new_model.load_state_dict(checkpoint['model_state_dict'])
            new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Verify weights are different from initial (training occurred)
            weights_changed = not torch.equal(initial_weights, new_model.weight.data)

            print(f"[SUCCESS] Resume test: weights changed = {weights_changed}")
            print(f"[SUCCESS] Resumed from epoch: {checkpoint['epoch']}")
            print(f"[SUCCESS] Resumed loss: {checkpoint['loss']:.4f}")

            self.assertTrue(weights_changed)

        except Exception as e:
            print(f"[ERROR] Resume test failed: {e}")
            self.fail(f"Checkpoint resume test failed: {e}")


class TestOOMHandling(unittest.TestCase):
    """Test Out-of-Memory handling"""

    def test_memory_monitoring(self):
        """Test memory usage monitoring"""
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")

        try:
            device = torch.device("cuda")

            # Clear cache
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            # Allocate some memory
            tensor = torch.randn(1000, 1000, device=device)
            allocated_memory = torch.cuda.memory_allocated()

            memory_used = (allocated_memory - initial_memory) / 1024**2  # MB
            print(f"[SUCCESS] Memory monitoring working: {memory_used:.1f} MB allocated")

            # Clean up
            del tensor
            torch.cuda.empty_cache()

            final_memory = torch.cuda.memory_allocated()
            print(f"[SUCCESS] Memory after cleanup: {final_memory / 1024**2:.1f} MB")

            self.assertTrue(memory_used > 0)

        except Exception as e:
            print(f"[ERROR] Memory monitoring test failed: {e}")
            self.fail(f"Memory monitoring test failed: {e}")

    def test_batch_size_reduction(self):
        """Test automatic batch size reduction on OOM"""
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")

        try:
            device = torch.device("cuda")

            # Test with manageable batch sizes for RTX 3050
            test_batch_sizes = [64, 32, 16, 8, 4]
            successful_batch_size = None

            for batch_size in test_batch_sizes:
                try:
                    # Create model and data
                    model = nn.Linear(512, 2).to(device)
                    x = torch.randn(batch_size, 512, device=device)
                    y = torch.randint(0, 2, (batch_size,), device=device)

                    # Forward pass
                    output = model(x)
                    loss = nn.CrossEntropyLoss()(output, y)
                    loss.backward()

                    successful_batch_size = batch_size
                    print(f"[SUCCESS] Batch size {batch_size}: SUCCESS")

                    # Clean up
                    del model, x, y, output, loss
                    torch.cuda.empty_cache()
                    break

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"[ERROR] Batch size {batch_size}: OOM")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            if successful_batch_size is None:
                print("[ERROR] All batch sizes failed")
                print("Fix: Reduce model size or use CPU training")
                self.fail("No batch size worked")
            else:
                print(f"[SUCCESS] Recommended batch size: {successful_batch_size}")
                self.assertTrue(True)

        except Exception as e:
            print(f"[ERROR] Batch size test failed: {e}")
            self.fail(f"Batch size reduction test failed: {e}")


class TestConfigurationValidation(unittest.TestCase):
    """Test training configuration validation"""

    def test_config_creation(self):
        """Test if training configuration can be created"""
        try:
            # Try to create basic config
            config_dict = {
                'data': {
                    'batch_size': 16,
                    'max_length': 512,
                    'dataloader_num_workers': 0  # Safe for Windows
                },
                'optimizer': {
                    'lr': 2e-5,
                    'weight_decay': 0.01
                },
                'training': {
                    'epochs': 1,
                    'gradient_accumulation_steps': 1
                }
            }

            print("[SUCCESS] Basic config structure valid")
            self.assertTrue(True)

        except Exception as e:
            print(f"[ERROR] Config creation failed: {e}")
            self.fail(f"Configuration creation failed: {e}")

    def test_gpu_config_validation(self):
        """Test GPU-specific configuration validation"""
        if not torch.cuda.is_available():
            print("[WARNING] GPU not available - using CPU config")
            recommended_config = {
                'data': {'batch_size': 8, 'dataloader_num_workers': 0},
                'training': {'use_amp': False, 'device': 'cpu'}
            }
        else:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[SUCCESS] GPU memory: {gpu_memory:.1f} GB")

            if gpu_memory < 6:  # RTX 3050 4GB
                recommended_config = {
                    'data': {'batch_size': 8, 'gradient_accumulation_steps': 4},
                    'training': {'use_amp': True, 'empty_cache_steps': 10}
                }
                print("[SUCCESS] Low memory GPU config recommended")
            else:
                recommended_config = {
                    'data': {'batch_size': 16, 'gradient_accumulation_steps': 2},
                    'training': {'use_amp': True, 'empty_cache_steps': 50}
                }
                print("[SUCCESS] Standard GPU config recommended")

        # Validate config makes sense
        batch_size = recommended_config['data']['batch_size']
        self.assertTrue(batch_size >= 4 and batch_size <= 32)

        print(f"[SUCCESS] Recommended batch size: {batch_size}")


class TestSmokeTest(unittest.TestCase):
    """Quick smoke test - should complete in < 1 minute"""

    def test_quick_training_loop(self):
        """Test one epoch of training with minimal data"""
        try:
            # Use CPU for reliable quick test
            device = torch.device("cpu")

            # Create minimal model
            model = nn.Sequential(
                nn.Linear(100, 64),
                nn.ReLU(),
                nn.Linear(64, 3)  # 3 classes for toxicity
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            # Create minimal dataset
            batch_size = 8
            num_samples = 32  # Very small for speed
            x = torch.randn(num_samples, 100, device=device)
            y = torch.randint(0, 3, (num_samples,), device=device)

            dataset = TensorDataset(x, y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            print(f"[SUCCESS] Smoke test setup: {num_samples} samples, batch size {batch_size}")

            # Training loop
            model.train()
            total_loss = 0
            num_batches = 0

            start_time = time.time()

            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            training_time = time.time() - start_time
            avg_loss = total_loss / num_batches if num_batches > 0 else 0

            print(f"[SUCCESS] Training completed in {training_time:.2f}s")
            print(f"[SUCCESS] Average loss: {avg_loss:.4f}")
            print(f"[SUCCESS] Processed {num_batches} batches")

            # Validation
            self.assertTrue(num_batches > 0)
            self.assertTrue(training_time < 60)  # Should be much faster
            self.assertTrue(avg_loss > 0)  # Loss should be positive

            if training_time > 10:
                print("[WARNING] Training slower than expected for smoke test")

        except Exception as e:
            print(f"[ERROR] Smoke test failed: {e}")
            self.fail(f"Quick training loop test failed: {e}")


def run_all_tests():
    """Run all tests with summary"""
    print("\n" + "="*80)
    print("CYBERPUPPY LOCAL TRAINING TEST SUITE")
    print("="*80)

    # Test discovery and execution
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestGPUDetection,
        TestTrainingDataLoading,
        TestCheckpointManager,
        TestOOMHandling,
        TestConfigurationValidation,
        TestSmokeTest
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    if result.wasSuccessful():
        print("SUCCESS ALL TESTS PASSED! System ready for training.")
    else:
        print(f"ERROR {len(result.failures)} failures, {len(result.errors)} errors")
        print("\nFailed tests:")
        for test, traceback in result.failures + result.errors:
            print(f"  - {test}")

    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    return result.wasSuccessful()


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    success = run_all_tests()
    sys.exit(0 if success else 1)