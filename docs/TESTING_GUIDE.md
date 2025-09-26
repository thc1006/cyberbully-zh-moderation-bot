# Local Training Test Suite

This comprehensive test suite ensures your local training environment is ready for the CyberPuppy project.

## Quick Start

### 1. Pre-flight Check (Recommended First Step)
```bash
python scripts/test_local_setup.py
```

This will check:
- ✅ GPU detection and CUDA availability
- ✅ System resources (RAM, disk space)
- ✅ Training data availability
- ✅ Model configurations
- ✅ Python dependencies
- ✅ Training recommendations for your hardware

### 2. Comprehensive Test Suite
```bash
python tests/test_local_training.py
```

Or using pytest:
```bash
pytest tests/test_local_training.py -v
```

### 3. Run Both (Full Test Suite)
```bash
python scripts/run_training_tests.py
```

## Test Categories

### 🖥️ GPU Detection Tests (`TestGPUDetection`)
- **GPU Availability**: Detects RTX 3050 and CUDA support
- **Memory Allocation**: Tests basic GPU memory operations
- **Mixed Precision**: Validates AMP (Automatic Mixed Precision) support
- **OOM Protection**: Ensures graceful handling of memory issues

**Expected for RTX 3050 4GB:**
```
[SUCCESS] GPU detected: NVIDIA GeForce RTX 3050 Laptop GPU
[SUCCESS] CUDA available: 12.4
[SUCCESS] GPU memory: 4.0 GB
[WARNING] Low GPU memory detected
[INFO] Recommendation: Use batch_size=8, gradient_accumulation=4
```

### 📊 Training Data Tests (`TestTrainingDataLoading`)
- **Data Files**: Checks for processed training datasets
- **Loading Performance**: Measures data loading speed
- **Format Validation**: Ensures correct label structure
- **Sample Counting**: Verifies data consistency

**Expected Output:**
```
[SUCCESS] Found: data/processed/training_dataset/train.json (25,659 samples)
[SUCCESS] Found: data/processed/training_dataset/dev.json (6,430 samples)
[SUCCESS] Total training samples: 37,409
[INFO] Large dataset - training ~1-3 hours
```

### 💾 Checkpoint Tests (`TestCheckpointManager`)
- **Save/Load**: Tests checkpoint persistence
- **Resume Training**: Validates training continuation
- **State Recovery**: Ensures model/optimizer state preservation

### 🧠 Memory Management Tests (`TestOOMHandling`)
- **Memory Monitoring**: Tracks GPU memory usage
- **Batch Size Reduction**: Tests automatic OOM recovery
- **Memory Cleanup**: Validates cache clearing

**RTX 3050 Optimizations:**
```
[SUCCESS] Recommended batch size: 8
[SUCCESS] Memory monitoring working: 3.8 MB allocated
[SUCCESS] Memory after cleanup: 0.0 MB
```

### ⚙️ Configuration Tests (`TestConfigurationValidation`)
- **Config Creation**: Tests training configuration setup
- **GPU-Specific Settings**: Validates hardware-appropriate settings
- **Parameter Validation**: Ensures sensible default values

### 🚀 Smoke Test (`TestSmokeTest`)
- **Quick Training Loop**: 1 epoch, 32 samples, <1 minute
- **Basic Functionality**: Validates end-to-end training
- **Performance Baseline**: Ensures reasonable speed

**Expected Performance:**
```
[SUCCESS] Smoke test setup: 32 samples, batch size 8
[SUCCESS] Training completed in 0.11s
[SUCCESS] Average loss: 1.1291
[SUCCESS] Processed 4 batches
```

## Hardware-Specific Recommendations

### RTX 3050 4GB (Your Current Setup)
- **Batch Size**: 8
- **Gradient Accumulation**: 4 steps (effective batch size: 32)
- **Mixed Precision**: Enabled (AMP)
- **Memory Cleanup**: Every 10 steps
- **Workers**: 4 dataloader workers

### RTX 3060/4060 8GB
- **Batch Size**: 16
- **Gradient Accumulation**: 2 steps
- **Mixed Precision**: Enabled
- **Memory Cleanup**: Every 50 steps

### High-End GPUs (12GB+)
- **Batch Size**: 32
- **Gradient Accumulation**: 1 step
- **Mixed Precision**: Enabled
- **Memory Cleanup**: Every 100 steps

### CPU Fallback
- **Batch Size**: 4
- **Gradient Accumulation**: 8 steps
- **Mixed Precision**: Disabled
- **Warning**: Significantly slower training

## Common Issues & Fixes

### GPU Not Detected
```
[ERROR] GPU not detected
Fix: Install CUDA-enabled PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory
```
[ERROR] GPU OOM on basic allocation
Fix: Close other GPU applications or reduce batch size
```

### Missing Training Data
```
[ERROR] Missing: data/processed/training_dataset/train.json
Fix: Run data preparation scripts to create missing files
     python scripts/create_unified_training_data.py
```

### Slow Data Loading
```
[WARNING] Data loading slower than expected
Fix: Increase num_workers or optimize data preprocessing
```

### Low Available RAM
```
[WARNING] Low available RAM
Fix: Close other applications before training
```

## Performance Benchmarks

### Expected Test Times (RTX 3050)
- **Pre-flight Check**: ~4 seconds
- **GPU Tests**: ~3 seconds
- **Data Loading Tests**: ~5 seconds
- **Checkpoint Tests**: ~2 seconds
- **Memory Tests**: ~8 seconds
- **Config Tests**: ~1 second
- **Smoke Test**: ~18 seconds
- **Total Runtime**: ~45 seconds

### Training Estimates
- **Small Dataset (<1K samples)**: 5-10 minutes
- **Medium Dataset (1K-10K)**: 30-60 minutes
- **Large Dataset (10K+)**: 1-3 hours
- **Full Dataset (37K samples)**: 2-4 hours

## Integration with Training Pipeline

After all tests pass, start training with:

```bash
# Use recommended configuration from pre-flight check
python scripts/training/train.py --config recommended

# Or manually specify settings
python scripts/training/train.py \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --use_amp \
  --empty_cache_steps 10
```

## Troubleshooting

### Tests Fail Due to Import Errors
Some advanced modules may not be available. Tests will skip gracefully:
```
[WARNING] Import error: No module named 'cyberpuppy.models.multitask'
Some tests may be skipped
```

### Coverage Warnings
Coverage requirements are for production code, not tests:
```
FAIL Required test coverage of 90.0% not reached. Total coverage: 6.76%
```
This is expected and doesn't affect test functionality.

### Unicode Issues on Windows
All special characters have been replaced with ASCII equivalents for Windows compatibility.

## Files Created

- `tests/test_local_training.py` - Comprehensive test suite
- `scripts/test_local_setup.py` - Pre-flight check script
- `scripts/run_training_tests.py` - Combined test runner
- `system_info.json` - System configuration export

## Next Steps

Once all tests pass:

1. ✅ Your system is ready for training
2. 🚀 Start with a small experiment to validate the pipeline
3. 📊 Monitor GPU memory usage during training
4. 🔧 Adjust batch size based on actual memory consumption
5. 📈 Scale up to full dataset training

The test suite ensures you catch configuration issues before they waste hours of training time!