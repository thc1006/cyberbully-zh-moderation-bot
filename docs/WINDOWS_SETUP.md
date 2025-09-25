# Windows Setup Guide for CyberPuppy

This guide addresses common Windows-specific issues when setting up the CyberPuppy Chinese cyberbullying detection system.

## Quick Start (Recommended)

```bash
# 1. Use the Windows-specific setup script
python scripts/windows_setup.py

# 2. Test the installation
python test_opencc.py

# 3. Run tests to verify everything works
python -m pytest tests/ -v
```

## Common Windows Issues & Solutions

### 1. NumPy Compilation Error (GCC >= 8.4 Required)

**Problem**: NumPy fails to compile due to GCC version requirements.

**Solution**: Use pre-compiled wheels:

```bash
# Install NumPy with pre-compiled wheels
python -m pip install --only-binary=numpy numpy>=1.24.4,<2.0.0

# Or use our Windows setup script
python scripts/windows_setup.py
```

### 2. Chinese Text Encoding Issues (CP950 Codec)

**Problem**: Console shows garbled Chinese text or encoding errors.

**Solutions**:

**Method 1: Set Console to UTF-8**
```bash
# Set Windows console to UTF-8
chcp 65001

# Set environment variable
set PYTHONIOENCODING=utf-8
```

**Method 2: PowerShell**
```powershell
# In PowerShell, set UTF-8 encoding
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new()
[Console]::InputEncoding = [Text.UTF8Encoding]::new()
```

**Method 3: Use Windows Terminal**
- Install Windows Terminal from Microsoft Store
- Automatically handles UTF-8 encoding

### 3. PyTorch Installation Issues

**Problem**: PyTorch CUDA compilation fails on Windows.

**Solution**: Install CPU version explicitly:

```bash
# Install PyTorch CPU version (avoids CUDA compilation)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. OpenCC Compilation Issues

**Problem**: OpenCC C++ compilation fails on Windows.

**Solution**: Use the reimplemented Python version:

```bash
# Use pure Python implementation (already in requirements.txt)
python -m pip install opencc-python-reimplemented>=0.1.7
```

## Manual Installation Steps

If the automated setup script fails, follow these manual steps:

### Step 1: Prepare Environment

```bash
# Upgrade pip and install build tools
python -m pip install --upgrade pip
python -m pip install wheel setuptools>=68.0.0 pip-tools

# Set encoding environment variables
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
```

### Step 2: Install Core Dependencies (Pre-compiled)

```bash
# Install NumPy first (many packages depend on it)
python -m pip install --only-binary=numpy "numpy>=1.24.4,<2.0.0"

# Install other commonly problematic packages
python -m pip install --only-binary=scipy scipy
python -m pip install --only-binary=pandas "pandas>=2.1.0"
python -m pip install --only-binary=scikit-learn "scikit-learn>=1.3.0"
python -m pip install --only-binary=matplotlib matplotlib
```

### Step 3: Install PyTorch

```bash
# CPU version (recommended for most users)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or GPU version if you have CUDA installed
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install Chinese NLP Packages

```bash
# Chinese text processing
python -m pip install jieba>=0.42.1
python -m pip install opencc-python-reimplemented>=0.1.7
```

### Step 5: Install Remaining Requirements

```bash
# Install with constraints for compatibility
python -m pip install --constraint constraints.txt -r requirements.txt --prefer-binary
```

## Testing Your Installation

### 1. Test Chinese Text Processing

```bash
python test_opencc.py
```

Expected output should show:
- ✅ Chinese character display
- ✅ Jieba segmentation working
- ✅ OpenCC traditional/simplified conversion

### 2. Test Core Imports

```python
# Create test_imports.py
import numpy as np
import pandas as pd
import torch
import jieba
from opencc import OpenCC
print("✅ All imports successful!")
```

### 3. Run Full Test Suite

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific tests
python -m pytest tests/test_imports.py -v
```

## Environment-Specific Configurations

### Python 3.9-3.11 (Windows)
```toml
# In pyproject.toml
numpy = ">=1.24.4,<2.0.0"
torch = ">=2.1.0,<3.0.0"
```

### Python 3.12+ (Windows)
```toml
# In pyproject.toml
numpy = ">=1.26.2,<2.0.0"
torch = ">=2.1.0,<3.0.0"
```

## Troubleshooting

### Issue: "Microsoft Visual C++ 14.0 is required"

**Solutions**:
1. Install pre-compiled wheels: `--only-binary=:all:`
2. Install Visual Studio Build Tools
3. Use conda instead of pip for problematic packages

### Issue: "Failed building wheel for [package]"

**Solutions**:
1. Use our Windows setup script
2. Install pre-compiled wheel explicitly
3. Use alternative package (e.g., `opencc-python-reimplemented` instead of `opencc`)

### Issue: Chinese text shows as "??" or boxes

**Solutions**:
1. Set console encoding: `chcp 65001`
2. Use Windows Terminal instead of Command Prompt
3. Set environment variable: `set PYTHONIOENCODING=utf-8`

### Issue: Tests fail with encoding errors

**Solutions**:
```bash
# Set UTF-8 environment for tests
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
python -m pytest tests/ -v
```

## Performance Optimization for Windows

### 1. Use Windows Terminal
- Better Unicode support
- Faster rendering
- Better color support

### 2. Enable UTF-8 System-wide
1. Open Region Settings
2. Administrative → Change system locale
3. Check "Beta: Use Unicode UTF-8 for worldwide language support"
4. Restart computer

### 3. Use SSD for Virtual Environment
- Place venv on SSD for faster package loading
- Use `--system-site-packages` if appropriate

## Automated Setup Script

The `scripts/windows_setup.py` script automatically handles:

- ✅ Encoding configuration
- ✅ Pre-compiled package installation
- ✅ PyTorch CPU installation
- ✅ Chinese NLP packages
- ✅ Import verification
- ✅ Chinese text processing test

```bash
# Run the automated setup
python scripts/windows_setup.py
```

## Development Environment (Windows)

### Recommended Setup
1. **Windows Terminal** + **PowerShell Core**
2. **Visual Studio Code** with Python extension
3. **Git for Windows** with UTF-8 support
4. **Python 3.9+** from python.org (not Microsoft Store)

### VS Code Configuration
```json
// In settings.json
{
    "python.defaultInterpreterPath": "./venv/Scripts/python.exe",
    "terminal.integrated.env.windows": {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1"
    },
    "files.encoding": "utf8"
}
```

## Support

If you encounter issues not covered here:

1. Check the error message for encoding/compilation keywords
2. Try the automated setup script: `python scripts/windows_setup.py`
3. Use pre-compiled wheels: `pip install --only-binary=:all:`
4. Consider using WSL2 for a Linux-like environment

## Next Steps

After successful installation:

1. **Download datasets**: `python scripts/download_datasets.py`
2. **Process data**: `python scripts/clean_normalize.py`
3. **Train model**: `python train.py --config configs/training_config.yaml`
4. **Start API**: `python api/app.py`
5. **Run tests**: `python -m pytest tests/ -v`

---

**Note**: This guide assumes you're using standard Windows Command Prompt or PowerShell. For WSL2, follow the standard Linux installation instructions.