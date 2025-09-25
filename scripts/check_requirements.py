#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
檢查專案必要檔案和資源
"""
import os
import sys
from pathlib import Path

# 設定 UTF-8 編碼
if sys.platform == "win32":
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def check_requirements():
    """檢查所有必要的檔案和目錄"""

    required_items = {
        "Models": {
            "models/macbert_base_demo/best.ckpt": "Model checkpoint (397MB)",
            "models/toxicity_only_demo/best.ckpt": "Model checkpoint (397MB)",
        },
        "Data": {
            "data/processed/unified/train_unified.json": "Training data",
            "data/processed/unified/test_unified.json": "Test data",
            "data/processed/unified/dev_unified.json": "Dev data",
        },
        "Core Files": {
            "api/app.py": "API application",
            "bot/line_bot.py": "LINE bot application",
            "src/cyberpuppy/models/baselines.py": "Model definitions",
        }
    }

    print("="*60)
    print("CyberPuppy Requirements Check")
    print("="*60)

    missing = []

    for category, items in required_items.items():
        print(f"\n{category}:")
        for path, description in items.items():
            if Path(path).exists():
                size = Path(path).stat().st_size / (1024*1024)  # MB
                print(f"  [OK] {path} ({size:.1f}MB)")
            else:
                print(f"  [MISSING] {path} - {description}")
                missing.append(path)

    print("\n" + "="*60)

    if missing:
        print(f"[WARNING] Missing {len(missing)} required files:")
        for item in missing[:5]:  # Show first 5
            print(f"  - {item}")
        if len(missing) > 5:
            print(f"  ... and {len(missing)-5} more")
        print("\n[ACTION] Please run: python scripts/download_datasets.py")
        return False
    else:
        print("[SUCCESS] All requirements satisfied!")
        return True

if __name__ == "__main__":
    check_requirements()