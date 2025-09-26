#!/usr/bin/env python3
"""
檢查 CyberPuppy 訓練環境需求
"""

import sys
import subprocess
import importlib
from pathlib import Path
import platform

def check_python_version():
    """檢查 Python 版本"""
    print("🐍 Python 版本檢查...")
    version = sys.version_info
    print(f"   當前版本: Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ 需要 Python 3.8 或更新版本")
        return False
    else:
        print("   ✅ Python 版本符合需求")
        return True

def check_package(package_name, import_name=None, version_attr='__version__'):
    """檢查套件是否安裝"""
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)
        if hasattr(module, version_attr):
            version = getattr(module, version_attr)
            print(f"   ✅ {package_name}: {version}")
        else:
            print(f"   ✅ {package_name}: 已安裝")
        return True
    except ImportError:
        print(f"   ❌ {package_name}: 未安裝")
        return False

def check_gpu():
    """檢查 GPU 支援"""
    print("\n🖥️ GPU 支援檢查...")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"   ✅ CUDA 可用，檢測到 {gpu_count} 個 GPU")

            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")

            return True
        else:
            print("   ⚠️  CUDA 不可用，將使用 CPU 訓練")
            return False
    except ImportError:
        print("   ❌ PyTorch 未安裝，無法檢查 GPU")
        return False

def check_directories():
    """檢查專案目錄結構"""
    print("\n📁 專案目錄檢查...")

    project_root = Path(__file__).parent.parent
    required_dirs = [
        "src/cyberpuppy",
        "data",
        "models",
        "logs",
        "checkpoints"
    ]

    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path}")
            if dir_path in ["models", "logs", "checkpoints"]:
                print(f"      → 將自動創建此目錄")
                full_path.mkdir(parents=True, exist_ok=True)
            else:
                all_exist = False

    return all_exist

def check_disk_space():
    """檢查磁碟空間"""
    print("\n💾 磁碟空間檢查...")

    try:
        import shutil
        project_root = Path(__file__).parent.parent
        total, used, free = shutil.disk_usage(project_root)

        free_gb = free / (1024**3)
        total_gb = total / (1024**3)

        print(f"   總空間: {total_gb:.1f}GB")
        print(f"   可用空間: {free_gb:.1f}GB")

        if free_gb < 5:
            print("   ⚠️  可用空間不足 5GB，建議清理磁碟")
            return False
        else:
            print("   ✅ 磁碟空間充足")
            return True
    except Exception as e:
        print(f"   ❌ 無法檢查磁碟空間: {e}")
        return False

def main():
    print("🔍 CyberPuppy 訓練環境需求檢查")
    print("=" * 50)

    checks = []

    # Python 版本
    checks.append(check_python_version())

    # 必要套件
    print("\n📦 必要套件檢查...")
    required_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
        ("scikit-learn", "sklearn"),
        ("colorama", "colorama"),
    ]

    package_results = []
    for package_name, import_name in required_packages:
        package_results.append(check_package(package_name, import_name))

    checks.append(all(package_results))

    # GPU 支援
    checks.append(check_gpu())

    # 目錄結構
    checks.append(check_directories())

    # 磁碟空間
    checks.append(check_disk_space())

    # 系統信息
    print(f"\n🖥️ 系統信息:")
    print(f"   作業系統: {platform.system()} {platform.release()}")
    print(f"   處理器: {platform.processor()}")
    print(f"   架構: {platform.architecture()[0]}")

    # 總結
    print("\n" + "=" * 50)
    if all(checks):
        print("✅ 所有檢查通過，可以開始訓練！")
        print("\n執行訓練：")
        print("  Windows: train_local.bat")
        print("  Python:  python scripts/train_local.py")
    else:
        print("❌ 部分檢查未通過，請先解決上述問題")
        print("\n安裝必要套件：")
        print("  pip install -r requirements.txt")
        print("  或")
        print("  pip install torch transformers numpy tqdm scikit-learn colorama")

    return all(checks)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)