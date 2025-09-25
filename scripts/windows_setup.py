#!/usr/bin/env python3
"""
Windows-specific setup script for CyberPuppy
Handles encoding issues and dependency installation for Windows environments
"""

import os
import sys
import subprocess
import locale
import platform
from pathlib import Path


def set_windows_encoding():
    """Configure Windows encoding to handle Chinese text properly"""
    if platform.system() == "Windows":
        # Set console output encoding to UTF-8
        os.environ["PYTHONIOENCODING"] = "utf-8"

        # Try to set console codepage to UTF-8
        try:
            subprocess.run(
                ["chcp", "65001"],
                shell=True,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

        # Set locale encoding
        try:
            locale.setlocale(locale.LC_ALL, "C.UTF-8")
        except locale.Error:
            try:
                locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
            except locale.Error:
                # Fallback to system default
                locale.setlocale(locale.LC_ALL, "")


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Error: Python 3.9 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")


def check_pip_version():
    """Ensure pip is up to date"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"✅ pip version: {result.stdout.strip()}")

        # Upgrade pip
        print("🔄 Upgrading pip...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error checking pip: {e}")
        sys.exit(1)


def install_build_tools():
    """Install essential build tools for Windows"""
    print("🔧 Installing build tools...")
    build_tools = ["wheel", "setuptools>=68.0.0", "pip-tools>=7.3.0"]

    for tool in build_tools:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", tool], check=True
            )
            print(f"✅ Installed: {tool}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Failed to install {tool}: {e}")


def install_precompiled_packages():
    """Install packages that commonly have compilation issues on Windows"""
    print("📦 Installing pre-compiled packages...")

    # Install numpy first (many packages depend on it)
    numpy_versions = [
        'numpy>=1.24.4,<2.0.0; python_version < "3.12"',
        'numpy>=1.26.2,<2.0.0; python_version >= "3.12"',
    ]

    for numpy_spec in numpy_versions:
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--only-binary=numpy",
                    numpy_spec,
                ],
                check=True,
            )
            print(f"✅ Installed NumPy (pre-compiled)")
            break
        except subprocess.CalledProcessError:
            continue

    # Install other commonly problematic packages
    precompiled_packages = [
        "--only-binary=scipy scipy",
        "--only-binary=pandas pandas>=2.1.0",
        "--only-binary=scikit-learn scikit-learn>=1.3.0",
        "--only-binary=matplotlib matplotlib>=3.8.0",
        "--only-binary=Pillow Pillow",
    ]

    for pkg_spec in precompiled_packages:
        try:
            args = [sys.executable, "-m", "pip", "install"] + pkg_spec.split()
            subprocess.run(args, check=True)
            print(f"✅ Installed: {pkg_spec.split()[-1]} (pre-compiled)")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Failed to install {pkg_spec}: {e}")


def install_pytorch_cpu():
    """Install PyTorch CPU version for Windows to avoid CUDA compilation issues"""
    print("🔥 Installing PyTorch (CPU version)...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            check=True,
        )
        print("✅ Installed PyTorch CPU version")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Failed to install PyTorch: {e}")
        print("You may need to install PyTorch manually from https://pytorch.org/")


def install_chinese_nlp_packages():
    """Install Chinese NLP packages with Windows compatibility"""
    print("🀄 Installing Chinese NLP packages...")

    chinese_packages = [
        "jieba>=0.42.1",
        "opencc-python-reimplemented>=0.1.7",  # Avoid C++ compilation
    ]

    for pkg in chinese_packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)
            print(f"✅ Installed: {pkg}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Failed to install {pkg}: {e}")


def install_requirements():
    """Install main requirements with Windows-specific handling"""
    print("📋 Installing requirements...")

    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    if not requirements_file.exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return

    try:
        # Use constraints file for Windows compatibility
        constraints_file = Path(__file__).parent.parent / "constraints.txt"
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--constraint",
            str(constraints_file),
            "--requirement",
            str(requirements_file),
            "--prefer-binary",  # Prefer wheels over source
            "--no-compile",  # Skip compilation step
        ]

        subprocess.run(cmd, check=True)
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        print("Trying fallback installation...")

        # Fallback: install without constraints
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--requirement",
                    str(requirements_file),
                    "--prefer-binary",
                ],
                check=True,
            )
            print("✅ Requirements installed (fallback)")
        except subprocess.CalledProcessError as e2:
            print(f"❌ Fallback installation also failed: {e2}")


def test_imports():
    """Test critical imports to ensure installation worked"""
    print("🧪 Testing critical imports...")

    test_imports = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("torch", "PyTorch"),
        ("sklearn", "scikit-learn"),
        ("jieba", "Jieba"),
        ("opencc", "OpenCC"),
    ]

    failed_imports = []

    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} import successful")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            failed_imports.append(name)

    if failed_imports:
        print(f"\n⚠️  Warning: Some imports failed: {', '.join(failed_imports)}")
        print("You may need to install these packages manually.")
    else:
        print("\n🎉 All critical imports successful!")


def test_chinese_text_processing():
    """Test Chinese text processing with proper encoding"""
    print("🀄 Testing Chinese text processing...")

    try:
        import jieba
        from opencc import OpenCC

        # Test jieba segmentation
        test_text = "这是一个中文测试句子"
        words = list(jieba.cut(test_text))
        print(f"✅ Jieba segmentation: {' / '.join(words)}")

        # Test OpenCC conversion
        cc = OpenCC("s2t")  # Simplified to Traditional
        converted = cc.convert(test_text)
        print(f"✅ OpenCC conversion: {test_text} → {converted}")

        print("🎉 Chinese text processing works correctly!")

    except Exception as e:
        print(f"❌ Chinese text processing test failed: {e}")


def main():
    """Main setup routine for Windows"""
    print("🐶 CyberPuppy Windows Setup")
    print("=" * 50)

    # Set up encoding first
    set_windows_encoding()

    # Check prerequisites
    check_python_version()
    check_pip_version()

    # Install build tools
    install_build_tools()

    # Install pre-compiled packages
    install_precompiled_packages()

    # Install PyTorch
    install_pytorch_cpu()

    # Install Chinese NLP packages
    install_chinese_nlp_packages()

    # Install main requirements
    install_requirements()

    # Test installation
    test_imports()
    test_chinese_text_processing()

    print("\n" + "=" * 50)
    print("🎉 Windows setup completed!")
    print("\nNext steps:")
    print("1. Run 'python test_opencc.py' to test Chinese text processing")
    print("2. Run 'python -m pytest tests/ -v' to run tests")
    print("3. Follow the main README.md for further setup instructions")


if __name__ == "__main__":
    main()
