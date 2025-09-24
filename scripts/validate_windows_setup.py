#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows setup validation script
Validates that all Windows-specific fixes are working correctly
"""

import os
import sys
import platform
import subprocess
import importlib
from pathlib import Path
from typing import List, Tuple, Optional

# Windows encoding fix - apply immediately
if platform.system() == "Windows":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        # Python < 3.7 doesn't have reconfigure
        pass


class ValidationResult:
    """Represents the result of a validation check"""

    def __init__(self, name: str, passed: bool, message: str, suggestion: Optional[str] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.suggestion = suggestion

    def __str__(self):
        # Use safe characters for Windows console
        try:
            status = "✅" if self.passed else "❌"
            suggestion_icon = "💡"
        except UnicodeEncodeError:
            status = "[PASS]" if self.passed else "[FAIL]"
            suggestion_icon = "TIP:"

        result = f"{status} {self.name}: {self.message}"
        if not self.passed and self.suggestion:
            result += f"\n   {suggestion_icon} Suggestion: {self.suggestion}"
        return result


class WindowsSetupValidator:
    """Validates Windows-specific setup"""

    def __init__(self):
        self.results: List[ValidationResult] = []

    def run_all_checks(self) -> List[ValidationResult]:
        """Run all validation checks"""
        # Use safe characters for Windows console
        try:
            print("🐶 CyberPuppy Windows Setup Validation")
        except UnicodeEncodeError:
            print("CyberPuppy Windows Setup Validation")
        print("=" * 50)

        checks = [
            self.check_platform,
            self.check_python_version,
            self.check_encoding_setup,
            self.check_console_encoding,
            self.check_numpy_installation,
            self.check_pytorch_installation,
            self.check_chinese_packages,
            self.check_chinese_text_processing,
            self.check_file_encoding,
            self.check_test_imports,
            self.check_environment_variables,
            self.check_pip_configuration,
        ]

        for check in checks:
            try:
                result = check()
                self.results.append(result)
                print(result)
            except Exception as e:
                error_result = ValidationResult(
                    check.__name__.replace('check_', '').replace('_', ' ').title(),
                    False,
                    f"Check failed with error: {e}",
                    "Review the error message and ensure all dependencies are installed"
                )
                self.results.append(error_result)
                print(error_result)

        return self.results

    def check_platform(self) -> ValidationResult:
        """Check if running on Windows"""
        if platform.system() == "Windows":
            version = platform.version()
            release = platform.release()
            return ValidationResult(
                "Platform Check",
                True,
                f"Running on Windows {release} (version {version})"
            )
        else:
            return ValidationResult(
                "Platform Check",
                False,
                f"Not running on Windows (detected: {platform.system()})",
                "This validation is specifically for Windows systems"
            )

    def check_python_version(self) -> ValidationResult:
        """Check Python version compatibility"""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 9:
            return ValidationResult(
                "Python Version",
                True,
                f"Python {version.major}.{version.minor}.{version.micro} is supported"
            )
        else:
            return ValidationResult(
                "Python Version",
                False,
                f"Python {version.major}.{version.minor}.{version.micro} is not supported",
                "Upgrade to Python 3.9 or higher"
            )

    def check_encoding_setup(self) -> ValidationResult:
        """Check encoding configuration"""
        issues = []

        # Check environment variables
        pythonioencoding = os.environ.get('PYTHONIOENCODING')
        if pythonioencoding != 'utf-8':
            issues.append("PYTHONIOENCODING not set to utf-8")

        # Check stdout/stderr encoding
        stdout_encoding = getattr(sys.stdout, 'encoding', None)
        if stdout_encoding and 'utf' not in stdout_encoding.lower():
            issues.append(f"stdout encoding is {stdout_encoding}, not UTF-8")

        # Test Unicode handling
        try:
            test_str = "測試中文編碼"  # Remove emoji that might cause issues
            test_str.encode('utf-8').decode('utf-8')
        except UnicodeError:
            issues.append("Unicode encoding/decoding failed")

        if issues:
            return ValidationResult(
                "Encoding Setup",
                False,
                "; ".join(issues),
                "Run 'chcp 65001' and set PYTHONIOENCODING=utf-8"
            )
        else:
            return ValidationResult(
                "Encoding Setup",
                True,
                f"Encoding configured correctly (stdout: {stdout_encoding})"
            )

    def check_console_encoding(self) -> ValidationResult:
        """Check Windows console encoding"""
        try:
            # Try to get current codepage
            result = subprocess.run(['chcp'], shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                output = result.stdout.strip()
                if '65001' in output:  # UTF-8 codepage
                    return ValidationResult(
                        "Console Encoding",
                        True,
                        "Console set to UTF-8 (codepage 65001)"
                    )
                else:
                    return ValidationResult(
                        "Console Encoding",
                        False,
                        f"Console not set to UTF-8: {output}",
                        "Run 'chcp 65001' to set UTF-8 encoding"
                    )
            else:
                return ValidationResult(
                    "Console Encoding",
                    False,
                    f"Failed to check codepage: {result.stderr}",
                    "Manually run 'chcp 65001' in command prompt"
                )

        except Exception as e:
            return ValidationResult(
                "Console Encoding",
                False,
                f"Could not check console encoding: {e}",
                "Try running 'chcp 65001' manually"
            )

    def check_numpy_installation(self) -> ValidationResult:
        """Check NumPy installation (common Windows issue)"""
        try:
            import numpy as np

            version = np.__version__
            # Check if it's a pre-compiled version (should load quickly)
            start_time = __import__('time').time()
            _ = np.array([1, 2, 3])
            load_time = __import__('time').time() - start_time

            if load_time < 1.0:  # Should be very fast for pre-compiled
                return ValidationResult(
                    "NumPy Installation",
                    True,
                    f"NumPy {version} installed correctly (pre-compiled)"
                )
            else:
                return ValidationResult(
                    "NumPy Installation",
                    True,
                    f"NumPy {version} installed but may be compiled from source",
                    "Consider reinstalling with --only-binary=numpy"
                )

        except ImportError:
            return ValidationResult(
                "NumPy Installation",
                False,
                "NumPy not installed",
                "Install with: pip install --only-binary=numpy numpy"
            )
        except Exception as e:
            return ValidationResult(
                "NumPy Installation",
                False,
                f"NumPy import failed: {e}",
                "Reinstall NumPy with pre-compiled wheels"
            )

    def check_pytorch_installation(self) -> ValidationResult:
        """Check PyTorch installation"""
        try:
            import torch

            version = torch.__version__
            # Check if CUDA is available (optional on Windows)
            cuda_available = torch.cuda.is_available()

            # Test basic tensor operations
            tensor = torch.tensor([1.0, 2.0, 3.0])
            result = tensor.sum()

            cuda_info = f" (CUDA: {'available' if cuda_available else 'not available'})"
            return ValidationResult(
                "PyTorch Installation",
                True,
                f"PyTorch {version} working correctly{cuda_info}"
            )

        except ImportError:
            return ValidationResult(
                "PyTorch Installation",
                False,
                "PyTorch not installed",
                "Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )
        except Exception as e:
            return ValidationResult(
                "PyTorch Installation",
                False,
                f"PyTorch test failed: {e}",
                "Reinstall PyTorch from official website"
            )

    def check_chinese_packages(self) -> ValidationResult:
        """Check Chinese NLP packages"""
        packages = {
            'jieba': 'Chinese word segmentation',
            'opencc': 'Traditional/Simplified conversion'
        }

        results = []
        for package, description in packages.items():
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                results.append(f"{package} {version} ({description})")
            except ImportError:
                return ValidationResult(
                    "Chinese Packages",
                    False,
                    f"Missing package: {package} ({description})",
                    f"Install with: pip install {package if package != 'opencc' else 'opencc-python-reimplemented'}"
                )

        return ValidationResult(
            "Chinese Packages",
            True,
            "; ".join(results)
        )

    def check_chinese_text_processing(self) -> ValidationResult:
        """Check Chinese text processing functionality"""
        try:
            import jieba
            from opencc import OpenCC

            # Test jieba
            test_text = "这是一个中文测试句子，用于验证分词功能。"
            words = list(jieba.cut(test_text))

            if len(words) < 5:  # Should produce multiple words
                return ValidationResult(
                    "Chinese Text Processing",
                    False,
                    f"Jieba segmentation produced too few words: {len(words)}",
                    "Check jieba installation and dictionary files"
                )

            # Test OpenCC
            cc = OpenCC('s2t')  # Simplified to Traditional
            converted = cc.convert(test_text)

            if converted == test_text:
                return ValidationResult(
                    "Chinese Text Processing",
                    False,
                    "OpenCC conversion did not change text",
                    "Check OpenCC configuration and data files"
                )

            return ValidationResult(
                "Chinese Text Processing",
                True,
                f"Jieba: {len(words)} words, OpenCC: conversion successful"
            )

        except Exception as e:
            return ValidationResult(
                "Chinese Text Processing",
                False,
                f"Text processing failed: {e}",
                "Reinstall Chinese NLP packages"
            )

    def check_file_encoding(self) -> ValidationResult:
        """Check file I/O encoding"""
        try:
            # Test writing and reading Chinese text
            test_content = "中文檔案測試\n網路霸凌檢測\n繁簡轉換功能"

            # Create temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                           delete=False, suffix='.txt') as f:
                f.write(test_content)
                temp_path = f.name

            # Read it back
            with open(temp_path, 'r', encoding='utf-8') as f:
                read_content = f.read()

            # Clean up
            os.unlink(temp_path)

            if read_content == test_content:
                return ValidationResult(
                    "File Encoding",
                    True,
                    "UTF-8 file I/O working correctly"
                )
            else:
                return ValidationResult(
                    "File Encoding",
                    False,
                    "File content mismatch after write/read",
                    "Check file system encoding support"
                )

        except Exception as e:
            return ValidationResult(
                "File Encoding",
                False,
                f"File I/O test failed: {e}",
                "Check file system permissions and encoding"
            )

    def check_test_imports(self) -> ValidationResult:
        """Check critical package imports"""
        critical_imports = [
            ('pandas', 'Data processing'),
            ('sklearn', 'Machine learning'),
            ('transformers', 'NLP models'),
            ('datasets', 'HuggingFace datasets'),
            ('fastapi', 'API framework'),
            ('pytest', 'Testing framework'),
        ]

        failed_imports = []
        success_count = 0

        for package, description in critical_imports:
            try:
                importlib.import_module(package)
                success_count += 1
            except ImportError:
                failed_imports.append(f"{package} ({description})")

        if failed_imports:
            return ValidationResult(
                "Critical Imports",
                False,
                f"Failed imports: {', '.join(failed_imports)}",
                "Install missing packages from requirements.txt"
            )
        else:
            return ValidationResult(
                "Critical Imports",
                True,
                f"All {success_count} critical packages imported successfully"
            )

    def check_environment_variables(self) -> ValidationResult:
        """Check required environment variables"""
        required_vars = {
            'PYTHONIOENCODING': 'utf-8',
        }

        recommended_vars = {
            'PYTHONUTF8': '1',
        }

        issues = []

        for var, expected in required_vars.items():
            actual = os.environ.get(var)
            if actual != expected:
                issues.append(f"{var} should be {expected} (got: {actual})")

        warnings = []
        for var, expected in recommended_vars.items():
            actual = os.environ.get(var)
            if actual != expected:
                warnings.append(f"{var} recommended to be {expected}")

        if issues:
            return ValidationResult(
                "Environment Variables",
                False,
                "; ".join(issues),
                "Set environment variables for proper encoding"
            )
        elif warnings:
            return ValidationResult(
                "Environment Variables",
                True,
                f"Required vars OK. Recommendations: {'; '.join(warnings)}"
            )
        else:
            return ValidationResult(
                "Environment Variables",
                True,
                "All encoding environment variables properly set"
            )

    def check_pip_configuration(self) -> ValidationResult:
        """Check pip configuration for Windows"""
        try:
            # Check pip version
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'],
                                  capture_output=True, text=True)

            if result.returncode == 0:
                pip_version = result.stdout.strip()

                # Check for pip-tools
                try:
                    import pip_tools
                    pip_tools_version = pip_tools.__version__
                    tools_info = f", pip-tools {pip_tools_version}"
                except ImportError:
                    tools_info = " (pip-tools not installed)"

                return ValidationResult(
                    "Pip Configuration",
                    True,
                    f"Pip working: {pip_version}{tools_info}"
                )
            else:
                return ValidationResult(
                    "Pip Configuration",
                    False,
                    f"Pip check failed: {result.stderr}",
                    "Reinstall or upgrade pip"
                )

        except Exception as e:
            return ValidationResult(
                "Pip Configuration",
                False,
                f"Pip check error: {e}",
                "Check pip installation"
            )

    def generate_report(self) -> str:
        """Generate a summary report"""
        if not self.results:
            return "No validation results available"

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        report = f"\n{'='*50}\n"
        try:
            report += f"🐶 CyberPuppy Windows Setup Validation Report\n"
            summary_icon = "📊"
            success_icon = "🎉"
            warning_icon = "⚠️"
            tip_icon = "💡"
        except UnicodeEncodeError:
            report += f"CyberPuppy Windows Setup Validation Report\n"
            summary_icon = "SUMMARY:"
            success_icon = "SUCCESS:"
            warning_icon = "WARNING:"
            tip_icon = "TIP:"

        report += f"{'='*50}\n\n"
        report += f"{summary_icon} Summary: {passed}/{total} checks passed\n\n"

        if passed == total:
            report += f"{success_icon} All checks passed! Your Windows setup is ready.\n\n"
            report += "Next steps:\n"
            report += "1. Run 'python test_opencc.py' to test Chinese processing\n"
            report += "2. Run 'python -m pytest tests/ -v' to run the test suite\n"
            report += "3. Follow the main README.md for project setup\n"
        else:
            report += f"{warning_icon} Some issues were found. Please address them:\n\n"

            failed_results = [r for r in self.results if not r.passed]
            for i, result in enumerate(failed_results, 1):
                report += f"{i}. {result.name}: {result.message}\n"
                if result.suggestion:
                    report += f"   {tip_icon} {result.suggestion}\n"
                report += "\n"

        return report


def main():
    """Main validation routine"""
    validator = WindowsSetupValidator()
    results = validator.run_all_checks()

    # Generate and display report
    report = validator.generate_report()
    print(report)

    # Exit with appropriate code
    failed_count = sum(1 for r in results if not r.passed)
    sys.exit(failed_count)


if __name__ == '__main__':
    main()