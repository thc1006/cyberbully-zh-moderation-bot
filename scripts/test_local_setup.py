#!/usr/bin/env python3
"""
Pre-flight check script for local training setup
Quick verification that everything is ready before training starts
"""

import os
import sys
import json
import time
import psutil
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_status(message: str, status: str = "info"):
    """Print colored status message"""
    if status == "success":
        print(f"{Colors.GREEN}[SUCCESS] {message}{Colors.END}")
    elif status == "error":
        print(f"{Colors.RED}[ERROR] {message}{Colors.END}")
    elif status == "warning":
        print(f"{Colors.YELLOW}[WARNING] {message}{Colors.END}")
    elif status == "info":
        print(f"{Colors.BLUE}[INFO] {message}{Colors.END}")
    else:
        print(f"  {message}")


def print_header(title: str):
    """Print section header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{title:^60}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")


def check_gpu() -> Dict:
    """Check GPU availability and specifications"""
    print_header("GPU Configuration Check")

    gpu_info = {
        "available": False,
        "name": None,
        "memory_gb": 0,
        "cuda_version": None,
        "status": "error"
    }

    try:
        gpu_info["available"] = torch.cuda.is_available()

        if gpu_info["available"]:
            gpu_info["name"] = torch.cuda.get_device_name(0)
            gpu_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_info["cuda_version"] = torch.version.cuda

            print_status(f"GPU detected: {gpu_info['name']}", "success")
            print_status(f"CUDA available: {torch.version.cuda}", "success")
            print_status(f"GPU memory: {gpu_info['memory_gb']:.1f} GB", "success")

            # Memory warning for RTX 3050
            if gpu_info["memory_gb"] < 6:
                print_status(f"Low GPU memory detected", "warning")
                print_status(f"Recommendation: Use batch_size=8, gradient_accumulation=4", "info")

            # Test GPU computation
            try:
                device = torch.device("cuda")
                test_tensor = torch.randn(100, 100, device=device)
                result = torch.matmul(test_tensor, test_tensor)
                del test_tensor, result
                torch.cuda.empty_cache()
                print_status("GPU computation test: PASSED", "success")
                gpu_info["status"] = "success"
            except Exception as e:
                print_status(f"GPU computation test: FAILED - {e}", "error")

        else:
            print_status("GPU not detected", "error")
            print_status("Install CUDA PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121", "info")

    except Exception as e:
        print_status(f"GPU check failed: {e}", "error")

    return gpu_info


def check_system_resources() -> Dict:
    """Check system resources (RAM, disk space)"""
    print_header("System Resources Check")

    resources = {
        "ram_gb": 0,
        "ram_available_gb": 0,
        "disk_space_gb": 0,
        "cpu_count": 0,
        "status": "success"
    }

    try:
        # RAM check
        memory = psutil.virtual_memory()
        resources["ram_gb"] = memory.total / 1024**3
        resources["ram_available_gb"] = memory.available / 1024**3

        print_status(f"Total RAM: {resources['ram_gb']:.1f} GB", "success")
        print_status(f"Available RAM: {resources['ram_available_gb']:.1f} GB", "success")

        if resources["ram_available_gb"] < 4:
            print_status("Low available RAM", "warning")
            print_status("Close other applications before training", "info")

        # Disk space check
        disk_usage = shutil.disk_usage(Path.cwd())
        resources["disk_space_gb"] = disk_usage.free / 1024**3

        print_status(f"Free disk space: {resources['disk_space_gb']:.1f} GB", "success")

        if resources["disk_space_gb"] < 10:
            print_status("Low disk space", "warning")
            print_status("Training requires ~5-10GB for models and logs", "info")

        # CPU check
        resources["cpu_count"] = psutil.cpu_count()
        print_status(f"CPU cores: {resources['cpu_count']}", "success")

    except Exception as e:
        print_status(f"System resource check failed: {e}", "error")
        resources["status"] = "error"

    return resources


def check_training_data() -> Dict:
    """Check training data availability"""
    print_header("Training Data Check")

    data_info = {
        "datasets_found": [],
        "total_samples": 0,
        "status": "error"
    }

    # Common data locations to check
    data_paths = [
        "data/processed/training_dataset/train.json",
        "data/processed/training_dataset/dev.json",
        "data/processed/training_dataset/test.json",
        "data/processed/cold_dataset.json",
        "data/processed/unified_training_data.json"
    ]

    found_datasets = []
    total_samples = 0

    for path in data_paths:
        full_path = Path(path)
        if full_path.exists():
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, dict) and 'texts' in data:
                    sample_count = len(data['texts'])
                    found_datasets.append((path, sample_count))
                    total_samples += sample_count
                    print_status(f"Found: {path} ({sample_count:,} samples)", "success")
                elif isinstance(data, list):
                    sample_count = len(data)
                    found_datasets.append((path, sample_count))
                    total_samples += sample_count
                    print_status(f"Found: {path} ({sample_count:,} samples)", "success")

            except Exception as e:
                print_status(f"Error reading {path}: {e}", "warning")

    data_info["datasets_found"] = found_datasets
    data_info["total_samples"] = total_samples

    if total_samples > 0:
        print_status(f"Total training samples: {total_samples:,}", "success")
        data_info["status"] = "success"

        # Estimate training time
        if total_samples < 1000:
            print_status("Small dataset - training will be quick", "info")
        elif total_samples < 10000:
            print_status("Medium dataset - training ~10-30 minutes", "info")
        else:
            print_status("Large dataset - training ~1-3 hours", "info")
    else:
        print_status("No training data found", "error")
        print_status("Run: python scripts/create_unified_training_data.py", "info")

    return data_info


def check_model_configs() -> Dict:
    """Check model configuration files"""
    print_header("Model Configuration Check")

    config_info = {
        "configs_found": [],
        "status": "error"
    }

    # Configuration locations to check
    config_paths = [
        "configs/training_config.json",
        "config/training_config.json",
        "src/cyberpuppy/training/config.py",
        "scripts/training/configs/"
    ]

    found_configs = []

    for path in config_paths:
        full_path = Path(path)
        if full_path.exists():
            if full_path.is_file():
                found_configs.append(str(path))
                print_status(f"Found config: {path}", "success")
            elif full_path.is_dir():
                config_files = list(full_path.glob("*.json")) + list(full_path.glob("*.yaml"))
                for config_file in config_files:
                    found_configs.append(str(config_file))
                    print_status(f"Found config: {config_file}", "success")

    config_info["configs_found"] = found_configs

    if found_configs:
        config_info["status"] = "success"
        print_status(f"Available configurations: {len(found_configs)}", "success")
    else:
        print_status("No model configurations found", "warning")
        print_status("Default configurations will be used", "info")
        config_info["status"] = "warning"

    return config_info


def check_dependencies() -> Dict:
    """Check required Python packages"""
    print_header("Dependencies Check")

    deps_info = {
        "missing_packages": [],
        "optional_missing": [],
        "status": "success"
    }

    # Required packages
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("tqdm", "Progress bars"),
    ]

    # Optional packages
    optional_packages = [
        ("tensorboard", "TensorBoard (training visualization)"),
        ("wandb", "Weights & Biases (experiment tracking)"),
        ("captum", "Model interpretability"),
        ("ckip_transformers", "Chinese NLP"),
    ]

    # Check required packages
    for package, description in required_packages:
        try:
            __import__(package)
            print_status(f"{description}: Available", "success")
        except ImportError:
            print_status(f"{description}: MISSING", "error")
            deps_info["missing_packages"].append(package)

    # Check optional packages
    for package, description in optional_packages:
        try:
            __import__(package)
            print_status(f"{description}: Available", "success")
        except ImportError:
            print_status(f"{description}: Not installed (optional)", "warning")
            deps_info["optional_missing"].append(package)

    if deps_info["missing_packages"]:
        deps_info["status"] = "error"
        print_status("Install missing packages with: pip install -r requirements.txt", "info")
    elif deps_info["optional_missing"]:
        deps_info["status"] = "warning"

    return deps_info


def generate_training_recommendations(gpu_info: Dict, resources: Dict, data_info: Dict) -> Dict:
    """Generate training configuration recommendations"""
    print_header("Training Recommendations")

    recommendations = {
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "use_amp": True,
        "num_workers": 0,
        "epochs": 3
    }

    # GPU-based recommendations
    if gpu_info["available"]:
        gpu_memory = gpu_info["memory_gb"]

        if gpu_memory < 6:  # RTX 3050 4GB
            recommendations.update({
                "batch_size": 8,
                "gradient_accumulation_steps": 4,
                "use_amp": True,
                "empty_cache_steps": 10
            })
            print_status("Low GPU memory detected", "info")
            print_status("Recommended: Small batch size with gradient accumulation", "info")

        elif gpu_memory < 12:  # RTX 3060/4060
            recommendations.update({
                "batch_size": 16,
                "gradient_accumulation_steps": 2,
                "use_amp": True
            })
            print_status("Medium GPU memory detected", "info")
            print_status("Recommended: Standard batch size", "info")

        else:  # High-end GPU
            recommendations.update({
                "batch_size": 32,
                "gradient_accumulation_steps": 1,
                "use_amp": True
            })
            print_status("High GPU memory detected", "info")
            print_status("Recommended: Large batch size", "info")

    else:
        # CPU training
        recommendations.update({
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "use_amp": False,
            "device": "cpu"
        })
        print_status("GPU not available - CPU training recommended", "warning")
        print_status("CPU training will be significantly slower", "info")

    # Data-based recommendations
    if data_info["total_samples"] > 0:
        sample_count = data_info["total_samples"]

        if sample_count < 1000:
            recommendations["epochs"] = 10
            print_status("Small dataset: More epochs recommended", "info")
        elif sample_count < 10000:
            recommendations["epochs"] = 5
            print_status("Medium dataset: Standard training", "info")
        else:
            recommendations["epochs"] = 3
            print_status("Large dataset: Fewer epochs to prevent overfitting", "info")

    # System resource recommendations
    if resources["cpu_count"] >= 4 and gpu_info["available"]:
        recommendations["num_workers"] = min(4, resources["cpu_count"] // 2)
        print_status(f"Recommended dataloader workers: {recommendations['num_workers']}", "info")

    # Print final recommendations
    print_status("Recommended training configuration:", "info")
    for key, value in recommendations.items():
        print(f"  {key}: {value}")

    return recommendations


def run_preflight_check():
    """Run complete pre-flight check"""
    start_time = time.time()

    print(f"{Colors.BOLD}{Colors.WHITE}")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                  CYBERPUPPY TRAINING                        │")
    print("│                  PRE-FLIGHT CHECK                           │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"{Colors.END}")

    # Run all checks
    checks = {
        "gpu": check_gpu(),
        "resources": check_system_resources(),
        "data": check_training_data(),
        "configs": check_model_configs(),
        "dependencies": check_dependencies()
    }

    # Generate recommendations
    recommendations = generate_training_recommendations(
        checks["gpu"], checks["resources"], checks["data"]
    )

    # Final summary
    print_header("FINAL SUMMARY")

    # Count successful checks
    successful_checks = []
    warning_checks = []
    failed_checks = []

    for check_name, check_result in checks.items():
        status = check_result.get("status", "error")
        if status == "success":
            successful_checks.append(check_name)
        elif status == "warning":
            warning_checks.append(check_name)
        else:
            failed_checks.append(check_name)

    # Print results
    if failed_checks:
        print_status(f"Failed checks: {', '.join(failed_checks)}", "error")
        print_status("Fix the above issues before training", "error")
        ready_status = False
    elif warning_checks:
        print_status(f"Warnings: {', '.join(warning_checks)}", "warning")
        print_status("Training can proceed with caution", "warning")
        ready_status = True
    else:
        print_status("All checks passed!", "success")
        ready_status = True

    print_status(f"Successful checks: {', '.join(successful_checks)}", "success")

    # Ready status
    if ready_status:
        print(f"\n{Colors.GREEN}{Colors.BOLD}>>> READY TO TRAIN! <<<{Colors.END}")
        print(f"\n{Colors.BLUE}To start training, run:{Colors.END}")
        print(f"{Colors.WHITE}python scripts/training/train.py --config recommended{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}>>> NOT READY - Please fix issues above <<<{Colors.END}")

    # Execution time
    execution_time = time.time() - start_time
    print(f"\n{Colors.CYAN}Pre-flight check completed in {execution_time:.2f} seconds{Colors.END}")

    return ready_status, checks, recommendations


def save_system_info(checks: Dict, recommendations: Dict, output_file: str = "system_info.json"):
    """Save system information for debugging"""
    system_info = {
        "timestamp": time.time(),
        "checks": checks,
        "recommendations": recommendations,
        "python_version": sys.version,
        "platform": sys.platform
    }

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(system_info, f, indent=2, ensure_ascii=False, default=str)
        print_status(f"System info saved to: {output_file}", "info")
    except Exception as e:
        print_status(f"Failed to save system info: {e}", "warning")


def main():
    """Main entry point"""
    try:
        ready, checks, recommendations = run_preflight_check()

        # Save system info for debugging
        save_system_info(checks, recommendations)

        # Exit with appropriate code
        sys.exit(0 if ready else 1)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Pre-flight check interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Pre-flight check failed with error: {e}{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()