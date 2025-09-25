#!/usr/bin/env python3
"""Test GPU availability after installation"""
import sys

def test_gpu():
    try:
        import torch
        print("="*60)
        print("GPU Configuration Test")
        print("="*60)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

            # Test GPU computation
            print("\nTesting GPU computation...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x)
            print("[SUCCESS] GPU computation test PASSED")

            # Test memory transfer
            cpu_tensor = torch.randn(100, 100)
            gpu_tensor = cpu_tensor.cuda()
            back_to_cpu = gpu_tensor.cpu()
            print("[SUCCESS] GPU memory transfer test PASSED")

            return True
        else:
            print("[WARNING] GPU not detected!")
            print("\nPossible reasons:")
            print("1. PyTorch CPU version still installed")
            print("2. CUDA drivers need update")
            print("3. Installation still in progress")
            return False

    except ImportError as e:
        print(f"[ERROR] PyTorch not installed or import error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_gpu()
    sys.exit(0 if success else 1)