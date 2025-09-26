#!/usr/bin/env python3
"""
CyberPuppy 訓練示例腳本
展示不同的訓練場景和最佳實踐
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_fast_development():
    """快速開發訓練示例"""
    print("🚀 執行快速開發訓練（3 epochs）...")

    cmd = [
        sys.executable, "scripts/train_improved_model.py",
        "--template", "fast_dev",
        "--experiment-name", "fast_dev_test",
        "--model-name", "hfl/chinese-macbert-base",
        "--batch-size", "8",
        "--gpu",
        "--fp16"
    ]

    return subprocess.run(cmd, capture_output=False)

def run_memory_efficient():
    """RTX 3050 記憶體優化訓練示例"""
    print("💾 執行RTX 3050記憶體優化訓練...")

    cmd = [
        sys.executable, "scripts/train_improved_model.py",
        "--config", "configs/training/rtx3050_optimized.yaml",
        "--experiment-name", "rtx3050_optimized_test"
    ]

    return subprocess.run(cmd, capture_output=False)

def run_hyperparameter_search():
    """超參數搜索示例"""
    print("🔍 執行超參數搜索...")

    # 定義搜索參數
    learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]
    batch_sizes = [4, 6, 8]

    for lr in learning_rates:
        for bs in batch_sizes:
            exp_name = f"hp_search_lr{lr}_bs{bs}"
            print(f"📊 測試 LR={lr}, BS={bs}")

            cmd = [
                sys.executable, "scripts/train_improved_model.py",
                "--config", "configs/training/hyperparameter_search.yaml",
                "--experiment-name", exp_name,
                "--learning-rate", str(lr),
                "--batch-size", str(bs)
            ]

            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                print(f"❌ 實驗 {exp_name} 失敗")
                continue

    print("✅ 超參數搜索完成")

def run_production_training():
    """生產環境訓練示例"""
    print("🏭 執行生產環境訓練...")

    cmd = [
        sys.executable, "scripts/train_improved_model.py",
        "--template", "production",
        "--experiment-name", "production_cyberpuppy",
        "--model-name", "hfl/chinese-roberta-wwm-ext",
        "--num-epochs", "20",
        "--gpu",
        "--fp16"
    ]

    return subprocess.run(cmd, capture_output=False)

def run_multitask_training():
    """多任務訓練示例"""
    print("🎯 執行多任務訓練...")

    cmd = [
        sys.executable, "scripts/train_improved_model.py",
        "--config", "configs/training/multi_task.yaml",
        "--experiment-name", "multitask_cyberpuppy"
    ]

    return subprocess.run(cmd, capture_output=False)

def run_model_comparison():
    """模型比較示例"""
    print("⚖️ 執行模型比較實驗...")

    models = [
        "hfl/chinese-macbert-base",
        "hfl/chinese-roberta-wwm-ext",
        "hfl/chinese-bert-wwm-ext"
    ]

    for model in models:
        model_name = model.split('/')[-1]
        exp_name = f"model_comparison_{model_name}"
        print(f"📈 測試模型: {model}")

        cmd = [
            sys.executable, "scripts/train_improved_model.py",
            "--template", "fast_dev",
            "--experiment-name", exp_name,
            "--model-name", model,
            "--num-epochs", "5"
        ]

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"❌ 模型 {model} 測試失敗")
            continue

    print("✅ 模型比較完成")

def run_ablation_study():
    """消融研究示例"""
    print("🧪 執行消融研究...")

    # 測試不同的優化策略
    experiments = [
        {
            "name": "baseline",
            "args": ["--template", "default"]
        },
        {
            "name": "with_fp16",
            "args": ["--template", "default", "--fp16"]
        },
        {
            "name": "with_gradient_accumulation",
            "args": ["--template", "default", "--batch-size", "4"]
        },
        {
            "name": "full_optimization",
            "args": ["--config", "configs/training/rtx3050_optimized.yaml"]
        }
    ]

    for exp in experiments:
        exp_name = f"ablation_{exp['name']}"
        print(f"🔬 執行消融實驗: {exp['name']}")

        cmd = [sys.executable, "scripts/train_improved_model.py"] + exp['args'] + [
            "--experiment-name", exp_name,
            "--num-epochs", "3"
        ]

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"❌ 消融實驗 {exp['name']} 失敗")
            continue

    print("✅ 消融研究完成")

def check_environment():
    """檢查訓練環境"""
    print("🔍 檢查訓練環境...")

    # 檢查 Python 版本
    print(f"Python 版本: {sys.version}")

    # 檢查 PyTorch
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"GPU {i}: {props.name}, 記憶體: {memory_gb:.1f}GB")
    except ImportError:
        print("❌ PyTorch 未安裝")
        return False

    # 檢查 Transformers
    try:
        import transformers
        print(f"Transformers 版本: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers 未安裝")
        return False

    # 檢查專案結構
    required_dirs = [
        "src/cyberpuppy",
        "configs/training",
        "scripts",
        "data"
    ]

    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"❌ 缺少目錄: {dir_path}")
            return False
        print(f"✅ 目錄存在: {dir_path}")

    print("✅ 環境檢查通過")
    return True

def show_training_tips():
    """顯示訓練提示"""
    print("""
🎯 CyberPuppy 訓練提示:

📊 實驗管理:
  • 使用有意義的實驗名稱
  • 記錄重要的配置變更
  • 定期備份最佳模型

💾 記憶體優化 (RTX 3050):
  • 使用 --fp16 啟用混合精度
  • 設置小批次大小 (4-8)
  • 啟用梯度檢查點
  • 監控 GPU 記憶體使用

⚡ 性能優化:
  • 使用 SSD 儲存資料
  • 適當設置 num_workers
  • 考慮使用更短的序列長度

📈 訓練策略:
  • 從快速實驗開始
  • 使用早停防止過擬合
  • 記錄訓練曲線
  • 比較不同的模型架構

🔧 故障排除:
  • 檢查資料格式
  • 驗證配置檔案
  • 監控訓練日誌
  • 測試更小的資料集
    """)

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="CyberPuppy 訓練示例")
    parser.add_argument("--example", type=str, required=True,
                       choices=[
                           "fast_dev",
                           "memory_efficient",
                           "hyperparameter_search",
                           "production",
                           "multitask",
                           "model_comparison",
                           "ablation_study",
                           "check_env",
                           "tips"
                       ],
                       help="選擇要執行的示例")

    args = parser.parse_args()

    # 檢查工作目錄
    if not Path("scripts/train_improved_model.py").exists():
        print("❌ 請在專案根目錄執行此腳本")
        sys.exit(1)

    if args.example == "check_env":
        if not check_environment():
            sys.exit(1)
        return

    if args.example == "tips":
        show_training_tips()
        return

    # 檢查環境
    if not check_environment():
        print("❌ 環境檢查失敗，請先解決問題")
        sys.exit(1)

    # 執行對應的示例
    examples = {
        "fast_dev": run_fast_development,
        "memory_efficient": run_memory_efficient,
        "hyperparameter_search": run_hyperparameter_search,
        "production": run_production_training,
        "multitask": run_multitask_training,
        "model_comparison": run_model_comparison,
        "ablation_study": run_ablation_study
    }

    example_func = examples.get(args.example)
    if example_func:
        print(f"\n🚀 開始執行示例: {args.example}")
        result = example_func()

        if result.returncode == 0:
            print(f"✅ 示例 {args.example} 執行成功")
        else:
            print(f"❌ 示例 {args.example} 執行失敗")
            sys.exit(1)
    else:
        print(f"❌ 未知示例: {args.example}")
        sys.exit(1)

if __name__ == "__main__":
    main()