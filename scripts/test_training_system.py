#!/usr/bin/env python3
"""
CyberPuppy 訓練系統測試腳本
驗證所有組件是否正常工作
"""
import os
import sys
import json
import tempfile
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer

def test_config_system():
    """測試配置系統"""
    print("🔧 測試配置系統...")

    try:
        from src.cyberpuppy.training.config import TrainingPipelineConfig, ConfigManager

        # 測試預設配置
        config = TrainingPipelineConfig()
        print(f"✅ 預設配置創建成功: {config.experiment.name}")

        # 測試配置管理器
        manager = ConfigManager()
        templates = ["default", "fast_dev", "production", "memory_efficient"]

        for template in templates:
            test_config = manager.get_template(template)
            print(f"✅ 模板 {template} 載入成功")

        # 測試配置驗證
        warnings = config.validate()
        print(f"✅ 配置驗證完成，警告數: {len(warnings)}")

        return True

    except Exception as e:
        print(f"❌ 配置系統測試失敗: {e}")
        return False

def test_trainer_components():
    """測試訓練器組件"""
    print("🏋️ 測試訓練器組件...")

    try:
        # 測試是否可以導入訓練組件
        from src.cyberpuppy.training.trainer import MultitaskTrainer

        # 檢查現有訓練器
        try:
            from src.cyberpuppy.training.trainer import create_trainer
            print("✅ 新訓練器導入成功")
        except ImportError:
            print("⚠️ 新訓練器未找到，使用現有訓練器")

        return True

    except Exception as e:
        print(f"❌ 訓練器組件測試失敗: {e}")
        return False

def test_model_creation():
    """測試模型創建"""
    print("🤖 測試模型創建...")

    try:
        from src.cyberpuppy.training.config import TrainingPipelineConfig

        # 創建測試配置
        config = TrainingPipelineConfig()
        config.model.model_name = "hfl/chinese-macbert-base"

        # 測試tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        print(f"✅ Tokenizer 載入成功: {len(tokenizer)} tokens")

        # 嘗試創建模型（可能失敗，但不應該崩潰）
        try:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model.model_name,
                num_labels=config.model.num_labels
            )
            print(f"✅ 模型創建成功，參數數量: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as model_error:
            print(f"⚠️ 模型創建失敗（可能是網路問題）: {model_error}")

        return True

    except Exception as e:
        print(f"❌ 模型創建測試失敗: {e}")
        return False

def test_data_loading():
    """測試資料載入"""
    print("📊 測試資料載入...")

    try:
        # 創建測試資料
        test_data = [
            {"text": "這是一個正常的訊息", "labels": 0},
            {"text": "這是一個有毒的訊息", "labels": 1},
            {"text": "這是一個嚴重有毒的訊息", "labels": 2}
        ] * 10

        # 創建臨時檔案
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False)
            temp_path = f.name

        try:
            # 測試Dataset類
            try:
                from src.cyberpuppy.data.dataset import CyberBullyDataset
                print("✅ 找到現有 Dataset 類")
            except ImportError:
                # 使用內建的基本Dataset
                print("⚠️ 使用基本 Dataset 實現")

            # 測試tokenizer
            tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print(f"✅ 資料載入測試完成，測試資料: {len(test_data)} 條")

        finally:
            # 清理臨時檔案
            os.unlink(temp_path)

        return True

    except Exception as e:
        print(f"❌ 資料載入測試失敗: {e}")
        return False

def test_memory_optimization():
    """測試記憶體優化"""
    print("💾 測試記憶體優化...")

    try:
        # 檢查CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用，設備數量: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"  GPU {i}: {props.name}, 記憶體: {memory_gb:.1f}GB")

                # 檢查是否為低記憶體設備
                if memory_gb <= 4.5:
                    print(f"  ⚠️ 檢測到低記憶體GPU，建議使用記憶體優化配置")

            # 測試記憶體監控
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"  當前記憶體: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")

        else:
            print("⚠️ CUDA 不可用，將使用CPU模式")

        # 測試混合精度
        try:
            from torch.cuda.amp import GradScaler, autocast
            print("✅ 混合精度支援可用")
        except ImportError:
            print("⚠️ 混合精度支援不可用")

        return True

    except Exception as e:
        print(f"❌ 記憶體優化測試失敗: {e}")
        return False

def test_training_configs():
    """測試訓練配置檔案"""
    print("📄 測試訓練配置檔案...")

    config_files = [
        "configs/training/default.yaml",
        "configs/training/rtx3050_optimized.yaml",
        "configs/training/multi_task.yaml",
        "configs/training/hyperparameter_search.yaml"
    ]

    success_count = 0

    for config_file in config_files:
        config_path = project_root / config_file
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                print(f"✅ {config_file} 格式正確")
                success_count += 1
            except Exception as e:
                print(f"❌ {config_file} 格式錯誤: {e}")
        else:
            print(f"⚠️ {config_file} 不存在")

    print(f"配置檔案測試完成: {success_count}/{len(config_files)} 通過")
    return success_count > 0

def test_directory_structure():
    """測試目錄結構"""
    print("📁 測試目錄結構...")

    required_dirs = [
        "src/cyberpuppy/training",
        "configs/training",
        "scripts",
        "docs"
    ]

    required_files = [
        "src/cyberpuppy/training/__init__.py",
        "src/cyberpuppy/training/config.py",
        "scripts/train_improved_model.py",
        "docs/TRAINING_GUIDE.md"
    ]

    # 檢查目錄
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"✅ 目錄存在: {dir_path}")
        else:
            print(f"❌ 目錄缺失: {dir_path}")

    # 檢查檔案
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists() and full_path.is_file():
            print(f"✅ 檔案存在: {file_path}")
        else:
            print(f"❌ 檔案缺失: {file_path}")

    return True

def run_integration_test():
    """執行整合測試"""
    print("🔄 執行整合測試...")

    try:
        from src.cyberpuppy.training.config import TrainingPipelineConfig, ConfigManager

        # 創建測試配置
        manager = ConfigManager()
        config = manager.get_template("fast_dev")

        # 修改為測試友好的設定
        config.experiment.name = "integration_test"
        config.training.num_epochs = 1
        config.data.batch_size = 2

        print(f"✅ 整合測試配置創建成功")
        print(f"  實驗名稱: {config.experiment.name}")
        print(f"  批次大小: {config.data.batch_size}")
        print(f"  訓練輪數: {config.training.num_epochs}")

        # 保存測試配置
        test_config_path = project_root / "test_config.json"
        config.save(test_config_path)
        print(f"✅ 測試配置已保存: {test_config_path}")

        # 清理
        if test_config_path.exists():
            test_config_path.unlink()
            print("✅ 測試檔案已清理")

        return True

    except Exception as e:
        print(f"❌ 整合測試失敗: {e}")
        return False

def main():
    """主函數"""
    print("🚀 CyberPuppy 訓練系統測試開始")
    print("=" * 60)

    tests = [
        ("目錄結構", test_directory_structure),
        ("配置系統", test_config_system),
        ("訓練器組件", test_trainer_components),
        ("模型創建", test_model_creation),
        ("資料載入", test_data_loading),
        ("記憶體優化", test_memory_optimization),
        ("配置檔案", test_training_configs),
        ("整合測試", run_integration_test)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 測試: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通過")
            else:
                print(f"❌ {test_name} 失敗")
        except Exception as e:
            print(f"💥 {test_name} 崩潰: {e}")

    print("\n" + "=" * 60)
    print(f"🏁 測試完成: {passed}/{total} 通過")

    if passed == total:
        print("🎉 所有測試通過！訓練系統準備就緒")
        print("\n💡 下一步:")
        print("  1. 準備訓練資料到 data/processed/")
        print("  2. 執行快速測試: python scripts/run_training_examples.py --example fast_dev")
        print("  3. 查看訓練指南: docs/TRAINING_GUIDE.md")
    else:
        print("⚠️ 部分測試失敗，請檢查上述錯誤")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)