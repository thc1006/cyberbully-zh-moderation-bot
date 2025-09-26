#!/usr/bin/env python3
"""
測試訓練設定和環境
確保所有組件都正常工作
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_packages() -> Dict[str, bool]:
    """檢查必要的Python套件"""
    required_packages = [
        'torch', 'transformers', 'numpy', 'pandas',
        'sklearn', 'yaml', 'tqdm', 'matplotlib', 'seaborn'
    ]

    results = {}
    for package in required_packages:
        try:
            __import__(package)
            results[package] = True
            logger.info(f"✅ {package} 已安裝")
        except ImportError:
            results[package] = False
            logger.error(f"❌ {package} 未安裝")

    return results


def check_gpu_availability():
    """檢查GPU可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            logger.info(f"✅ GPU可用: {gpu_name}")
            logger.info(f"📊 GPU記憶體: {gpu_memory:.1f}GB")
            logger.info(f"🔢 GPU數量: {gpu_count}")

            # RTX 3050特殊提示
            if "3050" in gpu_name:
                logger.info("💡 偵測到RTX 3050，建議使用記憶體優化配置")

            return True
        else:
            logger.warning("⚠️  GPU不可用，將使用CPU")
            return False
    except Exception as e:
        logger.error(f"❌ 檢查GPU時發生錯誤: {e}")
        return False


def check_config_files() -> Dict[str, bool]:
    """檢查配置檔案"""
    config_files = [
        "configs/training/bullying_f1_optimization.yaml",
        "configs/training/rtx3050_optimized.yaml",
        "configs/training/default.yaml"
    ]

    results = {}
    for config_file in config_files:
        config_path = project_root / config_file
        if config_path.exists():
            results[config_file] = True
            logger.info(f"✅ 配置檔案存在: {config_file}")
        else:
            results[config_file] = False
            logger.error(f"❌ 配置檔案不存在: {config_file}")

    return results


def check_model_files() -> Dict[str, bool]:
    """檢查模型相關檔案"""
    model_files = [
        "src/cyberpuppy/models/improved_detector.py",
        "src/cyberpuppy/models/baselines.py",
        "scripts/train_bullying_f1_optimizer.py"
    ]

    results = {}
    for model_file in model_files:
        file_path = project_root / model_file
        if file_path.exists():
            results[model_file] = True
            logger.info(f"✅ 模型檔案存在: {model_file}")
        else:
            results[model_file] = False
            logger.error(f"❌ 模型檔案不存在: {model_file}")

    return results


def check_data_files() -> Dict[str, bool]:
    """檢查資料檔案"""
    data_paths = [
        "data/processed/training_dataset/train.json",
        "data/processed/cold/train.json",
        "data/processed/cold",
        "data/processed"
    ]

    results = {}
    for data_path in data_paths:
        path = project_root / data_path
        if path.exists():
            results[data_path] = True
            if path.is_file():
                # 檢查檔案大小
                size_mb = path.stat().st_size / (1024 * 1024)
                logger.info(f"✅ 資料檔案存在: {data_path} ({size_mb:.1f}MB)")
            else:
                logger.info(f"✅ 資料目錄存在: {data_path}")
        else:
            results[data_path] = False
            logger.error(f"❌ 資料路徑不存在: {data_path}")

    return results


def test_model_import():
    """測試模型匯入"""
    try:
        from src.cyberpuppy.models.improved_detector import ImprovedDetector, ImprovedModelConfig
        logger.info("✅ 改進模型匯入成功")

        # 測試建立配置
        config = ImprovedModelConfig()
        logger.info("✅ 模型配置建立成功")

        return True
    except Exception as e:
        logger.error(f"❌ 模型匯入失敗: {e}")
        return False


def test_tokenizer():
    """測試tokenizer"""
    try:
        from transformers import AutoTokenizer

        model_name = "hfl/chinese-macbert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 測試中文文本
        test_text = "這是一個測試文本，用於檢查中文處理能力。"
        encoding = tokenizer(test_text, return_tensors='pt')

        logger.info("✅ Tokenizer測試成功")
        logger.info(f"📝 測試文本: {test_text}")
        logger.info(f"🔢 Token數量: {len(encoding['input_ids'][0])}")

        return True
    except Exception as e:
        logger.error(f"❌ Tokenizer測試失敗: {e}")
        return False


def test_config_loading():
    """測試配置檔案載入"""
    try:
        import yaml

        config_file = project_root / "configs/training/bullying_f1_optimization.yaml"
        if not config_file.exists():
            logger.error(f"❌ 配置檔案不存在: {config_file}")
            return False

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logger.info("✅ 配置檔案載入成功")
        logger.info(f"📋 模型: {config.get('model', {}).get('base_model', 'Unknown')}")
        logger.info(f"📋 批次大小: {config.get('training', {}).get('batch_size', 'Unknown')}")

        return True
    except Exception as e:
        logger.error(f"❌ 配置檔案載入失敗: {e}")
        return False


def create_test_data():
    """建立測試資料"""
    try:
        import json

        # 建立測試資料目錄
        test_data_dir = project_root / "data" / "test"
        test_data_dir.mkdir(parents=True, exist_ok=True)

        # 生成測試資料
        test_data = []
        test_texts = [
            "這是一個正常的訊息",
            "你真是個笨蛋",
            "我要揍你",
            "今天天氣很好",
            "這個電影很難看"
        ]

        labels = [
            {"toxicity": 0, "bullying": 0, "role": 0, "emotion": 1},
            {"toxicity": 1, "bullying": 1, "role": 1, "emotion": 0},
            {"toxicity": 2, "bullying": 2, "role": 1, "emotion": 0},
            {"toxicity": 0, "bullying": 0, "role": 0, "emotion": 2},
            {"toxicity": 0, "bullying": 0, "role": 0, "emotion": 0}
        ]

        for text, label in zip(test_texts, labels):
            test_data.append({"text": text, **label})

        # 儲存測試資料
        test_file = test_data_dir / "test_data.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 測試資料已建立: {test_file}")
        logger.info(f"📊 測試樣本數: {len(test_data)}")

        return True
    except Exception as e:
        logger.error(f"❌ 建立測試資料失敗: {e}")
        return False


def main():
    """主函數"""
    logger.info("🔍 開始測試訓練設定...")

    # 檢查項目
    checks = [
        ("Python套件", check_python_packages),
        ("GPU可用性", check_gpu_availability),
        ("配置檔案", check_config_files),
        ("模型檔案", check_model_files),
        ("資料檔案", check_data_files),
        ("模型匯入", test_model_import),
        ("Tokenizer", test_tokenizer),
        ("配置載入", test_config_loading),
        ("測試資料", create_test_data)
    ]

    results = {}
    for check_name, check_func in checks:
        logger.info(f"\n📋 檢查: {check_name}")
        logger.info("-" * 40)

        try:
            result = check_func()
            results[check_name] = result

            if isinstance(result, dict):
                # 字典結果 (如套件檢查)
                all_passed = all(result.values())
                results[check_name] = all_passed

                if all_passed:
                    logger.info(f"✅ {check_name}: 全部通過")
                else:
                    failed_items = [k for k, v in result.items() if not v]
                    logger.error(f"❌ {check_name}: 失敗項目 {failed_items}")
            else:
                # 布林結果
                if result:
                    logger.info(f"✅ {check_name}: 通過")
                else:
                    logger.error(f"❌ {check_name}: 失敗")

        except Exception as e:
            logger.error(f"❌ {check_name}: 檢查時發生錯誤 - {e}")
            results[check_name] = False

    # 總結
    logger.info("\n" + "="*50)
    logger.info("📊 檢查結果總結")
    logger.info("="*50)

    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for check_name, passed in results.items():
        status = "✅ 通過" if passed else "❌ 失敗"
        logger.info(f"{check_name:<15}: {status}")

    logger.info(f"\n總體結果: {passed_count}/{total_count} 項檢查通過")

    if passed_count == total_count:
        logger.info("🎉 所有檢查都通過！可以開始訓練")
        logger.info("💡 執行訓練:")
        logger.info("   Windows: scripts\\run_bullying_f1_training.bat")
        logger.info("   Linux/Mac: ./scripts/run_bullying_f1_training.sh")
        logger.info("   Python: python scripts/train_bullying_f1_optimizer.py")
        return 0
    else:
        logger.error("⚠️  部分檢查未通過，請先解決問題")
        logger.info("💡 常見解決方案:")

        if not results.get("Python套件", True):
            logger.info("   - 安裝缺少的套件: pip install [套件名稱]")

        if not results.get("配置檔案", True):
            logger.info("   - 檢查configs/目錄是否完整")

        if not results.get("資料檔案", True):
            logger.info("   - 執行資料下載和預處理腳本")
            logger.info("   - 或使用測試資料: data/test/test_data.json")

        return 1


if __name__ == "__main__":
    sys.exit(main())