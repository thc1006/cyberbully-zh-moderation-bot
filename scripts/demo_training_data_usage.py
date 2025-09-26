#!/usr/bin/env python3
"""
訓練資料使用示例
展示如何載入和使用準備好的訓練資料
"""

import sys
import os
from pathlib import Path
import torch

# 添加src路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cyberpuppy.data.loader import create_data_loader, TrainingDataset


def demo_basic_data_loading():
    """示例：基本資料載入"""
    print("=" * 60)
    print("1. 基本資料載入示例")
    print("=" * 60)

    # 檢查資料是否存在
    data_path = "./data/processed/training_dataset"
    if not Path(data_path).exists():
        print(f"❌ 資料目錄不存在: {data_path}")
        print("請先運行 python scripts/prepare_complete_training_data.py")
        return

    try:
        # 創建資料載入器
        data_loader = create_data_loader(
            data_path=data_path,
            batch_size=4,
            include_features=False,
            num_workers=0
        )

        print(f"✅ 成功創建資料載入器")
        print(f"可用分割: {list(data_loader.datasets.keys())}")

        # 獲取統計資訊
        stats = data_loader.get_statistics()
        print(f"總樣本數: {stats['total_samples']}")

        for split, split_stats in stats['splits'].items():
            print(f"{split}: {split_stats['size']} 樣本")

        # 測試訓練集載入
        train_loader = data_loader.get_dataloader('train')
        if train_loader:
            print("\n📊 訓練集第一個批次:")
            for batch in train_loader:
                print(f"  批次大小: {len(batch['texts'])}")
                print(f"  標籤類型: {list(batch['labels'].keys())}")
                print(f"  文字範例: {batch['texts'][0][:50]}...")

                # 顯示標籤分佈
                for label_type, label_tensor in batch['labels'].items():
                    unique_labels = torch.unique(label_tensor)
                    print(f"  {label_type} 標籤: {unique_labels.tolist()}")
                break

    except Exception as e:
        print(f"❌ 資料載入失敗: {e}")


def demo_feature_extraction():
    """示例：特徵提取"""
    print("\n" + "=" * 60)
    print("2. 特徵提取示例")
    print("=" * 60)

    data_path = "./data/processed/training_dataset"
    if not Path(data_path).exists():
        print("❌ 資料目錄不存在，跳過特徵提取示例")
        return

    try:
        # 載入帶特徵的資料
        data_loader = create_data_loader(
            data_path=data_path,
            batch_size=2,
            include_features=True,  # 啟用特徵提取
            num_workers=0
        )

        train_loader = data_loader.get_dataloader('train')
        if train_loader:
            print("✅ 成功創建帶特徵的資料載入器")

            for batch in train_loader:
                print(f"批次大小: {len(batch['texts'])}")

                if 'features' in batch:
                    print(f"特徵形狀: {batch['features'].shape}")
                    print(f"特徵數量: {len(batch['feature_names'])}")
                    print(f"特徵範例: {batch['feature_names'][:5]}")
                    print(f"特徵值範例: {batch['features'][0][:5].tolist()}")
                else:
                    print("⚠️  未找到特徵資料")
                break

    except Exception as e:
        print(f"❌ 特徵提取失敗: {e}")


def demo_class_weights():
    """示例：類別權重計算"""
    print("\n" + "=" * 60)
    print("3. 類別權重計算示例")
    print("=" * 60)

    data_path = "./data/processed/training_dataset"
    if not Path(data_path).exists():
        print("❌ 資料目錄不存在，跳過類別權重示例")
        return

    try:
        # 創建訓練資料集
        train_dataset = TrainingDataset(
            data_path=data_path,
            split='train'
        )

        print(f"✅ 載入訓練集: {len(train_dataset)} 樣本")

        # 獲取標籤分佈
        distributions = train_dataset.get_label_distributions()
        print("\n📊 標籤分佈:")
        for label_type, dist in distributions.items():
            print(f"  {label_type}: {dist}")

        # 計算類別權重
        print("\n⚖️ 類別權重:")
        for task in ['toxicity', 'bullying', 'emotion']:
            try:
                weights = train_dataset.get_class_weights(task)
                print(f"  {task}: {weights.tolist()}")
            except Exception as e:
                print(f"  {task}: 計算失敗 ({e})")

    except Exception as e:
        print(f"❌ 類別權重計算失敗: {e}")


def demo_sample_access():
    """示例：單個樣本存取"""
    print("\n" + "=" * 60)
    print("4. 單個樣本存取示例")
    print("=" * 60)

    data_path = "./data/processed/training_dataset"
    if not Path(data_path).exists():
        print("❌ 資料目錄不存在，跳過樣本存取示例")
        return

    try:
        # 創建資料集
        dataset = TrainingDataset(
            data_path=data_path,
            split='train',
            include_features=False
        )

        print(f"✅ 載入資料集: {len(dataset)} 樣本")

        # 存取前幾個樣本
        print("\n📝 樣本範例:")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n樣本 {i+1}:")
            print(f"  文字: {sample['text'][:80]}...")
            print(f"  毒性: {sample['original_labels']['toxicity']}")
            print(f"  霸凌: {sample['original_labels']['bullying']}")
            print(f"  情緒: {sample['original_labels']['emotion']}")
            print(f"  編碼標籤: {sample['labels']}")

    except Exception as e:
        print(f"❌ 樣本存取失敗: {e}")


def demo_multi_task_training_setup():
    """示例：多任務訓練設置"""
    print("\n" + "=" * 60)
    print("5. 多任務訓練設置示例")
    print("=" * 60)

    data_path = "./data/processed/training_dataset"
    if not Path(data_path).exists():
        print("❌ 資料目錄不存在，跳過訓練設置示例")
        return

    try:
        # 創建多任務資料載入器
        data_loader = create_data_loader(
            data_path=data_path,
            batch_size=8,
            include_features=False,
            num_workers=0
        )

        # 獲取訓練和驗證載入器
        train_loader = data_loader.get_dataloader('train')
        dev_loader = data_loader.get_dataloader('dev')

        print(f"✅ 訓練載入器: {len(train_loader)} 批次")
        print(f"✅ 驗證載入器: {len(dev_loader)} 批次")

        # 模擬訓練循環
        print("\n🚀 模擬訓練循環:")
        epoch = 1
        for i, batch in enumerate(train_loader):
            if i >= 2:  # 只示例前兩個批次
                break

            texts = batch['texts']
            labels = batch['labels']

            print(f"Epoch {epoch}, Batch {i+1}:")
            print(f"  批次大小: {len(texts)}")

            # 模擬多任務損失計算
            task_info = []
            for task, task_labels in labels.items():
                unique_labels = torch.unique(task_labels)
                task_info.append(f"{task}({len(unique_labels)}類)")

            print(f"  任務: {', '.join(task_info)}")

        # 獲取類別權重用於損失函數
        print("\n⚖️ 建議的損失函數權重:")
        toxicity_weights = data_loader.get_class_weights('toxicity')
        if toxicity_weights is not None:
            print(f"毒性任務權重: {toxicity_weights.tolist()}")

    except Exception as e:
        print(f"❌ 訓練設置失敗: {e}")


def main():
    """主函數"""
    print("CyberPuppy 訓練資料使用示例")
    print("=" * 80)

    # 運行所有示例
    demo_basic_data_loading()
    demo_feature_extraction()
    demo_class_weights()
    demo_sample_access()
    demo_multi_task_training_setup()

    print("\n" + "=" * 80)
    print("✅ 所有示例完成!")
    print("\n💡 下一步:")
    print("   1. 查看 docs/DATA_USAGE_GUIDE.md 獲取詳細使用說明")
    print("   2. 運行 python scripts/run_data_tests.py 驗證系統")
    print("   3. 開始模型訓練開發")


if __name__ == "__main__":
    main()