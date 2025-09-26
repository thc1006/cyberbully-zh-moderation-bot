#!/usr/bin/env python3
"""
Data Augmentation Examples for CyberPuppy

This script demonstrates various usage patterns for the data augmentation system,
including basic usage, advanced configurations, and quality validation.

Run examples:
    python examples/data_augmentation_examples.py --example basic
    python examples/data_augmentation_examples.py --example advanced
    python examples/data_augmentation_examples.py --example validation
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from cyberpuppy.data_augmentation import (
    AugmentationPipeline,
    AugmentationConfig,
    PipelineConfig,
    create_augmentation_pipeline,
    SynonymAugmenter,
    BackTranslationAugmenter,
    ContextualAugmenter,
    EDAugmenter
)
from cyberpuppy.data_augmentation.validation import (
    LabelConsistencyValidator,
    LabelConsistencyConfig,
    validate_augmented_dataset
)


def basic_usage_example():
    """Demonstrate basic augmentation usage."""
    print("=" * 60)
    print("基本資料增強範例")
    print("=" * 60)

    # 準備測試資料
    texts = [
        "我很討厭你",
        "今天天氣真好",
        "這個人很笨",
        "我很開心",
        "你給我滾開"
    ]

    labels = [
        {'toxicity': 'toxic', 'bullying': 'harassment', 'emotion': 'neg'},
        {'toxicity': 'none', 'bullying': 'none', 'emotion': 'pos'},
        {'toxicity': 'toxic', 'bullying': 'harassment', 'emotion': 'neg'},
        {'toxicity': 'none', 'bullying': 'none', 'emotion': 'pos'},
        {'toxicity': 'severe', 'bullying': 'threat', 'emotion': 'neg'}
    ]

    print(f"原始資料集大小: {len(texts)} 樣本")

    # 創建增強管道
    print("\n創建中度增強管道...")
    pipeline = create_augmentation_pipeline('medium')

    # 顯示配置
    print(f"啟用策略: {list(pipeline.augmenters.keys())}")
    print(f"增強比例: {pipeline.config.augmentation_ratio}")
    print(f"每文本增強數: {pipeline.config.augmentations_per_text}")

    # 執行增強
    print("\n執行增強...")
    try:
        augmented_texts, augmented_labels = pipeline.augment(
            texts, labels, verbose=True
        )

        print(f"\n增強完成!")
        print(f"增強後資料集大小: {len(augmented_texts)} 樣本")
        print(f"新增樣本數: {len(augmented_texts) - len(texts)}")

        # 顯示增強範例
        print("\n增強範例:")
        new_samples_start = len(texts)
        for i in range(min(3, len(augmented_texts) - new_samples_start)):
            idx = new_samples_start + i
            print(f"  增強樣本 {i+1}: {augmented_texts[idx]}")
            print(f"  標籤: {augmented_labels[idx]}")
            print()

        # 顯示統計資訊
        stats = pipeline.get_statistics()
        print("統計資訊:")
        print(f"  處理樣本數: {stats['total_processed']}")
        print(f"  增強樣本數: {stats['total_augmented']}")
        print(f"  品質過濾數: {stats['quality_filtered']}")
        print(f"  品質通過率: {stats['quality_pass_rate']:.2%}")

        print("\n策略使用分佈:")
        for strategy, percentage in stats['strategy_percentages'].items():
            print(f"  {strategy}: {percentage:.1f}%")

    except Exception as e:
        print(f"增強過程發生錯誤: {e}")
        print("這通常是因為缺少模型依賴，請確保已安裝 transformers 和相關模型")


def individual_augmenters_example():
    """Demonstrate individual augmenter usage."""
    print("=" * 60)
    print("個別增強器範例")
    print("=" * 60)

    test_text = "你真的很笨"
    print(f"測試文本: {test_text}")

    # 1. 同義詞替換
    print("\n1. 同義詞替換:")
    try:
        synonym_augmenter = SynonymAugmenter()
        synonyms = synonym_augmenter.augment(test_text, num_augmentations=3)
        for i, aug in enumerate(synonyms, 1):
            print(f"   {i}. {aug}")
    except Exception as e:
        print(f"   錯誤: {e}")

    # 2. EDA 操作
    print("\n2. EDA 操作:")
    try:
        eda_augmenter = EDAugmenter()
        eda_results = eda_augmenter.augment(test_text, num_augmentations=3)
        for i, aug in enumerate(eda_results, 1):
            print(f"   {i}. {aug}")
    except Exception as e:
        print(f"   錯誤: {e}")

    # 3. 回譯 (需要網路和模型)
    print("\n3. 回譯增強 (需要模型):")
    try:
        backtrans_augmenter = BackTranslationAugmenter()
        backtrans_results = backtrans_augmenter.augment(test_text, num_augmentations=2)
        for i, aug in enumerate(backtrans_results, 1):
            print(f"   {i}. {aug}")
    except Exception as e:
        print(f"   錯誤 (預期): {e}")
        print("   回譯需要下載翻譯模型，在此範例中跳過")

    # 4. 上下文擾動 (需要 MacBERT)
    print("\n4. 上下文擾動 (需要模型):")
    try:
        contextual_augmenter = ContextualAugmenter()
        contextual_results = contextual_augmenter.augment(test_text, num_augmentations=2)
        for i, aug in enumerate(contextual_results, 1):
            print(f"   {i}. {aug}")
    except Exception as e:
        print(f"   錯誤 (預期): {e}")
        print("   上下文擾動需要 MacBERT 模型，在此範例中跳過")


def advanced_configuration_example():
    """Demonstrate advanced configuration options."""
    print("=" * 60)
    print("進階配置範例")
    print("=" * 60)

    # 自訂增強配置
    augmentation_config = AugmentationConfig(
        synonym_prob=0.2,        # 增加同義詞替換機率
        backtrans_prob=0.1,      # 降低回譯機率
        contextual_prob=0.2,     # 增加上下文擾動機率
        eda_prob=0.15,          # 調整 EDA 機率
        quality_threshold=0.5    # 放寬品質門檻
    )

    # 自訂管道配置
    pipeline_config = PipelineConfig(
        use_synonym=True,
        use_backtranslation=False,  # 關閉回譯以避免模型載入
        use_contextual=False,       # 關閉上下文以避免模型載入
        use_eda=True,

        augmentation_ratio=0.4,     # 增強 40% 的資料
        augmentations_per_text=3,   # 每個文本產生 3 個增強版本

        # 品質控制
        quality_threshold=0.3,
        max_length_ratio=1.5,
        min_length_ratio=0.7,

        # 標籤分佈控制
        preserve_label_distribution=True,
        target_balance_ratio=1.2,

        # 處理選項
        batch_size=16,
        num_workers=2,
        use_multiprocessing=False,  # 在範例中關閉多程序
        random_seed=42,

        # 策略權重
        strategy_weights={
            'synonym': 0.6,  # 加重同義詞替換
            'eda': 0.4      # 其餘使用 EDA
        }
    )

    print("自訂配置:")
    print(f"  增強比例: {pipeline_config.augmentation_ratio}")
    print(f"  每文本增強數: {pipeline_config.augmentations_per_text}")
    print(f"  啟用策略: {[k for k, v in vars(pipeline_config).items() if k.startswith('use_') and v]}")
    print(f"  策略權重: {pipeline_config.strategy_weights}")

    # 創建管道
    pipeline = AugmentationPipeline(pipeline_config, augmentation_config)

    # 測試資料 (模擬不平衡資料集)
    texts = (
        ["我很開心", "今天真好", "很棒的一天"] * 2 +  # 正面樣本較多
        ["我討厭你", "很笨"]                        # 負面樣本較少
    )

    labels = (
        [{'toxicity': 'none', 'emotion': 'pos'}] * 6 +
        [{'toxicity': 'toxic', 'emotion': 'neg'}] * 2
    )

    print(f"\n原始標籤分佈:")
    toxicity_dist = {}
    for label in labels:
        tox = label['toxicity']
        toxicity_dist[tox] = toxicity_dist.get(tox, 0) + 1
    for tox, count in toxicity_dist.items():
        print(f"  {tox}: {count} ({count/len(labels)*100:.1f}%)")

    # 執行增強
    print("\n執行進階增強...")
    try:
        augmented_texts, augmented_labels = pipeline.augment(texts, labels)

        print(f"\n增強後標籤分佈:")
        aug_toxicity_dist = {}
        for label in augmented_labels:
            tox = label['toxicity']
            aug_toxicity_dist[tox] = aug_toxicity_dist.get(tox, 0) + 1
        for tox, count in aug_toxicity_dist.items():
            print(f"  {tox}: {count} ({count/len(augmented_labels)*100:.1f}%)")

        # 顯示統計
        stats = pipeline.get_statistics()
        print(f"\n增強統計:")
        print(f"  原始樣本: {len(texts)}")
        print(f"  增強後樣本: {len(augmented_texts)}")
        print(f"  增強比例: {stats['augmentation_ratio']:.2%}")

    except Exception as e:
        print(f"增強失敗: {e}")


def dataframe_processing_example():
    """Demonstrate DataFrame processing."""
    print("=" * 60)
    print("DataFrame 處理範例")
    print("=" * 60)

    # 創建示例 DataFrame
    data = {
        'text': [
            "我很討厭這個人",
            "今天天氣真好",
            "你真的很笨耶",
            "我很開心",
            "滾開啦你",
            "不錯的電影",
            "這個很煩人",
            "美好的一天"
        ],
        'toxicity': ['toxic', 'none', 'toxic', 'none', 'severe', 'none', 'toxic', 'none'],
        'bullying': ['harassment', 'none', 'harassment', 'none', 'threat', 'none', 'harassment', 'none'],
        'emotion': ['neg', 'pos', 'neg', 'pos', 'neg', 'pos', 'neg', 'pos'],
        'emotion_strength': [3, 2, 3, 3, 4, 2, 2, 3]
    }

    df = pd.DataFrame(data)
    print("原始 DataFrame:")
    print(df.to_string(index=False))
    print(f"\n形狀: {df.shape}")

    # 創建適合 DataFrame 的管道
    pipeline = create_augmentation_pipeline('light', strategies=['synonym', 'eda'])

    # 處理 DataFrame
    print("\n執行 DataFrame 增強...")
    try:
        augmented_df = pipeline.augment_dataframe(
            df,
            text_column='text',
            label_columns=['toxicity', 'bullying', 'emotion', 'emotion_strength'],
            verbose=True
        )

        print(f"\n增強後 DataFrame:")
        print(f"形狀: {augmented_df.shape}")
        print(f"新增樣本數: {len(augmented_df) - len(df)}")

        # 顯示新增的樣本
        new_samples = augmented_df[augmented_df['is_augmented'] == True]
        if len(new_samples) > 0:
            print(f"\n新增樣本範例 (顯示前 3 個):")
            display_cols = ['text', 'toxicity', 'emotion']
            print(new_samples[display_cols].head(3).to_string(index=False))

        # 標籤分佈分析
        print(f"\n標籤分佈變化:")
        for col in ['toxicity', 'emotion']:
            original_dist = df[col].value_counts(normalize=True)
            augmented_dist = augmented_df[col].value_counts(normalize=True)

            print(f"\n{col}:")
            for label in set(original_dist.index) | set(augmented_dist.index):
                orig_pct = original_dist.get(label, 0) * 100
                aug_pct = augmented_dist.get(label, 0) * 100
                change = aug_pct - orig_pct
                print(f"  {label}: {orig_pct:.1f}% → {aug_pct:.1f}% ({change:+.1f}%)")

    except Exception as e:
        print(f"DataFrame 處理失敗: {e}")


def validation_example():
    """Demonstrate validation and quality assurance."""
    print("=" * 60)
    print("品質驗證範例")
    print("=" * 60)

    # 準備測試資料
    original_texts = ["我很討厭你", "今天很開心", "這個人很笨"]
    augmented_texts = ["我很憎恨你", "今天很快樂", "這個人很愚蠢"]

    original_labels = [
        {'toxicity': 'toxic', 'emotion': 'neg'},
        {'toxicity': 'none', 'emotion': 'pos'},
        {'toxicity': 'toxic', 'emotion': 'neg'}
    ]
    augmented_labels = original_labels.copy()  # 標籤應該保持一致

    print("驗證設定:")
    print(f"  原始文本數: {len(original_texts)}")
    print(f"  增強文本數: {len(augmented_texts)}")

    # 配置驗證器
    validation_config = LabelConsistencyConfig(
        min_toxicity_confidence=0.6,
        min_emotion_confidence=0.5,
        max_length_change=0.3,
        min_semantic_similarity=0.4
    )

    validator = LabelConsistencyValidator(validation_config)

    # 單個樣本驗證
    print("\n單個樣本驗證:")
    for i, (orig, aug, label) in enumerate(zip(original_texts, augmented_texts, original_labels)):
        result = validator.validate_single_sample(orig, aug, label, label)

        print(f"\n樣本 {i+1}:")
        print(f"  原文: {orig}")
        print(f"  增強: {aug}")
        print(f"  驗證通過: {result.is_valid}")
        print(f"  信心度: {result.confidence:.3f}")

        if result.violations:
            print(f"  違規項目: {result.violations}")
        if result.warnings:
            print(f"  警告: {result.warnings}")

        if result.metrics:
            print(f"  指標:")
            for metric, value in result.metrics.items():
                print(f"    {metric}: {value:.3f}")

    # 批次驗證
    print("\n批次驗證:")
    try:
        batch_results, batch_stats = validator.validate_batch(
            original_texts, augmented_texts,
            original_labels, augmented_labels
        )

        print(f"  總樣本數: {batch_stats['total_samples']}")
        print(f"  有效樣本數: {batch_stats['valid_samples']}")
        print(f"  驗證通過率: {batch_stats['valid_samples']/batch_stats['total_samples']:.2%}")
        print(f"  平均信心度: {batch_stats['average_confidence']:.3f}")

        if batch_stats['violation_counts']:
            print(f"\n違規統計:")
            for violation, count in batch_stats['violation_counts'].items():
                print(f"    {violation}: {count}")

        # 生成完整報告
        from cyberpuppy.data_augmentation.validation import QualityAssuranceReport

        report = QualityAssuranceReport.generate_validation_report(batch_results, batch_stats)
        print(f"\n完整驗證報告:")
        print("-" * 60)
        print(report)

    except Exception as e:
        print(f"批次驗證失敗: {e}")


def performance_analysis_example():
    """Demonstrate performance analysis and monitoring."""
    print("=" * 60)
    print("性能分析範例")
    print("=" * 60)

    # 創建較大的測試資料集
    import time
    import random

    base_texts = [
        "我很討厭你", "今天很開心", "這個人很笨", "天氣真好",
        "你給我滾", "很棒的電影", "討厭的人", "美好的一天"
    ]

    base_labels = [
        {'toxicity': 'toxic', 'emotion': 'neg'},
        {'toxicity': 'none', 'emotion': 'pos'},
        {'toxicity': 'toxic', 'emotion': 'neg'},
        {'toxicity': 'none', 'emotion': 'pos'},
        {'toxicity': 'severe', 'emotion': 'neg'},
        {'toxicity': 'none', 'emotion': 'pos'},
        {'toxicity': 'toxic', 'emotion': 'neg'},
        {'toxicity': 'none', 'emotion': 'pos'}
    ]

    # 擴展資料集
    test_size = 50
    texts = random.choices(base_texts, k=test_size)
    labels = random.choices(base_labels, k=test_size)

    print(f"測試資料集大小: {test_size} 樣本")

    # 測試不同配置的性能
    configs = {
        'light': create_augmentation_pipeline('light', strategies=['synonym', 'eda']),
        'medium': create_augmentation_pipeline('medium', strategies=['synonym', 'eda']),
        'heavy': create_augmentation_pipeline('heavy', strategies=['synonym', 'eda'])
    }

    results = {}

    for name, pipeline in configs.items():
        print(f"\n測試 {name} 配置:")
        print(f"  增強比例: {pipeline.config.augmentation_ratio}")
        print(f"  每文本增強數: {pipeline.config.augmentations_per_text}")

        # 計時
        start_time = time.time()

        try:
            augmented_texts, augmented_labels = pipeline.augment(texts, labels, verbose=False)
            end_time = time.time()

            processing_time = end_time - start_time
            stats = pipeline.get_statistics()

            results[name] = {
                'processing_time': processing_time,
                'original_size': len(texts),
                'augmented_size': len(augmented_texts),
                'new_samples': len(augmented_texts) - len(texts),
                'stats': stats
            }

            print(f"  處理時間: {processing_time:.2f} 秒")
            print(f"  增強樣本數: {len(augmented_texts) - len(texts)}")
            print(f"  處理速度: {len(texts) / processing_time:.1f} 樣本/秒")
            print(f"  品質通過率: {stats['quality_pass_rate']:.2%}")

        except Exception as e:
            print(f"  配置 {name} 失敗: {e}")
            results[name] = {'error': str(e)}

    # 性能比較
    print(f"\n性能比較:")
    print(f"{'配置':<10} {'處理時間':<10} {'新樣本數':<10} {'速度':<15} {'品質率':<10}")
    print("-" * 60)

    for name, result in results.items():
        if 'error' not in result:
            speed = result['original_size'] / result['processing_time']
            quality_rate = result['stats']['quality_pass_rate']
            print(f"{name:<10} {result['processing_time']:<10.2f} {result['new_samples']:<10} {speed:<15.1f} {quality_rate:<10.2%}")
        else:
            print(f"{name:<10} 錯誤: {result['error']}")


def main():
    parser = argparse.ArgumentParser(description="Data Augmentation Examples")
    parser.add_argument('--example', choices=[
        'basic', 'individual', 'advanced', 'dataframe',
        'validation', 'performance', 'all'
    ], default='basic', help='Example to run')

    args = parser.parse_args()

    if args.example == 'basic' or args.example == 'all':
        basic_usage_example()

    if args.example == 'individual' or args.example == 'all':
        if args.example == 'all':
            print("\n" + "="*80 + "\n")
        individual_augmenters_example()

    if args.example == 'advanced' or args.example == 'all':
        if args.example == 'all':
            print("\n" + "="*80 + "\n")
        advanced_configuration_example()

    if args.example == 'dataframe' or args.example == 'all':
        if args.example == 'all':
            print("\n" + "="*80 + "\n")
        dataframe_processing_example()

    if args.example == 'validation' or args.example == 'all':
        if args.example == 'all':
            print("\n" + "="*80 + "\n")
        validation_example()

    if args.example == 'performance' or args.example == 'all':
        if args.example == 'all':
            print("\n" + "="*80 + "\n")
        performance_analysis_example()


if __name__ == "__main__":
    main()