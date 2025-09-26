#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
執行資料準備測試的簡化版本
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import json
import pandas as pd

# 添加src路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from cyberpuppy.data.normalizer import DataNormalizer, LabelUnifier
    from cyberpuppy.data.feature_extractor import TextFeatureExtractor
    from cyberpuppy.data.validator import DataQualityValidator
    from cyberpuppy.data.loader import TrainingDataset
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Running basic tests only...")


def test_data_normalizer():
    """測試資料正規化器"""
    print("Testing DataNormalizer...")

    try:
        normalizer = DataNormalizer()

        # 測試基本正規化
        result = normalizer.normalize_text("這是  測試   文字。\n")
        assert result == "這是 測試 文字。"

        # 測試空文字處理
        assert normalizer.normalize_text("") == ""
        assert normalizer.normalize_text(None) == ""

        print("  PASS: DataNormalizer basic tests")
        return True

    except Exception as e:
        print(f"  FAIL: DataNormalizer tests: {e}")
        return False


def test_label_unifier():
    """測試標籤統一化器"""
    print("Testing LabelUnifier...")

    try:
        unifier = LabelUnifier()

        # 測試預設標籤
        label = unifier.create_default_label()
        required_keys = {'toxicity', 'bullying', 'role', 'emotion', 'emotion_strength'}
        assert all(key in label for key in required_keys)

        # 測試COLD標籤轉換
        row = pd.Series({'label': 1})
        result = unifier.unify_cold_labels(row)
        assert result['toxicity'] == 'toxic'

        print("  PASS: LabelUnifier basic tests")
        return True

    except Exception as e:
        print(f"  FAIL: LabelUnifier tests: {e}")
        return False


def test_feature_extractor():
    """測試特徵提取器"""
    print("Testing FeatureExtractor...")

    try:
        extractor = TextFeatureExtractor()

        text = "這是一個測試文字！"
        features = extractor.extract_basic_features(text)

        assert 'text_length' in features
        assert 'chinese_ratio' in features
        assert features['text_length'] == len(text)

        print("  PASS: FeatureExtractor basic tests")
        return True

    except Exception as e:
        print(f"  FAIL: FeatureExtractor tests: {e}")
        return False


def test_validator():
    """測試驗證器"""
    print("Testing DataQualityValidator...")

    try:
        validator = DataQualityValidator()

        # 測試有效文字
        result = validator.validate_text_quality("這是一個有效的測試文字")
        assert result['valid'] == True

        # 測試無效文字
        result = validator.validate_text_quality("ab")  # 太短且中文比例低
        assert result['valid'] == False

        print("  PASS: DataQualityValidator basic tests")
        return True

    except Exception as e:
        print(f"  FAIL: DataQualityValidator tests: {e}")
        return False


def test_training_dataset():
    """測試訓練資料集"""
    print("Testing TrainingDataset...")

    temp_dir = tempfile.mkdtemp()
    try:
        # 創建測試資料
        test_data = [
            {
                'text': '這是一個測試文字',
                'label': {
                    'toxicity': 'toxic',
                    'bullying': 'harassment',
                    'role': 'none',
                    'emotion': 'neg',
                    'emotion_strength': 3
                }
            },
            {
                'text': '另一個測試文字',
                'label': {
                    'toxicity': 'none',
                    'bullying': 'none',
                    'role': 'none',
                    'emotion': 'pos',
                    'emotion_strength': 2
                }
            }
        ]

        data_path = Path(temp_dir)
        with open(data_path / 'train.json', 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False)

        # 測試資料集
        dataset = TrainingDataset(data_path, 'train')
        assert len(dataset) == 2

        item = dataset[0]
        assert 'text' in item
        assert 'labels' in item

        print("  PASS: TrainingDataset basic tests")
        return True

    except Exception as e:
        print(f"  FAIL: TrainingDataset tests: {e}")
        return False

    finally:
        shutil.rmtree(temp_dir)


def test_data_pipeline():
    """測試完整資料流水線"""
    print("Testing complete data pipeline...")

    try:
        # 檢查訓練資料是否存在
        data_dir = Path("./data/processed/training_dataset")

        if not data_dir.exists():
            print("  SKIP: Training data directory not found")
            return True

        # 檢查必要檔案
        required_files = ['train.json', 'dev.json', 'test.json', 'detailed_statistics.json']
        missing_files = [f for f in required_files if not (data_dir / f).exists()]

        if missing_files:
            print(f"  FAIL: Missing files: {missing_files}")
            return False

        # 驗證統計檔案
        with open(data_dir / 'detailed_statistics.json', 'r', encoding='utf-8') as f:
            stats = json.load(f)

        assert 'summary' in stats
        assert 'splits' in stats
        assert stats['summary']['total_samples'] > 0

        print("  PASS: Complete data pipeline verification")
        return True

    except Exception as e:
        print(f"  FAIL: Data pipeline tests: {e}")
        return False


def main():
    """主測試函數"""
    print("=" * 50)
    print("Running Data Preparation Tests")
    print("=" * 50)

    tests = [
        test_data_normalizer,
        test_label_unifier,
        test_feature_extractor,
        test_validator,
        test_training_dataset,
        test_data_pipeline,
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ERROR: {test_func.__name__}: {e}")

    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("All tests PASSED!")
        return 0
    else:
        print(f"{total - passed} tests FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)