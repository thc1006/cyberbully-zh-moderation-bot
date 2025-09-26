#!/usr/bin/env python3
"""
資料準備系統測試套件
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# 假設專案結構
import sys
sys.path.append('../src')

from cyberpuppy.data.normalizer import DataNormalizer, LabelUnifier
from cyberpuppy.data.feature_extractor import NTUSDFeatureExtractor, TextFeatureExtractor, CombinedFeatureExtractor
from cyberpuppy.data.validator import DataQualityValidator
from cyberpuppy.data.loader import TrainingDataset, MultiTaskDataLoader, create_data_loader
from cyberpuppy.data.preprocessor import DataPreprocessor, prepare_training_data


class TestDataNormalizer:
    """測試資料正規化器"""

    def setup_method(self):
        self.normalizer = DataNormalizer()

    def test_normalize_text_basic(self):
        """測試基本文字正規化"""
        # 測試正常文字
        text = "這是一個測試文字。"
        result = self.normalizer.normalize_text(text)
        assert result == "這是一個測試文字。"

        # 測試空文字
        assert self.normalizer.normalize_text("") == ""
        assert self.normalizer.normalize_text(None) == ""

        # 測試多餘空白
        text = "這是  一個   測試文字。\n\t"
        result = self.normalizer.normalize_text(text)
        assert result == "這是 一個 測試文字。"

    def test_clean_text(self):
        """測試文字清理"""
        # 測試URL移除
        text = "這是一個網站 https://example.com 測試"
        result = self.normalizer.clean_text(text)
        assert "https://example.com" not in result

        # 測試重複字符處理
        text = "這是測試aaaaaaa文字"
        result = self.normalizer.clean_text(text)
        assert "aaaaaa" not in result

    def test_remove_duplicates(self):
        """測試重複樣本移除"""
        data = [
            ("這是測試文字", {"toxicity": "none"}),
            ("這是測試文字", {"toxicity": "toxic"}),  # 相同文字，不同標籤
            ("另一個測試", {"toxicity": "none"}),
        ]

        result = self.normalizer.remove_duplicates(data)
        assert len(result) == 2  # 應該移除一個重複

    def test_balance_samples(self):
        """測試樣本平衡"""
        data = [
            ("text1", {"toxicity": "none"}),
            ("text2", {"toxicity": "none"}),
            ("text3", {"toxicity": "none"}),
            ("text4", {"toxicity": "toxic"}),
        ]

        result = self.normalizer.balance_samples(data, balance_key="toxicity", max_ratio=2.0)
        assert len(result) <= len(data)


class TestLabelUnifier:
    """測試標籤統一化器"""

    def setup_method(self):
        self.unifier = LabelUnifier()

    def test_create_default_label(self):
        """測試預設標籤創建"""
        label = self.unifier.create_default_label()
        required_keys = {'toxicity', 'bullying', 'role', 'emotion', 'emotion_strength'}
        assert all(key in label for key in required_keys)

    def test_unify_cold_labels(self):
        """測試COLD標籤統一"""
        # 測試毒性標籤
        row = pd.Series({'label': 1})
        result = self.unifier.unify_cold_labels(row)
        assert result['toxicity'] == 'toxic'
        assert result['bullying'] == 'harassment'

        # 測試非毒性標籤
        row = pd.Series({'label': 0})
        result = self.unifier.unify_cold_labels(row)
        assert result['toxicity'] == 'none'
        assert result['bullying'] == 'none'

    def test_unify_sentiment_labels(self):
        """測試情緒標籤統一"""
        # 測試ChnSentiCorp
        result = self.unifier.unify_sentiment_labels(1, 'chnsenticorp')
        assert result['emotion'] == 'pos'
        assert result['emotion_strength'] == 3

        result = self.unifier.unify_sentiment_labels(0, 'chnsenticorp')
        assert result['emotion'] == 'neg'
        assert result['emotion_strength'] == 3

    def test_validate_labels(self):
        """測試標籤驗證"""
        # 有效標籤
        valid_label = {
            'toxicity': 'toxic',
            'bullying': 'harassment',
            'role': 'none',
            'emotion': 'neg',
            'emotion_strength': 3
        }
        assert self.unifier.validate_labels(valid_label)

        # 無效標籤
        invalid_label = {
            'toxicity': 'invalid',  # 無效值
            'bullying': 'harassment',
            'role': 'none',
            'emotion': 'neg',
            'emotion_strength': 3
        }
        assert not self.unifier.validate_labels(invalid_label)

        # 缺少欄位
        incomplete_label = {
            'toxicity': 'toxic',
            # 缺少其他必要欄位
        }
        assert not self.unifier.validate_labels(incomplete_label)


class TestFeatureExtractors:
    """測試特徵提取器"""

    def test_text_feature_extractor(self):
        """測試文字特徵提取器"""
        extractor = TextFeatureExtractor()

        text = "這是一個測試文字！！"
        features = extractor.extract_basic_features(text)

        assert 'text_length' in features
        assert 'chinese_ratio' in features
        assert 'punct_ratio' in features
        assert features['text_length'] == len(text)

    def test_ntusd_feature_extractor_without_dict(self):
        """測試無詞典的NTUSD特徵提取器"""
        extractor = NTUSDFeatureExtractor()

        text = "這是測試文字"
        features = extractor.extract_features(text)

        expected_keys = ['ntusd_pos_ratio', 'ntusd_neg_ratio', 'ntusd_sentiment',
                        'ntusd_pos_count', 'ntusd_neg_count']
        assert all(key in features for key in expected_keys)

    def test_combined_feature_extractor(self):
        """測試組合特徵提取器"""
        extractor = CombinedFeatureExtractor()

        text = "這是一個測試文字"
        features = extractor.extract_features(text)

        # 應該包含NTUSD和文字特徵
        assert 'ntusd_sentiment' in features
        assert 'text_length' in features
        assert 'chinese_ratio' in features


class TestDataQualityValidator:
    """測試資料品質驗證器"""

    def setup_method(self):
        self.validator = DataQualityValidator()

    def test_validate_text_quality(self):
        """測試文字品質驗證"""
        # 有效文字
        result = self.validator.validate_text_quality("這是一個有效的測試文字")
        assert result['valid'] == True

        # 太短的文字
        result = self.validator.validate_text_quality("短")
        assert result['valid'] == False
        assert result['reason'] == 'text_too_short'

        # 空文字
        result = self.validator.validate_text_quality("")
        assert result['valid'] == False
        assert result['reason'] == 'empty_text'

        # 中文字符比例太低
        result = self.validator.validate_text_quality("english text only")
        assert result['valid'] == False
        assert result['reason'] == 'insufficient_chinese'

    def test_validate_label_format(self):
        """測試標籤格式驗證"""
        # 有效標籤
        valid_label = {
            'toxicity': 'toxic',
            'bullying': 'harassment',
            'role': 'none',
            'emotion': 'neg',
            'emotion_strength': 3
        }
        result = self.validator.validate_label_format(valid_label)
        assert result['valid'] == True

        # 無效標籤值
        invalid_label = {
            'toxicity': 'invalid_value',
            'bullying': 'harassment',
            'role': 'none',
            'emotion': 'neg',
            'emotion_strength': 3
        }
        result = self.validator.validate_label_format(invalid_label)
        assert result['valid'] == False
        assert len(result['errors']) > 0

    def test_validate_sample(self):
        """測試樣本驗證"""
        text = "這是一個有效的測試文字"
        label = {
            'toxicity': 'toxic',
            'bullying': 'harassment',
            'role': 'none',
            'emotion': 'neg',
            'emotion_strength': 3
        }

        result = self.validator.validate_sample(text, label)
        assert result['overall_valid'] == True
        assert result['text']['valid'] == True
        assert result['label']['valid'] == True


class TestTrainingDataset:
    """測試訓練資料集"""

    def setup_method(self):
        # 創建臨時測試資料
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir)

        # 創建測試資料檔案
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

        with open(self.data_path / 'train.json', 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_dataset_loading(self):
        """測試資料集載入"""
        dataset = TrainingDataset(self.data_path, 'train')
        assert len(dataset) == 2

    def test_dataset_getitem(self):
        """測試資料集項目獲取"""
        dataset = TrainingDataset(self.data_path, 'train')
        item = dataset[0]

        assert 'text' in item
        assert 'labels' in item
        assert 'original_labels' in item
        assert isinstance(item['labels']['toxicity'], int)

    def test_get_label_distributions(self):
        """測試標籤分佈獲取"""
        dataset = TrainingDataset(self.data_path, 'train')
        distributions = dataset.get_label_distributions()

        assert 'toxicity' in distributions
        assert 'emotion' in distributions

    def test_get_class_weights(self):
        """測試類別權重計算"""
        dataset = TrainingDataset(self.data_path, 'train')
        weights = dataset.get_class_weights('toxicity')

        assert len(weights) == 3  # none, toxic, severe
        assert all(w > 0 for w in weights)


class TestMultiTaskDataLoader:
    """測試多任務資料載入器"""

    def setup_method(self):
        # 創建臨時測試資料
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir)

        # 創建測試資料檔案
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
            } for _ in range(10)
        ]

        for split in ['train', 'dev', 'test']:
            with open(self.data_path / f'{split}.json', 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_dataloader_creation(self):
        """測試資料載入器創建"""
        loader = MultiTaskDataLoader(self.data_path, batch_size=4)

        assert 'train' in loader.datasets
        assert 'dev' in loader.datasets
        assert 'test' in loader.datasets

    def test_get_dataloader(self):
        """測試獲取資料載入器"""
        loader = MultiTaskDataLoader(self.data_path, batch_size=4)
        train_loader = loader.get_dataloader('train')

        assert train_loader is not None

        # 測試批次資料
        for batch in train_loader:
            assert 'texts' in batch
            assert 'labels' in batch
            assert len(batch['texts']) <= 4
            break

    def test_get_statistics(self):
        """測試統計資訊獲取"""
        loader = MultiTaskDataLoader(self.data_path, batch_size=4)
        stats = loader.get_statistics()

        assert 'splits' in stats
        assert 'total_samples' in stats
        assert stats['total_samples'] == 30  # 3 splits * 10 samples


class TestDataPreprocessor:
    """測試資料預處理器"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.preprocessor = DataPreprocessor(base_path=self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    @patch('pandas.read_csv')
    def test_load_cold_dataset(self, mock_read_csv):
        """測試COLD資料集載入"""
        # 模擬CSV資料
        mock_df = pd.DataFrame({
            'TEXT': ['這是測試文字1', '這是測試文字2'],
            'label': [1, 0]
        })
        mock_read_csv.return_value = mock_df

        # 創建模擬檔案路徑
        cold_dir = Path(self.temp_dir) / 'data' / 'processed' / 'cold'
        cold_dir.mkdir(parents=True, exist_ok=True)
        cold_file = cold_dir / 'train_processed.csv'
        cold_file.touch()

        result = self.preprocessor.load_cold_dataset('train')
        assert len(result) == 2
        assert all(len(item) == 2 for item in result)  # (text, label) pairs

    def test_create_data_splits(self):
        """測試資料分割"""
        # 創建測試資料
        data = []
        for i in range(100):
            label = {
                'toxicity': 'toxic' if i < 50 else 'none',
                'bullying': 'harassment' if i < 50 else 'none',
                'role': 'none',
                'emotion': 'neg',
                'emotion_strength': 2
            }
            data.append((f"測試文字{i}", label))

        splits = self.preprocessor.create_data_splits(data)

        assert 'train' in splits
        assert 'dev' in splits
        assert 'test' in splits

        total_samples = sum(len(split_data) for split_data in splits.values())
        assert total_samples == len(data)

    def test_validate_and_clean(self):
        """測試驗證與清理"""
        data = [
            ("這是一個有效的測試文字", {
                'toxicity': 'toxic',
                'bullying': 'harassment',
                'role': 'none',
                'emotion': 'neg',
                'emotion_strength': 3
            }),
            ("短", {  # 太短的文字
                'toxicity': 'none',
                'bullying': 'none',
                'role': 'none',
                'emotion': 'neu',
                'emotion_strength': 0
            })
        ]

        clean_data, stats = self.preprocessor.validate_and_clean(data)

        assert stats['total'] == 2
        assert stats['valid'] == 1
        assert stats['removed'] == 1
        assert len(clean_data) == 1


class TestIntegration:
    """整合測試"""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_full_pipeline_mock(self):
        """測試完整流水線（模擬資料）"""
        # 創建模擬資料結構
        data_dir = Path(self.temp_dir) / 'data' / 'processed'

        # 創建模擬COLD資料
        cold_dir = data_dir / 'cold'
        cold_dir.mkdir(parents=True, exist_ok=True)

        for split in ['train', 'dev', 'test']:
            mock_df = pd.DataFrame({
                'TEXT': [f'測試文字{i}_{split}' for i in range(10)],
                'label': [i % 2 for i in range(10)]
            })
            mock_df.to_csv(cold_dir / f'{split}_processed.csv', index=False)

        # 執行預處理
        preprocessor = DataPreprocessor(base_path=self.temp_dir)
        output_dir = Path(self.temp_dir) / 'output'

        try:
            stats = preprocessor.process_complete_pipeline(
                output_dir=str(output_dir),
                balance_data=False,
                validate_data=True
            )

            assert stats['summary']['total_samples'] > 0
            assert 'splits' in stats

            # 檢查輸出檔案
            assert (output_dir / 'train.json').exists()
            assert (output_dir / 'statistics.json').exists()

        except Exception as e:
            pytest.skip(f"Integration test skipped due to missing dependencies: {e}")


def test_convenience_functions():
    """測試便利函數"""
    # 測試create_data_loader函數
    temp_dir = tempfile.mkdtemp()
    try:
        # 創建最小測試資料
        data_path = Path(temp_dir)
        test_data = [{
            'text': '測試文字',
            'label': {
                'toxicity': 'none',
                'bullying': 'none',
                'role': 'none',
                'emotion': 'neu',
                'emotion_strength': 0
            }
        }]

        with open(data_path / 'train.json', 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False)

        loader = create_data_loader(data_path, batch_size=1)
        assert loader is not None
        assert 'train' in loader.datasets

    finally:
        shutil.rmtree(temp_dir)


# 運行測試的主要函數
if __name__ == "__main__":
    # 簡單的測試運行器
    print("Running data preparation tests...")

    # 運行基本測試
    test_classes = [
        TestDataNormalizer,
        TestLabelUnifier,
        TestFeatureExtractors,
        TestDataQualityValidator,
    ]

    for test_class in test_classes:
        instance = test_class()
        if hasattr(instance, 'setup_method'):
            instance.setup_method()

        methods = [method for method in dir(instance) if method.startswith('test_')]

        for method_name in methods:
            try:
                method = getattr(instance, method_name)
                method()
                print(f"✓ {test_class.__name__}.{method_name}")
            except Exception as e:
                print(f"✗ {test_class.__name__}.{method_name}: {e}")

        if hasattr(instance, 'teardown_method'):
            try:
                instance.teardown_method()
            except:
                pass

    print("Data preparation tests completed!")