"""
訓練資料載入器與資料集類別
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict, Counter

from .normalizer import DataNormalizer, LabelUnifier
from .feature_extractor import CombinedFeatureExtractor

logger = logging.getLogger(__name__)


class TrainingDataset(Dataset):
    """多任務學習訓練資料集"""

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str = 'train',
                 include_features: bool = False,
                 ntusd_path: Optional[str] = None,
                 max_length: int = 512):
        """
        初始化訓練資料集

        Args:
            data_path: 資料目錄路徑
            split: 資料分割 ('train', 'dev', 'test')
            include_features: 是否包含手工特徵
            ntusd_path: NTUSD詞典路徑
            max_length: 最大文字長度
        """
        self.data_path = Path(data_path)
        self.split = split
        self.include_features = include_features
        self.max_length = max_length

        # 初始化處理器
        self.normalizer = DataNormalizer()
        self.label_unifier = LabelUnifier()

        if include_features:
            self.feature_extractor = CombinedFeatureExtractor(ntusd_path)
        else:
            self.feature_extractor = None

        # 載入資料
        self.data = self.load_data()
        self.label_encoders = self._create_label_encoders()

    def load_data(self) -> List[Dict[str, Any]]:
        """載入資料"""
        file_path = self.data_path / f'{self.split}.json'

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"Loaded {len(data)} {self.split} samples from {file_path}")
            return data

        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise

    def _create_label_encoders(self) -> Dict[str, Dict[str, int]]:
        """創建標籤編碼器"""
        encoders = {
            'toxicity': {'none': 0, 'toxic': 1, 'severe': 2},
            'bullying': {'none': 0, 'harassment': 1, 'threat': 2},
            'role': {'none': 0, 'perpetrator': 1, 'victim': 2, 'bystander': 3},
            'emotion': {'neg': 0, 'neu': 1, 'pos': 2},
            'emotion_strength': {i: i for i in range(5)}  # 0-4
        }
        return encoders

    def encode_labels(self, label: Dict[str, Any]) -> Dict[str, int]:
        """編碼標籤"""
        encoded = {}
        for key, value in label.items():
            if key in self.label_encoders:
                encoded[key] = self.label_encoders[key].get(value, 0)
        return encoded

    def __len__(self) -> int:
        """資料集大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """獲取單個樣本"""
        item = self.data[idx]

        # 文字正規化
        text = self.normalizer.normalize_text(item['text'])

        # 截斷文字
        if len(text) > self.max_length:
            text = text[:self.max_length]

        # 編碼標籤
        encoded_labels = self.encode_labels(item['label'])

        result = {
            'text': text,
            'labels': encoded_labels,
            'original_labels': item['label']
        }

        # 添加手工特徵
        if self.include_features and self.feature_extractor:
            features = self.feature_extractor.extract_features(text)
            result['features'] = features

        return result

    def get_label_distributions(self) -> Dict[str, Dict[str, int]]:
        """獲取標籤分佈"""
        distributions = defaultdict(lambda: defaultdict(int))

        for item in self.data:
            label = item['label']
            for key, value in label.items():
                distributions[key][value] += 1

        return {k: dict(v) for k, v in distributions.items()}

    def get_class_weights(self, task: str = 'toxicity') -> torch.Tensor:
        """計算類別權重（用於不平衡資料集）"""
        if task not in self.label_encoders:
            raise ValueError(f"Unknown task: {task}")

        # 統計各類別數量
        label_counts = Counter()
        for item in self.data:
            label_value = item['label'][task]
            encoded_label = self.label_encoders[task][label_value]
            label_counts[encoded_label] += 1

        # 計算權重
        total_samples = len(self.data)
        num_classes = len(self.label_encoders[task])

        weights = torch.zeros(num_classes)
        for class_idx in range(num_classes):
            count = label_counts.get(class_idx, 1)  # 避免除零
            weights[class_idx] = total_samples / (num_classes * count)

        return weights

    def get_feature_names(self) -> List[str]:
        """獲取特徵名稱"""
        if self.feature_extractor:
            return self.feature_extractor.get_feature_names()
        return []


class MultiTaskDataLoader:
    """多任務資料載入器"""

    def __init__(self,
                 data_path: Union[str, Path],
                 batch_size: int = 32,
                 include_features: bool = False,
                 ntusd_path: Optional[str] = None,
                 num_workers: int = 0):
        """
        初始化多任務資料載入器

        Args:
            data_path: 資料目錄路徑
            batch_size: 批次大小
            include_features: 是否包含手工特徵
            ntusd_path: NTUSD詞典路徑
            num_workers: 資料載入工作進程數
        """
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.include_features = include_features
        self.ntusd_path = ntusd_path
        self.num_workers = num_workers

        # 創建資料集
        self.datasets = {}
        self.dataloaders = {}

        self._load_datasets()
        self._create_dataloaders()

    def _load_datasets(self):
        """載入所有資料集"""
        splits = ['train', 'dev', 'test']

        for split in splits:
            file_path = self.data_path / f'{split}.json'
            if file_path.exists():
                try:
                    dataset = TrainingDataset(
                        data_path=self.data_path,
                        split=split,
                        include_features=self.include_features,
                        ntusd_path=self.ntusd_path
                    )
                    self.datasets[split] = dataset
                    logger.info(f"Loaded {split} dataset: {len(dataset)} samples")

                except Exception as e:
                    logger.warning(f"Failed to load {split} dataset: {e}")

    def _create_dataloaders(self):
        """創建資料載入器"""
        for split, dataset in self.datasets.items():
            shuffle = (split == 'train')  # 只有訓練集需要打亂

            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                collate_fn=self._collate_fn
            )

            self.dataloaders[split] = dataloader

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批次資料整理函數"""
        texts = [item['text'] for item in batch]

        # 整理標籤
        labels = defaultdict(list)
        for item in batch:
            for key, value in item['labels'].items():
                labels[key].append(value)

        # 轉換為張量
        batch_labels = {}
        for key, values in labels.items():
            batch_labels[key] = torch.tensor(values, dtype=torch.long)

        result = {
            'texts': texts,
            'labels': batch_labels
        }

        # 整理特徵
        if self.include_features and 'features' in batch[0]:
            feature_names = list(batch[0]['features'].keys())
            features = []

            for item in batch:
                feature_vector = [item['features'][name] for name in feature_names]
                features.append(feature_vector)

            result['features'] = torch.tensor(features, dtype=torch.float32)
            result['feature_names'] = feature_names

        return result

    def get_dataloader(self, split: str) -> Optional[DataLoader]:
        """獲取指定分割的資料載入器"""
        return self.dataloaders.get(split)

    def get_dataset(self, split: str) -> Optional[TrainingDataset]:
        """獲取指定分割的資料集"""
        return self.datasets.get(split)

    def get_all_dataloaders(self) -> Dict[str, DataLoader]:
        """獲取所有資料載入器"""
        return self.dataloaders.copy()

    def get_class_weights(self, task: str = 'toxicity') -> Optional[torch.Tensor]:
        """獲取類別權重（從訓練集計算）"""
        if 'train' in self.datasets:
            return self.datasets['train'].get_class_weights(task)
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """獲取資料集統計資訊"""
        stats = {
            'splits': {},
            'total_samples': 0
        }

        for split, dataset in self.datasets.items():
            split_stats = {
                'size': len(dataset),
                'label_distributions': dataset.get_label_distributions()
            }
            stats['splits'][split] = split_stats
            stats['total_samples'] += len(dataset)

        return stats


def create_data_loader(data_path: Union[str, Path],
                      batch_size: int = 32,
                      include_features: bool = False,
                      ntusd_path: Optional[str] = None,
                      num_workers: int = 0) -> MultiTaskDataLoader:
    """
    創建多任務資料載入器的便利函數

    Args:
        data_path: 資料目錄路徑
        batch_size: 批次大小
        include_features: 是否包含手工特徵
        ntusd_path: NTUSD詞典路徑
        num_workers: 工作進程數

    Returns:
        多任務資料載入器
    """
    return MultiTaskDataLoader(
        data_path=data_path,
        batch_size=batch_size,
        include_features=include_features,
        ntusd_path=ntusd_path,
        num_workers=num_workers
    )


# 測試函數
if __name__ == "__main__":
    # 測試資料載入
    try:
        loader = create_data_loader(
            data_path="./data/processed/training_dataset",
            batch_size=4,
            include_features=True
        )

        # 測試訓練集
        train_dataloader = loader.get_dataloader('train')
        if train_dataloader:
            for batch in train_dataloader:
                print(f"Batch size: {len(batch['texts'])}")
                print(f"Labels: {list(batch['labels'].keys())}")
                if 'features' in batch:
                    print(f"Feature shape: {batch['features'].shape}")
                break

        # 顯示統計資訊
        stats = loader.get_statistics()
        print(f"Dataset statistics: {stats}")

    except Exception as e:
        logger.error(f"Test failed: {e}")