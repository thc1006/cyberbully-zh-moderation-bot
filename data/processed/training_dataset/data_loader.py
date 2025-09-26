#!/usr/bin/env python3
"""
訓練資料載入器
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from torch.utils.data import Dataset

class MultiTaskDataset(Dataset):
    """多任務學習資料集"""

    def __init__(self, data_path: str, split: str = 'train'):
        self.data_path = Path(data_path)
        self.split = split
        self.data = self.load_data()

    def load_data(self) -> List[Dict[str, Any]]:
        """載入資料"""
        file_path = self.data_path / f'{self.split}.json'

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def get_label_distributions(self) -> Dict[str, Dict[str, int]]:
        """獲取標籤分佈"""
        distributions = {
            'toxicity': {},
            'bullying': {},
            'role': {},
            'emotion': {},
            'emotion_strength': {}
        }

        for item in self.data:
            label = item['label']
            for key in distributions:
                value = label[key]
                distributions[key][value] = distributions[key].get(value, 0) + 1

        return distributions

def load_training_data(data_path: str, split: str = 'train') -> MultiTaskDataset:
    """載入訓練資料"""
    return MultiTaskDataset(data_path, split)

if __name__ == "__main__":
    # 測試資料載入
    dataset = load_training_data('.', 'train')
    print(f"Loaded {len(dataset)} training samples")
    print("Label distributions:", dataset.get_label_distributions())
