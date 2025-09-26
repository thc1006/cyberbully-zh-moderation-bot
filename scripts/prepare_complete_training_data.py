#!/usr/bin/env python3
"""
完整的訓練資料準備系統
整合所有資料來源並進行統一化處理

支援資料來源：
- COLD Dataset (毒性偵測)
- ChnSentiCorp (情緒分析)
- DMSC v2 (情緒分析)
- NTUSD (詞典特徵)
- 人工標註樣本
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
import re
from sklearn.model_selection import train_test_split
import opencc
import hashlib

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataNormalizer:
    """資料正規化處理器"""

    def __init__(self):
        # 繁簡轉換器
        self.s2t = opencc.OpenCC('s2t')  # 簡體轉繁體
        self.t2s = opencc.OpenCC('t2s')  # 繁體轉簡體

    def normalize_text(self, text: str, target_format: str = 'traditional') -> str:
        """文字正規化"""
        if not text or pd.isna(text):
            return ""

        text = str(text).strip()

        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text)

        # 繁簡轉換
        if target_format == 'traditional':
            text = self.s2t.convert(text)
        elif target_format == 'simplified':
            text = self.t2s.convert(text)

        return text

    def remove_duplicates(self, data: List[Tuple[str, Dict]],
                         threshold: float = 0.95) -> List[Tuple[str, Dict]]:
        """移除重複樣本"""
        unique_data = []
        seen_hashes = set()

        for text, label in data:
            # 計算文字雜湊
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_data.append((text, label))

        logger.info(f"Removed {len(data) - len(unique_data)} duplicate samples")
        return unique_data


class LabelUnifier:
    """標籤統一化處理器"""

    # 統一標籤映射
    TOXICITY_LABELS = {'none', 'toxic', 'severe'}
    BULLYING_LABELS = {'none', 'harassment', 'threat'}
    ROLE_LABELS = {'none', 'perpetrator', 'victim', 'bystander'}
    EMOTION_LABELS = {'pos', 'neu', 'neg'}
    EMOTION_STRENGTH_RANGE = list(range(5))  # 0-4

    def __init__(self):
        pass

    def unify_cold_labels(self, row: pd.Series) -> Dict[str, Any]:
        """統一COLD資料標籤"""
        label = int(row.get('label', 0))

        # 毒性判斷
        if label == 1:
            toxicity = "toxic"
            bullying = "harassment"
        else:
            toxicity = "none"
            bullying = "none"

        return {
            'toxicity': toxicity,
            'bullying': bullying,
            'role': 'none',
            'emotion': 'neu',
            'emotion_strength': 0
        }

    def unify_sentiment_labels(self, label: int, dataset_name: str) -> Dict[str, Any]:
        """統一情緒分析標籤"""
        if dataset_name == 'chnsenticorp':
            if label == 1:
                emotion = 'pos'
                emotion_strength = 3
            elif label == 0:
                emotion = 'neg'
                emotion_strength = 3
            else:
                emotion = 'neu'
                emotion_strength = 0
        elif dataset_name == 'dmsc':
            # DMSC 可能有多種標籤格式
            if label >= 3:
                emotion = 'pos'
                emotion_strength = min(4, label - 2)
            elif label <= 2:
                emotion = 'neg'
                emotion_strength = min(4, 3 - label)
            else:
                emotion = 'neu'
                emotion_strength = 0
        else:
            emotion = 'neu'
            emotion_strength = 0

        return {
            'toxicity': 'none',
            'bullying': 'none',
            'role': 'none',
            'emotion': emotion,
            'emotion_strength': emotion_strength
        }

    def validate_labels(self, label: Dict[str, Any]) -> bool:
        """驗證標籤格式"""
        required_keys = {'toxicity', 'bullying', 'role', 'emotion', 'emotion_strength'}

        if not all(key in label for key in required_keys):
            return False

        if label['toxicity'] not in self.TOXICITY_LABELS:
            return False
        if label['bullying'] not in self.BULLYING_LABELS:
            return False
        if label['role'] not in self.ROLE_LABELS:
            return False
        if label['emotion'] not in self.EMOTION_LABELS:
            return False
        if label['emotion_strength'] not in self.EMOTION_STRENGTH_RANGE:
            return False

        return True


class NTUSDFeatureExtractor:
    """NTUSD詞典特徵提取器"""

    def __init__(self, ntusd_path: Optional[str] = None):
        self.positive_words = set()
        self.negative_words = set()

        if ntusd_path and Path(ntusd_path).exists():
            self.load_ntusd(ntusd_path)

    def load_ntusd(self, ntusd_path: str):
        """載入NTUSD詞典"""
        try:
            # 假設NTUSD格式為JSON
            with open(ntusd_path, 'r', encoding='utf-8') as f:
                ntusd_data = json.load(f)

            self.positive_words = set(ntusd_data.get('positive', []))
            self.negative_words = set(ntusd_data.get('negative', []))

            logger.info(f"Loaded NTUSD: {len(self.positive_words)} positive, "
                       f"{len(self.negative_words)} negative words")
        except Exception as e:
            logger.warning(f"Failed to load NTUSD: {e}")

    def extract_features(self, text: str) -> Dict[str, float]:
        """提取NTUSD特徵"""
        if not text:
            return {'ntusd_pos_ratio': 0.0, 'ntusd_neg_ratio': 0.0, 'ntusd_sentiment': 0.0}

        words = list(text)  # 中文按字符分割

        pos_count = sum(1 for word in words if word in self.positive_words)
        neg_count = sum(1 for word in words if word in self.negative_words)
        total_words = len(words)

        pos_ratio = pos_count / total_words if total_words > 0 else 0.0
        neg_ratio = neg_count / total_words if total_words > 0 else 0.0
        sentiment = pos_ratio - neg_ratio

        return {
            'ntusd_pos_ratio': pos_ratio,
            'ntusd_neg_ratio': neg_ratio,
            'ntusd_sentiment': sentiment
        }


class DatasetIntegrator:
    """資料集整合器"""

    def __init__(self, base_path: str = '.'):
        self.base_path = Path(base_path)
        self.normalizer = DataNormalizer()
        self.label_unifier = LabelUnifier()
        self.ntusd_extractor = NTUSDFeatureExtractor()

    def load_cold_data(self, split: str) -> List[Tuple[str, Dict]]:
        """載入COLD資料"""
        data_path = self.base_path / 'data' / 'processed' / 'cold' / f'{split}_processed.csv'

        if not data_path.exists():
            logger.warning(f"COLD {split} data not found: {data_path}")
            return []

        data = []
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loading COLD {split}: {len(df)} samples")

            for _, row in df.iterrows():
                text = str(row.get('TEXT', ''))
                text = self.normalizer.normalize_text(text)

                if not text or len(text.strip()) < 3:
                    continue

                label = self.label_unifier.unify_cold_labels(row)

                if self.label_unifier.validate_labels(label):
                    data.append((text, label))

        except Exception as e:
            logger.error(f"Error loading COLD {split}: {e}")

        logger.info(f"Loaded {len(data)} COLD {split} samples")
        return data

    def load_sentiment_data(self, dataset_name: str, split: str,
                           max_samples: Optional[int] = None) -> List[Tuple[str, Dict]]:
        """載入情緒分析資料"""
        data_path = self.base_path / 'data' / 'processed' / dataset_name / f'{split}.json'

        if not data_path.exists():
            logger.warning(f"{dataset_name} {split} data not found: {data_path}")
            return []

        data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            logger.info(f"Loading {dataset_name} {split}: {len(raw_data)} samples")

            for item in raw_data:
                text = item.get('text', '')
                text = self.normalizer.normalize_text(text)

                if not text or len(text.strip()) < 3:
                    continue

                sentiment_label = item.get('label', 0)
                label = self.label_unifier.unify_sentiment_labels(sentiment_label, dataset_name)

                if self.label_unifier.validate_labels(label):
                    data.append((text, label))

            # 限制樣本數量
            if max_samples and len(data) > max_samples:
                data = data[:max_samples]
                logger.info(f"Limited to {max_samples} samples")

        except Exception as e:
            logger.warning(f"Failed to load {dataset_name} {split}: {e}")

        logger.info(f"Loaded {len(data)} {dataset_name} {split} samples")
        return data

    def load_manual_annotations(self, split: str) -> List[Tuple[str, Dict]]:
        """載入人工標註資料"""
        data_path = self.base_path / 'data' / 'processed' / 'manual' / f'{split}_annotated.json'

        if not data_path.exists():
            return []

        data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            for item in raw_data:
                text = self.normalizer.normalize_text(item.get('text', ''))
                if not text:
                    continue

                label = item.get('label', {})
                if self.label_unifier.validate_labels(label):
                    data.append((text, label))

        except Exception as e:
            logger.warning(f"Failed to load manual annotations {split}: {e}")

        logger.info(f"Loaded {len(data)} manual {split} samples")
        return data

    def integrate_datasets(self) -> Dict[str, List[Tuple[str, Dict]]]:
        """整合所有資料集"""
        datasets = {'train': [], 'dev': [], 'test': []}

        # 載入COLD資料（主要任務）
        for split in ['train', 'dev', 'test']:
            cold_data = self.load_cold_data(split)
            datasets[split].extend(cold_data)

        # 載入情緒分析資料（輔助任務）
        sentiment_datasets = ['chnsenticorp', 'dmsc']
        max_sentiment_samples = {'train': 2000, 'dev': 400, 'test': 400}

        for dataset_name in sentiment_datasets:
            for split in ['train', 'dev', 'test']:
                sentiment_data = self.load_sentiment_data(
                    dataset_name, split, max_sentiment_samples[split]
                )
                datasets[split].extend(sentiment_data)

        # 載入人工標註資料
        for split in ['train', 'dev', 'test']:
            manual_data = self.load_manual_annotations(split)
            datasets[split].extend(manual_data)

        # 移除重複樣本
        for split in datasets:
            datasets[split] = self.normalizer.remove_duplicates(datasets[split])

        return datasets

    def balance_dataset(self, data: List[Tuple[str, Dict]],
                       balance_key: str = 'toxicity') -> List[Tuple[str, Dict]]:
        """平衡資料集"""
        label_groups = defaultdict(list)

        for text, label in data:
            key_value = label[balance_key]
            label_groups[key_value].append((text, label))

        # 找到最小類別數量
        min_count = min(len(group) for group in label_groups.values())

        balanced_data = []
        for key_value, group in label_groups.items():
            if len(group) > min_count:
                # 隨機採樣
                np.random.shuffle(group)
                group = group[:min_count]
            balanced_data.extend(group)

        logger.info(f"Balanced dataset: {len(balanced_data)} samples")
        return balanced_data


class DataQualityValidator:
    """資料品質驗證器"""

    def __init__(self):
        pass

    def validate_text_quality(self, text: str) -> Dict[str, Any]:
        """驗證文字品質"""
        if not text:
            return {'valid': False, 'reason': 'Empty text'}

        # 檢查長度
        if len(text) < 3:
            return {'valid': False, 'reason': 'Text too short'}

        if len(text) > 1000:
            return {'valid': False, 'reason': 'Text too long'}

        # 檢查字符類型
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        total_chars = len(text)

        if chinese_chars / total_chars < 0.3:
            return {'valid': False, 'reason': 'Not enough Chinese characters'}

        return {'valid': True, 'reason': 'Valid'}

    def validate_dataset(self, datasets: Dict[str, List[Tuple[str, Dict]]]) -> Dict[str, Any]:
        """驗證整個資料集"""
        validation_report = {
            'splits': {},
            'overall': {
                'total_samples': 0,
                'valid_samples': 0,
                'invalid_samples': 0
            }
        }

        for split, data in datasets.items():
            valid_count = 0
            invalid_reasons = Counter()

            for text, label in data:
                quality = self.validate_text_quality(text)
                if quality['valid']:
                    valid_count += 1
                else:
                    invalid_reasons[quality['reason']] += 1

            validation_report['splits'][split] = {
                'total': len(data),
                'valid': valid_count,
                'invalid': len(data) - valid_count,
                'invalid_reasons': dict(invalid_reasons)
            }

            validation_report['overall']['total_samples'] += len(data)
            validation_report['overall']['valid_samples'] += valid_count
            validation_report['overall']['invalid_samples'] += len(data) - valid_count

        return validation_report


def generate_statistics_report(datasets: Dict[str, List[Tuple[str, Dict]]],
                             output_path: str):
    """生成詳細統計報告"""
    report = {
        'summary': {
            'creation_time': pd.Timestamp.now().isoformat(),
            'total_splits': len(datasets),
            'total_samples': sum(len(data) for data in datasets.values())
        },
        'splits': {}
    }

    for split, data in datasets.items():
        if not data:
            continue

        # 基本統計
        split_stats = {
            'total_samples': len(data),
            'text_lengths': [],
            'label_distributions': {
                'toxicity': Counter(),
                'bullying': Counter(),
                'role': Counter(),
                'emotion': Counter(),
                'emotion_strength': Counter()
            }
        }

        # 收集統計資訊
        for text, label in data:
            split_stats['text_lengths'].append(len(text))

            for key in split_stats['label_distributions']:
                split_stats['label_distributions'][key][label[key]] += 1

        # 計算文字長度統計
        lengths = split_stats['text_lengths']
        split_stats['text_length_stats'] = {
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths)),
            'median': float(np.median(lengths))
        }

        # 轉換Counter為dict
        for key in split_stats['label_distributions']:
            split_stats['label_distributions'][key] = dict(split_stats['label_distributions'][key])

        del split_stats['text_lengths']  # 移除原始長度列表
        report['splits'][split] = split_stats

    # 儲存報告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Statistics report saved to {output_path}")


def save_training_datasets(datasets: Dict[str, List[Tuple[str, Dict]]],
                          output_dir: str):
    """儲存訓練資料集"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, data in datasets.items():
        if not data:
            continue

        # 轉換為訓練格式
        training_data = []
        for text, label in data:
            training_data.append({
                'text': text,
                'label': label,
                'metadata': {
                    'text_length': len(text),
                    'source': 'integrated_dataset'
                }
            })

        # 儲存JSON格式
        output_file = output_dir / f'{split}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(training_data)} {split} samples to {output_file}")

    # 生成資料載入腳本
    loader_script = output_dir / 'data_loader.py'
    with open(loader_script, 'w', encoding='utf-8') as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
訓練資料載入器
\"\"\"

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from torch.utils.data import Dataset

class MultiTaskDataset(Dataset):
    \"\"\"多任務學習資料集\"\"\"

    def __init__(self, data_path: str, split: str = 'train'):
        self.data_path = Path(data_path)
        self.split = split
        self.data = self.load_data()

    def load_data(self) -> List[Dict[str, Any]]:
        \"\"\"載入資料\"\"\"
        file_path = self.data_path / f'{self.split}.json'

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def get_label_distributions(self) -> Dict[str, Dict[str, int]]:
        \"\"\"獲取標籤分佈\"\"\"
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
    \"\"\"載入訓練資料\"\"\"
    return MultiTaskDataset(data_path, split)

if __name__ == "__main__":
    # 測試資料載入
    dataset = load_training_data('.', 'train')
    print(f"Loaded {len(dataset)} training samples")
    print("Label distributions:", dataset.get_label_distributions())
""")

    logger.info(f"Data loader script saved to {loader_script}")


def main():
    """主要執行函數"""
    logger.info("Starting complete training data preparation...")

    # 初始化整合器
    integrator = DatasetIntegrator('.')

    # 整合所有資料集
    datasets = integrator.integrate_datasets()

    # 資料品質驗證
    validator = DataQualityValidator()
    validation_report = validator.validate_dataset(datasets)

    logger.info("Dataset validation completed:")
    for split, stats in validation_report['splits'].items():
        logger.info(f"  {split}: {stats['valid']}/{stats['total']} valid samples")

    # 儲存訓練資料集
    output_dir = './data/processed/training_dataset'
    save_training_datasets(datasets, output_dir)

    # 生成統計報告
    stats_file = f'{output_dir}/detailed_statistics.json'
    generate_statistics_report(datasets, stats_file)

    # 儲存驗證報告
    validation_file = f'{output_dir}/validation_report.json'
    with open(validation_file, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False)

    logger.info("Complete training data preparation finished!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total samples: {validation_report['overall']['total_samples']}")
    logger.info(f"Valid samples: {validation_report['overall']['valid_samples']}")


if __name__ == "__main__":
    main()