"""
資料預處理主模組
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from .feature_extractor import CombinedFeatureExtractor
from .normalizer import DataNormalizer, LabelUnifier
from .validator import DataQualityValidator

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """資料預處理主類"""

    def __init__(
        self,
        base_path: str = ".",
        target_format: str = "traditional",
        ntusd_path: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        初始化資料預處理器

        Args:
            base_path: 基礎路徑
            target_format: 目標文字格式
            ntusd_path: NTUSD詞典路徑
            random_state: 隨機種子
        """
        self.base_path = Path(base_path)
        self.random_state = random_state

        # 初始化處理器
        self.normalizer = DataNormalizer(target_format)
        self.label_unifier = LabelUnifier()
        self.feature_extractor = CombinedFeatureExtractor(ntusd_path)
        self.validator = DataQualityValidator()

        # 設定隨機種子
        np.random.seed(random_state)

    def load_cold_dataset(self, split: str) -> List[Tuple[str, Dict]]:
        """載入COLD資料集"""
        data_path = self.base_path / "data" / "processed" / "cold" / f"{split}_processed.csv"

        if not data_path.exists():
            logger.warning(f"COLD {split} data not found: {data_path}")
            return []

        try:
            import pandas as pd

            df = pd.read_csv(data_path)
            data = []

            for _, row in df.iterrows():
                text = self.normalizer.normalize_text(str(row.get("TEXT", "")))
                if not text or len(text.strip()) < 3:
                    continue

                label = self.label_unifier.unify_cold_labels(row)
                if self.label_unifier.validate_labels(label):
                    data.append((text, label))

            logger.info(f"Loaded {len(data)} COLD {split} samples")
            return data

        except Exception as e:
            logger.error(f"Error loading COLD {split}: {e}")
            return []

    def load_sentiment_dataset(
        self, dataset_name: str, split: str, max_samples: Optional[int] = None
    ) -> List[Tuple[str, Dict]]:
        """載入情緒分析資料集"""
        data_path = self.base_path / "data" / "processed" / dataset_name / f"{split}.json"

        if not data_path.exists():
            logger.warning(f"{dataset_name} {split} data not found: {data_path}")
            return []

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            data = []
            for item in raw_data:
                text = self.normalizer.normalize_text(item.get("text", ""))
                if not text or len(text.strip()) < 3:
                    continue

                sentiment_label = item.get("label", 0)
                label = self.label_unifier.unify_sentiment_labels(sentiment_label, dataset_name)

                if self.label_unifier.validate_labels(label):
                    data.append((text, label))

            # 限制樣本數量
            if max_samples and len(data) > max_samples:
                np.random.shuffle(data)
                data = data[:max_samples]

            logger.info(f"Loaded {len(data)} {dataset_name} {split} samples")
            return data

        except Exception as e:
            logger.warning(f"Failed to load {dataset_name} {split}: {e}")
            return []

    def load_manual_annotations(self, split: str) -> List[Tuple[str, Dict]]:
        """載入人工標註資料"""
        data_path = self.base_path / "data" / "processed" / "manual" / f"{split}_annotated.json"

        if not data_path.exists():
            return []

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            data = []
            for item in raw_data:
                text = self.normalizer.normalize_text(item.get("text", ""))
                if not text:
                    continue

                label = self.label_unifier.unify_manual_labels(item.get("label", {}))
                if self.label_unifier.validate_labels(label):
                    data.append((text, label))

            logger.info(f"Loaded {len(data)} manual {split} samples")
            return data

        except Exception as e:
            logger.warning(f"Failed to load manual annotations {split}: {e}")
            return []

    def integrate_all_datasets(
        self, balance_data: bool = True, sentiment_limits: Optional[Dict[str, int]] = None
    ) -> Dict[str, List[Tuple[str, Dict]]]:
        """整合所有資料集"""
        if sentiment_limits is None:
            sentiment_limits = {"train": 2000, "dev": 400, "test": 400}

        datasets = {"train": [], "dev": [], "test": []}

        # 載入COLD資料（主要任務）
        for split in ["train", "dev", "test"]:
            cold_data = self.load_cold_dataset(split)
            datasets[split].extend(cold_data)

        # 載入情緒分析資料（輔助任務）
        sentiment_datasets = ["chnsenticorp", "dmsc"]

        for dataset_name in sentiment_datasets:
            for split in ["train", "dev", "test"]:
                max_samples = sentiment_limits.get(split, 1000)
                sentiment_data = self.load_sentiment_dataset(dataset_name, split, max_samples)
                datasets[split].extend(sentiment_data)

        # 載入人工標註資料
        for split in ["train", "dev", "test"]:
            manual_data = self.load_manual_annotations(split)
            datasets[split].extend(manual_data)

        # 資料清理
        for split in datasets:
            # 移除重複
            datasets[split] = self.normalizer.remove_duplicates(datasets[split])

            # 平衡資料（僅訓練集）
            if balance_data and split == "train":
                datasets[split] = self.normalizer.balance_samples(datasets[split])

        logger.info("Dataset integration completed:")
        for split, data in datasets.items():
            logger.info(f"  {split}: {len(data)} samples")

        return datasets

    def create_data_splits(
        self,
        data: List[Tuple[str, Dict]],
        train_ratio: float = 0.7,
        dev_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, List[Tuple[str, Dict]]]:
        """創建資料分割"""
        assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, "比例總和必須為1"

        # 按標籤分層分割
        texts, labels = zip(*data)

        # 使用毒性標籤進行分層
        stratify_labels = [label["toxicity"] for label in labels]

        # 第一次分割：分離訓練集
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts,
            labels,
            train_size=train_ratio,
            stratify=stratify_labels,
            random_state=self.random_state,
        )

        # 第二次分割：分離開發集和測試集
        relative_dev_ratio = dev_ratio / (dev_ratio + test_ratio)
        temp_stratify = [label["toxicity"] for label in temp_labels]

        dev_texts, test_texts, dev_labels, test_labels = train_test_split(
            temp_texts,
            temp_labels,
            train_size=relative_dev_ratio,
            stratify=temp_stratify,
            random_state=self.random_state,
        )

        return {
            "train": list(zip(train_texts, train_labels)),
            "dev": list(zip(dev_texts, dev_labels)),
            "test": list(zip(test_texts, test_labels)),
        }

    def add_features(self, data: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict, Dict]]:
        """為資料添加特徵"""
        enhanced_data = []

        for text, label in data:
            features = self.feature_extractor.extract_features(text)
            enhanced_data.append((text, label, features))

        logger.info(f"Added features to {len(enhanced_data)} samples")
        return enhanced_data

    def validate_and_clean(
        self, data: List[Tuple[str, Dict]]
    ) -> Tuple[List[Tuple[str, Dict]], Dict]:
        """驗證並清理資料"""
        clean_data = []
        validation_stats = {"total": len(data), "valid": 0, "removed": 0, "issues": {}}

        for text, label in data:
            validation_result = self.validator.validate_sample(text, label)

            if validation_result["overall_valid"]:
                clean_data.append((text, label))
                validation_stats["valid"] += 1
            else:
                validation_stats["removed"] += 1

                # 記錄問題
                if not validation_result["text"]["valid"]:
                    reason = validation_result["text"]["reason"]
                    validation_stats["issues"][reason] = (
                        validation_stats["issues"].get(reason, 0) + 1
                    )

        logger.info(
            f"Data validation: {validation_stats['valid']}/{validation_stats['total']} valid samples"
        )
        return clean_data, validation_stats

    def save_processed_data(self, datasets: Dict[str, List[Tuple[str, Dict]]], output_dir: str):
        """儲存處理後的資料"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split, data in datasets.items():
            if not data:
                continue

            # 轉換為JSON格式
            json_data = []
            for text, label in data:
                json_data.append(
                    {
                        "text": text,
                        "label": label,
                        "metadata": {"text_length": len(text), "source": "preprocessed"},
                    }
                )

            # 儲存資料
            output_file = output_dir / f"{split}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(json_data)} {split} samples to {output_file}")

    def generate_statistics(self, datasets: Dict[str, List[Tuple[str, Dict]]]) -> Dict[str, Any]:
        """生成資料統計"""
        stats = {
            "summary": {
                "total_samples": sum(len(data) for data in datasets.values()),
                "splits": len(datasets),
            },
            "splits": {},
        }

        for split, data in datasets.items():
            if not data:
                continue

            # 基本統計
            texts, labels = zip(*data) if data else ([], [])
            text_lengths = [len(text) for text in texts]

            split_stats = {
                "total_samples": len(data),
                "text_length_stats": {
                    "mean": float(np.mean(text_lengths)) if text_lengths else 0,
                    "std": float(np.std(text_lengths)) if text_lengths else 0,
                    "min": int(np.min(text_lengths)) if text_lengths else 0,
                    "max": int(np.max(text_lengths)) if text_lengths else 0,
                    "median": float(np.median(text_lengths)) if text_lengths else 0,
                },
                "label_distributions": self.label_unifier.get_label_statistics(data),
            }

            stats["splits"][split] = split_stats

        return stats

    def process_complete_pipeline(
        self,
        output_dir: str = "./data/processed/training_dataset",
        balance_data: bool = True,
        validate_data: bool = True,
    ) -> Dict[str, Any]:
        """執行完整的資料處理流水線"""
        logger.info("Starting complete data processing pipeline...")

        # 1. 整合所有資料集
        datasets = self.integrate_all_datasets(balance_data=balance_data)

        # 2. 資料驗證與清理
        if validate_data:
            for split in datasets:
                clean_data, validation_stats = self.validate_and_clean(datasets[split])
                datasets[split] = clean_data
                logger.info(f"{split} validation: {validation_stats}")

        # 3. 儲存處理後的資料
        self.save_processed_data(datasets, output_dir)

        # 4. 生成統計報告
        statistics = self.generate_statistics(datasets)

        # 5. 儲存統計報告
        stats_file = Path(output_dir) / "statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)

        # 6. 生成驗證報告
        if validate_data:
            self.validator.generate_validation_report(
                output_dir, str(Path(output_dir) / "validation_report.json")
            )

        logger.info("Data processing pipeline completed!")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Total samples processed: {statistics['summary']['total_samples']}")

        return statistics


# 便利函數
def prepare_training_data(
    base_path: str = ".",
    output_dir: str = "./data/processed/training_dataset",
    ntusd_path: Optional[str] = None,
    balance_data: bool = True,
) -> Dict[str, Any]:
    """
    準備訓練資料的便利函數

    Args:
        base_path: 基礎路徑
        output_dir: 輸出目錄
        ntusd_path: NTUSD詞典路徑
        balance_data: 是否平衡資料

    Returns:
        處理統計資訊
    """
    preprocessor = DataPreprocessor(base_path=base_path, ntusd_path=ntusd_path)

    return preprocessor.process_complete_pipeline(output_dir=output_dir, balance_data=balance_data)


if __name__ == "__main__":
    # 執行資料預處理
    stats = prepare_training_data()
    print(f"Processing completed. Statistics: {stats}")
