#!/usr/bin/env python3
"""
處理已下載的資料集，轉換成統一格式
"""

import pandas as pd
import json
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(
        self,
        input_dir: str = "data/raw",
        output_dir: str = "data/processed"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_chnsenticorp(self):
        """處理 ChnSentiCorp 資料集"""
        logger.info("Processing ChnSentiCorp dataset...")

        input_path = self.input_dir / "chnsen"
            "ticorp" / 
        if not input_path.exists():
            logger.warning(f"ChnSentiCorp file not found: {input_path}")
            return False

        try:
            # 讀取 CSV
            df = pd.read_csv(input_path)
            logger.info(f"Loaded {len(df)} samples from ChnSentiCorp")
            logger.info(f"Columns: {df.columns.tolist()}")

            # 檢查並處理資料
            if 'review' in df.columns and 'label' in df.columns:
                # 清理資料
                df = df.dropna(subset=['review', 'label'])
                df['label'] = df['label'].astype(int)

                # 分割訓練集和測試集
                train_df, test_df = train_test_split(
                    df, test_size=0.2, random_state=42, stratify=df['label']
                )

                # 進一步分割訓練集和驗證集
                train_df, dev_df = train_test_split(
                    train_df, test_size=0.15, random_state=42,
                    stratify=train_df['label']
                )

                # 儲存處理後的資料
                output_dir = self.output_dir / "chnsenticorp"
                output_dir.mkdir(parents=True, exist_ok=True)

                # 儲存為 JSON 格式
                for split_name, split_df in [
                    ('train', train_df), ('dev', dev_df), ('test', test_df)
                ]:
                    output_file = output_dir / f"{split_name}.json"
                    data = split_df.to_dict('records')
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved {len(split_df)} s"
                        "amples to {output_file}")

                # 統計資訊
                logger.info("Label distribution:")
                logger.info(f"  Positive (1): {(df['label'] == 1).sum()}")
                logger.info(f"  Negative (0): {(df['label'] == 0).sum()}")

                return True

        except Exception as e:
            logger.error(f"Error processing ChnSentiCorp: {e}")
            return False

    def process_dmsc(self):
        """處理 DMSC 資料集"""
        logger.info("Processing DMSC dataset...")

        input_path = self.input_dir / "dmsc" / "DMSC.csv"
        if not input_path.exists():
            logger.warning(f"DMSC file not found: {input_path}")
            return False

        try:
            # 讀取 CSV (可能很大，分塊讀取)
            chunk_size = 10000
            chunks = []

            logger.info("Reading DMSC in chunks...")
            for chunk in pd.read_csv(
                input_path,
                chunksize=chunk_size,
                encoding='utf-8-sig'
            ):
                # 只保留需要的欄位
                if 'Star' in chunk.columns and 'Comment' in chunk.columns:
                    chunk_clean = chunk[['Star', 'Comment']].dropna()
                    chunks.append(chunk_clean)

                # 限制總數量（避免記憶體問題）
                if len(chunks) * chunk_size >= 100000:
                    logger.info("Limiting to 100,000 samp"
                        "les for memory efficiency")
                    break

            # 合併所有塊
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Loaded {len(df)} samples from DMSC")

            # 處理評分 (1-5星)
            df['Star'] = pd.to_numeric(df['Star'], errors='coerce')
            df = df.dropna(subset=['Star', 'Comment'])

            # 轉換為情感標籤 (1-2星:負面, 4-5星:正面, 3星:中性)
            df['sentiment'] = df['Star'].apply(lambda x: 0 if x <= 2 else (1 if
                x >= 4 else -1))

            # 只保留正負面（去除中性）
            df_binary = df[df['sentiment'] != -1].copy()
            df_binary = df_binary.rename(
                columns={'Comment': 'review',
                'sentiment': 'label'}
            )

            # 分割資料集
            train_df, test_df = train_test_split(
                df_binary, test_size=0.2, random_state=42,
                    stratify=df_binary['label']
            )

            train_df, dev_df = train_test_split(
                train_df, test_size=0.15, random_state=42,
                    stratify=train_df['label']
            )

            # 儲存處理後的資料
            output_dir = self.output_dir / "dmsc"
            output_dir.mkdir(parents=True, exist_ok=True)

            for split_name, split_df in [
                ('train', train_df), ('dev', dev_df), ('test', test_df)
            ]:
                output_file = output_dir / f"{split_name}.json"

                # 只保存需要的欄位
                data = split_df[['review', 'label', 'Star']].to_dict('records')

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(split_df)} samples to {output_file}")

            # 統計資訊
            logger.info("Rating distribution:")
            for star in sorted(df['Star'].unique()):
                count = (df['Star'] == star).sum()
                logger.info(f"  {star} stars: {count}")

            return True

        except Exception as e:
            logger.error(f"Error processing DMSC: {e}")
            return False

    def merge_sentiment_datasets(self):
        """合併所有情感分析資料集"""
        logger.info("Merging sentiment analysis datasets...")

        all_data = []
        sources = []

        # 讀取各個已處理的資料集
        for dataset in ['chnsenticorp', 'dmsc']:
            dataset_dir = self.output_dir / dataset
            if dataset_dir.exists():
                for split in ['train', 'dev', 'test']:
                    file_path = dataset_dir / f"{split}.json"
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            for item in data:
                                item['source'] = dataset
                                item['split'] = split
                            all_data.extend(data)
                            sources.append(f"{dataset}_{split}")
                            logger.info(
                                f"Added {len(data)} sample"
                                    "s from {dataset}/{split}"
                            )

        if all_data:
            # 創建合併資料集
            merged_dir = self.output_dir / "merged_sentiment"
            merged_dir.mkdir(parents=True, exist_ok=True)

            # 按 split 分組
            for split in ['train', 'dev', 'test']:
                split_data = [
                    item for item in all_data if item['split'] == split
                ]

                if split_data:
                    output_file = merged_dir / f"{split}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(split_data, f, ensure_ascii=False, indent=2)
                    logger.info(
                        f"Saved {len(split_data)} mer"
                            "ged samples to {output_file}"
                    )

            # 儲存統計資訊
            stats = {
                'total_samples': len(all_data),
                'sources': list(set(item['source'] for item in all_data)),
                'splits': {
                    split: len([
                        item for item in all_data if item['split'] == split
                    ])
                    for split in ['train', 'dev', 'test']
                },
                'label_distribution': {
                    'positive': len([
                        item for item in all_data if item['label'] == 1
                    ]),
                    'negative': len([
                        item for item in all_data if item['label'] == 0
                    ])
                }
            }

            stats_file = merged_dir / "stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            logger.info(f"Total merged samples: {stats['total_samples']}")
            logger.info(f"Positive: {stats['label_distribution']['positive']}")
            logger.info(f"Negative: {stats['label_distribution']['negative']}")

            return True

        return False

    def process_all(self):
        """處理所有資料集"""
        results = {}

        # 處理各個資料集
        results['chnsenticorp'] = self.process_chnsenticorp()
        results['dmsc'] = self.process_dmsc()

        # 合併資料集
        results['merged'] = self.merge_sentiment_datasets()

        # 總結
        print("\n" + "="*60)
        print("Data Processing Summary")
        print("="*60)
        for dataset, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"{dataset}: {status}")

        return results


def main():
    processor = DataProcessor()
    processor.process_all()

    print("\n" + "="*60)
    print("Processing complete!")
    print("Processed data available in: data/processed/")
    print("Merged sentiment data in: data/processed/merged_sentiment/")
    print("="*60)


if __name__ == "__main__":
    main()
