#!/usr/bin/env python3
"""
建立統一的多任務訓練資料集
結合 COLD (毒性)、ChnSentiCorp (情緒)、DMSC (情緒) 資料
"""

import json
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_cold_data(data_path: str) -> List[Tuple[str, Dict]]:
    """載入 COLD 毒性偵測資料"""
    logger.info(f"Loading COLD data from {data_path}")

    data = []
    df = pd.read_csv(data_path)

    for _, row in df.iterrows():
        text = str(row.get('text', ''))
        if not text or text == 'nan':
            continue

        # COLD 標籤映射
        offensive = row.get('offensive', 0)
        hate = row.get('hate', 0)

        # 毒性等級判斷
        if hate == 1:
            toxicity = "severe"
            bullying = "threat"
        elif offensive == 1:
            toxicity = "toxic"
            bullying = "harassment"
        else:
            toxicity = "none"
            bullying = "none"

        # 統一標籤
        unified_label = {
            'toxicity': toxicity,
            'bullying': bullying,
            'role': 'none',
            'emotion': 'neu',  # COLD 沒有情緒標籤
            'emotion_strength': 0
        }

        data.append((text, unified_label))

    logger.info(f"Loaded {len(data)} COLD samples")
    return data


def load_sentiment_data(
    data_path: str,
    dataset_name: str
) -> List[Tuple[str, Dict]]:
    """載入情緒分析資料 (ChnSentiCorp, DMSC)"""
    logger.info(f"Loading {dataset_name} data from {data_path}")

    data = []

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        for item in raw_data:
            text = item.get('text', '')
            if not text:
                continue

            # 情緒標籤映射
            label = item.get('label', 0)
            if label == 1:
                emotion = 'pos'
                emotion_strength = 3
            elif label == 0:
                emotion = 'neg'
                emotion_strength = 3
            else:
                emotion = 'neu'
                emotion_strength = 0

            # 統一標籤 (情緒資料沒有毒性標籤)
            unified_label = {
                'toxicity': 'none',
                'bullying': 'none',
                'role': 'none',
                'emotion': emotion,
                'emotion_strength': emotion_strength
            }

            data.append((text, unified_label))

    except Exception as e:
        logger.warning(f"Failed to load {dataset_name}: {e}")
        return []

    logger.info(f"Loaded {len(data)} {dataset_name} samples")
    return data


def create_unified_dataset(base_path: str) -> Dict[str, List[Tuple[str,
    Dict]]]:
    """建立統一的訓練資料集"""
    base_path = Path(base_path)

    datasets = {
        'train': [],
        'dev': [],
        'test': []
    }

    # 載入 COLD 資料
    for split in ['train', 'dev', 'test']:
        cold_path = base_path /
            'data' / 'processed' / 'cold' / f'{split}_processed.csv'
        if cold_path.exists():
            cold_data = load_cold_data(str(cold_path))
            datasets[split].extend(cold_data)
        else:
            logger.warning(f"COLD {split} data not found: {cold_path}")

    # 載入情緒資料
    sentiment_datasets = ['chnsenticorp', 'dmsc']

    for dataset_name in sentiment_datasets:
        for split in ['train', 'dev', 'test']:
            # 檢查 JSON 格式
            json_path = base_path / 'data' / 'processed' / dataset_name /
                f'{split}.json'
            if json_path.exists():
                sentiment_data = load_sentiment_data(
                    str(json_path),
                    dataset_name
                )
                datasets[split].extend(sentiment_data)
            else:
                logger.warning(f"{dataset_name} {split} da"
                    "ta not found: {json_path}")

    # 輸出統計
    for split in datasets:
        logger.info(f"{split.capitalize()} set: {l"
            "en(datasets[split])} samples")

        # 統計標籤分佈
        toxicity_counts = {}
        emotion_counts = {}

        for _, label in datasets[split]:
            tox = label['toxicity']
            emo = label['emotion']

            toxicity_counts[tox] = toxicity_counts.get(tox, 0) + 1
            emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

        logger.info(f"  Toxicity: {toxicity_counts}")
        logger.info(f"  Emotion: {emotion_counts}")

    return datasets


def save_unified_dataset(
    datasets: Dict[str,
    List[Tuple[str,
    Dict]]],
    output_dir: str
):
    """儲存統一資料集"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, data in datasets.items():
        output_file = output_dir / f'{split}_unified.json'

        # 轉換為 JSON 格式
        json_data = []
        for text, label in data:
            json_data.append({
                'text': text,
                'label': label
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(json_data)} samples to {output_file}")

    # 儲存資料集統計
    stats = {}
    for split, data in datasets.items():
        stats[split] = {
            'total_samples': len(data),
            'toxicity_distribution': {},
            'emotion_distribution': {}
        }

        for _, label in data:
            tox = label['toxicity']
            emo = label['emotion']

            if tox not in stats[split]['toxicity_distribution']:
                stats[split]['toxicity_distribution'][tox] = 0
            stats[split]['toxicity_distribution'][tox] += 1

            if emo not in stats[split]['emotion_distribution']:
                stats[split]['emotion_distribution'][emo] = 0
            stats[split]['emotion_distribution'][emo] += 1

    with open(output_dir / 'dataset_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Dataset statistics saved to {ou"
        "tput_dir / 'dataset_stats.json'}")


def main():
    base_path = '.'
    output_dir = f'{base_path}/data/processed/unified'

    logger.info("Creating unified multi-task training dataset...")

    # 建立統一資料集
    datasets = create_unified_dataset(base_path)

    # 儲存資料集
    save_unified_dataset(datasets, output_dir)

    logger.info("Unified dataset creation completed!")


if __name__ == "__main__":
    main()
