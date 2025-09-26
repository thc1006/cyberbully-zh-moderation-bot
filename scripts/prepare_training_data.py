#!/usr/bin/env python3
"""
準備訓練資料：將增強後的資料轉換為統一格式並分割為 train/dev/test
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_unified_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    創建統一的標籤格式，符合專案規範：
    - toxicity: {none, toxic, severe}
    - bullying: {none, harassment, threat}
    - role: {none, perpetrator, victim, bystander}
    - emotion: {pos, neu, neg}
    - emotion_strength: {0..4}
    """

    result_df = df.copy()

    # 基本毒性分類 (基於原始 label)
    result_df['toxicity'] = result_df['label'].map({
        0: 'none',
        1: 'toxic'  # 簡化為 toxic，後續可根據內容強度細分
    })

    # 霸凌分類 (基於主題和毒性)
    def classify_bullying(row):
        if row['label'] == 0:
            return 'none'

        # 根據主題和文本內容判斷霸凌類型
        text = row['TEXT'].lower()

        # 簡單的關鍵字分析來區分harassment和threat
        threat_keywords = ['死', '殺', '打', '揍', '滾', '去死', '該死']
        if any(keyword in text for keyword in threat_keywords):
            return 'threat'
        else:
            return 'harassment'

    result_df['bullying'] = result_df.apply(classify_bullying, axis=1)

    # 角色分類 (基於文本語態分析，這裡簡化處理)
    def classify_role(row):
        if row['label'] == 0:
            return 'none'

        text = row['TEXT']

        # 簡單的語態分析
        # 第一人稱 -> perpetrator
        # 第二人稱 -> victim or perpetrator
        # 第三人稱 -> bystander
        if any(pronoun in text for pronoun in ['我', '我們']):
            return 'perpetrator'
        elif any(pronoun in text for pronoun in ['你', '你們']):
            return 'perpetrator'  # 直接攻擊目標
        else:
            return 'bystander'  # 旁觀者評論

    result_df['role'] = result_df.apply(classify_role, axis=1)

    # 情緒分類
    def classify_emotion(row):
        if row['label'] == 0:
            return 'neu'  # 非毒性文本假設為中性
        else:
            return 'neg'  # 毒性文本為負面

    result_df['emotion'] = result_df.apply(classify_emotion, axis=1)

    # 情緒強度 (0-4，基於文本長度和毒性)
    def emotion_strength(row):
        if row['label'] == 0:
            return 0  # 非毒性文本強度為0

        text = row['TEXT']

        # 基於文本特徵計算強度
        strength = 1  # 基礎強度

        # 長度因子
        if len(text) > 50:
            strength += 1

        # 感嘆號因子
        strength += min(text.count('!'), 2)

        # 極端詞彙因子
        extreme_words = ['死', '滾', '垃圾', '蠢', '智障']
        strength += min(sum(1 for word in extreme_words if word in text), 2)

        return min(strength, 4)  # 最大強度為4

    result_df['emotion_strength'] = result_df.apply(emotion_strength, axis=1)

    return result_df

def split_dataset(df: pd.DataFrame, train_ratio: float = 0.8, dev_ratio: float = 0.1,
                  test_ratio: float = 0.1, random_seed: int = 42) -> tuple:
    """
    分割資料集為訓練、驗證和測試集
    """

    # 確保比例總和為1
    total_ratio = train_ratio + dev_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例總和必須為1，當前為: {total_ratio}")

    # 分層抽樣，確保各標籤比例一致
    train_df, temp_df = train_test_split(
        df,
        test_size=(dev_ratio + test_ratio),
        stratify=df['label'],
        random_state=random_seed
    )

    # 計算dev和test的相對比例
    dev_test_ratio = dev_ratio / (dev_ratio + test_ratio)

    dev_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - dev_test_ratio),
        stratify=temp_df['label'],
        random_state=random_seed
    )

    return train_df, dev_df, test_df

def save_datasets(train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame,
                  output_dir: str, format_type: str = "csv"):
    """
    保存分割後的資料集
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 定義檔案名稱
    files = {
        'train': output_path / f"train.{format_type}",
        'dev': output_path / f"dev.{format_type}",
        'test': output_path / f"test.{format_type}"
    }

    datasets = {
        'train': train_df,
        'dev': dev_df,
        'test': test_df
    }

    # 保存檔案
    for split_name, dataset in datasets.items():
        file_path = files[split_name]

        if format_type == "csv":
            dataset.to_csv(file_path, index=False, encoding='utf-8')
        elif format_type == "json":
            dataset.to_json(file_path, orient='records', force_ascii=False, indent=2)

        logger.info(f"已保存 {split_name} 集: {file_path} ({len(dataset):,} 樣本)")

    return files

def generate_dataset_info(train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame,
                         output_file: str):
    """
    生成資料集資訊文件
    """

    info = {
        "dataset_info": {
            "name": "COLD Augmented Dataset for Cyberbullying Detection",
            "version": "1.0",
            "description": "增強後的中文霸凌偵測資料集",
            "language": "Chinese (Simplified & Traditional)",
            "domain": "Social Media Text",
            "task": "Multi-label Classification"
        },

        "splits": {
            "train": {
                "size": len(train_df),
                "purpose": "模型訓練"
            },
            "dev": {
                "size": len(dev_df),
                "purpose": "超參數調優和模型驗證"
            },
            "test": {
                "size": len(test_df),
                "purpose": "最終模型評估"
            }
        },

        "labels": {
            "toxicity": {
                "description": "毒性檢測",
                "values": ["none", "toxic", "severe"],
                "distribution": {
                    split: df['toxicity'].value_counts().to_dict()
                    for split, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]
                }
            },
            "bullying": {
                "description": "霸凌類型",
                "values": ["none", "harassment", "threat"],
                "distribution": {
                    split: df['bullying'].value_counts().to_dict()
                    for split, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]
                }
            },
            "role": {
                "description": "角色分類",
                "values": ["none", "perpetrator", "victim", "bystander"],
                "distribution": {
                    split: df['role'].value_counts().to_dict()
                    for split, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]
                }
            },
            "emotion": {
                "description": "情緒極性",
                "values": ["pos", "neu", "neg"],
                "distribution": {
                    split: df['emotion'].value_counts().to_dict()
                    for split, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]
                }
            },
            "emotion_strength": {
                "description": "情緒強度",
                "values": [0, 1, 2, 3, 4],
                "distribution": {
                    split: df['emotion_strength'].value_counts().to_dict()
                    for split, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]
                }
            }
        },

        "statistics": {
            "total_samples": len(train_df) + len(dev_df) + len(test_df),
            "text_length": {
                split: {
                    "mean": float(df['TEXT'].str.len().mean()),
                    "std": float(df['TEXT'].str.len().std()),
                    "min": int(df['TEXT'].str.len().min()),
                    "max": int(df['TEXT'].str.len().max())
                }
                for split, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]
            }
        },

        "usage": {
            "recommended_models": [
                "hfl/chinese-roberta-wwm-ext",
                "hfl/chinese-macbert-base",
                "bert-base-chinese"
            ],
            "evaluation_metrics": [
                "macro F1-score",
                "micro F1-score",
                "precision per class",
                "recall per class"
            ],
            "training_tips": [
                "使用分層抽樣確保標籤平衡",
                "考慮使用類別權重處理不平衡",
                "多任務學習可能提升整體性能",
                "注意過擬合，使用early stopping"
            ]
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    logger.info(f"資料集資訊已保存: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="準備訓練資料")
    parser.add_argument("--input", required=True, help="增強後的資料檔案")
    parser.add_argument("--output-dir", required=True, help="輸出目錄")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="訓練集比例")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="驗證集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="測試集比例")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="輸出格式")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")

    args = parser.parse_args()

    logger.info("開始準備訓練資料...")

    # 讀取增強後的資料
    logger.info(f"讀取資料: {args.input}")
    df = pd.read_csv(args.input)

    # 創建統一標籤
    logger.info("創建統一標籤格式...")
    df_labeled = create_unified_labels(df)

    # 分割資料集
    logger.info("分割資料集...")
    train_df, dev_df, test_df = split_dataset(
        df_labeled,
        args.train_ratio,
        args.dev_ratio,
        args.test_ratio,
        args.seed
    )

    # 保存資料集
    logger.info("保存分割後的資料集...")
    files = save_datasets(train_df, dev_df, test_df, args.output_dir, args.format)

    # 生成資料集資訊
    info_file = Path(args.output_dir) / "dataset_info.json"
    logger.info("生成資料集資訊...")
    generate_dataset_info(train_df, dev_df, test_df, str(info_file))

    # 打印摘要
    print("\n" + "="*50)
    print("         訓練資料準備完成")
    print("="*50)
    print(f"訓練集: {len(train_df):,} 樣本")
    print(f"驗證集: {len(dev_df):,} 樣本")
    print(f"測試集: {len(test_df):,} 樣本")
    print(f"總計: {len(train_df) + len(dev_df) + len(test_df):,} 樣本")

    print(f"\n標籤分佈:")
    for label_name in ['toxicity', 'bullying', 'emotion']:
        train_dist = train_df[label_name].value_counts()
        print(f"  {label_name}: {dict(train_dist)}")

    print(f"\n輸出檔案:")
    for split_name, file_path in files.items():
        print(f"  {split_name}: {file_path}")
    print(f"  資訊檔案: {info_file}")

    logger.info("訓練資料準備完成！")

if __name__ == "__main__":
    main()