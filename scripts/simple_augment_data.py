#!/usr/bin/env python3
"""
簡化版資料增強腳本 - 避免複雜依賴問題
使用同義詞替換、句式變換和輕微文本變化來增強資料
"""

import pandas as pd
import numpy as np
import random
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import re
import logging

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleBullyingDataAugmenter:
    """簡化版中文霸凌偵測資料增強器"""

    def __init__(self, intensity: str = "medium"):
        """
        初始化增強器

        Args:
            intensity: 增強強度 ("light", "medium", "heavy")
        """
        self.intensity = intensity
        self.setup_augmentation_rules()

    def setup_augmentation_rules(self):
        """設定各種增強規則"""

        # 同義詞替換字典
        self.synonyms_dict = {
            # 負面情緒詞
            "討厭": ["厭惡", "反感", "嫌棄", "恨"],
            "噁心": ["惡心", "噁爛", "令人作嘔", "令人反胃"],
            "垃圾": ["廢物", "渣滓", "爛貨", "垃圾"],
            "智障": ["弱智", "腦殘", "白痴", "傻瓜"],
            "死": ["去死", "該死", "活該", "死掉"],
            "爛": ["爛掉", "爛透", "破爛", "糟糕"],
            "蠢": ["笨", "愚蠢", "傻", "呆"],

            # 族群相關
            "黑人": ["非洲人", "非裔"],
            "白人": ["歐美人", "洋人"],
            "日本人": ["日本人", "日人"],
            "韓國人": ["韓國人", "韓人"],

            # 地域相關
            "台灣人": ["台灣人", "台人"],
            "大陸人": ["內地人", "大陸人"],
            "香港人": ["港人", "香港人"],
            "東北人": ["東北人"],

            # 性別相關
            "女人": ["女性", "女的", "女士"],
            "男人": ["男性", "男的", "男士"],
            "女生": ["女孩", "女學生", "女同學"],
            "男生": ["男孩", "男學生", "男同學"],

            # 強化詞
            "很": ["非常", "特別", "超級", "極其"],
            "太": ["超", "過於", "實在太"],
            "真的": ["確實", "的確", "實在"],
            "完全": ["絕對", "徹底", "完全"],
        }

        # 語氣詞
        self.tone_words = {
            "強化": ["真的", "超級", "非常", "特別", "極其", "相當", "實在"],
            "疑問": ["難道", "莫非", "豈不是", "不是嗎", "是不是"],
            "感嘆": ["啊", "呀", "哎", "唉", "哇", "喔"],
            "確定": ["絕對", "一定", "肯定", "必然", "當然", "明明"]
        }

        # 句式變換模板
        self.sentence_templates = [
            # 疑問句
            lambda text: f"難道{text}？",
            lambda text: f"{text}，不是嗎？",
            lambda text: f"為什麼{text}？",
            lambda text: f"{text}嗎？",

            # 感嘆句
            lambda text: f"{text}！",
            lambda text: f"真的{text}啊！",
            lambda text: f"{text}，太過分了！",
            lambda text: f"怎麼{text}！",

            # 強調句
            lambda text: f"我覺得{text}",
            lambda text: f"就是{text}",
            lambda text: f"明明{text}",
            lambda text: f"根本{text}",
            lambda text: f"簡直{text}",
        ]

        # 設定增強強度參數
        if self.intensity == "light":
            self.synonym_prob = 0.15
            self.tone_prob = 0.08
            self.template_prob = 0.12
            self.augmentations_per_text = 1
        elif self.intensity == "medium":
            self.synonym_prob = 0.25
            self.tone_prob = 0.12
            self.template_prob = 0.18
            self.augmentations_per_text = 2
        elif self.intensity == "heavy":
            self.synonym_prob = 0.35
            self.tone_prob = 0.18
            self.template_prob = 0.25
            self.augmentations_per_text = 3

    def synonym_replacement(self, text: str) -> str:
        """同義詞替換"""
        result = text

        # 對每個同義詞組進行替換
        for original, synonyms in self.synonyms_dict.items():
            if original in result and random.random() < self.synonym_prob:
                synonym = random.choice(synonyms)
                # 只替換第一個出現的詞
                result = result.replace(original, synonym, 1)

        return result

    def add_tone_words(self, text: str) -> str:
        """添加語氣詞"""
        if random.random() < self.tone_prob:
            category = random.choice(list(self.tone_words.keys()))
            tone_word = random.choice(self.tone_words[category])

            # 隨機選擇插入位置
            position = random.choice(["start", "end", "middle"])

            if position == "start":
                text = f"{tone_word}{text}"
            elif position == "end":
                text = f"{text}{tone_word}"
            else:  # middle
                words = list(text)
                if len(words) > 3:
                    insert_pos = random.randint(1, len(words) - 2)
                    words.insert(insert_pos, tone_word)
                    text = ''.join(words)

        return text

    def sentence_transformation(self, text: str) -> str:
        """句式變換"""
        if random.random() < self.template_prob:
            template = random.choice(self.sentence_templates)
            try:
                # 移除原本的標點符號再應用模板
                clean_text = text.rstrip('。！？，；')
                transformed = template(clean_text)
                return transformed
            except Exception as e:
                logger.debug(f"句式變換失敗: {e}")
                return text
        return text

    def random_insertion(self, text: str) -> str:
        """隨機插入連接詞或語氣詞"""
        insertion_words = ["就是", "真的是", "簡直", "完全", "實在是", "根本", "明明"]

        if random.random() < 0.1:  # 10% 機率
            words = list(text)
            if len(words) > 5:
                insert_pos = random.randint(2, len(words) - 2)
                insert_word = random.choice(insertion_words)
                words.insert(insert_pos, insert_word)
                return ''.join(words)

        return text

    def minor_variation(self, text: str) -> str:
        """輕微變化（重複字符、標點變化等）"""
        result = text

        # 標點符號變化
        punct_map = {
            '。': ['！', '。。', '...'],
            '！': ['！！', '。', '！！！'],
            '？': ['？？', '。', '？！'],
            '，': ['，', '，，', '...']
        }

        for original, variations in punct_map.items():
            if original in result and random.random() < 0.1:
                replacement = random.choice(variations)
                result = result.replace(original, replacement, 1)

        return result

    def augment_single_text(self, text: str, label: int) -> List[str]:
        """對單個文本進行增強"""
        augmented_texts = []

        # 根據標籤調整增強次數（毒性文本增強更多）
        num_aug = self.augmentations_per_text
        if label == 1:  # 毒性文本
            num_aug += 1

        for i in range(num_aug):
            aug_text = text

            # 隨機應用增強技術
            techniques = [
                self.synonym_replacement,
                self.add_tone_words,
                self.sentence_transformation,
                self.random_insertion,
                self.minor_variation
            ]

            # 隨機選擇1-3個技術應用
            selected_techniques = random.sample(techniques, random.randint(1, 3))

            for technique in selected_techniques:
                aug_text = technique(aug_text)

            # 確保增強後的文本有效且有變化
            if aug_text.strip() and aug_text != text and len(aug_text) > 5:
                augmented_texts.append(aug_text.strip())

        return augmented_texts

    def augment_dataset(self, df: pd.DataFrame, target_ratio: float = 3.0) -> pd.DataFrame:
        """增強整個資料集"""
        logger.info(f"開始增強資料集，目標擴充比例: {target_ratio}x")

        original_size = len(df)
        target_size = int(original_size * target_ratio)
        needed_samples = target_size - original_size

        logger.info(f"原始大小: {original_size:,}")
        logger.info(f"目標大小: {target_size:,}")
        logger.info(f"需要生成: {needed_samples:,} 個樣本")

        # 分析標籤分佈
        label_counts = df['label'].value_counts()
        logger.info(f"原始標籤分佈: {dict(label_counts)}")

        augmented_rows = []

        # 為每個標籤按比例生成增強樣本
        for label in [0, 1]:
            label_df = df[df['label'] == label]
            label_ratio = len(label_df) / original_size
            label_needed = int(needed_samples * label_ratio)

            logger.info(f"標籤 {label}: 需要生成 {label_needed} 個樣本")

            generated = 0
            attempts = 0
            max_attempts = label_needed * 3  # 避免無限循環

            while generated < label_needed and attempts < max_attempts:
                attempts += 1

                # 隨機選擇一個原始樣本
                sample = label_df.sample(n=1).iloc[0]

                # 生成增強文本
                augmented_texts = self.augment_single_text(sample['TEXT'], sample['label'])

                for aug_text in augmented_texts:
                    if generated >= label_needed:
                        break

                    # 創建新的行
                    new_row = sample.copy()
                    new_row['TEXT'] = aug_text
                    augmented_rows.append(new_row)
                    generated += 1

            logger.info(f"標籤 {label}: 實際生成 {generated} 個樣本")

        # 合併原始和增強資料
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            final_df = pd.concat([df, augmented_df], ignore_index=True)
        else:
            final_df = df

        logger.info(f"增強完成！最終大小: {len(final_df):,}")
        logger.info(f"實際擴充比例: {len(final_df) / original_size:.2f}x")

        # 檢查最終標籤分佈
        final_label_counts = final_df['label'].value_counts()
        logger.info(f"最終標籤分佈: {dict(final_label_counts)}")

        return final_df


def main():
    parser = argparse.ArgumentParser(description="簡化版霸凌偵測資料集增強工具")
    parser.add_argument("--input", required=True, help="輸入 CSV 檔案路徑")
    parser.add_argument("--output", required=True, help="輸出 CSV 檔案路徑")
    parser.add_argument("--intensity", choices=["light", "medium", "heavy"],
                       default="medium", help="增強強度")
    parser.add_argument("--target-ratio", type=float, default=3.0,
                       help="目標擴充比例 (預設 3.0 倍)")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")

    args = parser.parse_args()

    # 設定隨機種子
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    logger.info(f"讀取資料: {args.input}")

    # 讀取資料
    df = pd.read_csv(args.input)
    logger.info(f"原始資料大小: {len(df):,} 筆")

    # 檢查必要欄位
    if 'TEXT' not in df.columns or 'label' not in df.columns:
        logger.error("資料集必須包含 'TEXT' 和 'label' 欄位")
        return

    # 初始化增強器
    augmenter = SimpleBullyingDataAugmenter(intensity=args.intensity)

    # 執行增強
    augmented_df = augmenter.augment_dataset(df, target_ratio=args.target_ratio)

    # 保存結果
    logger.info(f"保存增強後的資料: {args.output}")
    augmented_df.to_csv(args.output, index=False, encoding='utf-8')

    # 生成統計報告
    stats = {
        "original_size": len(df),
        "augmented_size": len(augmented_df),
        "expansion_ratio": len(augmented_df) / len(df),
        "original_distribution": df['label'].value_counts().to_dict(),
        "augmented_distribution": augmented_df['label'].value_counts().to_dict(),
        "parameters": {
            "intensity": args.intensity,
            "target_ratio": args.target_ratio,
            "seed": args.seed
        }
    }

    # 保存統計到 JSON
    stats_file = args.output.replace('.csv', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"統計報告保存至: {stats_file}")
    logger.info("資料增強完成！")


if __name__ == "__main__":
    main()