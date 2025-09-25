#!/usr/bin/env python3
"""
資料清理與正規化腳本
- 文字正規化：全形轉半形、空白折疊、URL/mention/hash處理、表情符號標記
- 繁簡轉換：使用OpenCC
- 敏感資訊去識別
- 產出對應mapping
"""

import re
import json
import argparse
import logging
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

import pandas as pd
from tqdm import tqdm

try:
    from opencc import OpenCC
    HAS_OPENCC = True
except ImportError:
    HAS_OPENCC = False
    print("Warning: OpenCC not installed. Install with: pip install opencc-python-reimplemented")

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextNormalizer:
    """文字正規化處理器"""

    def __init__(self,
                 convert_mode: str = None,
                 preserve_emoji: bool = True,
                 anonymize: bool = True):
        """
        初始化正規化器

        Args:
            convert_mode: 繁簡轉換模式 ('s2t', 't2s', 's2tw', 'tw2s', None)
            preserve_emoji: 是否保留表情符號
            anonymize: 是否進行去識別化
        """
        self.convert_mode = convert_mode
        self.preserve_emoji = preserve_emoji
        self.anonymize = anonymize

        # 初始化OpenCC轉換器
        self.converter = None
        if convert_mode and HAS_OPENCC:
            try:
                self.converter = OpenCC(convert_mode)
                logger.info(f"Initialized OpenCC with mode: {convert_mode}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenCC: {e}")

        # 編譯正則表達式
        self._compile_patterns()

        # 統計資訊
        self.stats = {
            'total_processed': 0,
            'urls_replaced': 0,
            'mentions_replaced': 0,
            'emails_replaced': 0,
            'phones_replaced': 0,
            'ids_replaced': 0,
            'emojis_marked': 0
        }

    def _compile_patterns(self):
        """編譯所有正則表達式模式"""
        # URL模式
        self.url_pattern = re.compile(
            r'(?:(?:https?|ftp):\/\/)?'  # 協議
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # 域名
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
            r'(?::\d+)?'  # 端口
            r'(?:/\S*)?',  # 路徑、查詢和片段
            re.IGNORECASE
        )

        # 社交媒體mention模式 (避免匹配email地址)
        self.mention_pattern = re.compile(
            r'@[\w\u4e00-\u9fff]+(?![\w.-]*\.[A-Za-z]{2,})'
        )

        # Hashtag模式
        self.hashtag_pattern = re.compile(r'#[\w\u4e00-\u9fff]+')

        # Email模式
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )

        # 電話號碼模式（支援多種格式）
        self.phone_patterns = [
            re.compile(r'\b(?:0?9\d{8})\b'),  # 台灣手機
            re.compile(r'\b(?:\d{2,4}[-\s]?\d{3,4}[-\s]?\d{3,4})\b'),  # 一般電話
            re.compile(r'\b(?:1[3-9]\d{9})\b'),  # 中國大陸手機
            re.compile(r'\b(?:\+?886[-\s]?9\d{8})\b'),  # 國際格式台灣手機
            re.compile(r'\b(?:\+?86[-\s]?1[3-9]\d{9})\b'),  # 國際格式大陸手機
        ]

        # 身分證號碼模式（簡化版，避免誤判）
        self.id_patterns = [
            re.compile(r'\b[A-Z][12]\d{8}\b'),  # 台灣身分證
            re.compile(r'\b\d{17}[\dXx]\b'),  # 大陸身分證
        ]

        # 表情符號模式（正確的Unicode範圍）
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # 表情
            "\U0001F300-\U0001F5FF"  # 符號和圖形
            "\U0001F680-\U0001F6FF"  # 交通和地圖
            "\U0001F1E0-\U0001F1FF"  # 國旗
            "\U00002702-\U000027B0"  # 雜項符號
            "\U000024C2-\U000024FF"  # 字母數字圓圈符號
            "\U0001F900-\U0001F9FF"  # 補充符號
            "]+",
            flags=re.UNICODE
        )

        # 多餘空白模式
        self.whitespace_pattern = re.compile(r'\s+')

    def fullwidth_to_halfwidth(self, text: str) -> str:
        """全形轉半形"""
        result = []
        for char in text:
            code = ord(char)
            # 全形空格轉半形空格
            if code == 0x3000:
                result.append(' ')
            # 全形字符（除空格）轉半形
            elif 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            else:
                result.append(char)
        return ''.join(result)

    def normalize_whitespace(self, text: str) -> str:
        """正規化空白字符"""
        # 將多個空白合併為單個空格
        text = self.whitespace_pattern.sub(' ', text)
        # 移除首尾空白
        return text.strip()

    def replace_urls(self, text: str) -> str:
        """替換URL"""
        def replace_url(match):
            self.stats['urls_replaced'] += 1
            return '[URL]'
        return self.url_pattern.sub(replace_url, text)

    def replace_mentions(self, text: str) -> str:
        """替換@mentions"""
        def replace_mention(match):
            self.stats['mentions_replaced'] += 1
            return '[MENTION]'
        return self.mention_pattern.sub(replace_mention, text)

    def replace_hashtags(self, text: str) -> str:
        """替換hashtags"""
        return self.hashtag_pattern.sub('[HASHTAG]', text)

    def mark_emojis(self, text: str) -> str:
        """標記表情符號"""
        if not self.preserve_emoji:
            return self.emoji_pattern.sub('[EMOJI]', text)

        # 保留表情但添加標記
        def mark_emoji(match):
            self.stats['emojis_marked'] += 1
            emoji = match.group()
            return f'[EMOJI:{emoji}]'

        return self.emoji_pattern.sub(mark_emoji, text)

    def anonymize_sensitive(self, text: str) -> str:
        """去識別化敏感資訊"""
        if not self.anonymize:
            return text

        # 替換Email
        def replace_email(match):
            self.stats['emails_replaced'] += 1
            return '[EMAIL]'
        text = self.email_pattern.sub(replace_email, text)

        # 替換電話號碼
        for pattern in self.phone_patterns:
            def replace_phone(match):
                self.stats['phones_replaced'] += 1
                return '[PHONE]'
            text = pattern.sub(replace_phone, text)

        # 替換身分證號碼
        for pattern in self.id_patterns:
            def replace_id(match):
                self.stats['ids_replaced'] += 1
                return '[ID]'
            text = pattern.sub(replace_id, text)

        return text

    def convert_chinese(self, text: str) -> str:
        """繁簡轉換"""
        if self.converter and self.convert_mode:
            try:
                return self.converter.convert(text)
            except Exception as e:
                logger.warning(f"Chinese conversion failed: {e}")
        return text

    def normalize(self, text: str) -> str:
        """
        完整的文字正規化流程

        Args:
            text: 輸入文字

        Returns:
            正規化後的文字
        """
        if not text or not isinstance(text, str):
            return text

        self.stats['total_processed'] += 1

        # 1. 全形轉半形
        text = self.fullwidth_to_halfwidth(text)

        # 2. 繁簡轉換
        text = self.convert_chinese(text)

        # 3. 去識別化（先處理敏感資訊）
        text = self.anonymize_sensitive(text)

        # 4. 替換URL
        text = self.replace_urls(text)

        # 5. 替換mentions
        text = self.replace_mentions(text)

        # 6. 替換hashtags
        text = self.replace_hashtags(text)

        # 7. 標記表情符號
        text = self.mark_emojis(text)

        # 8. 正規化空白
        text = self.normalize_whitespace(text)

        return text

    def get_stats(self) -> Dict:
        """獲取統計資訊"""
        return self.stats.copy()


class DatasetCleaner:
    """資料集清理器"""

    def __init__(self,
                 input_dir: str = "data/raw",
                 output_dir: str = "data/processed",
                 normalizer: Optional[TextNormalizer] = None):
        """
        初始化清理器

        Args:
            input_dir: 原始資料目錄
            output_dir: 處理後資料目錄
            normalizer: 文字正規化器
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.normalizer = normalizer or TextNormalizer()
        self.mapping_data = []

    def generate_id(self, text: str, index: int) -> str:
        """生成處理後的ID"""
        # 使用hash確保相同文字得到相同ID
        hash_obj = hashlib.md5(f"{text}_{index}".encode())
        return f"proc_{hash_obj.hexdigest()[:12]}"

    def clean_cold_dataset(self) -> Dict[str, Any]:
        """清理COLD資料集"""
        cold_dir = self.input_dir / "cold" / "COLDataset"
        if not cold_dir.exists():
            logger.warning(f"COLD dataset not found at {cold_dir}")
            return {}

        output_cold_dir = self.output_dir / "cold"
        output_cold_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for split in ['train', 'dev', 'test']:
            input_file = cold_dir / f"{split}.csv"
            if not input_file.exists():
                logger.warning(f"File not found: {input_file}")
                continue

            logger.info(f"Processing COLD {split} split...")

            # 讀取CSV
            try:
                df = pd.read_csv(input_file)

                # 處理每一筆資料
                processed_data = []
                for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"COLD {split}"):
                    # 假設文字欄位名稱為 'text' 或 'content'
                    text_col = None
                    for col in ['text', 'content', 'comment', 'sentence']:
                        if col in df.columns:
                            text_col = col
                            break

                    if text_col is None:
                        # 如果沒有找到，使用第一個字串類型的欄位
                        for col in df.columns:
                            if df[col].dtype == 'object':
                                text_col = col
                                break

                    if text_col:
                        original_text = str(row[text_col])
                        processed_text = self.normalizer.normalize(original_text)

                        # 生成ID
                        raw_id = f"cold_{split}_{idx}"
                        proc_id = self.generate_id(processed_text, idx)

                        # 保存對應關係
                        self.mapping_data.append({
                            'raw_id': raw_id,
                            'processed_id': proc_id,
                            'dataset': 'cold',
                            'split': split,
                            'original_hash': hashlib.md5(original_text.encode()).hexdigest()
                        })

                        # 建立處理後的資料
                        processed_row = row.copy()
                        processed_row[text_col] = processed_text
                        processed_row['processed_id'] = proc_id
                        processed_row['raw_id'] = raw_id
                        processed_data.append(processed_row)

                # 儲存處理後的資料
                processed_df = pd.DataFrame(processed_data)
                output_file = output_cold_dir / f"{split}_processed.csv"
                processed_df.to_csv(output_file, index=False, encoding='utf-8')

                results[f"cold_{split}"] = {
                    'input_file': str(input_file),
                    'output_file': str(output_file),
                    'total_samples': len(processed_df),
                    'columns': list(processed_df.columns)
                }

                logger.info(f"Saved {len(processed_df)} processed samples to {output_file}")

            except Exception as e:
                logger.error(f"Error processing {input_file}: {e}")

        return results

    def clean_json_dataset(
        self,
        dataset_name: str,
        text_field: str = 'text'
    ) -> Dict[str, Any]:
        """清理JSON格式的資料集"""
        dataset_dir = self.input_dir / dataset_name
        if not dataset_dir.exists():
            logger.warning(f"{dataset_name} dataset not found at {dataset_dir}")
            return {}

        output_dataset_dir = self.output_dir / dataset_name
        output_dataset_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # 處理所有JSON檔案
        for json_file in dataset_dir.glob("*.json"):
            logger.info(f"Processing {json_file.name}...")

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                processed_data = []

                # 處理資料（可能是列表或字典）
                if isinstance(data, list):
                    for idx, item in enumerate(
                        tqdm(data,
                        desc=json_file.stem)
                    ):
                        if text_field in item:
                            original_text = str(item[text_field])
                            processed_text = self.normalizer.normalize(original_text)

                            raw_id = f"{dataset_name}_{json_file.stem}_{idx}"
                            proc_id = self.generate_id(processed_text, idx)

                            self.mapping_data.append({
                                'raw_id': raw_id,
                                'processed_id': proc_id,
                                'dataset': dataset_name,
                                'file': json_file.stem,
                                'original_hash': hashlib.md5(original_text.encode()).hexdigest()
                            })

                            processed_item = item.copy()
                            processed_item[text_field] = processed_text
                            processed_item['processed_id'] = proc_id
                            processed_item['raw_id'] = raw_id
                            processed_data.append(processed_item)

                # 儲存處理後的資料
                output_file = output_dataset_dir / f"{json_file.stem}_processed.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)

                results[f"{dataset_name}_{json_file.stem}"] = {
                    'input_file': str(json_file),
                    'output_file': str(output_file),
                    'total_samples': len(processed_data)
                }

                logger.info(f"Saved {len(processed_data)} processed samples to {output_file}")

            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")

        return results

    def save_mapping(self):
        """儲存ID對應關係"""
        mapping_file = self.output_dir / "id_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.mapping_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(self.mapping_data)} ID mappings to {mapping_file}")

        # 也儲存為CSV以便查詢
        mapping_df = pd.DataFrame(self.mapping_data)
        csv_file = self.output_dir / "id_mapping.csv"
        mapping_df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"Also saved mappings as CSV to {csv_file}")

    def clean_all(self) -> Dict[str, Any]:
        """清理所有可用的資料集"""
        all_results = {}

        # 清理COLD資料集
        cold_results = self.clean_cold_dataset()
        all_results.update(cold_results)

        # 清理其他JSON格式的資料集
        for dataset_name in ['chnsenticorp', 'sccd', 'chnci']:
            dataset_results = self.clean_json_dataset(dataset_name)
            all_results.update(dataset_results)

        # 儲存mapping
        self.save_mapping()

        # 儲存處理統計
        stats_file = self.output_dir / "processing_stats.json"
        stats = {
            'timestamp': datetime.now().isoformat(),
            'datasets_processed': list(all_results.keys()),
            'total_mappings': len(self.mapping_data),
            'normalizer_stats': self.normalizer.get_stats(),
            'results': all_results
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"Processing complete. Stats saved to {stats_file}")

        return all_results


def main():
    parser = argparse.ArgumentParser(description="清理與正規化資料集")
    parser.add_argument(
        "--input-dir",
        default="data/raw",
        help="原始資料目錄"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="輸出目錄"
    )
    parser.add_argument(
        "--convert",
        choices=['s2t', 't2s', 's2tw', 'tw2s'],
        help="繁簡轉換模式"
    )
    parser.add_argument(
        "--no-anonymize",
        action="store_true",
        help="不進行去識別化"
    )
    parser.add_argument(
        "--no-emoji",
        action="store_true",
        help="移除表情符號而非標記"
    )
    parser.add_argument(
        "--dataset",
        choices=['cold', 'chnsenticorp', 'all'],
        default='all',
        help="要處理的資料集"
    )

    args = parser.parse_args()

    # 初始化正規化器
    normalizer = TextNormalizer(
        convert_mode=args.convert,
        preserve_emoji=not args.no_emoji,
        anonymize=not args.no_anonymize
    )

    # 初始化清理器
    cleaner = DatasetCleaner(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        normalizer=normalizer
    )

    # 執行清理
    if args.dataset == 'all':
        results = cleaner.clean_all()
    elif args.dataset == 'cold':
        results = cleaner.clean_cold_dataset()
        cleaner.save_mapping()
    else:
        results = cleaner.clean_json_dataset(args.dataset)
        cleaner.save_mapping()

    # 顯示結果摘要
    print("\n" + "="*60)
    print("Data Cleaning Summary")
    print("="*60)

    for dataset, info in results.items():
        print(f"\n[{dataset}]")
        print(f"  Samples: {info.get('total_samples', 'N/A')}")
        print(f"  Output: {info.get('output_file', 'N/A')}")

    print("\n" + "="*60)
    print("Normalizer Statistics")
    print("="*60)
    stats = normalizer.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("="*60)


if __name__ == "__main__":
    main()
