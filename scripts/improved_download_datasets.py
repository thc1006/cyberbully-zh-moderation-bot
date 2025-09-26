#!/usr/bin/env python3
"""
改進的資料集下載腳本
支援完整的 ChnSentiCorp、DMSC v2、NTUSD 下載與驗證
基於研究報告的建議實作
"""

import os
import sys
import json
import hashlib
import argparse
import logging
import zipfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import subprocess

import requests
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 更新的資料集元資料
DATASETS_META = {
    "chnsenticorp": {
        "name": "ChnSentiCorp",
        "description": "中文情感分析資料集（正負二元）",
        "sources": {
            "parquet": {
                "train": "https://huggingface.co/datasets/seamew/ChnSentiCorp/resolve/main/data/train-00000-of-00001.parquet",
                "validation": "https://huggingface.co/datasets/seamew/ChnSentiCorp/resolve/main/data/validation-00000-of-00001.parquet",
                "test": "https://huggingface.co/datasets/seamew/ChnSentiCorp/resolve/main/data/test-00000-of-00001.parquet"
            },
            "csv_backup": {
                "train": "https://raw.githubusercontent.com/fate233/sentiment_corpus/master/train.csv",
                "test": "https://raw.githubusercontent.com/fate233/sentiment_corpus/master/test.csv"
            }
        },
        "validation": {
            "min_samples": 7000,
            "required_columns": ["label", "text"],
            "expected_splits": ["train", "validation", "test"]
        }
    },
    "dmsc": {
        "name": "DMSC v2 (豆瓣電影短評)",
        "description": "豆瓣電影短評資料集，含評分",
        "sources": {
            "kaggle_zip": "https://github.com/candlewill/dmsc-v2/releases/download/v2.0/dmsc_v2.zip",
            "github_repo": "https://github.com/yhpc/DMSC"
        },
        "validation": {
            "expected_hash": "4bdb50bb400d23a8eebcc11b27195fc4",
            "min_samples": 2000000,
            "required_columns": ["ID", "Movie_Name_EN", "Movie_Name_CN", "Star", "Comment"],
            "file_size_mb": 400
        }
    },
    "ntusd": {
        "name": "NTUSD (臺大情感字典)",
        "description": "繁體中文情感詞典（正負極性）",
        "sources": {
            "complete": {
                "positive": "https://raw.githubusercontent.com/sweslo17/chinese_sentiment/master/dict/ntusd-positive.txt",
                "negative": "https://raw.githubusercontent.com/sweslo17/chinese_sentiment/master/dict/ntusd-negative.txt"
            },
            "official": {
                "repo": "https://github.com/ntunlplab/NTUSD.git",
                "positive_file": "data/正面詞無重複_9365詞.txt",
                "negative_file": "data/負面詞無重複_11230詞.txt"
            }
        },
        "validation": {
            "positive_min": 2500,
            "negative_min": 7500,
            "encoding": "utf-8",
            "total_expected": 11000
        }
    }
}


class ImprovedDatasetDownloader:
    """改進的資料集下載器"""

    def __init__(self, base_dir: str = "./data/raw", max_retries: int = 3):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CyberPuppy-Dataset-Downloader/2.0'
        })

        # 設定 proxy（如果有）
        if os.getenv("HTTP_PROXY"):
            self.session.proxies.update({
                'http': os.getenv("HTTP_PROXY"),
                'https': os.getenv("HTTPS_PROXY", os.getenv("HTTP_PROXY"))
            })

    def download_file_with_progress(self, url: str, dest_path: Path,
                                   expected_hash: Optional[str] = None,
                                   resume: bool = True) -> bool:
        """
        下載檔案，支援進度條、斷點續傳與雜湊驗證
        """
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dest_path.with_suffix('.tmp')

        headers = {}
        mode = 'wb'
        resume_pos = 0

        # 斷點續傳
        if resume and temp_path.exists():
            resume_pos = temp_path.stat().st_size
            headers['Range'] = f'bytes={resume_pos}-'
            mode = 'ab'
            logger.info(f"Resuming download from byte {resume_pos}")

        try:
            response = self.session.get(url, headers=headers, stream=True,
                                       timeout=int(os.getenv("DOWNLOAD_TIMEOUT", 300)))
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            if resume and response.status_code == 206:
                total_size += resume_pos

            with open(temp_path, mode) as f:
                with tqdm(total=total_size, initial=resume_pos,
                         unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # 驗證雜湊
            if expected_hash:
                if not self.verify_hash(temp_path, expected_hash):
                    logger.error(f"Hash verification failed for {dest_path.name}")
                    return False

            # 移動到最終位置
            shutil.move(str(temp_path), str(dest_path))
            logger.info(f"Successfully downloaded {dest_path.name}")
            return True

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def verify_hash(self, file_path: Path, expected_hash: str) -> bool:
        """驗證檔案雜湊"""
        hash_obj = hashlib.md5() if len(expected_hash) == 32 else hashlib.sha256()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_obj.update(chunk)

        actual_hash = hash_obj.hexdigest()
        return actual_hash == expected_hash

    def download_chnsenticorp_improved(self) -> bool:
        """改進的 ChnSentiCorp 下載方法"""
        logger.info("Downloading ChnSentiCorp dataset...")
        meta = DATASETS_META["chnsenticorp"]
        dest_dir = self.base_dir / "chnsenticorp"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 方法一：嘗試 Parquet 檔案
        logger.info("Attempting to download Parquet files...")
        try:
            for split, url in meta["sources"]["parquet"].items():
                parquet_file = dest_dir / f"{split}.parquet"
                if self.download_file_with_progress(url, parquet_file):
                    # 轉換為 JSON
                    df = pd.read_parquet(parquet_file)
                    json_file = dest_dir / f"{split}.json"
                    df.to_json(json_file, orient='records', force_ascii=False, indent=2)
                    logger.info(f"Converted {split}: {len(df)} samples")
                else:
                    raise Exception(f"Failed to download {split}")

            self._save_metadata(dest_dir, meta, "parquet")
            return True

        except Exception as e:
            logger.warning(f"Parquet download failed: {e}")

        # 方法二：備用 CSV 來源
        logger.info("Trying backup CSV sources...")
        try:
            for split, url in meta["sources"]["csv_backup"].items():
                csv_file = dest_dir / f"{split}.csv"
                if self.download_file_with_progress(url, csv_file):
                    # 轉換為 JSON
                    df = pd.read_csv(csv_file)
                    json_file = dest_dir / f"{split}.json"
                    df.to_json(json_file, orient='records', force_ascii=False, indent=2)
                    logger.info(f"Downloaded {split}: {len(df)} samples")
                else:
                    return False

            self._save_metadata(dest_dir, meta, "csv_backup")
            return True

        except Exception as e:
            logger.error(f"All ChnSentiCorp download methods failed: {e}")
            return False

    def download_dmsc_improved(self) -> bool:
        """改進的 DMSC v2 下載方法"""
        logger.info("Downloading DMSC v2 dataset...")
        meta = DATASETS_META["dmsc"]
        dest_dir = self.base_dir / "dmsc"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 檢查現有檔案
        csv_file = dest_dir / "DMSC.csv"
        if csv_file.exists():
            if self.validate_dmsc(csv_file, meta["validation"]):
                logger.info("DMSC dataset already exists and is valid")
                return True
            else:
                logger.warning("Existing DMSC file is invalid, re-downloading...")

        # 下載 ZIP 檔案
        zip_url = meta["sources"]["kaggle_zip"]
        zip_file = dest_dir / "dmsc_v2.zip"

        if not self.download_file_with_progress(zip_url, zip_file):
            return False

        # 解壓縮
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            logger.info("Successfully extracted DMSC dataset")

            # 驗證解壓後的檔案
            if csv_file.exists() and self.validate_dmsc(csv_file, meta["validation"]):
                self._save_metadata(dest_dir, meta, "kaggle_zip")
                return True
            else:
                logger.error("Extracted DMSC file validation failed")
                return False

        except Exception as e:
            logger.error(f"Failed to extract DMSC dataset: {e}")
            return False

    def download_ntusd_improved(self) -> bool:
        """改進的 NTUSD 下載方法"""
        logger.info("Downloading NTUSD dataset...")
        meta = DATASETS_META["ntusd"]
        dest_dir = self.base_dir / "ntusd"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 使用完整的詞典來源
        complete_sources = meta["sources"]["complete"]

        success = True
        for sentiment, url in complete_sources.items():
            filename = f"{sentiment}.txt"
            dest_file = dest_dir / filename

            logger.info(f"Downloading {sentiment} dictionary...")
            if not self.download_file_with_progress(url, dest_file):
                success = False
                continue

        if not success:
            return False

        # 驗證詞典
        if self.validate_ntusd(dest_dir, meta["validation"]):
            # 創建合併檔案
            self._create_merged_ntusd(dest_dir)
            self._save_metadata(dest_dir, meta, "complete")
            return True
        else:
            logger.error("NTUSD validation failed")
            return False

    def validate_dmsc(self, csv_file: Path, validation: Dict) -> bool:
        """驗證 DMSC 檔案"""
        try:
            # 檢查檔案大小
            file_size_mb = csv_file.stat().st_size / (1024 * 1024)
            if file_size_mb < validation["file_size_mb"] * 0.9:  # 允許10%誤差
                logger.warning(f"DMSC file size too small: {file_size_mb:.1f} MB")
                return False

            # 檢查雜湊值
            if "expected_hash" in validation:
                if not self.verify_hash(csv_file, validation["expected_hash"]):
                    logger.warning("DMSC hash verification failed")
                    return False

            # 檢查行數和欄位
            chunk = pd.read_csv(csv_file, nrows=1000)
            if not all(col in chunk.columns for col in validation["required_columns"]):
                logger.warning("DMSC missing required columns")
                return False

            logger.info("DMSC validation passed")
            return True

        except Exception as e:
            logger.error(f"DMSC validation error: {e}")
            return False

    def validate_ntusd(self, dest_dir: Path, validation: Dict) -> bool:
        """驗證 NTUSD 詞典"""
        try:
            pos_file = dest_dir / "positive.txt"
            neg_file = dest_dir / "negative.txt"

            if not pos_file.exists() or not neg_file.exists():
                logger.error("NTUSD files not found")
                return False

            # 計算詞彙數
            pos_content = pos_file.read_text(encoding=validation["encoding"]).strip()
            neg_content = neg_file.read_text(encoding=validation["encoding"]).strip()

            pos_count = len([line for line in pos_content.split('\n') if line.strip()])
            neg_count = len([line for line in neg_content.split('\n') if line.strip()])

            logger.info(f"NTUSD word counts: {pos_count} positive, {neg_count} negative")

            # 驗證最小詞彙數
            if pos_count < validation["positive_min"]:
                logger.warning(f"Too few positive words: {pos_count}")
                return False

            if neg_count < validation["negative_min"]:
                logger.warning(f"Too few negative words: {neg_count}")
                return False

            logger.info("NTUSD validation passed")
            return True

        except Exception as e:
            logger.error(f"NTUSD validation error: {e}")
            return False

    def _create_merged_ntusd(self, dest_dir: Path):
        """創建合併的 NTUSD 詞典檔案"""
        try:
            pos_file = dest_dir / "positive.txt"
            neg_file = dest_dir / "negative.txt"

            # 讀取並去重
            pos_words = set(line.strip() for line in pos_file.read_text(encoding='utf-8').split('\n') if line.strip())
            neg_words = set(line.strip() for line in neg_file.read_text(encoding='utf-8').split('\n') if line.strip())

            # 保存去重後的檔案
            merged_pos = dest_dir / "merged_positive.txt"
            merged_neg = dest_dir / "merged_negative.txt"

            merged_pos.write_text('\n'.join(sorted(pos_words)), encoding='utf-8')
            merged_neg.write_text('\n'.join(sorted(neg_words)), encoding='utf-8')

            logger.info(f"Created merged dictionaries: {len(pos_words)} positive, {len(neg_words)} negative")

        except Exception as e:
            logger.error(f"Failed to create merged NTUSD files: {e}")

    def _save_metadata(self, dest_dir: Path, meta: Dict, method: str):
        """保存資料集元資料"""
        metadata = {
            "name": meta["name"],
            "description": meta["description"],
            "download_method": method,
            "download_time": pd.Timestamp.now().isoformat(),
            "validation": meta.get("validation", {})
        }

        metadata_file = dest_dir / "download_metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')

    def validate_all_datasets(self) -> Dict[str, bool]:
        """驗證所有已下載的資料集"""
        results = {}

        for dataset_name, meta in DATASETS_META.items():
            dataset_dir = self.base_dir / dataset_name
            if not dataset_dir.exists():
                results[dataset_name] = False
                continue

            if dataset_name == "chnsenticorp":
                # 檢查是否有任何分割檔案
                json_files = list(dataset_dir.glob("*.json"))
                results[dataset_name] = len(json_files) >= 2
            elif dataset_name == "dmsc":
                csv_file = dataset_dir / "DMSC.csv"
                results[dataset_name] = csv_file.exists() and self.validate_dmsc(csv_file, meta["validation"])
            elif dataset_name == "ntusd":
                results[dataset_name] = self.validate_ntusd(dataset_dir, meta["validation"])

        return results

    def download_all(self, datasets: Optional[List[str]] = None) -> Dict[str, bool]:
        """下載所有或指定的資料集"""
        if datasets is None:
            datasets = list(DATASETS_META.keys())

        results = {}

        for dataset_name in datasets:
            if dataset_name not in DATASETS_META:
                logger.warning(f"Unknown dataset: {dataset_name}")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {DATASETS_META[dataset_name]['name']}")
            logger.info(f"{'='*60}")

            # 執行對應的下載方法
            if dataset_name == "chnsenticorp":
                results[dataset_name] = self.download_chnsenticorp_improved()
            elif dataset_name == "dmsc":
                results[dataset_name] = self.download_dmsc_improved()
            elif dataset_name == "ntusd":
                results[dataset_name] = self.download_ntusd_improved()
            else:
                logger.warning(f"No improved download method for {dataset_name}")
                results[dataset_name] = False

        # 總結
        self._print_summary(results)
        return results

    def _print_summary(self, results: Dict[str, bool]):
        """列印下載總結"""
        print("\n" + "="*60)
        print("Download Summary")
        print("="*60)

        for dataset_name, success in results.items():
            status = "[SUCCESS]" if success else "[FAILED]"
            print(f"{DATASETS_META[dataset_name]['name']}: {status}")

        # 驗證報告
        print("\nValidation Report:")
        validation_results = self.validate_all_datasets()
        for dataset_name, valid in validation_results.items():
            status = "[VALID]" if valid else "[INVALID]"
            print(f"  {dataset_name}: {status}")

        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="改進的 CyberPuppy 資料集下載器")
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=list(DATASETS_META.keys()) + ["all"],
        default=["all"],
        help="要下載的資料集"
    )
    parser.add_argument(
        "--output-dir",
        default="./data/raw",
        help="輸出目錄"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只驗證現有資料集，不下載"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="最大重試次數"
    )

    args = parser.parse_args()

    # 初始化下載器
    downloader = ImprovedDatasetDownloader(
        base_dir=args.output_dir,
        max_retries=args.max_retries
    )

    # 處理資料集清單
    datasets = None if "all" in args.dataset else args.dataset

    if args.validate_only:
        # 只驗證
        results = downloader.validate_all_datasets()
        downloader._print_summary({})  # 只顯示驗證結果
    else:
        # 執行下載
        results = downloader.download_all(datasets)

        # 檢查是否有失敗
        if results and not all(results.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()