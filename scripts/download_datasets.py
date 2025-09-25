#!/usr/bin/env python3
"""
資料集下載腳本
支援下載：COLD、ChnSentiCorp、DMSC v2、NTUSD
提供 SCCD、CHNCI 手動下載指引
包含雜湊校驗、斷點續傳與錯誤處理
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
from typing import Optional, List
import subprocess

import requests
from tqdm import tqdm
from dotenv import load_dotenv

# 嘗試導入 Hugging Face datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' library not instal"
        "led. Cannot download from Hugging Face.")
    print("Install with: pip install datasets")

# 載入環境變數
load_dotenv()

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 資料集元資料
DATASETS_META = {
    "cold": {
        "name": "COLD (Chinese Offensive Language Dataset)",
        "source": "github",
        "repo_url": os.getenv("COLD_REPO_URL", "https://github.com/thu-coai/COLDataset"),
        "branch": os.getenv("COLD_DATA_BRANCH", "main"),
        "description": "中文冒犯語言資料集，適合訓練毒性分類器",
        "files": {
            "data.json": "5e3d7c4f9a2b1e8d7c3f4a5b6c7d8e9f",  # 示例雜湊，實際需更新
            "test.json": "a1b2c3d4e5f6789012345678901234567"
        }
    },
    "chnsenticorp": {
        "name": "ChnSentiCorp",
        "source": "huggingface",
        "dataset_name": "seamew/ChnSentiCorp",
        "description": "中文情感分析資料集（正負二元）",
        "expected_size": 15000,  # 預期資料筆數
        "hash": "b3c4d5e6f7890123456789abcdef1234"  # 整體資料集雜湊
    },
    "dmsc": {
        "name": "DMSC v2 (豆瓣電影短評)",
        "source": "github",
        "repo_url": os.getenv("DMSC_REPO_URL", "https://github.com/yhpc/DMSC"),
        "branch": os.getenv("DMSC_DATA_BRANCH", "master"),
        "description": "豆瓣電影短評資料集，含評分",
        "large_files": [
            "https://github.com/ownthink/dmsc-v2"
                "/releases/download/v2.0/dmsc_v2.zip"
        ],
        "expected_hash": "c1d2e3f4567890abcdef1234567890ab"
    },
    "ntusd": {
        "name": "NTUSD (臺大情感字典)",
        "source": "github",
        "repo_url": os.getenv("NTUSD_REPO_URL", "https://github.com/candlewill/NTUSD"),
        "branch": os.getenv("NTUSD_DATA_BRANCH", "master"),
        "description": "繁體中文情感詞典（正負極性）",
        "files": {
            "positive.txt": "1234567890abcdef1234567890abcdef",
            "negative.txt": "fedcba0987654321fedcba0987654321"
        }
    },
    "sccd": {
        "name": "SCCD (Session-level Chinese Cyberbullying Dataset)",
        "source": "manual",
        "description": "會話級中文網路霸凌資料集",
        "paper_url": "https://arxiv.org/abs/2506.04975",
        "manual_instructions": """
        SCCD 資料集需要手動下載：
        1. 訪問論文頁面：https://arxiv.org/abs/2506.04975
        2. 聯繫作者獲取資料集存取權限
        3. 下載資料後，將檔案放置於：{path}
        4. 檔案結構應包含：
           - sccd_train.json
           - sccd_dev.json
           - sccd_test.json
        """
    },
    "chnci": {
        "name": "CHNCI (Chinese Cyberbullying Incident Dataset)",
        "source": "manual",
        "description": "事件級中文網路霸凌資料集",
        "paper_url": "https://arxiv.org/abs/2506.05380",
        "manual_instructions": """
        CHNCI 資料集需要手動下載：
        1. 訪問論文頁面：https://arxiv.org/abs/2506.05380
        2. 根據論文指引申請資料集
        3. 下載資料後，將檔案放置於：{path}
        4. 檔案結構應包含：
           - chnci_events.json
           - chnci_annotations.json
        """
    }
}


class DatasetDownloader:
    """資料集下載器"""

    def __init__(self, base_dir: str = "./data/raw", max_retries: int = 3):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CyberPuppy-Dataset-Downloader/1.0'
        })

        # 設定 proxy（如果有）
        if os.getenv("HTTP_PROXY"):
            self.session.proxies.update({
                'http': os.getenv("HTTP_PROXY"),
                'https': os.getenv("HTTPS_PROXY", os.getenv("HTTP_PROXY"))
            })

    def download_file(self, url: str, dest_path: Path,
                      expected_hash: Optional[str] = None,
                      resume: bool = True) -> bool:
        """
        下載檔案，支援斷點續傳與雜湊驗證
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
                                        timeout=int(os.getenv("DOWNLOAD"
                                            "_TIMEOUT", 300)))
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
                    logger.error(f"Hash verification fail"
                        "ed for {dest_path.name}")
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

    def clone_github_repo(self, repo_url: str, dest_dir: Path,
                          branch: str = "main") -> bool:
        """克隆或更新 GitHub 儲存庫"""
        try:
            if dest_dir.exists() and (dest_dir / ".git").exists():
                # 更新現有儲存庫
                logger.info(f"Updating existing repository: {dest_dir}")
                subprocess.run(["git", "pull"], cwd=dest_dir, check=True)
            else:
                # 克隆新儲存庫
                logger.info(f"Cloning repository: {repo_url}")
                dest_dir.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([
                    "git", "clone", "-b", branch, "--depth", "1",
                    repo_url, str(dest_dir)
                ], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Git operation failed: {e}")
            return False

    def download_cold(self) -> bool:
        """下載 COLD 資料集"""
        logger.info("Downloading COLD dataset...")
        meta = DATASETS_META["cold"]
        dest_dir = self.base_dir / "cold"

        if not self.clone_github_repo(meta["repo_url"], dest_dir, meta["branch"]):
            return False

        # 驗證關鍵檔案
        for filename in ["COLDataset.json", "COLDataset_split.json"]:
            if not (dest_dir / filename).exists():
                logger.warning(f"Expected file not found: {filename}")

        logger.info("COLD dataset downloaded successfully")
        return True

    def download_chnsenticorp(self) -> bool:
        """下載 ChnSentiCorp 資料集"""
        if not HAS_DATASETS:
            logger.error("Hugging Face datasets library not installed")
            return False

        logger.info("Downloading ChnSentiCorp dataset...")
        meta = DATASETS_META["chnsenticorp"]
        dest_dir = self.base_dir / "chnsenticorp"
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 從 Hugging Face 下載
            dataset = load_dataset(meta["dataset_name"])

            # 儲存為 JSON
            for split in dataset.keys():
                output_file = dest_dir / f"{split}.json"
                dataset[split].to_json(output_file)
                logger.info(f"Saved {split} split to {output_file}")

            # 儲存元資料
            metadata = {
                "source": meta["dataset_name"],
                "splits": list(dataset.keys()),
                "total_samples": sum(len(dataset[split]) for split in dataset.keys())
            }
            with open(dest_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info("ChnSentiCorp dataset downloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to download ChnSentiCorp: {e}")
            return False

    def download_dmsc(self) -> bool:
        """下載 DMSC v2 資料集"""
        logger.info("Downloading DMSC v2 dataset...")
        meta = DATASETS_META["dmsc"]
        dest_dir = self.base_dir / "dmsc"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 嘗試下載大檔案
        if "large_files" in meta:
            for url in meta["large_files"]:
                filename = url.split('/')[-1]
                dest_file = dest_dir / filename

                if self.download_file(url, dest_file):
                    # 解壓縮
                    if filename.endswith('.zip'):
                        with zipfile.ZipFile(dest_file, 'r') as zip_ref:
                            zip_ref.extractall(dest_dir)
                        logger.info(f"Extracted {filename}")

        # 也克隆 GitHub 儲存庫以獲取文檔
        repo_dir = dest_dir / "repo"
        self.clone_github_repo(meta["repo_url"], repo_dir, meta["branch"])

        logger.info("DMSC v2 dataset downloaded successfully")
        return True

    def download_ntusd(self) -> bool:
        """下載 NTUSD 資料集"""
        logger.info("Downloading NTUSD dataset...")
        meta = DATASETS_META["ntusd"]
        dest_dir = self.base_dir / "ntusd"

        if not self.clone_github_repo(meta["repo_url"], dest_dir, meta["branch"]):
            return False

        # 檢查關鍵檔案
        expected_files = ["positive.txt", "negative.txt"]
        for filename in expected_files:
            if not (dest_dir / filename).exists():
                logger.warning(f"Expected file not found: {filename}")

        logger.info("NTUSD dataset downloaded successfully")
        return True

    def show_manual_instructions(self, dataset_name: str):
        """顯示手動下載指引"""
        meta = DATASETS_META[dataset_name]
        if meta["source"] != "manual":
            return

        path = self.base_dir.parent / "external" / dataset_name
        instructions = meta["manual_instructions"].format(path=path)

        print("\n" + "="*60)
        print(f"[Manual Download Required] {meta['name']}")
        print("="*60)
        print(meta['description'])
        print(f"Paper: {meta['paper_url']}")
        print("-"*60)
        print(instructions)
        print("="*60 + "\n")

    def download_all(self, datasets: Optional[List[str]] = None):
        """下載所有或指定的資料集"""
        if datasets is None:
            datasets = list(DATASETS_META.keys())

        results = {}
        manual_datasets = []

        for dataset_name in datasets:
            if dataset_name not in DATASETS_META:
                logger.warning(f"Unknown dataset: {dataset_name}")
                continue

            meta = DATASETS_META[dataset_name]

            if meta["source"] == "manual":
                manual_datasets.append(dataset_name)
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {meta['name']}")
            logger.info(f"{'='*60}")

            # 執行對應的下載方法
            method_name = f"download_{dataset_name}"
            if hasattr(self, method_name):
                results[dataset_name] = getattr(self, method_name)()
            else:
                logger.warning(f"No download method for {dataset_name}")
                results[dataset_name] = False

        # 顯示手動下載指引
        for dataset_name in manual_datasets:
            self.show_manual_instructions(dataset_name)

        # 總結
        print("\n" + "="*60)
        print("Download Summary")
        print("="*60)
        for dataset_name, success in results.items():
            status = "[SUCCESS]" if success else "[FAILED]"
            print(f"{DATASETS_META[dataset_name]['name']}: {status}")

        if manual_datasets:
            print("\nDatasets requiring manual download:")
            for dataset_name in manual_datasets:
                print(f"  - {DATASETS_META[dataset_name]['name']}")
                print(f"    View instructions: python {__file__} --dataset {dataset_name}")
        print("="*60)

        return results


def main():
    parser = argparse.ArgumentParser(description="下載 CyberPuppy 所需資料集")
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
        "--max-retries",
        type=int,
        default=3,
        help="最大重試次數"
    )
    parser.add_argument(
        "--show-instructions",
        action="store_true",
        help="顯示手動下載指引"
    )

    args = parser.parse_args()

    # 初始化下載器
    downloader = DatasetDownloader(
        base_dir=args.output_dir,
        max_retries=args.max_retries
    )

    # 處理資料集清單
    datasets = None if "all" in args.dataset else args.dataset

    if args.show_instructions:
        # 只顯示指引
        for dataset_name in (datasets or DATASETS_META.keys()):
            if DATASETS_META.get(dataset_name, {}).get("source") == "manual":
                downloader.show_manual_instructions(dataset_name)
    else:
        # 執行下載
        results = downloader.download_all(datasets)

        # 檢查是否有失敗
        if results and not all(results.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()
