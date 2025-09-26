#!/usr/bin/env python3
"""
資料集修復腳本
解決 ChnSentiCorp、DMSC v2、NTUSD 的下載問題
"""

import os
import sys
import json
import zipfile
import logging
from pathlib import Path
from typing import Dict, Optional
import subprocess

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetFixer:
    """資料集修復器"""

    def __init__(self, base_dir: str = "./data/raw"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CyberPuppy-Dataset-Fixer/1.0'
        })

    def fix_chnsenticorp(self) -> bool:
        """
        修復 ChnSentiCorp 資料集
        使用直接 HTTP 下載而非 Hugging Face datasets 庫
        """
        logger.info("="*60)
        logger.info("修復 ChnSentiCorp 資料集")
        logger.info("="*60)

        dest_dir = self.base_dir / "chnsenticorp"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 方案 1: 從 Hugging Face Hub 直接下載 parquet 檔案
        hf_urls = {
            "train": "https://huggingface.co/datasets/seamew/ChnSentiCorp/resolve/main/train.parquet",
            "validation": "https://huggingface.co/datasets/seamew/ChnSentiCorp/resolve/main/validation.parquet",
            "test": "https://huggingface.co/datasets/seamew/ChnSentiCorp/resolve/main/test.parquet"
        }

        success_count = 0

        for split, url in hf_urls.items():
            output_file = dest_dir / f"{split}.parquet"

            if output_file.exists():
                logger.info(f"✓ {split}.parquet 已存在，跳過下載")
                success_count += 1
                continue

            try:
                logger.info(f"下載 {split} split...")
                response = self.session.get(url, stream=True, timeout=120)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))

                with open(output_file, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=split) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                logger.info(f"✓ {split}.parquet 下載成功")
                success_count += 1

            except Exception as e:
                logger.error(f"✗ 下載 {split} 失敗: {e}")

        # 轉換 parquet 到 JSON（需要 pandas 和 pyarrow）
        try:
            import pandas as pd

            logger.info("\n轉換 parquet 到 JSON 格式...")

            for split in ["train", "validation", "test"]:
                parquet_file = dest_dir / f"{split}.parquet"
                json_file = dest_dir / f"{split}.json"

                if not parquet_file.exists():
                    continue

                df = pd.read_parquet(parquet_file)
                df.to_json(json_file, orient='records', force_ascii=False, indent=2)
                logger.info(f"✓ {split}.json 轉換完成 ({len(df)} 筆資料)")

            # 儲存元資料
            metadata = {
                "source": "seamew/ChnSentiCorp",
                "download_method": "direct_http",
                "splits": ["train", "validation", "test"],
                "format": "parquet+json",
                "description": "中文情感分析資料集（正負二元）"
            }

            with open(dest_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"\n✓ ChnSentiCorp 修復完成！")
            return success_count == 3

        except ImportError:
            logger.warning("⚠ pandas 或 pyarrow 未安裝，無法轉換 parquet")
            logger.info("安裝指令: pip install pandas pyarrow")
            return success_count == 3
        except Exception as e:
            logger.error(f"✗ 轉換過程出錯: {e}")
            return False

    def fix_dmsc(self) -> bool:
        """
        修復 DMSC v2 資料集
        解壓縮已下載的 ZIP 檔案並驗證
        """
        logger.info("\n" + "="*60)
        logger.info("修復 DMSC v2 資料集")
        logger.info("="*60)

        dest_dir = self.base_dir / "dmsc"
        zip_file = dest_dir / "dmsc_kaggle.zip"

        if not zip_file.exists():
            logger.error(f"✗ ZIP 檔案不存在: {zip_file}")
            logger.info("請先下載: https://github.com/ownthink/dmsc-v2/releases/download/v2.0/dmsc_v2.zip")
            return False

        try:
            # 檢查 ZIP 檔案大小
            zip_size_mb = zip_file.stat().st_size / (1024 * 1024)
            logger.info(f"ZIP 檔案大小: {zip_size_mb:.2f} MB")

            # 解壓縮
            logger.info("解壓縮中...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                members = zip_ref.namelist()
                logger.info(f"ZIP 內包含 {len(members)} 個檔案")

                with tqdm(total=len(members), desc="解壓縮") as pbar:
                    for member in members:
                        zip_ref.extract(member, dest_dir)
                        pbar.update(1)

            logger.info("✓ 解壓縮完成")

            # 列出解壓縮後的檔案
            extracted_files = list(dest_dir.glob("**/*.csv"))
            logger.info(f"\n找到 {len(extracted_files)} 個 CSV 檔案:")
            for f in extracted_files[:5]:  # 只顯示前 5 個
                size_mb = f.stat().st_size / (1024 * 1024)
                logger.info(f"  - {f.name}: {size_mb:.2f} MB")

            if len(extracted_files) > 5:
                logger.info(f"  ... 和其他 {len(extracted_files) - 5} 個檔案")

            # 驗證主檔案
            main_csv = dest_dir / "DMSC.csv"
            if main_csv.exists():
                csv_size_mb = main_csv.stat().st_size / (1024 * 1024)
                logger.info(f"\n✓ 主檔案 DMSC.csv: {csv_size_mb:.2f} MB")

                # 讀取前幾行驗證格式
                import pandas as pd
                df_sample = pd.read_csv(main_csv, nrows=5)
                logger.info(f"✓ 資料格式驗證通過")
                logger.info(f"  欄位: {list(df_sample.columns)}")
                logger.info(f"  樣本數（估計）: {csv_size_mb * 2500:.0f} 筆")  # 估計

                return True
            else:
                logger.warning("⚠ 未找到 DMSC.csv 主檔案")
                return len(extracted_files) > 0

        except Exception as e:
            logger.error(f"✗ DMSC 修復失敗: {e}")
            return False

    def fix_ntusd(self) -> bool:
        """
        修復 NTUSD 資料集
        重新克隆完整儲存庫
        """
        logger.info("\n" + "="*60)
        logger.info("修復 NTUSD 資料集")
        logger.info("="*60)

        dest_dir = self.base_dir / "ntusd"
        repo_url = "https://github.com/candlewill/NTUSD.git"

        # 檢查現有檔案大小
        existing_files = {}
        for filename in ["positive.txt", "negative.txt"]:
            filepath = dest_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                existing_files[filename] = size
                logger.info(f"現有檔案 {filename}: {size} bytes")

        # 如果檔案太小（< 100 bytes），需要重新下載
        needs_update = any(size < 100 for size in existing_files.values()) or len(existing_files) < 2

        if not needs_update:
            logger.info("✓ NTUSD 檔案看起來正常，跳過更新")
            return True

        try:
            # 刪除舊的目錄
            if dest_dir.exists():
                logger.info(f"刪除舊目錄: {dest_dir}")
                import shutil
                shutil.rmtree(dest_dir)

            # 重新克隆
            logger.info(f"克隆儲存庫: {repo_url}")
            subprocess.run([
                "git", "clone", "--depth", "1", repo_url, str(dest_dir)
            ], check=True)

            # 驗證檔案
            expected_files = {
                "positive.txt": 3000,   # 至少 3KB
                "negative.txt": 3000,   # 至少 3KB
                "ntusd-positive.txt": 1000,
                "ntusd-negative.txt": 1000
            }

            logger.info("\n驗證檔案:")
            success = True
            for filename, min_size in expected_files.items():
                filepath = dest_dir / filename
                if filepath.exists():
                    size = filepath.stat().st_size
                    status = "✓" if size >= min_size else "⚠"
                    logger.info(f"{status} {filename}: {size} bytes")
                    if size < min_size:
                        success = False
                else:
                    logger.warning(f"✗ {filename}: 不存在")

            if success:
                logger.info("\n✓ NTUSD 修復完成！")
            else:
                logger.warning("\n⚠ NTUSD 部分檔案可能不完整")

            return success

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Git 克隆失敗: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ NTUSD 修復失敗: {e}")
            return False

    def verify_all(self) -> Dict[str, bool]:
        """驗證所有資料集"""
        logger.info("\n" + "="*60)
        logger.info("驗證所有資料集狀態")
        logger.info("="*60)

        results = {}

        # ChnSentiCorp
        chnsenti_dir = self.base_dir / "chnsenticorp"
        chnsenti_files = list(chnsenti_dir.glob("*.json")) + list(chnsenti_dir.glob("*.parquet"))
        results["ChnSentiCorp"] = len(chnsenti_files) >= 3
        logger.info(f"ChnSentiCorp: {'✓' if results['ChnSentiCorp'] else '✗'} ({len(chnsenti_files)} 個檔案)")

        # DMSC
        dmsc_dir = self.base_dir / "dmsc"
        dmsc_csv = dmsc_dir / "DMSC.csv"
        results["DMSC"] = dmsc_csv.exists() and dmsc_csv.stat().st_size > 100_000_000  # > 100MB
        size_mb = dmsc_csv.stat().st_size / (1024 * 1024) if dmsc_csv.exists() else 0
        logger.info(f"DMSC v2: {'✓' if results['DMSC'] else '✗'} ({size_mb:.2f} MB)")

        # NTUSD
        ntusd_dir = self.base_dir / "ntusd"
        ntusd_files = [ntusd_dir / f for f in ["positive.txt", "negative.txt"]]
        results["NTUSD"] = all(f.exists() and f.stat().st_size > 1000 for f in ntusd_files)
        logger.info(f"NTUSD: {'✓' if results['NTUSD'] else '✗'}")

        logger.info("="*60)
        return results

    def run_all_fixes(self):
        """執行所有修復"""
        results = {}

        logger.info("\n🔧 開始修復所有資料集...\n")

        # 1. 修復 ChnSentiCorp
        results["ChnSentiCorp"] = self.fix_chnsenticorp()

        # 2. 修復 DMSC
        results["DMSC"] = self.fix_dmsc()

        # 3. 修復 NTUSD
        results["NTUSD"] = self.fix_ntusd()

        # 4. 最終驗證
        verification = self.verify_all()

        # 總結
        logger.info("\n" + "="*60)
        logger.info("📊 修復總結")
        logger.info("="*60)

        all_success = all(results.values())

        for dataset, success in results.items():
            status = "✓ 成功" if success else "✗ 失敗"
            verify = "✓" if verification.get(dataset, False) else "✗"
            logger.info(f"{dataset}: {status} (驗證: {verify})")

        logger.info("="*60)

        if all_success:
            logger.info("\n🎉 所有資料集修復完成！")
        else:
            logger.warning("\n⚠ 部分資料集修復失敗，請查看上方日誌")

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="修復資料集下載問題")
    parser.add_argument(
        "--dataset",
        choices=["chnsenticorp", "dmsc", "ntusd", "all"],
        default="all",
        help="要修復的資料集"
    )
    parser.add_argument(
        "--output-dir",
        default="./data/raw",
        help="資料集目錄"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只驗證，不執行修復"
    )

    args = parser.parse_args()

    fixer = DatasetFixer(base_dir=args.output_dir)

    if args.verify_only:
        fixer.verify_all()
    elif args.dataset == "all":
        fixer.run_all_fixes()
    elif args.dataset == "chnsenticorp":
        fixer.fix_chnsenticorp()
        fixer.verify_all()
    elif args.dataset == "dmsc":
        fixer.fix_dmsc()
        fixer.verify_all()
    elif args.dataset == "ntusd":
        fixer.fix_ntusd()
        fixer.verify_all()


if __name__ == "__main__":
    main()