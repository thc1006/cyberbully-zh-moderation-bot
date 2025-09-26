#!/usr/bin/env python3
"""
完整資料集下載與修復腳本
支援 ChnSentiCorp (.arrow)、DMSC v2 (ZIP)、NTUSD (Git) 等資料集的正確下載與處理
包含進度條、錯誤處理、斷點續傳、詳細日誌等功能
"""

import os
import sys
import json
import zipfile
import logging
import hashlib
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import subprocess
import argparse
from datetime import datetime

try:
    import requests
    from tqdm import tqdm
    import pandas as pd
    from datasets import load_dataset, Dataset
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError as e:
    print(f"缺少必要的依賴套件: {e}")
    print("請執行: pip install requests tqdm pandas datasets huggingface_hub pyarrow")
    sys.exit(1)

# 設定日誌
class SafeStreamHandler(logging.StreamHandler):
    """安全的串流處理器，避免 Unicode 編碼問題"""
    def emit(self, record):
        try:
            msg = self.format(record)
            # 移除 emoji 和特殊字符，在 Windows 命令列中使用純文字
            msg = (msg.replace('🔍', '[INFO]')
                     .replace('📊', '[REPORT]')
                     .replace('✅', '[OK]')
                     .replace('❌', '[FAIL]')
                     .replace('⚠️', '[WARN]')
                     .replace('🚫', '[ERROR]')
                     .replace('📈', '[SUMMARY]')
                     .replace('📝', '[SAVE]')
                     .replace('📄', '[FILE]')
                     .replace('💾', '[DATA]')
                     .replace('🚀', '[START]')
                     .replace('🔄', '[PROCESS]')
                     .replace('⏱️', '[TIME]')
                     .replace('🎉', '[SUCCESS]'))
            self.stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # 如果仍然有編碼問題，使用 ASCII 安全版本
            safe_msg = record.getMessage().encode('ascii', 'ignore').decode('ascii')
            self.stream.write(f"{record.levelname}: {safe_msg}{self.terminator}")
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_setup.log', encoding='utf-8'),
        SafeStreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """完整資料集下載器"""

    def __init__(self, base_dir: str = "./data/raw", force_download: bool = False):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.force_download = force_download
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CyberPuppy-Complete-Dataset-Setup/2.0'
        })

        # 資料集配置
        self.datasets_config = {
            'cold': {
                'name': 'COLD Dataset',
                'description': '中文攻擊性語言檢測資料集',
                'method': 'git_clone',
                'url': 'https://github.com/thu-coai/COLDataset.git',
                'expected_files': ['COLDataset/train.csv', 'COLDataset/dev.csv', 'COLDataset/test.csv'],
                'size_estimate': '10MB'
            },
            'chnsenticorp': {
                'name': 'ChnSentiCorp',
                'description': '中文情感分析資料集',
                'method': 'huggingface_arrow',
                'dataset_id': 'seamew/ChnSentiCorp',
                'expected_files': ['train.arrow', 'validation.arrow', 'test.arrow', 'ChnSentiCorp_htl_all.csv'],
                'size_estimate': '5MB'
            },
            'dmsc': {
                'name': 'DMSC v2',
                'description': '大眾點評情感分析資料集',
                'method': 'zip_download',
                'url': 'https://github.com/ownthink/dmsc-v2/releases/download/v2.0/dmsc_v2.zip',
                'expected_files': ['DMSC.csv'],
                'size_estimate': '144MB'
            },
            'ntusd': {
                'name': 'NTUSD',
                'description': '台大中文情感詞典',
                'method': 'git_clone',
                'url': 'https://github.com/ntunlplab/NTUSD.git',
                'expected_files': ['data/正面詞無重複_9365詞.txt', 'data/負面詞無重複_11230詞.txt'],
                'size_estimate': '1MB'
            }
        }

    def _calculate_file_hash(self, file_path: Path) -> str:
        """計算檔案 SHA256 hash"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _download_with_progress(self, url: str, output_path: Path,
                               chunk_size: int = 8192) -> bool:
        """帶進度條的檔案下載，支援斷點續傳"""
        headers = {}
        initial_pos = 0

        # 檢查是否支援斷點續傳
        if output_path.exists() and not self.force_download:
            initial_pos = output_path.stat().st_size
            headers['Range'] = f'bytes={initial_pos}-'
            logger.info(f"檢測到部分下載檔案，從 {initial_pos} bytes 繼續")

        try:
            response = self.session.get(url, headers=headers, stream=True, timeout=30)

            # 檢查 HTTP 狀態碼
            if response.status_code not in [200, 206]:
                if response.status_code == 416:  # Range Not Satisfiable
                    logger.info("檔案已完全下載")
                    return True
                response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            if initial_pos > 0:
                total_size += initial_pos

            mode = 'ab' if initial_pos > 0 else 'wb'

            with open(output_path, mode) as f:
                with tqdm(
                    total=total_size,
                    initial=initial_pos,
                    unit='B',
                    unit_scale=True,
                    desc=output_path.name
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"✓ 下載完成: {output_path}")
            return True

        except Exception as e:
            logger.error(f"✗ 下載失敗: {e}")
            if output_path.exists():
                output_path.unlink()  # 刪除不完整的檔案
            return False

    def download_huggingface_arrow(self, dataset_id: str, dest_dir: Path) -> bool:
        """下載 Hugging Face 資料集的 .arrow 檔案"""
        logger.info(f"="*60)
        logger.info(f"下載 Hugging Face 資料集: {dataset_id}")
        logger.info(f"="*60)

        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 方法 1: 嘗試直接從 Hub 下載 parquet 檔案
            logger.info("嘗試從 Hugging Face Hub 直接下載...")

            success_count = 0
            try:
                # 列出儲存庫中的所有檔案
                repo_files = list_repo_files(dataset_id, repo_type="dataset")

                # 優先下載 arrow 檔案
                arrow_files = [f for f in repo_files if f.endswith('.arrow')]
                parquet_files = [f for f in repo_files if f.endswith('.parquet')]

                if arrow_files:
                    logger.info(f"找到 {len(arrow_files)} 個 arrow 檔案")
                    for arrow_file in arrow_files:
                        # 解析檔案名稱格式: chn_senti_corp-train.arrow -> train.arrow
                        original_name = Path(arrow_file).name
                        if '-' in original_name:
                            split_name = original_name.split('-')[-1]  # 取最後一部分
                        else:
                            split_name = original_name

                        local_path = dest_dir / split_name

                        if local_path.exists() and not self.force_download:
                            logger.info(f"✓ {split_name} 已存在，跳過下載")
                            success_count += 1
                            continue

                        logger.info(f"下載 {arrow_file} -> {split_name}...")
                        downloaded_path = hf_hub_download(
                            repo_id=dataset_id,
                            filename=arrow_file,
                            local_dir=str(dest_dir),
                            repo_type="dataset"
                        )

                        # 重新命名為標準格式
                        Path(downloaded_path).rename(local_path)

                        success_count += 1
                        logger.info(f"✓ {split_name} 下載完成")

                elif parquet_files:
                    logger.info(f"找到 {len(parquet_files)} 個 parquet 檔案")
                    for parquet_file in parquet_files:
                        local_path = dest_dir / Path(parquet_file).name
                        if local_path.exists() and not self.force_download:
                            logger.info(f"✓ {local_path.name} 已存在，跳過下載")
                            success_count += 1
                            continue

                        logger.info(f"下載 {parquet_file}...")
                        hf_hub_download(
                            repo_id=dataset_id,
                            filename=parquet_file,
                            local_dir=str(dest_dir),
                            repo_type="dataset"
                        )
                        success_count += 1
                        logger.info(f"✓ {local_path.name} 下載完成")

                        # 嘗試轉換 parquet 到 arrow
                        try:
                            import pyarrow.parquet as pq
                            import pyarrow as pa

                            split_name = Path(parquet_file).stem
                            arrow_file = dest_dir / f"{split_name}.arrow"

                            if not arrow_file.exists() or self.force_download:
                                table = pq.read_table(local_path)
                                with pa.OSFile(str(arrow_file), 'wb') as sink:
                                    with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                                        writer.write_table(table)
                                logger.info(f"✓ 轉換 {split_name}.parquet 到 {split_name}.arrow")
                        except ImportError:
                            logger.warning("pyarrow 未安裝，無法轉換 parquet 到 arrow")
                        except Exception as e:
                            logger.warning(f"轉換 parquet 到 arrow 失敗: {e}")

                else:
                    logger.warning("未找到 arrow 或 parquet 檔案")

            except Exception as e:
                logger.error(f"Hub 檔案列表失敗: {e}")

            # 方法 2: 如果 Hub 下載失敗，嘗試使用 datasets 庫（無腳本模式）
            if success_count == 0:
                logger.info("嘗試使用 datasets 庫下載...")

                try:
                    dataset = load_dataset(dataset_id, trust_remote_code=False)

                    for split_name, split_dataset in dataset.items():
                        arrow_file = dest_dir / f"{split_name}.arrow"

                        if arrow_file.exists() and not self.force_download:
                            logger.info(f"✓ {split_name}.arrow 已存在，跳過下載")
                            success_count += 1
                            continue

                        logger.info(f"保存 {split_name} split 到 .arrow 格式...")

                        # 直接儲存為 arrow 格式
                        split_dataset.save_to_disk(str(dest_dir / split_name))

                        # 檢查並移動 .arrow 檔案
                        arrow_files = list((dest_dir / split_name).glob("*.arrow"))
                        if arrow_files:
                            main_arrow = arrow_files[0]
                            main_arrow.rename(arrow_file)

                            # 清理臨時目錄
                            import shutil
                            shutil.rmtree(dest_dir / split_name)

                            logger.info(f"✓ {split_name}.arrow 儲存完成 ({len(split_dataset)} 筆資料)")
                            success_count += 1
                        else:
                            # 後備方案：儲存為 parquet 和 JSON
                            parquet_file = dest_dir / f"{split_name}.parquet"
                            json_file = dest_dir / f"{split_name}.json"

                            split_dataset.to_parquet(str(parquet_file))
                            split_dataset.to_json(str(json_file))

                            logger.info(f"✓ {split_name} 儲存為 parquet 和 JSON")
                            success_count += 1

                except Exception as e:
                    logger.error(f"datasets 庫下載失敗: {e}")

            # 儲存元資料
            if success_count > 0:
                metadata = {
                    "dataset_id": dataset_id,
                    "download_method": "huggingface_hub",
                    "download_time": datetime.now().isoformat(),
                    "success_count": success_count,
                    "format": "parquet+arrow",
                }

                with open(dest_dir / "metadata.json", 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

            return success_count > 0

        except Exception as e:
            logger.error(f"✗ Hugging Face 下載失敗: {e}")
            return False

    def download_zip_dataset(self, url: str, dest_dir: Path) -> bool:
        """下載並解壓縮 ZIP 資料集"""
        logger.info(f"="*60)
        logger.info(f"下載 ZIP 資料集: {url}")
        logger.info(f"="*60)

        dest_dir.mkdir(parents=True, exist_ok=True)
        zip_filename = Path(url).name
        zip_path = dest_dir / zip_filename

        # 下載 ZIP 檔案
        if not zip_path.exists() or self.force_download:
            logger.info(f"下載 {zip_filename}...")
            if not self._download_with_progress(url, zip_path):
                return False
        else:
            logger.info(f"✓ {zip_filename} 已存在，跳過下載")

        # 驗證 ZIP 檔案
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_info = zip_ref.infolist()
                logger.info(f"ZIP 檔案包含 {len(zip_info)} 個檔案")

            # 檢查是否已解壓縮
            csv_files = list(dest_dir.glob("**/*.csv"))
            if csv_files and not self.force_download:
                logger.info(f"✓ 檢測到 {len(csv_files)} 個 CSV 檔案，跳過解壓縮")
                return True

            # 解壓縮
            logger.info("解壓縮中...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = zip_ref.namelist()

                with tqdm(total=len(members), desc="解壓縮") as pbar:
                    for member in members:
                        zip_ref.extract(member, dest_dir)
                        pbar.update(1)

            logger.info("✓ 解壓縮完成")

            # 驗證解壓縮結果
            extracted_files = list(dest_dir.glob("**/*.csv"))
            logger.info(f"解壓縮後找到 {len(extracted_files)} 個 CSV 檔案")

            # 顯示主要檔案資訊
            for csv_file in extracted_files[:5]:  # 顯示前 5 個
                size_mb = csv_file.stat().st_size / (1024 * 1024)
                logger.info(f"  - {csv_file.name}: {size_mb:.2f} MB")

            # 儲存元資料
            metadata = {
                "source_url": url,
                "download_method": "zip_download",
                "download_time": datetime.now().isoformat(),
                "zip_file": zip_filename,
                "extracted_files": [f.name for f in extracted_files],
                "total_size_mb": sum(f.stat().st_size for f in extracted_files) / (1024 * 1024)
            }

            with open(dest_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            return len(extracted_files) > 0

        except Exception as e:
            logger.error(f"✗ ZIP 處理失敗: {e}")
            return False

    def clone_git_repository(self, url: str, dest_dir: Path, depth: int = 1) -> bool:
        """克隆 Git 儲存庫"""
        logger.info(f"="*60)
        logger.info(f"克隆 Git 儲存庫: {url}")
        logger.info(f"="*60)

        # 檢查是否已存在
        if dest_dir.exists() and not self.force_download:
            if (dest_dir / ".git").exists():
                logger.info("檢測到現有 Git 儲存庫，嘗試更新...")
                try:
                    subprocess.run(["git", "pull"], cwd=dest_dir, check=True,
                                 capture_output=True, text=True)
                    logger.info("✓ Git 儲存庫更新完成")
                    return True
                except subprocess.CalledProcessError:
                    logger.warning("Git 更新失敗，將重新克隆...")
                    import shutil
                    shutil.rmtree(dest_dir)
            else:
                logger.info("檢測到現有目錄但非 Git 儲存庫，檢查是否需要重新下載...")
                # 這裡可以檢查檔案是否完整，如果不完整就重新下載
                return True  # 暫時跳過重新下載

        # 如果是強制下載模式，刪除現有目錄
        elif dest_dir.exists() and self.force_download:
            logger.info(f"強制模式：刪除現有目錄 {dest_dir}")
            import shutil
            shutil.rmtree(dest_dir)

        try:
            # 克隆儲存庫
            logger.info(f"克隆到: {dest_dir}")

            cmd = ["git", "clone"]
            if depth > 0:
                cmd.extend(["--depth", str(depth)])
            cmd.extend([url, str(dest_dir)])

            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("✓ Git 克隆完成")

            # 顯示儲存庫資訊
            try:
                # 獲取最新提交資訊
                commit_result = subprocess.run(
                    ["git", "log", "-1", "--oneline"],
                    cwd=dest_dir,
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"最新提交: {commit_result.stdout.strip()}")

                # 計算檔案統計
                file_count = len(list(dest_dir.rglob("*")))
                total_size = sum(f.stat().st_size for f in dest_dir.rglob("*") if f.is_file())
                logger.info(f"檔案數量: {file_count}, 總大小: {total_size / (1024 * 1024):.2f} MB")

            except subprocess.CalledProcessError:
                pass  # 不是關鍵錯誤

            # 儲存元資料
            metadata = {
                "repository_url": url,
                "download_method": "git_clone",
                "download_time": datetime.now().isoformat(),
                "clone_depth": depth,
                "file_count": file_count if 'file_count' in locals() else 0,
                "total_size_mb": total_size / (1024 * 1024) if 'total_size' in locals() else 0
            }

            with open(dest_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Git 克隆失敗: {e}")
            if e.stderr:
                logger.error(f"錯誤詳情: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"✗ Git 操作失敗: {e}")
            return False

    def verify_dataset(self, dataset_name: str) -> Tuple[bool, Dict]:
        """驗證單個資料集"""
        config = self.datasets_config.get(dataset_name)
        if not config:
            return False, {"error": f"未知資料集: {dataset_name}"}

        dest_dir = self.base_dir / dataset_name
        if not dest_dir.exists():
            return False, {"error": "資料集目錄不存在"}

        result = {
            "name": config["name"],
            "description": config["description"],
            "path": str(dest_dir),
            "files": [],
            "total_size_mb": 0,
            "status": "unknown"
        }

        try:
            # 檢查預期檔案
            found_files = []
            missing_files = []
            total_size = 0

            for expected_file in config["expected_files"]:
                file_patterns = [
                    dest_dir / expected_file,
                ]

                # 如果包含路徑分隔符，則使用遞歸搜索
                if '/' in expected_file:
                    file_patterns.append(dest_dir / expected_file)
                else:
                    # 否則在所有子目錄中搜索
                    file_patterns.extend(list(dest_dir.rglob(expected_file)))

                found = False
                for pattern in file_patterns:
                    if isinstance(pattern, Path) and pattern.exists():
                        file_path = pattern
                        size = file_path.stat().st_size
                        found_files.append({
                            "name": expected_file,
                            "path": str(file_path.relative_to(dest_dir)),
                            "size_mb": size / (1024 * 1024),
                            "exists": True
                        })
                        total_size += size
                        found = True
                        break
                    elif not isinstance(pattern, Path):
                        # 處理 glob 結果
                        if pattern.exists():
                            file_path = pattern
                            size = file_path.stat().st_size
                            found_files.append({
                                "name": expected_file,
                                "path": str(file_path.relative_to(dest_dir)),
                                "size_mb": size / (1024 * 1024),
                                "exists": True
                            })
                            total_size += size
                            found = True
                            break

                if not found:
                    missing_files.append(expected_file)

            result["files"] = found_files
            result["missing_files"] = missing_files
            result["total_size_mb"] = total_size / (1024 * 1024)

            # 判斷狀態
            if not missing_files:
                result["status"] = "complete"
            elif found_files:
                result["status"] = "partial"
            else:
                result["status"] = "missing"

            # 讀取元資料
            metadata_file = dest_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    result["metadata"] = json.load(f)

            return result["status"] in ["complete", "partial"], result

        except Exception as e:
            result["error"] = str(e)
            result["status"] = "error"
            return False, result

    def download_dataset(self, dataset_name: str) -> bool:
        """下載單個資料集"""
        config = self.datasets_config.get(dataset_name)
        if not config:
            logger.error(f"未知資料集: {dataset_name}")
            return False

        logger.info(f"\n🚀 開始下載: {config['name']}")
        logger.info(f"描述: {config['description']}")
        logger.info(f"預估大小: {config['size_estimate']}")

        dest_dir = self.base_dir / dataset_name
        method = config["method"]

        try:
            if method == "huggingface_arrow":
                return self.download_huggingface_arrow(config["dataset_id"], dest_dir)
            elif method == "zip_download":
                return self.download_zip_dataset(config["url"], dest_dir)
            elif method == "git_clone":
                return self.clone_git_repository(config["url"], dest_dir)
            else:
                logger.error(f"不支援的下載方法: {method}")
                return False

        except Exception as e:
            logger.error(f"✗ 下載 {dataset_name} 失敗: {e}")
            return False

    def download_all_datasets(self, datasets: Optional[List[str]] = None) -> Dict[str, bool]:
        """下載所有或指定的資料集"""
        if datasets is None:
            datasets = list(self.datasets_config.keys())

        results = {}

        logger.info(f"\n🔄 開始下載 {len(datasets)} 個資料集...")
        start_time = time.time()

        for i, dataset_name in enumerate(datasets, 1):
            logger.info(f"\n[{i}/{len(datasets)}] 處理資料集: {dataset_name}")
            results[dataset_name] = self.download_dataset(dataset_name)

        elapsed_time = time.time() - start_time
        logger.info(f"\n⏱️ 總下載時間: {elapsed_time:.2f} 秒")

        return results

    def generate_final_report(self, download_results: Optional[Dict[str, bool]] = None) -> Dict:
        """生成最終驗證報告"""
        logger.info(f"\n" + "="*80)
        logger.info("📊 最終驗證報告")
        logger.info("="*80)

        report = {
            "timestamp": datetime.now().isoformat(),
            "base_directory": str(self.base_dir),
            "datasets": {},
            "summary": {
                "total": len(self.datasets_config),
                "complete": 0,
                "partial": 0,
                "missing": 0,
                "error": 0
            }
        }

        for dataset_name in self.datasets_config.keys():
            success, details = self.verify_dataset(dataset_name)
            report["datasets"][dataset_name] = details

            status = details.get("status", "error")
            if status in report["summary"]:
                report["summary"][status] += 1

            # 顯示資料集狀態
            status_emoji = {
                "complete": "[OK]",
                "partial": "[WARN]",
                "missing": "[FAIL]",
                "error": "[ERROR]"
            }

            emoji = status_emoji.get(status, "[UNKNOWN]")
            size_info = f"({details.get('total_size_mb', 0):.2f} MB)" if details.get('total_size_mb') else ""

            logger.info(f"{emoji} {details.get('name', dataset_name)}: {status.upper()} {size_info}")

            if details.get('missing_files'):
                logger.info(f"   缺少檔案: {', '.join(details['missing_files'])}")

            if download_results and dataset_name in download_results:
                download_status = "成功" if download_results[dataset_name] else "失敗"
                logger.info(f"   下載狀態: {download_status}")

        # 顯示總結
        logger.info(f"\n📈 總結:")
        logger.info(f"   完整: {report['summary']['complete']}/{report['summary']['total']}")
        logger.info(f"   部分: {report['summary']['partial']}/{report['summary']['total']}")
        logger.info(f"   缺失: {report['summary']['missing']}/{report['summary']['total']}")
        logger.info(f"   錯誤: {report['summary']['error']}/{report['summary']['total']}")

        # 計算總大小
        total_size_mb = sum(
            details.get('total_size_mb', 0)
            for details in report["datasets"].values()
        )
        logger.info(f"   總大小: {total_size_mb:.2f} MB")

        # 儲存報告
        report_file = self.base_dir / "dataset_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"\n📝 詳細報告已儲存至: {report_file}")
        logger.info("="*80)

        return report


def main():
    parser = argparse.ArgumentParser(
        description="完整資料集下載與修復腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法:
  python complete_dataset_setup.py --dataset all           # 下載所有資料集
  python complete_dataset_setup.py --dataset chnsenticorp # 只下載 ChnSentiCorp
  python complete_dataset_setup.py --verify-only          # 只驗證，不下載
  python complete_dataset_setup.py --force                # 強制重新下載
        """
    )

    parser.add_argument(
        "--dataset",
        nargs='+',
        choices=["cold", "chnsenticorp", "dmsc", "ntusd", "all"],
        default=["all"],
        help="要下載的資料集 (可選擇多個)"
    )

    parser.add_argument(
        "--output-dir",
        default="./data/raw",
        help="資料集輸出目錄 (預設: ./data/raw)"
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只驗證現有資料集，不執行下載"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="強制重新下載，即使檔案已存在"
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="只生成驗證報告"
    )

    args = parser.parse_args()

    # 展開 "all" 選項
    if "all" in args.dataset:
        datasets = ["cold", "chnsenticorp", "dmsc", "ntusd"]
    else:
        datasets = args.dataset

    # 建立下載器
    downloader = DatasetDownloader(
        base_dir=args.output_dir,
        force_download=args.force
    )

    # 執行操作
    download_results = None

    if args.report_only or args.verify_only:
        logger.info("🔍 只執行驗證，不下載資料集")
    else:
        # 執行下載
        download_results = downloader.download_all_datasets(datasets)

    # 生成最終報告
    final_report = downloader.generate_final_report(download_results)

    # 結果總結
    if download_results:
        success_count = sum(1 for success in download_results.values() if success)
        total_count = len(download_results)

        if success_count == total_count:
            logger.info("\n🎉 所有資料集下載完成！")
            return 0
        else:
            logger.warning(f"\n⚠️ {success_count}/{total_count} 個資料集下載成功")
            return 1
    else:
        # 只驗證模式
        complete_count = final_report["summary"]["complete"]
        total_count = final_report["summary"]["total"]

        if complete_count == total_count:
            logger.info("\n✅ 所有資料集驗證通過！")
            return 0
        else:
            logger.info(f"\n📊 {complete_count}/{total_count} 個資料集完整")
            return 0


if __name__ == "__main__":
    sys.exit(main())