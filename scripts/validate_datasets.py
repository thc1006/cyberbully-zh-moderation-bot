#!/usr/bin/env python3
"""
資料集驗證腳本
驗證 ChnSentiCorp、DMSC v2、NTUSD 的完整性和格式
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 預期檔案規格
DATASET_SPECS = {
    "chnsenticorp": {
        "name": "ChnSentiCorp 中文情感分析",
        "files": {
            "train.json": {"min_size_kb": 1000, "min_records": 5000},
            "validation.json": {"min_size_kb": 100, "min_records": 400},
            "test.json": {"min_size_kb": 100, "min_records": 200}
        },
        "required_fields": ["label", "text"],
        "label_values": [0, 1]
    },
    "dmsc": {
        "name": "DMSC v2 豆瓣電影短評",
        "files": {
            "DMSC.csv": {
                "expected_hash": "4bdb50bb400d23a8eebcc11b27195fc4",
                "min_size_mb": 300,
                "min_records": 2000000
            }
        },
        "required_fields": ["ID", "Movie_Name_EN", "Movie_Name_CN", "Star", "Comment"],
        "star_range": [1, 2, 3, 4, 5]
    },
    "ntusd": {
        "name": "NTUSD 臺大情感詞典",
        "files": {
            "positive.txt": {"min_size_kb": 20, "min_words": 2500},
            "negative.txt": {"min_size_kb": 70, "min_words": 7500}
        },
        "encoding": "utf-8",
        "total_min_words": 10000
    }
}


class DatasetValidator:
    """資料集驗證器"""

    def __init__(self, base_dir: str = "./data/raw"):
        self.base_dir = Path(base_dir)
        self.results = {}

    def calculate_file_hash(self, file_path: Path) -> str:
        """計算檔案 MD5 雜湊值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def validate_chnsenticorp(self) -> Dict:
        """驗證 ChnSentiCorp 資料集"""
        logger.info("Validating ChnSentiCorp dataset...")

        dataset_dir = self.base_dir / "chnsenticorp"
        spec = DATASET_SPECS["chnsenticorp"]
        validation_result = {
            "dataset": "chnsenticorp",
            "name": spec["name"],
            "status": "PASS",
            "issues": [],
            "files": {},
            "summary": {}
        }

        if not dataset_dir.exists():
            validation_result["status"] = "FAIL"
            validation_result["issues"].append("Dataset directory not found")
            return validation_result

        total_records = 0

        # 檢查每個檔案
        for filename, file_spec in spec["files"].items():
            file_path = dataset_dir / filename
            file_result = {"exists": False, "size_kb": 0, "records": 0, "valid": False}

            if file_path.exists():
                file_result["exists"] = True
                file_size_kb = file_path.stat().st_size / 1024
                file_result["size_kb"] = round(file_size_kb, 2)

                try:
                    # 讀取 JSON 檔案
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if isinstance(data, list):
                        file_result["records"] = len(data)
                        total_records += len(data)

                        # 檢查必要欄位
                        if data and all(field in data[0] for field in spec["required_fields"]):
                            file_result["valid"] = True
                        else:
                            validation_result["issues"].append(f"{filename}: Missing required fields")

                    # 檢查檔案大小和記錄數
                    if file_size_kb < file_spec["min_size_kb"]:
                        validation_result["issues"].append(f"{filename}: File too small ({file_size_kb:.1f} KB)")

                    if file_result["records"] < file_spec["min_records"]:
                        validation_result["issues"].append(f"{filename}: Too few records ({file_result['records']})")

                except Exception as e:
                    validation_result["issues"].append(f"{filename}: Read error - {e}")
            else:
                validation_result["issues"].append(f"{filename}: File not found")

            validation_result["files"][filename] = file_result

        validation_result["summary"] = {
            "total_records": total_records,
            "files_found": sum(1 for f in validation_result["files"].values() if f["exists"]),
            "files_valid": sum(1 for f in validation_result["files"].values() if f["valid"])
        }

        if validation_result["issues"]:
            validation_result["status"] = "FAIL"

        return validation_result

    def validate_dmsc(self) -> Dict:
        """驗證 DMSC v2 資料集"""
        logger.info("Validating DMSC v2 dataset...")

        dataset_dir = self.base_dir / "dmsc"
        spec = DATASET_SPECS["dmsc"]
        validation_result = {
            "dataset": "dmsc",
            "name": spec["name"],
            "status": "PASS",
            "issues": [],
            "files": {},
            "summary": {}
        }

        if not dataset_dir.exists():
            validation_result["status"] = "FAIL"
            validation_result["issues"].append("Dataset directory not found")
            return validation_result

        # 檢查 DMSC.csv
        csv_file = dataset_dir / "DMSC.csv"
        file_result = {"exists": False, "size_mb": 0, "records": 0, "hash_valid": False, "columns_valid": False}

        if csv_file.exists():
            file_result["exists"] = True
            file_size_mb = csv_file.stat().st_size / (1024 * 1024)
            file_result["size_mb"] = round(file_size_mb, 2)

            # 檢查檔案大小
            min_size = spec["files"]["DMSC.csv"]["min_size_mb"]
            if file_size_mb < min_size:
                validation_result["issues"].append(f"DMSC.csv: File too small ({file_size_mb:.1f} MB)")

            # 檢查雜湊值
            try:
                actual_hash = self.calculate_file_hash(csv_file)
                expected_hash = spec["files"]["DMSC.csv"]["expected_hash"]
                file_result["hash_valid"] = (actual_hash == expected_hash)
                file_result["actual_hash"] = actual_hash

                if not file_result["hash_valid"]:
                    validation_result["issues"].append(f"DMSC.csv: Hash mismatch")
            except Exception as e:
                validation_result["issues"].append(f"DMSC.csv: Hash calculation error - {e}")

            # 檢查欄位和記錄數（僅讀取前1000行以節省時間）
            try:
                sample_df = pd.read_csv(csv_file, nrows=1000, encoding='utf-8')
                file_result["columns_valid"] = all(col in sample_df.columns for col in spec["required_fields"])

                if not file_result["columns_valid"]:
                    missing_cols = [col for col in spec["required_fields"] if col not in sample_df.columns]
                    validation_result["issues"].append(f"DMSC.csv: Missing columns: {missing_cols}")

                # 估算總記錄數（基於檔案大小）
                estimated_records = int(file_size_mb * 1024 * 1024 / 190)  # 估算每行約190 bytes
                file_result["estimated_records"] = estimated_records

                if estimated_records < spec["files"]["DMSC.csv"]["min_records"]:
                    validation_result["issues"].append(f"DMSC.csv: Too few records (estimated: {estimated_records})")

            except Exception as e:
                validation_result["issues"].append(f"DMSC.csv: Read error - {e}")
        else:
            validation_result["issues"].append("DMSC.csv: File not found")

        validation_result["files"]["DMSC.csv"] = file_result
        validation_result["summary"] = {
            "file_size_mb": file_result.get("size_mb", 0),
            "estimated_records": file_result.get("estimated_records", 0),
            "hash_valid": file_result.get("hash_valid", False)
        }

        if validation_result["issues"]:
            validation_result["status"] = "FAIL"

        return validation_result

    def validate_ntusd(self) -> Dict:
        """驗證 NTUSD 資料集"""
        logger.info("Validating NTUSD dataset...")

        dataset_dir = self.base_dir / "ntusd"
        spec = DATASET_SPECS["ntusd"]
        validation_result = {
            "dataset": "ntusd",
            "name": spec["name"],
            "status": "PASS",
            "issues": [],
            "files": {},
            "summary": {}
        }

        if not dataset_dir.exists():
            validation_result["status"] = "FAIL"
            validation_result["issues"].append("Dataset directory not found")
            return validation_result

        total_words = 0

        # 檢查正負面詞典檔案
        for filename, file_spec in spec["files"].items():
            file_path = dataset_dir / filename
            file_result = {"exists": False, "size_kb": 0, "word_count": 0, "encoding_valid": False}

            if file_path.exists():
                file_result["exists"] = True
                file_size_kb = file_path.stat().st_size / 1024
                file_result["size_kb"] = round(file_size_kb, 2)

                try:
                    # 讀取詞典檔案
                    content = file_path.read_text(encoding=spec["encoding"])
                    words = [line.strip() for line in content.split('\n') if line.strip()]
                    file_result["word_count"] = len(words)
                    file_result["encoding_valid"] = True
                    total_words += len(words)

                    # 檢查詞彙數量
                    if len(words) < file_spec["min_words"]:
                        validation_result["issues"].append(
                            f"{filename}: Too few words ({len(words)} < {file_spec['min_words']})"
                        )

                    # 檢查檔案大小
                    if file_size_kb < file_spec["min_size_kb"]:
                        validation_result["issues"].append(
                            f"{filename}: File too small ({file_size_kb:.1f} KB)"
                        )

                    # 檢查詞彙品質（樣本檢查）
                    sample_words = words[:10]
                    if not all(any('\u4e00' <= char <= '\u9fff' for char in word) for word in sample_words):
                        validation_result["issues"].append(f"{filename}: Contains non-Chinese words")

                except UnicodeDecodeError:
                    validation_result["issues"].append(f"{filename}: Encoding error")
                except Exception as e:
                    validation_result["issues"].append(f"{filename}: Read error - {e}")
            else:
                validation_result["issues"].append(f"{filename}: File not found")

            validation_result["files"][filename] = file_result

        # 檢查總詞彙數
        validation_result["summary"] = {
            "total_words": total_words,
            "positive_words": validation_result["files"].get("positive.txt", {}).get("word_count", 0),
            "negative_words": validation_result["files"].get("negative.txt", {}).get("word_count", 0)
        }

        if total_words < spec["total_min_words"]:
            validation_result["issues"].append(f"Total words too few ({total_words} < {spec['total_min_words']})")

        if validation_result["issues"]:
            validation_result["status"] = "FAIL"

        return validation_result

    def validate_all(self) -> Dict:
        """驗證所有資料集"""
        logger.info("Starting comprehensive dataset validation...")

        all_results = {
            "validation_time": pd.Timestamp.now().isoformat(),
            "base_directory": str(self.base_dir),
            "datasets": {},
            "overall_status": "PASS",
            "summary": {
                "total_datasets": 0,
                "passed": 0,
                "failed": 0
            }
        }

        # 執行各個資料集的驗證
        validators = {
            "chnsenticorp": self.validate_chnsenticorp,
            "dmsc": self.validate_dmsc,
            "ntusd": self.validate_ntusd
        }

        for dataset_name, validator_func in validators.items():
            try:
                result = validator_func()
                all_results["datasets"][dataset_name] = result
                all_results["summary"]["total_datasets"] += 1

                if result["status"] == "PASS":
                    all_results["summary"]["passed"] += 1
                else:
                    all_results["summary"]["failed"] += 1
                    all_results["overall_status"] = "FAIL"

            except Exception as e:
                logger.error(f"Validation error for {dataset_name}: {e}")
                all_results["datasets"][dataset_name] = {
                    "dataset": dataset_name,
                    "status": "ERROR",
                    "error": str(e)
                }
                all_results["summary"]["failed"] += 1
                all_results["overall_status"] = "FAIL"

        return all_results

    def print_validation_report(self, results: Dict):
        """列印驗證報告"""
        print("\n" + "="*80)
        print("DATASET VALIDATION REPORT")
        print("="*80)
        print(f"Validation Time: {results['validation_time']}")
        print(f"Base Directory: {results['base_directory']}")
        print(f"Overall Status: {results['overall_status']}")

        # 總結
        summary = results['summary']
        print(f"\nSummary:")
        print(f"  Total Datasets: {summary['total_datasets']}")
        print(f"  Passed: {summary['passed']}")
        print(f"  Failed: {summary['failed']}")

        # 詳細結果
        for dataset_name, dataset_result in results['datasets'].items():
            print(f"\n{'-'*60}")
            print(f"Dataset: {dataset_result.get('name', dataset_name)}")
            print(f"Status: {dataset_result['status']}")

            if 'summary' in dataset_result:
                print("Summary:")
                for key, value in dataset_result['summary'].items():
                    print(f"  {key}: {value}")

            if dataset_result.get('issues'):
                print("Issues:")
                for issue in dataset_result['issues']:
                    print(f"  ⚠️  {issue}")

            if 'files' in dataset_result:
                print("Files:")
                for filename, file_info in dataset_result['files'].items():
                    status = "✅" if file_info.get('exists', False) else "❌"
                    print(f"  {status} {filename}")
                    if file_info.get('exists'):
                        for key, value in file_info.items():
                            if key != 'exists':
                                print(f"      {key}: {value}")

        print("="*80)

    def save_validation_report(self, results: Dict, output_file: str = None):
        """保存驗證報告"""
        if output_file is None:
            output_file = f"validation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_path = Path(output_file)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
        logger.info(f"Validation report saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="驗證 CyberPuppy 資料集")
    parser.add_argument(
        "--data-dir",
        default="./data/raw",
        help="資料集目錄"
    )
    parser.add_argument(
        "--output",
        help="輸出報告檔案路徑"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="只顯示錯誤"
    )

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # 執行驗證
    validator = DatasetValidator(base_dir=args.data_dir)
    results = validator.validate_all()

    # 顯示結果
    validator.print_validation_report(results)

    # 保存報告
    if args.output:
        validator.save_validation_report(results, args.output)

    # 根據結果設定退出碼
    if results['overall_status'] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()