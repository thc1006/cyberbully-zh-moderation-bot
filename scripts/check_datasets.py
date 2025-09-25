#!/usr/bin/env python3
"""
檢查資料集下載狀態
"""
from pathlib import Path


def check_dataset_status():
    """檢查各資料集的下載狀態"""

    base_dir = Path("data/raw")
    external_dir = Path("data/external")

    datasets = {
        "COLD": {
            "path": base_dir / "cold" / "COLDataset",
            "required_files": ["train.csv", "dev.csv", "test.csv"],
            "status": "Not Found"
        },
        "ChnSentiCorp": {
            "path": base_dir / "chnsenticorp",
            "required_files": ["train.json", "test.json"],
            "alternate_files": [
                "train.csv", "test.csv", "data.json",
                "ChnSentiCorp_htl_all.csv", "ChnSentiCorp_htl_all_2.csv"
            ],
            "status": "Not Found"
        },
        "DMSC v2": {
            "path": base_dir / "dmsc",
            "required_files": ["dmsc_v2.csv"],
            "alternate_files": [
                "train.csv", "test.csv", "ratings.txt",
                "DMSC.csv", "dmsc_kaggle.zip"
            ],
            "status": "Not Found"
        },
        "NTUSD": {
            "path": base_dir / "ntusd",
            "required_files": ["positive.txt", "negative.txt"],
            "alternate_files": ["ntusd-positive.txt", "ntusd-negative.txt"],
            "status": "Not Found"
        },
        "SCCD": {
            "path": external_dir / "sccd",
            "required_files": ["sccd_events.json", "sccd_annotations.json"],
            "status": "Not Found"
        },
        "CHNCI": {
            "path": external_dir / "chnci",
            "required_files": ["chnci_events.json", "chnci_annotations.json"],
            "status": "Not Found"
        }
    }

    print("="*60)
    print("CyberPuppy Dataset Status Check")
    print("="*60)

    for name, info in datasets.items():
        path = info["path"]

        if path.exists():
            # 檢查必需檔案
            files_found = []
            files_missing = []

            for file in info["required_files"]:
                if (path / file).exists():
                    files_found.append(file)
                else:
                    files_missing.append(file)

            # 檢查備選檔案
            if "alternate_files" in info and files_missing:
                for file in info["alternate_files"]:
                    if (path / file).exists():
                        files_found.append(f"{file} (alternate)")

            if files_found and not files_missing:
                info["status"] = "Complete"

                # 計算資料集大小
                total_size = 0
                for file in path.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size

                size_mb = total_size / (1024 * 1024)
                info["size"] = f"{size_mb:.2f} MB"

            elif files_found:
                info["status"] = "Partial"
                info["found"] = files_found
                info["missing"] = files_missing
            else:
                info["status"] = "Empty Directory"
        else:
            info["status"] = "Not Downloaded"

    # 輸出結果
    for name, info in datasets.items():
        status = info["status"]

        if status == "Complete":
            print(f"\n[OK] {name}: Complete")
            if "size" in info:
                print(f"   Size: {info['size']}")
            print(f"   Path: {info['path']}")

        elif status == "Partial":
            print(f"\n[WARN] {name}: Partial")
            print(f"   Found: {', '.join(info['found'])}")
            print(f"   Missing: {', '.join(info['missing'])}")
            print(f"   Path: {info['path']}")

        elif status == "Empty Directory":
            print(f"\n[DIR] {name}: Directory exists but empty")
            print(f"   Path: {info['path']}")

        else:
            print(f"\n[MISS] {name}: Not Downloaded")
            print(f"   Expected path: {info['path']}")

    # 統計
    print("\n" + "="*60)
    print("Summary:")
    complete = sum(1 for info in datasets.values() if info["sta"
        "tus"] == 
    partial = sum(1 for info in datasets.values() if info["sta"
        "tus"] == 
    not_downloaded = sum(
        1 for info in datasets.values()
        if info["status"] in ["Not Downloaded", "Empty Directory"]
    )

    print(f"  Complete: {complete}/{len(datasets)}")
    print(f"  Partial: {partial}/{len(datasets)}")
    print(f"  Not Downloaded: {not_downloaded}/{len(datasets)}")

    if not_downloaded > 0:
        print("\n[INFO] Next Steps:")
        print("  Please refer to docs/DATASET_DOWNLO"
            "AD_GUIDE.md for download instructions")

    print("="*60)


if __name__ == "__main__":
    check_dataset_status()
