#!/usr/bin/env python3
"""
簡化版 ChnSentiCorp 下載腳本 - 直接從 Hugging Face 下載
"""
import json
from pathlib import Path


def download_chnsenticorp():
    output_dir = Path("data/raw/chnsenticorp")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading ChnSentiCorp dataset from Hugging Face...")

    try:
        from datasets import load_dataset

        # 嘗試從不同的來源下載
        dataset_names = [
            "seamew/ChnSentiCorp",
            "chnsenticorp",
            "eriktks/ChnSentiCorp"
        ]

        dataset = None
        for name in dataset_names:
            try:
                print(f"Trying to load from: {name}")
                dataset = load_dataset(name, trust_remote_code=True)
                print(f"Successfully loaded from: {name}")
                break
            except Exception as e:
                print(f"Failed with {name}: {str(e)[:100]}")
                continue

        if dataset is None:
            print("Failed to load dataset from all sources")
            return False

        # 保存資料集
        for split in dataset.keys():
            output_file = output_dir / f"{split}.json"
            print(f"Saving {split} split ({len(dataset[split])} samples)...")

            # 轉換為 JSON 格式
            data = []
            for item in dataset[split]:
                data.append(item)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"Saved to: {output_file}")

        # 保存元資料
        metadata = {
            "dataset_name": "ChnSentiCorp",
            "source": "Hugging Face",
            "splits": list(dataset.keys()),
            "total_samples": sum(
                len(dataset[split]) for split in dataset.keys()
            ),
            "samples_per_split": {
                split: len(dataset[split]) for split in dataset.keys()
            }
        }

        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print("\nDataset successfully downloaded!")
        print(f"Total samples: {metadata['total_samples']}")
        for split, count in metadata['samples_per_split'].items():
            print(f"  {split}: {count} samples")

        return True

    except ImportError:
        print("Error: datasets library not installed")
        print("Please run: pip install datasets")
        return False
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


if __name__ == "__main__":
    success = download_chnsenticorp()
    exit(0 if success else 1)
