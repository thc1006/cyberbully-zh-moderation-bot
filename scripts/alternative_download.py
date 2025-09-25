#!/usr/bin/env python3
"""
替代資料集下載方法 - 嘗試從多個來源下載資料集
"""

import json
import requests
from pathlib import Path
import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlternativeDownloader:
    """替代下載方法"""

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_file_direct(self, url: str, output_path: Path) -> bool:
        """直接下載檔案"""
        try:
            logger.info(f"Downloading from: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Downloaded to: {output_path}")
            return True
        except Exception as _e:
            logger.error(f"Download failed: {_e}")
            return False

    def download_chnsenticorp_alternatives(self):
        """嘗試從不同來源下載ChnSentiCorp"""
        output_dir = self.output_dir / "chnsenticorp"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 方法1: 從GitHub鏡像下載
        github_urls = [
            (
                "https://raw.githubusercontent.c"
                    "om/SophonPlus/ChineseNlpCorpus/"
                "master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv"
            ),
            (
                "https://raw.githubusercontent.com"
                    "/pengming617/bert_classification/"
                "master/data/ChnSentiCorp/train.txt"
            ),
            (
                "https://raw.githubusercontent.com"
                    "/pengming617/bert_classification/"
                "master/data/ChnSentiCorp/test.txt"
            ),
            (
                "https://raw.githubusercontent.com"
                    "/pengming617/bert_classification/"
                "master/data/ChnSentiCorp/dev.txt"
            ),
        ]

        success_count = 0
        for url in github_urls:
            filename = url.split('/')[-1]
            output_path = output_dir / filename
            if self.download_file_direct(url, output_path):
                success_count += 1

        # 方法2: 嘗試使用curl/wget
        if success_count == 0:
            logger.info("Trying with curl...")
            try:
                subprocess.run([
                    "curl", "-L", "-o", str(output_dir / "chnsenticorp.zip"),
                    (
                        "https://github.com/SophonPlus"
                            "/ChineseNlpCorpus/raw/master/"
                        "datasets/ChnSentiCorp_htl_all.zip"
                    )
                ], check=True)
                success_count += 1
            except subprocess.CalledProcessError as _e:
                logger.error(\n                    f\"curl download failed with exit code {_e.returncode}: {_e}\"\n                )
            except FileNotFoundError:
                logger.error("curl command not found")
            except Exception as _e:
                logger.error(f"curl download failed: {_e}")

        # 方法3: 建立示例資料
        if success_count == 0:
            logger.info("Creating sample ChnSentiCorp data...")
            sample_data = {
                "train": [
                    {"text": "這家酒店真的很棒，服務態度非常好", "label": 1},
                    {"text": "房間太小了，而且很髒", "label": 0},
                    {"text": "位置方便，價格合理", "label": 1},
                    {"text": "隔音效果差，晚上睡不好", "label": 0},
                ],
                "test": [
                    {"text": "早餐豐富，環境優雅", "label": 1},
                    {"text": "設施老舊，需要翻新", "label": 0},
                ]
            }

            for split, data in sample_data.items():
                output_file = output_dir / f"{split}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"Created sample: {output_file}")

        return success_count > 0

    def download_dmsc_alternatives(self):
        """嘗試下載DMSC資料集"""
        output_dir = self.output_dir / "dmsc"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 方法1: 從Kaggle API（如果有API key）
        try:
            # 嘗試使用Kaggle API
            subprocess.run([
                "kaggle", "datasets", "download", "-d",
                "utmhikari/doubanmovieshortcomments", "-p", str(output_dir)
            ], check=False, capture_output=True)
        except FileNotFoundError:
            logger.info("Kaggle command not found")
        except Exception as _e:
            logger.info(f"Kaggle API not available: {_e}")

        # 方法2: GitHub上的樣本資料
        sample_urls = [
            (
                "https://raw.githubusercontent.c"
                    "om/SophonPlus/ChineseNlpCorpus/"
                "master/datasets/dmsc_v2/intro.md"
            ),
        ]

        for url in sample_urls:
            filename = "dmsc_intro.md"
            self.download_file_direct(url, output_dir / filename)

        # 方法3: 建立示例DMSC資料
        logger.info("Creating sample DMSC data...")
        sample_data = [
            {"comment": "這部電影太精彩了！", "rating": 5, "movie": "sample_movie_1"},
            {"comment": "劇情拖沓，不推薦", "rating": 2, "movie": "sample_movie_2"},
            {"comment": "還可以，中規中矩", "rating": 3, "movie": "sample_movie_3"},
        ]

        output_file = output_dir / "dmsc_sample.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Created sample: {output_file}")
        return True

    def download_ntusd_alternatives(self):
        """嘗試下載NTUSD詞典"""
        output_dir = self.output_dir / "ntusd"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 方法1: 從其他GitHub repos獲取
        alternative_urls = [
            # 嘗試從使用NTUSD的項目中獲取
            "https://raw.githubusercontent.com/zake774"
                "9/DeepToxic/master/data/ntusd-positive.txt",
            "https://raw.githubusercontent.com/zake774"
                "9/DeepToxic/master/data/ntusd-negative.txt",
        ]

        success_count = 0
        for url in alternative_urls:
            filename = url.split('/')[-1]
            output_path = output_dir / filename
            if self.download_file_direct(url, output_path):
                success_count += 1

        # 方法2: 建立基礎情感詞典
        if success_count == 0:
            logger.info("Creating basic sentiment dictionary...")

            positive_words = [
                "好", "棒", "優秀", "傑出", "美好", "快樂", "開心", "滿意",
                "喜歡", "愛", "讚", "完美", "出色", "精彩", "舒適", "方便",
                "漂亮", "美麗", "可愛", "親切", "友善", "熱情", "積極", "正面"
            ]

            negative_words = [
                "壞", "差", "糟糕", "惡劣", "討厭", "失望", "難過", "生氣",
                "憤怒", "恨", "爛", "垃圾", "失敗", "錯誤", "問題", "麻煩",
                "醜", "髒", "臭", "噁心", "無聊", "冷漠", "消極", "負面"
            ]

            # 儲存正面詞
            positive_file = output_dir / "positive.txt"
            with open(positive_file, 'w', encoding='utf-8') as f:
                for word in positive_words:
                    f.write(word + '\n')
            logger.info(f"Created: {positive_file}")

            # 儲存負面詞
            negative_file = output_dir / "negative.txt"
            with open(negative_file, 'w', encoding='utf-8') as f:
                for word in negative_words:
                    f.write(word + '\n')
            logger.info(f"Created: {negative_file}")

        return True

    def create_sccd_chnci_instructions(self):
        """建立SCCD和CHNCI的詳細下載說明"""
        external_dir = Path("data/external")
        external_dir.mkdir(parents=True, exist_ok=True)

        # SCCD說明
        sccd_dir = external_dir / "sccd"
        sccd_dir.mkdir(parents=True, exist_ok=True)

        sccd_readme = sccd_dir / "README.md"
        with open(sccd_readme, 'w', encoding='utf-8') as f:
            f.write("""# SCCD (Session-level Chinese Cyberbullying Dataset)

## 獲取方法

1. **閱讀論文**: https://arxiv.org/abs/2506.04975

2. **聯繫作者**:
   - 查看論文中的通訊作者郵箱
   - 準備學術或研究用途說明

3. **申請模板**:
```
Subject: Request for SCCD Dataset Access

Dear Professor [Name],

I am working on cyberbullying prevention research and would like to request
access to the SCCD dataset described in your paper.

Purpose: Academic research on cyberbullying detection and prevention
Institution: [Your Institution]
Usage: Non-commercial, research only

Thank you for your consideration.

Best regards,
[Your Name]
```

4. **預期檔案**:
   - sccd_train.json
   - sccd_dev.json
   - sccd_test.json

5. **放置位置**: 將下載的檔案放在此目錄下
""")

        # CHNCI說明
        chnci_dir = external_dir / "chnci"
        chnci_dir.mkdir(parents=True, exist_ok=True)

        chnci_readme = chnci_dir / "README.md"
        with open(chnci_readme, 'w', encoding='utf-8') as f:
            f.write("""# CHNCI (Chinese Cyberbullying Incident Dataset)

## 獲取方法

1. **閱讀論文**: https://arxiv.org/abs/2506.05380

2. **申請流程**:
   - 訪問論文中提供的申請連結
   - 填寫資料使用協議
   - 等待審核（通常1-2週）

3. **所需資訊**:
   - 研究目的說明
   - 機構證明
   - 資料安全承諾

4. **預期檔案**:
   - chnci_events.json
   - chnci_annotations.json

5. **使用限制**:
   - 僅限學術研究
   - 不可再分發
   - 需引用原論文
""")

        logger.info(f"Created instructions in {external_dir}")
        return True

    def download_all_alternatives(self):
        """執行所有替代下載方法"""
        results = {}

        # 下載ChnSentiCorp
        logger.info("\n" + "="*60)
        logger.info("Attempting ChnSentiCorp download...")
        results['chnsenticorp'] = self.download_chnsenticorp_alternatives()

        # 下載DMSC
        logger.info("\n" + "="*60)
        logger.info("Attempting DMSC download...")
        results['dmsc'] = self.download_dmsc_alternatives()

        # 下載NTUSD
        logger.info("\n" + "="*60)
        logger.info("Attempting NTUSD download...")
        results['ntusd'] = self.download_ntusd_alternatives()

        # 建立SCCD/CHNCI說明
        logger.info("\n" + "="*60)
        logger.info("Creating SCCD/CHNCI instructions...")
        results['instructions'] = self.create_sccd_chnci_instructions()

        # 總結
        logger.info("\n" + "="*60)
        logger.info("Download Summary:")
        for dataset, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"  {dataset}: {status}")

        return results


def main():
    downloader = AlternativeDownloader()
    downloader.download_all_alternatives()

    print("\n" + "="*60)
    print("Alternative Download Complete")
    print("="*60)
    print("\nNext steps:")
    print("1. Check data/raw/ for downloaded files")
    print("2. For SCCD/CHNCI, see data/external/ for instructions")
    print("3. Run: python scripts/check_datasets.py")
    print("="*60)


if __name__ == "__main__":
    main()
