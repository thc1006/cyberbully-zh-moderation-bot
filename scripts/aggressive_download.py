#!/usr/bin/env python3
"""
積極嘗試下載所有可能的資料集來源
"""

import json
import requests
import zipfile
from pathlib import Path
import logging
import subprocess
import csv
import time
import urllib.request
import ssl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 忽略 SSL 警告（某些站點可能有證書問題）
ssl._create_default_https_context = ssl._create_unverified_context


class AggressiveDownloader:
    """積極下載器 - 嘗試所有可能的來源"""

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
        })

    def download_with_urllib(self, url: str, output_path: Path) -> bool:
        """使用 urllib 下載（備選方案）"""
        try:
            logger.info(f"Downloading with urllib: {url}")
            urllib.request.urlretrieve(url, output_path)
            logger.info(f"Success: {output_path}")
            return True
        except Exception as e:
            logger.error(f"urllib failed: {e}")
            return False

    def download_with_requests(self, url: str, output_path: Path) -> bool:
        """使用 requests 下載"""
        try:
            logger.info(f"Downloading with requests: {url}")
            response = self.session.get(
                url, stream=True, timeout=60, verify=False
            )
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Success: {output_path}")
            return True
        except Exception as e:
            logger.error(f"requests failed: {e}")
            return False

    def download_with_wget(self, url: str, output_path: Path) -> bool:
        """使用 wget 命令下載"""
        try:
            logger.info(f"Downloading with wget: {url}")
            subprocess.run([
                'wget', '-O', str(output_path), '--no-check-certificate',
                '--user-agent=Mozilla/5.0', url
            ], check=True, capture_output=True)
            logger.info(f"Success: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"wget command failed: {e}")
            return False
        except Exception as e:
            logger.error(f"wget download failed: {e}")
            return False

    def download_with_curl(self, url: str, output_path: Path) -> bool:
        """使用 curl 命令下載"""
        try:
            logger.info(f"Downloading with curl: {url}")
            subprocess.run([
                'curl', '-L', '-o', str(output_path), '-k', url
            ], check=True, capture_output=True)
            logger.info(f"Success: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"curl command failed: {e}")
            return False
        except Exception as e:
            logger.error(f"curl download failed: {e}")
            return False

    def try_all_methods(self, url: str, output_path: Path) -> bool:
        """嘗試所有下載方法"""
        methods = [
            self.download_with_requests,
            self.download_with_urllib,
            self.download_with_curl,
            self.download_with_wget
        ]

        for method in methods:
            if method(url, output_path):
                return True

        return False

    def download_chnsenticorp_comprehensive(self):
        """全面嘗試下載 ChnSentiCorp"""
        output_dir = self.output_dir / "chnsenticorp"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 已經下載的檔案
        existing = list(output_dir.glob("*.csv"))
        if existing:
            logger.info(f"ChnSentiCorp already has {len(existing)} files")

        # 更多可能的來源
        sources = [
            # GitHub 上的各種版本
            (
                "https://github.com/SophonPlus/ChineseNlpCorpus/raw/master/"
                "datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
                "ChnSentiCorp_htl_all_2.csv"
            ),
            (
                "https://raw.githubusercontent.c"
                    "om/fate233/nlp-datasets/master/"
                "ChnSentiCorp_htl_all.csv",
                "ChnSentiCorp_from_fate233.csv"
            ),

            # Gitee 鏡像
            (
                "https://gitee.com/sophonplus/ChineseNlpCorpus/raw/master/"
                "datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
                "ChnSentiCorp_gitee.csv"
            ),

            # 其他項目中的版本
            (
                "https://github.com/gaussic/text-classification-cnn-rnn/raw/"
                "master/data/cnews.train.txt",
                "sample_chinese_text.txt"
            ),
        ]

        success_count = 0
        for url, filename in sources:
            output_path = output_dir / filename
            if output_path.exists():
                logger.info(f"Already exists: {filename}")
                success_count += 1
                continue

            if self.try_all_methods(url, output_path):
                success_count += 1
            time.sleep(1)  # 避免過快請求

        # 如果還沒有數據，創建更完整的示例
        if success_count == 0:
            logger.info("Creating comprehensive ChnSentiCorp sample data...")
            self.create_comprehensive_sample_data(output_dir, "chnsenticorp")

        return success_count > 0

    def download_dmsc_comprehensive(self):
        """全面嘗試下載 DMSC"""
        output_dir = self.output_dir / "dmsc"
        output_dir.mkdir(parents=True, exist_ok=True)

        sources = [
            # 可能的 GitHub 來源
            (
                "https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/"
                "datasets/dmsc_v2/intro.txt",
                "dmsc_intro.txt"
            ),

            # 從其他項目獲取樣本
            (
                "https://raw.githubusercontent.com"
                    "/fxsjy/jieba/master/test/test.txt",
                "chinese_sample.txt"
            ),

            # Kaggle API 下載（如果有 key）
            (
                "https://www.kaggle.com/api/v1/datasets/download/utmhikari/"
                "doubanmovieshortcomments",
                "dmsc_kaggle.zip"
            ),
        ]

        success_count = 0
        for url, filename in sources:
            output_path = output_dir / filename
            if output_path.exists():
                success_count += 1
                continue

            if self.try_all_methods(url, output_path):
                success_count += 1

                # 如果是 zip 檔案，解壓
                if filename.endswith('.zip') and output_path.exists():
                    try:
                        with zipfile.ZipFile(output_path, 'r') as zip_ref:
                            zip_ref.extractall(output_dir)
                        logger.info(f"Extracted: {filename}")
                    except zipfile.BadZipFile as e:
                        logger.error(f"Invalid zip file {filename}: {e}")
                    except Exception as e:
                        logger.error(f"Extraction failed for {filename}: {e}")

        # 創建更豐富的樣本資料
        if success_count == 0:
            self.create_comprehensive_sample_data(output_dir, "dmsc")

        return success_count > 0

    def download_ntusd_comprehensive(self):
        """全面嘗試下載 NTUSD"""
        output_dir = self.output_dir / "ntusd"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 已有基礎詞典，擴充更多來源
        sources = [
            # 從其他情感詞典項目
            (
                "https://raw.githubusercontent.com/"
                    "Tony607/Chinese_sentiment_analysis/"
                "master/dict/positive.txt",
                "positive_extended.txt"
            ),
            (
                "https://raw.githubusercontent.com/"
                    "Tony607/Chinese_sentiment_analysis/"
                "master/dict/negative.txt",
                "negative_extended.txt"
            ),

            # BosonNLP 情感詞典
            (
                "https://raw.githubusercontent.com"
                    "/BosonNLP/sentiment-analysis-dict/"
                "master/positive.txt",
                "boson_positive.txt"
            ),
            (
                "https://raw.githubusercontent.com"
                    "/BosonNLP/sentiment-analysis-dict/"
                "master/negative.txt",
                "boson_negative.txt"
            ),

            # 知網情感詞典（部分）
            (
                "https://raw.githubusercontent.com"
                    "/rainarch/SentiBridge/master/data/"
                "hownet_positive.txt",
                "hownet_positive.txt"
            ),
            (
                "https://raw.githubusercontent.com"
                    "/rainarch/SentiBridge/master/data/"
                "hownet_negative.txt",
                "hownet_negative.txt"
            ),
        ]

        success_count = 0
        for url, filename in sources:
            output_path = output_dir / filename
            if output_path.exists():
                success_count += 1
                continue

            if self.try_all_methods(url, output_path):
                success_count += 1

        # 合併所有詞典
        if success_count > 0:
            self.merge_sentiment_dictionaries(output_dir)

        return success_count > 0

    def merge_sentiment_dictionaries(self, output_dir: Path):
        """合併多個情感詞典"""
        positive_words = set()
        negative_words = set()

        # 讀取所有正面詞檔案
        for pos_file in output_dir.glob("*positive*.txt"):
            try:
                with open(pos_file, 'r', encoding='utf-8') as f:
                    words = f.read().split('\n')
                    positive_words.update(
                        w.strip() for w in words if w.strip()
                    )
            except FileNotFoundError:
                logger.warning(
                    f"Positive dictionary file not found: {pos_file}"
                )
            except UnicodeDecodeError as e:
                logger.error(f"Encoding error reading {pos_file}: {e}")
            except Exception as e:
                logger.error(
                    f"Error reading positive dictionary {pos_file}: {e}"
                )

        # 讀取所有負面詞檔案
        for neg_file in output_dir.glob("*negative*.txt"):
            try:
                with open(neg_file, 'r', encoding='utf-8') as f:
                    words = f.read().split('\n')
                    negative_words.update(w.strip() for w in words if
                        w.strip())
            except FileNotFoundError:
                logger.warning(f"Negative dictionary fil"
                    "e not found: {neg_file}")
            except UnicodeDecodeError as e:
                logger.error(f"Encoding error reading {neg_file}: {e}")
            except Exception as e:
                logger.error(f"Error reading negative d"
                    "ictionary {neg_file}: {e}")

        # 儲存合併後的詞典
        if positive_words:
            merged_pos = output_dir / "merged_positive.txt"
            with open(merged_pos, 'w', encoding='utf-8') as f:
                for word in sorted(positive_words):
                    f.write(word + '\n')
            logger.info(f"Merged {len(positive_words)} positive words")

        if negative_words:
            merged_neg = output_dir / "merged_negative.txt"
            with open(merged_neg, 'w', encoding='utf-8') as f:
                for word in sorted(negative_words):
                    f.write(word + '\n')
            logger.info(f"Merged {len(negative_words)} negative words")

    def create_comprehensive_sample_data(
        self,
        output_dir: Path,
        dataset_type: str
    ):
        """創建更完整的示例資料集"""

        if dataset_type == "chnsenticorp":
            # 創建更豐富的情感分析樣本
            samples = []

            # 正面評論
            positive_samples = [
                "這家酒店真的太棒了！服務一流，環境優雅，下次還會再來。",
                "房間乾淨整潔，床很舒服，早餐種類豐富，物超所值。",
                "地理位置優越，交通便利，工作人員態度親切友善。",
                "設施先進，裝修有品味，整體體驗非常滿意。",
                "性價比很高，推薦給需要出差的朋友們。",
            ]

            # 負面評論
            negative_samples = [
                "太失望了，房間小又髒，隔音效果差，根本睡不好。",
                "服務態度惡劣，設施陳舊，完全不值這個價格。",
                "位置偏僻，周圍什麼都沒有，WiFi還經常斷線。",
                "早餐難吃，房間有異味，再也不會來了。",
                "預訂的房型和實際不符，投訴也沒人理，體驗極差。",
            ]

            # 構建CSV格式
            for text in positive_samples:
                samples.append({'label': 1, 'review': text})
            for text in negative_samples:
                samples.append({'label': 0, 'review': text})

            # 儲存為CSV
            output_file = output_dir / "sample_sentiment.csv"
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['label', 'review'])
                writer.writeheader()
                writer.writerows(samples)

            logger.info(f"Created sample sentiment data: {output_file}")

        elif dataset_type == "dmsc":
            # 創建電影評論樣本
            samples = [
                {"mo"
                    "vie": 
                {"movie": "戰狼2", "rating": 4, "comment": "動作場面精彩，愛國情懷濃厚。"},
                {"mo"
                    "vie": 
                {"movie": "某爛片", "rating": 1, "comment": "浪費時間，劇情無聊，演技尷尬。"},
                {"movie": "普通電影", "rating": 3, "comment": "中規中矩，沒什麼特別的亮點。"},
            ]

            output_file = output_dir / "sample_movie_reviews.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)

            logger.info(f"Created sample movie review data: {output_file}")

    def try_huggingface_api(self):
        """嘗試使用 Hugging Face API 下載"""
        output_dir = self.output_dir / "chnsenticorp"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 嘗試使用 Hugging Face datasets API
        api_urls = [
            "https://huggingface.co/api/datasets/seamew/ChnSentiCorp",
            "https://huggingface.co/api/datasets/lansinuote/ChnSentiCorp",
        ]

        for url in api_urls:
            try:
                response = self.session.get(url)
                if response.status_code == 200:
                    response.json()  # Check if valid JSON
                    logger.info(f"Found dataset metadata: {url}")

                    # 嘗試獲取檔案列表
                    files_url = url.replace(
                        '/api/datasets/',
                        '/datasets/'
                    ) + '/tree/main'
                    self.try_all_methods(files_url, output_dir / "hf_metad"
                        "ata.json")
            except requests.RequestException as e:
                logger.error(f"Request failed for {url}: {e}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from {url}: {e}")
            except Exception as e:
                logger.error(f"Error accessing Hugging Face API {url}: {e}")

    def download_all_aggressive(self):
        """執行所有積極下載策略"""
        results = {}

        logger.info("\n" + "="*60)
        logger.info("Starting aggressive download attempts...")
        logger.info("="*60)

        # ChnSentiCorp
        logger.info("\n[ChnSentiCorp] Attempting downloads...")
        results['chnsenticorp'] = self.download_chnsenticorp_comprehensive()

        # DMSC
        logger.info("\n[DMSC] Attempting downloads...")
        results['dmsc'] = self.download_dmsc_comprehensive()

        # NTUSD
        logger.info("\n[NTUSD] Attempting downloads...")
        results['ntusd'] = self.download_ntusd_comprehensive()

        # 嘗試 Hugging Face API
        logger.info("\n[Hugging Face] Trying API access...")
        self.try_huggingface_api()

        # 總結
        logger.info("\n" + "="*60)
        logger.info("Aggressive Download Summary:")
        for dataset, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"  {dataset}: {status}")
        logger.info("="*60)

        return results


def main():
    downloader = AggressiveDownloader()

    # 執行積極下載
    downloader.download_all_aggressive()

    # 檢查結果
    print("\n" + "="*60)
    print("Final Status:")
    print("="*60)

    for dataset_name in ['chnsenticorp', 'dmsc', 'ntusd']:
        dataset_dir = Path("data/raw") / dataset_name
        if dataset_dir.exists():
            files = list(dataset_dir.glob("*"))
            print(f"\n{dataset_name.upper()}:")
            print(f"  Directory: {dataset_dir}")
            print(f"  Files found: {len(files)}")
            for f in files[:5]:  # 顯示前5個檔案
                size = f.stat().st_size / 1024
                print(f"    - {f.name} ({size:.1f} KB)")
            if len(files) > 5:
                print(f"    ... and {len(files)-5} more files")

    print("\n" + "="*60)
    print("Next steps:")
    print("1. Run: python scripts/check_datasets.py")
    print("2. Process data: python scripts/clean_normalize.py")
    print("="*60)


if __name__ == "__main__":
    main()
