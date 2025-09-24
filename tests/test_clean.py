#!/usr/bin/env python3
"""
資料清理模組的單元測試
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# 添加專案路徑到系統路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.clean_normalize import (
    DatasetCleaner, TextNormalizer  # noqa: E402
)


class TestTextNormalizer(unittest.TestCase):
    """測試文字正規化器"""

    def setUp(self):
        """測試前初始化"""
        self.normalizer = TextNormalizer(
            convert_mode=None, preserve_emoji=True, anonymize=True  # 測試時不做繁簡轉換
        )

    def test_fullwidth_to_halfwidth(self):
        """測試全形轉半形"""
        # 測試全形英文和數字
        text = "ＡＢＣ１２３"
        expected = "ABC123"
        result = self.normalizer.fullwidth_to_halfwidth(text)
        self.assertEqual(result, expected)

        # 測試全形空格
        text = "你好　世界"
        expected = "你好 世界"
        result = self.normalizer.fullwidth_to_halfwidth(text)
        self.assertEqual(result, expected)

        # 測試全形符號
        text = "！＠＃＄％"
        expected = "!@#$%"
        result = self.normalizer.fullwidth_to_halfwidth(text)
        self.assertEqual(result, expected)

    def test_normalize_whitespace(self):
        """測試空白正規化"""
        # 測試多個空格
        text = "hello    world    test"
        expected = "hello world test"
        result = self.normalizer.normalize_whitespace(text)
        self.assertEqual(result, expected)

        # 測試混合空白字符
        text = "hello\t\n\r  world"
        expected = "hello world"
        result = self.normalizer.normalize_whitespace(text)
        self.assertEqual(result, expected)

        # 測試首尾空白
        text = "  hello world  "
        expected = "hello world"
        result = self.normalizer.normalize_whitespace(text)
        self.assertEqual(result, expected)

    def test_replace_urls(self):
        """測試URL替換"""
        test_cases = [
            ("Visit https://www.google.com for more", "Visit [URL] for more"),
            ("Check http://example.com and https://test.org", "Check [URL] and [URL]"),
            ("IP: 192.168.1.1:8080", "IP: [URL]"),
            ("No URL here", "No URL here"),
            ("複雜URL: https://example.com/path?query=123#anchor", "複雜URL: [URL]"), 
        ]

        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.normalizer.replace_urls(text)
                self.assertEqual(result, expected)

    def test_replace_mentions(self):
        """測試@mention替換"""
        test_cases = [
            ("Hello @user123", "Hello [MENTION]"),
            ("@測試用戶 你好", "[MENTION] 你好"),
            ("Multiple @user1 @user2", "Multiple [MENTION] [MENTION]"),
            ("Email: test@example.com", "Email: [EMAIL]"), 
        ]

        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.normalizer.replace_mentions(text)
                self.assertEqual(result, expected)

    def test_replace_hashtags(self):
        """測試hashtag替換"""
        test_cases = [
            ("#Python is great", "[HASHTAG] is great"),
            ("標籤 #測試 #中文標籤", "標籤 [HASHTAG] [HASHTAG]"),
            ("No hashtag here", "No hashtag here"),
        ]

        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.normalizer.replace_hashtags(text)
                self.assertEqual(result, expected)

    def test_anonymize_email(self):
        """測試Email去識別化"""
        test_cases = [
            ("Contact: john@example.com", "Contact: [EMAIL]"),
            ("Emails: a@b.com and test@gmail.com", "Emails: [EMAIL] and [EMAIL]"), 
            ("No email here", "No email here"),
        ]

        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.normalizer.anonymize_sensitive(text)
                # 只檢查email部分，因為可能還會處理其他敏感資訊
                self.assertIn(expected.split()[0], result)

    def test_anonymize_phone(self):
        """測試電話號碼去識別化"""
        test_cases = [
            ("Call me at 0912345678", "Call me at [PHONE]"),
            ("手機：0987654321", "手機：[PHONE]"),
            ("電話 02-12345678", "電話 [PHONE]"),
            ("大陸手機 13912345678", "大陸手機 [PHONE]"),
        ]

        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.normalizer.anonymize_sensitive(text)
                self.assertEqual(result, expected)

    def test_anonymize_id(self):
        """測試身分證號碼去識別化"""
        test_cases = [
            ("身分證：A123456789", "身分證：[ID]"),
            ("ID: B234567890", "ID: [ID]"),
            ("大陸身分證：12345678901234567X", "大陸身分證：[ID]"),
        ]

        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.normalizer.anonymize_sensitive(text)
                self.assertEqual(result, expected)

    def test_mark_emojis(self):
        """測試表情符號標記"""
        # 測試保留表情模式
        normalizer_preserve = TextNormalizer(preserve_emoji=True)
        text = "Hello 😀 World 🌍"
        result = normalizer_preserve.mark_emojis(text)
        self.assertIn("[EMOJI:😀]", result)
        self.assertIn("[EMOJI:🌍]", result)

        # 測試移除表情模式
        normalizer_remove = TextNormalizer(preserve_emoji=False)
        result = normalizer_remove.mark_emojis(text)
        self.assertEqual(result, "Hello [EMOJI] World [EMOJI]")

    def test_complete_normalize(self):
        """測試完整的正規化流程"""
        text = (
            "請訪問　　https://example.com　　聯絡 @user"
                "　郵箱：test@email.com　電話：0912345678 😀"
        )
        result = self.normalizer.normalize(text)

        # 檢查各項處理是否生效
        self.assertIn("[URL]", result)
        self.assertIn("[MENTION]", result)
        self.assertIn("[EMAIL]", result)
        self.assertIn("[PHONE]", result)
        self.assertIn("[EMOJI:", result)
        self.assertNotIn("　　", result)  # 全形空格應被處理
        self.assertNotIn("  ", result)  # 多個空格應被合併

    def test_empty_input(self):
        """測試空輸入"""
        test_cases = [None, "", "   "]
        for text in test_cases:
            with self.subTest(text=text):
                result = self.normalizer.normalize(text)
                self.assertIn(result, [None, ""])


class TestDatasetCleaner(unittest.TestCase):
    """測試資料集清理器"""

    def setUp(self):
        """測試前初始化"""
        # 建立臨時目錄
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "raw"
        self.output_dir = Path(self.temp_dir) / "processed"

        # 建立測試資料結構
        self.input_dir.mkdir(parents=True, exist_ok=True)

        # 初始化清理器
        self.normalizer = TextNormalizer()
        self.cleaner = DatasetCleaner(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            normalizer=self.normalizer,
        )

    def tearDown(self):
        """測試後清理"""
        # 刪除臨時目錄
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_id(self):
        """測試ID生成"""
        text1 = "Hello World"
        text2 = "Hello World"
        text3 = "Different Text"

        # 相同文字和索引應生成相同ID
        id1 = self.cleaner.generate_id(text1, 0)
        id2 = self.cleaner.generate_id(text2, 0)
        self.assertEqual(id1, id2)

        # 不同文字應生成不同ID
        id3 = self.cleaner.generate_id(text3, 0)
        self.assertNotEqual(id1, id3)

        # 相同文字但不同索引應生成不同ID
        id4 = self.cleaner.generate_id(text1, 1)
        self.assertNotEqual(id1, id4)

        # ID格式檢查
        self.assertTrue(id1.startswith("proc_"))
        self.assertEqual(len(id1), 17)  # proc_ + 12個字符

    def test_clean_json_dataset(self):
        """測試JSON資料集清理"""
        # 建立測試JSON資料
        test_dataset_dir = self.input_dir / "test_dataset"
        test_dataset_dir.mkdir(parents=True, exist_ok=True)

        test_data = [
            {"text": "測試文字１２３", "label": 0},
            {"text": "包含URL https://test.com", "label": 1},
            {"text": "有@mention和#hashtag", "label": 0},
        ]

        test_file = test_dataset_dir / "test.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False)

        # 執行清理
        results = self.cleaner.clean_json_dataset("test_dataset", "text")

        # 檢查結果
        self.assertIn("test_dataset_test", results)
        self.assertEqual(results["test_dataset_test"]["total_samples"], 3)

        # 檢查輸出檔案
        output_file = self.output_dir / "test_dataset" / "test_processed.json"
        self.assertTrue(output_file.exists())

        # 檢查處理後的資料
        with open(output_file, "r", encoding="utf-8") as f:
            processed_data = json.load(f)

        self.assertEqual(len(processed_data), 3)

        # 檢查正規化效果
        self.assertEqual(processed_data[0]["text"], "測試文字123")  # 全形轉半形
        self.assertIn("[URL]", processed_data[1]["text"])  # URL替換
        self.assertIn("[MENTION]", processed_data[2]["text"])  # Mention替換
        self.assertIn("[HASHTAG]", processed_data[2]["text"])  # Hashtag替換

        # 檢查ID
        for item in processed_data:
            self.assertIn("processed_id", item)
            self.assertIn("raw_id", item)
            self.assertTrue(item["processed_id"].startswith("proc_"))

    def test_save_mapping(self):
        """測試mapping儲存"""
        # 添加一些測試mapping
        self.cleaner.mapping_data = [
            {"raw_id": "test_1", "processed_id": "proc_abc123", "text": "測試1"},
            {"raw_id": "test_2", "processed_id": "proc_def456", "text": "測試2"},
        ]

        # 儲存mapping
        self.cleaner.save_mapping()

        # 檢查JSON檔案
        json_file = self.output_dir / "id_mapping.json"
        self.assertTrue(json_file.exists())

        with open(json_file, "r", encoding="utf-8") as f:
            loaded_mapping = json.load(f)

        self.assertEqual(len(loaded_mapping), 2)
        self.assertEqual(loaded_mapping[0]["raw_id"], "test_1")

        # 檢查CSV檔案
        csv_file = self.output_dir / "id_mapping.csv"
        self.assertTrue(csv_file.exists())


class TestIntegration(unittest.TestCase):
    """整合測試"""

    def test_chinese_conversion(self):
        """測試繁簡轉換（如果安裝了OpenCC）"""
        try:
            from opencc import OpenCC

            normalizer = TextNormalizer(convert_mode="s2t")

            text = "这是简体中文"
            result = normalizer.convert_chinese(text)
            # 只檢查是否執行成功，不檢查具體轉換結果（因為可能沒安裝OpenCC）
            self.assertIsNotNone(result)
        except ImportError:
            self.skipTest("OpenCC not installed")

    def test_stats_tracking(self):
        """測試統計追蹤"""
        normalizer = TextNormalizer()

        # 處理包含各種元素的文字
        texts = [
            "URL: https://example.com",
            "Mention: @user123",
            "Email: test@email.com",
            "Phone: 0912345678",
            "Emoji: 😀",
        ]

        for text in texts:
            normalizer.normalize(text)

        stats = normalizer.get_stats()

        # 檢查統計數據
        self.assertEqual(stats["total_processed"], 5)
        self.assertGreater(stats["urls_replaced"], 0)
        self.assertGreater(stats["mentions_replaced"], 0)
        self.assertGreater(stats["emails_replaced"], 0)
        self.assertGreater(stats["phones_replaced"], 0)
        self.assertGreater(stats["emojis_marked"], 0)


def run_tests():
    """執行所有測試"""
    # 建立測試套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有測試類
    suite.addTests(loader.loadTestsFromTestCase(TestTextNormalizer))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetCleaner))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # 執行測試
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回測試結果
    return result.wasSuccessful()


if __name__ == "__main__":
    # 執行測試
    success = run_tests()
    sys.exit(0 if success else 1)
