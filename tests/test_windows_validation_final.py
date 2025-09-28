#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final validation test for Windows encoding and path issues
"""

import os
import platform
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestWindowsFinalValidation:
    """Final comprehensive validation for Windows compatibility"""

    def test_environment_setup(self):
        """Test that environment is properly configured for Chinese text"""
        # Check critical encoding settings
        assert sys.getdefaultencoding() == "utf-8", "Default encoding should be UTF-8"
        assert sys.getfilesystemencoding() == "utf-8", "Filesystem encoding should be UTF-8"

        # Platform check
        if platform.system() == "Windows":
            print(f"Windows version: {platform.version()}")
            print(f"Console encoding: {sys.stdout.encoding}")

    def test_chinese_text_in_memory(self):
        """Test Chinese text handling in memory (no console output)"""
        chinese_texts = [
            "網路霸凌檢測系統",
            "毒性內容分析",
            "情緒分析模組",
            "中文自然語言處理",
            "繁體中文測試",
            "简体中文测试",
        ]

        for text in chinese_texts:
            # Should handle encoding/decoding without issues
            encoded = text.encode("utf-8")
            decoded = encoded.decode("utf-8")
            assert decoded == text

            # Should have reasonable length
            assert len(text) > 0
            assert len(encoded) >= len(text)  # UTF-8 might be longer

    def test_project_modules_import(self):
        """Test that all main project modules can be imported"""
        modules_to_test = [
            "cyberpuppy.config",
            "cyberpuppy.labeling.label_map",
            "cyberpuppy.models.baselines",
            "cyberpuppy.safety.rules",
        ]

        for module_name in modules_to_test:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Failed to import {module_name}: {e}")

    def test_config_with_chinese_content(self):
        """Test configuration handling with Chinese content"""
        from cyberpuppy.config import Settings

        # Test creating config
        config = Settings()
        assert config.APP_NAME == "CyberPuppy"

        # Test path handling (should use pathlib.Path)
        assert isinstance(config.PROJECT_ROOT, Path)

        # Test that paths work on Windows
        data_path = config.get_data_path("test.txt")
        assert isinstance(data_path, Path)

        # Test Chinese file paths
        chinese_path = config.get_data_path("中文檔案.txt")
        assert isinstance(chinese_path, Path)

        # Should be able to handle the Chinese path
        assert "中文檔案.txt" in str(chinese_path)

    def test_file_operations_chinese(self):
        """Test file operations with Chinese content and paths"""
        chinese_content = """# 測試檔案
這是一個包含中文的測試檔案。
內容包括：
- 網路霸凌檢測
- 毒性內容分析
- 情緒分析模組

繁體中文：繁體中文測試
簡體中文：简体中文测试
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test Chinese directory name
            chinese_dir = temp_path / "中文目錄"
            chinese_dir.mkdir()
            assert chinese_dir.exists()

            # Test Chinese file name
            chinese_file = chinese_dir / "測試檔案.txt"

            # Write Chinese content
            chinese_file.write_text(chinese_content, encoding="utf-8")
            assert chinese_file.exists()

            # Read Chinese content
            read_content = chinese_file.read_text(encoding="utf-8")
            assert read_content == chinese_content

            # Test file operations
            assert chinese_file.is_file()
            assert chinese_dir.is_dir()
            assert chinese_file.parent == chinese_dir

            # Test globbing
            found_files = list(chinese_dir.glob("*.txt"))
            assert len(found_files) == 1
            assert found_files[0] == chinese_file

    def test_json_with_chinese(self):
        """Test JSON serialization with Chinese content"""
        import json

        chinese_data = {
            "系統名稱": "網路霸凌檢測系統",
            "模組": {
                "毒性檢測": "檢測有毒內容",
                "情緒分析": "分析情緒狀態",
                "角色識別": "識別霸凌角色",
            },
            "標籤": ["正常", "有毒", "嚴重"],
            "測試文本": ["這是正常的中文句子。", "這可能包含問題內容。"],
        }

        # Test serialization
        json_str = json.dumps(chinese_data, ensure_ascii=False, indent=2)

        # Should contain Chinese characters directly
        assert "網路霸凌檢測系統" in json_str
        assert "毒性檢測" in json_str

        # Test deserialization
        loaded_data = json.loads(json_str)
        assert loaded_data == chinese_data

    def test_chinese_nlp_libraries(self):
        """Test Chinese NLP libraries work correctly"""
        try:
            import jieba

            chinese_text = "中文分詞測試句子包含網路霸凌內容檢測"
            words = list(jieba.cut(chinese_text))

            assert len(words) > 0

            # All words should be valid Unicode strings
            for word in words:
                assert isinstance(word, str)
                # Should be able to encode/decode
                encoded = word.encode("utf-8")
                decoded = encoded.decode("utf-8")
                assert decoded == word

        except ImportError:
            pytest.skip("jieba not installed")

        try:
            from opencc import OpenCC

            converter = OpenCC("s2t")  # Simplified to Traditional
            simplified = "这是简体中文测试"
            traditional = converter.convert(simplified)

            assert len(traditional) > 0
            assert traditional != simplified  # Should be different

            # Should be valid Unicode
            traditional.encode("utf-8").decode("utf-8")

        except ImportError:
            pytest.skip("opencc not installed")

    def test_windows_path_separators(self):
        """Test that Windows path separators are handled correctly"""
        if platform.system() != "Windows":
            pytest.skip("Windows-specific test")

        from cyberpuppy.config import Settings

        config = Settings()

        # Test that paths use correct separators
        data_path = config.get_data_path("subdir/file.txt")
        path_str = str(data_path)

        # On Windows, pathlib should handle separators correctly
        # Either forward or backward slashes should work
        assert "subdir" in path_str
        assert "file.txt" in path_str

        # Test with Chinese paths
        chinese_path = config.get_data_path("中文目錄/中文檔案.txt")
        chinese_str = str(chinese_path)

        assert "中文目錄" in chinese_str
        assert "中文檔案.txt" in chinese_str

    def test_environment_variables_encoding(self):
        """Test encoding-related environment variables"""
        # These should be set for proper Unicode handling
        recommended_vars = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

        missing_vars = []
        for var, expected in recommended_vars.items():
            actual = os.environ.get(var)
            if actual != expected:
                missing_vars.append(f"{var}={expected}")

        if missing_vars:
            print(f"\nRecommended environment variables: {missing_vars}")
            print("Consider setting these for better Unicode support on Windows")

    def test_no_hardcoded_paths(self):
        """Test that no hardcoded Unix paths exist in config"""
        from cyberpuppy.config import Settings

        config = Settings()

        # All paths should be Path objects, not strings
        assert isinstance(config.PROJECT_ROOT, Path)
        if config.DATA_DIR:
            assert isinstance(config.DATA_DIR, Path)
        if config.MODEL_DIR:
            assert isinstance(config.MODEL_DIR, Path)
        if config.CACHE_DIR:
            assert isinstance(config.CACHE_DIR, Path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
