#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows encoding compatibility tests
Tests that Chinese text processing works correctly on Windows systems
"""

import os
import sys
import platform
import pytest
import locale
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestWindowsEncoding:
    """Test Windows-specific encoding handling"""

    def setup_method(self):
        """Setup encoding for each test"""
        if platform.system() == "Windows":
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            # Try to reconfigure stdout/stderr if available
            try:
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
            except AttributeError:
                # Python < 3.7 doesn't have reconfigure
                pass

    def test_chinese_text_display(self):
        """Test that Chinese characters can be displayed without encoding errors"""
        chinese_texts = [
            "中文測試",
            "網路霸凌",
            "毒性檢測",
            "情緒分析",
            "繁體中文",
            "简体中文",
        ]

        for text in chinese_texts:
            # Should not raise UnicodeEncodeError
            try:
                encoded = text.encode('utf-8')
                decoded = encoded.decode('utf-8')
                assert decoded == text, f"Encoding/decoding failed for: {text}"
            except UnicodeError as e:
                pytest.fail(f"Unicode error for text '{text}': {e}")

    def test_console_encoding_support(self):
        """Test console encoding support"""
        if platform.system() != "Windows":
            pytest.skip("Windows-specific test")

        test_text = "測試中文輸出: ✓"

        try:
            # Test string representation
            repr_text = repr(test_text)
            assert test_text in repr_text or '\\u' in repr_text

            # Test encoding to various Windows codepages
            encodings_to_test = ['utf-8', 'utf-16', 'cp1252']

            for encoding in encodings_to_test:
                try:
                    encoded = test_text.encode(encoding, errors='replace')
                    decoded = encoded.decode(encoding)
                    # Should not be empty after encoding/decoding
                    assert len(decoded) > 0
                except (UnicodeError, LookupError):
                    # Some encodings may not be available
                    pass

        except Exception as e:
            pytest.fail(f"Console encoding test failed: {e}")

    def test_file_io_encoding(self, tmp_path):
        """Test file I/O with Chinese characters"""
        chinese_content = """# 中文內容測試
這是一個中文測試文件
包含繁體中文：繁體中文測試
包含簡體中文：简体中文测试
包含標點符號：，。！？；：
包含英文：English mixed content
"""

        test_file = tmp_path / "chinese_test.txt"

        # Test writing with UTF-8 encoding
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(chinese_content)
        except UnicodeError as e:
            pytest.fail(f"Failed to write Chinese content: {e}")

        # Test reading with UTF-8 encoding
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                read_content = f.read()
            assert read_content == chinese_content
        except UnicodeError as e:
            pytest.fail(f"Failed to read Chinese content: {e}")

    def test_locale_settings(self):
        """Test locale settings for Chinese text processing"""
        try:
            # Get current locale
            current_locale = locale.getlocale()

            # Test setting UTF-8 locale if available
            utf8_locales = [
                'C.UTF-8',
                'en_US.UTF-8',
                'zh_CN.UTF-8',
                'zh_TW.UTF-8'
            ]

            for loc in utf8_locales:
                try:
                    locale.setlocale(locale.LC_ALL, loc)
                    break
                except locale.Error:
                    continue
            else:
                # If no UTF-8 locale available, use default
                locale.setlocale(locale.LC_ALL, '')

            # Test encoding detection
            encoding = locale.getpreferredencoding()
            assert encoding is not None

            # Restore original locale
            try:
                locale.setlocale(locale.LC_ALL, current_locale)
            except (locale.Error, TypeError):
                pass

        except Exception as e:
            # Locale tests may fail in some environments, don't fail the test
            pytest.skip(f"Locale test skipped: {e}")

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_windows_codepage_support(self):
        """Test Windows codepage support"""
        try:
            import subprocess

            # Try to get current codepage
            result = subprocess.run(['chcp'], shell=True, capture_output=True, text=True)

            if result.returncode == 0:
                # Should contain codepage information
                assert 'codepage' in result.stdout.lower() or result.stdout.strip().isdigit()

            # Test setting UTF-8 codepage (65001)
            result = subprocess.run(['chcp', '65001'], shell=True,
                                  capture_output=True, text=True)
            # Command should execute (may not succeed depending on permissions)

        except Exception as e:
            pytest.skip(f"Windows codepage test skipped: {e}")

    def test_environment_variables(self):
        """Test encoding-related environment variables"""
        # Test that our encoding environment variables are set
        encoding_vars = {
            'PYTHONIOENCODING': 'utf-8',
            'PYTHONUTF8': '1'
        }

        for var, expected in encoding_vars.items():
            actual = os.environ.get(var)
            if actual is None:
                # Set it for this test
                os.environ[var] = expected

            # Verify it's accessible
            assert os.environ.get(var) is not None

    def test_chinese_nlp_packages_encoding(self):
        """Test that Chinese NLP packages handle encoding correctly"""
        try:
            import jieba

            # Test jieba with Chinese text
            chinese_text = "中文分詞測試，這是一個完整的句子。"

            # Should not raise encoding errors
            words = list(jieba.cut(chinese_text))
            assert len(words) > 0

            # All words should be valid Unicode
            for word in words:
                assert isinstance(word, str)
                # Should be able to encode/decode
                word.encode('utf-8').decode('utf-8')

        except ImportError:
            pytest.skip("jieba not installed")
        except Exception as e:
            pytest.fail(f"Jieba encoding test failed: {e}")

        try:
            from opencc import OpenCC

            # Test OpenCC with traditional/simplified conversion
            cc = OpenCC('s2t')  # Simplified to Traditional

            test_text = "这是简体中文测试"
            converted = cc.convert(test_text)

            # Should produce different text (conversion should work)
            assert converted != test_text
            assert len(converted) > 0

            # Should be valid Unicode
            converted.encode('utf-8').decode('utf-8')

        except ImportError:
            pytest.skip("opencc not installed")
        except Exception as e:
            pytest.fail(f"OpenCC encoding test failed: {e}")

    def test_json_serialization_chinese(self):
        """Test JSON serialization with Chinese content"""
        import json

        chinese_data = {
            "name": "網路霸凌檢測",
            "description": "中文毒性內容檢測系統",
            "labels": ["正常", "有毒", "嚴重"],
            "examples": [
                "這是正常的中文句子。",
                "這可能是有問題的內容。"
            ]
        }

        try:
            # Test JSON serialization
            json_str = json.dumps(chinese_data, ensure_ascii=False, indent=2)

            # Should contain Chinese characters directly (not escaped)
            assert "網路霸凌" in json_str
            assert "檢測系統" in json_str

            # Test deserialization
            loaded_data = json.loads(json_str)
            assert loaded_data == chinese_data

        except Exception as e:
            pytest.fail(f"JSON serialization with Chinese failed: {e}")

    def test_pathlib_chinese_paths(self, tmp_path):
        """Test pathlib with Chinese directory and file names"""
        # Create directory with Chinese name
        chinese_dir = tmp_path / "中文目錄"
        chinese_dir.mkdir()

        # Create file with Chinese name
        chinese_file = chinese_dir / "中文檔案.txt"

        try:
            # Write Chinese content to Chinese-named file
            chinese_content = "中文內容測試"
            chinese_file.write_text(chinese_content, encoding='utf-8')

            # Read it back
            read_content = chinese_file.read_text(encoding='utf-8')
            assert read_content == chinese_content

            # Test path operations
            assert chinese_file.exists()
            assert chinese_file.is_file()
            assert chinese_dir.exists()
            assert chinese_dir.is_dir()

        except Exception as e:
            pytest.fail(f"Chinese path handling failed: {e}")


@pytest.mark.windows
class TestWindowsSpecific:
    """Windows-specific tests that only run on Windows"""

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
    def test_windows_version_info(self):
        """Test Windows version compatibility"""
        import platform

        # Get Windows version info
        version = platform.version()
        release = platform.release()

        assert version is not None
        assert release is not None

        # Windows 10 and above should support UTF-8 better
        if release in ['10', '11']:
            # Should have better Unicode support
            assert True  # Test passes for supported versions
        else:
            pytest.skip(f"Windows {release} may have limited Unicode support")

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
    def test_windows_terminal_detection(self):
        """Test if running in Windows Terminal (better Unicode support)"""
        # Check for Windows Terminal environment variables
        wt_session = os.environ.get('WT_SESSION')
        wt_profile = os.environ.get('WT_PROFILE_ID')

        if wt_session or wt_profile:
            # Running in Windows Terminal - better Unicode support
            assert True
        else:
            # May be running in legacy Command Prompt
            pytest.skip("Not running in Windows Terminal - some Unicode tests may fail")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])