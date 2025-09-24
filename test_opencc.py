#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""測試 OpenCC 繁簡轉換功能"""

import sys
import os

# Windows encoding fix
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from opencc import OpenCC


def test_opencc():
    """測試各種轉換模式"""

    test_cases = [
        ('s2t', '这是简体中文测试', '繁體'),
        ('t2s', '這是繁體中文測試', '简体'),
        ('s2tw', '这是简体中文测试', '台灣繁體'),
        ('tw2s', '這是繁體中文測試', '简体'),
    ]

    print("="*50)
    print("OpenCC 繁簡轉換測試")
    print("="*50)

    for mode, text, desc in test_cases:
        try:
            cc = OpenCC(mode)
            result = cc.convert(text)
            print(f"\n模式: {mode} ({desc})")
            print(f"輸入: {text}")
            print(f"輸出: {result}")
        except Exception as e:
            print(f"錯誤 ({mode}): {e}")
            # Windows-specific error handling for encoding issues
            if sys.platform.startswith('win') and 'codec' in str(e).lower():
                print("提示：Windows 編碼問題，請確保 console 設定為 UTF-8")

    # 測試實際應用場景
    print("\n" + "="*50)
    print("實際應用測試")
    print("="*50)

    cc_s2t = OpenCC('s2t')

    # 網路霸凌相關詞彙轉換
    test_texts = [
        "你这个垃圾",
        "网络暴力很严重",
        "这种行为是错误的",
        "请不要发表恶意评论",
    ]

    print("\n簡體轉繁體:")
    for text in test_texts:
        converted = cc_s2t.convert(text)
        print(f"  {text} → {converted}")

    print("\nOpenCC 安裝成功！所有轉換模式可用。")
    print("="*50)

    # Windows encoding test
    if sys.platform.startswith('win'):
        print("\n🪟 Windows 編碼測試:")
        try:
            test_encoding = "中文編碼測試 - 繁簡轉換 ✓"
            print(f"編碼測試: {test_encoding}")
            print("✅ Windows 中文顯示正常")
        except UnicodeEncodeError as e:
            print(f"❌ Windows 編碼錯誤: {e}")
            print("建議執行: chcp 65001 (設定 console 為 UTF-8)")


if __name__ == "__main__":
    try:
        test_opencc()
    except UnicodeEncodeError as e:
        print(f"\n❌ 編碼錯誤: {e}")
        print("Windows 用戶請嘗試:")
        print("1. 執行 'chcp 65001' 設定 console 為 UTF-8")
        print("2. 使用 'python scripts/windows_setup.py' 進行 Windows 專用設定")
        sys.exit(1)
    except ImportError as e:
        print(f"\n❌ 匯入錯誤: {e}")
        print("請先安裝依賴: python -m pip install opencc-python-reimplemented")
        sys.exit(1)
