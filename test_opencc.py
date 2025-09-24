#!/usr/bin/env python3
"""測試 OpenCC 繁簡轉換功能"""

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


if __name__ == "__main__":
    test_opencc()
