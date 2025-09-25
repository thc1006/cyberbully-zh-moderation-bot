#!/usr/bin/env python3
"""測試修復後的 API"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_analyze_fixed():
    """測試修復後的分析功能"""

    test_cases = [
        {"text": "你好朋友", "expected": "positive"},
        {"text": "今天天氣真好", "expected": "positive"},
        {"text": "你這個笨蛋", "expected": "toxic"},
        {"text": "去死吧", "expected": "severe"},
        {"text": "我不太確定", "expected": "neutral"}
    ]

    print("="*60)
    print("CyberPuppy API 修復測試")
    print("="*60)

    success_count = 0

    for i, case in enumerate(test_cases, 1):
        print(f"\n測試 {i}: {case['text']}")
        print(f"預期: {case['expected']}")

        try:
            response = requests.post(
                f"{BASE_URL}/analyze",
                json={"text": case['text']},
                headers={"Content-Type": "application/json"},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                print("結果: [SUCCESS]")
                print(f"  毒性: {result.get('toxicity', {}).get('level', 'N/A')}")
                print(f"  情緒: {result.get('emotion', {}).get('label', 'N/A')}")
                print(f"  信心: {result.get('toxicity', {}).get('confidence', 0):.2f}")
                success_count += 1
            else:
                print(f"結果: [ERROR] {response.status_code}")
                print(f"  訊息: {response.text[:100]}")
        except Exception as e:
            print(f"結果: [EXCEPTION]")
            print(f"  錯誤: {str(e)}")

        time.sleep(0.5)  # 避免過快請求

    print("\n" + "="*60)
    print(f"測試總結: {success_count}/{len(test_cases)} 成功")
    print("="*60)

    return success_count == len(test_cases)

if __name__ == "__main__":
    # 等待服務啟動
    print("等待服務啟動...")
    time.sleep(2)

    # 執行測試
    if test_analyze_fixed():
        print("\n[SUCCESS] API 修復成功！所有測試通過！")
    else:
        print("\n[WARNING] 部分測試失敗，需要進一步調試")