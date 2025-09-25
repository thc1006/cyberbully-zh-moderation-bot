#!/usr/bin/env python3
"""測試 API 功能"""
import requests
import json

# API 端點
BASE_URL = "http://localhost:8000"

def test_health():
    """測試健康檢查"""
    response = requests.get(f"{BASE_URL}/healthz")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()

def test_analyze(text):
    """測試文本分析"""
    response = requests.post(
        f"{BASE_URL}/analyze",
        json={"text": text},
        headers={"Content-Type": "application/json"}
    )

    print(f"分析文本: {text}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    else:
        print(f"錯誤: {response.status_code} - {response.text}")
    print("-" * 50)

if __name__ == "__main__":
    # 測試健康檢查
    test_health()

    # 測試不同類型的文本
    test_texts = [
        "你真是個好人",
        "我很喜歡這個產品",
        "你這個廢物",
        "去死吧",
        "今天天氣不錯"
    ]

    for text in test_texts:
        test_analyze(text)