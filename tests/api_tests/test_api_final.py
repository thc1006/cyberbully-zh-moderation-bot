#!/usr/bin/env python3
"""最終 API 測試"""
import requests
import json

url = "http://localhost:8000/analyze"
headers = {"Content-Type": "application/json"}

test_cases = [
    "你好朋友",
    "今天天氣真好",
    "你這個笨蛋",
    "去死吧",
    "謝謝你"
]

print("CyberPuppy API 最終測試")
print("="*40)

for text in test_cases:
    print(f"\n測試: {text}")

    try:
        response = requests.post(url, json={"text": text}, headers=headers)

        if response.status_code == 200:
            data = response.json()
            print(f"[SUCCESS]")
            print(f"  毒性: {data['toxicity']['level']}")
            print(f"  情緒: {data['emotion']['label']}")
        else:
            print(f"[ERROR] {response.status_code}")
            print(f"  {response.text[:100]}")
    except Exception as e:
        print(f"[EXCEPTION] {e}")

print("\n" + "="*40)
print("測試完成！")