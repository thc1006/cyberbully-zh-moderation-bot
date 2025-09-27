#!/usr/bin/env python3
"""直接測試模型載入器

這是一個獨立的測試腳本，不是 pytest 測試。
直接運行：python tests/api_tests/test_model_directly.py
"""
import sys
import os


def main():
    """主測試函數"""
    # 添加路徑
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))
    sys.path.insert(0, os.path.dirname(__file__))

    print("="*60)
    print("直接測試模型載入器")
    print("="*60)

    # 測試 import
    try:
        print("\n1. 測試 import model_loader_fixed...")
        from api.model_loader_fixed import get_model_loader
        print("   [OK] import 成功")
    except ImportError as e:
        print(f"   [ERROR] import 失敗: {e}")
        print("\n2. 嘗試 import model_loader_simple...")
        try:
            from api.model_loader_simple import get_model_loader
            print("   [OK] 使用 simple 版本")
        except ImportError as e2:
            print(f"   [ERROR] 兩個版本都失敗: {e2}")
            return 1

    # 測試模型載入
    try:
        print("\n3. 初始化模型載入器...")
        loader = get_model_loader()
        print("   [OK] 載入器初始化成功")

        print("\n4. 測試狀態檢查...")
        status = loader.get_status()
        print(f"   模型載入: {status.get('models_loaded', False)}")
        print(f"   設備: {status.get('device', 'unknown')}")
        print(f"   準備就緒: {status.get('warmup_complete', False)}")

        print("\n5. 測試預測功能...")
        test_texts = [
            "你好世界",
            "今天天氣很好",
            "你這個笨蛋"
        ]

        for text in test_texts:
            print(f"\n   測試文本: '{text}'")
            try:
                result = loader.predict(text)
                if result:
                    print(f"   毒性: {result.get('toxicity', {}).get('level', 'N/A')}")
                    print(f"   情緒: {result.get('emotion', {}).get('label', 'N/A')}")
                else:
                    print("   [WARNING] 預測返回空結果")
            except Exception as e:
                print(f"   [ERROR] 預測失敗: {str(e)[:100]}")
                import traceback
                print("   詳細錯誤:")
                traceback.print_exc()

    except Exception as e:
        print(f"\n[ERROR] 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())