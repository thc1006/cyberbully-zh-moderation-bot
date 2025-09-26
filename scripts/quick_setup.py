#!/usr/bin/env python3
"""
快速設置腳本
一鍵下載並驗證所有必要的資料集
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """主函數"""
    print("[START] CyberPuppy 資料集快速設置")
    print("=" * 50)

    # 檢查腳本位置
    script_dir = Path(__file__).parent
    main_script = script_dir / "complete_dataset_setup.py"

    if not main_script.exists():
        print("[FAIL] 找不到 complete_dataset_setup.py")
        sys.exit(1)

    print("[INFO] 即將下載以下資料集：")
    print("  - COLD Dataset (中文攻擊性語言檢測)")
    print("  - ChnSentiCorp (中文情感分析)")
    print("  - DMSC v2 (大眾點評情感分析)")
    print("  - NTUSD (台大中文情感詞典)")
    print()

    # 確認
    response = input("[QUESTION] 是否繼續？(y/N): ").strip().lower()
    if response not in ['y', 'yes', '是']:
        print("[CANCEL] 取消設置")
        sys.exit(0)

    print("\n[PROCESS] 開始下載資料集...")

    try:
        # 執行主要下載腳本
        result = subprocess.run([
            sys.executable, str(main_script),
            "--dataset", "all"
        ], capture_output=True, text=True, encoding='utf-8')

        # 顯示輸出
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("錯誤:", result.stderr)

        if result.returncode == 0:
            print("\n[SUCCESS] 資料集設置完成！")

            # 執行驗證
            print("\n[VERIFY] 執行最終驗證...")
            verify_result = subprocess.run([
                sys.executable, str(main_script),
                "--verify-only"
            ], capture_output=True, text=True, encoding='utf-8')

            if verify_result.stdout:
                print(verify_result.stdout)

            print("\n[SUMMARY] 設置總結：")
            print("  - 日誌檔案: dataset_setup.log")
            print("  - 驗證報告: data/raw/dataset_report.json")
            print("  - 資料集目錄: data/raw/")
            print("\n[COMPLETE] 全部完成！")

        else:
            print(f"\n[ERROR] 下載過程中出現錯誤 (退出碼: {result.returncode})")
            sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] 執行錯誤: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()