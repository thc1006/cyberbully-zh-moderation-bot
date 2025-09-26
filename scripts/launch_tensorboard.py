#!/usr/bin/env python3
"""
TensorBoard 啟動和管理腳本
支援多實驗監控和自動化視覺化

功能特色:
1. 自動偵測實驗目錄
2. 多實驗比較
3. 自動化報告生成
4. 性能指標監控
"""

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import List, Optional

import yaml


def find_tensorboard_logs(base_dir: str = "experiments") -> List[Path]:
    """尋找所有TensorBoard日誌目錄"""
    base_path = Path(base_dir)
    tensorboard_dirs = []

    if base_path.exists():
        # 尋找所有tensorboard_logs目錄
        for root, dirs, files in os.walk(base_path):
            if "tensorboard_logs" in dirs:
                tensorboard_dirs.append(Path(root) / "tensorboard_logs")

    return tensorboard_dirs


def get_recent_experiments(base_dir: str = "experiments", limit: int = 5) -> List[Path]:
    """取得最近的實驗"""
    tensorboard_dirs = find_tensorboard_logs(base_dir)

    # 按修改時間排序
    tensorboard_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return tensorboard_dirs[:limit]


def launch_tensorboard(log_dirs: List[Path], port: int = 6006,
                      open_browser: bool = True) -> subprocess.Popen:
    """啟動TensorBoard"""
    if not log_dirs:
        raise ValueError("沒有找到TensorBoard日誌目錄")

    # 建立logdir參數
    if len(log_dirs) == 1:
        logdir_arg = str(log_dirs[0])
    else:
        # 多個實驗目錄
        logdir_parts = []
        for i, log_dir in enumerate(log_dirs):
            experiment_name = log_dir.parent.name
            logdir_parts.append(f"{experiment_name}:{log_dir}")
        logdir_arg = ",".join(logdir_parts)

    # 啟動TensorBoard
    cmd = [
        "tensorboard",
        "--logdir", logdir_arg,
        "--port", str(port),
        "--host", "localhost"
    ]

    print(f"🚀 啟動TensorBoard...")
    print(f"📊 日誌目錄: {logdir_arg}")
    print(f"🌐 URL: http://localhost:{port}")

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)

        # 等待啟動
        time.sleep(3)

        if process.poll() is None:  # 進程仍在運行
            print(f"✅ TensorBoard 已啟動 (PID: {process.pid})")

            if open_browser:
                webbrowser.open(f"http://localhost:{port}")
                print(f"🌐 已在瀏覽器中打開")
        else:
            stdout, stderr = process.communicate()
            print(f"❌ TensorBoard 啟動失敗")
            print(f"Error: {stderr}")
            return None

        return process

    except FileNotFoundError:
        print("❌ TensorBoard 未安裝")
        print("請執行: pip install tensorboard")
        return None
    except Exception as e:
        print(f"❌ 啟動TensorBoard時發生錯誤: {e}")
        return None


def monitor_training_progress(experiment_dir: Path, check_interval: int = 30):
    """監控訓練進度"""
    print(f"📈 監控訓練進度: {experiment_dir}")
    print(f"🔄 檢查間隔: {check_interval}秒")
    print("按 Ctrl+C 停止監控")

    try:
        while True:
            # 檢查最新指標
            results_file = experiment_dir / "final_results.json"
            checkpoints_dir = experiment_dir / "checkpoints"

            if results_file.exists():
                print("🎉 訓練已完成!")
                # 顯示最終結果
                import json
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)

                test_metrics = results.get('test_metrics', {})
                targets = results.get('target_achieved', {})

                print(f"📊 最終結果:")
                print(f"  霸凌F1: {test_metrics.get('bullying_f1', 0):.4f} ({'✅' if targets.get('bullying_f1_075', False) else '❌'})")
                print(f"  毒性F1: {test_metrics.get('toxicity_f1', 0):.4f} ({'✅' if targets.get('toxicity_f1_078', False) else '❌'})")
                print(f"  總體F1: {test_metrics.get('overall_macro_f1', 0):.4f} ({'✅' if targets.get('overall_macro_f1_076', False) else '❌'})")
                break

            elif checkpoints_dir.exists():
                # 檢查最新檢查點
                checkpoints = list(checkpoints_dir.glob("*.ckpt"))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    print(f"📁 最新檢查點: {latest_checkpoint.name} ({time.ctime(latest_checkpoint.stat().st_mtime)})")

            else:
                print(f"⏳ 等待訓練開始...")

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n⏹️  監控已停止")


def compare_experiments(experiment_dirs: List[Path]):
    """比較多個實驗結果"""
    print(f"📊 比較 {len(experiment_dirs)} 個實驗:")

    results = []
    for exp_dir in experiment_dirs:
        results_file = exp_dir / "final_results.json"
        if results_file.exists():
            import json
            with open(results_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                result['experiment_dir'] = str(exp_dir)
                results.append(result)

    if not results:
        print("❌ 沒有找到完整的實驗結果")
        return

    # 顯示比較表格
    print(f"\n{'實驗名稱':<20} {'霸凌F1':<8} {'毒性F1':<8} {'總體F1':<8} {'目標達成':<8}")
    print("-" * 60)

    for result in results:
        exp_name = result.get('experiment_name', 'Unknown')[:19]
        test_metrics = result.get('test_metrics', {})
        targets = result.get('target_achieved', {})

        bullying_f1 = test_metrics.get('bullying_f1', 0)
        toxicity_f1 = test_metrics.get('toxicity_f1', 0)
        overall_f1 = test_metrics.get('overall_macro_f1', 0)

        all_achieved = all(targets.values()) if targets else False

        print(f"{exp_name:<20} {bullying_f1:<8.4f} {toxicity_f1:<8.4f} {overall_f1:<8.4f} {'✅' if all_achieved else '❌':<8}")


def main():
    parser = argparse.ArgumentParser(description="TensorBoard 啟動和管理工具")
    parser.add_argument(
        "--log-dir", "-l",
        type=str,
        help="指定TensorBoard日誌目錄"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        help="指定實驗名稱"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=6006,
        help="TensorBoard端口 (預設: 6006)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="不自動打開瀏覽器"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="監控訓練進度"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="比較多個實驗"
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=5,
        help="顯示最近的N個實驗 (預設: 5)"
    )

    args = parser.parse_args()

    # 專案根目錄
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    print("🎯 TensorBoard 管理工具")
    print(f"📁 專案根目錄: {project_root}")

    if args.log_dir:
        # 指定日誌目錄
        log_dirs = [Path(args.log_dir)]
    elif args.experiment:
        # 指定實驗
        exp_dir = Path("experiments") / args.experiment
        if not exp_dir.exists():
            print(f"❌ 實驗目錄不存在: {exp_dir}")
            return 1
        log_dirs = [exp_dir / "tensorboard_logs"]
    else:
        # 自動尋找
        recent_experiments = get_recent_experiments(limit=args.recent)

        if not recent_experiments:
            print("❌ 沒有找到TensorBoard日誌目錄")
            print("請先執行訓練或指定日誌目錄")
            return 1

        print(f"📋 找到 {len(recent_experiments)} 個最近的實驗:")
        for i, log_dir in enumerate(recent_experiments):
            exp_name = log_dir.parent.name
            mod_time = time.ctime(log_dir.stat().st_mtime)
            print(f"  {i+1}. {exp_name} ({mod_time})")

        if args.compare:
            # 比較模式
            compare_experiments([log_dir.parent for log_dir in recent_experiments])
            return 0

        # 選擇實驗
        if len(recent_experiments) == 1:
            log_dirs = recent_experiments
        else:
            try:
                choice = input(f"\n選擇實驗 (1-{len(recent_experiments)}, 或 'all'): ").strip()
                if choice.lower() == 'all':
                    log_dirs = recent_experiments
                else:
                    idx = int(choice) - 1
                    if 0 <= idx < len(recent_experiments):
                        log_dirs = [recent_experiments[idx]]
                    else:
                        print("❌ 無效選擇")
                        return 1
            except (ValueError, KeyboardInterrupt):
                print("❌ 操作取消")
                return 1

    # 檢查日誌目錄是否存在
    valid_log_dirs = [log_dir for log_dir in log_dirs if log_dir.exists()]
    if not valid_log_dirs:
        print("❌ 沒有找到有效的TensorBoard日誌目錄")
        return 1

    if args.monitor:
        # 監控模式
        if len(valid_log_dirs) != 1:
            print("❌ 監控模式只能選擇一個實驗")
            return 1

        experiment_dir = valid_log_dirs[0].parent
        monitor_training_progress(experiment_dir)
        return 0

    # 啟動TensorBoard
    process = launch_tensorboard(
        valid_log_dirs,
        port=args.port,
        open_browser=not args.no_browser
    )

    if process is None:
        return 1

    try:
        print("\n按 Ctrl+C 停止TensorBoard")
        process.wait()
    except KeyboardInterrupt:
        print("\n⏹️  正在停止TensorBoard...")
        process.terminate()
        process.wait()
        print("✅ TensorBoard 已停止")

    return 0


if __name__ == "__main__":
    sys.exit(main())