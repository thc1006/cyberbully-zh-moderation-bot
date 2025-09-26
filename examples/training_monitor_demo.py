#!/usr/bin/env python3
"""
CyberPuppy 訓練監控器演示
展示如何使用實時訓練監控器
"""

import sys
import time
import random
from pathlib import Path

# 添加 src 路徑
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberpuppy.training.monitor import TrainingMonitor


def simulate_training():
    """模擬訓練過程"""
    # 訓練參數
    total_epochs = 10
    steps_per_epoch = 100
    model_name = "macbert_aggressive"

    # 創建監控器
    monitor = TrainingMonitor(
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        model_name=model_name,
        use_rich=True,  # 自動偵測 Rich 可用性
        export_dir="./training_metrics"
    )

    try:
        # 開始訓練
        monitor.start_training()

        # 模擬訓練循環
        best_f1 = 0.0
        patience = 3
        early_stop_counter = 0

        for epoch in range(total_epochs):
            monitor.start_epoch(epoch)
            monitor.set_early_stopping(patience, early_stop_counter)

            # 模擬 epoch 內的步驟
            for step in range(steps_per_epoch):
                # 模擬訓練指標
                train_loss = max(0.1, 1.0 - (epoch * 0.1 + step * 0.001) + random.uniform(-0.05, 0.05))
                val_loss = train_loss + random.uniform(0.01, 0.1)
                val_f1 = min(0.95, 0.5 + (epoch * 0.05 + step * 0.0005) + random.uniform(-0.02, 0.02))
                val_accuracy = val_f1 + random.uniform(-0.05, 0.05)
                learning_rate = 2e-5 * (0.9 ** epoch)

                # 更新監控器
                monitor.update_step(
                    step=step,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_f1=val_f1,
                    val_accuracy=val_accuracy,
                    learning_rate=learning_rate
                )

                # 模擬訓練時間
                time.sleep(0.1)

            # 檢查 early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            monitor.end_epoch(epoch)

            # Early stopping
            if early_stop_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

    finally:
        # 停止監控
        monitor.stop_monitoring()

    print(f"\nTraining completed! Best F1: {best_f1:.4f}")


def test_simple_monitor():
    """測試簡單監控器（無 Rich）"""
    from cyberpuppy.training.monitor import SimpleTrainingMonitor

    print("Testing Simple Monitor (no Rich)...")

    monitor = SimpleTrainingMonitor(
        total_epochs=3,
        steps_per_epoch=20,
        model_name="test_model"
    )

    monitor.start_training()

    for epoch in range(3):
        monitor.start_epoch(epoch)

        for step in range(20):
            # 模擬指標
            from cyberpuppy.training.monitor import TrainingMetrics
            metrics = TrainingMetrics(
                epoch=epoch,
                step=step,
                total_steps=20,
                train_loss=0.5 - step * 0.01,
                val_f1=0.7 + step * 0.01,
                val_accuracy=0.75 + step * 0.005,
                learning_rate=1e-4
            )

            monitor.update_step(step, metrics)
            time.sleep(0.05)

        monitor.end_epoch(epoch)

    monitor.stop_monitoring()


def test_gpu_monitor():
    """測試 GPU 監控功能"""
    from cyberpuppy.training.monitor import GPUMonitor

    print("Testing GPU Monitor...")

    gpu_monitor = GPUMonitor()
    print(f"GPU Available: {gpu_monitor.available}")
    print(f"Device Count: {gpu_monitor.device_count}")

    if gpu_monitor.available:
        stats = gpu_monitor.get_memory_stats()
        print(f"GPU Memory: {stats['allocated_gb']:.2f}GB / {stats['total_gb']:.2f}GB")
        print(f"Utilization: {stats['utilization']:.1f}%")
        print(f"Device: {gpu_monitor.get_device_name()}")
    else:
        print("No GPU available - using CPU")


def test_metrics_export():
    """測試指標匯出功能"""
    from cyberpuppy.training.monitor import MetricsHistory

    print("Testing Metrics Export...")

    history = MetricsHistory()

    # 添加模擬指標
    from cyberpuppy.training.monitor import TrainingMetrics
    for i in range(10):
        metrics = TrainingMetrics(
            epoch=0,
            step=i,
            total_steps=10,
            train_loss=1.0 - i * 0.1,
            val_f1=0.5 + i * 0.05,
            val_accuracy=0.6 + i * 0.04
        )
        history.add_metrics(metrics)

    # 匯出到 JSON
    export_path = Path("./test_metrics.json")
    history.export_to_json(export_path)
    print(f"Metrics exported to: {export_path}")

    # 檢查平均值計算
    avg_loss = history.get_recent_avg('train_loss', 5)
    avg_f1 = history.get_recent_avg('val_f1', 5)
    print(f"Recent avg train loss: {avg_loss:.4f}")
    print(f"Recent avg val F1: {avg_f1:.4f}")


if __name__ == "__main__":
    print("CyberPuppy Training Monitor Demo")
    print("=" * 40)

    # 選擇演示模式
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("Available demos:")
        print("  full    - Full training simulation")
        print("  simple  - Simple monitor test")
        print("  gpu     - GPU monitor test")
        print("  export  - Metrics export test")
        print("  all     - Run all tests")
        mode = input("Select demo mode [full]: ").strip() or "full"

    if mode == "full":
        simulate_training()
    elif mode == "simple":
        test_simple_monitor()
    elif mode == "gpu":
        test_gpu_monitor()
    elif mode == "export":
        test_metrics_export()
    elif mode == "all":
        test_gpu_monitor()
        print("\n" + "=" * 40 + "\n")
        test_metrics_export()
        print("\n" + "=" * 40 + "\n")
        test_simple_monitor()
        print("\n" + "=" * 40 + "\n")
        simulate_training()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)