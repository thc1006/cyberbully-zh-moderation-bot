#!/usr/bin/env python3
"""
CyberPuppy 訓練監控器整合範例
展示如何在實際訓練腳本中整合監控器
"""

import sys
import time
import torch
import torch.nn as nn
from pathlib import Path

# 添加 src 路徑
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cyberpuppy.training.monitor import TrainingMonitor


class SimpleModel(nn.Module):
    """簡單的模型範例"""
    def __init__(self, input_size=100, hidden_size=50, num_classes=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


def training_with_monitor():
    """展示如何在訓練循環中使用監控器"""

    # 訓練配置
    config = {
        'total_epochs': 15,
        'steps_per_epoch': 50,
        'model_name': 'macbert_aggressive',
        'batch_size': 32,
        'learning_rate': 2e-5,
        'early_stopping_patience': 3
    }

    # 初始化模型和優化器（模擬）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # 創建訓練監控器
    monitor = TrainingMonitor(
        total_epochs=config['total_epochs'],
        steps_per_epoch=config['steps_per_epoch'],
        model_name=config['model_name'],
        use_rich=True,  # 自動偵測 Rich 可用性
        export_dir='./training_logs'
    )

    print(f"🚀 Starting training on {device}")
    print(f"📊 Model: {config['model_name']}")
    print(f"🔧 Config: {config}")
    print("=" * 60)

    try:
        # 開始監控
        monitor.start_training()

        # 訓練變數
        best_f1 = 0.0
        patience_counter = 0
        global_step = 0

        for epoch in range(config['total_epochs']):
            monitor.start_epoch(epoch)
            monitor.set_early_stopping(config['early_stopping_patience'], patience_counter)

            # 模擬訓練過程
            epoch_train_losses = []

            for step in range(config['steps_per_epoch']):
                # 模擬前向傳播和損失計算
                batch_size = config['batch_size']
                fake_input = torch.randn(batch_size, 100).to(device)
                fake_target = torch.randint(0, 3, (batch_size,)).to(device)

                optimizer.zero_grad()

                # 前向傳播
                outputs = model(fake_input)
                loss = nn.CrossEntropyLoss()(outputs, fake_target)

                # 反向傳播
                loss.backward()
                optimizer.step()

                train_loss = loss.item()
                epoch_train_losses.append(train_loss)

                # 每10步計算一次驗證指標（模擬）
                if step % 10 == 0 or step == config['steps_per_epoch'] - 1:
                    # 模擬驗證指標（隨時間改善）
                    progress = (epoch * config['steps_per_epoch'] + step) / (config['total_epochs'] * config['steps_per_epoch'])

                    val_loss = train_loss + 0.1 + torch.rand(1).item() * 0.1
                    val_f1 = min(0.95, 0.5 + progress * 0.4 + torch.rand(1).item() * 0.1)
                    val_accuracy = val_f1 + torch.rand(1).item() * 0.05

                    # 更新學習率（模擬 scheduler）
                    current_lr = config['learning_rate'] * (0.95 ** epoch)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr

                    # 更新監控器
                    monitor.update_step(
                        step=step,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        val_f1=val_f1,
                        val_accuracy=val_accuracy,
                        learning_rate=current_lr
                    )

                    # 檢查是否是最佳模型
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        patience_counter = 0
                        # 在實際情況下，這裡會保存最佳模型
                        print(f"💾 New best model saved! F1: {best_f1:.4f}")

                global_step += 1

                # 模擬訓練時間
                time.sleep(0.05)

            # 結束 epoch
            monitor.end_epoch(epoch, export_metrics=True)

            # Early stopping 檢查
            epoch_val_f1 = val_f1  # 使用最後的驗證 F1
            if epoch_val_f1 <= best_f1:
                patience_counter += 1
                if patience_counter >= config['early_stopping_patience']:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
                    print(f"   Best F1: {best_f1:.4f}")
                    break
            else:
                patience_counter = 0

        print(f"\n✅ Training completed!")
        print(f"📈 Final best F1: {best_f1:.4f}")

    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
    finally:
        # 停止監控
        monitor.stop_monitoring()

        # 匯出最終指標
        final_metrics_path = './training_logs/final_training_metrics.json'
        monitor.export_metrics(final_metrics_path)
        print(f"📊 Final metrics exported to: {final_metrics_path}")


def demonstrate_gpu_monitoring():
    """展示 GPU 記憶體監控功能"""
    from cyberpuppy.training.monitor import GPUMonitor

    print("\n🖥️  GPU Memory Monitoring Demo")
    print("=" * 40)

    gpu_monitor = GPUMonitor()

    if gpu_monitor.available:
        print(f"🎯 GPU Available: {gpu_monitor.device_count} device(s)")

        for i in range(gpu_monitor.device_count):
            device_name = gpu_monitor.get_device_name(i)
            stats = gpu_monitor.get_memory_stats(i)

            print(f"\n📱 Device {i}: {device_name}")
            print(f"   Memory: {stats['allocated_gb']:.2f}GB / {stats['total_gb']:.2f}GB")
            print(f"   Utilization: {stats['utilization']:.1f}%")
            print(f"   Reserved: {stats['reserved_gb']:.2f}GB")

        # 分配一些記憶體來展示監控
        print(f"\n🧪 Allocating tensors to demonstrate monitoring...")
        tensors = []
        for i in range(5):
            tensor = torch.randn(1000, 1000).cuda()
            tensors.append(tensor)

            stats = gpu_monitor.get_memory_stats()
            print(f"   Step {i+1}: {stats['allocated_gb']:.2f}GB allocated")
            time.sleep(0.5)

        # 清理記憶體
        del tensors
        torch.cuda.empty_cache()

        final_stats = gpu_monitor.get_memory_stats()
        print(f"   After cleanup: {final_stats['allocated_gb']:.2f}GB allocated")
    else:
        print("❌ No GPU available - using CPU")


if __name__ == "__main__":
    print("CyberPuppy Training Monitor Integration Example")
    print("=" * 50)

    # 展示 GPU 監控
    demonstrate_gpu_monitoring()

    print("\n" + "=" * 50)
    input("Press Enter to start training demo...")

    # 執行訓練示範
    training_with_monitor()

    print("\n🎉 Demo completed!")