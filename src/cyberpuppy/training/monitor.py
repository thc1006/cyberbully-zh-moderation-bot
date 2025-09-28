"""
CyberPuppy 實時訓練監控器
提供美觀的實時訓練進度顯示，支援 GPU 記憶體監控、最佳模型追蹤等功能
"""

import json
import os
import platform
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

# Rich imports with fallback
try:
    from rich import box
    from rich.align import Align
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                               SpinnerColumn, TaskProgressColumn, TextColumn,
                               TimeRemainingColumn)
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Import tqdm regardless (needed for fallback)
# Always import tqdm for fallback
from tqdm.auto import tqdm


@dataclass
class TrainingMetrics:
    """訓練指標數據結構"""

    epoch: int
    step: int
    total_steps: int
    train_loss: float
    val_loss: Optional[float] = None
    val_f1: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float = 0.0
    best_f1: float = 0.0
    best_epoch: int = 0
    early_stopping_patience: int = 0
    early_stopping_counter: int = 0
    eta_seconds: float = 0.0
    epoch_time: float = 0.0
    step_time: float = 0.0
    gpu_memory_allocated: float = 0.0
    gpu_memory_reserved: float = 0.0
    gpu_memory_total: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class GPUMonitor:
    """GPU 記憶體監控器"""

    def __init__(self):
        self.available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.available else 0
        self.device_names = []

        if self.available:
            for i in range(self.device_count):
                self.device_names.append(torch.cuda.get_device_name(i))

    def get_memory_stats(self, device: int = 0) -> Dict[str, float]:
        """獲取 GPU 記憶體統計"""
        if not self.available or device >= self.device_count:
            return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0, "utilization": 0.0}

        # 獲取當前設備記憶體統計
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3

        # 獲取總記憶體
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3

        utilization = (allocated / total * 100) if total > 0 else 0.0

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "utilization": utilization,
        }

    def get_device_name(self, device: int = 0) -> str:
        """獲取設備名稱"""
        if self.available and device < len(self.device_names):
            return self.device_names[device]
        return "CPU"


class MetricsHistory:
    """訓練指標歷史記錄"""

    def __init__(self, maxlen: int = 1000):
        self.maxlen = maxlen
        self.train_losses = deque(maxlen=maxlen)
        self.val_losses = deque(maxlen=maxlen)
        self.val_f1_scores = deque(maxlen=maxlen)
        self.val_accuracies = deque(maxlen=maxlen)
        self.learning_rates = deque(maxlen=maxlen)
        self.timestamps = deque(maxlen=maxlen)
        self.epochs = deque(maxlen=maxlen)
        self.steps = deque(maxlen=maxlen)

    def add_metrics(self, metrics: TrainingMetrics):
        """添加訓練指標"""
        self.train_losses.append(metrics.train_loss)
        self.val_losses.append(metrics.val_loss)
        self.val_f1_scores.append(metrics.val_f1)
        self.val_accuracies.append(metrics.val_accuracy)
        self.learning_rates.append(metrics.learning_rate)
        self.timestamps.append(metrics.timestamp)
        self.epochs.append(metrics.epoch)
        self.steps.append(metrics.step)

    def get_recent_avg(self, metric_name: str, window: int = 10) -> float:
        """獲取最近 N 個指標的平均值"""
        metrics_map = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "val_f1": self.val_f1_scores,
            "val_accuracy": self.val_accuracies,
            "learning_rate": self.learning_rates,
        }

        if metric_name not in metrics_map:
            return 0.0

        metrics = metrics_map[metric_name]
        if not metrics:
            return 0.0

        recent_metrics = [m for m in list(metrics)[-window:] if m is not None]
        return np.mean(recent_metrics) if recent_metrics else 0.0

    def export_to_json(self, filepath: Union[str, Path]) -> None:
        """匯出歷史記錄到 JSON"""
        data = {
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses),
            "val_f1_scores": list(self.val_f1_scores),
            "val_accuracies": list(self.val_accuracies),
            "learning_rates": list(self.learning_rates),
            "timestamps": list(self.timestamps),
            "epochs": list(self.epochs),
            "steps": list(self.steps),
            "exported_at": datetime.now().isoformat(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class RichTrainingMonitor:
    """Rich 美觀訓練監控器"""

    def __init__(self, total_epochs: int, steps_per_epoch: int, model_name: str = "CyberPuppy"):
        self.console = Console()
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.model_name = model_name
        self.gpu_monitor = GPUMonitor()
        self.metrics_history = MetricsHistory()

        # 進度追蹤
        self.current_epoch = 0
        self.current_step = 0
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        self.step_times = deque(maxlen=50)  # 保留最近50步的時間

        # 最佳指標追蹤
        self.best_metrics = {"f1": 0.0, "accuracy": 0.0, "epoch": 0}

        # Early stopping
        self.early_stopping_patience = 0
        self.early_stopping_counter = 0

        # 設定進度條
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )

        self.epoch_task = None
        self.step_task = None
        self.live = None

    def start_training(self):
        """開始訓練監控"""
        self.start_time = time.time()

        # 建立進度任務
        self.epoch_task = self.progress.add_task("[cyan]Epochs", total=self.total_epochs)

        # 啟動 Live 顯示
        layout = self._create_layout()
        self.live = Live(layout, console=self.console, refresh_per_second=4)
        self.live.start()

    def start_epoch(self, epoch: int):
        """開始新的 epoch"""
        self.current_epoch = epoch
        self.current_step = 0
        self.epoch_start_time = time.time()

        # 更新 epoch 進度
        if self.epoch_task is not None:
            self.progress.update(self.epoch_task, completed=epoch)

        # 重置或建立 step 任務
        if self.step_task is not None:
            self.progress.remove_task(self.step_task)

        self.step_task = self.progress.add_task(
            f"[green]Epoch {epoch+1}/{self.total_epochs}", total=self.steps_per_epoch
        )

    def update_step(self, step: int, metrics: TrainingMetrics):
        """更新步驟進度和指標"""
        self.current_step = step

        # 記錄步驟時間
        current_time = time.time()
        if len(self.step_times) > 0:
            step_time = current_time - self.step_times[-1]
        else:
            step_time = 0.0
        self.step_times.append(current_time)

        # 更新指標
        metrics.step_time = step_time
        self.metrics_history.add_metrics(metrics)

        # 更新步驟進度
        if self.step_task is not None:
            self.progress.update(self.step_task, completed=step)

        # 更新最佳指標
        if metrics.val_f1 and metrics.val_f1 > self.best_metrics["f1"]:
            self.best_metrics["f1"] = metrics.val_f1
            self.best_metrics["epoch"] = metrics.epoch

        if metrics.val_accuracy and metrics.val_accuracy > self.best_metrics["accuracy"]:
            self.best_metrics["accuracy"] = metrics.val_accuracy

    def end_epoch(self, epoch: int):
        """結束 epoch"""
        if self.step_task is not None:
            self.progress.update(self.step_task, completed=self.steps_per_epoch)

    def set_early_stopping(self, patience: int, counter: int):
        """設定 early stopping 狀態"""
        self.early_stopping_patience = patience
        self.early_stopping_counter = counter

    def _create_layout(self) -> Layout:
        """建立顯示布局"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=7),
            Layout(name="progress", size=6),
            Layout(name="metrics", size=10),
            Layout(name="footer", size=3),
        )

        return layout

    def _update_display(self):
        """更新顯示內容"""
        if not self.live:
            return

        # 更新各個區域
        self.live.update(self._create_layout())

    def _create_header(self) -> Panel:
        """建立標題區域"""
        gpu_info = self.gpu_monitor.get_memory_stats()
        gpu_name = self.gpu_monitor.get_device_name()

        header_text = f"""[bold cyan]CyberPuppy Local Training Monitor[/bold cyan]
[bold]Model:[/bold] {self.model_name}
[bold]GPU:[/bold] {gpu_name} ({gpu_info['allocated_gb']:.1f}GB / {gpu_info['total_gb']:.1f}GB)"""

        return Panel(Align.center(header_text), box=box.DOUBLE, style="blue")

    def _create_metrics_table(self) -> Table:
        """建立指標表格"""
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", width=15)
        table.add_column("Current", style="green", justify="right", width=12)
        table.add_column("Best", style="yellow", justify="right", width=12)
        table.add_column("Trend", style="blue", justify="center", width=8)

        # 獲取最近指標
        recent_train_loss = self.metrics_history.get_recent_avg("train_loss", 1)
        recent_val_f1 = self.metrics_history.get_recent_avg("val_f1", 1)
        recent_val_acc = self.metrics_history.get_recent_avg("val_accuracy", 1)

        # 計算趨勢
        train_loss_trend = self._get_trend("train_loss")
        f1_trend = self._get_trend("val_f1")
        acc_trend = self._get_trend("val_accuracy")

        table.add_row("Train Loss", f"{recent_train_loss:.4f}", "-", train_loss_trend)
        table.add_row("Val F1", f"{recent_val_f1:.4f}", f"{self.best_metrics['f1']:.4f}", f1_trend)
        table.add_row(
            "Val Accuracy",
            f"{recent_val_acc:.4f}",
            f"{self.best_metrics['accuracy']:.4f}",
            acc_trend,
        )

        return table

    def _get_trend(self, metric_name: str) -> str:
        """獲取指標趨勢"""
        current = self.metrics_history.get_recent_avg(metric_name, 1)
        previous = self.metrics_history.get_recent_avg(metric_name, 2)

        if current == 0 or previous == 0:
            return "➡️"

        if current > previous:
            return "⬆️" if metric_name != "train_loss" else "⬇️"
        elif current < previous:
            return "⬇️" if metric_name != "train_loss" else "⬆️"
        else:
            return "➡️"

    def _estimate_eta(self) -> str:
        """估算剩餘時間"""
        if len(self.step_times) < 2:
            return "Calculating..."

        # 計算平均步驟時間
        recent_times = list(self.step_times)[-10:]  # 使用最近10步
        if len(recent_times) < 2:
            return "Calculating..."

        step_durations = [
            recent_times[i] - recent_times[i - 1] for i in range(1, len(recent_times))
        ]
        avg_step_time = np.mean(step_durations)

        # 計算剩餘步驟
        remaining_steps_this_epoch = self.steps_per_epoch - self.current_step
        remaining_epochs = self.total_epochs - self.current_epoch - 1
        total_remaining_steps = remaining_steps_this_epoch + (
            remaining_epochs * self.steps_per_epoch
        )

        eta_seconds = total_remaining_steps * avg_step_time
        eta_timedelta = timedelta(seconds=int(eta_seconds))

        return str(eta_timedelta)

    def stop_monitoring(self):
        """停止監控"""
        if self.live:
            self.live.stop()

    def export_metrics(self, filepath: Union[str, Path]):
        """匯出訓練指標"""
        self.metrics_history.export_to_json(filepath)


class SimpleTrainingMonitor:
    """簡單的 tqdm 訓練監控器（Rich 不可用時的備選）"""

    def __init__(self, total_epochs: int, steps_per_epoch: int, model_name: str = "CyberPuppy"):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.model_name = model_name
        self.gpu_monitor = GPUMonitor()
        self.metrics_history = MetricsHistory()

        self.current_epoch = 0
        self.current_step = 0
        self.start_time = time.time()
        self.epoch_start_time = time.time()

        self.best_metrics = {"f1": 0.0, "accuracy": 0.0, "epoch": 0}

        self.early_stopping_patience = 0
        self.early_stopping_counter = 0

        self.epoch_pbar = None
        self.step_pbar = None

    def start_training(self):
        """開始訓練監控"""
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"  {self.model_name} Local Training Monitor")
        print(f"{'='*60}")

        gpu_info = self.gpu_monitor.get_memory_stats()
        gpu_name = self.gpu_monitor.get_device_name()
        print(f"Model: {self.model_name}")
        print(f"GPU: {gpu_name} ({gpu_info['allocated_gb']:.1f}GB / {gpu_info['total_gb']:.1f}GB)")
        print("")

        self.epoch_pbar = tqdm(
            total=self.total_epochs,
            desc="Epochs",
            position=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

    def start_epoch(self, epoch: int):
        """開始新的 epoch"""
        self.current_epoch = epoch
        self.current_step = 0
        self.epoch_start_time = time.time()

        if self.epoch_pbar:
            self.epoch_pbar.set_description(f"Epoch {epoch+1}/{self.total_epochs}")

        if self.step_pbar:
            self.step_pbar.close()

        self.step_pbar = tqdm(
            total=self.steps_per_epoch,
            desc=f"Steps (Epoch {epoch+1})",
            position=1,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )

    def update_step(self, step: int, metrics: TrainingMetrics):
        """更新步驟進度和指標"""
        self.current_step = step
        self.metrics_history.add_metrics(metrics)

        # 更新最佳指標
        if metrics.val_f1 and metrics.val_f1 > self.best_metrics["f1"]:
            self.best_metrics["f1"] = metrics.val_f1
            self.best_metrics["epoch"] = metrics.epoch

        # 更新進度條
        if self.step_pbar:
            postfix = {
                "Loss": f"{metrics.train_loss:.4f}",
                "F1": f"{metrics.val_f1:.4f}" if metrics.val_f1 else "N/A",
                "Best F1": f"{self.best_metrics['f1']:.4f}",
                "ES": f"{self.early_stopping_counter}/{self.early_stopping_patience}",
            }
            self.step_pbar.set_postfix(postfix)
            self.step_pbar.update(1)

    def end_epoch(self, epoch: int):
        """結束 epoch"""
        if self.epoch_pbar:
            self.epoch_pbar.update(1)

    def set_early_stopping(self, patience: int, counter: int):
        """設定 early stopping 狀態"""
        self.early_stopping_patience = patience
        self.early_stopping_counter = counter

    def stop_monitoring(self):
        """停止監控"""
        if self.step_pbar:
            self.step_pbar.close()
        if self.epoch_pbar:
            self.epoch_pbar.close()

        print("\nTraining completed!")
        print(f"Best F1: {self.best_metrics['f1']:.4f} (Epoch {self.best_metrics['epoch']+1})")

    def export_metrics(self, filepath: Union[str, Path]):
        """匯出訓練指標"""
        self.metrics_history.export_to_json(filepath)


class TrainingMonitor:
    """統一的訓練監控接口"""

    def __init__(
        self,
        total_epochs: int,
        steps_per_epoch: int,
        model_name: str = "CyberPuppy",
        use_rich: bool = True,
        export_dir: Optional[Union[str, Path]] = None,
    ):
        """
        初始化訓練監控器

        Args:
            total_epochs: 總 epoch 數
            steps_per_epoch: 每個 epoch 的步驟數
            model_name: 模型名稱
            use_rich: 是否使用 Rich 美觀顯示（自動偵測可用性）
            export_dir: 指標匯出目錄
        """
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.model_name = model_name
        self.export_dir = Path(export_dir) if export_dir else Path("./training_metrics")
        self.export_dir.mkdir(exist_ok=True)

        # 自動選擇監控器
        if use_rich and HAS_RICH and self._supports_rich():
            self.monitor = RichTrainingMonitor(total_epochs, steps_per_epoch, model_name)
            self.monitor_type = "rich"
        else:
            self.monitor = SimpleTrainingMonitor(total_epochs, steps_per_epoch, model_name)
            self.monitor_type = "simple"

        print(f"Using {self.monitor_type} training monitor")

    def _supports_rich(self) -> bool:
        """檢查是否支援 Rich 顯示"""
        # 檢查是否在支援的終端環境中
        if platform.system() == "Windows":
            # Windows Terminal, ConEmu, 等支援 Rich
            return (
                os.environ.get("WT_SESSION") or os.environ.get("ConEmuANSI") or sys.stdout.isatty()
            )
        else:
            # Unix-like 系統一般都支援
            return sys.stdout.isatty()

    def start_training(self):
        """開始訓練監控"""
        self.monitor.start_training()

    def start_epoch(self, epoch: int):
        """開始新的 epoch"""
        self.monitor.start_epoch(epoch)

    def update_step(
        self,
        step: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        val_f1: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        learning_rate: float = 0.0,
        **kwargs,
    ):
        """
        更新步驟進度和指標

        Args:
            step: 當前步驟
            train_loss: 訓練損失
            val_loss: 驗證損失
            val_f1: 驗證 F1 分數
            val_accuracy: 驗證準確率
            learning_rate: 學習率
            **kwargs: 其他指標
        """
        # 獲取 GPU 記憶體資訊
        gpu_stats = self.monitor.gpu_monitor.get_memory_stats()

        metrics = TrainingMetrics(
            epoch=self.monitor.current_epoch,
            step=step,
            total_steps=self.steps_per_epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_f1=val_f1,
            val_accuracy=val_accuracy,
            learning_rate=learning_rate,
            best_f1=self.monitor.best_metrics["f1"],
            best_epoch=self.monitor.best_metrics["epoch"],
            early_stopping_patience=self.monitor.early_stopping_patience,
            early_stopping_counter=self.monitor.early_stopping_counter,
            gpu_memory_allocated=gpu_stats["allocated_gb"],
            gpu_memory_reserved=gpu_stats["reserved_gb"],
            gpu_memory_total=gpu_stats["total_gb"],
        )

        self.monitor.update_step(step, metrics)

    def end_epoch(self, epoch: int, export_metrics: bool = True):
        """結束 epoch"""
        self.monitor.end_epoch(epoch)

        # 自動匯出指標
        if export_metrics:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_metrics_epoch_{epoch:03d}_{timestamp}.json"
            filepath = self.export_dir / filename
            self.export_metrics(filepath)

    def set_early_stopping(self, patience: int, counter: int):
        """設定 early stopping 狀態"""
        self.monitor.set_early_stopping(patience, counter)

    def stop_monitoring(self):
        """停止監控"""
        self.monitor.stop_monitoring()

    def export_metrics(self, filepath: Union[str, Path]):
        """匯出訓練指標"""
        self.monitor.export_metrics(filepath)
        print(f"Metrics exported to: {filepath}")


# 向後相容性
class LegacyTrainingMonitor:
    """向後相容的簡化監控器（與 utils.py 中的介面一致）"""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_times = []
        self.losses = []
        self.start_time = None

    def start_epoch(self):
        """開始一個epoch"""
        self.start_time = time.time()
        self.step_times = []
        self.losses = []

    def log_step(self, step: int, loss: float, lr: float):
        """記錄一個步驟"""
        step_time = time.time()
        if len(self.step_times) > 0:
            time_per_step = step_time - self.step_times[-1]
        else:
            time_per_step = 0

        self.step_times.append(step_time)
        self.losses.append(loss)

        if step % self.log_interval == 0:
            avg_loss = np.mean(self.losses[-self.log_interval :])

            print(
                f"Step {step} | Loss: {loss:.4f} | Avg Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | Time/Step: {time_per_step:.3f}s"
            )

    def end_epoch(self, epoch: int) -> Dict[str, float]:
        """結束一個epoch並返回統計"""
        if self.start_time is None:
            return {}

        epoch_time = time.time() - self.start_time
        avg_loss = np.mean(self.losses) if self.losses else 0
        steps_per_second = len(self.losses) / epoch_time if epoch_time > 0 else 0

        return {
            "epoch_time": epoch_time,
            "avg_loss": avg_loss,
            "steps_per_second": steps_per_second,
        }


# 主要匯出
__all__ = [
    "TrainingMonitor",
    "TrainingMetrics",
    "GPUMonitor",
    "MetricsHistory",
    "RichTrainingMonitor",
    "SimpleTrainingMonitor",
    "LegacyTrainingMonitor",
]
