"""
評估結果視覺化工具

提供各種圖表生成功能，用於評估報告的視覺化
"""

import warnings
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# 設定中文字體
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class MetricsVisualizer:
    """指標視覺化工具"""

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        初始化視覺化工具

        Args:
            style: matplotlib 風格
        """
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 10)

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str],
        title: str = "混淆矩陣",
        normalize: bool = True,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """
        繪製混淆矩陣

        Args:
            cm: 混淆矩陣
            labels: 類別標籤
            title: 圖表標題
            normalize: 是否正規化
            figsize: 圖表大小

        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            fmt = ".2f"
        else:
            fmt = "d"

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"label": "比例" if normalize else "數量"},
        )

        ax.set_xlabel("預測標籤")
        ax.set_ylabel("真實標籤")
        ax.set_title(title)

        plt.tight_layout()
        return fig

    def plot_pr_curves(
        self,
        pr_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
        title: str = "Precision-Recall 曲線",
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        繪製 PR 曲線

        Args:
            pr_data: {類別名稱: (precision, recall, average_precision)}
            title: 圖表標題
            figsize: 圖表大小

        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        for i, (label, (precision, recall, _ap)) in enumerate(pr_data.items()):
            ax.plot(
                recall,
                precision,
                color=self.colors[i],
                lw=2,
                label=f"{label} (A" "P={ap:.3f})",
            )
            ax.fill_between(recall, precision, alpha=0.2, color=self.colors[i])

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        return fig

    def plot_training_curves(
        self,
        history: Dict[str, List],
        metrics: List[str] = None,
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        繪製訓練曲線

        Args:
            history: 訓練歷史
            metrics: 要繪製的指標
            figsize: 圖表大小

        Returns:
            matplotlib figure
        """
        if metrics is None:
            metrics = ["loss", "accuracy", "f1_score"]
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        steps = history.get("step", range(len(history[metrics[0]])))

        for i, metric in enumerate(metrics):
            if metric not in history:
                continue

            ax = axes[i]
            values = history[metric]

            # 繪製原始值
            ax.plot(steps, values, alpha=0.3, label=f"{metric} (原始)")

            # 繪製移動平均（如果有）
            if f"{metric}_avg" in history:
                avg_values = history[f"{metric}_avg"]
                ax.plot(steps, avg_values, linewidth=2, label=f"{metric} (平均)")

            ax.set_xlabel("步驟")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f'{metric.replace("_", " ").title()} 趨勢')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 隱藏多餘的子圖
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("訓練監控", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_session_analysis(
        self, sessions_data: List[Dict[str, Any]], figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        繪製會話分析圖表

        Args:
            sessions_data: 會話資料清單
            figsize: 圖表大小

        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. 會話長度分佈
        session_lengths = [s["message_count"] for s in sessions_data]
        axes[0, 0].hist(session_lengths, bins=20, color="skyblue", edgecolor="black")
        axes[0, 0].set_xlabel("訊息數量")
        axes[0, 0].set_ylabel("會話數")
        axes[0, 0].set_title("會話長度分佈")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 毒性升級率
        escalation_rates = [s.get("escalation_rate", 0) for s in sessions_data]
        axes[0, 1].bar(
            ["升級", "穩定", "降級"],
            [
                sum(r > 0.1 for r in escalation_rates),
                sum(-0.1 <= r <= 0.1 for r in escalation_rates),
                sum(r < -0.1 for r in escalation_rates),
            ],
            color=["red", "yellow", "green"],
            alpha=0.7,
        )
        axes[0, 1].set_ylabel("會話數")
        axes[0, 1].set_title("會話毒性變化分佈")
        axes[0, 1].grid(True, axis="y", alpha=0.3)

        # 3. 介入成功率時間線
        intervention_times = []
        intervention_successes = []
        for s in sessions_data:
            if "interventions" in s:
                for intervention in s["interventions"]:
                    intervention_times.append(intervention["time"])
                    intervention_successes.append(intervention["successful"])

        if intervention_times:
            df = pd.DataFrame({"time": intervention_times})
            df["hour"] = pd.to_datetime(df["time"]).dt.hour
            success_by_hour = df.groupby("hour")["success"].mean()

            axes[1, 0].plot(success_by_hour.index, success_by_hour.values, marker="" "o")
            axes[1, 0].set_xlabel("小時")
            axes[1, 0].set_ylabel("成功率")
            axes[1, 0].set_title("介入成功率（按時段）")
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, "無介入資料", ha="center", va="center")

        # 4. 平均會話持續時間
        durations = [s.get("duration" "_seconds", 0) / 60 for s in sessions_data]  # 轉換為分鐘
        axes[1, 1].boxplot(durations)
        axes[1, 1].set_ylabel("持續時間（分鐘）")
        axes[1, 1].set_title("會話持續時間分佈")
        axes[1, 1].grid(True, axis="y", alpha=0.3)

        plt.suptitle("會話級分析", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_benchmark_comparison(
        self,
        benchmarks: Dict[str, Dict[str, float]],
        title: str = "效能基準比較",
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        繪製基準比較圖

        Args:
            benchmarks: {模型名稱: {指標: 數值}}
            title: 圖表標題
            figsize: 圖表大小

        Returns:
            matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        models = list(benchmarks.keys())
        metrics = list(next(iter(benchmarks.values())).keys())

        x = np.arange(len(metrics))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            values = [benchmarks[model].get(m, 0) for m in metrics]
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model, alpha=0.8)

            # 添加數值標籤
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_xlabel("指標")
        ax.set_ylabel("分數")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_metric_heatmap(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        title: str = "指標熱圖",
        figsize: Tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """
        繪製指標熱圖

        Args:
            metrics_dict: {任務: {指標: 數值}}
            title: 圖表標題
            figsize: 圖表大小

        Returns:
            matplotlib figure
        """
        # 轉換為 DataFrame
        df = pd.DataFrame(metrics_dict).T

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            df,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": "分數"},
            vmin=0,
            vmax=1,
        )

        ax.set_title(title)
        ax.set_xlabel("指標")
        ax.set_ylabel("任務")

        plt.tight_layout()
        return fig

    def save_all_figures(self, figures: Dict[str, plt.Figure], output_dir: str = "./figures"):
        """
        儲存所有圖表

        Args:
            figures: {檔名: figure}
            output_dir: 輸出目錄
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        for name, fig in figures.items():
            filepath = os.path.join(output_dir, f"{name}.png")
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"圖表已儲存: {filepath}")


# 使用範例
def example_visualizations():
    """視覺化範例"""

    visualizer = MetricsVisualizer()

    # 1. 混淆矩陣
    cm = np.array([[85, 10, 5], [15, 75, 10], [5, 15, 80]])
    fig1 = visualizer.plot_confusion_matrix(
        cm, labels=["None", "Toxic", "Severe"], title="毒性偵測混淆矩陣"
    )

    # 2. 訓練曲線
    history = {
        "step": list(range(100)),
        "loss": [1.0 / (i + 1) + np.random.random() * 0.1 for i in range(100)],
        "accuracy": [min(0.95, 0.5 + i * 0.005) for i in range(100)],
        "f1_score": [min(0.9, 0.4 + i * 0.005) for i in range(100)],
    }
    visualizer.plot_training_curves(history)

    # 3. 基準比較
    benchmarks = {
        "CyberPuppy": {
            "基準模型": {"F1": 0.72, "Precision": 0.75, "Recall": 0.70, "AUCPR": 0.68},
            "目標": {"F1": 0.80, "Precision": 0.85, "Recall": 0.78, "AUCPR": 0.75},
        }
    }
    visualizer.plot_benchmark_comparison(benchmarks)

    # 顯示所有圖表
    plt.show()

    # 儲存圖表
    figures = {"confusion_matrix": fig1}
    visualizer.save_all_figures(figures)


if __name__ == "__main__":
    example_visualizations()
