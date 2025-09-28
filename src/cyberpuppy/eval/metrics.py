"""
評估指標計算與監控模組

提供離線評估（F1、AUCPR、會話級指標）與線上收斂監控功能
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
# 評估指標相關
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """單一指標結果"""

    name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "metadata": self.metadata,
        }


@dataclass
class SessionContext:
    """會話上下文"""

    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def add_message(self, message: Dict[str, Any]):
        """添加訊息到會話"""
        self.messages.append(message)
        self.end_time = datetime.now()

    def get_duration(self) -> float:
        """取得會話持續時間（秒）"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()


class MetricsCalculator:
    """評估指標計算器"""

    def __init__(self):
        """初始化計算器"""
        self.label_mapping = {
            "toxicity": ["none", "toxic", "severe"],
            "bullying": ["none", "harassment", "threat"],
            "emotion": ["pos", "neu", "neg"],
            "role": ["none", "perpetrator", "victim", "bystander"],
        }

    def calculate_classification_metrics(
        self,
        y_true: List[Union[int, str]],
        y_pred: List[Union[int, str]],
        task_name: str = "toxicity",
        average: str = "macro",
    ) -> Dict[str, MetricResult]:
        """
        計算分類指標

        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
            task_name: 任務名稱
            average: 平均方式 ('macro', 'micro', 'weighted')

        Returns:
            指標結果字典
        """
        metrics = {}

        # 基本指標
        metrics["accuracy"] = MetricResult(
            name=f"{task_name}_accuracy", value=accuracy_score(y_true, y_pred)
        )

        metrics["f1_score"] = MetricResult(
            name=f"{task_name}_f1_{average}",
            value=f1_score(y_true, y_pred, average=average),
            metadata={"average": average},
        )

        metrics["precision"] = MetricResult(
            name=f"{task_name}_precision_{average}",
            value=precision_score(y_true, y_pred, average=average, zero_division=0),
            metadata={"average": average},
        )

        metrics["recall"] = MetricResult(
            name=f"{task_name}_recall_{average}",
            value=recall_score(y_true, y_pred, average=average, zero_division=0),
            metadata={"average": average},
        )

        # 混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = MetricResult(
            name=f"{task_name}_confusion_matrix",
            value=0.0,  # 矩陣不是單一數值
            metadata={"matrix": cm.tolist()},
        )

        # 類別級指標
        if task_name in self.label_mapping:
            labels = self.label_mapping[task_name]
            report = classification_report(
                y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
            )

            for label in labels:
                if label in report:
                    metrics[f"f1_{label}"] = MetricResult(
                        name=f"{task_name}_f1_{label}",
                        value=report[label]["f1-score"],
                        metadata={"label": label},
                    )

        return metrics

    def calculate_probability_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray, task_name: str = "toxi" "city"
    ) -> Dict[str, MetricResult]:
        """
        計算機率相關指標

        Args:
            y_true: 真實標籤（one-hot 或整數）
            y_prob: 預測機率
            task_name: 任務名稱

        Returns:
            指標結果字典
        """
        metrics = {}

        # 確保是 numpy array
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        # AUC-ROC (多類別)
        try:
            if y_prob.shape[1] > 2:
                # 多類別 AUC
                auc_roc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            else:
                # 二元分類
                if len(y_true.shape) > 1:
                    auc_roc = roc_auc_score(y_true[:, 1], y_prob[:, 1])
                else:
                    auc_roc = roc_auc_score(y_true, y_prob[:, 1])

            metrics["auc_roc"] = MetricResult(
                name=f"{task_name}_auc_roc",
                value=auc_roc,
                metadata={"task": task_name, "metric_type": "auc_roc"},
            )
        except Exception as e:
            logger.warning(f"無法計算 AUC-ROC: {e}")

        # Average Precision (AUCPR)
        try:
            if y_prob.shape[1] > 2:
                # 多類別 AP
                aucpr_scores = []
                for i in range(y_prob.shape[1]):
                    if len(np.unique(y_true[:, i] if len(y_true.shape) > 1 else (y_true == i))) > 1:
                        ap = average_precision_score(
                            y_true[:, i] if len(y_true.shape) > 1 else (y_true == i),
                            y_prob[:, i],
                        )
                        aucpr_scores.append(ap)
                aucpr = np.mean(aucpr_scores) if aucpr_scores else 0.0
            else:
                # 二元分類
                aucpr = average_precision_score(
                    y_true[:, 1] if len(y_true.shape) > 1 else y_true, y_prob[:, 1]
                )

            metrics["aucpr"] = MetricResult(
                name=f"{task_name}_aucpr",
                value=aucpr,
                metadata={"task": task_name, "metric_type": "aucpr"},
            )
        except Exception as e:
            logger.warning(f"無法計算 AUCPR: {e}")

        return metrics

    def calculate_session_metrics(
        self, sessions: List[SessionContext], task_name: str = "toxicity"
    ) -> Dict[str, MetricResult]:
        """
        計算會話級指標

        Args:
            sessions: 會話清單
            task_name: 任務名稱

        Returns:
            會話級指標
        """
        metrics = {}

        if not sessions:
            return metrics

        # 會話統計
        total_sessions = len(sessions)
        total_messages = sum(len(s.messages) for s in sessions)
        avg_messages_per_session = total_messages / total_sessions if total_sessions > 0 else 0

        metrics["total_sessions"] = MetricResult(
            name="total_sessions",
            value=total_sessions,
            metadata={"task": task_name, "metric_type": "session_count"},
        )

        metrics["avg_messages_per_session"] = MetricResult(
            name="avg_messages_per_session", value=avg_messages_per_session
        )

        # 會話持續時間
        durations = [s.get_duration() for s in sessions]
        metrics["avg_session_duration"] = MetricResult(
            name="avg_session_du" "ration_seconds",
            value=np.mean(durations) if durations else 0.0,
        )

        # 毒性升級率（會話內毒性增加的比例）
        escalation_count = 0
        de_escalation_count = 0

        for session in sessions:
            if len(session.messages) < 2:
                continue

            # 檢查會話內的毒性趨勢
            toxicity_scores = []
            for msg in session.messages:
                if "scores" in msg and task_name in msg["scores"]:
                    score = msg["scores"][task_name]
                    if isinstance(score, dict):
                        # 取最高分數（假設是毒性）
                        toxicity_scores.append(max(score.values()))
                    else:
                        toxicity_scores.append(score)

            if len(toxicity_scores) >= 2:
                # 比較首尾
                if toxicity_scores[-1] > toxicity_scores[0] + 0.1:
                    escalation_count += 1
                elif toxicity_scores[-1] < toxicity_scores[0] - 0.1:
                    de_escalation_count += 1

        metrics["escalation_rate"] = MetricResult(
            name=f"{task_name}_escalation_rate",
            value=escalation_count / total_sessions if total_sessions > 0 else 0.0,
            metadata={"escalated_sessions": escalation_count},
        )

        metrics["de_escalation_rate"] = MetricResult(
            name=f"{task_name}_de_escalation_rate",
            value=de_escalation_count / total_sessions if total_sessions > 0 else 0.0,
            metadata={"de_escalated_sessions": de_escalation_count},
        )

        # 介入成功率（介入後毒性降低）
        intervention_success = 0
        intervention_total = 0

        for session in sessions:
            for i, msg in enumerate(session.messages[:-1]):
                if msg.get("intervention", False):
                    intervention_total += 1
                    # 檢查下一條訊息的毒性是否降低
                    current_score = msg.get("scores", {}).get(task_name, 0)
                    next_score = session.messages[i + 1].get("sco" "res", {}).get(task_name, 0)

                    if isinstance(current_score, dict):
                        current_score = max(current_score.values())
                    if isinstance(next_score, dict):
                        next_score = max(next_score.values())

                    if next_score < current_score - 0.1:
                        intervention_success += 1

        metrics["intervention_success_rate"] = MetricResult(
            name="intervention_success_rate",
            value=(intervention_success / intervention_total if intervention_total > 0 else 0.0),
            metadata={
                "successful_interventions": intervention_success,
                "total_interventions": intervention_total,
            },
        )

        return metrics


class OnlineMonitor:
    """線上收斂監控器"""

    def __init__(self, window_size: int = 100, checkpoint_interval: int = 1000):
        """
        初始化監控器

        Args:
            window_size: 移動窗口大小
            checkpoint_interval: 檢查點間隔
        """
        self.window_size = window_size
        self.checkpoint_interval = checkpoint_interval

        # 移動窗口
        self.loss_window = deque(maxlen=window_size)
        self.accuracy_window = deque(maxlen=window_size)
        self.f1_window = deque(maxlen=window_size)

        # 歷史記錄
        self.history = {
            "step": [],
            "loss": [],
            "accuracy": [],
            "f1_score": [],
            "learning_rate": [],
            "timestamp": [],
        }

        # 收斂檢測
        self.convergence_patience = 10
        self.convergence_threshold = 0.001
        self.no_improvement_count = 0
        self.best_loss = float("inf")

        # 統計
        self.total_steps = 0
        self.start_time = time.time()

    def update(
        self, loss: float, accuracy: float, f1_score: float, learning_rate: float = 0.0
    ) -> Dict[str, Any]:
        """
        更新監控指標

        Args:
            loss: 損失值
            accuracy: 準確率
            f1_score: F1 分數
            learning_rate: 學習率

        Returns:
            監控狀態
        """
        self.total_steps += 1

        # 更新窗口
        self.loss_window.append(loss)
        self.accuracy_window.append(accuracy)
        self.f1_window.append(f1_score)

        # 檢查收斂
        converged = self._check_convergence(loss)

        # 計算統計
        stats = {
            "step": self.total_steps,
            "loss": loss,
            "loss_avg": np.mean(self.loss_window),
            "loss_std": np.std(self.loss_window),
            "accuracy": accuracy,
            "accuracy_avg": np.mean(self.accuracy_window),
            "f1_score": f1_score,
            "f1_avg": np.mean(self.f1_window),
            "learning_rate": learning_rate,
            "converged": converged,
            "improvem" "ent_count": self.convergence_patience - self.no_improvement_count,
            "elapsed_time": time.time() - self.start_time,
        }

        # 儲存歷史（每個檢查點）
        if self.total_steps % self.checkpoint_interval == 0:
            self.history["step"].append(self.total_steps)
            self.history["loss"].append(loss)
            self.history["accuracy"].append(accuracy)
            self.history["f1_score"].append(f1_score)
            self.history["learning_rate"].append(learning_rate)
            self.history["timestamp"].append(datetime.now().isoformat())

        return stats

    def _check_convergence(self, loss: float) -> bool:
        """檢查是否收斂"""
        if loss < self.best_loss - self.convergence_threshold:
            self.best_loss = loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        return self.no_improvement_count >= self.convergence_patience

    def get_summary(self) -> Dict[str, Any]:
        """取得監控摘要"""
        if not self.loss_window:
            return {}

        return {
            "total_steps": self.total_steps,
            "best_loss": self.best_loss,
            "current_loss_avg": np.mean(self.loss_window),
            "current_accuracy_avg": np.mean(self.accuracy_window),
            "current_f1_avg": np.mean(self.f1_window),
            "conv" "erged": self.no_improvement_count >= self.convergence_patience,
            "training_time": time.time() - self.start_time,
            "steps_pe" "r_second": self.total_steps / (time.time() - self.start_time),
        }

    def export_history(self) -> Dict[str, List]:
        """匯出歷史記錄"""
        return self.history.copy()


class PrometheusExporter:
    """Prometheus 指標匯出器"""

    def __init__(self, job_name: str = "cyberpuppy", instance: str = "localhost:8000"):
        """
        初始化 Prometheus 匯出器

        Args:
            job_name: 工作名稱
            instance: 實例標識
        """
        self.job_name = job_name
        self.instance = instance
        self.metrics = {}

    def update_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """更新指標"""
        key = self._generate_key(name, labels)
        self.metrics[key] = {
            "name": name,
            "value": value,
            "labels": labels or {},
            "timestamp": time.time(),
        }

    def _generate_key(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> str:
        """生成唯一鍵值"""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def export(self) -> str:
        """
        匯出 Prometheus 格式

        Returns:
            Prometheus 格式的指標字串
        """
        lines = []

        # 添加說明
        lines.append("# HELP cyberpuppy_info CyberPuppy service information")
        lines.append("# TYPE cyberpuppy_info gauge")
        lines.append(f'cyberpuppy_info{{job="{self.job_name}",instance="{self.instance_name}"}} 1')

        # 匯出所有指標
        for _key, metric in self.metrics.items():
            name = metric["name"]
            value = metric["value"]
            labels = metric["labels"]

            # 構建標籤字串
            label_parts = [
                f'job="{self.job_name}"',
                f'instance="{self.instance_name}"',
            ]
            label_parts.extend(f'{k}="{v}"' for k, v in labels.items())
            label_str = ",".join(label_parts)

            # 添加指標
            lines.append(f"# HELP {name} {name}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name}{{{label_str}}} {value}")

        return "\n".join(lines)

    def push_to_gateway(self, gateway_url: str = "localhost:9091") -> bool:
        """
        推送到 Prometheus Pushgateway

        Args:
            gateway_url: Pushgateway URL

        Returns:
            是否成功
        """
        try:
            import requests

            url = (
                f"http://{gateway_url}/metrics/job/{self.job_name}"
                f"/instance/{self.instance_name}"
            )
            response = requests.post(url, data=self.export())
            return response.status_code == 200
        except Exception as e:
            logger.error(f"推送到 Pushgateway 失敗: {e}")
            return False


class CSVExporter:
    """CSV 格式匯出器"""

    def __init__(self, output_dir: str = "./metrics"):
        """
        初始化 CSV 匯出器

        Args:
            output_dir: 輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_metrics(
        self, metrics: Dict[str, MetricResult], filename: Optional[str] = None
    ) -> str:
        """
        匯出指標到 CSV

        Args:
            metrics: 指標字典
            filename: 檔案名稱

        Returns:
            檔案路徑
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.csv"

        filepath = self.output_dir / filename

        lines = ["metric_name,value,metadata"]
        for _name, result in metrics.items():
            metadata_str = json.dumps(result.metadata) if result.metadata else ""
            lines.append(f'{result.name},{result.value},"{metadata_str}"')

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"指標已匯出到: {filepath}")
        return str(filepath)

    def export_history(
        self,
        history: Dict[str, List],
        filename: Optional[str] = None,
    ) -> str:
        """
        匯出歷史記錄到 CSV

        Args:
            history: 歷史記錄
            filename: 檔案名稱

        Returns:
            檔案路徑
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"history_{timestamp}.csv"

        filepath = self.output_dir / filename

        # 轉換為 CSV 格式
        if history:
            import csv

            keys = list(history.keys())
            rows = []

            # 找出最大長度
            max_len = max(len(v) for v in history.values())

            for i in range(max_len):
                row = {}
                for key in keys:
                    if i < len(history[key]):
                        row[key] = history[key][i]
                    else:
                        row[key] = ""
                rows.append(row)

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)

        logger.info(f"歷史記錄已匯出到: {filepath}")
        return str(filepath)


class EvaluationReport:
    """評估報告生成器"""

    def __init__(self):
        """初始化報告生成器"""
        self.calculator = MetricsCalculator()
        self.sessions: List[SessionContext] = []
        self.results = {}

    def add_predictions(
        self,
        y_true: List,
        y_pred: List,
        y_prob: Optional[np.ndarray] = None,
        task_name: str = "toxicity",
    ):
        """添加預測結果"""
        # 計算分類指標
        classification_metrics = self.calculator.calculate_classification_metrics(
            y_true, y_pred, task_name
        )
        self.results[f"{task_name}_classification"] = classification_metrics

        # 計算機率指標
        if y_prob is not None:
            prob_metrics = self.calculator.calculate_probability_metrics(y_true, y_prob, task_name)
            self.results[f"{task_name}_probability"] = prob_metrics

    def add_session(self, session: SessionContext):
        """添加會話"""
        self.sessions.append(session)

    def generate_report(self) -> Dict[str, Any]:
        """生成完整報告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "session_metrics": {},
            "summary": {},
        }

        # 彙整所有指標
        for task_name, metrics in self.results.items():
            report["metrics"][task_name] = {
                name: result.to_dict() for name, result in metrics.items()
            }

        # 計算會話級指標
        if self.sessions:
            session_metrics = self.calculator.calculate_session_metrics(self.sessions)
            report["session_metrics"] = {
                name: result.to_dict() for name, result in session_metrics.items()
            }

        # 生成摘要
        report["summary"] = self._generate_summary()

        return report

    def _generate_summary(self) -> Dict[str, Any]:
        """生成摘要"""
        summary = {"total_evaluations": 0, "main_metrics": {}}

        # 提取主要指標
        for task_name, metrics in self.results.items():
            if "f1_score" in metrics:
                summary["main_metrics"][f"{task_name}_f1"] = metrics["f1_score"].value
            if "aucpr" in metrics:
                summary["main_metrics"][f"{task_name}_aucpr"] = metrics["aucpr"].value

        # 會話統計
        if self.sessions:
            summary["total_sessions"] = len(self.sessions)
            summary["total_messages"] = sum(len(s.messages) for s in self.sessions)

        return summary

    def save_report(self, filepath: str):
        """儲存報告到檔案"""
        report = self.generate_report()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"報告已儲存到: {filepath}")


# 使用範例
def example_usage():
    """使用範例"""
    import random

    # 1. 離線評估
    print("=== 離線評估範例 ===")

    # 生成模擬資料
    labels = ["none", "toxic", "severe"]
    y_true = [random.choice(labels) for _ in range(100)]
    y_pred = [random.choice(labels) for _ in range(100)]

    # 計算指標
    calculator = MetricsCalculator()
    metrics = calculator.calculate_classification_metrics(y_true, y_pred, "toxi" "city")

    for name, result in metrics.items():
        if name != "toxicity_confusion_matrix":
            print(f"{name}: {result.value:.3f}")

    # 2. 線上監控
    print("\n=== 線上監控範例 ===")

    monitor = OnlineMonitor(window_size=10)

    # 模擬訓練過程
    for step in range(50):
        loss = 1.0 / (step + 1) + random.random() * 0.1
        accuracy = min(0.95, 0.5 + step * 0.01)
        f1 = min(0.9, 0.4 + step * 0.01)

        stats = monitor.update(loss, accuracy, f1, learning_rate=0.001)

        if step % 10 == 0:
            print(
                f"Step {step}: Loss={stats['loss_avg']:.3f}, "
                f"Acc={stats['accuracy_avg']:.3f}, F1={stats['f1_avg']:.3f}"
            )

    # 3. 匯出指標
    print("\n=== 匯出範例 ===")

    # Prometheus 格式
    prometheus = PrometheusExporter()
    prometheus.update_metric("toxicity_f1_score", 0.78, {"model": "baseline", "dataset": "test"})
    prometheus.update_metric("accuracy", 0.85, {"dataset": "test"})
    print("Prometheus 格式:")
    print(prometheus.export()[:200] + "...")

    # CSV 格式
    csv_exporter = CSVExporter()
    csv_path = csv_exporter.export_metrics(metrics)
    print(f"CSV 已匯出到: {csv_path}")


if __name__ == "__main__":
    example_usage()
