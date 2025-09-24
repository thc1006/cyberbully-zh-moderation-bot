"""
回歸測試套件
自動化檢測系統變更對既有功能的影響
包含：
- 基準預測結果比較
- API 回應格式穩定性
- 效能回歸檢測
- 模型輸出一致性驗證
"""

import asyncio
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib

import pytest
import numpy as np
from sklearn.metrics import accuracy_score

from tests.integration.fixtures.chinese_toxicity_examples import (
    get_balanced_test_set
)


@pytest.mark.regression
class TestRegressionBaseline:
    """回歸測試基準管理"""

    @pytest.fixture
    def baseline_dir(self, temp_dir):
        """基準資料目錄"""
        baseline_path = temp_dir / "baselines"
        baseline_path.mkdir(exist_ok=True)
        return baseline_path

    @pytest.fixture
    def regression_test_data(self):
        """標準回歸測試資料集"""
        return get_balanced_test_set(100)  # 100個平衡樣本

    def test_create_prediction_baseline(self, api_server, http_client,
                                       regression_test_data, baseline_dir):
        """建立預測基準"""
        baseline_file = baseline_dir / "prediction_baseline.json"

        if baseline_file.exists():
            pytest.skip("基準已存在，跳過建立")

        baseline_results = []

        async def collect_predictions():
            for example in regression_test_data:
                payload = {"text": example["text"]}

                try:
                    response = await http_client.post(
                        f"{api_server}/analyze",
                        json=payload,
                        timeout=30.0
                    )

                    if response.status_code == 200:
                        result = response.json()
                        baseline_results.append({
                            "text"
                                "_hash": hashlib.sha256(example[
                            "expected": example["expected"],
                            "prediction": {
                                "toxicity": result["toxicity"],
                                "bullying": result["bullying"],
                                "emotion": result["emotion"],
                                "role": result["role"]
                            },
                            "scores": result["scores"],
                            "timestamp": datetime.now().isoformat()
                        })
                except Exception as e:
                    print(f"Failed to get prediction for text: {e}")

        # 收集基準預測結果
        asyncio.run(collect_predictions())

        # 儲存基準
        with open(baseline_file, "w", encoding="utf-8") as f:
            json.dump({
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_examples": len(baseline_results),
                "results": baseline_results
            }, f, indent=2, ensure_ascii=False)

        print(f"Created prediction baseline with"
            " {len(baseline_results)} examples")
        assert len(baseline_results) > 50  # 至少要有足夠的基準資料

    def test_create_performance_baseline(self, api_server, http_client,
                                        regression_test_data, baseline_dir):
        """建立效能基準"""
        baseline_file = baseline_dir / "performance_baseline.json"

        if baseline_file.exists():
            pytest.skip("效能基準已存在")

        response_times = []

        async def measure_performance():
            for example in regression_test_data[:20]:  # 使用20個樣本測量
                payload = {"text": example["text"]}

                start_time = time.time()
                try:
                    response = await http_client.post(
                        f"{api_server}/analyze",
                        json=payload,
                        timeout=30.0
                    )
                    end_time = time.time()

                    if response.status_code == 200:
                        response_times.append(end_time - start_time)

                except Exception as e:
                    print(f"Performance measurement failed: {e}")

        asyncio.run(measure_performance())

        if response_times:
            performance_baseline = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "sample_size": len(response_times),
                "metrics": {
                    "mean_response_time": float(np.mean(response_times)),
                    "median_response_time": float(np.median(response_times)),
                    "p95_resp"
                        "onse_time": float(np.percentile(response_times, 95)),
                    "p99_resp"
                        "onse_time": float(np.percentile(response_times, 99)),
                    "max_response_time": float(np.max(response_times)),
                    "std_response_time": float(np.std(response_times))
                }
            }

            with open(baseline_file, "w", encoding="utf-8") as f:
                json.dump(performance_baseline, f, indent=2)

            print(f"Created performance baseline: {"
                "performance_baseline['metrics']}")


@pytest.mark.regression
class TestPredictionRegression:
    """預測結果回歸測試"""

    def load_baseline(
        self,
        baseline_dir: Path,
        baseline_name: str
    ) -> Optional[Dict]::
        """載入基準資料"""
        baseline_file = baseline_dir / baseline_name
        if not baseline_file.exists():
            return None

        with open(baseline_file, "r", encoding="utf-8") as f:
            return json.load(f)

    async def test_prediction_consistency(self, api_server, http_client,
                                        regression_test_data, baseline_dir):
        """測試預測一致性"""
        baseline = self.load_baseline(baseline_dir, "prediction_baseline.json")

        if baseline is None:
            pytest.skip("未找到預測基準，請先執行基準建立測試")

        current_results = []
        baseline_lookup = {r["text_hash"]: r for r in baseline["results"]}

        # 收集當前預測結果
        for example in regression_test_data:
            text_hash = hashlib.sha256(example["te"
                "xt"].encode()).hexdigest()[:16]

            if text_hash not in baseline_lookup:
                continue  # 跳過不在基準中的樣本

            payload = {"text": example["text"]}

            try:
                response = await http_client.post(
                    f"{api_server}/analyze",
                    json=payload,
                    timeout=30.0
                )

                if response.status_code == 200:
                    result = response.json()
                    current_results.append({
                        "text_hash": text_hash,
                        "prediction": {
                            "toxicity": result["toxicity"],
                            "bullying": result["bullying"],
                            "emotion": result["emotion"],
                            "role": result["role"]
                        }
                    })
            except Exception as e:
                print(f"Failed to get current prediction: {e}")

        # 比較預測結果
        consistency_metrics = self._compare_predictions(
            baseline["results"],
            current_results
        )

        print(f"Prediction consistency metrics: {consistency_metrics}")

        # 一致性要求
        assert consistency_metrics["toxicity_accuracy"] >= 0.9, \
            f"毒性預測一致性不足: {consistency_metrics['toxicity_accuracy']:.3f}"
        assert consistency_metrics["emotion_accuracy"] >= 0.85, \
            f"情緒預測一致性不足: {consistency_metrics['emotion_accuracy']:.3f}"
        assert consistency_metrics["overall_consistency"] >= 0.8, \
            f"整體預測一致性不足: {consistency_metrics['overall_consistency']:.3f}"

    def _compare_predictions(self, baseline_results: List[Dict],
                           current_results: List[Dict]) -> Dict[str, float]:
        """比較預測結果"""
        # 建立查找表
        current_lookup = {r["text_hash"]: r for r in current_results}

        comparisons = {
            "toxicity": {"baseline": [], "current": []},
            "bullying": {"baseline": [], "current": []},
            "emotion": {"baseline": [], "current": []},
            "role": {"baseline": [], "current": []}
        }

        for baseline_result in baseline_results:
            text_hash = baseline_result["text_hash"]

            if text_hash in current_lookup:
                current_result = current_lookup[text_hash]

                for task in comparisons.keys():
                    baseline_pred = baseline_result["prediction"][task]
                    current_pred = current_result["prediction"][task]

                    comparisons[task]["baseline"].append(baseline_pred)
                    comparisons[task]["current"].append(current_pred)

        # 計算一致性指標
        metrics = {}
        for task, data in comparisons.items():
            if data["baseline"]:
                accuracy = accuracy_score(data["baseline"], data["current"])
                metrics[f"{task}_accuracy"] = accuracy

        # 計算整體一致性
        if metrics:
            metrics["overall_consistency"] = np.mean(list(metrics.values()))
            metrics["compared"
                "_samples"] = len(comparisons[

        return metrics


@pytest.mark.regression
class TestPerformanceRegression:
    """效能回歸測試"""

    async def test_response_time_regression(self, api_server, http_client,
                                          regression_test_data, baseline_dir):
        """測試回應時間回歸"""
        baseline = self.load_baseline(baseline_dir, "performance_"
            "baseline.json")

        if baseline is None:
            pytest.skip("未找到效能基準")

        # 測量當前效能
        current_response_times = []

        for example in regression_test_data[:20]:
            payload = {"text": example["text"]}

            start_time = time.time()
            try:
                response = await http_client.post(
                    f"{api_server}/analyze",
                    json=payload,
                    timeout=30.0
                )
                end_time = time.time()

                if response.status_code == 200:
                    current_response_times.append(end_time - start_time)

            except Exception as e:
                print(f"Performance test failed: {e}")

        if not current_response_times:
            pytest.fail("無法收集當前效能資料")

        # 計算當前效能指標
        current_metrics = {
            "mean_response_time": float(np.mean(current_response_times)),
            "p95_resp"
                "onse_time": float(np.percentile(current_response_times, 95)),
            "max_response_time": float(np.max(current_response_times))
        }

        baseline_metrics = baseline["metrics"]

        print(f"Baseline metrics: {baseline_metrics}")
        print(f"Current metrics: {current_metrics}")

        # 效能回歸檢查
        mean_regression_ratio = current_metrics["mean_resp"
            "onse_time"] / baseline_metrics[
        p95_regression_ratio = current_metrics["p95_resp"
            "onse_time"] / baseline_metrics[

        print(f"Mean response time regression ra"
            "tio: {mean_regression_ratio:.3f}")
        print(f"P95 response time regression ra"
            "tio: {p95_regression_ratio:.3f}")

        # 允許最多 20% 的效能退化
        assert mean_regression_ratio <= 1.2, \
            f"平均回應時間退化過多: {mean_regression_ratio:.3f}x"
        assert p95_regression_ratio <= 1.3, \
            f"P95 回應時間退化過多: {p95_regression_ratio:.3f}x"

        # 絕對限制
        assert current_metrics["mean_response_time"] <= 2.0, \
            f"平均回應時間超過限制: {current_metrics['mean_response_time']:.3f}s"

    def load_baseline(
        self,
        baseline_dir: Path,
        baseline_name: str
    ) -> Optional[Dict]::
        """載入基準資料"""
        baseline_file = baseline_dir / baseline_name
        if not baseline_file.exists():
            return None

        with open(baseline_file, "r", encoding="utf-8") as f:
            return json.load(f)


@pytest.mark.regression
class TestAPIContractRegression:
    """API 契約回歸測試"""

    async def test_response_schema_stability(self, api_server, http_client):
        """測試回應結構穩定性"""
        test_payload = {"text": "API 契約測試"}

        response = await http_client.post(
            f"{api_server}/analyze",
            json=test_payload,
            timeout=30.0
        )

        assert response.status_code == 200
        data = response.json()

        # 驗證必要欄位存在
        required_fields = [
            "toxicity", "bullying", "role", "emotion", "emotion_strength",
            "sco"
                "res", 
        ]

        for field in required_fields:
            assert field in data, f"API 契約違規：缺少必要欄位 {field}"

        # 驗證欄位型別
        assert isinstance(data["toxicity"], str)
        assert data["toxicity"] in ["none", "toxic", "severe"]
        assert isinstance(data["emotion_strength"], int)
        assert 0 <= data["emotion_strength"] <= 4
        assert isinstance(data["scores"], dict)
        assert isinstance(data["processing_time_ms"], (int, float))

    async def test_health_endpoint_stability(self, api_server, http_client):
        """測試健康檢查端點穩定性"""
        response = await http_client.get(f"{api_server}/healthz")

        assert response.status_code == 200
        data = response.json()

        # 驗證健康檢查回應格式
        required_health_fields = ["sta"
            "tus", 

        for field in required_health_fields:
            assert field in data, f"健康檢查契約違規：缺少欄位 {field}"

        assert data["status"] in ["healthy", "degraded", "starting"]
        assert isinstance(data["uptime_seconds"], (int, float))


@pytest.mark.regression
class TestModelOutputRegression:
    """模型輸出回歸測試"""

    def test_prediction_distribution_stability(self, api_server, http_client,
                                             regression_test_data):
        """測試預測分佈穩定性"""
        predictions = {"toxicity": [], "emotion": [], "bullying": []}

        async def collect_predictions():
            for example in regression_test_data:
                payload = {"text": example["text"]}

                try:
                    response = await http_client.post(
                        f"{api_server}/analyze",
                        json=payload,
                        timeout=30.0
                    )

                    if response.status_code == 200:
                        result = response.json()
                        predictions["toxicity"].append(result["toxicity"])
                        predictions["emotion"].append(result["emotion"])
                        predictions["bullying"].append(result["bullying"])

                except Exception as e:
                    print(f"預測收集失敗: {e}")

        asyncio.run(collect_predictions())

        # 檢查預測分佈的合理性
        for task, preds in predictions.items():
            if not preds:
                continue

            pred_counts = {}
            for pred in preds:
                pred_counts[pred] = pred_counts.get(pred, 0) + 1

            print(f"{task} 預測分佈: {pred_counts}")

            # 基本合理性檢查
            assert len(pred_counts) > 1, f"{task} 預測缺乏多樣性"

            # 不應該有單一類別佔絕大多數（除非測試資料真的這樣分佈）
            max_ratio = max(pred_counts.values()) / len(preds)
            assert max_ratio < 0.95, f"{task} 預測過度集中於單一類別: {max_ratio:.3f}"


@pytest.mark.regression
class TestRegressionReporting:
    """回歸測試報告"""

    def test_generate_regression_report(self, baseline_dir, temp_dir):
        """生成回歸測試報告"""
        report_data = {
            "test_run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "regression_detected": False
            },
            "details": []
        }

        # 檢查是否有回歸問題
        # 這裡可以整合前面測試的結果

        report_file = temp_dir / "regression_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"回歸測試報告已生成: {report_file}")


# 輔助函數
def calculate_prediction_drift(
    baseline_scores: Dict,
    current_scores: Dict
) -> float::
    """計算預測漂移程度"""
    drift_scores = []

    for task in ["toxicity", "bullying", "emotion"]:
        if task in baseline_scores and task in current_scores:
            baseline_dist = list(baseline_scores[task].values())
            current_dist = list(current_scores[task].values())

            # 使用 KL 散度或簡單的歐氏距離
            if len(baseline_dist) == len(current_dist):
                drift = np.sqrt(np.sum((np.array(baseline_dist) -
                    np.array(current_dist)) ** 2))
                drift_scores.append(drift)

    return np.mean(drift_scores) if drift_scores else 0.0


def save_regression_artifacts(baseline_dir: Path, artifacts: Dict[str, Any]):
    """儲存回歸測試相關文件"""
    artifacts_file = baseline_dir / "regression_artifacts.pkl"

    with open(artifacts_file, "wb") as f:
        pickle.dump(artifacts, f)


def load_regression_artifacts(baseline_dir: Path) -> Dict[str, Any]:
    """載入回歸測試相關文件"""
    artifacts_file = baseline_dir / "regression_artifacts.pkl"

    if artifacts_file.exists():
        with open(artifacts_file, "rb") as f:
            return pickle.load(f)

    return {}
