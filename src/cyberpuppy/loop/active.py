"""
主動學習循環模組

用於識別高不確定性和爭議性樣本，生成標註任務
"""

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningConfig:
    """主動學習配置"""

    # 不確定性閾值
    uncertainty_threshold: float = 0.3  # 熵值閾值
    confidence_margin: float = 0.2  # 置信度差異閾值
    controversy_threshold: float = 0.15  # 爭議性閾值

    # 樣本選擇
    max_samples_per_batch: int = 100  # 每批最大樣本數
    min_samples_for_retrain: int = 500  # 觸發再訓練的最小樣本數
    diversity_weight: float = 0.3  # 多樣性權重

    # 策略權重
    strategy_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "uncertainty": 0.4,
            "margin": 0.2,
            "controversy": 0.2,
            "diversity": 0.2,
        }
    )

    # 輸出配置
    output_dir: str = "./data/active_learning"
    annotation_csv: str = "annotation_tasks.csv"
    completed_csv: str = "completed_annotations.csv"

    # 時間配置
    sample_window_hours: int = 24  # 樣本收集窗口
    retrain_interval_days: int = 7  # 再訓練間隔


class UncertaintySampler:
    """不確定性採樣器"""

    def __init__(self, config: ActiveLearningConfig):
        self.config = config

    def calculate_entropy(self, probs: np.ndarray) -> float:
        """
        計算熵值

        Args:
            probs: 機率分佈

        Returns:
            熵值
        """
        # 避免 log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs * np.log(probs))

    def calculate_margin(self, probs: np.ndarray) -> float:
        """
        計算置信度邊界（前兩個最高機率的差）

        Args:
            probs: 機率分佈

        Returns:
            邊界值
        """
        sorted_probs = np.sort(probs)[::-1]
        if len(sorted_probs) >= 2:
            return sorted_probs[0] - sorted_probs[1]
        return 1.0

    def calculate_least_confidence(self, probs: np.ndarray) -> float:
        """
        計算最小置信度

        Args:
            probs: 機率分佈

        Returns:
            不確定性分數（1 - 最大機率）
        """
        return 1.0 - np.max(probs)

    def select_uncertain_samples(
        self,
        predictions: List[Dict[str, Any]],
        k: int
    ) -> List[int]:
        """
        選擇最不確定的樣本

        Args:
            predictions: 預測結果列表
            k: 選擇樣本數

        Returns:
            選中樣本的索引
        """
        uncertainties = []

        for i, pred in enumerate(predictions):
            # 計算各種不確定性指標
            probs = np.array(pred.get("probabilities", []))
            if len(probs) == 0:
                continue

            entropy = self.calculate_entropy(probs)
            margin = self.calculate_margin(probs)
            least_conf = self.calculate_least_confidence(probs)

            # 綜合不確定性分數
            uncertainty = (
                self.config.strategy_weights["uncertainty"] * entropy
                + (1 - margin) * self.config.strategy_weights["margin"]
                + least_conf * 0.2
            )

            uncertainties.append((i, uncertainty))

        # 排序並選擇 top-k
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in uncertainties[:k]]


class ControversyDetector:
    """爭議性檢測器"""

    def __init__(self, config: ActiveLearningConfig):
        self.config = config
        self.model_predictions = defaultdict(list)

    def add_model_prediction(
        self,
        sample_id: str,
        model_name: str,
        prediction: Dict[str, Any]
    ):
        """
        添加模型預測

        Args:
            sample_id: 樣本ID
            model_name: 模型名稱
            prediction: 預測結果
        """
        self.model_predictions[sample_id].append({"model": model_name, "prediction": prediction}) 

    def calculate_controversy(self, sample_id: str) -> float:
        """
        計算爭議性分數（多個模型預測的不一致程度）

        Args:
            sample_id: 樣本ID

        Returns:
            爭議性分數
        """
        predictions = self.model_predictions.get(sample_id, [])

        if len(predictions) < 2:
            return 0.0

        # 提取所有預測標籤
        labels = []
        probs_list = []

        for pred in predictions:
            label = pred["prediction"].get("label")
            probs = pred["prediction"].get("probabilities", [])

            if label is not None:
                labels.append(label)
            if len(probs) > 0:
                probs_list.append(probs)

        # 計算標籤不一致性
        if len(labels) > 1:
            unique_labels = set(labels)
            label_controversy = 1.0 - (1.0 / len(unique_labels))
        else:
            label_controversy = 0.0

        # 計算機率分佈的 KL 散度
        kl_controversy = 0.0
        if len(probs_list) > 1:
            for i in range(len(probs_list)):
                for j in range(i + 1, len(probs_list)):
                    kl_div = self._kl_divergence(
                        np.array(probs_list[i]),
                        np.array(probs_list[j])
                    )
                    kl_controversy += kl_div

            kl_controversy /= len(probs_list) * (len(probs_list) - 1) / 2

        return 0.6 * label_controversy + 0.4 * kl_controversy

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """計算 KL 散度"""
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        return np.sum(p * np.log(p / q))

    def select_controversial_samples(self, k: int) -> List[str]:
        """
        選擇最具爭議性的樣本

        Args:
            k: 選擇樣本數

        Returns:
            選中樣本的ID列表
        """
        controversies = []

        for sample_id in self.model_predictions:
            controversy = self.calculate_controversy(sample_id)
            if controversy > self.config.controversy_threshold:
                controversies.append((sample_id, controversy))

        # 排序並選擇 top-k
        controversies.sort(key=lambda x: x[1], reverse=True)
        return [sample_id for sample_id, _ in controversies[:k]]


class DiversitySampler:
    """多樣性採樣器"""

    def __init__(self, config: ActiveLearningConfig):
        self.config = config

    def calculate_embeddings_diversity(
        self, embeddings: np.ndarray, selected_indices: List[int],
            candidate_index: int
    ) -> float:
        """
        計算候選樣本與已選樣本的多樣性

        Args:
            embeddings: 所有樣本的嵌入向量
            selected_indices: 已選樣本索引
            candidate_index: 候選樣本索引

        Returns:
            多樣性分數
        """
        if len(selected_indices) == 0:
            return 1.0

        candidate_emb = embeddings[candidate_index:candidate_index + 1]
        selected_embs = embeddings[selected_indices]

        # 計算與已選樣本的最小距離
        distances = pairwise_distances(candidate_emb, selected_embs, metric="cos"
            "ine")

        return np.min(distances)

    def select_diverse_samples(
        self, embeddings: np.ndarray, uncertainties: np.ndarray, k: int
    ) -> List[int]:
        """
        選擇多樣化的樣本（結合不確定性和多樣性）

        Args:
            embeddings: 樣本嵌入向量
            uncertainties: 不確定性分數
            k: 選擇樣本數

        Returns:
            選中樣本的索引
        """
        n_samples = len(embeddings)
        selected = []

        # 初始選擇最不確定的樣本
        first_idx = np.argmax(uncertainties)
        selected.append(first_idx)

        # 迭代選擇
        while len(selected) < k and len(selected) < n_samples:
            scores = []

            for i in range(n_samples):
                if i in selected:
                    continue

                # 計算多樣性
                diversity = self.calculate_embeddings_diversity(
                    embeddings,
                    selected,
                    i
                )

                # 結合不確定性和多樣性
                score = (
                    uncertainties[i] * (1 - self.config.diversity_weight)
                    + diversity * self.config.diversity_weight
                )

                scores.append((i, score))

            if not scores:
                break

            # 選擇得分最高的
            scores.sort(key=lambda x: x[1], reverse=True)
            selected.append(scores[0][0])

        return selected


class ActiveLearningLoop:
    """主動學習循環主類"""

    def __init__(self, config: Optional[ActiveLearningConfig] = None):
        self.config = config or ActiveLearningConfig()
        self.uncertainty_sampler = UncertaintySampler(self.config)
        self.controversy_detector = ControversyDetector(self.config)
        self.diversity_sampler = DiversitySampler(self.config)

        # 創建輸出目錄
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # 樣本池
        self.sample_pool = []
        self.selected_samples = []

    def add_prediction(
        self,
        text: str,
        prediction: Dict[str, Any],
        metadata: Optional[Dict] = None,
        embeddings: Optional[np.ndarray] = None,
        model_name: str = "default",
    ):
        """
        添加預測結果到樣本池

        Args:
            text: 原始文本
            prediction: 預測結果
            metadata: 元數據
            embeddings: 文本嵌入向量
            model_name: 模型名稱
        """
        # 生成樣本ID
        sample_id = hashlib.sha256(text.encode()).hexdigest()[:16]

        sample = {
            "id": sample_id,
            "text": text,
            "prediction": prediction,
            "metadata": metadata or {},
            "embeddings": embeddings,
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
        }

        self.sample_pool.append(sample)

        # 添加到爭議檢測器
        self.controversy_detector.add_model_prediction(
            sample_id,
            model_name,
            prediction
        )

    def select_samples_for_annotation(self) -> List[Dict]:
        """
        選擇需要標註的樣本

        Returns:
            選中的樣本列表
        """
        if len(self.sample_pool) == 0:
            return []

        # 準備數據
        predictions = [s["prediction"] for s in self.sample_pool]

        # 1. 不確定性採樣
        uncertain_indices = self.uncertainty_sampler.select_uncertain_samples(
            predictions, self.config.max_samples_per_batch // 3
        )

        # 2. 爭議性採樣
        controversial_ids = self.controversy_detector.select_controversial_samples(
            self.config.max_samples_per_batch // 3
        )

        # 3. 多樣性採樣（如果有嵌入向量）
        diverse_indices = []
        embeddings_list = [s.get("embeddings") for s in self.sample_pool]
        if any(e is not None for e in embeddings_list):
            # 構建嵌入矩陣
            valid_embeddings = []
            valid_indices = []
            uncertainties = []

            for i, (emb, pred) in enumerate(zip(embeddings_list, predictions)):
                if emb is not None:
                    valid_embeddings.append(emb)
                    valid_indices.append(i)

                    # 計算不確定性
                    probs = np.array(pred.get("probabilities", []))
                    if len(probs) > 0:
                        entropy = self.uncertainty_sampler.calculate_entropy(probs)
                        uncertainties.append(entropy)
                    else:
                        uncertainties.append(0.0)

            if len(valid_embeddings) > 0:
                embeddings_matrix = np.vstack(valid_embeddings)
                uncertainties_array = np.array(uncertainties)

                diverse_local_indices = self.diversity_sampler.select_diverse_samples(
                    embeddings_matrix, uncertainties_array,
                        self.config.max_samples_per_batch // 3
                )

                diverse_indices = [valid_indices[i] for i in
                    diverse_local_indices]

        # 合併選中的樣本
        selected_indices = set(uncertain_indices + diverse_indices)

        # 根據 ID 添加爭議樣本
        id_to_index = {s["id"]: i for i, s in enumerate(self.sample_pool)}
        for controversial_id in controversial_ids:
            if controversial_id in id_to_index:
                selected_indices.add(id_to_index[controversial_id])

        # 限制總數
        selected_indices = list(selected_indices)[:
            self.config.max_samples_per_batch]

        # 獲取選中的樣本
        selected_samples = [self.sample_pool[i] for i in selected_indices]

        # 計算選擇原因
        for sample in selected_samples:
            reasons = []

            # 檢查不確定性
            if sample.get("prediction"):
                probs = np.array(sample["prediction"].get("probabilities", []))
                if len(probs) > 0:
                    entropy = self.uncertainty_sampler.calculate_entropy(probs)
                    if entropy > self.config.uncertainty_threshold:
                        reasons.append(f"high_uncertainty({entropy:.3f})")

            # 檢查爭議性
            controversy = self.controversy_detector.calculate_controversy(sample["i"
                "d"])
            if controversy > self.config.controversy_threshold:
                reasons.append(f"controversial({controversy:.3f})")

            sample["selection_reasons"] = reasons

        self.selected_samples = selected_samples
        return selected_samples

    def export_annotation_tasks(
        self,
        output_path: Optional[str] = None
    ) -> str:
        """
        輸出標註任務到 CSV

        Args:
            output_path: 輸出路徑

        Returns:
            輸出檔案路徑
        """
        if not self.selected_samples:
            self.select_samples_for_annotation()

        if not self.selected_samples:
            logger.warning("No samples selected for annotation")
            return ""

        # 設定輸出路徑
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.config.output_dir) / f"annotation_tasks_{timestamp}.csv"

        # 準備 CSV 數據
        csv_data = []
        for sample in self.selected_samples:
            prediction = sample.get("prediction", {})

            row = {
                "sample_id": sample["id"],
                "text": sample["text"],
                "predicted_toxicity": prediction.get("toxicity", "none"),
                "toxicity_confidence": prediction.get("toxicity_confidence", 0.0),
                "predicted_emotion": prediction.get("emotion", "neu"),
                "emotion_confidence": prediction.get("emotion_confidence", 0.0),
                "confidence": prediction.get("confidence", 0.0),
                "selection_reasons": prediction.get("selection_reasons", []),
                "timestamp": sample["timestamp"],
                "true_toxicity": "",  # 待標註
                "true_emotion": "",  # 待標註
                "annotator": "",  # 待填寫
                "annotation_notes": "",  # 標註備註
            }

            # 添加元數據
            for key, value in sample.get("metadata", {}).items():
                if key not in ["text", "embeddings"]:
                    row[f"meta_{key}"] = value

            csv_data.append(row)

        # 寫入 CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        logger.info(f"Exported {len(csv_data)} anno"
            "tation tasks to {output_path}")
        return str(output_path)

    def load_completed_annotations(self, csv_path: str) -> List[Dict]:
        """
        載入已完成的標註

        Args:
            csv_path: 標註完成的 CSV 路徑

        Returns:
            標註數據列表
        """
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

        annotations = []
        for _, row in df.iterrows():
            # 只選擇已標註的樣本
            if pd.notna(row.get("true_toxicity")) and pd.notna(row.get("true_emotion")):
                annotation = {
                    "sample_id": row["sample_id"],
                    "text": row["text"],
                    "true_labels": {
                        "toxicity": row["true_toxicity"],
                        "emotion": row["true_emotion"],
                    },
                    "predicted_labels": {
                        "toxicity": row.get("predicted_toxicity", ""),
                        "emotion": row.get("predicted_emotion", ""),
                    },
                    "annotator": row.get("annotator", "unknown"),
                    "annotation_notes": row.get("annotation_notes", ""),
                    "timestamp": row.get("timestamp", ""),
                }

                # 添加元數據
                meta = {}
                for col in row.index:
                    if col.startswith("meta_"):
                        meta[col[5:]] = row[col]
                annotation["metadata"] = meta

                annotations.append(annotation)

        logger.info(f"Loaded {len(annotations)} completed annotations")
        return annotations

    def prepare_retraining_data(
        self, annotations: List[Dict], output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        準備再訓練數據

        Args:
            annotations: 標註數據
            output_dir: 輸出目錄

        Returns:
            輸出檔案路徑字典
        """
        if output_dir is None:
            output_dir = Path(self.config.output_dir) / "retrain_data"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 按任務分組數據
        toxicity_data = []
        emotion_data = []

        for ann in annotations:
            # 毒性偵測數據
            toxicity_data.append(
                {
                    "text": ann["text"],
                    "label": ann["true_labels"]["toxicity"],
                    "sample_id": ann["sample_id"],
                }
            )

            # 情緒分類數據
            emotion_data.append(
                {
                    "text": ann["text"],
                    "label": ann["true_labels"]["emotion"],
                    "sample_id": ann["sample_id"],
                }
            )

        # 輸出檔案路徑
        output_files = {}

        # 儲存毒性偵測數據
        toxicity_path = output_dir / "toxicity_retrain.jsonl"
        with open(toxicity_path, "w", encoding="utf-8") as f:
            for item in toxicity_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        output_files["toxicity"] = str(toxicity_path)

        # 儲存情緒分類數據
        emotion_path = output_dir / "emotion_retrain.jsonl"
        with open(emotion_path, "w", encoding="utf-8") as f:
            for item in emotion_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        output_files["emotion"] = str(emotion_path)

        # 生成統計報告
        stats = {
            "total_samples": len(annotations),
            "toxicity_distribution": pd.DataFrame(toxicity_data),
            "emotion_distribution": pd.DataFrame(emotion_data),
            "timestamp": datetime.now().isoformat(),
        }

        stats_path = output_dir / "retrain_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        output_files["stats"] = str(stats_path)

        logger.info(f"Prepared retraining data: {stats}")
        return output_files

    def should_trigger_retraining(self) -> bool:
        """
        檢查是否應該觸發再訓練

        Returns:
            是否應該再訓練
        """
        # 檢查已完成標註的數量
        completed_pattern = Path(self.config.output_dir).glob("complet"
            "ed_*.csv")
        total_annotations = 0

        for csv_file in completed_pattern:
            df = pd.read_csv(csv_file)
            # 計算已標註的樣本
            annotated = df[df["true_toxicity"].notna() & df["true_emotion"].notna()]
            total_annotations += len(annotated)

        # 檢查是否達到閾值
        if total_annotations >= self.config.min_samples_for_retrain:
            logger.info(f"Retraining triggered: {total_annotations} annotations available")
            return True

        logger.info(
            f"Not enough annotations for retraining: "
            f"{total_annotations}/{self.config.min_samples_for_retrain}"
        )
        return False


# 使用範例
def example_usage():
    """示範如何使用主動學習循環"""

    # 初始化
    config = ActiveLearningConfig(
        uncertainty_threshold=0.3, max_samples_per_batch=50,
            min_samples_for_retrain=100
    )

    active_loop = ActiveLearningLoop(config)

    # 模擬添加預測結果
    sample_texts = [
        "這個人真的很討厭",
        "今天天氣很好",
        "你是不是有問題啊",
        "謝謝你的幫助",
        "去死吧你這個廢物",
    ]

    for text in sample_texts:
        # 模擬預測結果
        prediction = {
            "toxicity": {
                "label": "toxic" if "討厭" in text or "廢物" in text else "none",
                "confidence": np.random.uniform(0.3, 0.9),
                "probabilities": np.random.dirichlet([1, 1, 1]).tolist(),
            },
            "emotion": {
                "label": np.random.choice(["positive", "negative", "neutral"]),
                "confidence": np.random.uniform(0.4, 0.95),
                "probabilities": np.random.dirichlet([1, 1, 1]).tolist(),
            },
        }

        # 模擬嵌入向量
        embeddings = np.random.randn(768)

        active_loop.add_prediction(
            text=text,
            prediction=prediction,
            embeddings=embeddings,
            metadata={"source": "simulation"},
            model_name="model_v1",
        )

    # 選擇樣本並輸出
    selected = active_loop.select_samples_for_annotation()
    print(f"Selected {len(selected)} samples for annotation")

    # 輸出到 CSV
    csv_path = active_loop.export_annotation_tasks()
    print(f"Exported to {csv_path}")

    # 檢查是否需要再訓練
    if active_loop.should_trigger_retraining():
        print("Ready for retraining!")


if __name__ == "__main__":
    example_usage()
