#!/usr/bin/env python3
"""
SHAP 可解釋性AI實現
使用SHAP庫為中文網路霸凌偵測模型提供可解釋性分析

主要功能:
- Force plots (局部解釋)
- Waterfall plots (層次化解釋)
- Text plots (文本級解釋)
- Summary plots (全局解釋)
- Dependence plots (特徵依賴分析)
- 誤判分析
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Conditional SHAP import to avoid numba/coverage conflicts
try:
    import shap

    SHAP_AVAILABLE = True
except (ImportError, AttributeError):
    SHAP_AVAILABLE = False
    shap = None

from ..models.improved_detector import ImprovedDetector
from .ig import ExplanationResult

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


@dataclass
class SHAPResult:
    """SHAP解釋結果資料結構"""

    text: str
    tokens: List[str]

    # 預測結果
    toxicity_pred: int
    toxicity_prob: float
    emotion_pred: int
    emotion_prob: float
    bullying_pred: int
    bullying_prob: float
    role_pred: int
    role_prob: float

    # SHAP值
    toxicity_shap_values: np.ndarray
    emotion_shap_values: np.ndarray
    bullying_shap_values: np.ndarray
    role_shap_values: np.ndarray

    # 基線值
    toxicity_base_value: float
    emotion_base_value: float
    bullying_base_value: float
    role_base_value: float

    # 額外資訊
    prediction_confidence: Dict[str, float]
    feature_importance: Dict[str, float]


class SHAPModelWrapper:
    """SHAP模型包裝器"""

    def __init__(self, model: ImprovedDetector, tokenizer: AutoTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def __call__(self, texts: List[str]) -> np.ndarray:
        """
        SHAP需要的預測函數介面

        Args:
            texts: 文本列表

        Returns:
            預測概率數組 [batch_size, num_classes * num_tasks]
        """
        if isinstance(texts, str):
            texts = [texts]

        all_probs = []

        with torch.no_grad():
            for text in texts:
                # 分詞
                inputs = self.tokenizer(
                    text,
                    max_length=self.model.config.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                # 模型預測
                outputs = self.model(input_ids, attention_mask)

                # 轉換為概率
                toxicity_probs = F.softmax(outputs["toxicity"], dim=-1)
                bullying_probs = F.softmax(outputs["bullying"], dim=-1)
                role_probs = F.softmax(outputs["role"], dim=-1)
                emotion_probs = F.softmax(outputs["emotion"], dim=-1)

                # 合併所有任務的概率
                combined_probs = torch.cat(
                    [toxicity_probs, bullying_probs, role_probs, emotion_probs], dim=-1
                )

                all_probs.append(combined_probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)


class SHAPExplainer:
    """
    SHAP解釋器

    支援多種SHAP可視化方法和模型解釋
    """

    def __init__(self, model: ImprovedDetector, device: Optional[torch.device] = None):
        """
        初始化SHAP解釋器

        Args:
            model: 已訓練的模型
            device: 計算設備
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.tokenizer = model.tokenizer

        # 創建模型包裝器
        self.model_wrapper = SHAPModelWrapper(model, self.tokenizer, self.device)

        # 初始化SHAP解釋器
        self.explainer = None
        self._setup_explainer()

        # 任務配置
        self.task_configs = {
            "toxicity": {"start_idx": 0, "end_idx": 3, "labels": ["非毒性", "毒性", "嚴重毒性"]},
            "bullying": {"start_idx": 3, "end_idx": 6, "labels": ["非霸凌", "騷擾", "威脅"]},
            "role": {
                "start_idx": 6,
                "end_idx": 10,
                "labels": ["無角色", "施暴者", "受害者", "旁觀者"],
            },
            "emotion": {"start_idx": 10, "end_idx": 13, "labels": ["正面", "中性", "負面"]},
        }

        logger.info(f"SHAP Explainer initialized on device: {self.device}")

    def _setup_explainer(self):
        """設定SHAP解釋器"""
        # 使用Transformer專用的解釋器
        try:
            # 嘗試使用Transformer解釋器
            self.explainer = shap.Explainer(self.model_wrapper)
            logger.info("Using SHAP Transformer Explainer")
        except Exception as e:
            logger.warning(f"Failed to use Transformer explainer: {e}")
            # 後備：使用Permutation解釋器
            try:
                self.explainer = shap.PermutationExplainer(self.model_wrapper)
                logger.info("Using SHAP Permutation Explainer")
            except Exception as e:
                logger.error(f"Failed to initialize SHAP explainer: {e}")
                raise

    def explain_text(self, text: str, max_evals: int = 2000) -> SHAPResult:
        """
        解釋單個文本的模型決策

        Args:
            text: 輸入文本
            max_evals: 最大評估次數

        Returns:
            SHAP解釋結果
        """
        logger.info(f"Explaining text: {text[:50]}...")

        # 分詞
        inputs = self.tokenizer(
            text,
            max_length=self.model.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # 獲取預測結果
        with torch.no_grad():
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            outputs = self.model(input_ids, attention_mask)

            # 各任務預測
            toxicity_logits = outputs["toxicity"]
            bullying_logits = outputs["bullying"]
            role_logits = outputs["role"]
            emotion_logits = outputs["emotion"]

            toxicity_probs = F.softmax(toxicity_logits, dim=-1)
            bullying_probs = F.softmax(bullying_logits, dim=-1)
            role_probs = F.softmax(role_logits, dim=-1)
            emotion_probs = F.softmax(emotion_logits, dim=-1)

            # 預測類別
            toxicity_pred = torch.argmax(toxicity_logits, dim=-1).item()
            bullying_pred = torch.argmax(bullying_logits, dim=-1).item()
            role_pred = torch.argmax(role_logits, dim=-1).item()
            emotion_pred = torch.argmax(emotion_logits, dim=-1).item()

        # 計算SHAP值
        try:
            shap_values = self.explainer([text], max_evals=max_evals)

            # 提取各任務的SHAP值
            if hasattr(shap_values, "values"):
                values = shap_values.values[0]  # 第一個（也是唯一的）樣本
                base_values = (
                    shap_values.base_values[0] if hasattr(shap_values, "base_values") else 0
                )
            else:
                values = shap_values[0]
                base_values = 0

            # 分割各任務的SHAP值
            toxicity_shap = values[
                self.task_configs["toxicity"]["start_idx"] : self.task_configs["toxicity"][
                    "end_idx"
                ]
            ]
            bullying_shap = values[
                self.task_configs["bullying"]["start_idx"] : self.task_configs["bullying"][
                    "end_idx"
                ]
            ]
            role_shap = values[
                self.task_configs["role"]["start_idx"] : self.task_configs["role"]["end_idx"]
            ]
            emotion_shap = values[
                self.task_configs["emotion"]["start_idx"] : self.task_configs["emotion"]["end_idx"]
            ]

            # 基線值
            if isinstance(base_values, (list, np.ndarray)):
                toxicity_base = (
                    base_values[toxicity_pred]
                    if len(base_values) > toxicity_pred
                    else base_values[0]
                )
                bullying_base = (
                    base_values[bullying_pred + 3]
                    if len(base_values) > bullying_pred + 3
                    else base_values[0]
                )
                role_base = (
                    base_values[role_pred + 6]
                    if len(base_values) > role_pred + 6
                    else base_values[0]
                )
                emotion_base = (
                    base_values[emotion_pred + 10]
                    if len(base_values) > emotion_pred + 10
                    else base_values[0]
                )
            else:
                toxicity_base = bullying_base = role_base = emotion_base = base_values

        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            # 使用零值作為後備
            seq_len = len(tokens)
            toxicity_shap = np.zeros(seq_len)
            bullying_shap = np.zeros(seq_len)
            role_shap = np.zeros(seq_len)
            emotion_shap = np.zeros(seq_len)
            toxicity_base = bullying_base = role_base = emotion_base = 0.0

        # 計算特徵重要性
        feature_importance = {
            "toxicity": np.abs(toxicity_shap).sum(),
            "bullying": np.abs(bullying_shap).sum(),
            "role": np.abs(role_shap).sum(),
            "emotion": np.abs(emotion_shap).sum(),
        }

        # 構建結果
        result = SHAPResult(
            text=text,
            tokens=tokens,
            toxicity_pred=toxicity_pred,
            toxicity_prob=toxicity_probs[0, toxicity_pred].item(),
            bullying_pred=bullying_pred,
            bullying_prob=bullying_probs[0, bullying_pred].item(),
            role_pred=role_pred,
            role_prob=role_probs[0, role_pred].item(),
            emotion_pred=emotion_pred,
            emotion_prob=emotion_probs[0, emotion_pred].item(),
            toxicity_shap_values=toxicity_shap,
            bullying_shap_values=bullying_shap,
            role_shap_values=role_shap,
            emotion_shap_values=emotion_shap,
            toxicity_base_value=toxicity_base,
            bullying_base_value=bullying_base,
            role_base_value=role_base,
            emotion_base_value=emotion_base,
            prediction_confidence={
                "toxicity": toxicity_probs[0, toxicity_pred].item(),
                "bullying": bullying_probs[0, bullying_pred].item(),
                "role": role_probs[0, role_pred].item(),
                "emotion": emotion_probs[0, emotion_pred].item(),
            },
            feature_importance=feature_importance,
        )

        logger.info("SHAP explanation completed")
        return result


class SHAPVisualizer:
    """SHAP可視化器"""

    def __init__(self, explainer: SHAPExplainer):
        self.explainer = explainer

        # 設定中文字體
        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    def create_force_plot(
        self, result: SHAPResult, task: str = "toxicity", save_path: Optional[str] = None
    ) -> None:
        """
        創建SHAP force plot

        Args:
            result: SHAP解釋結果
            task: 任務類型
            save_path: 保存路徑
        """
        # 獲取任務相關數據
        task_config = self.explainer.task_configs[task]
        shap_values = getattr(result, f"{task}_shap_values")
        base_value = getattr(result, f"{task}_base_value")
        pred_class = getattr(result, f"{task}_pred")

        # 過濾有效token
        valid_tokens = []
        valid_shap_values = []

        for i, token in enumerate(result.tokens):
            if token not in ["[CLS]", "[SEP]", "[PAD]"] and i < len(shap_values):
                clean_token = token.replace("##", "")
                valid_tokens.append(clean_token)
                valid_shap_values.append(shap_values[i])

        # 創建force plot
        try:
            shap.force_plot(
                base_value=base_value,
                shap_values=np.array(valid_shap_values),
                features=valid_tokens,
                out_names=task_config["labels"][pred_class],
                show=False,
                matplotlib=True,
            )

            plt.title(
                f"SHAP Force Plot - {task.capitalize()}\n"
                f"Prediction: {task_config['labels'][pred_class]}\n"
                f"Text: {result.text[:60]}..."
            )

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Force plot saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to create force plot: {e}")
            # 創建手動force plot
            self._manual_force_plot(
                valid_tokens,
                valid_shap_values,
                base_value,
                task_config["labels"][pred_class],
                result.text,
                save_path,
            )

    def _manual_force_plot(
        self,
        tokens: List[str],
        shap_values: List[float],
        base_value: float,
        prediction: str,
        text: str,
        save_path: Optional[str] = None,
    ):
        """手動創建force plot風格的可視化"""
        fig, ax = plt.subplots(figsize=(15, 8))

        # 計算位置
        x_pos = 0

        # 繪製基線
        ax.axhline(
            y=base_value,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Base value: {base_value:.3f}",
        )

        # 繪製SHAP值
        for _i, (token, shap_val) in enumerate(zip(tokens, shap_values)):
            color = "red" if shap_val > 0 else "blue"
            alpha = min(abs(shap_val) * 2, 1.0)

            # 繪製箭頭
            ax.arrow(
                x_pos,
                base_value,
                0,
                shap_val,
                head_width=0.5,
                head_length=abs(shap_val) * 0.1,
                fc=color,
                ec=color,
                alpha=alpha,
            )

            # 標註token
            ax.text(
                x_pos,
                base_value + shap_val + (0.1 if shap_val > 0 else -0.1),
                f"{token}\n{shap_val:.3f}",
                ha="center",
                va="bottom" if shap_val > 0 else "top",
                fontsize=8,
                rotation=45,
            )

            x_pos += 1

        ax.set_xlabel("Tokens")
        ax.set_ylabel("SHAP Value")
        ax.set_title(f"SHAP Force Plot\nPrediction: {prediction}\nText: {text[:60]}...")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Manual force plot saved to {save_path}")

    def create_waterfall_plot(
        self, result: SHAPResult, task: str = "toxicity", save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        創建SHAP waterfall plot

        Args:
            result: SHAP解釋結果
            task: 任務類型
            save_path: 保存路徑
        """
        task_config = self.explainer.task_configs[task]
        shap_values = getattr(result, f"{task}_shap_values")
        base_value = getattr(result, f"{task}_base_value")
        pred_class = getattr(result, f"{task}_pred")

        # 過濾和排序特徵
        valid_features = []
        valid_shap_values = []

        for i, token in enumerate(result.tokens):
            if token not in ["[CLS]", "[SEP]", "[PAD]"] and i < len(shap_values):
                clean_token = token.replace("##", "")
                valid_features.append(clean_token)
                valid_shap_values.append(shap_values[i])

        # 按絕對值排序，取Top 15
        sorted_indices = np.argsort(np.abs(valid_shap_values))[::-1][:15]
        top_features = [valid_features[i] for i in sorted_indices]
        top_shap_values = [valid_shap_values[i] for i in sorted_indices]

        # 創建waterfall plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # 計算累積值
        cumulative = base_value
        prev_cumulative = base_value
        positions = []
        values = []
        colors = []

        for i, (_feature, shap_val) in enumerate(zip(top_features, top_shap_values)):
            positions.append(i)
            values.append(abs(shap_val))
            colors.append("red" if shap_val > 0 else "blue")

            # 繪製條形
            ax.bar(
                i,
                abs(shap_val),
                bottom=cumulative if shap_val > 0 else cumulative - abs(shap_val),
                color=colors[-1],
                alpha=0.7,
                width=0.6,
            )

            # 添加連接線
            if i > 0:
                ax.plot([i - 0.3, i - 0.3], [prev_cumulative, cumulative], "k--", alpha=0.3)

            # 標註數值
            ax.text(
                i,
                cumulative + shap_val / 2,
                f"{shap_val:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                rotation=90,
            )

            prev_cumulative = cumulative
            cumulative += shap_val

        # 設定標籤
        ax.set_xticks(range(len(top_features)))
        ax.set_xticklabels(top_features, rotation=45, ha="right")
        ax.set_ylabel("SHAP Value")
        ax.set_title(
            f"SHAP Waterfall Plot - {task.capitalize()}\n"
            f'Prediction: {task_config["labels"][pred_class]}\n'
            f"Base: {base_value:.3f} → Final: {cumulative:.3f}"
        )

        # 添加基線和最終預測線
        ax.axhline(
            y=base_value, color="gray", linestyle="--", alpha=0.5, label=f"Base: {base_value:.3f}"
        )
        ax.axhline(
            y=cumulative, color="green", linestyle="-", alpha=0.7, label=f"Final: {cumulative:.3f}"
        )

        ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Waterfall plot saved to {save_path}")

        return fig

    def create_text_plot(
        self, result: SHAPResult, task: str = "toxicity", save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        創建SHAP text plot

        Args:
            result: SHAP解釋結果
            task: 任務類型
            save_path: 保存路徑
        """
        task_config = self.explainer.task_configs[task]
        shap_values = getattr(result, f"{task}_shap_values")
        pred_class = getattr(result, f"{task}_pred")

        # 準備文本和SHAP值
        text_with_scores = []
        colors = []

        for i, token in enumerate(result.tokens):
            if token not in ["[CLS]", "[SEP]", "[PAD]"] and i < len(shap_values):
                clean_token = token.replace("##", "")
                shap_val = shap_values[i]

                text_with_scores.append((clean_token, shap_val))

                # 設定顏色強度
                if shap_val > 0:
                    colors.append(plt.cm.Reds(min(abs(shap_val) * 2, 1.0)))
                else:
                    colors.append(plt.cm.Blues(min(abs(shap_val) * 2, 1.0)))

        # 創建文本可視化
        fig, ax = plt.subplots(figsize=(16, 6))

        x_pos = 0
        for (token, shap_val), color in zip(text_with_scores, colors):
            # 繪製文本背景
            bbox = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7)
            ax.text(x_pos, 0.5, token, fontsize=12, bbox=bbox, ha="left", va="center")

            # 添加SHAP值標註
            ax.text(x_pos, 0.2, f"{shap_val:.3f}", fontsize=8, ha="left", va="center")

            x_pos += len(token) + 1

        # 設定圖表
        ax.set_xlim(-1, x_pos)
        ax.set_ylim(0, 1)
        ax.set_title(
            f"SHAP Text Plot - {task.capitalize()}\n"
            f'Prediction: {task_config["labels"][pred_class]}\n'
            f"Red: Positive Contribution, Blue: Negative Contribution"
        )
        ax.axis("off")

        # 添加顏色說明
        ax.text(
            0.02,
            0.98,
            "Color intensity represents SHAP value magnitude",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Text plot saved to {save_path}")

        return fig

    def create_summary_plot(
        self,
        results: List[SHAPResult],
        task: str = "toxicity",
        max_features: int = 20,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        創建SHAP summary plot

        Args:
            results: SHAP解釋結果列表
            task: 任務類型
            max_features: 最大特徵數
            save_path: 保存路徑
        """
        if not results:
            raise ValueError("No results provided for summary plot")

        # 收集所有SHAP值
        all_shap_values = []
        all_features = []

        for result in results:
            shap_values = getattr(result, f"{task}_shap_values")

            for i, token in enumerate(result.tokens):
                if token not in ["[CLS]", "[SEP]", "[PAD]"] and i < len(shap_values):
                    clean_token = token.replace("##", "")
                    all_features.append(clean_token)
                    all_shap_values.append(shap_values[i])

        # 創建DataFrame並分析
        df = pd.DataFrame({"feature": all_features, "shap_value": all_shap_values})

        # 計算特徵重要性統計
        feature_stats = (
            df.groupby("feature")
            .agg({"shap_value": ["mean", "std", "count", lambda x: np.abs(x).mean()]})
            .round(4)
        )

        feature_stats.columns = ["mean_shap", "std_shap", "count", "abs_mean_shap"]
        feature_stats = feature_stats.sort_values("abs_mean_shap", ascending=True).tail(
            max_features
        )

        # 創建summary plot
        fig, ax = plt.subplots(figsize=(10, max_features * 0.5))

        y_pos = np.arange(len(feature_stats))

        # 繪製平均SHAP值
        colors = ["red" if x > 0 else "blue" for x in feature_stats["mean_shap"]]
        bars = ax.barh(y_pos, feature_stats["abs_mean_shap"], color=colors, alpha=0.7)

        # 添加誤差條
        ax.errorbar(
            feature_stats["abs_mean_shap"],
            y_pos,
            xerr=feature_stats["std_shap"],
            fmt="none",
            color="black",
            alpha=0.5,
        )

        # 設定標籤
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_stats.index)
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title(
            f"SHAP Summary Plot - {task.capitalize()}\n"
            f"Top {max_features} Most Important Features"
        )

        # 添加數值標註
        for i, (bar, mean_val, count) in enumerate(
            zip(bars, feature_stats["mean_shap"], feature_stats["count"])
        ):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{mean_val:.3f} (n={count})",
                va="center",
                fontsize=8,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Summary plot saved to {save_path}")

        return fig


class MisclassificationAnalyzer:
    """誤判分析器"""

    def __init__(self, explainer: SHAPExplainer):
        self.explainer = explainer

    def analyze_misclassifications(
        self, texts: List[str], true_labels: List[Dict[str, int]], task: str = "toxicity"
    ) -> Dict:
        """
        分析誤判案例

        Args:
            texts: 文本列表
            true_labels: 真實標籤列表
            task: 分析的任務

        Returns:
            誤判分析結果
        """
        misclassified_cases = []
        correct_cases = []

        for text, true_label_dict in zip(texts, true_labels):
            # 獲取SHAP解釋
            result = self.explainer.explain_text(text)

            true_label = true_label_dict.get(f"{task}_label", 0)
            pred_label = getattr(result, f"{task}_pred")

            case_info = {
                "text": text,
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": getattr(result, f"{task}_prob"),
                "shap_values": getattr(result, f"{task}_shap_values"),
                "base_value": getattr(result, f"{task}_base_value"),
                "feature_importance": result.feature_importance[task],
                "tokens": result.tokens,
            }

            if true_label != pred_label:
                misclassified_cases.append(case_info)
            else:
                correct_cases.append(case_info)

        # 分析誤判模式
        analysis = self._analyze_error_patterns(misclassified_cases, correct_cases, task)

        return {
            "misclassified_cases": misclassified_cases,
            "correct_cases": correct_cases,
            "error_analysis": analysis,
            "misclassification_rate": len(misclassified_cases) / len(texts) if texts else 0,
        }

    def _analyze_error_patterns(
        self, misclassified: List[Dict], correct: List[Dict], task: str
    ) -> Dict:
        """分析錯誤模式"""
        if not misclassified:
            return {"error_patterns": "No misclassifications found"}

        # 分析置信度分佈
        misc_confidences = [case["confidence"] for case in misclassified]
        correct_confidences = [case["confidence"] for case in correct]

        # 分析特徵重要性
        misc_importance = [case["feature_importance"] for case in misclassified]
        correct_importance = [case["feature_importance"] for case in correct]

        # 分析高頻錯誤特徵
        error_features = {}
        for case in misclassified:
            shap_values = case["shap_values"]
            tokens = case["tokens"]

            # 找出最重要的特徵
            top_indices = np.argsort(np.abs(shap_values))[-5:]
            for idx in top_indices:
                if idx < len(tokens):
                    token = tokens[idx].replace("##", "")
                    if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                        error_features[token] = error_features.get(token, 0) + 1

        # 排序錯誤特徵
        sorted_error_features = sorted(error_features.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        return {
            "avg_misclassified_confidence": np.mean(misc_confidences) if misc_confidences else 0,
            "avg_correct_confidence": np.mean(correct_confidences) if correct_confidences else 0,
            "avg_misclassified_importance": np.mean(misc_importance) if misc_importance else 0,
            "avg_correct_importance": np.mean(correct_importance) if correct_importance else 0,
            "top_error_features": sorted_error_features,
            "confidence_gap": (
                (np.mean(correct_confidences) - np.mean(misc_confidences))
                if (misc_confidences and correct_confidences)
                else 0
            ),
        }

    def generate_misclassification_report(self, analysis_result: Dict, output_path: str):
        """生成誤判分析報告"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# 誤判分析報告\n\n")

            error_analysis = analysis_result["error_analysis"]

            f.write("## 總體統計\n")
            f.write(f"- 誤判率: {analysis_result['misclassification_rate']:.2%}\n")
            f.write(f"- 誤判案例數: {len(analysis_result['misclassified_cases'])}\n")
            f.write(f"- 正確案例數: {len(analysis_result['correct_cases'])}\n\n")

            if isinstance(error_analysis, dict):
                f.write("## 置信度分析\n")
                f.write(
                    f"- 誤判案例平均置信度: {error_analysis['avg_misclassified_confidence']:.3f}\n"
                )
                f.write(f"- 正確案例平均置信度: {error_analysis['avg_correct_confidence']:.3f}\n")
                f.write(f"- 置信度差距: {error_analysis['confidence_gap']:.3f}\n\n")

                f.write("## 特徵重要性分析\n")
                f.write(
                    f"- 誤判案例平均特徵重要性: {error_analysis['avg_misclassified_importance']:.3f}\n"
                )
                f.write(
                    f"- 正確案例平均特徵重要性: {error_analysis['avg_correct_importance']:.3f}\n\n"
                )

                f.write("## 高頻錯誤特徵\n")
                for feature, count in error_analysis["top_error_features"]:
                    f.write(f"- {feature}: {count} 次\n")

            f.write("\n## 誤判案例詳情\n")
            for i, case in enumerate(analysis_result["misclassified_cases"][:10]):  # 顯示前10個
                f.write(f"\n### 案例 {i+1}\n")
                f.write(f"- 文本: {case['text']}\n")
                f.write(f"- 真實標籤: {case['true_label']}\n")
                f.write(f"- 預測標籤: {case['predicted_label']}\n")
                f.write(f"- 置信度: {case['confidence']:.3f}\n")
                f.write(f"- 特徵重要性: {case['feature_importance']:.3f}\n")

        logger.info(f"Misclassification report saved to {output_path}")


def compare_ig_shap_explanations(
    ig_result: ExplanationResult, shap_result: SHAPResult, task: str = "toxicity"
) -> Dict:
    """
    比較IG和SHAP解釋結果

    Args:
        ig_result: IG解釋結果
        shap_result: SHAP解釋結果
        task: 比較的任務

    Returns:
        比較分析結果
    """
    # 獲取對應任務的attribution
    ig_attr = getattr(ig_result, f"{task}_attributions")
    shap_attr = getattr(shap_result, f"{task}_shap_values")

    # 確保長度一致
    min_len = min(len(ig_attr), len(shap_attr))
    ig_attr = ig_attr[:min_len]
    shap_attr = shap_attr[:min_len]

    # 計算相關性
    correlation = np.corrcoef(ig_attr, shap_attr)[0, 1] if min_len > 1 else 0

    # 計算排序一致性（Spearman相關）
    from scipy.stats import spearmanr

    rank_correlation, _ = spearmanr(ig_attr, shap_attr) if min_len > 1 else (0, 1)

    # 找出最重要的特徵
    ig_top_indices = np.argsort(np.abs(ig_attr))[-5:]
    shap_top_indices = np.argsort(np.abs(shap_attr))[-5:]

    # 計算特徵重疊
    overlap = len(set(ig_top_indices) & set(shap_top_indices))
    overlap_ratio = overlap / 5

    return {
        "pearson_correlation": correlation,
        "spearman_correlation": rank_correlation,
        "top_features_overlap": overlap_ratio,
        "ig_top_features": ig_top_indices.tolist(),
        "shap_top_features": shap_top_indices.tolist(),
        "attribution_difference": np.mean(np.abs(ig_attr - shap_attr)),
    }


if __name__ == "__main__":
    # 示例使用
    logger.info("SHAP explainer module loaded successfully")
    print("SHAP可解釋性模組已準備就緒")
