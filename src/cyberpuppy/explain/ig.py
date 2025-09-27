#!/usr/bin/env python3
"""
Integrated Gradients 解釋性AI實現
使用Captum庫為中文網路霸凌偵測模型提供可解釋性分析

參考文獻與範例：
- Captum官方文檔: https://captum.ai/
- Integrated Gradients論文: https://arxiv.org/abs/1703.01365
- Captum教程: https://captum.ai/tutorials/
- 中文NLP解釋性範例: https://github.com/pytorch/captum/blob/master/tutorials/
"""

import csv
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Captum imports
# 參考: https://captum.ai/api/integrated_gradients.html
from captum.attr import IntegratedGradients, TokenReferenceBase

# 本地imports
from ..models.baselines import BaselineModel

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


@dataclass
class ExplanationResult:
    """解釋結果資料結構"""

    text: str
    tokens: List[str]

    # 預測結果
    toxicity_pred: int
    toxicity_prob: float
    emotion_pred: int
    emotion_prob: float
    bullying_pred: int
    bullying_prob: float

    # Attribution分數
    toxicity_attributions: np.ndarray  # [seq_len]
    emotion_attributions: np.ndarray  # [seq_len]
    bullying_attributions: np.ndarray  # [seq_len]

    # 額外資訊
    convergence_delta: float
    prediction_confidence: Dict[str, float]


class IntegratedGradientsExplainer:
    """
    Integrated Gradients解釋器

    基於Captum實現，參考範例：
    https://captum.ai/tutorials/Bert_SQUAD_Interpret
    """

    def __init__(self, model: BaselineModel, device: Optional[torch.device] = None):
        """
        初始化IG解釋器

        Args:
            model: 已訓練的基線模型
            device: 計算設備
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

        # 設定參考baseline (PAD token)
        # 參考: https://captum.ai/api/base_classes.html#tokenreferencebase
        try:
            self.pad_token_id = self.model.tokenizer.pad_token_id
            if self.pad_token_id is None:
                self.pad_token_id = 0
        except AttributeError:
            self.pad_token_id = 0

        self.token_reference = TokenReferenceBase(reference_token_idx=self.pad_token_id)

        # 初始化IG解釋器 - 針對不同任務
        # 參考: https://captum.ai/api/integrated_gradients.html
        self.ig_toxicity = IntegratedGradients(self._forward_toxicity)
        self.ig_emotion = IntegratedGradients(self._forward_emotion)
        self.ig_bullying = IntegratedGradients(self._forward_bullying)

        logger.info(f"IG Explainer initialized on device: {self.device}")

    def _forward_toxicity(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        毒性分類的前向傳播wrapper

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            毒性預測邏輯值 [batch_size, num_classes]
        """
        outputs = self.model(input_ids, attention_mask)
        return outputs["toxicity"]

    def _forward_emotion(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """情緒分類的前向傳播wrapper"""
        outputs = self.model(input_ids, attention_mask)
        return outputs["emotion"]

    def _forward_bullying(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """霸凌分類的前向傳播wrapper"""
        outputs = self.model(input_ids, attention_mask)
        return outputs["bullying"]

    def explain_text(
        self,
        text: str,
        target_class: Optional[int] = None,
        n_steps: int = 50,
        internal_batch_size: int = 10,
    ) -> ExplanationResult:
        """
        解釋單個文本的模型決策

        Args:
            text: 輸入文本
            target_class: 目標類別（None表示使用預測類別）
            n_steps: IG積分步數
            internal_batch_size: 內部批次大小

        Returns:
            解釋結果

        參考Captum教程：
        https://captum.ai/tutorials/Bert_SQUAD_Interpret2
        """
        logger.info(f"Explaining text: {text[:50]}...")

        # 分詞
        inputs = self.model.tokenizer(
            text,
            max_length=self.model.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # 獲取tokens（用於可視化）
        tokens = self.model.tokenizer.convert_ids_to_tokens(input_ids[0])

        # 獲取預測結果
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

            toxicity_logits = outputs["toxicity"]
            emotion_logits = outputs["emotion"]
            bullying_logits = outputs["bullying"]

            toxicity_probs = F.softmax(toxicity_logits, dim=-1)
            emotion_probs = F.softmax(emotion_logits, dim=-1)
            bullying_probs = F.softmax(bullying_logits, dim=-1)

            toxicity_pred = torch.argmax(toxicity_logits, dim=-1).item()
            emotion_pred = torch.argmax(emotion_logits, dim=-1).item()
            bullying_pred = torch.argmax(bullying_logits, dim=-1).item()

        # 設定基線（參考輸入）
        # 使用PAD token作為基線 - 參考Captum文檔建議
        reference_indices = self.token_reference.generate_reference(
            sequence_length=input_ids.size(1), device=self.device
        ).unsqueeze(0)

        # 計算Integrated Gradients attribution
        # 參考: https://captum.ai/api/integrated_gradients.html
        # #captum.attr.IntegratedGradients.attribute
        try:
            # 毒性attribution
            toxicity_target = (
                target_class if target_class is not None else toxicity_pred
            )
            toxicity_attributions = self.ig_toxicity.attribute(
                inputs=(input_ids, attention_mask),
                baselines=(reference_indices, torch.zeros_like(attention_mask)),
                target=toxicity_target,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=True,
            )

            toxicity_attr = toxicity_attributions[0][0]  # [seq_len]
            convergence_delta = toxicity_attributions[1].item()

            # 情緒attribution
            emotion_target = emotion_pred
            emotion_attr = self.ig_emotion.attribute(
                inputs=(input_ids, attention_mask),
                baselines=(reference_indices, torch.zeros_like(attention_mask)),
                target=emotion_target,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
            )[
                0
            ]  # [seq_len]

            # 霸凌attribution
            bullying_target = bullying_pred
            bullying_attr = self.ig_bullying.attribute(
                inputs=(input_ids, attention_mask),
                baselines=(reference_indices, torch.zeros_like(attention_mask)),
                target=bullying_target,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
            )[
                0
            ]  # [seq_len]

        except Exception as e:
            logger.error(f"Attribution computation failed: {e}")
            seq_len = input_ids.size(1)
            toxicity_attr = torch.zeros(seq_len)
            emotion_attr = torch.zeros(seq_len)
            bullying_attr = torch.zeros(seq_len)
            convergence_delta = 0.0

        # 構建結果
        result = ExplanationResult(
            text=text,
            tokens=tokens,
            toxicity_pred=toxicity_pred,
            toxicity_prob=toxicity_probs[0, toxicity_pred].item(),
            emotion_pred=emotion_pred,
            emotion_prob=emotion_probs[0, emotion_pred].item(),
            bullying_pred=bullying_pred,
            bullying_prob=bullying_probs[0, bullying_pred].item(),
            toxicity_attributions=toxicity_attr.detach().cpu().numpy(),
            emotion_attributions=emotion_attr.detach().cpu().numpy(),
            bullying_attributions=bullying_attr.detach().cpu().numpy(),
            convergence_delta=convergence_delta,
            prediction_confidence={
                "toxicity": toxicity_probs[0, toxicity_pred].item(),
                "emotion": emotion_probs[0, emotion_pred].item(),
                "bullying": bullying_probs[0, bullying_pred].item(),
            },
        )

        logger.info(
            f"Attribution computed. Convergenc" "e delta: {convergence_delta:.4f}"
        )
        return result


class BiasAnalyzer:
    """
    模型偏見分析器
    檢測身份攻擊、辱罵詞等偏見模式
    """

    def __init__(self, explainer: IntegratedGradientsExplainer):
        """
        初始化偏見分析器

        Args:
            explainer: IG解釋器實例
        """
        self.explainer = explainer

        # 定義偏見詞彙集合（可擴展）
        self.identity_terms = {
            # 性別相關
            "gender": [
                "男",
                "女",
                "男性",
                "女性",
                "男人",
                "女人",
                "先生",
                "女士",
                "小姐",
            ],
            # 種族/地域相關
            "ethnicity": [
                "中國",
                "日本",
                "韓國",
                "美國",
                "台灣",
                "香港",
                "大陸",
                "內地",
            ],
            # 職業相關
            "occupation": ["老師", "學生", "醫生", "護士", "工程師", "農民", "工人"],
            # 年齡相關
            "age": ["老人", "年輕", "小孩", "嬰兒", "青少年", "中年", "老年"],
        }

        # 辱罵詞類別（使用溫和的代表詞，實際使用時需要擴展）
        self.offensive_terms = {
            "mild_insult": ["笨", "蠢", "傻"],
            "strong_insult": ["廢物", "垃圾"],  # 實際使用時需要更全面的詞庫
            "discriminatory": ["歧視", "偏見", "仇恨"],
        }

    def analyze_bias_patterns(
        self, texts: List[str], top_k: int = 10, output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        分析文本集合中的偏見模式

        Args:
            texts: 待分析的文本列表
            top_k: 返回每個類別的前K個重要詞
            output_csv: CSV輸出路徑

        Returns:
            包含偏見分析結果的DataFrame
        """
        logger.info(f"Analyzing bias patterns for {len(texts)} texts...")

        bias_results = []

        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info(f"Processing text {i+1}/{len(texts)}")

            try:
                # 獲取解釋結果
                result = self.explainer.explain_text(text)

                # 檢查每個token的偏見屬性
                for j, token in enumerate(result.tokens):
                    # 清理token（移除##等sub-word標記）
                    clean_token = token.replace("##", "").strip()
                    if len(clean_token) < 1 or clean_token in ["[CLS]"]:
                        continue

                    # 檢查是否為偏見相關詞彙
                    bias_category = self._classify_bias_term(clean_token)
                    if bias_category is not None:
                        bias_results.append(
                            {
                                "text_id": i,
                                "text": text[:100] + "...",
                                "token": clean_token,
                                "token_position": j,
                                "bias_category": bias_category[0],
                                "bias_subcategory": bias_category[1],
                                "toxicity_a"
                                "ttribution": result.toxicity_attributions[j],
                                "emotion_a"
                                "ttribution": result.emotion_attributions[j],
                                "bullying_a"
                                "ttribution": result.bullying_attributions[j],
                                "toxicity_pred": result.toxicity_pred,
                                "toxicity_prob": result.toxicity_prob,
                                "emotion_pred": result.emotion_pred,
                                "emotion_prob": result.emotion_prob,
                                "convergence_delta": result.convergence_delta,
                            }
                        )

            except Exception as e:
                logger.warning(f"Failed to analyze text {i}: {e}")
                continue

        # 轉換為DataFrame
        df = pd.DataFrame(bias_results)

        if len(df) == 0:
            logger.warning("No bias patterns found in the provided texts")
            return pd.DataFrame()

        # 計算統計資訊
        df = self._compute_bias_statistics(df, top_k)

        # 儲存CSV報告
        if output_csv:
            self.save_bias_report(df, output_csv, top_k)

        logger.info(f"Bias analysis completed. Found" " {len(df)} bias-related tokens.")
        return df

    def _classify_bias_term(self, token: str) -> Optional[Tuple[str, str]]:
        """
        分類偏見詞彙

        Args:
            token: 待分類的token

        Returns:
            (主類別, 子類別) 或 None
        """
        token_lower = token.lower()

        # 檢查身份相關詞彙
        for subcategory, terms in self.identity_terms.items():
            if any(term in token_lower for term in terms):
                return ("identity", subcategory)

        # 檢查辱罵詞彙
        for subcategory, terms in self.offensive_terms.items():
            if any(term in token_lower for term in terms):
                return ("offensive", subcategory)

        return None

    def _compute_bias_statistics(self, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """計算偏見統計資訊"""
        # 按類別和attribution分數排序
        df["abs_toxicity_attr"] = df["toxicity_attribution"].abs()
        df["abs_emotion_attr"] = df["emotion_attribution"].abs()
        df["abs_bullying_attr"] = df["bullying_attribution"].abs()

        # 計算總體重要度
        df["total_importance"] = (
            df["abs_toxicity_attr"] + df["abs_emotion_attr"] + df["abs_bullying_attr"]
        ) / 3

        return df

    def save_bias_report(self, df: pd.DataFrame, output_path: str, top_k: int):
        """
        儲存偏見分析報告

        Args:
            df: 偏見分析結果
            output_path: 輸出CSV路徑
            top_k: 每類別顯示的top-k結果
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 生成總體報告
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)

            # 寫入標題
            writer.writerow(
                [
                    "rank",
                    "token",
                    "bias_category",
                    "bias_subcategory",
                    "toxicity_attribution",
                    "emotion_attribution",
                    "bullying_attribution",
                    "total_importance",
                    "frequency",
                    "avg_toxicity_prob",
                    "sample_text",
                ]
            )

            # 按類別分組並寫入Top-K結果
            for category in df["bias_category"].unique():
                writer.writerow([f"=== {category.upper()} BIAS PATTERNS ==="])

                category_df = df[df["bias_category"] == category]

                # 按總重要度排序並取Top-K
                top_tokens = (
                    category_df.groupby("token")
                    .agg(
                        {
                            "toxicity_attribution": "mean",
                            "emotion_attribution": "mean",
                            "bullying_attribution": "mean",
                            "total_importance": "mean",
                            "toxicity_prob": "mean",
                            "bias_subcategory": "first",
                            "text": "first",
                        }
                    )
                    .reset_index()
                    .sort_values("total_importance", ascending=False)
                    .head(top_k)
                )

                for i, row in enumerate(top_tokens.itertuples(), 1):
                    frequency = len(category_df[category_df["to" "ken"] == row.token])
                    writer.writerow(
                        [
                            i,
                            row.token,
                            category,
                            row.bias_subcategory,
                            f"{row.toxicity_attribution:.4f}",
                            f"{row.emotion_attribution:.4f}",
                            f"{row.bullying_attribution:.4f}",
                            f"{row.total_importance:.4f}",
                            frequency,
                            f"{row.toxicity_prob:.3f}",
                            (
                                row.text[:50] + "." ".."
                                if len(row.text) > 50
                                else row.text
                            ),
                        ]
                    )

                writer.writerow([])  # 空行分隔

        logger.info(f"Bias analysis report saved to {output_path}")


def create_attribution_heatmap(
    result: ExplanationResult,
    task: str = "toxicity",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    創建attribution熱力圖

    Args:
        result: 解釋結果
        task: 任務類型 ('toxicity', 'emotion', 'bullying')
        save_path: 圖片儲存路徑
        figsize: 圖片大小

    Returns:
        matplotlib Figure對象

    參考Captum可視化範例:
    https://captum.ai/api/utilities.html#captum.attr.visualization.visualize_text
    """
    # 選擇對應任務的attribution
    if task == "toxicity":
        attributions = result.toxicity_attributions
        pred_class = result.toxicity_pred
        prob = result.toxicity_prob
        class_names = ["非毒性", "毒性", "嚴重毒性"]
    elif task == "emotion":
        attributions = result.emotion_attributions
        pred_class = result.emotion_pred
        prob = result.emotion_prob
        class_names = ["正面", "中性", "負面"]
    elif task == "bullying":
        attributions = result.bullying_attributions
        pred_class = result.bullying_pred
        prob = result.bullying_prob
        class_names = ["非霸凌", "騷擾", "威脅"]
    else:
        raise ValueError(f"Unsupported task: {task}")

    # 過濾特殊token
    tokens = []
    attrs = []
    for i, token in enumerate(result.tokens):
        if token not in ["[CLS]", "[SEP]", "[PAD]"] and i < len(attributions):
            # 清理sub-word token
            clean_token = token.replace("##", "")
            tokens.append(clean_token)
            attrs.append(attributions[i])

    # 正規化attribution分數到[-1, 1]
    attrs = np.array(attrs)
    if np.max(np.abs(attrs)) > 0:
        attrs = attrs / np.max(np.abs(attrs))

    # 創建熱力圖
    fig, ax = plt.subplots(figsize=figsize)

    # 設定顏色映射（紅色=正貢獻，藍色=負貢獻）
    colors = []
    for attr in attrs:
        if attr > 0:
            colors.append(plt.cm.Reds(attr))  # 正向attribution用紅色
        else:
            colors.append(plt.cm.Blues(-attr))  # 負向attribution用藍色

    # 繪製條形圖
    ax.barh(range(len(tokens)), attrs, color=colors)

    # 設定標籤和標題
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=10)
    ax.set_xlabel("Attribution Score", fontsize=12)
    ax.set_title(
        f"{task.capitalize()} Attribution Heatmap\n"
        f"Prediction: {class_names[pred_class]} (Prob: {prob:.3f})\n"
        f"Text: {result.text[:60]}...",
        fontsize=14,
    )

    # 添加垂直線表示零點
    ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)

    # 添加顏色說明
    ax.text(
        0.02,
        0.98,
        "Red: Positive Attribution\nBlue: Negative Attribution",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # 儲存圖片
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Heatmap saved to {save_path}")

    return fig


def save_attribution_report(
    results: List[ExplanationResult], output_path: str, include_tokens: bool = True
):
    """
    儲存attribution報告到CSV

    Args:
        results: 解釋結果列表
        output_path: CSV輸出路徑
        include_tokens: 是否包含token級詳細資訊
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 準備資料
    report_data = []

    for i, result in enumerate(results):
        # 基本資訊
        base_info = {
            "text_id": i,
            "text": result.text,
            "toxicity_pred": result.toxicity_pred,
            "toxicity_prob": result.toxicity_prob,
            "emotion_pred": result.emotion_pred,
            "emotion_prob": result.emotion_prob,
            "bullying_pred": result.bullying_pred,
            "bullying_prob": result.bullying_prob,
            "convergence_delta": result.convergence_delta,
        }

        if include_tokens:
            # Token級詳細資訊
            for j, token in enumerate(result.tokens):
                if token not in ["[CLS]", "[SEP]", "[PAD]"] and j < len(
                    result.toxicity_attributions
                ):
                    token_info = base_info.copy()
                    token_info.update(
                        {
                            "token_position": j,
                            "token": token.replace("##", ""),
                            "toxicity_attr": result.toxicity_attributions[j],
                            "emotion_attr": result.emotion_attributions[j],
                            "bullying_attr": result.bullying_attributions[j],
                        }
                    )
                    report_data.append(token_info)
        else:
            # 僅文本級資訊
            base_info.update(
                {
                    "top_toxicity_tokens": _get_top_tokens(
                        result.tokens, result.toxicity_attributions, 3
                    ),
                    "top_emotion_tokens": _get_top_tokens(
                        result.tokens, result.emotion_attributions, 3
                    ),
                    "top_bullying_tokens": _get_top_tokens(
                        result.tokens, result.bullying_attributions, 3
                    ),
                }
            )
            report_data.append(base_info)

    # 儲存CSV
    df = pd.DataFrame(report_data)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info(f"Attribution report saved to {output_path}")


def _get_top_tokens(tokens: List[str], attributions: np.ndarray, k: int = 3) -> str:
    """獲取attribution分數最高的k個tokens"""
    # 過濾特殊tokens
    valid_indices = []
    for i, token in enumerate(tokens):
        if token not in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]:
            valid_indices.append(i)

    if not valid_indices:
        return ""

    # 獲取top-k
    valid_attrs = attributions[valid_indices]
    valid_tokens = [tokens[i].replace("##", "") for i in valid_indices]

    top_k_indices = np.argsort(np.abs(valid_attrs))[-k:][::-1]
    top_tokens = [f"{valid_tokens[i]}({valid_attrs[i]:.3f})" for i in top_k_indices]

    return "; ".join(top_tokens)


# 使用範例和測試函數
def demo_integrated_gradients():
    """
    示範如何使用Integrated Gradients解釋器

    參考Captum教程進行實現:
    https://captum.ai/tutorials/Bert_SQUAD_Interpret
    """
    logger.info("Running Integrated Gradients demo...")

    # 注意：實際使用時需要載入已訓練的模型
    # model = BaselineModel.load_model('path/to/model')
    # explainer = IntegratedGradientsExplainer(model)

    # 示例文本（暫時未使用）
    # test_texts = [
    #     "你這個垃圾，滾開！",
    #     "今天天氣很好，心情也不錯",
    #     "這個政策真的很爛，完全沒用",
    #     "謝謝你的幫助，非常感謝",
    # ]

    # results = []
    # for text in test_texts:
    #     result = explainer.explain_text(text)
    #     results.append(result)
    #
    #     # 創建熱力圖
    #     fig = create_attribution_heatmap(result, 'toxicity')
    #     plt.show()

    # # 偏見分析
    # bias_analyzer = BiasAnalyzer(explainer)
    # bias_df = bias_analyzer.analyze_bias_patterns(
    #     test_texts,
    #     top_k=5,
    #     output_csv='reports/bias_analysis.csv'
    # )

    logger.info("Demo completed. Check output files for results.")


# Backwards compatibility aliases
IGExplainer = IntegratedGradientsExplainer

# Mock config for compatibility
class IGConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


if __name__ == "__main__":
    demo_integrated_gradients()
