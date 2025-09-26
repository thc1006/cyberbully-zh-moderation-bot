#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主動學習樣本選擇器
用於選擇最有價值的樣本進行人工標註
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
import jieba

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ActiveLearningSelector:
    """主動學習樣本選擇器"""

    def __init__(self, model_name: str = "hfl/chinese-macbert-base", device: str = "auto"):
        """初始化選擇器

        Args:
            model_name: 預訓練模型名稱
            device: 計算裝置 (cpu/cuda/auto)
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name

        # 載入預訓練模型和分詞器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        logger.info(f"初始化完成，使用模型: {model_name}, 裝置: {device}")

    def preprocess_text(self, text: str) -> str:
        """文字預處理

        Args:
            text: 原始文字

        Returns:
            處理後的文字
        """
        if not isinstance(text, str):
            return ""

        # 移除多餘空白字元
        text = ' '.join(text.split())

        # 如果文字過短，直接返回
        if len(text) < 5:
            return text

        return text

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """獲取文字嵌入向量

        Args:
            texts: 文字列表
            batch_size: 批次大小

        Returns:
            嵌入向量矩陣 (n_samples, embedding_dim)
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # 預處理文字
            processed_texts = [self.preprocess_text(text) for text in batch_texts]

            # 分詞和編碼
            inputs = self.tokenizer(
                processed_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # 獲取嵌入向量
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用 [CLS] token 的嵌入作為句子表示
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def uncertainty_sampling(self, probabilities: np.ndarray, n_samples: int) -> List[int]:
        """不確定性採樣

        Args:
            probabilities: 模型預測機率 (n_samples, n_classes)
            n_samples: 選擇的樣本數量

        Returns:
            選中樣本的索引列表
        """
        # 計算熵作為不確定性度量
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)

        # 選擇熵最高的樣本
        uncertain_indices = np.argsort(entropy)[-n_samples:]

        return uncertain_indices.tolist()

    def diversity_sampling(self, embeddings: np.ndarray, n_samples: int) -> List[int]:
        """多樣性採樣

        Args:
            embeddings: 文字嵌入向量
            n_samples: 選擇的樣本數量

        Returns:
            選中樣本的索引列表
        """
        selected_indices = []
        remaining_indices = list(range(len(embeddings)))

        # 隨機選擇第一個樣本
        first_idx = np.random.choice(remaining_indices)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # 迭代選擇最不相似的樣本
        for _ in range(n_samples - 1):
            if not remaining_indices:
                break

            # 計算剩餘樣本與已選樣本的最大相似度
            selected_embeddings = embeddings[selected_indices]
            remaining_embeddings = embeddings[remaining_indices]

            # 計算餘弦相似度
            similarities = cosine_similarity(remaining_embeddings, selected_embeddings)
            max_similarities = np.max(similarities, axis=1)

            # 選擇最不相似的樣本
            min_sim_idx = np.argmin(max_similarities)
            selected_idx = remaining_indices[min_sim_idx]

            selected_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)

        return selected_indices

    def combined_sampling(self, texts: List[str], probabilities: Optional[np.ndarray] = None,
                         n_samples: int = 500, uncertainty_ratio: float = 0.7) -> List[int]:
        """結合不確定性和多樣性的採樣策略

        Args:
            texts: 文字列表
            probabilities: 模型預測機率（可選）
            n_samples: 選擇的樣本數量
            uncertainty_ratio: 不確定性採樣的比例

        Returns:
            選中樣本的索引列表
        """
        logger.info(f"開始樣本選擇，目標數量: {n_samples}")

        # 計算各採樣策略的數量
        n_uncertainty = int(n_samples * uncertainty_ratio)
        n_diversity = n_samples - n_uncertainty

        selected_indices = []

        # 不確定性採樣
        if probabilities is not None and n_uncertainty > 0:
            logger.info(f"執行不確定性採樣，選擇 {n_uncertainty} 個樣本")
            uncertainty_indices = self.uncertainty_sampling(probabilities, n_uncertainty)
            selected_indices.extend(uncertainty_indices)

        # 多樣性採樣
        if n_diversity > 0:
            logger.info(f"執行多樣性採樣，選擇 {n_diversity} 個樣本")
            logger.info("正在計算文字嵌入向量...")
            embeddings = self.get_embeddings(texts)

            # 排除已選樣本
            available_indices = [i for i in range(len(texts)) if i not in selected_indices]
            if len(available_indices) < n_diversity:
                n_diversity = len(available_indices)

            if available_indices:
                available_embeddings = embeddings[available_indices]
                diversity_indices_relative = self.diversity_sampling(available_embeddings, n_diversity)
                diversity_indices = [available_indices[i] for i in diversity_indices_relative]
                selected_indices.extend(diversity_indices)

        # 如果沒有提供機率且不執行多樣性採樣，隨機選擇
        if not selected_indices:
            logger.info(f"隨機選擇 {n_samples} 個樣本")
            selected_indices = np.random.choice(len(texts), size=min(n_samples, len(texts)), replace=False).tolist()

        logger.info(f"樣本選擇完成，共選擇 {len(selected_indices)} 個樣本")
        return selected_indices

    def load_candidate_data(self, data_path: str) -> Tuple[List[str], List[Dict]]:
        """載入候選標註資料

        Args:
            data_path: 資料檔案路徑

        Returns:
            (文字列表, 原始資料列表)
        """
        logger.info(f"載入候選資料: {data_path}")

        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_path.endswith('.jsonl'):
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"不支援的檔案格式: {data_path}")

        # 提取文字
        texts = []
        for item in data:
            if isinstance(item, dict):
                # 嘗試常見的文字欄位名稱
                text = item.get('text') or item.get('content') or item.get('message') or item.get('sentence')
                if text:
                    texts.append(str(text))
                else:
                    texts.append("")
            else:
                texts.append(str(item))

        logger.info(f"載入 {len(texts)} 個樣本")
        return texts, data

    def save_selected_samples(self, selected_indices: List[int], original_data: List[Dict],
                            output_path: str, include_metadata: bool = True):
        """儲存選中的樣本

        Args:
            selected_indices: 選中樣本的索引
            original_data: 原始資料
            output_path: 輸出檔案路徑
            include_metadata: 是否包含元資料
        """
        logger.info(f"儲存選中樣本到: {output_path}")

        # 準備輸出資料
        selected_data = []
        for idx in selected_indices:
            if idx < len(original_data):
                sample = original_data[idx].copy()

                if include_metadata:
                    sample['annotation_metadata'] = {
                        'original_index': idx,
                        'selection_method': 'active_learning',
                        'annotation_status': 'pending',
                        'annotation_priority': 'high' if idx in selected_indices[:100] else 'medium'
                    }

                selected_data.append(sample)

        # 儲存檔案
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(selected_data, f, ensure_ascii=False, indent=2)
        elif output_path.endswith('.jsonl'):
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in selected_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif output_path.endswith('.csv'):
            df = pd.DataFrame(selected_data)
            df.to_csv(output_path, index=False, encoding='utf-8')

        logger.info(f"成功儲存 {len(selected_data)} 個樣本")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="主動學習樣本選擇器")

    parser.add_argument("--input", required=True, help="輸入資料檔案路徑")
    parser.add_argument("--output", required=True, help="輸出檔案路徑")
    parser.add_argument("--n_samples", type=int, default=500, help="選擇的樣本數量")
    parser.add_argument("--uncertainty_ratio", type=float, default=0.7, help="不確定性採樣比例")
    parser.add_argument("--model_name", default="hfl/chinese-macbert-base", help="預訓練模型名稱")
    parser.add_argument("--device", default="auto", help="計算裝置")
    parser.add_argument("--probabilities", help="模型預測機率檔案路徑（可選）")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")

    args = parser.parse_args()

    # 設定隨機種子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 初始化選擇器
    selector = ActiveLearningSelector(
        model_name=args.model_name,
        device=args.device
    )

    # 載入候選資料
    texts, original_data = selector.load_candidate_data(args.input)

    # 載入預測機率（如果提供）
    probabilities = None
    if args.probabilities:
        logger.info(f"載入預測機率: {args.probabilities}")
        probabilities = np.load(args.probabilities)

    # 執行樣本選擇
    selected_indices = selector.combined_sampling(
        texts=texts,
        probabilities=probabilities,
        n_samples=args.n_samples,
        uncertainty_ratio=args.uncertainty_ratio
    )

    # 儲存結果
    selector.save_selected_samples(
        selected_indices=selected_indices,
        original_data=original_data,
        output_path=args.output
    )

    # 輸出統計資訊
    print(f"\n=== 樣本選擇完成 ===")
    print(f"輸入樣本數量: {len(texts)}")
    print(f"選擇樣本數量: {len(selected_indices)}")
    print(f"選擇比例: {len(selected_indices)/len(texts)*100:.2f}%")
    print(f"輸出檔案: {args.output}")


if __name__ == "__main__":
    main()