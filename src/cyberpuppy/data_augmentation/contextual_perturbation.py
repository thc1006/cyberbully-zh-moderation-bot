"""
上下文擾動模組

使用 MacBERT 的 [MASK] 預測來生成語義相似的文本變體。
"""

import random
import logging
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

logger = logging.getLogger(__name__)

try:
    from transformers import BertTokenizer, BertForMaskedLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers 未安裝，請執行: pip install transformers torch")

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("jieba 未安裝，請執行: pip install jieba")


class ContextualPerturber:
    """上下文擾動器"""

    def __init__(
        self,
        model_name: str = "hfl/chinese-macbert-base",
        mask_probability: float = 0.15,
        max_predictions: int = 5,
        min_similarity_threshold: float = 0.3,
        preserve_keywords: Optional[Set[str]] = None,
        device: Optional[str] = None
    ):
        """
        初始化上下文擾動器

        Args:
            model_name: 預訓練模型名稱
            mask_probability: 遮蔽概率
            max_predictions: 最大預測候選數
            min_similarity_threshold: 最小相似度閾值
            preserve_keywords: 需要保護的關鍵詞
            device: 計算設備
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("需要安裝 transformers 和 torch")

        if not JIEBA_AVAILABLE:
            raise ImportError("需要安裝 jieba")

        self.model_name = model_name
        self.mask_probability = mask_probability
        self.max_predictions = max_predictions
        self.min_similarity_threshold = min_similarity_threshold
        self.preserve_keywords = preserve_keywords or set()

        # 設置設備
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 載入模型和分詞器
        self._load_model()

        # 霸凌關鍵詞 (預設保護詞彙)
        self.default_preserve_keywords = {
            "死", "殺", "滾", "廢物", "垃圾", "白癡", "智障", "腦殘",
            "去死", "找死", "該死", "霸凌", "威脅", "恐嚇", "排擠"
        }

        logger.info(f"上下文擾動器初始化完成，使用模型: {model_name}")
        logger.info(f"設備: {self.device}")

    def _load_model(self):
        """載入預訓練模型"""
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertForMaskedLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"模型載入成功: {self.model_name}")
        except Exception as e:
            logger.error(f"模型載入失敗: {e}")
            raise e

    def augment(
        self,
        text: str,
        num_augmented: int = 1,
        custom_mask_prob: Optional[float] = None
    ) -> List[str]:
        """
        對文本進行上下文擾動增強

        Args:
            text: 原始文本
            num_augmented: 生成增強樣本數量
            custom_mask_prob: 自定義遮蔽概率

        Returns:
            上下文擾動後的文本列表
        """
        if not text.strip():
            return [text] * num_augmented

        mask_prob = custom_mask_prob or self.mask_probability
        augmented_texts = []

        for i in range(num_augmented):
            try:
                perturbed_text = self._contextual_perturbation(text, mask_prob)
                augmented_texts.append(perturbed_text)
            except Exception as e:
                logger.warning(f"上下文擾動失敗: {e}")
                augmented_texts.append(text)

        return augmented_texts

    def _contextual_perturbation(self, text: str, mask_prob: float) -> str:
        """執行上下文擾動"""
        # 分詞
        words = list(jieba.cut(text))

        if len(words) <= 1:
            return text

        # 選擇要遮蔽的詞位置
        maskable_positions = self._get_maskable_positions(words)

        if not maskable_positions:
            return text

        # 計算遮蔽數量
        num_masks = max(1, int(len(maskable_positions) * mask_prob))
        positions_to_mask = random.sample(maskable_positions, min(num_masks, len(maskable_positions)))

        # 逐個位置進行遮蔽和預測
        result_words = words.copy()
        for pos in positions_to_mask:
            masked_text = self._create_masked_text(result_words, pos)
            predicted_word = self._predict_masked_word(masked_text, pos)
            if predicted_word and predicted_word != words[pos]:
                result_words[pos] = predicted_word

        return ''.join(result_words)

    def _get_maskable_positions(self, words: List[str]) -> List[int]:
        """獲取可遮蔽的詞位置"""
        maskable_positions = []

        for i, word in enumerate(words):
            if self._is_maskable(word):
                maskable_positions.append(i)

        return maskable_positions

    def _is_maskable(self, word: str) -> bool:
        """判斷詞是否可遮蔽"""
        # 過濾標點符號
        if word in ['，', '。', '！', '？', '；', '：', '"', '"', ''', ''', '（', '）', '【', '】']:
            return False

        # 過濾過短的詞
        if len(word.strip()) < 1:
            return False

        # 保護關鍵詞
        all_preserve_keywords = self.preserve_keywords | self.default_preserve_keywords
        if word in all_preserve_keywords:
            return False

        # 過濾數字
        if word.isdigit():
            return False

        return True

    def _create_masked_text(self, words: List[str], mask_position: int) -> str:
        """創建遮蔽文本"""
        masked_words = words.copy()
        masked_words[mask_position] = '[MASK]'
        return ''.join(masked_words)

    def _predict_masked_word(self, masked_text: str, original_position: int) -> Optional[str]:
        """預測遮蔽位置的詞"""
        try:
            # 分詞和編碼
            inputs = self.tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 找到 [MASK] 標記的位置
            mask_token_id = self.tokenizer.mask_token_id
            mask_positions = (inputs['input_ids'] == mask_token_id).nonzero(as_tuple=True)

            if len(mask_positions[1]) == 0:
                return None

            mask_pos = mask_positions[1][0].item()

            # 預測
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits[0, mask_pos]

                # 獲取 top-k 候選
                top_k_tokens = torch.topk(predictions, self.max_predictions)

                # 轉換為詞彙並過濾
                candidates = []
                for token_id, score in zip(top_k_tokens.indices, top_k_tokens.values):
                    token = self.tokenizer.decode(token_id.item()).strip()

                    # 過濾候選
                    if self._is_valid_candidate(token, score.item()):
                        candidates.append((token, score.item()))

                # 隨機選擇一個候選 (按概率權重)
                if candidates:
                    if len(candidates) == 1:
                        return candidates[0][0]
                    else:
                        # 使用 softmax 加權隨機選擇
                        scores = [score for _, score in candidates]
                        probs = torch.softmax(torch.tensor(scores), dim=0).numpy()
                        chosen_idx = np.random.choice(len(candidates), p=probs)
                        return candidates[chosen_idx][0]

                return None

        except Exception as e:
            logger.warning(f"預測遮蔽詞失敗: {e}")
            return None

    def _is_valid_candidate(self, token: str, score: float) -> bool:
        """檢查候選詞是否有效"""
        # 分數閾值
        if score < self.min_similarity_threshold:
            return False

        # 過濾特殊標記
        special_tokens = ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']
        if token in special_tokens:
            return False

        # 過濾空白和標點
        if not token.strip() or token in ['，', '。', '！', '？']:
            return False

        # 過濾英文字母 (主要處理中文)
        if token.isalpha() and all(ord(c) < 128 for c in token):
            return False

        return True

    def batch_augment(
        self,
        texts: List[str],
        num_augmented: int = 1
    ) -> List[List[str]]:
        """批量上下文擾動"""
        results = []
        for text in texts:
            try:
                augmented = self.augment(text, num_augmented)
                results.append(augmented)
            except Exception as e:
                logger.warning(f"批量處理失敗: {e}")
                results.append([text] * num_augmented)

        return results

    def add_preserve_keyword(self, keyword: str) -> None:
        """添加保護關鍵詞"""
        self.preserve_keywords.add(keyword)
        logger.info(f"已添加保護關鍵詞: {keyword}")

    def remove_preserve_keyword(self, keyword: str) -> None:
        """移除保護關鍵詞"""
        self.preserve_keywords.discard(keyword)
        logger.info(f"已移除保護關鍵詞: {keyword}")

    def set_mask_probability(self, prob: float) -> None:
        """設置遮蔽概率"""
        if 0 < prob <= 1:
            self.mask_probability = prob
            logger.info(f"遮蔽概率已設置為: {prob}")
        else:
            raise ValueError("遮蔽概率必須在 (0, 1] 範圍內")

    def get_model_info(self) -> Dict[str, str]:
        """獲取模型信息"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "vocab_size": len(self.tokenizer),
            "mask_token": self.tokenizer.mask_token,
            "model_type": type(self.model).__name__
        }

    def get_stats(self) -> Dict[str, any]:
        """獲取統計信息"""
        return {
            "model_name": self.model_name,
            "mask_probability": self.mask_probability,
            "max_predictions": self.max_predictions,
            "min_similarity_threshold": self.min_similarity_threshold,
            "preserve_keywords_count": len(self.preserve_keywords),
            "device": str(self.device)
        }


class LightweightPerturber:
    """輕量級擾動器 (不依賴大型模型)"""

    def __init__(self):
        """初始化輕量級擾動器"""
        if not JIEBA_AVAILABLE:
            raise ImportError("需要安裝 jieba")

        # 預定義替換規則
        self.replacement_rules = {
            "很": ["非常", "特別", "相當", "極其"],
            "說": ["講", "談", "表示", "提到"],
            "好": ["棒", "讚", "優秀", "不錯"],
            "看": ["瞧", "觀察", "注視"],
            "想": ["思考", "考慮", "認為"],
            "大": ["巨大", "龐大", "寬廣"],
            "小": ["微小", "細小", "迷你"],
            "快": ["迅速", "敏捷", "急速"],
            "慢": ["緩慢", "遲緩", "徐緩"]
        }

        logger.info("輕量級擾動器初始化完成")

    def augment(self, text: str, num_augmented: int = 1) -> List[str]:
        """輕量級上下文擾動"""
        if not text.strip():
            return [text] * num_augmented

        augmented_texts = []
        words = list(jieba.cut(text))

        for _ in range(num_augmented):
            new_words = words.copy()

            # 隨機替換一些詞
            for i, word in enumerate(words):
                if word in self.replacement_rules and random.random() < 0.3:
                    new_words[i] = random.choice(self.replacement_rules[word])

            augmented_texts.append(''.join(new_words))

        return augmented_texts


if __name__ == "__main__":
    # 測試範例
    print("=== 上下文擾動測試 ===")

    # 使用輕量級擾動器進行快速測試
    print("使用輕量級擾動器:")
    lightweight_perturber = LightweightPerturber()

    test_texts = [
        "你說話很難聽",
        "這個想法不錯",
        "我覺得很好",
        "看起來不太對"
    ]

    for text in test_texts:
        print(f"原文: {text}")
        augmented = lightweight_perturber.augment(text, num_augmented=3)
        for i, aug_text in enumerate(augmented, 1):
            print(f"擾動{i}: {aug_text}")
        print("-" * 50)

    # 如果有 transformers，測試完整功能
    if TRANSFORMERS_AVAILABLE:
        print("\n使用 MacBERT 上下文擾動器:")
        try:
            perturber = ContextualPerturber()

            test_text = "這個人說話很難聽"
            print(f"原文: {test_text}")

            augmented = perturber.augment(test_text, num_augmented=2)
            for i, aug_text in enumerate(augmented, 1):
                print(f"BERT擾動{i}: {aug_text}")

            # 模型信息
            model_info = perturber.get_model_info()
            print(f"模型信息: {model_info}")

        except Exception as e:
            print(f"MacBERT 測試失敗: {e}")
    else:
        print("transformers 未安裝，跳過 MacBERT 測試")